"""
NBA Playoff Predictor - Data Collection
=======================================
Collects two datasets from Basketball-Reference.com for the 1989-90 through
2025-26 NBA seasons:

  1. Team advanced stats (SRS, ORtg, DRtg, NRtg, Pace, Four Factors), one row
     per team-season, from each season-summary page (37 requests).
  2. Playoff game results (date, teams, score, overtime), from each season's
     "Playoffs Schedule and Results" page, which lists EVERY playoff game in a
     single table (37 requests). No per-game box-score requests are needed.

What has and has not been verified
----------------------------------
Page structure (table headers, link patterns, column order) was checked against
the live 2024-25 season-summary page and the live 2025 playoffs schedule page on
4 Oct 2026. Older seasons (1990-2000s) have NOT been checked: layouts there may
differ. All parsing is therefore header-driven rather than position-driven, and
every season is validated before it is checkpointed (see below). Run one old
season first (--start-year 1990 --end-year 1990 --include-playoffs).

Data integrity
--------------
A season is only checkpointed if it passes validation:
  - team stats: exact expected team count for that season, unique team codes
  - playoff games: 16 teams, 15 series, 8/4/2/1 series per round, every series
    ends exactly when one side reaches the required win count, no duplicates,
    and every box-score URL agrees with the parsed date and home team.
Seasons that fail are logged and NEVER checkpointed, so a resumed run cannot
silently trust them. With --allow-incomplete they are written to
<prefix>_<year>_INCOMPLETE.csv for inspection only. The combined output files
are written only when no season failed. If a season legitimately fails, do not
loosen the validator: list it in EXCLUDED_SEASONS with the reason and use
load_checkpoints(), which refuses to run if any other season is missing.

Team-stats value checks (confirmed on the 2024-25 table only; run 1990 and a
few other seasons before trusting them): mean SRS is about 0 and total wins
equal total losses, since every game has one winner and one loser.

Baseline
--------
build_series_table() records home_court_team (the team hosting game 1). The
project's naive baseline is "the home-court team wins the series". Call it that
in reports, not "higher seed": seed and home court can disagree (pre-2015-16
seeding guaranteed division winners a top-4 seed) and the 2020 bubble had no
real home court (see the neutral_site_playoffs flag).

Politeness
----------
Sports Reference enforces roughly 20 requests/minute. Requests are spaced out,
retried with exponential backoff and jitter, and every fetched page is cached
on disk so that a parsing fix never requires re-downloading. The scraper
identifies itself honestly (project name and repo URL in the User-Agent). If
the site blocks it, the run STOPS after the retries are exhausted instead of
hammering a site that has said no; wait, then re-run (checkpointed seasons are
reused). A 404 only skips that one season. Do not remove the delays or run
several copies in parallel. Check Sports Reference's data-use
terms before redistributing anything derived from their pages.

Usage:
    python nba_scraper.py                                    # team stats, full range
    python nba_scraper.py --include-playoffs                 # + playoff games
    python nba_scraper.py --start-year 1990 --end-year 1990 --include-playoffs   # old-season check
    python nba_scraper.py --start-year 2025 --end-year 2025 --include-playoffs   # recent-season check

Tests:
    pip install -r requirements-dev.txt
    pytest -v
"""

import argparse
import logging
import os
import random
import re
import time
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_URL = "https://www.basketball-reference.com"
DEFAULT_USER_AGENT = (
    "nba-playoff-predictor/1.0 (research project; "
    "https://github.com/ChampionshipCube-Sloan2027/Nba-Playoff-Predictor)"
)
USER_AGENT_ENV_VAR = "NBA_SCRAPER_USER_AGENT"
REQUEST_DELAY_SECONDS = 4.0    # ~13 req/min including jitter - under the ~20/min limit
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 30   # doubles on every retry, plus jitter

START_SEASON_DEFAULT = 1990    # 1989-90 season
END_SEASON_DEFAULT = 2026      # 2025-26 season (complete as of Oct 2026)

# Rule-era / anomalous-season / format flags, from documented NBA rule history.
LOCKOUT_OR_DISRUPTED_SEASONS = {1999, 2012, 2020, 2021}   # 50-game, 66-game, bubble, reduced-crowd
SHORTENED_3PT_LINE_SEASONS = set(range(1995, 1998))        # 1994-95 to 1996-97
BEST_OF_5_FIRST_ROUND_CUTOFF = 2002   # seasons <=2002 had a best-of-5 first round
BUBBLE_SEASON = 2020                  # playoffs held at a single neutral site

# Basketball-Reference team codes that denote the same franchise lineage as a
# later code. Anything not listed maps to itself. Follows BR's own franchise
# lineage (e.g. the original Charlotte Hornets, CHH, are the franchise now in
# New Orleans). Used for continuity analysis; joins between team stats and game
# results should use the per-season team_code, which is consistent on both pages.
FRANCHISE_CODE_MAP = {
    "CHH": "NOP", "NOH": "NOP", "NOK": "NOP",
    "CHA": "CHO",
    "SEA": "OKC",
    "NJN": "BRK",
    "VAN": "MEM",
    "WSB": "WAS",
}

TEAM_HREF_RE = re.compile(r"/teams/([A-Z]{3})/\d{4}\.html")
BOX_SCORE_SLUG_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d)([A-Z]{3})$")
OVERTIME_RE = re.compile(r"^(\d?)OT$")

log = logging.getLogger("nba_scraper")


class FetchError(RuntimeError):
    """A page could not be retrieved after retries."""


class NotFoundError(FetchError):
    """The server answered 404; retrying will not help."""


def setup_logging(log_file="scrape.log"):
    """Console + file logging, so a long background run leaves a record."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


# ---------------------------------------------------------------
# Networking - polite, retrying, cached. Network code is tested with fake
# sessions (see test_nba_scraper.py); nothing here needs a live connection to test.
# ---------------------------------------------------------------
def _headers():
    return {"User-Agent": os.environ.get(USER_AGENT_ENV_VAR, DEFAULT_USER_AGENT)}


def _retry_after_seconds(resp):
    """Seconds requested by a Retry-After header (numeric form only), else None."""
    value = getattr(resp, "headers", {}).get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def polite_get(url, session=None, sleep=time.sleep):
    """
    GET a URL and return its text. Sleeps after every success, retries on
    429/403/5xx and on network errors with exponential backoff plus jitter,
    honours Retry-After, and fails fast on 404. `sleep` is injectable for tests.
    """
    session = session or requests.Session()
    backoff = INITIAL_BACKOFF_SECONDS
    last_problem = "no attempt made"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last_problem = f"{type(exc).__name__}: {exc}"
            wait = backoff + random.uniform(0, 5)
            log.warning(f"Network error on {url} ({last_problem}), attempt {attempt}/{MAX_RETRIES}. "
                        f"Backing off {wait:.0f}s.")
            sleep(wait)
            backoff *= 2
            continue

        if resp.status_code == 200:
            sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 1))
            return resp.text
        if resp.status_code == 404:
            raise NotFoundError(f"404 Not Found: {url}")
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            last_problem = f"HTTP {resp.status_code}"
            wait = _retry_after_seconds(resp) or (backoff + random.uniform(0, 5))
            log.warning(f"Blocked/rate-limited on {url} (status {resp.status_code}), "
                        f"attempt {attempt}/{MAX_RETRIES}. Backing off {wait:.0f}s.")
            sleep(wait)
            backoff *= 2
            continue
        raise FetchError(f"Unexpected HTTP {resp.status_code} for {url}")

    raise FetchError(
        f"Failed to fetch {url} after {MAX_RETRIES} attempts (last problem: {last_problem}). "
        f"A 403 or 429 means you are probably blocked: stop, wait before retrying, and "
        f"check Sports Reference's bot and data-use policy. (Set {USER_AGENT_ENV_VAR} to "
        f"override the User-Agent, e.g. to add your own contact details.)"
    )


def cache_path_for(cache_dir, url):
    """Deterministic on-disk path for a URL's raw HTML, e.g. leagues_NBA_2025.html."""
    name = urlparse(url).path.strip("/").replace("/", "_") or "index.html"
    if not name.endswith(".html"):
        name += ".html"
    return Path(cache_dir) / name


def make_fetcher(cache_dir=None, session=None, sleep=time.sleep):
    """
    Returns fetch(url) -> html. With a cache_dir, raw HTML is stored on disk and
    reused, so a parsing fix never means re-downloading from a rate-limited site.
    """
    session = session or requests.Session()

    def fetch(url):
        path = cache_path_for(cache_dir, url) if cache_dir else None
        if path is not None and path.exists():
            log.info(f"cache hit: {path.name}")
            return path.read_text(encoding="utf-8")
        html = polite_get(url, session=session, sleep=sleep)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
        return html

    return fetch


# ---------------------------------------------------------------
# Pure helpers - no network
# ---------------------------------------------------------------
def _uncomment(html):
    """Basketball-Reference hides some tables inside HTML comments so naive
    scrapers and caches miss them. Removing the comment markers reveals them."""
    return html.replace("<!--", "").replace("-->", "")


def _soup(html):
    return BeautifulSoup(_uncomment(html), "lxml")


def get_all_tables(html):
    """
    Every table on the page as a DataFrame, including ones hidden in comments.
    Callers pick the table they want by its actual column names, not an element id.
    """
    try:
        # StringIO is required: recent pandas treats a bare string as a filename.
        # flavor="lxml" is pinned: otherwise pandas falls back to html5lib on a
        # zero-table page and raises ImportError instead of the ValueError caught here.
        return pd.read_html(StringIO(_uncomment(html)), flavor="lxml")
    except ValueError:
        return []


def flatten_columns(df):
    """
    Collapse MultiIndex columns (grouped headers such as Four Factors) into single
    strings, e.g. ('Offense Four Factors', 'eFG%') -> 'Offense Four Factors_eFG%'.
    'Unnamed' header levels are dropped, and columns left with no name at all
    (the blank spacer columns on Basketball-Reference tables) are removed.
    Returns a new DataFrame; the input is not modified.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(level) for level in col if "Unnamed" not in str(level)).strip("_")
            for col in df.columns
        ]
    keep = [str(c) != "" for c in df.columns]
    return df.loc[:, keep]


def _find_table(soup, required_headers):
    """First <table> whose header cells include every name in required_headers."""
    for table in soup.find_all("table"):
        headers = {th.get_text(strip=True) for th in table.find_all("th")}
        if required_headers <= headers:
            return table
    return None


def extract_team_code(href):
    """'/teams/IND/2025.html' (or a full URL) -> 'IND'; None if it isn't a team link."""
    match = TEAM_HREF_RE.search(href or "")
    return match.group(1) if match else None


def franchise_code(team_code):
    """Map a season-specific team code to its franchise lineage code."""
    return FRANCHISE_CODE_MAP.get(team_code, team_code)


def parse_box_score_slug(url):
    """
    Date and home team code from a box-score URL.
    e.g. https://www.basketball-reference.com/boxscores/202606130SAS.html
      -> ("2026-06-13", "SAS")
    The single digit after the date is a game index that belongs to neither the
    date nor the team code. Raises ValueError if the slug is malformed.
    """
    slug = url.rstrip("/").split("/")[-1]
    if slug.endswith(".html"):
        slug = slug[:-5]
    match = BOX_SCORE_SLUG_RE.match(slug)
    if not match:
        raise ValueError(f"Unrecognised box-score slug: {url!r}")
    year, month, day, _game_index, home_team = match.groups()
    return f"{year}-{month}-{day}", home_team


def classify_rule_era(year):
    """Which defensive-rules era a season falls in, by documented NBA history."""
    if year <= 2001:
        return "illegal_defense_era"           # zone defense banned
    elif year <= 2004:
        return "post_illegal_defense_pre_fom"
    else:
        return "freedom_of_movement_era"       # hand-checking tightened, 2004-05+


def add_context_flags(df, year_col="season_end_year"):
    """Adds rule-era, anomalous-season, playoff-format and neutral-site columns."""
    df = df.copy()
    df["anomalous_season"] = df[year_col].isin(LOCKOUT_OR_DISRUPTED_SEASONS)
    df["shortened_3pt_line"] = df[year_col].isin(SHORTENED_3PT_LINE_SEASONS)
    df["first_round_best_of_5"] = df[year_col] <= BEST_OF_5_FIRST_ROUND_CUTOFF
    df["neutral_site_playoffs"] = df[year_col] == BUBBLE_SEASON
    df["rule_era"] = df[year_col].apply(classify_rule_era)
    return df


def expected_team_count(season_end_year):
    """NBA team count: 27 through 1994-95, 29 from 1995-96, 30 from 2004-05."""
    if season_end_year >= 2005:
        return 30
    if season_end_year >= 1996:
        return 29
    return 27


def wins_needed(season_end_year, round_number):
    """Wins required to take a series: 3 in the best-of-5 first round (<=2002), else 4."""
    if round_number == 1 and season_end_year <= BEST_OF_5_FIRST_ROUND_CUTOFF:
        return 3
    return 4


# ---------------------------------------------------------------
# Team advanced stats
# ---------------------------------------------------------------
def parse_team_advanced_stats(html, season_end_year):
    """
    Parse the 'Advanced Stats' team table from a season-summary page.
    Returns one row per team with numeric stats, plus team_code (from the team
    link, matching the playoff-schedule page), franchise_code, made_playoffs
    (the asterisk marker) and season_end_year. Empty DataFrame if no table found.
    Raises ValueError if table rows and team links cannot be aligned.
    """
    table = _find_table(_soup(html), {"SRS", "ORtg", "DRtg"})
    if table is None:
        return pd.DataFrame()

    df = flatten_columns(pd.read_html(StringIO(str(table)), flavor="lxml")[0])
    team_col = next(c for c in df.columns if c == "Team" or c.endswith("_Team"))

    codes = []
    for tr in table.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        link = tr.find("a", href=TEAM_HREF_RE)
        if link is not None:
            codes.append(extract_team_code(link["href"]))

    df = df[df[team_col].notna() & ~df[team_col].isin(["League Average", "Team"])]
    df = df.reset_index(drop=True)
    if len(df) != len(codes):
        raise ValueError(
            f"{season_end_year}: {len(df)} team rows but {len(codes)} team links; "
            f"cannot align team codes safely."
        )

    non_numeric = {team_col, "Arena"}
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    names = df[team_col].astype(str)
    df["made_playoffs"] = names.str.endswith("*")
    df[team_col] = names.str.rstrip("*").str.strip()
    df["team_code"] = codes
    df["franchise_code"] = df["team_code"].map(franchise_code)
    df["season_end_year"] = season_end_year
    return df


def validate_team_stats(df, season_end_year):
    """Sanity-check one scraped season. Returns a list of problems (also logged)."""
    issues = []
    expected = expected_team_count(season_end_year)
    if len(df) != expected:
        issues.append(f"expected {expected} teams, got {len(df)}")
    if "team_code" in df.columns and df["team_code"].duplicated().any():
        dupes = sorted(df.loc[df["team_code"].duplicated(), "team_code"].unique())
        issues.append(f"duplicate team codes: {dupes}")
    # SRS is measured against the league average, so it averages to ~0 by construction;
    # every game has one winner and one loser, so league wins must equal league losses.
    # NaNs are checked first: a mean/sum over a half-parsed column would pass silently.
    if "SRS" in df.columns:
        if df["SRS"].isna().any():
            issues.append("missing SRS values")
        elif abs(df["SRS"].mean()) > 0.15:
            issues.append(f"mean SRS {df['SRS'].mean():.2f}, expected about 0")
    if {"W", "L"} <= set(df.columns):
        if df[["W", "L"]].isna().any().any():
            issues.append("missing W/L values")
        elif df["W"].sum() != df["L"].sum():
            issues.append("total wins do not equal total losses")
    for issue in issues:
        log.warning(f"{season_end_year}: {issue}. Inspect this season before trusting it.")
    return issues


def scrape_team_advanced_stats(season_end_year, fetch):
    url = f"{BASE_URL}/leagues/NBA_{season_end_year}.html"
    df = parse_team_advanced_stats(fetch(url), season_end_year)
    if df.empty:
        log.warning(f"No advanced-stats table found for {season_end_year} - inspect the page manually.")
    return df


# ---------------------------------------------------------------
# Playoff games
# ---------------------------------------------------------------
def _cell_team(cell):
    """(display name, team code) from a visitor/home cell."""
    link = cell.find("a", href=TEAM_HREF_RE)
    if link is None:
        return cell.get_text(strip=True), None
    return link.get_text(strip=True), extract_team_code(link["href"])


def parse_playoff_schedule(html, season_end_year):
    """
    Parse the 'Playoffs Schedule' table (every playoff game of a season).
    Columns are located by header name, not position, because older seasons omit
    some columns (e.g. start time). Rows that cannot be parsed are skipped and
    counted in the log; season validation then catches the resulting gap.
    """
    table = _find_table(_soup(html), {"Date", "Visitor/Neutral", "Home/Neutral"})
    if table is None:
        return pd.DataFrame()

    head = table.find("thead")
    header_row = (head.find_all("tr")[-1] if head is not None else table.find("tr"))
    header = [th.get_text(strip=True) for th in header_row.find_all("th")]
    i_date = header.index("Date")
    i_away = header.index("Visitor/Neutral")
    i_home = header.index("Home/Neutral")
    i_start = header.index("Start (ET)") if "Start (ET)" in header else None
    i_arena = header.index("Arena") if "Arena" in header else None
    i_attend = header.index("Attend.") if "Attend." in header else None

    body = table.find("tbody") or table
    rows, skipped = [], 0
    for tr in body.find_all("tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cells = tr.find_all(["th", "td"])
        if len(cells) != len(header):
            skipped += 1
            continue
        away_team, away_code = _cell_team(cells[i_away])
        home_team, home_code = _cell_team(cells[i_home])
        try:
            date = pd.to_datetime(cells[i_date].get_text(strip=True), format="%a, %b %d, %Y")
        except ValueError:
            skipped += 1
            continue
        if away_code is None or home_code is None:
            skipped += 1
            continue

        overtime = 0
        for cell in cells:
            match = OVERTIME_RE.match(cell.get_text(strip=True))
            if match:
                overtime = int(match.group(1) or 1)
        box = next((a["href"] for a in tr.find_all("a", href=True)
                    if "/boxscores/" in a["href"] and a["href"].endswith(".html")), None)

        rows.append({
            "season_end_year": season_end_year,
            "date": date,
            "start_et": cells[i_start].get_text(strip=True) if i_start is not None else None,
            "away_team": away_team,
            "away_code": away_code,
            "away_pts": cells[i_away + 1].get_text(strip=True),
            "home_team": home_team,
            "home_code": home_code,
            "home_pts": cells[i_home + 1].get_text(strip=True),
            "overtime_periods": overtime,
            "box_score_url": BASE_URL + box if box else None,
            "arena": cells[i_arena].get_text(strip=True) if i_arena is not None else None,
            "attendance": cells[i_attend].get_text(strip=True) if i_attend is not None else None,
        })
    if skipped:
        log.warning(f"{season_end_year}: skipped {skipped} unparseable schedule row(s).")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("away_pts", "home_pts"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["attendance"] = pd.to_numeric(df["attendance"].str.replace(",", "", regex=False),
                                     errors="coerce").astype("Int64")
    df["home_win"] = df["home_pts"] > df["away_pts"]
    return df


def _assign_series_id(games):
    """series_id = '<year>-<code>-<code>' with the two codes sorted. Two teams meet
    at most once per playoffs, so the pairing identifies the series."""
    games = games.copy()
    pair = games[["home_code", "away_code"]].apply(lambda r: "-".join(sorted(r)), axis=1)
    games["series_id"] = games["season_end_year"].astype(str) + "-" + pair
    return games


def build_series_table(games):
    """
    One row per series, derived from game results: wins per side, winner, the
    round (a team's k-th series is round k), the wins needed, and whether the
    series ended cleanly (exactly one side reached the required wins and no
    game was played after the clinch), and home_court_team (host of game 1; the
    basis of the "home-court team wins" baseline).
    """
    columns = ["series_id", "season_end_year", "team_a", "team_b", "games", "wins_a",
               "wins_b", "winner", "round", "wins_needed", "complete", "home_court_team",
               "first_date"]
    if games.empty:
        return pd.DataFrame(columns=columns)
    games = games if "series_id" in games.columns else _assign_series_id(games)
    played = games.dropna(subset=["home_pts", "away_pts"])

    rows = []
    for series_id, g in played.groupby("series_id", sort=False):
        team_a, team_b = series_id.split("-")[1:3]
        home_won = g["home_pts"].astype(float) > g["away_pts"].astype(float)
        winners = np.where(home_won, g["home_code"], g["away_code"])
        wins_a, wins_b = int((winners == team_a).sum()), int((winners == team_b).sum())
        rows.append({
            "series_id": series_id,
            "season_end_year": int(g["season_end_year"].iloc[0]),
            "team_a": team_a, "team_b": team_b,
            "games": len(g), "wins_a": wins_a, "wins_b": wins_b,
            "winner": team_a if wins_a > wins_b else team_b if wins_b > wins_a else None,
            # Game 1 is always hosted by the home-court team, so this is known before
            # the series starts (not leakage). Baseline: "the home-court team wins".
            "home_court_team": g.sort_values("date", kind="stable")["home_code"].iloc[0],
            "first_date": g["date"].min(),
        })
    series = pd.DataFrame(rows)
    if series.empty:
        return pd.DataFrame(columns=columns)

    rounds = {}
    for _, season in series.groupby("season_end_year"):
        series_of_team = {}
        for r in season.sort_values("first_date", kind="stable").itertuples():
            for team in (r.team_a, r.team_b):
                series_of_team.setdefault(team, []).append(r.series_id)
        for r in season.itertuples():
            round_a = series_of_team[r.team_a].index(r.series_id) + 1
            round_b = series_of_team[r.team_b].index(r.series_id) + 1
            rounds[r.series_id] = round_a if round_a == round_b else pd.NA
    series["round"] = series["series_id"].map(rounds).astype("Int64")
    series["wins_needed"] = [
        wins_needed(r.season_end_year, 4 if pd.isna(r.round) else int(r.round))
        for r in series.itertuples()
    ]
    top = series[["wins_a", "wins_b"]].max(axis=1)
    low = series[["wins_a", "wins_b"]].min(axis=1)
    series["complete"] = (top == series["wins_needed"]) & (low < series["wins_needed"])
    return series[columns]


def add_series_context(games):
    """Adds series_id, round and game_in_series (1-based) to a games frame."""
    if games.empty:
        return games
    games = _assign_series_id(games).sort_values(["season_end_year", "date"], kind="stable")
    games["game_in_series"] = games.groupby("series_id").cumcount() + 1
    rounds = build_series_table(games)[["series_id", "round"]]
    return games.merge(rounds, on="series_id", how="left").reset_index(drop=True)


def validate_playoff_games(games, season_end_year):
    """
    Integrity checks for one season's playoff games. Returns a list of problems
    (also logged). An empty list means the season is internally consistent:
    16 teams, 15 series in an 8/4/2/1 bracket, every series ended cleanly, no
    duplicate games, no missing scores, and box-score URLs agree with the data.
    """
    issues = []
    if games.empty:
        return [f"no playoff games parsed for {season_end_year}"]

    teams = set(games["home_code"]) | set(games["away_code"])
    if len(teams) != 16:
        issues.append(f"expected 16 playoff teams, got {len(teams)}")
    missing_scores = int(games[["home_pts", "away_pts"]].isna().any(axis=1).sum())
    if missing_scores:
        issues.append(f"{missing_scores} game(s) with a missing score")
    dupes = int(games.duplicated(subset=["date", "home_code", "away_code"]).sum())
    if dupes:
        issues.append(f"{dupes} duplicate game row(s)")

    series = build_series_table(games)
    if len(series) != 15:
        issues.append(f"expected 15 series, got {len(series)}")
    round_counts = series["round"].value_counts().to_dict()
    for rnd, expected in {1: 8, 2: 4, 3: 2, 4: 1}.items():
        if round_counts.get(rnd, 0) != expected:
            issues.append(f"round {rnd}: expected {expected} series, got {round_counts.get(rnd, 0)}")
    if series["round"].isna().any():
        issues.append("could not assign a round to every series")
    unfinished = series.loc[~series["complete"], "series_id"].tolist()
    if unfinished:
        issues.append(f"series not ending cleanly (missing or extra games): {unfinished}")

    if "box_score_url" in games.columns:
        bad = 0
        for row in games.dropna(subset=["box_score_url"]).itertuples():
            try:
                slug_date, slug_home = parse_box_score_slug(row.box_score_url)
            except ValueError:
                bad += 1
                continue
            if slug_date != row.date.strftime("%Y-%m-%d") or slug_home != row.home_code:
                bad += 1
        if bad:
            issues.append(f"{bad} game(s) whose box-score URL disagrees with the parsed date/home team")

    for issue in issues:
        log.warning(f"{season_end_year}: {issue}")
    return issues


def scrape_playoff_games(season_end_year, fetch):
    """All playoff games for one season, from the single 'Playoffs Schedule' page."""
    url = f"{BASE_URL}/playoffs/NBA_{season_end_year}_games.html"
    games = parse_playoff_schedule(fetch(url), season_end_year)
    if games.empty:
        log.warning(f"No playoff schedule table found for {season_end_year} - inspect the page manually.")
        return games
    return add_context_flags(add_series_context(games))


# ---------------------------------------------------------------
# Orchestration - checkpointed, resumable, validated
# ---------------------------------------------------------------
def _collect_seasons(years, output_dir, prefix, scrape_one, validate, resume,
                     allow_incomplete, read_csv_kwargs=None):
    """
    Scrape each season, validate it, and checkpoint only seasons that pass.
    Failing seasons are never checkpointed (so --resume cannot trust them); with
    allow_incomplete they go to <prefix>_<year>_INCOMPLETE.csv for inspection.
    A 404 or an unparseable page skips that season; any other fetch failure
    (blocked / rate-limited after all retries) stops the run.
    Returns (frames, failed_years, blocked).
    """
    years = list(years)
    frames, failed, blocked = [], [], False
    for position, year in enumerate(years):
        checkpoint = Path(output_dir) / f"{prefix}_{year}.csv"
        if resume and checkpoint.exists():
            log.info(f"[{year}] {prefix} checkpoint exists, skipping scrape.")
            frames.append(pd.read_csv(checkpoint, **(read_csv_kwargs or {})))
            continue
        log.info(f"[{year}] scraping {prefix}...")
        try:
            df = scrape_one(year)
        except NotFoundError as exc:
            log.error(f"[{year}] {prefix}: {exc}")
            failed.append(year)
            continue
        except FetchError as exc:
            log.error(f"[{year}] {prefix}: {exc}")
            failed.append(year)
            log.error(f"Stopping {prefix}: the site may be blocking us. Not attempted: "
                      f"{years[position + 1:]}. Wait before re-running; checkpointed "
                      f"seasons will be reused.")
            blocked = True
            break
        except ValueError:   # the parsers' contract for "cannot parse safely"
            # ValueError is also what pandas raises for many malformed inputs, so a
            # genuine parser bug can land here. log.exception keeps the traceback.
            log.exception(f"[{year}] {prefix}: could not parse page; season skipped")
            failed.append(year)
            continue
        if df.empty:
            failed.append(year)
            continue
        issues = validate(df, year)
        if issues:
            if not allow_incomplete:
                log.error(f"[{year}] {prefix}: {len(issues)} validation issue(s); NOT checkpointed.")
            else:
                incomplete = Path(output_dir) / f"{prefix}_{year}_INCOMPLETE.csv"
                df.to_csv(incomplete, index=False)
                log.error(f"[{year}] {prefix}: {len(issues)} validation issue(s); wrote "
                          f"{incomplete.name} for inspection only (not a checkpoint).")
            failed.append(year)
            continue
        df.to_csv(checkpoint, index=False)
        frames.append(df)
    return frames, failed, blocked


def _log_failure_summary(failures):
    for kind, failed in failures.items():
        if failed:
            log.error(f"{kind}: {len(failed)} season(s) failed or were rejected: {failed}")
    if any(failures.values()):
        log.error(f"Per-season checkpoints for passing seasons are still usable: see "
                  f"load_checkpoints() in {Path(__file__).name}.")


def scrape_range(start_year, end_year, output_dir, include_playoffs, resume, fetch,
                 allow_incomplete=False):
    """Returns {'team_stats': [failed years], 'playoff_games': [failed years]}.
    The failure summary is logged on every exit path, including a block."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    years = range(start_year, end_year + 1)
    failures = {"team_stats": [], "playoff_games": []}
    try:
        frames, failures["team_stats"], blocked = _collect_seasons(
            years, output_dir, "team_advanced_stats",
            scrape_one=lambda y: _with_flags(scrape_team_advanced_stats(y, fetch)),
            validate=validate_team_stats, resume=resume, allow_incomplete=allow_incomplete)
        if frames and not failures["team_stats"]:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(output_dir / f"team_advanced_stats_{start_year}_{end_year}.csv", index=False)
            log.info(f"Saved {len(combined)} team-season rows (combined file).")
        elif failures["team_stats"]:
            log.error(f"Combined team-stats file NOT written: {len(failures['team_stats'])} "
                      f"season(s) failed: {failures['team_stats']}")
        if blocked:
            log.error("Skipping playoff games because the site is blocking requests.")
            return failures

        if not include_playoffs:
            log.info("Skipping playoff games (pass --include-playoffs). Recommended: check one old "
                     "season (1990) and one recent season (2025) before the full range.")
            return failures

        frames, failures["playoff_games"], _ = _collect_seasons(
            years, output_dir, "playoff_games",
            scrape_one=lambda y: scrape_playoff_games(y, fetch),
            validate=validate_playoff_games, resume=resume, allow_incomplete=allow_incomplete,
            read_csv_kwargs={"parse_dates": ["date"]})
        if frames and not failures["playoff_games"]:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(output_dir / f"playoff_games_{start_year}_{end_year}.csv", index=False)
            series = build_series_table(combined)
            series.to_csv(output_dir / f"playoff_series_{start_year}_{end_year}.csv", index=False)
            log.info(f"Saved {len(combined)} game rows and {len(series)} series (combined files).")
        elif failures["playoff_games"]:
            log.error(f"Combined playoff files NOT written: {len(failures['playoff_games'])} "
                      f"season(s) failed: {failures['playoff_games']}")
        return failures
    finally:
        _log_failure_summary(failures)


# Seasons deliberately left out of an analysis, with the reason. Empty until a live
# run shows a season that legitimately fails validation; add it here WITH the reason.
EXCLUDED_SEASONS = {}


def load_checkpoints(output_dir, prefix, start_year, end_year, exclude=None):
    """
    Concatenate per-season checkpoints (<prefix>_<year>.csv) into one DataFrame.
    This is the alternative to the all-or-nothing combined file when one season
    legitimately fails validation. It is NOT lenient: every year in the range must
    either have a checkpoint (which only exists if the season passed validation) or
    be listed in `exclude` as {year: reason}. A missing, unexcluded season raises
    FileNotFoundError, so a gap can never slip into the data silently.
    `exclude` defaults to EXCLUDED_SEASONS; pass {} to require every season.
    prefix is "team_advanced_stats" or "playoff_games".
    """
    exclude = EXCLUDED_SEASONS if exclude is None else exclude
    output_dir = Path(output_dir)
    read_kwargs = {"parse_dates": ["date"]} if prefix == "playoff_games" else {}
    frames, missing = [], []
    for year in range(start_year, end_year + 1):
        if year in exclude:
            log.warning(f"{prefix}: excluding {year}: {exclude[year]}")
            continue
        path = output_dir / f"{prefix}_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, **read_kwargs))
        else:
            missing.append(year)
    if missing:
        raise FileNotFoundError(
            f"No validated {prefix} checkpoint for {missing}. Re-scrape them, or add them to "
            f"`exclude` with a reason.")
    if not frames:
        raise FileNotFoundError(f"No {prefix} checkpoints in {output_dir} for {start_year}-{end_year}.")
    return pd.concat(frames, ignore_index=True)


def _with_flags(df):
    return df if df.empty else add_context_flags(df)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect team stats and playoff games from Basketball-Reference "
                    "for the NBA Playoff Predictor project.")
    parser.add_argument("--start-year", type=int, default=START_SEASON_DEFAULT)
    parser.add_argument("--end-year", type=int, default=END_SEASON_DEFAULT)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--cache-dir", default="data/raw_html",
                        help="Where raw HTML is cached so parsing fixes never re-download.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the raw-HTML cache.")
    parser.add_argument("--include-playoffs", action="store_true",
                        help="Also collect playoff games (validated per season).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing per-season checkpoints and re-scrape.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Keep seasons that fail validation, for inspection.")
    args = parser.parse_args(argv)

    setup_logging()
    fetch = make_fetcher(None if args.no_cache else args.cache_dir)
    failures = scrape_range(
        start_year=args.start_year, end_year=args.end_year, output_dir=args.output_dir,
        include_playoffs=args.include_playoffs, resume=not args.no_resume, fetch=fetch,
        allow_incomplete=args.allow_incomplete)
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
