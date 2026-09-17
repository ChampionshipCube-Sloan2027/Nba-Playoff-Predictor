"""
NBA Playoff Predictor - Data Collection Script
================================================
Scrapes team-level advanced stats and playoff game results from
Basketball-Reference.com for the 1989-90 through 2025-26 NBA seasons.

READ THIS BEFORE RUNNING
-------------------------
1. Rate limiting is real. Sports Reference enforces ~20 requests/minute
   via Cloudflare. This script sleeps between every request, with
   exponential backoff and jitter if it gets blocked. Do NOT remove the
   delays or run multiple copies in parallel.

2. Confidence levels differ between the two scrapers in this file:
   - scrape_team_advanced_stats(): HIGH confidence. Fetched a live page
     and confirmed this exact table (SRS, ORtg, DRtg, NRtg, Pace, Four
     Factors) exists and parses the way this code expects.
   - scrape_playoff_games(): LOWER confidence. The playoffs page structure
     could not be verified live (bot-detection block on first request).
     Run it on ONE season first (--start-year 2025 --end-year 2025
     --include-playoffs), inspect the CSV, and fix anything that doesn't
     match before running the full range.

3. Runtime: ~2,500+ playoff games across 36 seasons, one request per
   game -> a few hours. This script checkpoints per season and resumes
   automatically, so it's safe to stop and restart.

Usage:
    pip install -r requirements.txt
    python nba_scraper.py                              # team stats only, full range
    python nba_scraper.py --include-playoffs            # team stats + playoff games
    python nba_scraper.py --start-year 2025 --end-year 2025 --include-playoffs   # test run

Tests:
    pip install -r requirements-dev.txt
    pytest test_nba_scraper.py -v
"""

import argparse
import logging
import random
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASE_URL = "https://www.basketball-reference.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY_SECONDS = 4.0    # ~15 req/min - comfortably under the 20/min limit
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 30   # doubles on every retry, plus jitter

START_SEASON_DEFAULT = 1990    # 1989-90 season
END_SEASON_DEFAULT = 2026      # 2025-26 season (complete as of Sept 2026)

# Rule-era / anomalous-season / format flags, from documented NBA rule history.
LOCKOUT_OR_DISRUPTED_SEASONS = {1999, 2012, 2020, 2021}   # 50-game, 66-game, bubble, reduced-crowd
SHORTENED_3PT_LINE_SEASONS = set(range(1995, 1998))        # 1994-95 to 1996-97
BEST_OF_5_FIRST_ROUND_CUTOFF = 2002   # seasons <=2002 had a best-of-5 first round

log = logging.getLogger("nba_scraper")


def setup_logging(log_file="scrape.log"):
    """Console + file logging, so a multi-hour background run leaves a record."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


# ---------------------------------------------------------------
# Networking - not unit tested (needs live network), but built defensively
# ---------------------------------------------------------------
def polite_get(url, session=None):
    """GET a URL with a delay, a browser-like header, and exponential
    backoff with jitter on blocks - polite under real rate limiting,
    not just under the happy path."""
    session = session or requests
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            time.sleep(REQUEST_DELAY_SECONDS + random.uniform(0, 1))
            return resp.text
        if resp.status_code in (429, 403):
            wait = backoff + random.uniform(0, 5)
            log.warning(
                f"Blocked/rate-limited on {url} (status {resp.status_code}), "
                f"attempt {attempt}/{MAX_RETRIES}. Backing off {wait:.0f}s."
            )
            time.sleep(wait)
            backoff *= 2
        else:
            log.warning(f"Unexpected status {resp.status_code} on {url}")
            time.sleep(REQUEST_DELAY_SECONDS)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------
# Pure functions - no network - fully unit tested in test_nba_scraper.py
# ---------------------------------------------------------------
def get_all_tables(html):
    """
    Basketball-Reference hides some tables inside HTML comments so they don't
    get picked up by naive scrapers/caches. Stripping the comment markers
    before parsing reveals them. Returns every table on the page as a
    DataFrame - callers pick the one they want by its actual columns, not
    by guessing an element id (ids can shift; column names like 'SRS' won't).
    """
    uncommented = html.replace("<!--", "").replace("-->", "")
    try:
        # StringIO is required here: recent pandas treats a bare string as a
        # filename/URL to open, not literal HTML, and raises FileNotFoundError.
        # flavor="lxml" is pinned explicitly: without it, pandas falls back to
        # html5lib on a zero-table page, which raises ImportError rather than
        # the ValueError this function is built to catch, if html5lib isn't
        # installed (it isn't, and doesn't need to be - lxml is our parser).
        return pd.read_html(StringIO(uncommented), flavor="lxml")
    except ValueError:
        return []


def flatten_columns(df):
    """Collapse pandas' MultiIndex columns (from grouped headers like the
    Four Factors sections) into single strings, e.g. ('Offense Four Factors', 'eFG%')
    becomes 'Offense Four Factors_eFG%'. Bare 'Unnamed' levels are dropped so a
    plain column like Team doesn't become 'Unnamed: 1_Team'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(level) for level in col if "Unnamed" not in str(level)]).strip("_")
            for col in df.columns
        ]
    return df


def parse_box_score_slug(url):
    """
    Extract game date and home team code from a Basketball-Reference box
    score URL. Pure string parsing, no network - the date and home team are
    encoded directly in the URL so there's nothing to scrape for these two
    fields.
    e.g. https://www.basketball-reference.com/boxscores/202606130SAS.html
      -> ("2026-06-13", "SAS")

    Note the single digit after the date (game index, almost always '0')
    belongs to neither the date nor the team code - it's easy to slice past
    it by one character and silently prefix the team code with a stray '0'.
    """
    slug = url.rstrip("/").split("/")[-1]
    if slug.endswith(".html"):
        slug = slug[:-5]
    date_str, home_team = slug[:8], slug[9:]   # slug[8] is the game-index digit, skipped
    game_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return game_date, home_team


def classify_rule_era(year):
    """Which defensive-rules era a season falls in, by documented NBA history."""
    if year <= 2001:
        return "illegal_defense_era"           # zone defense banned
    elif year <= 2004:
        return "post_illegal_defense_pre_fom"
    else:
        return "freedom_of_movement_era"       # hand-checking tightened, 2004-05+


def add_context_flags(df, year_col="season_end_year"):
    """Adds the rule-era, anomalous-season, and playoff-format control columns."""
    df = df.copy()
    df["anomalous_season"] = df[year_col].isin(LOCKOUT_OR_DISRUPTED_SEASONS)
    df["shortened_3pt_line"] = df[year_col].isin(SHORTENED_3PT_LINE_SEASONS)
    df["first_round_best_of_5"] = df[year_col] <= BEST_OF_5_FIRST_ROUND_CUTOFF
    df["rule_era"] = df[year_col].apply(classify_rule_era)
    return df


def validate_team_stats(df, season_end_year):
    """Sanity-check a scraped season: the NBA has had 25-30 teams since 1990,
    so a row count outside a generous window signals a parsing problem, not
    a real league event. Warns rather than raises, so one bad season doesn't
    kill an otherwise-good run."""
    if not (20 <= len(df) <= 32):
        log.warning(
            f"{season_end_year}: got {len(df)} team rows - expected roughly "
            f"25-30. Inspect this season's output before trusting it."
        )


# ---------------------------------------------------------------
# HIGH CONFIDENCE - verified live
# ---------------------------------------------------------------
def scrape_team_advanced_stats(season_end_year):
    """
    Pulls the 'Advanced Stats' team table from the season summary page.
    season_end_year: the year the season ENDS in (e.g. 2025 for the 2024-25 season).
    Returns one row per team: SRS, ORtg, DRtg, NRtg, Pace, Four Factors, etc.
    """
    url = f"{BASE_URL}/leagues/NBA_{season_end_year}.html"
    html = polite_get(url)

    for table in get_all_tables(html):
        table = flatten_columns(table)
        cols = [str(c) for c in table.columns]
        if any("SRS" in c for c in cols) and any("ORtg" in c for c in cols):
            df = table.copy()
            team_col = next(c for c in df.columns if c == "Team" or c.endswith("_Team"))
            df = df[~df[team_col].isin(["League Average", "Team"])]
            df[team_col] = df[team_col].astype(str).str.replace("*", "", regex=False).str.strip()
            df["season_end_year"] = season_end_year
            return df

    log.warning(f"No advanced-stats table found for {season_end_year} - inspect the page manually.")
    return pd.DataFrame()


# ---------------------------------------------------------------
# LOWER CONFIDENCE - not verified live, test before trusting
# ---------------------------------------------------------------
def scrape_playoff_series_links(season_end_year):
    """
    Pulls the playoff series summary for one season and returns the box score
    URL for every game played. Box score URL pattern is verified live; the
    series-listing page itself is NOT - check the first season's output.
    """
    url = f"{BASE_URL}/playoffs/NBA_{season_end_year}.html"
    html = polite_get(url)
    soup = BeautifulSoup(html, "lxml")

    game_links = set()
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("/boxscores/") and a["href"].endswith(".html"):
            game_links.add(BASE_URL + a["href"])

    if not game_links:
        log.warning(f"No box score links found on playoffs page for {season_end_year}.")
    return sorted(game_links)


def scrape_box_score(url):
    """
    Pulls date, home team, and final score from a single box score page.
    Date/home team come from parse_box_score_slug() (unit tested, no network
    needed). The score table itself still needs verification against real
    output - see the module docstring.
    """
    game_date, home_team = parse_box_score_slug(url)
    html = polite_get(url)

    scores = {}
    for table in get_all_tables(html):
        table = flatten_columns(table)
        cols = [str(c) for c in table.columns]
        if any("PTS" in c for c in cols) and len(table) <= 2:
            # Line-score-style small table - adjust once you've seen real output
            scores["raw_table"] = table.to_dict()

    return {
        "date": game_date,
        "home_team": home_team,
        "box_score_url": url,
        **scores,
    }


def scrape_playoff_games(season_end_year):
    """One season's playoff games: find every box score link, then fetch each.
    This is the slow part across 36 seasons - see checkpointing in main()."""
    links = scrape_playoff_series_links(season_end_year)
    rows = []
    for i, link in enumerate(links, 1):
        log.info(f"  [{season_end_year}] game {i}/{len(links)}: {link}")
        try:
            rows.append(scrape_box_score(link))
        except Exception as e:
            log.error(f"  Failed on {link}: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["season_end_year"] = season_end_year
    return df


# ---------------------------------------------------------------
# Orchestration - checkpointed and resumable
# ---------------------------------------------------------------
def scrape_range(start_year, end_year, output_dir, include_playoffs, resume):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    team_frames = []
    for year in range(start_year, end_year + 1):
        checkpoint = output_dir / f"team_advanced_stats_{year}.csv"
        if resume and checkpoint.exists():
            log.info(f"[{year}] team stats checkpoint exists, skipping scrape.")
            df = pd.read_csv(checkpoint)
        else:
            log.info(f"[{year}] scraping team advanced stats...")
            df = scrape_team_advanced_stats(year)
            if not df.empty:
                df = add_context_flags(df)
                validate_team_stats(df, year)
                df.to_csv(checkpoint, index=False)
        if not df.empty:
            team_frames.append(df)

    if team_frames:
        combined = pd.concat(team_frames, ignore_index=True)
        combined.to_csv(output_dir / f"team_advanced_stats_{start_year}_{end_year}.csv", index=False)
        log.info(f"Saved {len(combined)} team-season rows (combined file).")

    if not include_playoffs:
        log.info("Skipping playoff game scraping (pass --include-playoffs to enable). "
                 "Recommended: test a single season before running the full range.")
        return

    game_frames = []
    for year in range(start_year, end_year + 1):
        checkpoint = output_dir / f"playoff_games_{year}.csv"
        if resume and checkpoint.exists():
            log.info(f"[{year}] playoff games checkpoint exists, skipping scrape.")
            df = pd.read_csv(checkpoint)
        else:
            log.info(f"[{year}] scraping playoff games...")
            df = scrape_playoff_games(year)
            if not df.empty:
                df = add_context_flags(df)
                df.to_csv(checkpoint, index=False)
        if not df.empty:
            game_frames.append(df)

    if game_frames:
        combined = pd.concat(game_frames, ignore_index=True)
        combined.to_csv(output_dir / f"playoff_games_{start_year}_{end_year}.csv", index=False)
        log.info(f"Saved {len(combined)} game rows (combined file).")


def main():
    parser = argparse.ArgumentParser(description="Scrape Basketball-Reference for the NBA Playoff Predictor project.")
    parser.add_argument("--start-year", type=int, default=START_SEASON_DEFAULT)
    parser.add_argument("--end-year", type=int, default=END_SEASON_DEFAULT)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--include-playoffs", action="store_true",
                         help="Also scrape playoff games. Off by default - verify on one "
                              "season first, see module docstring.")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore existing per-season checkpoint files and re-scrape everything.")
    args = parser.parse_args()

    setup_logging()
    scrape_range(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output_dir,
        include_playoffs=args.include_playoffs,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
