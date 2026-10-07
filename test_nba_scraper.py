"""
Tests for nba_scraper.py. No test touches the network.

Fixtures in ./fixtures are structure-faithful reconstructions of the live
Basketball-Reference pages (layout checked 4 Oct 2026, real 2024-25 values,
trimmed). They prove the parsers handle that layout; they are not byte-for-byte
copies of the live HTML. Boundary-year logic is tested explicitly because those
are the bugs that produce a plausible-looking but wrong dataset.
"""

import re
from pathlib import Path

import pandas as pd
import pytest
import requests

import nba_scraper as n
from nba_scraper import (
    FetchError,
    NotFoundError,
    add_context_flags,
    add_series_context,
    build_series_table,
    cache_path_for,
    classify_rule_era,
    expected_team_count,
    extract_team_code,
    flatten_columns,
    franchise_code,
    get_all_tables,
    load_checkpoints,
    make_fetcher,
    parse_box_score_slug,
    parse_playoff_schedule,
    parse_team_advanced_stats,
    polite_get,
    scrape_range,
    validate_playoff_games,
    validate_team_stats,
    wins_needed,
)

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------
# Test doubles for the network layer
# ---------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code=200, text="<html></html>", headers=None):
        self.status_code, self.text, self.headers = status_code, text, headers or {}


class FakeSession:
    """Plays back scripted responses (or raises scripted exceptions) in order."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class SleepRecorder:
    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


# ---------------------------------------------------------------
# parse_box_score_slug / team-code helpers
# ---------------------------------------------------------------
def test_parse_box_score_slug_basic():
    date, home = parse_box_score_slug("https://www.basketball-reference.com/boxscores/202606130SAS.html")
    assert (date, home) == ("2026-06-13", "SAS")


def test_parse_box_score_slug_does_not_leak_game_index_into_team_code():
    # Regression: slicing one character too early produced "0SAS".
    date, home = parse_box_score_slug("https://www.basketball-reference.com/boxscores/199906190NYK.html")
    assert (date, home) == ("1999-06-19", "NYK")


@pytest.mark.parametrize("url", [
    "https://www.basketball-reference.com/boxscores/index.fcgi?month=4&day=19&year=2025",
    "https://www.basketball-reference.com/boxscores/2025041.html",
    "https://www.basketball-reference.com/boxscores/202504190indiana.html",
])
def test_parse_box_score_slug_rejects_malformed_slugs(url):
    with pytest.raises(ValueError):
        parse_box_score_slug(url)


@pytest.mark.parametrize("href, expected", [
    ("/teams/IND/2025.html", "IND"),
    ("https://www.basketball-reference.com/teams/BRK/2025.html", "BRK"),
    ("/boxscores/202504190IND.html", None),
    ("", None),
    (None, None),
])
def test_extract_team_code(href, expected):
    assert extract_team_code(href) == expected


def test_franchise_code_follows_lineage_and_defaults_to_identity():
    assert franchise_code("SEA") == "OKC"
    assert franchise_code("NJN") == "BRK"
    assert franchise_code("CHH") == "NOP"   # original Hornets are the franchise now in New Orleans
    assert franchise_code("CHA") == "CHO"   # Bobcats lineage
    assert franchise_code("LAL") == "LAL"


# ---------------------------------------------------------------
# Era / context flags / format helpers
# ---------------------------------------------------------------
def test_classify_rule_era_boundaries():
    assert classify_rule_era(1990) == "illegal_defense_era"
    assert classify_rule_era(2001) == "illegal_defense_era"
    assert classify_rule_era(2002) == "post_illegal_defense_pre_fom"
    assert classify_rule_era(2004) == "post_illegal_defense_pre_fom"
    assert classify_rule_era(2005) == "freedom_of_movement_era"
    assert classify_rule_era(2026) == "freedom_of_movement_era"


def test_add_context_flags_anomalous_seasons():
    df = pd.DataFrame({"season_end_year": [1998, 1999, 2000, 2012, 2020, 2021, 2022]})
    assert list(add_context_flags(df)["anomalous_season"]) == [False, True, False, True, True, True, False]


def test_add_context_flags_shortened_3pt_line():
    df = pd.DataFrame({"season_end_year": [1994, 1995, 1996, 1997, 1998]})
    assert list(add_context_flags(df)["shortened_3pt_line"]) == [False, True, True, True, False]


def test_add_context_flags_first_round_format():
    df = pd.DataFrame({"season_end_year": [2001, 2002, 2003, 2004]})
    assert list(add_context_flags(df)["first_round_best_of_5"]) == [True, True, False, False]


def test_add_context_flags_marks_only_the_bubble_as_neutral_site():
    df = pd.DataFrame({"season_end_year": [2019, 2020, 2021]})
    assert list(add_context_flags(df)["neutral_site_playoffs"]) == [False, True, False]


def test_add_context_flags_does_not_mutate_input():
    df = pd.DataFrame({"season_end_year": [2010]})
    add_context_flags(df)
    assert list(df.columns) == ["season_end_year"]


@pytest.mark.parametrize("year, expected", [
    (1990, 27), (1995, 27), (1996, 29), (2004, 29), (2005, 30), (2026, 30),
])
def test_expected_team_count_boundaries(year, expected):
    assert expected_team_count(year) == expected


@pytest.mark.parametrize("year, rnd, expected", [
    (2002, 1, 3), (2002, 2, 4), (2003, 1, 4), (1990, 4, 4),
])
def test_wins_needed(year, rnd, expected):
    assert wins_needed(year, rnd) == expected


# ---------------------------------------------------------------
# flatten_columns / get_all_tables
# ---------------------------------------------------------------
def test_flatten_columns_multiindex():
    df = pd.DataFrame(
        [[0.55, "Lakers"]],
        columns=pd.MultiIndex.from_tuples([("Offense Four Factors", "eFG%"), ("Unnamed: 1_level_0", "Team")]),
    )
    out = flatten_columns(df)
    assert "Offense Four Factors_eFG%" in out.columns
    assert "Team" in out.columns


def test_flatten_columns_drops_blank_spacer_columns():
    df = pd.DataFrame(
        [[1, 2, 3]],
        columns=pd.MultiIndex.from_tuples(
            [("Unnamed: 0_level_0", "Team"), ("Unnamed: 1_level_0", "Unnamed: 1_level_1"), ("Defense Four Factors", "eFG%")]
        ),
    )
    assert list(flatten_columns(df).columns) == ["Team", "Defense Four Factors_eFG%"]


def test_flatten_columns_leaves_normal_columns_alone_and_does_not_mutate():
    df = pd.DataFrame([[1, 2]], columns=["Team", "SRS"])
    out = flatten_columns(df)
    assert list(out.columns) == ["Team", "SRS"]
    out["Team"] = 99
    assert df.loc[0, "Team"] == 1


def test_get_all_tables_finds_commented_out_table():
    html = ('<html><body><!--<table><tr><th>SRS</th><th>ORtg</th></tr>'
            '<tr><td>1.2</td><td>110</td></tr></table>--></body></html>')
    tables = get_all_tables(html)
    assert len(tables) == 1 and "SRS" in tables[0].columns


def test_get_all_tables_finds_normal_table_too():
    html = '<table><tr><th>Team</th><th>W</th></tr><tr><td>Celtics</td><td>60</td></tr></table>'
    assert "Team" in get_all_tables(html)[0].columns


def test_get_all_tables_returns_empty_list_for_no_tables():
    assert get_all_tables("<html><body><p>No tables here.</p></body></html>") == []


# ---------------------------------------------------------------
# Team stats: parsing the fixture, validation
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def team_stats():
    return parse_team_advanced_stats(read_fixture("season_2025_sample.html"), 2025)


def test_team_stats_selects_the_advanced_table_not_the_decoys(team_stats):
    # Fixture also holds a standings table (has SRS, no ORtg) and a per-game table.
    assert len(team_stats) == 5
    assert "ORtg" in team_stats.columns and "DRtg" in team_stats.columns


def test_team_stats_finds_table_hidden_in_a_comment(team_stats):
    assert not team_stats.empty   # the fixture wraps the real table in <!-- -->


def test_team_stats_drops_league_average_and_blank_spacer_columns(team_stats):
    assert "League Average" not in set(team_stats["Team"])
    assert all(str(c) != "" for c in team_stats.columns)
    assert not any("Unnamed" in str(c) for c in team_stats.columns)


def test_team_stats_team_codes_come_from_links_and_align_with_rows(team_stats):
    got = dict(zip(team_stats["Team"], team_stats["team_code"]))
    assert got == {
        "Oklahoma City Thunder": "OKC", "Boston Celtics": "BOS", "Cleveland Cavaliers": "CLE",
        "Atlanta Hawks": "ATL", "Washington Wizards": "WAS",
    }


def test_team_stats_asterisk_becomes_a_flag_and_is_stripped_from_names(team_stats):
    flags = dict(zip(team_stats["team_code"], team_stats["made_playoffs"]))
    assert flags == {"OKC": True, "BOS": True, "CLE": True, "ATL": False, "WAS": False}
    assert not team_stats["Team"].str.contains(r"\*").any()


def test_team_stats_values_are_numeric_and_correct(team_stats):
    okc = team_stats.set_index("team_code").loc["OKC"]
    assert okc["SRS"] == pytest.approx(12.70)
    assert okc["NRtg"] == pytest.approx(12.8)          # "+12.8" parsed, not left as text
    assert okc["Offense Four Factors_eFG%"] == pytest.approx(0.560)
    assert okc["Defense Four Factors_eFG%"] == pytest.approx(0.513)
    assert okc["Attend."] == 754832                     # thousands separator handled
    assert okc["Arena"] == "Paycom Center"
    assert pd.api.types.is_numeric_dtype(team_stats["ORtg"])


def test_team_stats_adds_season_and_franchise_columns(team_stats):
    assert set(team_stats["season_end_year"]) == {2025}
    assert list(team_stats["franchise_code"]) == list(team_stats["team_code"])


def test_parse_team_stats_returns_empty_when_no_table():
    assert parse_team_advanced_stats("<html><body><p>nothing</p></body></html>", 2025).empty


def test_parse_team_stats_refuses_to_guess_when_rows_and_links_misalign():
    html = ('<table><thead><tr><th>Rk</th><th>Team</th><th>SRS</th><th>ORtg</th><th>DRtg</th></tr></thead><tbody>'
            '<tr><th>1</th><td><a href="/teams/BOS/2025.html">Boston Celtics</a></td><td>8</td><td>120</td><td>111</td></tr>'
            '<tr><th>2</th><td>Mystery Team (no link)</td><td>1</td><td>110</td><td>110</td></tr></tbody></table>')
    with pytest.raises(ValueError, match="cannot align"):
        parse_team_advanced_stats(html, 2025)


def test_validate_team_stats_silent_when_count_matches(caplog):
    df = pd.DataFrame({"team_code": [f"T{i:02d}" for i in range(30)]})
    with caplog.at_level("WARNING"):
        assert validate_team_stats(df, 2025) == []
    assert caplog.text == ""


@pytest.mark.parametrize("rows, year", [(2, 2025), (29, 2025), (31, 2025), (30, 1996), (28, 1990)])
def test_validate_team_stats_flags_any_count_mismatch(rows, year, caplog):
    df = pd.DataFrame({"team_code": [f"T{i:02d}" for i in range(rows)]})
    with caplog.at_level("WARNING"):
        issues = validate_team_stats(df, year)
    assert any("expected" in i for i in issues) and "expected" in caplog.text


# Real 2024-25 regular-season W, L and SRS for all 30 teams (Basketball-Reference, Advanced Stats).
REAL_2025_W = [68,61,64,49,50,48,52,51,50,48,48,50,44,50,37,40,41,40,39,39,34,36,36,30,24,26,17,19,21,18]
REAL_2025_L = [14,21,18,33,32,34,30,31,32,34,34,32,38,32,45,42,41,42,43,43,48,46,46,52,58,56,65,63,61,64]
REAL_2025_SRS = [12.70,8.28,8.81,5.15,4.84,4.79,4.97,3.59,3.97,3.56,2.12,1.68,1.73,1.45,0.11,0.58,
                 -0.70,-1.41,-0.74,-1.83,-2.45,-2.67,-2.55,-4.40,-6.29,-6.95,-8.51,-9.10,-8.59,-12.14]


def real_2025_frame():
    return pd.DataFrame({"team_code": [f"T{i:02d}" for i in range(30)],
                         "W": REAL_2025_W, "L": REAL_2025_L, "SRS": REAL_2025_SRS})


def test_value_checks_pass_on_the_real_2024_25_table():
    # Confirmed on ONE real season only; run other seasons before trusting the checks.
    assert validate_team_stats(real_2025_frame(), 2025) == []


def test_value_check_flags_srs_that_does_not_centre_on_zero():
    df = real_2025_frame().assign(SRS=lambda d: d["SRS"] + 1.0)
    assert any("mean SRS" in i for i in validate_team_stats(df, 2025))


def test_value_check_flags_wins_not_equal_to_losses():
    df = real_2025_frame()
    df.loc[0, "W"] += 1
    assert any("wins do not equal" in i for i in validate_team_stats(df, 2025))


def test_value_checks_flag_missing_values_instead_of_passing_silently():
    df = real_2025_frame()
    df.loc[3, "SRS"] = float("nan")
    assert any("missing SRS" in i for i in validate_team_stats(df, 2025))
    df = real_2025_frame()
    df.loc[3, "W"] = float("nan")
    assert any("missing W/L" in i for i in validate_team_stats(df, 2025))


def test_validate_team_stats_flags_duplicate_team_codes():
    df = pd.DataFrame({"team_code": ["BOS"] * 2 + [f"T{i:02d}" for i in range(28)]})
    assert any("duplicate" in i for i in validate_team_stats(df, 2025))


# ---------------------------------------------------------------
# Playoff schedule: parsing the fixture
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def schedule():
    return parse_playoff_schedule(read_fixture("playoffs_2025_games_sample.html"), 2025)


def test_schedule_parses_every_row(schedule):
    assert len(schedule) == 8


def test_schedule_parses_teams_scores_and_winner_from_header_positions(schedule):
    first = schedule.iloc[0]
    assert (first["away_code"], first["away_pts"], first["home_code"], first["home_pts"]) == ("MIL", 98, "IND", 117)
    assert first["date"] == pd.Timestamp("2025-04-19")
    assert bool(first["home_win"]) is True
    assert bool(schedule.iloc[3]["home_win"]) is False          # GSW 99 @ MIN 88: visitor won


def test_schedule_parses_overtime(schedule):
    assert list(schedule["overtime_periods"]) == [0, 1, 0, 0, 1, 0, 0, 0]


def test_schedule_keeps_box_score_url_and_it_agrees_with_row(schedule):
    row = schedule.iloc[1]
    assert row["box_score_url"].endswith("/boxscores/202504190DEN.html")
    assert parse_box_score_slug(row["box_score_url"]) == ("2025-04-19", row["home_code"])


def test_schedule_handles_missing_log_and_attendance_formatting(schedule):
    assert schedule.iloc[0]["attendance"] == 17274               # thousands separator removed


def test_schedule_works_without_optional_columns():
    html = ('<table><thead><tr><th>Date</th><th>Visitor/Neutral</th><th>PTS</th><th>Home/Neutral</th><th>PTS</th></tr></thead>'
            '<tbody><tr><th>Sat, Apr 21, 1990</th><td><a href="/teams/BOS/1990.html">Boston Celtics</a></td><td>101</td>'
            '<td><a href="/teams/NYK/1990.html">New York Knicks</a></td><td>96</td></tr></tbody></table>')
    df = parse_playoff_schedule(html, 1990)
    assert len(df) == 1 and df.loc[0, "start_et"] is None
    assert (df.loc[0, "away_code"], df.loc[0, "home_pts"]) == ("BOS", 96)


def test_schedule_skips_malformed_rows_and_logs_them(caplog):
    html = ('<table><thead><tr><th>Date</th><th>Visitor/Neutral</th><th>PTS</th><th>Home/Neutral</th><th>PTS</th></tr></thead><tbody>'
            '<tr><th>not a date</th><td><a href="/teams/BOS/1990.html">Boston Celtics</a></td><td>1</td>'
            '<td><a href="/teams/NYK/1990.html">New York Knicks</a></td><td>2</td></tr>'
            '<tr><td colspan="5">Playoffs note</td></tr></tbody></table>')
    with caplog.at_level("WARNING"):
        df = parse_playoff_schedule(html, 1990)
    assert df.empty and "skipped 2" in caplog.text


def test_schedule_returns_empty_when_table_missing():
    assert parse_playoff_schedule("<html><body></body></html>", 2025).empty


# ---------------------------------------------------------------
# Series reconstruction and season validation (synthetic full bracket)
# ---------------------------------------------------------------
CODES = [c * 3 for c in "ABCDEFGHIJKLMNOP"]   # 16 fake teams: AAA..PPP


def make_season(year=2025, first_round_wins=4):
    """A complete, valid 16-team bracket. Lower-lettered team always wins its series."""
    games, base = [], pd.Timestamp(f"{year}-04-19")

    def play(rnd, a, b, wins):
        start = base + pd.Timedelta(days=14 * (rnd - 1))
        results = ["a"] * wins + ["b"] * (1 if wins == 4 else 0)   # winner sweeps, or wins 4-1 / 3-1
        for i, side in enumerate(results):
            home, away = (a, b) if i % 2 == 0 else (b, a)
            winner = a if side == "a" else b
            home_pts, away_pts = (100, 90) if winner == home else (90, 100)
            games.append({"season_end_year": year, "date": start + pd.Timedelta(days=2 * i),
                          "home_code": home, "away_code": away, "home_pts": home_pts, "away_pts": away_pts})

    alive = CODES[:]
    for rnd in range(1, 5):
        wins = first_round_wins if rnd == 1 else 4
        nxt = []
        for a, b in zip(alive[::2], alive[1::2]):
            play(rnd, a, b, wins)
            nxt.append(a)
        alive = nxt
    return pd.DataFrame(games)


def test_valid_bracket_passes_validation():
    assert validate_playoff_games(make_season(), 2025) == []


def test_valid_best_of_five_era_bracket_passes_validation():
    assert validate_playoff_games(make_season(year=2001, first_round_wins=3), 2001) == []


def test_best_of_five_bracket_fails_if_treated_as_a_modern_season():
    # Same games, but 2003+ rules require 4 wins in round 1, so first-round series look unfinished.
    issues = validate_playoff_games(make_season(year=2001, first_round_wins=3).assign(season_end_year=2005), 2005)
    assert any("not ending cleanly" in i for i in issues)


def test_series_table_assigns_rounds_8_4_2_1():
    series = build_series_table(make_season())
    assert series["round"].value_counts().sort_index().to_dict() == {1: 8, 2: 4, 3: 2, 4: 1}
    assert series["complete"].all()
    final = series[series["round"] == 4].iloc[0]
    assert (final["winner"], final["wins_a"] + final["wins_b"]) == ("AAA", final["games"])


def test_home_court_team_is_the_host_of_game_one_regardless_of_row_order_or_code_order():
    # 2-2 hosting pattern: ZZZ hosts games 1-2, AAA hosts games 3-4. The first host (ZZZ) differs
    # from both the last host (AAA) and the alphabetically first code (AAA); rows are shuffled so
    # row order cannot help either.
    def game(day, home, away, home_pts, away_pts):
        return {"season_end_year": 2025, "date": pd.Timestamp(f"2025-04-{day}"), "home_code": home,
                "away_code": away, "home_pts": home_pts, "away_pts": away_pts}

    games = pd.DataFrame([
        game(23, "AAA", "ZZZ", 99, 100),
        game(19, "ZZZ", "AAA", 110, 100),
        game(25, "AAA", "ZZZ", 90, 95),
        game(21, "ZZZ", "AAA", 105, 95),
    ])
    assert list(build_series_table(games)["home_court_team"]) == ["ZZZ"]


def test_home_court_baseline_is_computable_from_the_series_table():
    series = build_series_table(make_season())
    assert "home_court_team" in series.columns
    # In the synthetic bracket the home-court team (game-1 host) wins every series.
    assert (series["winner"] == series["home_court_team"]).all()


def test_add_series_context_numbers_games_within_a_series():
    games = add_series_context(make_season())
    one = games[games["series_id"] == "2025-AAA-BBB"]
    assert list(one["game_in_series"]) == [1, 2, 3, 4, 5]
    assert set(one["round"]) == {1}


def _series_rows(games, a, b):
    return games[games["home_code"].isin([a, b]) & games["away_code"].isin([a, b])]


def test_missing_game_that_the_series_winner_won_is_detected():
    games = make_season()
    games = games.drop(_series_rows(games, "AAA", "BBB").index[0])      # AAA won game 1: 4-1 becomes 3-1
    assert any("2025-AAA-BBB" in i for i in validate_playoff_games(games, 2025))


def test_known_limit_a_missing_game_that_does_not_change_the_clinch_is_invisible():
    # Win-count validation cannot see a dropped game from the losing side of a series:
    # a 4-1 series missing the loser's win looks exactly like a legitimate 4-0 sweep.
    # (Cross-checking against the series records on the playoffs summary page would catch it.)
    games = make_season()
    games = games.drop(_series_rows(games, "AAA", "BBB").index[-1])     # BBB's only win
    assert validate_playoff_games(games, 2025) == []


def test_game_after_the_clinch_is_detected():
    games = make_season()
    extra = games[(games["home_code"] == "CCC") & (games["away_code"] == "DDD")].iloc[[0]].copy()
    extra["date"] += pd.Timedelta(days=30)
    issues = validate_playoff_games(pd.concat([games, extra], ignore_index=True), 2025)
    assert any("2025-CCC-DDD" in i for i in issues)


def test_duplicate_game_row_is_detected():
    games = make_season()
    issues = validate_playoff_games(pd.concat([games, games.iloc[[0]]], ignore_index=True), 2025)
    assert any("duplicate" in i for i in issues)


def test_wrong_team_count_is_detected():
    games = make_season()
    games = games[~games["home_code"].isin(["OOO", "PPP"]) & ~games["away_code"].isin(["OOO", "PPP"])]
    issues = validate_playoff_games(games, 2025)
    assert any("16 playoff teams" in i for i in issues)


def test_missing_score_is_detected():
    games = make_season()
    games["home_pts"] = games["home_pts"].astype("Int64")
    games.loc[0, "home_pts"] = pd.NA
    assert any("missing score" in i for i in validate_playoff_games(games, 2025))


def test_box_score_url_disagreeing_with_row_is_detected():
    games = make_season()
    games["box_score_url"] = [
        f"https://www.basketball-reference.com/boxscores/{d:%Y%m%d}0{h}.html"
        for d, h in zip(games["date"], games["home_code"])
    ]
    assert validate_playoff_games(games, 2025) == []              # consistent URLs pass
    games.loc[0, "box_score_url"] = "https://www.basketball-reference.com/boxscores/202504190ZZZ.html"
    assert any("box-score URL disagrees" in i for i in validate_playoff_games(games, 2025))


def test_empty_season_is_reported():
    assert validate_playoff_games(pd.DataFrame(), 2025)


# ---------------------------------------------------------------
# polite_get: retry / backoff / failure behaviour
# ---------------------------------------------------------------
def test_polite_get_returns_text_and_pauses_after_success():
    sleep = SleepRecorder()
    assert polite_get("u", FakeSession([FakeResponse(200, "hello")]), sleep=sleep) == "hello"
    assert len(sleep.waits) == 1 and sleep.waits[0] >= n.REQUEST_DELAY_SECONDS


def test_polite_get_backs_off_on_429_then_succeeds():
    sleep, session = SleepRecorder(), FakeSession([FakeResponse(429), FakeResponse(200, "ok")])
    assert polite_get("u", session, sleep=sleep) == "ok"
    assert session.calls == 2 and sleep.waits[0] >= n.INITIAL_BACKOFF_SECONDS


def test_polite_get_backoff_grows_between_attempts():
    sleep = SleepRecorder()
    polite_get("u", FakeSession([FakeResponse(403), FakeResponse(403), FakeResponse(200)]), sleep=sleep)
    assert sleep.waits[1] >= 2 * n.INITIAL_BACKOFF_SECONDS


def test_polite_get_honours_retry_after_header():
    sleep = SleepRecorder()
    polite_get("u", FakeSession([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200)]), sleep=sleep)
    assert sleep.waits[0] == 7


def test_polite_get_does_not_retry_a_404():
    session = FakeSession([FakeResponse(404), FakeResponse(200)])
    with pytest.raises(NotFoundError):
        polite_get("u", session, sleep=SleepRecorder())
    assert session.calls == 1


def test_polite_get_retries_network_errors():
    session = FakeSession([requests.Timeout("slow"), requests.ConnectionError("down"), FakeResponse(200, "back")])
    assert polite_get("u", session, sleep=SleepRecorder()) == "back"
    assert session.calls == 3


def test_polite_get_gives_up_after_max_retries_with_actionable_message():
    session = FakeSession([FakeResponse(403)] * n.MAX_RETRIES)
    with pytest.raises(FetchError, match="HTTP 403"):
        polite_get("u", session, sleep=SleepRecorder())
    assert session.calls == n.MAX_RETRIES


def test_polite_get_fails_fast_on_unexpected_status():
    session = FakeSession([FakeResponse(418)])
    with pytest.raises(FetchError, match="418"):
        polite_get("u", session, sleep=SleepRecorder())
    assert session.calls == 1


def test_default_user_agent_identifies_the_project_and_is_not_a_browser(monkeypatch):
    monkeypatch.delenv(n.USER_AGENT_ENV_VAR, raising=False)
    agent = n._headers()["User-Agent"]
    assert "nba-playoff-predictor" in agent and "github.com" in agent
    assert "Mozilla" not in agent and "Chrome" not in agent


def test_user_agent_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv(n.USER_AGENT_ENV_VAR, "my-research-bot/1.0")
    assert n._headers()["User-Agent"] == "my-research-bot/1.0"


# ---------------------------------------------------------------
# Raw-HTML cache
# ---------------------------------------------------------------
def test_cache_path_is_deterministic_and_flat(tmp_path):
    p = cache_path_for(tmp_path, "https://www.basketball-reference.com/playoffs/NBA_2025_games.html")
    assert p == tmp_path / "playoffs_NBA_2025_games.html"


def test_fetcher_downloads_once_then_serves_from_cache(tmp_path):
    session = FakeSession([FakeResponse(200, "<p>page</p>")])
    fetch = make_fetcher(tmp_path, session=session, sleep=SleepRecorder())
    assert fetch("https://www.basketball-reference.com/leagues/NBA_2025.html") == "<p>page</p>"
    assert fetch("https://www.basketball-reference.com/leagues/NBA_2025.html") == "<p>page</p>"
    assert session.calls == 1


def test_fetcher_without_cache_always_downloads():
    session = FakeSession([FakeResponse(200, "a"), FakeResponse(200, "b")])
    fetch = make_fetcher(None, session=session, sleep=SleepRecorder())
    assert (fetch("u"), fetch("u")) == ("a", "b")


# ---------------------------------------------------------------
# Orchestration: checkpointing, resume, rejection of bad seasons
# ---------------------------------------------------------------
def _team_table_html(year):
    """A valid 30-team advanced-stats table: wins mirror losses and SRS sums to zero."""
    codes = [f"T{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(30)]
    offsets = list(range(1, 16)) + [-o for o in range(1, 16)]
    rows = "".join(
        f'<tr><th>{i + 1}</th><td><a href="/teams/{c}/{year}.html">Team {c}</a>*</td>'
        f"<td>{41 + o}</td><td>{41 - o}</td><td>{o / 2}</td><td>{110 + o}</td><td>{110 - o}</td></tr>"
        for i, (c, o) in enumerate(zip(codes, offsets))
    )
    return ("<table><thead><tr><th>Rk</th><th>Team</th><th>W</th><th>L</th><th>SRS</th><th>ORtg</th><th>DRtg</th></tr></thead>"
            f"<tbody>{rows}<tr><td></td><td>League Average</td><td></td><td></td><td>0.0</td><td>110</td><td>110</td></tr></tbody></table>")


def _playoff_table_html(year):
    games = make_season(year=year)
    rows = "".join(
        f'<tr><th>{g.date.strftime("%a, %b %d, %Y")}</th>'
        f'<td><a href="/teams/{g.away_code}/{year}.html">{g.away_code}</a></td><td>{g.away_pts}</td>'
        f'<td><a href="/teams/{g.home_code}/{year}.html">{g.home_code}</a></td><td>{g.home_pts}</td></tr>'
        for g in games.itertuples()
    )
    return ("<table><thead><tr><th>Date</th><th>Visitor/Neutral</th><th>PTS</th><th>Home/Neutral</th><th>PTS</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def valid_fetch(url):
    """A fake site whose every season is internally consistent, so the happy path can be tested."""
    year = int(re.search(r"NBA_(\d{4})", url).group(1))
    return _playoff_table_html(year) if "/playoffs/" in url else _team_table_html(year)


def fixture_fetch(url):
    if "/playoffs/" in url:
        return read_fixture("playoffs_2025_games_sample.html")
    return read_fixture("season_2025_sample.html")


def test_bad_seasons_are_rejected_not_checkpointed(tmp_path):
    # Fixtures hold 5 teams / 8 games, so validation must reject both datasets.
    failures = scrape_range(2025, 2025, tmp_path, include_playoffs=True, resume=True, fetch=fixture_fetch)
    assert failures == {"team_stats": [2025], "playoff_games": [2025]}
    assert not (tmp_path / "team_advanced_stats_2025.csv").exists()
    assert not (tmp_path / "playoff_games_2025.csv").exists()


def test_allow_incomplete_writes_failing_seasons_to_separate_incomplete_files(tmp_path):
    failures = scrape_range(2025, 2025, tmp_path, include_playoffs=True, resume=True, fetch=fixture_fetch,
                            allow_incomplete=True)
    assert failures == {"team_stats": [2025], "playoff_games": [2025]}   # still reported as failed
    assert (tmp_path / "team_advanced_stats_2025_INCOMPLETE.csv").exists()
    games = pd.read_csv(tmp_path / "playoff_games_2025_INCOMPLETE.csv", parse_dates=["date"])
    assert len(games) == 8 and {"series_id", "round", "game_in_series", "neutral_site_playoffs"} <= set(games.columns)
    # Not trusted as checkpoints, and no combined files from failing data:
    assert not (tmp_path / "team_advanced_stats_2025.csv").exists()
    assert not (tmp_path / "playoff_games_2025.csv").exists()
    assert not list(tmp_path.glob("*_2025_2025.csv"))


def test_failing_season_is_rescraped_on_resume_not_silently_trusted(tmp_path):
    scrape_range(2025, 2025, tmp_path, include_playoffs=False, resume=True, fetch=fixture_fetch,
                 allow_incomplete=True)
    calls = []

    def counting_fetch(url):
        calls.append(url)
        return read_fixture("season_2025_sample.html")

    scrape_range(2025, 2025, tmp_path, include_playoffs=False, resume=True, fetch=counting_fetch,
                 allow_incomplete=True)
    assert len(calls) == 1      # the INCOMPLETE file was ignored; the season was fetched again


def test_combined_files_are_written_only_when_every_season_passes(tmp_path):
    failures = scrape_range(2024, 2025, tmp_path, include_playoffs=True, resume=True, fetch=valid_fetch)
    assert failures == {"team_stats": [], "playoff_games": []}
    assert len(pd.read_csv(tmp_path / "team_advanced_stats_2024_2025.csv")) == 60
    games = pd.read_csv(tmp_path / "playoff_games_2024_2025.csv")
    assert len(games) == 2 * len(make_season())
    series = pd.read_csv(tmp_path / "playoff_series_2024_2025.csv")
    assert len(series) == 30 and "home_court_team" in series.columns
    assert (tmp_path / "team_advanced_stats_2024.csv").exists()          # per-season checkpoints too


def test_no_combined_file_when_any_season_failed(tmp_path):
    def one_bad_season(url):
        if "NBA_2024" in url and "/playoffs/" not in url:
            return read_fixture("season_2025_sample.html")              # 5 teams: fails validation
        return valid_fetch(url)

    failures = scrape_range(2024, 2025, tmp_path, include_playoffs=False, resume=True, fetch=one_bad_season)
    assert failures["team_stats"] == [2024]
    assert (tmp_path / "team_advanced_stats_2025.csv").exists()          # the good season is kept...
    assert not (tmp_path / "team_advanced_stats_2024_2025.csv").exists() # ...but nothing is combined


def test_resume_uses_checkpoints_without_touching_the_network(tmp_path):
    pd.DataFrame({"team_code": ["AAA"], "season_end_year": [2025]}).to_csv(
        tmp_path / "team_advanced_stats_2025.csv", index=False)

    def exploding_fetch(url):
        raise AssertionError("network must not be used when a checkpoint exists")

    assert scrape_range(2025, 2025, tmp_path, include_playoffs=False, resume=True,
                        fetch=exploding_fetch) == {"team_stats": [], "playoff_games": []}


def test_no_resume_rescrapes_even_if_checkpoint_exists(tmp_path):
    pd.DataFrame({"team_code": ["AAA"], "season_end_year": [2025]}).to_csv(
        tmp_path / "team_advanced_stats_2025.csv", index=False)
    calls = []

    def counting_fetch(url):
        calls.append(url)
        return read_fixture("season_2025_sample.html")

    scrape_range(2025, 2025, tmp_path, include_playoffs=False, resume=False, fetch=counting_fetch)
    assert len(calls) == 1


def test_a_block_stops_the_run_and_later_seasons_are_not_attempted(tmp_path, caplog):
    attempted = []

    def blocked_fetch(url):
        attempted.append(url)
        raise FetchError("blocked")

    with caplog.at_level("ERROR"):
        failures = scrape_range(2024, 2026, tmp_path, include_playoffs=True, resume=True, fetch=blocked_fetch)
    assert failures["team_stats"] == [2024]
    assert len(attempted) == 1                                           # stopped at the first block
    assert "[2025, 2026]" in caplog.text                                 # says what was not attempted


def test_a_block_during_team_stats_skips_the_playoff_stage_entirely(tmp_path):
    attempted = []

    def blocked_fetch(url):
        attempted.append(url)
        raise FetchError("blocked")

    scrape_range(2025, 2025, tmp_path, include_playoffs=True, resume=True, fetch=blocked_fetch)
    assert not any("/playoffs/" in u for u in attempted)


def test_a_404_skips_that_season_and_continues(tmp_path):
    def fetch_with_one_404(url):
        if "NBA_2024" in url:
            raise NotFoundError("404")
        return valid_fetch(url)

    failures = scrape_range(2024, 2025, tmp_path, include_playoffs=False, resume=True, fetch=fetch_with_one_404)
    assert failures["team_stats"] == [2024]
    assert (tmp_path / "team_advanced_stats_2025.csv").exists()


def test_an_unparseable_page_skips_that_season_instead_of_crashing_the_run(tmp_path):
    misaligned = ('<table><thead><tr><th>Rk</th><th>Team</th><th>SRS</th><th>ORtg</th><th>DRtg</th></tr></thead><tbody>'
                  '<tr><th>1</th><td><a href="/teams/BOS/2024.html">Boston Celtics</a></td><td>8</td><td>120</td><td>111</td></tr>'
                  '<tr><th>2</th><td>No link here</td><td>1</td><td>110</td><td>110</td></tr></tbody></table>')

    def fetch(url):
        return misaligned if "NBA_2024" in url else valid_fetch(url)

    failures = scrape_range(2024, 2025, tmp_path, include_playoffs=False, resume=True, fetch=fetch)
    assert failures["team_stats"] == [2024]
    assert (tmp_path / "team_advanced_stats_2025.csv").exists()


# ---------------------------------------------------------------
# Review follow-ups: traceback kept, summary on every exit, checkpoint loader
# ---------------------------------------------------------------
def test_parse_failure_keeps_the_traceback_in_the_log(tmp_path, caplog):
    def exploding_fetch(url):
        return ("<table><thead><tr><th>Rk</th><th>Team</th><th>SRS</th><th>ORtg</th><th>DRtg</th></tr></thead><tbody>"
                '<tr><th>1</th><td><a href="/teams/BOS/2024.html">Boston</a></td><td>8</td><td>120</td><td>111</td></tr>'
                "<tr><th>2</th><td>No link</td><td>1</td><td>110</td><td>110</td></tr></tbody></table>")

    with caplog.at_level("ERROR"):
        scrape_range(2024, 2024, tmp_path, include_playoffs=False, resume=True, fetch=exploding_fetch)
    assert "Traceback" in caplog.text and "cannot align team codes" in caplog.text


def test_closing_summary_is_logged_when_the_run_is_blocked(tmp_path, caplog):
    def blocked_fetch(url):
        raise FetchError("blocked")

    with caplog.at_level("ERROR"):
        scrape_range(2024, 2026, tmp_path, include_playoffs=True, resume=True, fetch=blocked_fetch)
    assert "team_stats: 1 season(s) failed or were rejected: [2024]" in caplog.text


def test_closing_summary_is_logged_once_on_the_normal_path(tmp_path, caplog):
    def one_bad_season(url):
        return read_bad_team_page() if "NBA_2024" in url and "/playoffs/" not in url else valid_fetch(url)

    with caplog.at_level("ERROR"):
        scrape_range(2024, 2025, tmp_path, include_playoffs=False, resume=True, fetch=one_bad_season)
    assert caplog.text.count("season(s) failed or were rejected") == 1


def read_bad_team_page():
    """A well-formed table with only 5 teams: parses fine, fails the team-count check."""
    rows = "".join(
        f'<tr><th>{i}</th><td><a href="/teams/T{i}A/2024.html">Team {i}</a></td><td>{i}</td><td>{110 + i}</td><td>{110 - i}</td></tr>'
        for i in range(1, 6)
    )
    return ("<table><thead><tr><th>Rk</th><th>Team</th><th>SRS</th><th>ORtg</th><th>DRtg</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def test_no_summary_noise_when_everything_passes(tmp_path, caplog):
    with caplog.at_level("ERROR"):
        scrape_range(2024, 2025, tmp_path, include_playoffs=False, resume=True, fetch=valid_fetch)
    assert caplog.text == ""


def _season_checkpoints_with_2024_failing(tmp_path):
    def one_bad_season(url):
        return read_bad_team_page() if "NBA_2024" in url and "/playoffs/" not in url else valid_fetch(url)

    scrape_range(2023, 2025, tmp_path, include_playoffs=False, resume=True, fetch=one_bad_season)


def test_loader_concatenates_checkpoints_when_a_season_is_explicitly_excluded(tmp_path, caplog):
    _season_checkpoints_with_2024_failing(tmp_path)
    assert not (tmp_path / "team_advanced_stats_2023_2025.csv").exists()     # combined file refused...
    with caplog.at_level("WARNING"):
        df = load_checkpoints(tmp_path, "team_advanced_stats", 2023, 2025,
                              exclude={2024: "bad table in test"})           # ...loader still works
    assert sorted(df["season_end_year"].unique()) == [2023, 2025] and len(df) == 60
    assert "excluding 2024: bad table in test" in caplog.text


def test_loader_refuses_a_missing_season_that_was_not_excluded(tmp_path):
    _season_checkpoints_with_2024_failing(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"\[2024\]"):
        load_checkpoints(tmp_path, "team_advanced_stats", 2023, 2025, exclude={})


def test_loader_ignores_incomplete_and_combined_files(tmp_path):
    scrape_range(2024, 2025, tmp_path, include_playoffs=False, resume=True, fetch=valid_fetch)
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "team_advanced_stats_2024_INCOMPLETE.csv", index=False)
    df = load_checkpoints(tmp_path, "team_advanced_stats", 2024, 2025, exclude={})
    assert len(df) == 60            # per-season files only; neither the combined nor INCOMPLETE file is double-counted


def test_loader_parses_dates_for_playoff_games(tmp_path):
    scrape_range(2024, 2025, tmp_path, include_playoffs=True, resume=True, fetch=valid_fetch)
    games = load_checkpoints(tmp_path, "playoff_games", 2024, 2025, exclude={})
    assert pd.api.types.is_datetime64_any_dtype(games["date"])
    assert len(build_series_table(games)) == 30
