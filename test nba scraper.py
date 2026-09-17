"""
Unit tests for the parts of nba_scraper.py that don't need a live network
call. These are the parts that get silently wrong logic if you're not
careful with boundary years - exactly the kind of bug that produces a
plausible-looking but incorrect dataset.
"""

import pandas as pd

from nba_scraper import (
    add_context_flags,
    classify_rule_era,
    flatten_columns,
    get_all_tables,
    parse_box_score_slug,
    validate_team_stats,
)


# ---- parse_box_score_slug ----

def test_parse_box_score_slug_basic():
    url = "https://www.basketball-reference.com/boxscores/202606130SAS.html"
    date, home = parse_box_score_slug(url)
    assert date == "2026-06-13"
    assert home == "SAS"


def test_parse_box_score_slug_three_letter_codes():
    url = "https://www.basketball-reference.com/boxscores/199906190NYK.html"
    date, home = parse_box_score_slug(url)
    assert date == "1999-06-19"
    assert home == "NYK"


# ---- classify_rule_era ----

def test_classify_rule_era_boundaries():
    assert classify_rule_era(1990) == "illegal_defense_era"
    assert classify_rule_era(2001) == "illegal_defense_era"
    assert classify_rule_era(2002) == "post_illegal_defense_pre_fom"
    assert classify_rule_era(2004) == "post_illegal_defense_pre_fom"
    assert classify_rule_era(2005) == "freedom_of_movement_era"
    assert classify_rule_era(2026) == "freedom_of_movement_era"


# ---- add_context_flags ----

def test_add_context_flags_anomalous_seasons():
    df = pd.DataFrame({"season_end_year": [1998, 1999, 2000, 2012, 2020, 2021, 2022]})
    out = add_context_flags(df)
    expected = [False, True, False, True, True, True, False]
    assert list(out["anomalous_season"]) == expected


def test_add_context_flags_shortened_3pt_line():
    df = pd.DataFrame({"season_end_year": [1994, 1995, 1996, 1997, 1998]})
    out = add_context_flags(df)
    assert list(out["shortened_3pt_line"]) == [False, True, True, True, False]


def test_add_context_flags_first_round_format():
    df = pd.DataFrame({"season_end_year": [2001, 2002, 2003, 2004]})
    out = add_context_flags(df)
    assert list(out["first_round_best_of_5"]) == [True, True, False, False]


def test_add_context_flags_does_not_mutate_input():
    df = pd.DataFrame({"season_end_year": [2010]})
    original_columns = list(df.columns)
    add_context_flags(df)
    assert list(df.columns) == original_columns  # caller's df is untouched


# ---- flatten_columns ----

def test_flatten_columns_multiindex():
    df = pd.DataFrame(
        [[0.55, "Lakers"]],
        columns=pd.MultiIndex.from_tuples(
            [("Offense Four Factors", "eFG%"), ("Unnamed: 1_level_0", "Team")]
        ),
    )
    out = flatten_columns(df)
    assert "Offense Four Factors_eFG%" in out.columns
    assert "Team" in out.columns  # Unnamed level dropped, not literalised


def test_flatten_columns_leaves_normal_columns_alone():
    df = pd.DataFrame([[1, 2]], columns=["Team", "SRS"])
    out = flatten_columns(df)
    assert list(out.columns) == ["Team", "SRS"]


# ---- get_all_tables ----

def test_get_all_tables_finds_commented_out_table():
    html = """
    <html><body>
    <!--
    <table id="hidden"><tr><th>SRS</th><th>ORtg</th></tr><tr><td>1.2</td><td>110</td></tr></table>
    -->
    </body></html>
    """
    tables = get_all_tables(html)
    assert len(tables) == 1
    assert "SRS" in tables[0].columns


def test_get_all_tables_finds_normal_table_too():
    html = """
    <html><body>
    <table id="visible"><tr><th>Team</th><th>W</th></tr><tr><td>Celtics</td><td>60</td></tr></table>
    </body></html>
    """
    tables = get_all_tables(html)
    assert len(tables) == 1
    assert "Team" in tables[0].columns


def test_get_all_tables_returns_empty_list_for_no_tables():
    html = "<html><body><p>No tables here.</p></body></html>"
    assert get_all_tables(html) == []


# ---- validate_team_stats ----

def test_validate_team_stats_warns_on_low_row_count(caplog):
    df = pd.DataFrame({"Team": ["A", "B"]})  # only 2 teams - clearly wrong
    with caplog.at_level("WARNING"):
        validate_team_stats(df, 2025)
    assert "expected roughly" in caplog.text


def test_validate_team_stats_silent_on_normal_row_count(caplog):
    df = pd.DataFrame({"Team": [f"Team{i}" for i in range(29)]})
    with caplog.at_level("WARNING"):
        validate_team_stats(df, 2025)
    assert caplog.text == ""
