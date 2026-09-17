# NBA Playoff Predictor

A quantitative companion project to **The Championship Cube**. Where Championship Cube evaluates championship contention across a season using an evidence-based six-dimensional framework, this project asks a narrower, testable question: can pre-series team performance data predict individual playoff game outcomes, and does that prediction actually beat a naive baseline?

The project produces two related but distinct outputs — a validated game-level win probability, and a downstream Monte Carlo bracket simulation that chains those probabilities into estimated championship odds. See below for why that distinction matters.

## Repository structure

```
.
├── nba_scraper.py         # Basketball-Reference scraper: team advanced stats + playoff games
├── test_nba_scraper.py    # Unit tests for all network-free logic
├── requirements.txt
├── requirements-dev.txt   # requirements.txt + pytest
├── .gitignore
├── LICENSE
└── README.md
```

Modelling code (feature engineering, baseline comparison, logistic regression, cross-validation, Monte Carlo simulation) will be added as each pipeline stage is built — see Status below for what currently exists.

## Data sources

- **Team advanced stats** (SRS, ORtg, DRtg, Net Rating, Pace, Four Factors): scraped from Basketball-Reference's season summary "Advanced Stats" table, as published for each season, 1989-90 through 2025-26.
- **Playoff game results** (date, matchup, home team, final score): scraped from Basketball-Reference box scores. Not yet collected — scraper implemented, pending live verification (see Status).
- **Context flags** (rule era, anomalous season, first-round format): not scraped. Hand-derived from documented NBA rule and schedule history — see `classify_rule_era()` and `add_context_flags()` in `nba_scraper.py` for the exact season cutoffs used.

## Game-level model vs. bracket simulation: two connected but distinct outputs

This distinction matters and will be stated in the code, any reports, and here:

The **game-level model** predicts the winner of a single playoff game from each team's pre-series stats. Its performance will be reported directly — cross-validated, tested out of sample, and compared against a naive higher-seed baseline. This is the model's actual, measured accuracy.

The **bracket simulation** takes those game-level probabilities and runs them forward through a Monte Carlo simulation across a full series and bracket to estimate each team's championship odds. This is a downstream application of the model, not an independently validated instrument — its outputs compound the game-level model's own uncertainty across many simulated games. Bracket-level odds should be read as an exploratory estimate, not a calibrated probability of winning.

## Testing

The parts of `nba_scraper.py` that don't require a live network call (rule-era classification, anomalous-season flags, box-score URL parsing, HTML table extraction, row-count validation) are covered by `test_nba_scraper.py` — 14 tests, currently passing.

This isn't a formality: running this suite caught two real bugs before any live scraping was attempted —

1. Box-score URLs encode a single-digit game index between the date and the team code (`.../202606130SAS.html`). The original date/team slicing didn't account for it and would have silently prefixed every home team code with a stray `0` (`"0SAS"` instead of `"SAS"`).
2. Basketball-Reference tables hidden inside HTML comments require the raw HTML to be parsed correctly — the original code passed a bare string to `pandas.read_html()`, which current pandas versions interpret as a filename to open, not literal HTML, causing a hard failure.

Both are the kind of bug that produces a plausible-looking but wrong dataset if it isn't caught before a real run, not after one.

```bash
pip install -r requirements-dev.txt
pytest test_nba_scraper.py -v
```

## Quickstart

```bash
git clone <this-repo>
cd nba-playoff-predictor
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python nba_scraper.py                                          # team stats only, full range
python nba_scraper.py --include-playoffs                       # + playoff games
python nba_scraper.py --start-year 2025 --end-year 2025 --include-playoffs   # test run
```

The scraper checkpoints per season and resumes automatically if interrupted — safe to stop and restart on a multi-hour run. `--no-resume` forces a full re-scrape.

What to expect: `scrape_team_advanced_stats()` has been checked against a live fetch of the season summary page and should run cleanly across the full range. `scrape_playoff_games()` is implemented against Basketball-Reference's known URL and box-score conventions but has not been verified against a live run — test it on a single season (as above) and inspect the CSV before trusting output across 1990–2026.

## Status

*As of September 2026 — data collection stage.*

- Team advanced-stats scraper: built, structurally verified against a live page fetch. Not yet run across the full 1990–2026 range.
- Playoff-game scraper: built against known Basketball-Reference conventions. Not yet verified live — needs a single-season test before the full pull.
- Network-free logic (era classification, URL parsing, table extraction, validation): unit tested, 14/14 passing, two real bugs already caught and fixed pre-live-run.
- No dataset has been finalised.
- No model has been trained. No accuracy, baseline comparison, or feature-importance result exists yet, and none will be reported here until the full pipeline — data collection, baseline, logistic regression, cross-validation, out-of-sample test — has actually been run.

This section will be updated as each stage is completed and verified, not in advance of it.

## License

MIT — see LICENSE.
