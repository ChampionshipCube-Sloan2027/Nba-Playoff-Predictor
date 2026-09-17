# NBA Playoff Predictor

A quantitative companion project to **The Championship Cube**. Where Championship Cube evaluates championship contention across a season using an evidence-based six-dimensional framework, this project asks a narrower, testable question: can pre-series team performance data predict individual playoff game outcomes, and does that prediction actually beat a naive baseline?

The project produces two related but distinct outputs — a validated game-level win probability, and a downstream Monte Carlo bracket simulation that chains those probabilities into estimated championship odds. See below for why that distinction matters.

## Repository structure

```
.
├── nba_scraper.py       # Basketball-Reference scraper: team advanced stats + playoff games
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Modelling code (feature engineering, baseline comparison, logistic regression, cross-validation, Monte Carlo simulation) will be added as each pipeline stage is built — see Status below for what currently exists.

## Data sources

- **Team advanced stats** (SRS, ORtg, DRtg, Net Rating, Pace, Four Factors): scraped from Basketball-Reference's season summary "Advanced Stats" table, as published for each season, 1989-90 through 2025-26.
- **Playoff game results** (date, matchup, home team, final score): scraped from Basketball-Reference box scores. Not yet collected — scraper implemented, pending live verification (see Status).
- **Context flags** (rule era, anomalous season, first-round format): not scraped. Hand-derived from documented NBA rule and schedule history — see `add_context_flags()` in `nba_scraper.py` for the exact season cutoffs used.

## Game-level model vs. bracket simulation: two connected but distinct outputs

This distinction matters and will be stated in the code, any reports, and here:

The **game-level model** predicts the winner of a single playoff game from each team's pre-series stats. Its performance will be reported directly — cross-validated, tested out of sample, and compared against a naive higher-seed baseline. This is the model's actual, measured accuracy.

The **bracket simulation** takes those game-level probabilities and runs them forward through a Monte Carlo simulation across a full series and bracket to estimate each team's championship odds. This is a downstream application of the model, not an independently validated instrument — its outputs compound the game-level model's own uncertainty across many simulated games. Bracket-level odds should be read as an exploratory estimate, not a calibrated probability of winning.

## Quickstart

```bash
git clone <this-repo>
cd nba-playoff-predictor
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python nba_scraper.py
```

What to expect: `scrape_team_advanced_stats()` has been checked against a live fetch of the season summary page and should run cleanly across the full range. `scrape_playoff_games()` is implemented against Basketball-Reference's known URL and box-score conventions but has not been verified against a live run — test it on a single season before trusting output across 1990–2026:

```python
from nba_scraper import scrape_playoff_games
scrape_playoff_games(2025)   # inspect the output before running the full range
```

## Status

*As of September 2026 — data collection stage.*

- Team advanced-stats scraper: built, structurally verified against a live page fetch. Not yet run across the full 1990–2026 range.
- Playoff-game scraper: built against known Basketball-Reference conventions. Not yet verified live — needs a single-season test before the full pull.
- No dataset has been finalised.
- No model has been trained. No accuracy, baseline comparison, or feature-importance result exists yet, and none will be reported here until the full pipeline — data collection, baseline, logistic regression, cross-validation, out-of-sample test — has actually been run.

This section will be updated as each stage is completed and verified, not in advance of it.

## License

MIT — see LICENSE.
