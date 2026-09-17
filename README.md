# NBA Playoff Predictor

A companion project to **The Championship Cube**. Where Championship Cube asks
which team-building dimensions correlate with championship outcomes across a
season, this project narrows the question to something more falsifiable: can
team-level performance data predict individual playoff game outcomes — and
does that prediction actually beat a naive baseline?

## Research question

Given two teams' end-of-regular-season performance profiles, can we predict
the winner of a playoff game, and how does that prediction compare to (a) a
higher-seed-always-wins baseline and (b) market-implied odds where available?

## Methodology

This project follows a deliberately linear, falsifiable pipeline rather than
reporting a single headline accuracy number:

1. Research question & hypotheses
2. Data collection (Basketball-Reference, 1989-90 to 2025-26 seasons)
3. Feature engineering (Four Factors, efficiency ratings, SRS, pace, rest, seed, home court)
4. Baseline comparison (naive seed-based baseline; market odds where available)
5. Train/test methodology (chronological split — no look-ahead leakage)
6. Logistic regression baseline
7. Non-linear model (gradient boosting / random forest)
8. Cross-validation
9. Feature importance / coefficients
10. Out-of-sample performance vs. baseline
11. Failure analysis
12. Front-office implications (Monte Carlo simulation through series and bracket to championship odds)

## Scope

- **Target:** individual playoff game outcomes, chained via Monte Carlo
  simulation to series- and bracket-level championship probabilities
- **Features:** pre-series team-level stats only (no in-series state —
  a live win-probability model is a separate, later project)
- **Seasons:** 1989-90 through 2025-26
- **Controls:** explicit flags for rule era (illegal defense / hand-checking
  eras), anomalous seasons (1999 and 2012 lockouts, 2020 bubble, 2021
  reduced-crowd season), and playoff format (best-of-5 first round pre-2003
  vs. best-of-7 from 2003)
- **Excludes:** play-in tournament games

## Status

Data collection in progress. Team-level advanced stats scraper is
verified working; playoff game-level scraper is in testing.

## Setup

```bash
pip install -r requirements.txt
python nba_scraper.py
```

See `nba_scraper.py`'s docstring for rate-limiting notes and current
scraper reliability per function.

## Why this exists

Built to demonstrate a defensible research pipeline — not a claimed
accuracy figure — for sports analytics / research analyst roles, and as a
step toward the front-office analytics skillset (observe → measure → model
→ predict → decide).
