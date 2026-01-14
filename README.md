# NFL EV Betting Engine

[![CI Status](https://github.com/AbeneilMagpantay/BDP_PM/actions/workflows/daily_update.yml/badge.svg)](https://github.com/AbeneilMagpantay/BDP_PM/actions/workflows/daily_update.yml)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An automated Expected Value (+EV) sports betting system that leverages machine learning to identify discrepancies between model probabilities and bookmaker odds.

The system operates as a closed-loop pipeline on GitHub Actions, performing daily data ingestion, model retraining, inference, and result grading without manual intervention.

## Architectural Overview

The project consists of three main components:

1.  **Data Pipeline & Retraining**: 
    - Fetches historical play-by-play data (NFL), advanced stats (NBA), and match results (Soccer).
    - Dynamically detects the current season context to ensure models train on the latest available data.
    - Rebuilds XGBoost classification models daily to adapt to recent team form.

2.  **Inference Engine**:
    - Queries **The Odds API** for live spreads, money lines, and totals across 40+ bookmakers.
    - Calculates the "Edge" (difference between Model Win % and Implied Odds %).
    - Applies the Kelly Criterion to determine optimal bankroll allocation.

3.  **Automated Accountability**:
    - Archives daily predictions to a persistent history file.
    - Grades previous bets using confirmed game scores fetched from external APIs.
    - Tracks aggregate performance (ROI, Win Rate) over time.

## Directory Structure

```bash
├── .github/workflows/    # CI/CD definitions (cron schedule)
├── docs/                 # Static dashboard assets (GitHub Pages root)
│   ├── data/             # JSON artifacts (predictions, history)
├── nfl_ev_betting_engine/
│   ├── scripts/          # ETL and execution scripts
│   ├── src/              # Core library (models, fetchers, utils)
```

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/AbeneilMagpantay/BDP_PM.git
cd BDP_PM
pip install -r nfl_ev_betting_engine/requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```ini
ODDS_API_KEY=your_api_key_here
```

## Usage

### Manual Execution

To run the full update cycle locally:

```bash
# 1. Update dashboard (Fetch odds -> Predict -> Save JSON)
python nfl_ev_betting_engine/scripts/update_dashboard.py

# 2. Grade historical bets
python nfl_ev_betting_engine/scripts/grade_history.py
```

### Automation

The system is configured to run automatically via **GitHub Actions** (`.github/workflows/daily_update.yml`):
- **Cron**: Runs daily at 00:00 UTC and 09:00 UTC.
- **Triggers**: Auto-updates whenever changes are pushed to `main`.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. I am not responsible for any financial losses incurred from using this software.
