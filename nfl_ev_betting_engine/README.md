# NFL EV Betting Engine

Finding edge in NFL betting using data + machine learning.

## What is this?

A Python project that:
1. Pulls NFL play-by-play data (via `nfl_data_py`)
2. Trains an XGBoost model to predict game outcomes
3. Fetches live odds from sportsbooks
4. Compares model predictions vs Vegas lines
5. Alerts you on Discord when it finds +EV bets

Built as a portfolio project for a sports betting data engineer role.

## How it works

```
NFL Data (nfl_data_py) ──┐
                         ├──> XGBoost Model ──> Compare vs Odds ──> Discord Alert
Live Odds (The Odds API)─┘
```

The model looks at team efficiency metrics (EPA, success rate, turnovers) to predict win probability. If the model thinks a team has a 60% chance but the odds only imply 50%, that's a +EV opportunity.

## Setup

```bash
# Clone and install
git clone https://github.com/AbeneilMagpantay/BDP_PM.git
cd BDP_PM/nfl_ev_betting_engine
pip install -r requirements.txt

# Add your API key (get free one at the-odds-api.com)
cp .env.example .env
# edit .env with your keys

# Train the model
python scripts/train_model.py --fast

# Run it
python scripts/daily_runner.py
```

## Project structure

```
nfl_ev_betting_engine/
├── src/
│   ├── data/          # data fetching + preprocessing
│   ├── model/         # XGBoost training + prediction
│   ├── betting/       # EV math
│   └── alerts/        # Discord webhook
├── scripts/
│   ├── train_model.py
│   └── daily_runner.py
└── tests/
```

## Model performance

After training on 2021-2024 data:
- Accuracy: ~86%
- ROC AUC: 0.94

Note: These numbers are on historical data. Real-world betting performance would be lower since bookmakers already price in most of this information.

## EV calculation

```python
# Basic idea
model_prob = 0.55  # model says 55% chance
implied_prob = 0.50  # odds imply 50%
edge = model_prob - implied_prob  # 5% edge
```

If edge > 0, bet has positive expected value.

## Key files

| File | What it does |
|------|--------------|
| `src/data/nfl_data_fetcher.py` | Pulls play-by-play data |
| `src/data/odds_fetcher.py` | Gets live odds from API |
| `src/model/trainer.py` | Trains XGBoost model |
| `src/betting/ev_calculator.py` | EV math (odds conversion, Kelly) |
| `scripts/daily_runner.py` | Main script - run this daily |

## Limitations

- Model is overconfident (needs calibration)
- Doesn't account for injuries, weather, etc.
- Historical accuracy != future performance
- Bookmakers are good at their job

## Future improvements

- Add injury data
- Calibrate probability outputs
- Track actual betting performance
- Add spread/totals predictions (currently just moneyline)

## License

MIT
