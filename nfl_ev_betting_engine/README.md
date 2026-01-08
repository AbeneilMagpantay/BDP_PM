# NFL EV Betting Engine 🏈💰

A quantitative sports betting engine that finds **Expected Value (EV)** opportunities in NFL games by comparing machine learning predictions against Vegas odds.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)

## 🎯 What This Does

1. **Ingests NFL Data** - Historical play-by-play data via `nfl_data_py`
2. **Trains ML Model** - XGBoost model predicts game outcomes
3. **Fetches Live Odds** - Real-time odds from The Odds API
4. **Calculates EV** - Compares model probability vs implied odds
5. **Sends Alerts** - Discord notifications for +EV opportunities

## 📊 Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   nfl_data_py   │     │  The Odds API   │
│  (Historical)   │     │  (Live Odds)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Data Pipeline & Preprocessing    │
│  • Game-level aggregation               │
│  • Feature engineering                  │
│  • Rolling averages                     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│           XGBoost Predictor             │
│  • Win probability for each team        │
│  • Trained on 4+ years of data          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│            EV Calculator                │
│  • Compare model vs Vegas odds          │
│  • Calculate edge & expected value      │
│  • Kelly Criterion bet sizing           │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│           Discord Alerts                │
│  • Real-time +EV notifications          │
│  • Rich embeds with all details         │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/nfl_ev_betting_engine.git
cd nfl_ev_betting_engine

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example config
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your keys:
# ODDS_API_KEY=your_key_here        (get free at https://the-odds-api.com/)
# DISCORD_WEBHOOK_URL=your_url_here (optional, for alerts)
```

### 3. Train the Model

```bash
python scripts/train_model.py
```

This will:
- Download NFL play-by-play data (2021-2024)
- Engineer features (EPA, success rate, etc.)
- Train XGBoost with hyperparameter tuning
- Save the model to `data/models/`

### 4. Run Daily Analysis

```bash
python scripts/daily_runner.py
```

This will:
- Fetch current NFL odds
- Generate predictions for each game
- Find +EV opportunities
- Send Discord alerts (if configured)

## 📁 Project Structure

```
nfl_ev_betting_engine/
├── src/
│   ├── data/
│   │   ├── nfl_data_fetcher.py   # Historical NFL data
│   │   ├── odds_fetcher.py       # Live odds from API
│   │   └── preprocessor.py       # Feature engineering
│   ├── model/
│   │   ├── features.py           # Feature definitions
│   │   ├── trainer.py            # XGBoost training
│   │   └── predictor.py          # Prediction interface
│   ├── betting/
│   │   ├── ev_calculator.py      # EV math
│   │   └── edge_detector.py      # Find opportunities
│   └── alerts/
│       └── discord_notifier.py   # Discord webhooks
├── scripts/
│   ├── train_model.py            # Model training
│   └── daily_runner.py           # Daily automation
├── tests/
│   └── test_ev_calculator.py
├── data/
│   ├── raw/                      # Cached raw data
│   ├── processed/                # Processed features
│   └── models/                   # Saved models
├── run_analysis.py               # CLI entry point
├── requirements.txt
└── README.md
```

## 🧮 EV Calculation Explained

**Expected Value (EV)** tells you the average profit/loss per dollar wagered:

```
EV = (Win Probability × Profit) - (Lose Probability × Stake)
```

### Example:
- **Your Model**: 55% chance home team wins
- **Odds**: -110 (implies 52.4%)
- **Edge**: 55% - 52.4% = **2.6%**
- **EV**: +4.5% (you expect to profit $4.50 per $100 bet long-term)

A **+EV bet** means the odds are in your favor over the long run.

## 🔧 Configuration

Edit `.env` to customize:

```bash
# API Keys
ODDS_API_KEY=your_api_key_here
DISCORD_WEBHOOK_URL=your_webhook_url_here

# Thresholds
EV_THRESHOLD=5.0          # Minimum EV % to alert
MIN_CONFIDENCE=0.55       # Minimum model confidence
```

## 📈 Model Performance

Target metrics on validation set:
- **Accuracy**: > 55% (better than coin flip)
- **ROC AUC**: > 0.55
- **Log Loss**: < 0.68

The model uses these features:
- Yards per play (home/away)
- EPA per play (Expected Points Added)
- Success rate
- Turnover differential
- Pass rate

## 🤖 Automation

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (Daily at 9 AM)
4. Action: Start a Program
   - Program: `python`
   - Arguments: `scripts/daily_runner.py`
   - Start in: `C:\path\to\nfl_ev_betting_engine`

### Linux/Mac Cron

```bash
# Edit crontab
crontab -e

# Add line (runs daily at 9 AM)
0 9 * * * cd /path/to/nfl_ev_betting_engine && python scripts/daily_runner.py
```

## 📚 Dependencies

- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - ML utilities
- **xgboost** - Gradient boosting model
- **nfl_data_py** - NFL data source
- **requests** - API calls
- **python-dotenv** - Environment variables

## ⚠️ Disclaimer

This project is for **educational purposes only**. Sports betting involves risk and you should only bet what you can afford to lose. Past performance does not guarantee future results. Always gamble responsibly.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

---

Built with ❤️ for data-driven sports analysis
