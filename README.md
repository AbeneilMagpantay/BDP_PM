# 🤖 AI Sports Betting Dashboard

**A fully autonomous, closed-loop AI system for detecting +EV (Expected Value) sports betting opportunities.**

![Dashboard Preview](https://via.placeholder.com/800x400?text=AI+Betting+Dashboard)

## 🚀 Key Features

### 1. Multi-Sport Intelligence
*   **NFL:** Trained on play-by-play data (2015-Present) using advanced efficiency metrics (EPA, DVOA).
*   **NBA:** Evaluates Pace, Offensive Rating, and Rest Days (Dynamic daily retraining).
*   **Soccer (EPL):** Analyzes form momentum and head-to-head history.

### 2. Autonomous "Closed Loop" System
This project runs entirely on **GitHub Actions**, requiring zero manual maintenance.
*   **Daily Retraining:** Every day, the AI retrains itself on the absolute latest data (including games played yesterday).
*   **Real-Time Market Scan:** Fetches live odds from 40+ bookmakers via **TheOddsApi** to identify discrepancies.
*   **Auto-Grading:** Automatically tracks every recommended bet, checks the real game score the next day, and grades it (WON/LOST) to track long-term performance.

### 3. +EV Discovery Engine
The system only recommends bets where the **AI's calculated win probability** is significantly higher than the **implied probability** of the bookmaker's odds.
*   **Edge:** The percentage difference between AI confidence and Market price.
*   **Kelly Criterion:** Suggested bankroll allocation for each bet.

## 🛠️ Technology Stack
*   **Core:** Python 3.10
*   **ML:** XGBoost, Scikit-Learn, Pandas
*   **Data Sources:** nfl_data_py, nba_api, soccerdata, TheOddsApi
*   **Automation:** GitHub Actions (Cron Jobs)
*   **Frontend:** HTML5/CSS3 (Hosted on Vercel/GitHub Pages)

## 🔄 Workflow (Daily Cycle)

1.  **08:00 AM (PH Time):** GitHub Action wakes up.
2.  **Retrain:** Scripts fetch the very latest season data (e.g., 2026 Season) and rebuild the XGBoost models.
3.  **Scan:** The EV Engine fetches live odds and compares them against the new models.
4.  **Publish:** Profitable opportunities are pushed to `docs/data/predictions.json`.
5.  **Grade:** The Archiver checks yesterday's pending bets, pulls real scores from the API, updates `history.json`, and calculates profit.
6.  **Deploy:** Vercel automatically deploys the updated dashboard.

## 📂 Project Structure

```
├── .github/workflows/    # Automation logic (daily_update.yml)
├── docs/                 # Frontend (index.html) & Data storage
│   ├── data/
│   │   ├── predictions.json  # Current Live Bets
│   │   └── history.json     # All-time Graded Record
├── nfl_ev_betting_engine/
│   ├── scripts/          # Core execution scripts
│   │   ├── train_model.py       # NFL Training
│   │   ├── update_dashboard.py  # Odds Fetching & EV Calc
│   │   └── grade_history.py     # Score Checking & Grading
│   └── src/              # Shared library code
```

## 🔧 Setup & Installation

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/AbeneilMagpantay/BDP_PM.git
    cd BDP_PM
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r nfl_ev_betting_engine/requirements.txt
    ```

3.  **Set Environment Variables (.env):**
    ```
    ODDS_API_KEY=your_api_key_here
    ```

4.  **Run Locally:**
    ```bash
    python nfl_ev_betting_engine/scripts/update_dashboard.py
    ```

---
*Auto-generated & Maintained by Antigravity*
