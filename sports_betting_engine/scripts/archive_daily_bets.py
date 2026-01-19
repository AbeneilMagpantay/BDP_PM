
import json
import os
from datetime import datetime
from pathlib import Path

# Paths - Go up 3 levels: scripts/ -> sports_betting_engine/ -> BDP_PM/ (repo root)
REPO_ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = DOCS_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"
HISTORY_FILE = DATA_DIR / "history.json"

def archive_bets():
    """
    Reads current predictions.json and archives new bets to history.json.
    """
    if not PREDICTIONS_FILE.exists():
        print("No predictions file found.")
        return

    # Load Predictions
    with open(PREDICTIONS_FILE, 'r') as f:
        data = json.load(f)
        
    current_bets = []
    
    # The structure is: { "sports": { "nfl": { "edges": [...] } } }
    sports_data = data.get('sports', {})
    
    for sport_key in ['nfl', 'nba', 'soccer']:
        sport_info = sports_data.get(sport_key, {})
        edges = sport_info.get('edges', [])
        
        for edge in edges:
            # Edge structure from update_dashboard.py:
            # matchup: "Away @ Home", bet_team, odds, ev, commence_time, bookmaker
            matchup_str = edge.get('matchup', 'Unknown @ Unknown')
            
            # Parse teams from "Away @ Home" format
            if ' @ ' in matchup_str:
                away_team, home_team = matchup_str.split(' @ ', 1)
            else:
                away_team, home_team = 'Unknown', matchup_str
            
            bet = {
                'sport': sport_key,
                'date': edge.get('commence_time', ''),
                'match': matchup_str,
                'pick': edge.get('bet_team', ''),
                'odds': edge.get('odds', 0),
                'ev': edge.get('ev', 0),
                'kelly': edge.get('kelly_bet', 0),  # Kelly % recommendation
                'bookmaker': edge.get('bookmaker', 'Unknown'),
                'home_team': home_team,
                'away_team': away_team
            }
            current_bets.append(bet)
    
    # Load History
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
        
    # Archive Logic
    # We identify bets by a unique hash of (Date + Match + Pick) to avoid duplicates
    # Since we don't have IDs in predictions, we generate one.
    
    new_bets_count = 0
    
    existing_ids = {h.get('id') for h in history}
    
    for bet in current_bets:
        # Generate ID using DATE ONLY (not time) to avoid duplicates from different prediction runs
        # Extract just the date portion from ISO timestamp
        date_str = bet['date'].split('T')[0] if 'T' in bet['date'] else bet['date'][:10]
        bet_id = f"{bet['sport']}_{date_str}_{bet['match']}_{bet['pick']}".replace(" ", "").replace("@", "").lower()
        
        if bet_id not in existing_ids:
            # Create History Entry
            entry = {
                "id": bet_id,
                "sport": bet['sport'],
                "date": bet['date'],
                "match": bet['match'],
                "pick": bet['pick'],
                "odds": bet['odds'],
                "ev": bet['ev'],
                "result": None,
                "profit": None,
                "status": "PENDING"
            }
            history.append(entry)
            existing_ids.add(bet_id)
            new_bets_count += 1
            
    # Save History
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"Archived {new_bets_count} new bets to history.json")

if __name__ == "__main__":
    archive_bets()
