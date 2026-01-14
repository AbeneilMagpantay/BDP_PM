
import json
import os
from datetime import datetime
from pathlib import Path

# Paths
DOCS_DIR = Path(__file__).parent.parent / "docs"
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
    
    # Flatten bets from all sports
    for sport in ['nfl', 'nba', 'soccer']:
        if sport in data and isinstance(data[sport], list):
            for bet in data[sport]:
                bet['sport'] = sport
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
        # Generate ID
        bet_id = f"{bet['sport']}_{bet['date']}_{bet['match']}_{bet['pick']}".replace(" ", "").replace("@", "").lower()
        
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
