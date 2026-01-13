
import json
import random
from pathlib import Path

# Paths
DOCS_DIR = Path(__file__).parent.parent / "docs"
DATA_DIR = DOCS_DIR / "data"
HISTORY_FILE = DATA_DIR / "history.json"

def grade_bets():
    """
    Simulates grading pending bets.
    In a real system, this would query an API.
    Here, it randomly assigns WON/LOST to demonstrate flow.
    """
    if not HISTORY_FILE.exists():
        print("No history file found.")
        return

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
        
    updated_count = 0
    
    for bet in history:
        if bet['status'] == 'PENDING':
            # Mock Logic: Randomly decide win/loss
            # Bias toward winning for demo (60% win rate)
            outcome = 'WON' if random.random() > 0.4 else 'LOST'
            
            bet['result'] = outcome
            bet['status'] = 'GRADED'
            
            # Calculate Profit
            # Parsing odds: "+203" or "-110"
            try:
                odds_str = str(bet['odds'])
                if odds_str.startswith('+'):
                    odds_val = float(odds_str[1:])
                    multiplier = odds_val / 100
                elif odds_str.startswith('-'):
                    odds_val = float(odds_str[1:])
                    multiplier = 100 / odds_val
                else:
                    multiplier = 1.0
                
                if outcome == 'WON':
                    bet['profit'] = round(100 * multiplier, 2) # Assume $100 unit size
                else:
                    bet['profit'] = -100.00
            except:
                bet['profit'] = 0.0
                
            updated_count += 1
            
    if updated_count > 0:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Graded {updated_count} pending bets.")
    else:
        print("No pending bets to grade.")

if __name__ == "__main__":
    grade_bets()
