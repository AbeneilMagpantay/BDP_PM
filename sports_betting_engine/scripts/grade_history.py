import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.odds_fetcher import fetch_scores

# Paths
DOCS_DIR = project_root.parent / "docs" # Adjusted relative to root
DATA_DIR = DOCS_DIR / "data"
HISTORY_FILE = DATA_DIR / "history.json"

SPORT_KEY_MAP = {
    'nfl': 'americanfootball_nfl',
    'nba': 'basketball_nba',
    'soccer': 'soccer_epl'
}

def grade_bets():
    """
    Grades pending bets using real scores from The Odds API.
    """
    print(f"Starting grading process at {datetime.now()}...")
    
    if not HISTORY_FILE.exists():
        print("No history file found.")
        return

    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
        
    pending_bets = [b for b in history if b.get('status') == 'PENDING']
    
    if not pending_bets:
        print("No pending bets to grade.")
        return

    print(f"Found {len(pending_bets)} pending bets.")
    
    # optimize by fetching scores only for relevant sports
    needed_sports = set(b.get('sport_key', b.get('sport', 'nba')) for b in pending_bets)
    
    # Store scores: sport -> {(home, away): game_data}
    scores_cache = {}
    
    for sport in needed_sports:
        api_key = SPORT_KEY_MAP.get(sport, 'basketball_nba')
        print(f"Fetching scores for {sport} ({api_key})...")
        
        # Fetch last 3 days of scores
        games = fetch_scores(api_key, days_from=3)
        
        scores_map = {}
        for game in games:
            if not game.get('completed'):
                continue
                
            home = game.get('home_team')
            away = game.get('away_team')
            # Get game date (just the date part, not time)
            commence_time = game.get('commence_time', '')
            game_date = commence_time[:10] if commence_time else ''  # e.g., "2026-01-15"
            
            scores = game.get('scores', [])
            
            # Parse scores
            if scores:
                home_score = next((s['score'] for s in scores if s['name'] == home), None)
                away_score = next((s['score'] for s in scores if s['name'] == away), None)
                
                # Use (home, away, date) as key to avoid matching wrong games
                scores_map[(home, away, game_date)] = {
                    'home_score': int(home_score) if home_score is not None else 0,
                    'away_score': int(away_score) if away_score is not None else 0,
                    'completed': True
                }
        
        scores_cache[sport] = scores_map

    updated_count = 0
    
    for bet in pending_bets:
        sport = bet.get('sport_key', bet.get('sport', 'nba'))
        matchup_games = scores_cache.get(sport, {})
        
        # Try to find the game
        # bet['match'] usually "Away @ Home" or similar. 
        # But we stored 'home_team' and 'away_team' in edges, hopefully history has them?
        # If history structure lacks home/away explicitly, we might need to parse 'match' string.
        # Let's assume history objects have home_team/away_team or we parse.
        # Based on previous code, history items come from 'edges'.
        
        home = bet.get('home_team')
        away = bet.get('away_team')
        
        # Fallback parsing if keys missing (legacy data)
        if not home or not away:
             if '@' in bet.get('match', ''):
                 away, home = bet['match'].split(' @ ')
             else:
                 continue # Cannot identify teams

        # Get bet date (just the date part) for matching
        bet_date_str = bet.get('date', '')
        bet_date = bet_date_str[:10] if bet_date_str else ''  # e.g., "2026-01-15"
        
        # Look up using (home, away, date) to match correct game
        game_result = matchup_games.get((home, away, bet_date))
        
        if game_result and game_result['completed']:
            home_score = game_result['home_score']
            away_score = game_result['away_score']
            
            # Determine winner
            winner = home if home_score > away_score else away
            if home_score == away_score:
                winner = 'PUSH'
                
            pick = bet.get('pick')
            
            outcome = 'PUSH'
            if winner == 'PUSH':
                outcome = 'PUSH'
            elif pick == winner:
                outcome = 'WON'
            else:
                outcome = 'LOST'
            
            # Grading
            bet['result'] = outcome
            bet['status'] = 'GRADED'
            bet['score'] = f"{home_score}-{away_score}"
            
            # Calculate Profit
            bet_amt = 100 # Standard unit
            odds = bet.get('odds', -110)
            
            profit = 0.0
            if outcome == 'WON':
                if odds > 0:
                    profit = bet_amt * (odds / 100)
                else:
                    profit = bet_amt * (100 / abs(odds))
            elif outcome == 'LOST':
                profit = -bet_amt
                
            bet['profit'] = round(profit, 2)
            updated_count += 1
            print(f"Graded {bet['match']}: {outcome} (${bet['profit']})")

    if updated_count > 0:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Successfully updated {updated_count} bets.")
    else:
        print("No matches found/completed yet.")

if __name__ == "__main__":
    grade_bets()
