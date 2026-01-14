"""
Generate Web Predictions
========================

Generates predictions in JSON format for the static website.
Outputs to docs/data/ for GitHub Pages compatibility.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use same imports as daily_runner.py
from src.data.nfl_data_fetcher import aggregate_to_game_stats, fetch_play_by_play_data
from src.data.odds_fetcher import fetch_nfl_odds, parse_odds_for_game
from src.data.preprocessor import get_team_recent_stats
from src.model.predictor import GamePredictionService
from src.betting.edge_detector import EdgeDetector


# Team name mapping
TEAM_NAME_MAP = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}


def get_team_abbrev(full_name: str) -> str:
    return TEAM_NAME_MAP.get(full_name, full_name)


def main():
    print("=" * 60)
    print("GENERATING WEB PREDICTIONS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create output directory (relative to repo root)
    output_dir = project_root.parent / "docs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Fetch odds
        print("\n📊 Fetching live NFL odds...")
        raw_odds = fetch_nfl_odds()
        
        if not raw_odds:
            print("No games found.")
            save_empty_predictions(output_dir)
            return
        
        # Parse odds into structured format for edge detector
        parsed_odds = [parse_odds_for_game(game) for game in raw_odds]
        print(f"   Found {len(parsed_odds)} upcoming games")
        
        # Step 2: Load model
        print("\n🤖 Loading prediction model...")
        prediction_service = GamePredictionService()
        model_accuracy = 0.86  # Known from training
        print(f"   Model loaded")
        
        # Step 3: Get team stats
        print("\n📈 Fetching recent team statistics...")
        current_year = datetime.now().year
        seasons = [current_year - 1, current_year]
        
        pbp_data = fetch_play_by_play_data(seasons)
        game_stats = aggregate_to_game_stats(pbp_data) if pbp_data is not None else None
        
        # Step 4: Generate predictions
        print("\n🎯 Generating predictions...")
        predictions = []
        
        for game in parsed_odds:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            home_abbrev = get_team_abbrev(home_team)
            away_abbrev = get_team_abbrev(away_team)
            
            home_stats = get_team_recent_stats(game_stats, home_abbrev)
            away_stats = get_team_recent_stats(game_stats, away_abbrev)
            
            if home_stats is None or away_stats is None:
                print(f"   ⚠️ Skipping {away_team} @ {home_team} - missing stats")
                continue
            
            prediction = prediction_service.predict_game(home_stats, away_stats)
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': prediction['home_win_prob'],
                'away_win_prob': prediction['away_win_prob'],
                'confidence': prediction['confidence'],
                'commence_time': game.get('commence_time', ''),
            })
        
        print(f"   Generated predictions for {len(predictions)} games")
        
        # Step 5: Find edges
        print("\n🔍 Scanning for +EV opportunities...")
        edge_detector = EdgeDetector()
        
        edges_result = edge_detector.find_edges(predictions, parsed_odds)
        edges = edge_detector.get_edges_as_dicts(edges_result)
        
        print(f"   Found {len(edges)} +EV opportunities!")
        
        # Format predictions for JSON output (convert to percentages)
        predictions_output = []
        for pred in predictions:
            predictions_output.append({
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'home_win_prob': round(pred['home_win_prob'] * 100, 1),
                'away_win_prob': round(pred['away_win_prob'] * 100, 1),
                'confidence': round(pred['confidence'] * 100, 1),
                'commence_time': pred['commence_time'],
            })
        
        # Format edges for JSON output
        edges_output = []
        for edge in edges:
            edges_output.append({
                'home_team': edge.get('matchup', '').split(' @ ')[1] if ' @ ' in edge.get('matchup', '') else '',
                'away_team': edge.get('matchup', '').split(' @ ')[0] if ' @ ' in edge.get('matchup', '') else '',
                'bet_on': edge.get('bet_team'),
                'side': edge.get('bet_side'),
                'odds': edge.get('odds'),
                'bookmaker': edge.get('bookmaker'),
                'model_prob': round(edge.get('model_prob', 0) * 100, 1),
                'implied_prob': round(edge.get('implied_prob', 0) * 100, 1),
                'edge': round(edge.get('edge', 0), 1),
                'ev': round(edge.get('ev', 0), 1),
                'kelly': round(edge.get('kelly_bet', 0), 1),
                'commence_time': edge.get('commence_time', '')
            })
        
        # Step 6: Save to JSON
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'model_accuracy': model_accuracy,
            'total_games': len(predictions_output),
            'total_edges': len(edges_output),
            'predictions': predictions_output,
            'edges': edges_output
        }
        
        # Save current predictions
        predictions_file = output_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Saved predictions to {predictions_file}")
        
        # Append to history
        history_file = output_dir / "history.json"
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_edges': len(edges_output),
            'edges': edges_output
        })
        
        # Keep last 30 days
        history = history[-30:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Updated history ({len(history)} days)")
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        save_empty_predictions(output_dir)


def save_empty_predictions(output_dir):
    """Save empty predictions when no data available."""
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'model_accuracy': 0,
        'total_games': 0,
        'total_edges': 0,
        'predictions': [],
        'edges': [],
        'error': 'No games available or error occurred'
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
