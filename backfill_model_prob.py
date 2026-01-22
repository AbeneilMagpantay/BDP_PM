"""
Backfill model_prob into history.json from predictions.json.
This script adds model_prob to existing history entries that are missing it.
"""
import json
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs" / "data"
HISTORY_FILE = DOCS_DIR / "history.json"
PREDICTIONS_FILE = DOCS_DIR / "predictions.json"

def backfill():
    # Load history
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    # Load predictions
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)
    
    # Build lookup from predictions edges: key = (match, pick) -> model_prob
    edge_lookup = {}
    
    for sport_key in ['nfl', 'nba', 'soccer']:
        sport_data = predictions.get('sports', {}).get(sport_key, {})
        edges = sport_data.get('edges', [])
        
        for edge in edges:
            match = edge.get('matchup', '')
            pick = edge.get('bet_team', '')
            model_prob = edge.get('model_prob', 0)
            
            if match and pick and model_prob:
                key = (match, pick)
                edge_lookup[key] = model_prob
    
    print(f"Loaded {len(edge_lookup)} edges from predictions.json")
    
    # Update history entries
    updated_count = 0
    for entry in history:
        # Skip if already has model_prob
        if entry.get('model_prob') and entry['model_prob'] != 0:
            continue
        
        match = entry.get('match', '')
        pick = entry.get('pick', '')
        key = (match, pick)
        
        if key in edge_lookup:
            entry['model_prob'] = edge_lookup[key]
            updated_count += 1
            print(f"  Updated: {match} -> {pick} = {edge_lookup[key]:.2f}")
    
    print(f"\nUpdated {updated_count} history entries")
    
    # Save history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    
    print("Saved history.json")

if __name__ == "__main__":
    backfill()
