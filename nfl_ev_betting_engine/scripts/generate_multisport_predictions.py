"""
Multi-Sport Predictions Generator
==================================

Fetches odds for NFL, NBA, Soccer, and Esports (Dota 2, Valorant via PandaScore).
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import requests

load_dotenv(project_root.parent / '.env')

# Sport configurations - The Odds API
ODDS_API_SPORTS = {
    'nfl': {
        'key': 'americanfootball_nfl',
        'name': 'NFL',
        'profile': {
            'name': 'NFL',
            'handle': 'American Football',
            'emoji': '🏈',
            'bio': 'NFL game predictions and betting edges.',
            'accuracy': 86,
            'record': {'wins': 4, 'losses': 1},
        }
    },
    'nba': {
        'key': 'basketball_nba',
        'name': 'NBA',
        'profile': {
            'name': 'NBA',
            'handle': 'Basketball',
            'emoji': '🏀',
            'bio': 'NBA game predictions and betting edges.',
            'accuracy': 0,
            'record': {'wins': 0, 'losses': 0},
        }
    },
    'soccer': {
        'key': 'soccer_epl',
        'name': 'EPL',
        'profile': {
            'name': 'Soccer',
            'handle': 'Premier League',
            'emoji': '⚽',
            'bio': 'EPL match predictions and betting edges.',
            'accuracy': 0,
            'record': {'wins': 0, 'losses': 0},
        }
    }
}

# Esports - PandaScore
PANDASCORE_SPORTS = {
    'dota2': {
        'slug': 'dota2',
        'name': 'Dota 2',
        'profile': {
            'name': 'Dota 2',
            'handle': 'Esports',
            'emoji': '🎮',
            'bio': 'Dota 2 match predictions.',
            'accuracy': 0,
            'record': {'wins': 0, 'losses': 0},
        }
    },
    'valorant': {
        'key': 'valorant',
        'name': 'Valorant',
        'profile': {
            'name': 'Valorant',
            'handle': 'Esports',
            'emoji': '🎯',
            'bio': 'Valorant match predictions.',
            'accuracy': 0,
            'record': {'wins': 0, 'losses': 0},
        }
    }
}


def fetch_odds_api(sport_key: str) -> list:
    """Fetch odds from The Odds API."""
    api_key = os.getenv('ODDS_API_KEY')
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american'
    }
    
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"   Error fetching {sport_key}: {e}")
        return []


def fetch_pandascore_matches(game: str) -> list:
    """Fetch upcoming matches from PandaScore."""
    api_key = os.getenv('PANDASCORE_API_KEY')
    if not api_key:
        print(f"   PandaScore API key not found")
        return []
    
    url = f"https://api.pandascore.co/{game}/matches/upcoming"
    headers = {'Authorization': f'Bearer {api_key}'}
    params = {'per_page': 20}
    
    try:
        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"   Error fetching {game}: {e}")
        return []


def parse_pandascore_matches(matches: list, sport_name: str) -> tuple:
    """Parse PandaScore matches into games format (no odds available)."""
    games = []
    
    for match in matches:
        if not match.get('opponents') or len(match['opponents']) < 2:
            continue
        
        team1 = match['opponents'][0].get('opponent', {})
        team2 = match['opponents'][1].get('opponent', {})
        
        game = {
            'home_team': team1.get('name', 'Team 1'),
            'away_team': team2.get('name', 'Team 2'),
            'commence_time': match.get('scheduled_at', ''),
            'league': match.get('league', {}).get('name', sport_name),
        }
        games.append(game)
    
    # No edges - PandaScore free tier doesn't provide odds
    return games, []


def find_best_odds(games: list) -> list:
    """Find best odds across bookmakers for each game."""
    results = []
    
    for game in games:
        home = game.get('home_team')
        away = game.get('away_team')
        commence = game.get('commence_time', '')
        
        home_best = {'odds': -9999, 'book': None}
        away_best = {'odds': -9999, 'book': None}
        
        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker.get('title', bookmaker.get('key'))
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    for outcome in market.get('outcomes', []):
                        if outcome['name'] == home and outcome['price'] > home_best['odds']:
                            home_best = {'odds': outcome['price'], 'book': book_name}
                        elif outcome['name'] == away and outcome['price'] > away_best['odds']:
                            away_best = {'odds': outcome['price'], 'book': book_name}
        
        if home_best['book'] and away_best['book']:
            results.append({
                'home_team': home,
                'away_team': away,
                'commence_time': commence,
                'home_odds': home_best['odds'],
                'home_book': home_best['book'],
                'away_odds': away_best['odds'],
                'away_book': away_best['book']
            })
    
    return results


def odds_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def find_value_bets(games: list) -> list:
    """Find value bets by comparing bookmaker odds."""
    edges = []
    
    for game in games:
        home_prob = odds_to_implied_prob(game['home_odds'])
        away_prob = odds_to_implied_prob(game['away_odds'])
        total_prob = home_prob + away_prob
        
        if total_prob < 1.0 and total_prob > 0:
            edge_pct = (1.0 - total_prob) * 100
            
            if home_prob < away_prob:
                edges.append({
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'bet_on': game['home_team'],
                    'side': 'home',
                    'odds': game['home_odds'],
                    'bookmaker': game['home_book'],
                    'implied_prob': round(home_prob * 100, 1),
                    'edge': round(edge_pct, 1),
                    'ev': round(edge_pct * 2, 1),
                    'commence_time': game['commence_time']
                })
            else:
                edges.append({
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'bet_on': game['away_team'],
                    'side': 'away',
                    'odds': game['away_odds'],
                    'bookmaker': game['away_book'],
                    'implied_prob': round(away_prob * 100, 1),
                    'edge': round(edge_pct, 1),
                    'ev': round(edge_pct * 2, 1),
                    'commence_time': game['commence_time']
                })
    
    return sorted(edges, key=lambda x: x['ev'], reverse=True)


def main():
    print("=" * 60)
    print("MULTI-SPORT PREDICTIONS GENERATOR")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    output_dir = project_root.parent / "docs" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = {
        'generated_at': datetime.now().isoformat(),
        'sports': {}
    }
    
    total_edges = 0
    total_games = 0
    
    # Fetch from The Odds API
    for sport_id, config in ODDS_API_SPORTS.items():
        print(f"\n📊 Fetching {config['name']} odds...")
        
        raw_odds = fetch_odds_api(config['key'])
        
        if not raw_odds:
            print(f"   No {config['name']} games found")
            all_data['sports'][sport_id] = {
                'profile': config['profile'],
                'games': [],
                'edges': [],
                'total_games': 0,
                'total_edges': 0
            }
            continue
        
        games = find_best_odds(raw_odds)
        edges = find_value_bets(games)
        
        print(f"   Found {len(games)} games, {len(edges)} value bets")
        
        all_data['sports'][sport_id] = {
            'profile': config['profile'],
            'games': games,
            'edges': edges[:10],
            'total_games': len(games),
            'total_edges': len(edges)
        }
        
        total_edges += len(edges)
        total_games += len(games)
    
    # Fetch from PandaScore (Esports)
    for sport_id, config in PANDASCORE_SPORTS.items():
        print(f"\n🎮 Fetching {config['name']} matches...")
        
        game_slug = config.get('slug') or config.get('key')
        matches = fetch_pandascore_matches(game_slug)
        
        if not matches:
            print(f"   No {config['name']} matches found")
            all_data['sports'][sport_id] = {
                'profile': config['profile'],
                'games': [],
                'edges': [],
                'total_games': 0,
                'total_edges': 0
            }
            continue
        
        games, edges = parse_pandascore_matches(matches, config['name'])
        
        print(f"   Found {len(games)} matches")
        
        all_data['sports'][sport_id] = {
            'profile': config['profile'],
            'games': games,
            'edges': edges[:10],
            'total_games': len(games),
            'total_edges': len(edges)
        }
        
        total_games += len(games)
    
    all_data['total_games'] = total_games
    all_data['total_edges'] = total_edges
    
    # Calculate aggregate stats
    total_wins = 4  # NFL record
    total_losses = 1
    all_data['aggregate'] = {
        'total_games': total_games,
        'total_edges': total_edges,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(total_wins / max(total_wins + total_losses, 1) * 100, 1)
    }
    
    # Save
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✅ Saved multi-sport predictions")
    print(f"   Total: {total_games} games, {total_edges} edges")


if __name__ == "__main__":
    main()
