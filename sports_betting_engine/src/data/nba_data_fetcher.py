"""
NBA Data Fetcher Module
========================

Fetches historical NBA game data for training and recent stats for predictions.
Uses the official NBA API via nba_api package.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

try:
    from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, teamestimatedmetrics
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("Warning: nba_api not installed. Run: pip install nba_api")

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "nba_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_all_nba_teams() -> Dict[str, dict]:
    """Get all NBA teams with their IDs and abbreviations."""
    if not NBA_API_AVAILABLE:
        return {}
    
    nba_teams = teams.get_teams()
    return {team['full_name']: team for team in nba_teams}


def fetch_nba_games(seasons: List[str] = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch NBA game data for specified seasons.
    
    Args:
        seasons: List of seasons in format "2024-25", defaults to last 2 seasons
        use_cache: Whether to use cached data
        
    Returns:
        DataFrame with game-level statistics
    """
    if not NBA_API_AVAILABLE:
        raise ImportError("nba_api is required. Install with: pip install nba_api")
    
    if seasons is None:
        current_year = datetime.now().year
        current_month = datetime.now().month
        # NBA season starts in October
        if current_month >= 10:
            seasons = [f"{current_year}-{str(current_year+1)[-2:]}"]
        else:
            seasons = [f"{current_year-1}-{str(current_year)[-2:]}"]
    
    cache_file = CACHE_DIR / f"nba_games_{'_'.join(seasons)}.parquet"
    
    if use_cache and cache_file.exists():
        print(f"Loading NBA data from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    print(f"Fetching NBA game data for seasons: {seasons}")
    all_games = []
    
    for season in seasons:
        print(f"  Fetching season {season}...")
        try:
            # Get all games for the season
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00'  # NBA
            )
            time.sleep(1)  # Rate limiting
            
            games_df = gamefinder.get_data_frames()[0]
            games_df['SEASON'] = season
            all_games.append(games_df)
            
        except Exception as e:
            print(f"  Error fetching season {season}: {e}")
            continue
    
    if not all_games:
        raise ValueError("No game data retrieved")
    
    df = pd.concat(all_games, ignore_index=True)
    
    # Save to cache
    df.to_parquet(cache_file)
    print(f"Cached NBA data to: {cache_file}")
    
    return df


def process_nba_game_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw NBA games into matchup-level statistics.
    
    Returns DataFrame with one row per game, including home/away team stats.
    """
    # Filter to regular season and finished games
    games_df = games_df[games_df['WL'].notna()].copy()
    
    # Determine home/away from matchup string
    games_df['IS_HOME'] = games_df['MATCHUP'].str.contains(' vs. ')
    
    # Create game features
    games_df['WIN'] = (games_df['WL'] == 'W').astype(int)
    
    # Calculate per-game efficiency metrics
    games_df['OFF_RATING'] = games_df['PTS'] / (games_df['FGA'] + 0.44 * games_df['FTA'] + games_df['TOV']) * 100
    games_df['TS_PCT'] = games_df['PTS'] / (2 * (games_df['FGA'] + 0.44 * games_df['FTA']))
    games_df['AST_RATIO'] = games_df['AST'] / games_df['FGM']
    games_df['REB_PCT'] = (games_df['OREB'] + games_df['DREB']) / games_df['MIN'] * 48
    
    return games_df


def get_team_recent_form(games_df: pd.DataFrame, team_name: str, n_games: int = 10) -> Optional[Dict]:
    """
    Get recent form statistics for a team.
    
    Args:
        games_df: Processed games DataFrame
        team_name: Full team name (e.g., "Los Angeles Lakers")
        n_games: Number of recent games to consider
        
    Returns:
        Dictionary of averaged statistics or None if team not found
    """
    # Find team games
    team_games = games_df[games_df['TEAM_NAME'] == team_name].copy()
    
    if len(team_games) == 0:
        # Try partial match
        for name in games_df['TEAM_NAME'].unique():
            if team_name.split()[-1].lower() in name.lower():
                team_games = games_df[games_df['TEAM_NAME'] == name].copy()
                break
    
    if len(team_games) == 0:
        return None
    
    # Sort by date and take most recent
    team_games = team_games.sort_values('GAME_DATE', ascending=False).head(n_games)
    
    # Calculate averages
    stats = {
        'pts_avg': team_games['PTS'].mean(),
        'fg_pct': team_games['FG_PCT'].mean(),
        'fg3_pct': team_games['FG3_PCT'].mean(),
        'ft_pct': team_games['FT_PCT'].mean(),
        'reb_avg': (team_games['OREB'] + team_games['DREB']).mean(),
        'ast_avg': team_games['AST'].mean(),
        'stl_avg': team_games['STL'].mean(),
        'blk_avg': team_games['BLK'].mean(),
        'tov_avg': team_games['TOV'].mean(),
        'plus_minus_avg': team_games['PLUS_MINUS'].mean(),
        'win_pct': team_games['WIN'].mean(),
        'off_rating': team_games['OFF_RATING'].mean() if 'OFF_RATING' in team_games else 100,
        'ts_pct': team_games['TS_PCT'].mean() if 'TS_PCT' in team_games else 0.5,
    }
    
    return stats


def build_nba_training_data(seasons: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build training dataset for NBA game prediction.
    
    Returns:
        X: Feature DataFrame
        y: Target Series (1 for home win, 0 for away win)
    """
    # Fetch games
    games_df = fetch_nba_games(seasons)
    games_df = process_nba_game_stats(games_df)
    
    print(f"Processing {len(games_df)} team-games into matchups...")
    
    # Group by game ID to get home vs away
    features = []
    targets = []
    
    game_ids = games_df['GAME_ID'].unique()
    
    for game_id in game_ids:
        game_data = games_df[games_df['GAME_ID'] == game_id]
        
        if len(game_data) != 2:
            continue
        
        home_row = game_data[game_data['IS_HOME'] == True]
        away_row = game_data[game_data['IS_HOME'] == False]
        
        if len(home_row) != 1 or len(away_row) != 1:
            continue
        
        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]
        
        # Build feature dict
        feat = {
            'home_pts': home_row['PTS'],
            'away_pts': away_row['PTS'],
            'home_fg_pct': home_row['FG_PCT'],
            'away_fg_pct': away_row['FG_PCT'],
            'home_fg3_pct': home_row['FG3_PCT'],
            'away_fg3_pct': away_row['FG3_PCT'],
            'home_ft_pct': home_row['FT_PCT'],
            'away_ft_pct': away_row['FT_PCT'],
            'home_reb': home_row['OREB'] + home_row['DREB'],
            'away_reb': away_row['OREB'] + away_row['DREB'],
            'home_ast': home_row['AST'],
            'away_ast': away_row['AST'],
            'home_tov': home_row['TOV'],
            'away_tov': away_row['TOV'],
            'home_stl': home_row['STL'],
            'away_stl': away_row['STL'],
            'home_blk': home_row['BLK'],
            'away_blk': away_row['BLK'],
        }
        
        # Add differentials
        feat['pts_diff'] = feat['home_pts'] - feat['away_pts']
        feat['fg_pct_diff'] = feat['home_fg_pct'] - feat['away_fg_pct']
        feat['reb_diff'] = feat['home_reb'] - feat['away_reb']
        feat['ast_diff'] = feat['home_ast'] - feat['away_ast']
        feat['tov_diff'] = feat['home_tov'] - feat['away_tov']
        
        features.append(feat)
        targets.append(1 if home_row['WIN'] == 1 else 0)
    
    X = pd.DataFrame(features)
    y = pd.Series(targets)
    
    print(f"Created {len(X)} training samples")
    print(f"Home win rate: {y.mean():.1%}")
    
    return X, y


if __name__ == "__main__":
    print("Testing NBA Data Fetcher...")
    
    try:
        X, y = build_nba_training_data()
        print(f"\nFeatures: {list(X.columns)}")
        print(f"Samples: {len(X)}")
        print(f"Home win rate: {y.mean():.1%}")
    except Exception as e:
        print(f"Error: {e}")
