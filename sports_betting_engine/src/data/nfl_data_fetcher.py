"""
NFL Data Fetcher Module
=======================

Fetches historical NFL play-by-play data using nfl_data_py library.
Aggregates play-level data to game-level efficiency metrics for model training.

Data Source: nflfastR via nfl_data_py
"""

import os
import pandas as pd
import nfl_data_py as nfl
from typing import List, Optional
from pathlib import Path


# Define the base data directory
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def fetch_play_by_play_data(
    years: List[int],
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch play-by-play data for specified NFL seasons.
    Fetches year-by-year to handle missing future data gracefully.
    """
    if cache_dir is None:
        cache_dir = RAW_DATA_DIR
        
    cache_file = cache_dir / f"pbp_{'_'.join(map(str, sorted(years)))}.parquet"
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        print(f"Loading play-by-play data from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    # Fetch from nfl_data_py
    print(f"Fetching play-by-play data for years: {years}")
    
    # Select only the columns we need to reduce memory usage
    columns = [
        'game_id', 'season', 'week', 'season_type', 'game_date',
        'home_team', 'away_team', 'posteam', 'defteam',
        'play_type', 'yards_gained', 'epa', 'success', 'down',
        'ydstogo', 'yardline_100', 'pass', 'rush',
        'touchdown', 'interception', 'fumble_lost',
        'first_down', 'third_down_converted', 'fourth_down_converted',
        'home_score', 'away_score', 'result', 'total'
    ]
    
    all_pbp = []
    
    for year in years:
        try:
            print(f"Fetching NFL {year}...")
            # Fetch single year
            pbp_year = nfl.import_pbp_data([year], columns)
            if pbp_year is not None and not pbp_year.empty:
                all_pbp.append(pbp_year)
                print(f"Successfully loaded {year} ({len(pbp_year)} rows)")
        except Exception as e:
            # Catch all exceptions, including likely NameError in nfl_data_py or HTTPError
            print(f"Warning: Could not fetch data for {year}. Skipping. Error: {e}")
            continue

    if not all_pbp:
        raise ValueError(f"No data could be fetched for any of the requested years: {years}")

    combined_pbp = pd.concat(all_pbp, ignore_index=True)

    try:
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        combined_pbp.to_parquet(cache_file)
        print(f"Cached play-by-play data to: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to cache data: {e}")
        
    return combined_pbp


def fetch_schedule_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch NFL schedule data for specified seasons.
    
    Args:
        years: List of years to fetch
        
    Returns:
        DataFrame with schedule information including game results
    """
    print(f"Fetching schedule data for years: {years}")
    try:
        schedule = nfl.import_schedules(years)
        return schedule
    except Exception as e:
        print(f"Error fetching schedule data: {e}")
        raise


def fetch_team_data() -> pd.DataFrame:
    """
    Fetch current NFL team information.
    
    Returns:
        DataFrame with team abbreviations, names, and divisions
    """
    try:
        teams = nfl.import_team_desc()
        return teams
    except Exception as e:
        print(f"Error fetching team data: {e}")
        raise


def aggregate_to_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-by-play data to game-level statistics.
    
    Calculates efficiency metrics for each team in each game:
    - Yards per play (rushing and passing)
    - EPA per play
    - Success rate
    - Turnover counts
    - Red zone efficiency
    - Third down conversion rate
    
    Args:
        pbp: Play-by-play DataFrame
        
    Returns:
        DataFrame with one row per team per game
    """
    print("Aggregating play-by-play data to game-level statistics...")
    
    # Filter to actual plays (exclude penalties, timeouts, etc.)
    plays = pbp[pbp['play_type'].isin(['pass', 'run'])].copy()
    
    # Group by game and possession team
    game_stats = plays.groupby(['game_id', 'season', 'week', 'posteam']).agg({
        # Volume stats
        'play_type': 'count',
        'yards_gained': ['sum', 'mean'],
        'epa': ['sum', 'mean'],
        'success': 'mean',
        
        # Passing
        'pass': 'sum',
        
        # Rushing  
        'rush': 'sum',
        
        # Scoring and turnovers
        'touchdown': 'sum',
        'interception': 'sum',
        'fumble_lost': 'sum',
        
        # Efficiency
        'first_down': 'sum',
        'third_down_converted': 'sum',
    }).reset_index()
    
    # Flatten column names
    game_stats.columns = [
        'game_id', 'season', 'week', 'team',
        'total_plays', 'total_yards', 'yards_per_play',
        'total_epa', 'epa_per_play', 'success_rate',
        'pass_plays', 'rush_plays',
        'touchdowns', 'interceptions', 'fumbles_lost',
        'first_downs', 'third_down_conversions'
    ]
    
    # Calculate turnovers
    game_stats['turnovers'] = game_stats['interceptions'] + game_stats['fumbles_lost']
    
    # Calculate pass/rush ratio
    game_stats['pass_rate'] = game_stats['pass_plays'] / game_stats['total_plays']
    
    print(f"Created game stats for {len(game_stats)} team-games")
    return game_stats


def get_game_results(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Extract game results (winner/loser, scores) from schedule data.
    
    Args:
        schedule: Schedule DataFrame from nfl_data_py
        
    Returns:
        DataFrame with game results including point differential
    """
    # Filter to completed games only
    completed = schedule[schedule['result'].notna()].copy()
    
    completed['home_win'] = (completed['result'] > 0).astype(int)
    completed['point_diff'] = completed['result']  # Positive = home win
    completed['total_points'] = completed['home_score'] + completed['away_score']
    
    return completed[[
        'game_id', 'season', 'week', 'game_type',
        'home_team', 'away_team',
        'home_score', 'away_score',
        'home_win', 'point_diff', 'total_points',
        'spread_line', 'total_line'
    ]]


def build_training_dataset(years: List[int], use_cache: bool = True) -> pd.DataFrame:
    """
    Build complete training dataset from raw NFL data.
    
    Combines play-by-play aggregations with game results to create
    a dataset ready for model training.
    
    Args:
        years: List of years to include
        use_cache: Whether to use cached data
        
    Returns:
        DataFrame ready for model training
    """
    cache_file = PROCESSED_DATA_DIR / f"training_data_{'_'.join(map(str, years))}.parquet"
    
    if use_cache and cache_file.exists():
        print(f"Loading training data from cache: {cache_file}")
        return pd.read_parquet(cache_file)
    
    # Fetch raw data
    pbp = fetch_play_by_play_data(years, use_cache=use_cache)
    schedule = fetch_schedule_data(years)
    
    # Process data
    game_stats = aggregate_to_game_stats(pbp)
    game_results = get_game_results(schedule)
    
    # Merge stats with results
    # We need home and away team stats for each game
    home_stats = game_stats.copy()
    home_stats.columns = ['game_id', 'season', 'week', 'home_team'] + \
                         [f'home_{col}' for col in game_stats.columns[4:]]
    
    away_stats = game_stats.copy()
    away_stats.columns = ['game_id', 'season', 'week', 'away_team'] + \
                         [f'away_{col}' for col in game_stats.columns[4:]]
    
    # Merge everything together
    training_data = game_results.merge(
        home_stats, on=['game_id', 'season', 'week', 'home_team'], how='left'
    ).merge(
        away_stats, on=['game_id', 'season', 'week', 'away_team'], how='left'
    )
    
    # Drop rows with missing stats (incomplete games)
    training_data = training_data.dropna()
    
    # Save to cache
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    training_data.to_parquet(cache_file)
    print(f"Saved training data to: {cache_file}")
    
    return training_data


if __name__ == "__main__":
    # Test the module
    print("Testing NFL Data Fetcher...")
    
    # Fetch a single year for testing
    pbp = fetch_play_by_play_data([2023])
    print(f"\nPlay-by-play data shape: {pbp.shape}")
    print(f"Columns: {list(pbp.columns)}")
    
    # Aggregate to game stats
    game_stats = aggregate_to_game_stats(pbp)
    print(f"\nGame stats shape: {game_stats.shape}")
    print(game_stats.head())
