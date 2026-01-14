"""
Data Preprocessor Module
========================

Feature engineering for NFL game prediction model.
Transforms raw game data into features suitable for XGBoost training.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def calculate_rolling_averages(
    df: pd.DataFrame,
    group_col: str,
    value_cols: List[str],
    windows: List[int] = [3, 5],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Calculate rolling averages for specified columns grouped by team.
    
    Args:
        df: DataFrame with game data (should be sorted by date)
        group_col: Column to group by (e.g., 'team')
        value_cols: Columns to calculate rolling averages for
        windows: List of window sizes (e.g., [3, 5] for last 3 and 5 games)
        min_periods: Minimum observations required
        
    Returns:
        DataFrame with additional rolling average columns
    """
    result = df.copy()
    
    for col in value_cols:
        for window in windows:
            new_col = f"{col}_rolling_{window}"
            result[new_col] = (
                result
                .groupby(group_col)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean())
            )
    
    return result


def create_team_features(game_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Create team-level features from game statistics.
    
    Features include:
    - Rolling averages of efficiency metrics
    - Momentum indicators
    - Rest days (if available)
    
    Args:
        game_stats: DataFrame with game-level statistics per team
        
    Returns:
        DataFrame with additional feature columns
    """
    # Sort by season and week for proper rolling calculations
    df = game_stats.sort_values(['season', 'week']).copy()
    
    # Key metrics to create rolling averages for
    efficiency_cols = [
        'yards_per_play', 'epa_per_play', 'success_rate',
        'pass_rate', 'turnovers', 'touchdowns'
    ]
    
    # Calculate rolling averages for last 3 and 5 games
    df = calculate_rolling_averages(df, 'team', efficiency_cols, windows=[3, 5])
    
    # Calculate season-to-date averages
    for col in efficiency_cols:
        df[f'{col}_season_avg'] = (
            df.groupby(['team', 'season'])[col]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    
    return df


def create_matchup_features(training_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture matchup-specific information.
    
    Args:
        training_data: DataFrame with home and away team stats
        
    Returns:
        DataFrame with matchup differential features
    """
    df = training_data.copy()
    
    # Create differential features (home - away)
    stat_pairs = [
        ('home_yards_per_play', 'away_yards_per_play', 'ypp_diff'),
        ('home_epa_per_play', 'away_epa_per_play', 'epa_diff'),
        ('home_success_rate', 'away_success_rate', 'success_diff'),
        ('home_turnovers', 'away_turnovers', 'turnover_diff'),
        ('home_pass_rate', 'away_pass_rate', 'pass_rate_diff'),
    ]
    
    for home_col, away_col, diff_col in stat_pairs:
        if home_col in df.columns and away_col in df.columns:
            df[diff_col] = df[home_col] - df[away_col]
    
    # Home field advantage indicator (always 1 for home team perspective)
    df['home_field'] = 1
    
    return df


def prepare_features_for_training(
    training_data: pd.DataFrame,
    target_col: str = 'home_win'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare final feature matrix and target variable for model training.
    
    Args:
        training_data: Full training dataset
        target_col: Column name for target variable
        
    Returns:
        Tuple of (X features, y target, feature names)
    """
    # Define feature columns to use
    feature_cols = [
        # Home team stats
        'home_yards_per_play', 'home_epa_per_play', 'home_success_rate',
        'home_pass_rate', 'home_turnovers', 'home_touchdowns',
        
        # Away team stats
        'away_yards_per_play', 'away_epa_per_play', 'away_success_rate',
        'away_pass_rate', 'away_turnovers', 'away_touchdowns',
    ]
    
    # Add differential features if they exist
    diff_cols = ['ypp_diff', 'epa_diff', 'success_diff', 'turnover_diff', 'pass_rate_diff']
    feature_cols.extend([col for col in diff_cols if col in training_data.columns])
    
    # Filter to only existing columns
    available_cols = [col for col in feature_cols if col in training_data.columns]
    
    print(f"Using {len(available_cols)} features: {available_cols}")
    
    X = training_data[available_cols].copy()
    y = training_data[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    return X, y, available_cols


def prepare_prediction_features(
    home_team_stats: dict,
    away_team_stats: dict,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Prepare features for a single game prediction.
    
    Args:
        home_team_stats: Dictionary of home team statistics
        away_team_stats: Dictionary of away team statistics
        feature_cols: List of feature columns expected by the model
        
    Returns:
        DataFrame with one row of features ready for prediction
    """
    features = {}
    
    # Map stats to feature names
    for col in feature_cols:
        if col.startswith('home_'):
            stat_name = col.replace('home_', '')
            features[col] = home_team_stats.get(stat_name, 0)
        elif col.startswith('away_'):
            stat_name = col.replace('away_', '')
            features[col] = away_team_stats.get(stat_name, 0)
        elif col.endswith('_diff'):
            # Calculate differential
            base_name = col.replace('_diff', '')
            home_val = home_team_stats.get(base_name, 0)
            away_val = away_team_stats.get(base_name, 0)
            features[col] = home_val - away_val
    
    return pd.DataFrame([features])


def get_team_recent_stats(
    game_stats: pd.DataFrame,
    team: str,
    n_games: int = 5
) -> dict:
    """
    Get a team's average statistics over their last N games.
    
    Args:
        game_stats: DataFrame with game-level stats
        team: Team abbreviation
        n_games: Number of recent games to average
        
    Returns:
        Dictionary of average statistics
    """
    team_games = (
        game_stats[game_stats['team'] == team]
        .sort_values(['season', 'week'], ascending=False)
        .head(n_games)
    )
    
    if len(team_games) == 0:
        return {}
    
    stats = {
        'yards_per_play': team_games['yards_per_play'].mean(),
        'epa_per_play': team_games['epa_per_play'].mean(),
        'success_rate': team_games['success_rate'].mean(),
        'pass_rate': team_games['pass_rate'].mean(),
        'turnovers': team_games['turnovers'].mean(),
        'touchdowns': team_games['touchdowns'].mean(),
    }
    
    return stats


if __name__ == "__main__":
    # Test with sample data
    print("Testing Preprocessor...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'season': [2023] * 10,
        'week': list(range(1, 11)),
        'team': ['KC'] * 10,
        'yards_per_play': np.random.uniform(5, 7, 10),
        'epa_per_play': np.random.uniform(-0.1, 0.3, 10),
        'success_rate': np.random.uniform(0.4, 0.6, 10),
        'pass_rate': np.random.uniform(0.5, 0.7, 10),
        'turnovers': np.random.randint(0, 3, 10),
        'touchdowns': np.random.randint(1, 5, 10),
    })
    
    # Test rolling averages
    result = create_team_features(sample_data)
    print("\nRolling average columns added:")
    print([col for col in result.columns if 'rolling' in col])
