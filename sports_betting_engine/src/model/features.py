"""
Feature Definitions Module
==========================

Defines the features used for NFL game prediction models.
Provides documentation and utilities for feature understanding.
"""

from typing import Dict, List


# Core efficiency features
EFFICIENCY_FEATURES = {
    'yards_per_play': {
        'description': 'Average yards gained per offensive play',
        'type': 'continuous',
        'higher_is_better': True,
        'typical_range': (4.0, 7.0)
    },
    'epa_per_play': {
        'description': 'Expected Points Added per play (advanced efficiency metric)',
        'type': 'continuous',
        'higher_is_better': True,
        'typical_range': (-0.3, 0.3)
    },
    'success_rate': {
        'description': 'Percentage of plays that are considered successful',
        'type': 'continuous',
        'higher_is_better': True,
        'typical_range': (0.35, 0.55)
    },
    'pass_rate': {
        'description': 'Percentage of plays that are pass attempts',
        'type': 'continuous',
        'higher_is_better': None,  # Context dependent
        'typical_range': (0.45, 0.70)
    },
    'turnovers': {
        'description': 'Number of turnovers committed (interceptions + fumbles lost)',
        'type': 'count',
        'higher_is_better': False,
        'typical_range': (0, 4)
    },
    'touchdowns': {
        'description': 'Number of touchdowns scored',
        'type': 'count',
        'higher_is_better': True,
        'typical_range': (1, 6)
    }
}


# Model feature configuration
MODEL_FEATURES = [
    # Home team offensive efficiency
    'home_yards_per_play',
    'home_epa_per_play',
    'home_success_rate',
    'home_pass_rate',
    'home_turnovers',
    'home_touchdowns',
    
    # Away team offensive efficiency
    'away_yards_per_play',
    'away_epa_per_play',
    'away_success_rate',
    'away_pass_rate',
    'away_turnovers',
    'away_touchdowns',
    
    # Differential features
    'ypp_diff',
    'epa_diff',
    'success_diff',
    'turnover_diff',
    'pass_rate_diff',
]


# XGBoost hyperparameter search space
XGBOOST_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3, 5],
}


# Reduced param grid for faster training
XGBOOST_PARAM_GRID_FAST = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}


def get_feature_names(prefix: str = 'home') -> List[str]:
    """
    Get feature names with a specific prefix.
    
    Args:
        prefix: 'home' or 'away'
        
    Returns:
        List of feature column names
    """
    return [f"{prefix}_{feat}" for feat in EFFICIENCY_FEATURES.keys()]


def get_feature_importance_names() -> Dict[str, str]:
    """
    Get human-readable names for features (for visualization).
    
    Returns:
        Dictionary mapping feature names to display names
    """
    names = {}
    
    for feat, info in EFFICIENCY_FEATURES.items():
        names[f'home_{feat}'] = f"Home {info['description']}"
        names[f'away_{feat}'] = f"Away {info['description']}"
    
    # Differential features
    names['ypp_diff'] = "Yards Per Play Differential"
    names['epa_diff'] = "EPA Per Play Differential"
    names['success_diff'] = "Success Rate Differential"
    names['turnover_diff'] = "Turnover Differential"
    names['pass_rate_diff'] = "Pass Rate Differential"
    
    return names


def validate_features(df, required_features: List[str] = None) -> bool:
    """
    Validate that a DataFrame contains all required features.
    
    Args:
        df: DataFrame to validate
        required_features: List of required column names
        
    Returns:
        True if all features present, False otherwise
    """
    if required_features is None:
        required_features = MODEL_FEATURES
    
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        print(f"Missing features: {missing}")
        return False
    
    return True
