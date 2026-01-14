"""
Predictor Module
================

High-level interface for making predictions on upcoming NFL games.
Loads trained model and prepares features for prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .trainer import NFLGamePredictor


class GamePredictionService:
    """
    Service for predicting NFL game outcomes.
    
    Provides a clean interface for making predictions on upcoming games
    using the latest trained model.
    
    Example:
        >>> service = GamePredictionService()
        >>> prediction = service.predict_game(
        ...     home_team_stats={'yards_per_play': 6.2, 'epa_per_play': 0.15, ...},
        ...     away_team_stats={'yards_per_play': 5.5, 'epa_per_play': 0.05, ...}
        ... )
        >>> print(f"Home win probability: {prediction['home_win_prob']:.1%}")
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the prediction service.
        
        Args:
            model_path: Path to specific model file, or None to load latest
        """
        if model_path:
            self.predictor = NFLGamePredictor.load(model_path)
        else:
            self.predictor = NFLGamePredictor.load_latest()
    
    def predict_game(
        self,
        home_team_stats: Dict[str, float],
        away_team_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict the outcome of a single game.
        
        Args:
            home_team_stats: Dictionary of home team statistics
            away_team_stats: Dictionary of away team statistics
            
        Returns:
            Dictionary with prediction results:
            - home_win_prob: Probability of home team winning
            - away_win_prob: Probability of away team winning
            - predicted_winner: 'home' or 'away'
            - confidence: How confident the model is (abs distance from 0.5)
        """
        # Build feature DataFrame
        features = self._build_features(home_team_stats, away_team_stats)
        
        # Get prediction
        home_win_prob = self.predictor.predict_proba(features)[0]
        away_win_prob = 1 - home_win_prob
        
        predicted_winner = 'home' if home_win_prob >= 0.5 else 'away'
        confidence = abs(home_win_prob - 0.5) * 2  # Scale to 0-1
        
        return {
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence
        }
    
    def predict_multiple_games(
        self,
        games: List[Dict]
    ) -> List[Dict]:
        """
        Predict outcomes for multiple games.
        
        Args:
            games: List of game dictionaries, each containing:
                - home_team: Team name/abbreviation
                - away_team: Team name/abbreviation
                - home_stats: Statistics dictionary
                - away_stats: Statistics dictionary
                
        Returns:
            List of prediction dictionaries with game info and predictions
        """
        predictions = []
        
        for game in games:
            prediction = self.predict_game(
                game['home_stats'],
                game['away_stats']
            )
            
            predictions.append({
                'home_team': game.get('home_team', 'Unknown'),
                'away_team': game.get('away_team', 'Unknown'),
                'commence_time': game.get('commence_time'),
                **prediction
            })
        
        return predictions
    
    def _build_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Build feature DataFrame from team statistics.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            DataFrame with one row of features
        """
        features = {}
        
        # Map statistics to feature columns
        stat_keys = [
            'yards_per_play', 'epa_per_play', 'success_rate',
            'pass_rate', 'turnovers', 'touchdowns'
        ]
        
        for key in stat_keys:
            features[f'home_{key}'] = home_stats.get(key, 0)
            features[f'away_{key}'] = away_stats.get(key, 0)
        
        # Calculate differential features
        for key in ['yards_per_play', 'epa_per_play', 'success_rate', 'turnovers', 'pass_rate']:
            home_val = home_stats.get(key, 0)
            away_val = away_stats.get(key, 0)
            
            # Map to differential column names
            diff_name_map = {
                'yards_per_play': 'ypp_diff',
                'epa_per_play': 'epa_diff',
                'success_rate': 'success_diff',
                'turnovers': 'turnover_diff',
                'pass_rate': 'pass_rate_diff'
            }
            
            features[diff_name_map[key]] = home_val - away_val
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([features])
        
        # Reorder to match model's expected features
        expected_features = self.predictor.feature_names
        
        # Add any missing features with default value 0
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0
        
        return df[expected_features]
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'trained_at': self.predictor.trained_at,
            'accuracy': self.predictor.training_metrics.get('accuracy'),
            'roc_auc': self.predictor.training_metrics.get('roc_auc'),
            'feature_count': len(self.predictor.feature_names),
            'features': self.predictor.feature_names
        }


def make_prediction(
    home_team: str,
    away_team: str,
    home_stats: Dict[str, float],
    away_stats: Dict[str, float],
    model_path: Optional[Path] = None
) -> Dict:
    """
    Convenience function for making a single game prediction.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_stats: Home team statistics
        away_stats: Away team statistics
        model_path: Optional path to specific model
        
    Returns:
        Prediction dictionary with all details
    """
    service = GamePredictionService(model_path)
    prediction = service.predict_game(home_stats, away_stats)
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        **prediction
    }


if __name__ == "__main__":
    print("Testing Predictor Module...")
    print("\nNote: This requires a trained model to exist.")
    print("Run scripts/train_model.py first if you haven't already.")
    
    try:
        service = GamePredictionService()
        
        # Example prediction
        prediction = service.predict_game(
            home_team_stats={
                'yards_per_play': 6.2,
                'epa_per_play': 0.15,
                'success_rate': 0.52,
                'pass_rate': 0.58,
                'turnovers': 1.2,
                'touchdowns': 3.5
            },
            away_team_stats={
                'yards_per_play': 5.5,
                'epa_per_play': 0.05,
                'success_rate': 0.45,
                'pass_rate': 0.55,
                'turnovers': 1.8,
                'touchdowns': 2.5
            }
        )
        
        print(f"\nPrediction:")
        print(f"  Home win probability: {prediction['home_win_prob']:.1%}")
        print(f"  Away win probability: {prediction['away_win_prob']:.1%}")
        print(f"  Predicted winner: {prediction['predicted_winner']}")
        print(f"  Confidence: {prediction['confidence']:.1%}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
