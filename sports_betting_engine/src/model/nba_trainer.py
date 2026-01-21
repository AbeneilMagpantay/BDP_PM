"""
NBA Model Trainer
=================

Trains XGBoost model for NBA game outcome prediction.
Similar architecture to NFL model but optimized for basketball.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

# Hyperparameter grid for tuning
NBA_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Model save directory
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class NBAGamePredictor:
    """
    XGBoost-based predictor for NBA game outcomes.
    Predicts the probability that the home team wins.
    """
    
    # Features used for prediction (based on recent team form)
    PREDICTION_FEATURES = [
        'home_pts_avg', 'away_pts_avg',
        'home_fg_pct', 'away_fg_pct',
        'home_fg3_pct', 'away_fg3_pct',
        'home_reb_avg', 'away_reb_avg',
        'home_ast_avg', 'away_ast_avg',
        'home_tov_avg', 'away_tov_avg',
        'home_win_pct', 'away_win_pct',
        'pts_diff', 'fg_pct_diff', 'reb_diff', 'ast_diff', 'tov_diff'
    ]
    
    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict[str, Any] = {}
        self.trained_at: Optional[datetime] = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        tune_hyperparameters: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the XGBoost model on provided data."""
        print("=" * 60)
        print("NBA GAME PREDICTOR - MODEL TRAINING")
        print("=" * 60)
        
        # Remove non-predictive features (actual game results)
        drop_cols = ['home_pts', 'away_pts', 'pts_diff']
        X_train_data = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        
        self.feature_names = list(X_train_data.columns)
        print(f"\nFeatures ({len(self.feature_names)}): {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_data, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Train model with or without hyperparameter tuning
        if tune_hyperparameters:
            print("\nPerforming GridSearchCV hyperparameter tuning...")
            base_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=random_state,
                use_label_encoder=False
            )
            
            grid_search = GridSearchCV(
                base_model,
                NBA_PARAM_GRID,
                cv=3,
                scoring='neg_log_loss',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV log loss: {-grid_search.best_score_:.4f}")
            raw_model = grid_search.best_estimator_
        else:
            print("\nTraining XGBoost model with default params...")
            raw_model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=random_state,
                use_label_encoder=False
            )
            raw_model.fit(X_train, y_train)
        
        # Apply Platt Scaling for probability calibration
        print("\nApplying Platt Scaling (probability calibration)...")
        self.model = CalibratedClassifierCV(
            raw_model,
            method='sigmoid',  # Platt Scaling
            cv='prefit'  # Model is already fitted
        )
        self.model.fit(X_train, y_train)
        print("Calibration complete.")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Log Loss: {logloss:.4f}")
        print(f"  ROC AUC: {auc:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_data, y, cv=5, scoring='accuracy')
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
        
        self.training_metrics = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'roc_auc': auc,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std()
        }
        self.trained_at = datetime.now()
        
        return self.training_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of home team winning."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features are in correct order
        X_ordered = X[self.feature_names]
        return self.model.predict_proba(X_ordered)[:, 1]
    
    def predict_game(
        self,
        home_team_stats: Dict[str, float],
        away_team_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict the outcome of a single game.
        
        Args:
            home_team_stats: Dictionary of home team recent stats
            away_team_stats: Dictionary of away team recent stats
            
        Returns:
            Dictionary with prediction results
        """
        # Build feature dict
        features = {}
        
        # Map stats to features
        stat_keys = ['pts_avg', 'fg_pct', 'fg3_pct', 'reb_avg', 'ast_avg', 'tov_avg', 'win_pct']
        
        for key in stat_keys:
            features[f'home_{key}'] = home_team_stats.get(key, 0)
            features[f'away_{key}'] = away_team_stats.get(key, 0)
        
        # Calculate differentials
        features['pts_diff'] = features.get('home_pts_avg', 0) - features.get('away_pts_avg', 0)
        features['fg_pct_diff'] = features.get('home_fg_pct', 0) - features.get('away_fg_pct', 0)
        features['reb_diff'] = features.get('home_reb_avg', 0) - features.get('away_reb_avg', 0)
        features['ast_diff'] = features.get('home_ast_avg', 0) - features.get('away_ast_avg', 0)
        features['tov_diff'] = features.get('home_tov_avg', 0) - features.get('away_tov_avg', 0)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Add missing features with default values
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        # Get prediction
        home_win_prob = self.predict_proba(df[self.feature_names])[0]
        
        # Clamp to realistic NBA probability range (no team ever has >85% or <15% true odds)
        home_win_prob = max(0.15, min(0.85, home_win_prob))
        
        return {
            'home_win_prob': float(home_win_prob),
            'away_win_prob': float(1 - home_win_prob),
            'predicted_winner': 'home' if home_win_prob >= 0.5 else 'away',
            'confidence': float(abs(home_win_prob - 0.5) * 2)
        }
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODEL_DIR / f"nba_predictor_{timestamp}.joblib"
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'trained_at': self.trained_at
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nNBA Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'NBAGamePredictor':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.training_metrics = model_data['training_metrics']
        predictor.trained_at = model_data.get('trained_at')
        
        print(f"NBA Model loaded from: {filepath}")
        print(f"Accuracy: {predictor.training_metrics.get('accuracy', 'N/A'):.1%}")
        
        return predictor
    
    @classmethod
    def load_latest(cls) -> 'NBAGamePredictor':
        """Load the most recently saved NBA model."""
        model_files = list(MODEL_DIR.glob("nba_predictor_*.joblib"))
        
        if not model_files:
            raise FileNotFoundError(
                f"No NBA models found in {MODEL_DIR}. "
                "Train a model first using train_nba_model.py"
            )
        
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return cls.load(latest)


def train_and_save_nba_model():
    """Main function to train and save NBA model."""
    from src.data.nba_data_fetcher import build_nba_training_data
    
    print("Building NBA training data...")
    X, y = build_nba_training_data()
    
    print("\nTraining NBA model...")
    predictor = NBAGamePredictor()
    metrics = predictor.train(X, y)
    
    print("\nSaving model...")
    predictor.save()
    
    return predictor, metrics


if __name__ == "__main__":
    train_and_save_nba_model()
