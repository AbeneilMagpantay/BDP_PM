"""
Soccer Model Trainer
====================

Trains XGBoost model for soccer match outcome prediction.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

# Model save directory
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class SoccerMatchPredictor:
    """
    XGBoost-based predictor for soccer match outcomes.
    Predicts the probability that the home team wins.
    """
    
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
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the XGBoost model on provided data."""
        print("=" * 60)
        print("SOCCER MATCH PREDICTOR - MODEL TRAINING")
        print("=" * 60)
        
        self.feature_names = list(X.columns)
        print(f"\nFeatures ({len(self.feature_names)}): {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Train model
        print("\nTraining XGBoost model...")
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False
        )
        
        self.model.fit(X_train, y_train)
        
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
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
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
        
        # Ensure features are in correct order, add missing with 0
        for feat in self.feature_names:
            if feat not in X.columns:
                X[feat] = 0
        
        X_ordered = X[self.feature_names]
        return self.model.predict_proba(X_ordered)[:, 1]
    
    def predict_match(
        self,
        home_team_stats: Dict[str, float],
        away_team_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Predict the outcome of a single match with form and H2H.
        """
        features = {}
        
        # Map stats to features
        for key in ['strength', 'xg', 'shots', 'possession', 'form', 'h2h_win_rate']:
            features[f'home_{key}'] = home_team_stats.get(key, 0.5)
            features[f'away_{key}'] = away_team_stats.get(key, 0.5)
        
        # Calculate differentials
        features['strength_diff'] = features.get('home_strength', 0.5) - features.get('away_strength', 0.5)
        features['xg_diff'] = features.get('home_xg', 1) - features.get('away_xg', 1)
        features['shots_diff'] = features.get('home_shots', 10) - features.get('away_shots', 10)
        features['possession_diff'] = features.get('home_possession', 50) - features.get('away_possession', 50)
        features['form_diff'] = features.get('home_form', 0.5) - features.get('away_form', 0.5)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Get prediction
        home_win_prob = self.predict_proba(df)[0]
        
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
            filepath = MODEL_DIR / f"soccer_predictor_{timestamp}.joblib"
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'trained_at': self.trained_at
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nSoccer Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'SoccerMatchPredictor':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.training_metrics = model_data['training_metrics']
        predictor.trained_at = model_data.get('trained_at')
        
        print(f"Soccer Model loaded from: {filepath}")
        print(f"Accuracy: {predictor.training_metrics.get('accuracy', 'N/A'):.1%}")
        
        return predictor
    
    @classmethod
    def load_latest(cls) -> 'SoccerMatchPredictor':
        """Load the most recently saved Soccer model."""
        model_files = list(MODEL_DIR.glob("soccer_predictor_*.joblib"))
        
        if not model_files:
            raise FileNotFoundError(
                f"No Soccer models found in {MODEL_DIR}. "
                "Train a model first using train_soccer_model.py"
            )
        
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return cls.load(latest)


if __name__ == "__main__":
    from src.data.soccer_data_fetcher import build_soccer_training_data
    
    print("Building Soccer training data...")
    X, y = build_soccer_training_data()
    
    print("\nTraining Soccer model...")
    predictor = SoccerMatchPredictor()
    metrics = predictor.train(X, y)
    
    print("\nSaving model...")
    predictor.save()
