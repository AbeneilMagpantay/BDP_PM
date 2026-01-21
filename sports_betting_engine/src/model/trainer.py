"""
Model Trainer Module
====================

Trains XGBoost model for NFL game outcome prediction.
Includes hyperparameter tuning, cross-validation, and model persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, 
    classification_report, confusion_matrix, brier_score_loss
)
from xgboost import XGBClassifier

from .features import XGBOOST_PARAM_GRID, XGBOOST_PARAM_GRID_FAST, MODEL_FEATURES


# Model save directory
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"


class NFLGamePredictor:
    """
    XGBoost-based predictor for NFL game outcomes.
    
    Predicts the probability that the home team wins.
    
    Attributes:
        model: Trained XGBClassifier
        feature_names: List of feature column names used by the model
        training_metrics: Dictionary of evaluation metrics from training
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
        tune_hyperparameters: bool = True,
        fast_mode: bool = True,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model on provided data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (1 for home win, 0 for away win)
            test_size: Fraction of data to hold out for testing
            tune_hyperparameters: If True, perform GridSearchCV
            fast_mode: If True, use reduced parameter grid
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of training metrics
        """
        print("=" * 60)
        print("NFL GAME PREDICTOR - MODEL TRAINING")
        print("=" * 60)
        
        # Store feature names
        self.feature_names = list(X.columns)
        print(f"\nFeatures ({len(self.feature_names)}): {self.feature_names}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nDataset split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Home win rate (train): {y_train.mean():.1%}")
        print(f"  Home win rate (test): {y_test.mean():.1%}")
        
        if tune_hyperparameters:
            self.model, base_model = self._train_with_tuning(
                X_train, y_train, fast_mode, random_state
            )
        else:
            self.model, base_model = self._train_default(X_train, y_train, random_state)
        
        # Evaluate on test set
        metrics = self._evaluate(X_test, y_test)
        
        # Cross-validation score (use base XGBClassifier, not calibrated model)
        # CalibratedClassifierCV with cv='prefit' cannot be used with cross_val_score
        cv_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False
        )
        cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
        
        self.training_metrics = metrics
        self.trained_at = datetime.now()
        
        return metrics
    
    def _train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fast_mode: bool,
        random_state: int
    ) -> tuple:
        """Train with hyperparameter tuning via GridSearchCV, then apply Platt Scaling.
        
        Returns:
            Tuple of (calibrated_model, base_model) for cross-validation support.
        """
        print("\nPerforming hyperparameter tuning...")
        
        param_grid = XGBOOST_PARAM_GRID_FAST if fast_mode else XGBOOST_PARAM_GRID
        
        base_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False
        )
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV log loss: {-grid_search.best_score_:.4f}")
        
        # Apply Platt Scaling for probability calibration
        print("\nApplying Platt Scaling (probability calibration)...")
        calibrated_model = CalibratedClassifierCV(
            grid_search.best_estimator_,
            method='sigmoid',  # Platt Scaling
            cv='prefit'  # Model is already fitted
        )
        calibrated_model.fit(X_train, y_train)
        print("Calibration complete.")
        
        return calibrated_model, grid_search.best_estimator_
    
    def _train_default(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int
    ) -> tuple:
        """Train with default hyperparameters, then apply Platt Scaling.
        
        Returns:
            Tuple of (calibrated_model, base_model) for cross-validation support.
        """
        print("\nTraining with default parameters...")
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_state,
            use_label_encoder=False
        )
        
        model.fit(X_train, y_train)
        
        # Apply Platt Scaling for probability calibration
        print("Applying Platt Scaling (probability calibration)...")
        calibrated_model = CalibratedClassifierCV(
            model,
            method='sigmoid',  # Platt Scaling
            cv='prefit'  # Model is already fitted
        )
        calibrated_model.fit(X_train, y_train)
        print("Calibration complete.")
        
        return calibrated_model, model
    
    def _evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        print("\n" + "-" * 40)
        print("TEST SET EVALUATION")
        print("-" * 40)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\nAccuracy: {accuracy:.1%}")
        print(f"Log Loss: {logloss:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}]")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        return {
            'accuracy': accuracy,
            'log_loss': logloss,
            'roc_auc': auc,
            'confusion_matrix': cm.tolist(),
            'test_size': len(y_test)
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of home team winning.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probabilities (home win probability)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure features are in correct order
        X_ordered = X[self.feature_names]
        return self.model.predict_proba(X_ordered)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict game outcomes.
        
        Args:
            X: Feature DataFrame
            threshold: Probability threshold for home win prediction
            
        Returns:
            Array of predictions (1 = home win, 0 = away win)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Access underlying XGBoost model from CalibratedClassifierCV
        if hasattr(self.model, 'estimator'):
            # CalibratedClassifierCV wraps the estimator
            base_model = self.model.estimator
        elif hasattr(self.model, 'calibrated_classifiers_'):
            # Access first calibrated classifier's base estimator
            base_model = self.model.calibrated_classifiers_[0].estimator
        else:
            base_model = self.model
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': base_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (default: auto-generated)
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = MODEL_DIR / f"nfl_predictor_{timestamp}.joblib"
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'trained_at': self.trained_at
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'NFLGamePredictor':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            NFLGamePredictor instance with loaded model
        """
        model_data = joblib.load(filepath)
        
        predictor = cls()
        predictor.model = model_data['model']
        predictor.feature_names = model_data['feature_names']
        predictor.training_metrics = model_data['training_metrics']
        predictor.trained_at = model_data.get('trained_at')
        
        print(f"Model loaded from: {filepath}")
        print(f"Trained at: {predictor.trained_at}")
        print(f"Test accuracy: {predictor.training_metrics.get('accuracy', 'N/A'):.1%}")
        
        return predictor
    
    @classmethod
    def load_latest(cls) -> 'NFLGamePredictor':
        """
        Load the most recently saved model.
        
        Returns:
            NFLGamePredictor instance with loaded model
        """
        model_files = list(MODEL_DIR.glob("nfl_predictor_*.joblib"))
        
        # DEBUG: Print search path and results
        print(f"DEBUG: Searching for models in: {MODEL_DIR.absolute()}")
        if MODEL_DIR.exists():
            print(f"DEBUG: Directory contents: {[f.name for f in MODEL_DIR.iterdir()]}")
        else:
            print("DEBUG: Directory does not exist!")

        if not model_files:
            raise FileNotFoundError(
                f"No saved models found in {MODEL_DIR}. "
                "Train a model first using train_model.py"
            )
        
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return cls.load(latest)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing Model Trainer...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'home_yards_per_play': np.random.uniform(4, 7, n_samples),
        'home_epa_per_play': np.random.uniform(-0.2, 0.3, n_samples),
        'home_success_rate': np.random.uniform(0.35, 0.55, n_samples),
        'home_pass_rate': np.random.uniform(0.5, 0.7, n_samples),
        'home_turnovers': np.random.randint(0, 4, n_samples),
        'home_touchdowns': np.random.randint(1, 5, n_samples),
        'away_yards_per_play': np.random.uniform(4, 7, n_samples),
        'away_epa_per_play': np.random.uniform(-0.2, 0.3, n_samples),
        'away_success_rate': np.random.uniform(0.35, 0.55, n_samples),
        'away_pass_rate': np.random.uniform(0.5, 0.7, n_samples),
        'away_turnovers': np.random.randint(0, 4, n_samples),
        'away_touchdowns': np.random.randint(1, 5, n_samples),
    })
    
    # Create differential features
    X['ypp_diff'] = X['home_yards_per_play'] - X['away_yards_per_play']
    X['epa_diff'] = X['home_epa_per_play'] - X['away_epa_per_play']
    X['success_diff'] = X['home_success_rate'] - X['away_success_rate']
    X['turnover_diff'] = X['home_turnovers'] - X['away_turnovers']
    X['pass_rate_diff'] = X['home_pass_rate'] - X['away_pass_rate']
    
    # Create target (biased toward efficiency metrics)
    y = (
        (X['epa_diff'] > 0).astype(int) * 0.4 +
        (X['ypp_diff'] > 0).astype(int) * 0.3 +
        np.random.uniform(0, 1, n_samples) * 0.3
    ) > 0.5
    y = y.astype(int)
    
    # Train model
    predictor = NFLGamePredictor()
    metrics = predictor.train(X, y, tune_hyperparameters=False)
    
    # Show feature importance
    print("\nFeature Importance:")
    print(predictor.get_feature_importance())
