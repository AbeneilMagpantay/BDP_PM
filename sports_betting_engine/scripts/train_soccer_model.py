"""
Train Soccer Prediction Model
==============================

Fetches historical soccer data and trains an XGBoost model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.soccer_data_fetcher import build_soccer_training_data
from src.model.soccer_trainer import SoccerMatchPredictor


def main():
    print("=" * 60)
    print("SOCCER MODEL TRAINING")
    print("=" * 60)
    
    # Build training data
    print("\n1. Building training dataset...")
    X, y = build_soccer_training_data()
    
    # Train model
    print("\n2. Training model...")
    predictor = SoccerMatchPredictor()
    metrics = predictor.train(X, y)
    
    # Save model
    print("\n3. Saving model...")
    model_path = predictor.save()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return predictor


if __name__ == "__main__":
    main()
