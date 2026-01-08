"""
Model Training Script
=====================

One-time script to train the NFL game prediction model.
Fetches historical data, engineers features, and trains XGBoost model.

Usage:
    python scripts/train_model.py [--years 2020 2021 2022 2023 2024] [--fast]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nfl_data_fetcher import build_training_dataset, aggregate_to_game_stats
from src.data.preprocessor import create_matchup_features, prepare_features_for_training
from src.model.trainer import NFLGamePredictor


def main():
    parser = argparse.ArgumentParser(
        description='Train NFL game prediction model'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=[2021, 2022, 2023, 2024],
        help='Years of data to use for training (default: 2021-2024)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use reduced hyperparameter grid for faster training'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (use defaults)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NFL EV BETTING ENGINE - MODEL TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Years: {args.years}")
    print(f"  Fast mode: {args.fast}")
    print(f"  Hyperparameter tuning: {not args.no_tune}")
    print(f"  Test size: {args.test_size}")
    
    # Step 1: Build training dataset
    print("\n" + "-" * 70)
    print("STEP 1: Building Training Dataset")
    print("-" * 70)
    
    training_data = build_training_dataset(args.years)
    print(f"\nDataset shape: {training_data.shape}")
    print(f"Seasons: {sorted(training_data['season'].unique())}")
    print(f"Total games: {len(training_data)}")
    print(f"Home win rate: {training_data['home_win'].mean():.1%}")
    
    # Step 2: Feature engineering
    print("\n" + "-" * 70)
    print("STEP 2: Feature Engineering")
    print("-" * 70)
    
    training_data = create_matchup_features(training_data)
    X, y, feature_names = prepare_features_for_training(training_data)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # Step 3: Train model
    print("\n" + "-" * 70)
    print("STEP 3: Training Model")
    print("-" * 70)
    
    predictor = NFLGamePredictor()
    metrics = predictor.train(
        X, y,
        test_size=args.test_size,
        tune_hyperparameters=not args.no_tune,
        fast_mode=args.fast
    )
    
    # Step 4: Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE")
    print("-" * 70)
    
    importance = predictor.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Step 5: Save model
    print("\n" + "-" * 70)
    print("SAVING MODEL")
    print("-" * 70)
    
    model_path = predictor.save()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.1%} (+/- {metrics['cv_accuracy_std']*2:.1%})")
    print(f"\nModel saved to: {model_path}")
    print("\nNext steps:")
    print("  1. Add your API keys to .env")
    print("  2. Run: python scripts/daily_runner.py")


if __name__ == "__main__":
    main()
