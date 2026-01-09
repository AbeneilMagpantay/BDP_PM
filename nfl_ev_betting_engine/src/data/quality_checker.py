"""
Data Quality Checker
====================

Validates data integrity before training or predictions.
Catches issues early to prevent garbage-in-garbage-out.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class DataQualityReport:
    """Container for data quality check results."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
        self.stats = {}
    
    def add_pass(self, check_name: str):
        self.checks_passed += 1
        
    def add_fail(self, check_name: str, message: str):
        self.checks_failed += 1
        self.errors.append(f"[FAIL] {check_name}: {message}")
    
    def add_warning(self, check_name: str, message: str):
        self.warnings.append(f"[WARN] {check_name}: {message}")
    
    def is_passing(self) -> bool:
        return self.checks_failed == 0
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            f"Checks passed: {self.checks_passed}",
            f"Checks failed: {self.checks_failed}",
            f"Warnings: {len(self.warnings)}",
            ""
        ]
        
        if self.errors:
            lines.append("ERRORS:")
            lines.extend(self.errors)
            lines.append("")
        
        if self.warnings:
            lines.append("WARNINGS:")
            lines.extend(self.warnings)
            lines.append("")
        
        if self.stats:
            lines.append("STATISTICS:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def check_play_by_play_data(df: pd.DataFrame) -> DataQualityReport:
    """
    Validate play-by-play data quality.
    
    Checks:
    - No empty dataframe
    - Required columns exist
    - Reasonable value ranges
    - Missing value thresholds
    """
    report = DataQualityReport()
    
    # Check 1: Not empty
    if len(df) == 0:
        report.add_fail("non_empty", "DataFrame is empty")
        return report
    report.add_pass("non_empty")
    report.stats["total_rows"] = len(df)
    
    # Check 2: Required columns
    required_cols = ['game_id', 'play_type', 'yards_gained', 'posteam']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        report.add_fail("required_columns", f"Missing: {missing_cols}")
    else:
        report.add_pass("required_columns")
    
    # Check 3: Yards gained in reasonable range
    if 'yards_gained' in df.columns:
        min_yards = df['yards_gained'].min()
        max_yards = df['yards_gained'].max()
        if min_yards < -50 or max_yards > 120:
            report.add_warning("yards_range", f"Unusual range: {min_yards} to {max_yards}")
        else:
            report.add_pass("yards_range")
        report.stats["yards_range"] = f"{min_yards} to {max_yards}"
    
    # Check 4: Missing values
    if 'epa' in df.columns:
        epa_missing = df['epa'].isna().sum()
        epa_missing_pct = epa_missing / len(df) * 100
        if epa_missing_pct > 20:
            report.add_warning("epa_missing", f"{epa_missing_pct:.1f}% missing EPA values")
        else:
            report.add_pass("epa_completeness")
        report.stats["epa_missing_pct"] = f"{epa_missing_pct:.1f}%"
    
    # Check 5: Unique games
    if 'game_id' in df.columns:
        unique_games = df['game_id'].nunique()
        report.stats["unique_games"] = unique_games
        if unique_games < 10:
            report.add_warning("game_count", f"Only {unique_games} unique games")
        else:
            report.add_pass("game_count")
    
    # Check 6: Teams
    if 'posteam' in df.columns:
        unique_teams = df['posteam'].dropna().nunique()
        report.stats["unique_teams"] = unique_teams
        if unique_teams < 32:
            report.add_warning("team_count", f"Only {unique_teams} teams (expected 32)")
        else:
            report.add_pass("team_count")
    
    return report


def check_game_stats(df: pd.DataFrame) -> DataQualityReport:
    """Validate aggregated game statistics."""
    report = DataQualityReport()
    
    # Check 1: Not empty
    if len(df) == 0:
        report.add_fail("non_empty", "DataFrame is empty")
        return report
    report.add_pass("non_empty")
    report.stats["total_team_games"] = len(df)
    
    # Check 2: Reasonable efficiency metrics
    if 'yards_per_play' in df.columns:
        ypp_mean = df['yards_per_play'].mean()
        ypp_std = df['yards_per_play'].std()
        if ypp_mean < 3 or ypp_mean > 8:
            report.add_warning("ypp_range", f"Unusual mean YPP: {ypp_mean:.2f}")
        else:
            report.add_pass("ypp_range")
        report.stats["avg_yards_per_play"] = f"{ypp_mean:.2f} (std: {ypp_std:.2f})"
    
    # Check 3: EPA sanity
    if 'epa_per_play' in df.columns:
        epa_mean = df['epa_per_play'].mean()
        if abs(epa_mean) > 0.5:
            report.add_warning("epa_bias", f"EPA mean far from 0: {epa_mean:.3f}")
        else:
            report.add_pass("epa_centered")
        report.stats["avg_epa_per_play"] = f"{epa_mean:.3f}"
    
    # Check 4: Duplicate detection
    if 'game_id' in df.columns and 'team' in df.columns:
        duplicates = df.duplicated(subset=['game_id', 'team']).sum()
        if duplicates > 0:
            report.add_fail("no_duplicates", f"{duplicates} duplicate team-games found")
        else:
            report.add_pass("no_duplicates")
    
    # Check 5: Success rate bounds
    if 'success_rate' in df.columns:
        invalid = ((df['success_rate'] < 0) | (df['success_rate'] > 1)).sum()
        if invalid > 0:
            report.add_fail("success_rate_bounds", f"{invalid} rows with invalid success_rate")
        else:
            report.add_pass("success_rate_bounds")
    
    return report


def check_training_data(df: pd.DataFrame) -> DataQualityReport:
    """Validate training dataset before model training."""
    report = DataQualityReport()
    
    # Check 1: Not empty
    if len(df) == 0:
        report.add_fail("non_empty", "Training data is empty")
        return report
    report.add_pass("non_empty")
    report.stats["training_samples"] = len(df)
    
    # Check 2: Target variable
    if 'home_win' in df.columns:
        win_rate = df['home_win'].mean()
        report.stats["home_win_rate"] = f"{win_rate:.1%}"
        if win_rate < 0.4 or win_rate > 0.6:
            report.add_warning("class_balance", f"Imbalanced: {win_rate:.1%} home wins")
        else:
            report.add_pass("class_balance")
    else:
        report.add_fail("target_exists", "Missing 'home_win' column")
    
    # Check 3: No all-null columns
    null_cols = df.columns[df.isna().all()].tolist()
    if null_cols:
        report.add_fail("no_empty_columns", f"All-null columns: {null_cols}")
    else:
        report.add_pass("no_empty_columns")
    
    # Check 4: Minimum sample size
    min_samples = 100
    if len(df) < min_samples:
        report.add_warning("sample_size", f"Only {len(df)} samples (recommend {min_samples}+)")
    else:
        report.add_pass("sample_size")
    
    # Check 5: Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report.stats["numeric_features"] = len(numeric_cols)
    
    # Check for infinite values
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        report.add_fail("no_infinites", f"{inf_count} infinite values found")
    else:
        report.add_pass("no_infinites")
    
    return report


def check_odds_data(odds_list: list) -> DataQualityReport:
    """Validate odds data from API."""
    report = DataQualityReport()
    
    # Check 1: Not empty
    if not odds_list:
        report.add_fail("non_empty", "No odds data received")
        return report
    report.add_pass("non_empty")
    report.stats["games_with_odds"] = len(odds_list)
    
    # Check 2: Required fields
    required = ['home_team', 'away_team', 'bookmakers']
    for game in odds_list:
        for field in required:
            if field not in game:
                report.add_fail("required_fields", f"Missing '{field}' in game data")
                return report
    report.add_pass("required_fields")
    
    # Check 3: Has bookmaker odds
    games_with_odds = sum(1 for g in odds_list if g.get('bookmakers'))
    if games_with_odds < len(odds_list):
        report.add_warning("bookmaker_coverage", 
                          f"Only {games_with_odds}/{len(odds_list)} games have odds")
    else:
        report.add_pass("bookmaker_coverage")
    
    return report


def run_full_quality_check(
    pbp_data: pd.DataFrame = None,
    game_stats: pd.DataFrame = None,
    training_data: pd.DataFrame = None,
    odds_data: list = None,
    verbose: bool = True
) -> bool:
    """
    Run all applicable quality checks.
    
    Returns True if all critical checks pass.
    """
    all_passed = True
    
    if pbp_data is not None:
        if verbose:
            print("\nChecking play-by-play data...")
        report = check_play_by_play_data(pbp_data)
        if verbose:
            print(report.summary())
        all_passed = all_passed and report.is_passing()
    
    if game_stats is not None:
        if verbose:
            print("\nChecking game statistics...")
        report = check_game_stats(game_stats)
        if verbose:
            print(report.summary())
        all_passed = all_passed and report.is_passing()
    
    if training_data is not None:
        if verbose:
            print("\nChecking training data...")
        report = check_training_data(training_data)
        if verbose:
            print(report.summary())
        all_passed = all_passed and report.is_passing()
    
    if odds_data is not None:
        if verbose:
            print("\nChecking odds data...")
        report = check_odds_data(odds_data)
        if verbose:
            print(report.summary())
        all_passed = all_passed and report.is_passing()
    
    return all_passed


if __name__ == "__main__":
    print("Data Quality Checker - Demo")
    print("-" * 40)
    
    # Create sample data
    sample_df = pd.DataFrame({
        'game_id': ['2023_01_KC_DET'] * 100,
        'play_type': ['pass', 'run'] * 50,
        'yards_gained': np.random.randint(-5, 30, 100),
        'posteam': ['KC'] * 50 + ['DET'] * 50,
        'epa': np.random.normal(0, 0.5, 100)
    })
    
    report = check_play_by_play_data(sample_df)
    print(report.summary())
