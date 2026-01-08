"""
Daily Runner Script
===================

Automated script that runs daily to find +EV betting opportunities.
Fetches latest odds, generates predictions, and sends Discord alerts.

Usage:
    python scripts/daily_runner.py [--no-alerts] [--verbose]
    
Schedule with Windows Task Scheduler or cron for automation.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nfl_data_fetcher import aggregate_to_game_stats, fetch_play_by_play_data
from src.data.odds_fetcher import fetch_nfl_odds, parse_odds_for_game, get_best_odds
from src.data.preprocessor import get_team_recent_stats
from src.model.predictor import GamePredictionService
from src.betting.edge_detector import EdgeDetector
from src.alerts.discord_notifier import DiscordNotifier


def main():
    parser = argparse.ArgumentParser(
        description='Daily NFL EV opportunity scanner'
    )
    parser.add_argument(
        '--no-alerts',
        action='store_true',
        help='Skip sending Discord alerts (just print results)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed output'
    )
    parser.add_argument(
        '--ev-threshold',
        type=float,
        default=5.0,
        help='Minimum EV percentage to alert (default: 5.0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NFL EV BETTING ENGINE - DAILY SCAN")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Step 1: Fetch current odds
        print("\n📊 Fetching live NFL odds...")
        try:
            odds_data = fetch_nfl_odds()
            print(f"   Found {len(odds_data)} upcoming games")
        except ValueError as e:
            print(f"   ⚠️ {e}")
            print("   Please add your ODDS_API_KEY to .env")
            return
        
        if not odds_data:
            print("   No upcoming games found (might be off-season)")
            return
        
        # Parse odds
        parsed_odds = [parse_odds_for_game(game) for game in odds_data]
        
        if args.verbose:
            best = get_best_odds(odds_data)
            for game in best[:3]:
                print(f"   {game['away_team']} @ {game['home_team']}")
        
        # Step 2: Load prediction model
        print("\n🤖 Loading prediction model...")
        try:
            prediction_service = GamePredictionService()
            model_info = prediction_service.get_model_info()
            print(f"   Model accuracy: {model_info['accuracy']:.1%}")
        except FileNotFoundError as e:
            print(f"   ⚠️ {e}")
            print("   Run 'python scripts/train_model.py' first")
            return
        
        # Step 3: Get recent team stats
        print("\n📈 Fetching recent team statistics...")
        current_year = datetime.now().year
        years = [current_year - 1, current_year]  # Get current and last season
        
        try:
            pbp = fetch_play_by_play_data(years, use_cache=True)
            game_stats = aggregate_to_game_stats(pbp)
            print(f"   Loaded stats for {game_stats['team'].nunique()} teams")
        except Exception as e:
            print(f"   ⚠️ Error loading stats: {e}")
            return
        
        # Step 4: Generate predictions for each game
        print("\n🎯 Generating predictions...")
        predictions = []
        
        for game in parsed_odds:
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get recent stats for each team
            # Note: Team names from odds API might not match exactly
            # This is a simplified matching - production would need better mapping
            home_stats = get_team_recent_stats(game_stats, home_team)
            away_stats = get_team_recent_stats(game_stats, away_team)
            
            # If stats not found, try partial matching
            if not home_stats:
                matching_teams = game_stats[
                    game_stats['team'].str.contains(home_team.split()[-1], case=False)
                ]['team'].unique()
                if len(matching_teams) > 0:
                    home_stats = get_team_recent_stats(game_stats, matching_teams[0])
            
            if not away_stats:
                matching_teams = game_stats[
                    game_stats['team'].str.contains(away_team.split()[-1], case=False)
                ]['team'].unique()
                if len(matching_teams) > 0:
                    away_stats = get_team_recent_stats(game_stats, matching_teams[0])
            
            # Skip if we couldn't find stats
            if not home_stats or not away_stats:
                if args.verbose:
                    print(f"   ⚠️ Skipping {away_team} @ {home_team} (missing stats)")
                continue
            
            # Generate prediction
            prediction = prediction_service.predict_game(home_stats, away_stats)
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': prediction['home_win_prob'],
                'away_win_prob': prediction['away_win_prob'],
                'confidence': prediction['confidence'],
                'commence_time': game.get('commence_time', '')
            })
            
            if args.verbose:
                print(f"   {away_team} @ {home_team}: "
                      f"Home {prediction['home_win_prob']:.1%} | "
                      f"Away {prediction['away_win_prob']:.1%}")
        
        print(f"   Generated predictions for {len(predictions)} games")
        
        # Step 5: Find edges
        print("\n🔍 Scanning for +EV opportunities...")
        detector = EdgeDetector(ev_threshold=args.ev_threshold)
        edges = detector.find_edges(predictions, parsed_odds)
        
        if edges:
            print(f"\n   ✅ Found {len(edges)} +EV opportunities!")
            print(detector.format_edges_report(edges))
        else:
            print("   No +EV opportunities found at current thresholds")
        
        # Step 6: Send Discord alerts
        if edges and not args.no_alerts:
            print("\n📱 Sending Discord alerts...")
            try:
                notifier = DiscordNotifier()
                edge_dicts = detector.get_edges_as_dicts(edges)
                sent = notifier.send_edge_alerts(edge_dicts)
                print(f"   Sent {sent} alerts to Discord")
            except ValueError as e:
                print(f"   ⚠️ {e}")
                print("   Add DISCORD_WEBHOOK_URL to .env to enable alerts")
        
        print("\n" + "=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
