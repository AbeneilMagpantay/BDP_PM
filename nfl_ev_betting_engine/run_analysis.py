"""
NFL EV Betting Engine - Main Entry Point
=========================================

Interactive analysis and reporting tool for NFL betting opportunities.

Usage:
    python run_analysis.py analyze      # Analyze current opportunities
    python run_analysis.py train        # Train/retrain the model
    python run_analysis.py test-odds    # Test odds API connection
    python run_analysis.py test-discord # Test Discord webhook
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.data.odds_fetcher import fetch_nfl_odds, format_odds_display, get_api_key
from src.data.nfl_data_fetcher import fetch_play_by_play_data, fetch_team_data
from src.betting.ev_calculator import (
    american_to_implied_probability,
    calculate_ev,
    calculate_edge
)
from src.alerts.discord_notifier import DiscordNotifier


def cmd_analyze(args):
    """Run full analysis pipeline."""
    print("Running full analysis...")
    print("Use 'python scripts/daily_runner.py' for the complete pipeline")
    
    # Quick odds check
    try:
        odds = fetch_nfl_odds()
        print(format_odds_display(odds))
    except Exception as e:
        print(f"Error: {e}")


def cmd_train(args):
    """Train or retrain the prediction model."""
    import subprocess
    script = Path(__file__).parent / "scripts" / "train_model.py"
    
    cmd = [sys.executable, str(script)]
    if args.fast:
        cmd.append("--fast")
    if args.no_tune:
        cmd.append("--no-tune")
    
    subprocess.run(cmd)


def cmd_test_odds(args):
    """Test the odds API connection."""
    print("Testing Odds API connection...")
    print("-" * 40)
    
    try:
        api_key = get_api_key()
        print(f"✅ API key configured: {api_key[:8]}...")
        
        odds = fetch_nfl_odds()
        print(f"✅ API connection successful")
        print(f"   Found {len(odds)} upcoming games")
        
        if odds:
            print(format_odds_display(odds))
        else:
            print("   (No games currently scheduled)")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ API error: {e}")


def cmd_test_discord(args):
    """Test Discord webhook connection."""
    print("Testing Discord webhook...")
    print("-" * 40)
    
    try:
        notifier = DiscordNotifier()
        
        if notifier.test_connection():
            print("✅ Discord webhook test successful!")
            print("   Check your Discord channel for the test message.")
        else:
            print("❌ Failed to send test message")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def cmd_ev_calc(args):
    """Interactive EV calculator."""
    print("=" * 50)
    print("EXPECTED VALUE CALCULATOR")
    print("=" * 50)
    
    while True:
        try:
            print("\nEnter values (or 'q' to quit):")
            
            prob_input = input("Your predicted probability (0-100%): ").strip()
            if prob_input.lower() == 'q':
                break
            prob = float(prob_input.replace('%', '')) / 100
            
            odds_input = input("American odds (e.g., -110, +150): ").strip()
            if odds_input.lower() == 'q':
                break
            odds = int(odds_input.replace('+', ''))
            
            # Calculate
            implied = american_to_implied_probability(odds)
            ev = calculate_ev(prob, odds)
            edge = calculate_edge(prob, odds)
            
            print(f"\nResults:")
            print(f"  Implied Probability: {implied:.1%}")
            print(f"  Your Probability: {prob:.1%}")
            print(f"  Edge: {edge:+.1f}%")
            print(f"  Expected Value: {ev:+.1f}%")
            
            if ev > 0:
                print(f"  ✅ This is a +EV bet!")
            else:
                print(f"  ❌ This is a -EV bet")
                
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def cmd_info(args):
    """Show project information."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║               NFL EV BETTING ENGINE v1.0.0                        ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  A quantitative sports betting engine that finds Expected Value  ║
║  (EV) opportunities in NFL games.                                ║
║                                                                   ║
║  Features:                                                        ║
║  • Historical NFL data ingestion via nfl_data_py                 ║
║  • Live odds fetching from The Odds API                          ║
║  • XGBoost prediction model for game outcomes                    ║
║  • EV calculation and edge detection                             ║
║  • Discord alerts for +EV opportunities                          ║
║                                                                   ║
║  Quick Start:                                                     ║
║  1. Copy .env.example to .env and add your API keys              ║
║  2. Run: python scripts/train_model.py                           ║
║  3. Run: python scripts/daily_runner.py                          ║
║                                                                   ║
║  Commands:                                                        ║
║  • python run_analysis.py analyze    - Run analysis              ║
║  • python run_analysis.py train      - Train model               ║
║  • python run_analysis.py test-odds  - Test odds API             ║
║  • python run_analysis.py test-discord - Test Discord            ║
║  • python run_analysis.py ev-calc    - EV calculator             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description='NFL EV Betting Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_analysis.py info         Show project information
    python run_analysis.py test-odds    Test odds API connection
    python run_analysis.py test-discord Test Discord webhook
    python run_analysis.py train        Train the prediction model
    python run_analysis.py ev-calc      Interactive EV calculator
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    subparsers.add_parser('info', help='Show project information')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Run full analysis')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train prediction model')
    train_parser.add_argument('--fast', action='store_true', help='Fast training mode')
    train_parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    
    # Test commands
    subparsers.add_parser('test-odds', help='Test odds API connection')
    subparsers.add_parser('test-discord', help='Test Discord webhook')
    
    # EV calculator
    subparsers.add_parser('ev-calc', help='Interactive EV calculator')
    
    args = parser.parse_args()
    
    commands = {
        'info': cmd_info,
        'analyze': cmd_analyze,
        'train': cmd_train,
        'test-odds': cmd_test_odds,
        'test-discord': cmd_test_discord,
        'ev-calc': cmd_ev_calc,
    }
    
    if args.command is None:
        cmd_info(args)
    elif args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
