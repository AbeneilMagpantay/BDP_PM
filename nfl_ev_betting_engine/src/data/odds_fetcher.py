"""
Odds Fetcher Module
===================

Fetches live NFL betting odds from The Odds API.
Supports moneyline, spread, and totals markets.

API Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""

import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# API Configuration
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_ODDS_FORMAT = "american"


def get_api_key() -> str:
    """
    Get The Odds API key from environment variables.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not configured
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "ODDS_API_KEY not configured. "
            "Please set it in your .env file. "
            "Get a free key at: https://the-odds-api.com/"
        )
    return api_key


def fetch_odds(
    sport_key: str,
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    odds_format: str = DEFAULT_ODDS_FORMAT
) -> List[Dict[str, Any]]:
    """
    Fetch current odds for any sport from The Odds API.
    
    Args:
        sport_key: Sport identifier (e.g., 'americanfootball_nfl', 'basketball_nba', 'soccer_epl')
        markets: Comma-separated markets (h2h, spreads, totals)
        regions: Comma-separated regions for bookmakers (us, uk, eu, au)
        odds_format: 'american' or 'decimal'
        
    Returns:
        List of game dictionaries with odds from multiple bookmakers
    """
    api_key = get_api_key()
    
    url = f"{BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format
    }
    
    sport_name = sport_key.split('_')[-1].upper()
    print(f"Fetching {sport_name} odds from The Odds API...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        remaining = response.headers.get('x-requests-remaining', 'unknown')
        used = response.headers.get('x-requests-used', 'unknown')
        print(f"API requests: {used} used, {remaining} remaining this month")
        
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise ValueError("Invalid API key. Please check your ODDS_API_KEY.")
        elif response.status_code == 429:
            raise ValueError("API rate limit exceeded. Please wait or upgrade your plan.")
        else:
            raise e
    except requests.exceptions.RequestException as e:
        print(f"Error fetching odds: {e}")
        raise


def fetch_nfl_odds(
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    odds_format: str = DEFAULT_ODDS_FORMAT
) -> List[Dict[str, Any]]:
    """Fetch current NFL odds. Wrapper for backwards compatibility."""
    return fetch_odds(SPORT_KEY, markets, regions, odds_format)


def fetch_nba_odds(
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    odds_format: str = DEFAULT_ODDS_FORMAT
) -> List[Dict[str, Any]]:
    """Fetch current NBA odds."""
    return fetch_odds("basketball_nba", markets, regions, odds_format)


def fetch_soccer_odds(
    league: str = "soccer_epl",  # English Premier League by default
    markets: str = "h2h",  # Soccer typically just h2h
    regions: str = "us,uk,eu",
    odds_format: str = DEFAULT_ODDS_FORMAT
) -> List[Dict[str, Any]]:
    """Fetch current Soccer odds for a specific league."""
    return fetch_odds(league, markets, regions, odds_format)


def parse_odds_for_game(game: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single game's odds into a structured format.
    
    Args:
        game: Raw game dictionary from API
        
    Returns:
        Structured dictionary with parsed odds for all markets
    """
    parsed = {
        "game_id": game.get("id"),
        "sport": game.get("sport_key"),
        "commence_time": game.get("commence_time"),
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "bookmakers": {}
    }
    
    # Parse each bookmaker's odds
    for bookmaker in game.get("bookmakers", []):
        book_name = bookmaker.get("key")
        parsed["bookmakers"][book_name] = {
            "last_update": bookmaker.get("last_update"),
            "markets": {}
        }
        
        for market in bookmaker.get("markets", []):
            market_key = market.get("key")
            outcomes = {}
            
            for outcome in market.get("outcomes", []):
                team = outcome.get("name")
                price = outcome.get("price")
                point = outcome.get("point")  # For spreads/totals
                
                outcomes[team] = {
                    "odds": price,
                    "point": point
                }
            
            parsed["bookmakers"][book_name]["markets"][market_key] = outcomes
    
    return parsed


def get_best_odds(games: List[Dict[str, Any]], market: str = "h2h") -> List[Dict[str, Any]]:
    """
    Find the best available odds for each team across all bookmakers.
    
    Args:
        games: List of game dictionaries from fetch_nfl_odds()
        market: Market to analyze ('h2h', 'spreads', 'totals')
        
    Returns:
        List of dictionaries with best odds for each game
    """
    best_odds = []
    
    for game in games:
        parsed = parse_odds_for_game(game)
        
        game_best = {
            "game_id": parsed["game_id"],
            "home_team": parsed["home_team"],
            "away_team": parsed["away_team"],
            "commence_time": parsed["commence_time"],
            "home_best_odds": None,
            "home_best_book": None,
            "away_best_odds": None,
            "away_best_book": None,
        }
        
        # Find best odds for each team
        for book_name, book_data in parsed["bookmakers"].items():
            if market not in book_data["markets"]:
                continue
                
            market_data = book_data["markets"][market]
            
            # Home team
            if parsed["home_team"] in market_data:
                home_odds = market_data[parsed["home_team"]]["odds"]
                if game_best["home_best_odds"] is None or home_odds > game_best["home_best_odds"]:
                    game_best["home_best_odds"] = home_odds
                    game_best["home_best_book"] = book_name
            
            # Away team
            if parsed["away_team"] in market_data:
                away_odds = market_data[parsed["away_team"]]["odds"]
                if game_best["away_best_odds"] is None or away_odds > game_best["away_best_odds"]:
                    game_best["away_best_odds"] = away_odds
                    game_best["away_best_book"] = book_name
        
        if game_best["home_best_odds"] is not None:
            best_odds.append(game_best)
    
    return best_odds


def american_to_implied_probability(odds: int) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -150, +120)
        
    Returns:
        Implied probability as decimal (0 to 1)
        
    Example:
        >>> american_to_implied_probability(-150)
        0.6
        >>> american_to_implied_probability(+150)
        0.4
    """
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.
    
    Args:
        odds: American odds
        
    Returns:
        Decimal odds (e.g., 2.50)
    """
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def format_odds_display(games: List[Dict[str, Any]]) -> str:
    """
    Format odds data for display.
    
    Args:
        games: List of parsed game dictionaries
        
    Returns:
        Formatted string for console output
    """
    output = []
    output.append("=" * 70)
    output.append("NFL BETTING ODDS")
    output.append(f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 70)
    
    best = get_best_odds(games)
    
    for game in best:
        output.append(f"\n{game['away_team']} @ {game['home_team']}")
        output.append(f"  Game Time: {game['commence_time']}")
        
        if game['home_best_odds']:
            home_imp = american_to_implied_probability(game['home_best_odds'])
            output.append(
                f"  {game['home_team']}: {game['home_best_odds']:+d} "
                f"({home_imp:.1%} implied) @ {game['home_best_book']}"
            )
        
        if game['away_best_odds']:
            away_imp = american_to_implied_probability(game['away_best_odds'])
            output.append(
                f"  {game['away_team']}: {game['away_best_odds']:+d} "
                f"({away_imp:.1%} implied) @ {game['away_best_book']}"
            )
    
    return "\n".join(output)


if __name__ == "__main__":
    # Test the module
    print("Testing Odds Fetcher...")
    
    try:
        odds = fetch_nfl_odds()
        print(f"\nFound {len(odds)} upcoming games")
        
        if odds:
            print(format_odds_display(odds))
        else:
            print("No upcoming games found (might be off-season)")
            
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
