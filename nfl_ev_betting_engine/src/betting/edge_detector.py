"""
Edge Detector Module
====================

Identifies +EV betting opportunities by comparing model predictions
against live betting odds from bookmakers.

This is the core "edge detection" engine that finds profitable bets.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any

from .ev_calculator import (
    calculate_ev,
    calculate_edge,
    calculate_kelly_criterion,
    american_to_implied_probability,
)


@dataclass
class BettingEdge:
    """
    Represents a detected betting edge opportunity.
    
    Attributes:
        game_id: Unique identifier for the game
        home_team: Home team name
        away_team: Away team name
        commence_time: When the game starts
        bet_side: Which team to bet on ('home' or 'away')
        bet_team: Team name to bet on
        american_odds: The betting odds
        bookmaker: Which bookmaker offers these odds
        model_probability: Model's predicted win probability
        implied_probability: Odds-implied probability
        edge: Difference between model and implied probability
        expected_value: Expected value as percentage
        kelly_bet: Recommended bet size (% of bankroll)
        confidence: Model confidence level
    """
    game_id: str
    home_team: str
    away_team: str
    commence_time: str
    bet_side: str
    bet_team: str
    american_odds: int
    bookmaker: str
    model_probability: float
    implied_probability: float
    edge: float
    expected_value: float
    kelly_bet: float
    confidence: float


class EdgeDetector:
    """
    Detects +EV betting opportunities.
    
    Compares model predictions against bookmaker odds to find
    situations where the model predicts a higher probability
    than the market implies.
    
    Attributes:
        ev_threshold: Minimum EV % to consider a bet (default: 5.0)
        edge_threshold: Minimum edge % to consider a bet (default: 3.0)
        min_confidence: Minimum model confidence (default: 0.55)
        kelly_fraction: Fraction of Kelly to recommend (default: 0.25)
    """
    
    def __init__(
        self,
        ev_threshold: float = None,
        edge_threshold: float = 3.0,
        min_confidence: float = 0.55,
        kelly_fraction: float = 0.25
    ):
        """
        Initialize the edge detector.
        
        Args:
            ev_threshold: Minimum EV % to flag (default from env or 5.0)
            edge_threshold: Minimum edge % to flag
            min_confidence: Minimum model confidence required
            kelly_fraction: Fraction of full Kelly for bet sizing
        """
        self.ev_threshold = ev_threshold or float(os.getenv('EV_THRESHOLD', 5.0))
        self.edge_threshold = edge_threshold
        self.min_confidence = min_confidence
        self.kelly_fraction = kelly_fraction
    
    def find_edges(
        self,
        predictions: List[Dict],
        odds_data: List[Dict],
        best_odds_only: bool = True
    ) -> List[BettingEdge]:
        """
        Find all +EV opportunities.
        
        Args:
            predictions: List of game predictions from the model
            odds_data: List of odds data from odds fetcher
            best_odds_only: If True, only consider best available odds
            
        Returns:
            List of BettingEdge objects sorted by EV (highest first)
        """
        edges = []
        
        # Match predictions with odds
        for prediction in predictions:
            # Find matching odds data
            game_odds = self._find_matching_odds(prediction, odds_data)
            
            if not game_odds:
                continue
            
            # Check both sides of the bet
            for side in ['home', 'away']:
                team = prediction[f'{side}_team']
                prob = prediction[f'{side}_win_prob']
                
                # Get best odds for this team
                best = self._get_best_odds_for_team(game_odds, team, side)
                
                if not best:
                    continue
                
                odds = best['odds']
                bookmaker = best['bookmaker']
                
                # Calculate metrics
                implied = american_to_implied_probability(odds)
                edge = calculate_edge(prob, odds)
                ev = calculate_ev(prob, odds)
                kelly = calculate_kelly_criterion(
                    prob, odds, self.kelly_fraction
                )
                
                # Check if this qualifies as an edge
                if self._qualifies_as_edge(ev, edge, prediction['confidence']):
                    edges.append(BettingEdge(
                        game_id=game_odds.get('game_id', 'unknown'),
                        home_team=prediction['home_team'],
                        away_team=prediction['away_team'],
                        commence_time=game_odds.get('commence_time', ''),
                        bet_side=side,
                        bet_team=team,
                        american_odds=odds,
                        bookmaker=bookmaker,
                        model_probability=prob,
                        implied_probability=implied,
                        edge=edge,
                        expected_value=ev,
                        kelly_bet=kelly,
                        confidence=prediction['confidence']
                    ))
        
        # Sort by EV (highest first)
        edges.sort(key=lambda x: x.expected_value, reverse=True)
        
        return edges
    
    def _find_matching_odds(
        self,
        prediction: Dict,
        odds_data: List[Dict]
    ) -> Optional[Dict]:
        """Find odds data matching a prediction."""
        home = prediction.get('home_team', '').lower()
        away = prediction.get('away_team', '').lower()
        
        for game in odds_data:
            game_home = game.get('home_team', '').lower()
            game_away = game.get('away_team', '').lower()
            
            # Fuzzy match (teams might have different abbreviations)
            if (home in game_home or game_home in home) and \
               (away in game_away or game_away in away):
                return game
        
        return None
    
    def _get_best_odds_for_team(
        self,
        game_odds: Dict,
        team: str,
        side: str
    ) -> Optional[Dict]:
        """Get best odds for a specific team."""
        best_odds = None
        best_bookmaker = None
        
        bookmakers = game_odds.get('bookmakers', {})
        
        for book_name, book_data in bookmakers.items():
            markets = book_data.get('markets', {})
            h2h = markets.get('h2h', {})
            
            # Find this team's odds
            for team_name, odds_info in h2h.items():
                if team.lower() in team_name.lower() or \
                   team_name.lower() in team.lower():
                    odds = odds_info.get('odds')
                    if odds is not None:
                        if best_odds is None or odds > best_odds:
                            best_odds = odds
                            best_bookmaker = book_name
        
        if best_odds:
            return {'odds': best_odds, 'bookmaker': best_bookmaker}
        
        return None
    
    def _qualifies_as_edge(
        self,
        ev: float,
        edge: float,
        confidence: float
    ) -> bool:
        """Check if a bet qualifies as an edge."""
        return (
            ev >= self.ev_threshold and
            edge >= self.edge_threshold and
            confidence >= self.min_confidence
        )
    
    def format_edges_report(self, edges: List[BettingEdge]) -> str:
        """
        Format edges into a readable report.
        
        Args:
            edges: List of detected edges
            
        Returns:
            Formatted string report
        """
        if not edges:
            return "No +EV opportunities found."
        
        lines = []
        lines.append("=" * 70)
        lines.append("🎯 EDGE ALERT - BETTING OPPORTUNITIES DETECTED")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        for i, edge in enumerate(edges, 1):
            lines.append(f"\n#{i} {edge.away_team} @ {edge.home_team}")
            lines.append(f"    🏈 Bet: {edge.bet_team} ({edge.bet_side.upper()})")
            lines.append(f"    📊 Odds: {edge.american_odds:+d} @ {edge.bookmaker}")
            lines.append(f"    🎲 Model Prob: {edge.model_probability:.1%} | "
                        f"Implied: {edge.implied_probability:.1%}")
            lines.append(f"    📈 Edge: {edge.edge:+.1f}% | "
                        f"EV: {edge.expected_value:+.1f}%")
            lines.append(f"    💰 Kelly Bet: {edge.kelly_bet:.1f}% of bankroll")
            lines.append(f"    ⏰ Game Time: {edge.commence_time}")
            lines.append("-" * 70)
        
        lines.append(f"\nTotal opportunities: {len(edges)}")
        
        return "\n".join(lines)
    
    def get_edges_as_dicts(self, edges: List[BettingEdge]) -> List[Dict]:
        """Convert edges to dictionary format for JSON serialization."""
        return [
            {
                'game_id': e.game_id,
                'matchup': f"{e.away_team} @ {e.home_team}",
                'bet_team': e.bet_team,
                'bet_side': e.bet_side,
                'odds': e.american_odds,
                'bookmaker': e.bookmaker,
                'model_prob': round(e.model_probability, 3),
                'implied_prob': round(e.implied_probability, 3),
                'edge': e.edge,
                'ev': e.expected_value,
                'kelly_bet': e.kelly_bet,
                'confidence': round(e.confidence, 3),
                'commence_time': e.commence_time
            }
            for e in edges
        ]


if __name__ == "__main__":
    # Test with sample data
    print("Testing Edge Detector...")
    
    detector = EdgeDetector(ev_threshold=3.0, edge_threshold=2.0)
    
    # Sample prediction
    predictions = [
        {
            'home_team': 'Kansas City Chiefs',
            'away_team': 'Buffalo Bills',
            'home_win_prob': 0.58,
            'away_win_prob': 0.42,
            'confidence': 0.60
        }
    ]
    
    # Sample odds (simplified structure)
    odds_data = [
        {
            'game_id': 'test123',
            'home_team': 'Kansas City Chiefs',
            'away_team': 'Buffalo Bills',
            'commence_time': '2024-01-21T18:00:00Z',
            'bookmakers': {
                'draftkings': {
                    'markets': {
                        'h2h': {
                            'Kansas City Chiefs': {'odds': -130},
                            'Buffalo Bills': {'odds': +110}
                        }
                    }
                },
                'fanduel': {
                    'markets': {
                        'h2h': {
                            'Kansas City Chiefs': {'odds': -125},
                            'Buffalo Bills': {'odds': +115}
                        }
                    }
                }
            }
        }
    ]
    
    edges = detector.find_edges(predictions, odds_data)
    print(detector.format_edges_report(edges))
