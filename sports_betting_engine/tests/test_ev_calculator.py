"""
Unit Tests for EV Calculator
=============================

Tests the core betting mathematics functions.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.ev_calculator import (
    american_to_implied_probability,
    implied_probability_to_american,
    american_to_decimal,
    decimal_to_american,
    calculate_ev,
    calculate_edge,
    calculate_kelly_criterion,
    calculate_payout,
    calculate_no_vig_probability,
)


class TestOddsConversion:
    """Tests for odds format conversions."""
    
    def test_american_to_implied_favorite(self):
        """Test conversion for favorite (negative odds)."""
        # -150 means bet $150 to win $100
        # Implied prob = 150 / (150 + 100) = 60%
        prob = american_to_implied_probability(-150)
        assert abs(prob - 0.6) < 0.001
    
    def test_american_to_implied_underdog(self):
        """Test conversion for underdog (positive odds)."""
        # +150 means bet $100 to win $150
        # Implied prob = 100 / (150 + 100) = 40%
        prob = american_to_implied_probability(+150)
        assert abs(prob - 0.4) < 0.001
    
    def test_american_to_implied_even(self):
        """Test conversion for even odds."""
        # +100 means bet $100 to win $100
        # Implied prob = 100 / (100 + 100) = 50%
        prob = american_to_implied_probability(+100)
        assert abs(prob - 0.5) < 0.001
    
    def test_implied_to_american_favorite(self):
        """Test reverse conversion for favorite."""
        odds = implied_probability_to_american(0.6)
        assert odds == -150
    
    def test_implied_to_american_underdog(self):
        """Test reverse conversion for underdog."""
        odds = implied_probability_to_american(0.4)
        assert odds == 150
    
    def test_american_to_decimal_positive(self):
        """Test American to decimal conversion (positive odds)."""
        decimal = american_to_decimal(+150)
        assert abs(decimal - 2.5) < 0.01
    
    def test_american_to_decimal_negative(self):
        """Test American to decimal conversion (negative odds)."""
        decimal = american_to_decimal(-150)
        assert abs(decimal - 1.667) < 0.01
    
    def test_decimal_to_american_underdog(self):
        """Test decimal to American conversion (underdog)."""
        odds = decimal_to_american(2.5)
        assert odds == 150
    
    def test_decimal_to_american_favorite(self):
        """Test decimal to American conversion (favorite)."""
        odds = decimal_to_american(1.5)
        assert odds == -200


class TestEVCalculations:
    """Tests for Expected Value calculations."""
    
    def test_ev_positive(self):
        """Test +EV calculation when model has edge."""
        # Model says 55%, odds imply ~52.4% (-110)
        ev = calculate_ev(0.55, -110)
        assert ev > 0
        assert abs(ev - 4.55) < 0.5  # Approximately +4.55%
    
    def test_ev_negative(self):
        """Test -EV calculation when no edge exists."""
        # Model says 50%, odds imply ~52.4% (-110)
        ev = calculate_ev(0.50, -110)
        assert ev < 0
    
    def test_ev_zero(self):
        """Test near-zero EV when model matches implied."""
        # If model probability equals implied, EV should be near 0
        prob = american_to_implied_probability(-110)
        ev = calculate_ev(prob, -110)
        assert abs(ev) < 1  # Should be very small
    
    def test_edge_positive(self):
        """Test edge calculation with positive edge."""
        edge = calculate_edge(0.55, -110)
        implied = american_to_implied_probability(-110)
        expected_edge = (0.55 - implied) * 100
        assert abs(edge - expected_edge) < 0.1
    
    def test_edge_negative(self):
        """Test edge calculation with negative edge."""
        edge = calculate_edge(0.45, -110)
        assert edge < 0


class TestKellyCriterion:
    """Tests for Kelly Criterion bet sizing."""
    
    def test_kelly_with_edge(self):
        """Test Kelly bet size when edge exists."""
        kelly = calculate_kelly_criterion(0.55, -110, kelly_fraction=1.0)
        assert kelly > 0
    
    def test_kelly_no_edge(self):
        """Test Kelly returns 0 when no edge."""
        kelly = calculate_kelly_criterion(0.50, -110, kelly_fraction=1.0)
        assert kelly == 0
    
    def test_kelly_fraction(self):
        """Test that fractional Kelly reduces bet size."""
        full_kelly = calculate_kelly_criterion(0.60, -110, kelly_fraction=1.0)
        quarter_kelly = calculate_kelly_criterion(0.60, -110, kelly_fraction=0.25)
        assert abs(quarter_kelly - full_kelly * 0.25) < 0.1


class TestPayout:
    """Tests for payout calculations."""
    
    def test_payout_favorite(self):
        """Test payout for favorite bet."""
        profit, total = calculate_payout(-150, 150)
        assert abs(profit - 100) < 0.1
        assert abs(total - 250) < 0.1
    
    def test_payout_underdog(self):
        """Test payout for underdog bet."""
        profit, total = calculate_payout(+150, 100)
        assert abs(profit - 150) < 0.1
        assert abs(total - 250) < 0.1


class TestNoVigProbability:
    """Tests for removing vigorish."""
    
    def test_no_vig_standard(self):
        """Test vig removal for standard -110/-110 line."""
        fair_a, fair_b = calculate_no_vig_probability(-110, -110)
        assert abs(fair_a - 0.5) < 0.01
        assert abs(fair_b - 0.5) < 0.01
    
    def test_no_vig_sums_to_one(self):
        """Test that no-vig probabilities sum to 1."""
        fair_a, fair_b = calculate_no_vig_probability(-150, +130)
        assert abs(fair_a + fair_b - 1.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
