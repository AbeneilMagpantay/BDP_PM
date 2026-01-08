"""
Expected Value Calculator Module
================================

Core betting mathematics for calculating Expected Value (EV).
Converts odds formats and computes EV for betting opportunities.

Key Concepts:
- Implied Probability: What the odds suggest the probability is
- True Probability: What your model predicts the probability is
- Expected Value: The expected profit/loss per dollar wagered
"""

from typing import Union, Tuple


def american_to_implied_probability(odds: int) -> float:
    """
    Convert American odds to implied probability.
    
    American odds represent how much you need to bet to win $100 (negative)
    or how much you win on a $100 bet (positive).
    
    Args:
        odds: American odds (e.g., -150, +120)
        
    Returns:
        Implied probability as decimal (0 to 1)
        
    Examples:
        >>> american_to_implied_probability(-150)  # Favorite
        0.6  # 60% implied probability
        
        >>> american_to_implied_probability(+150)  # Underdog
        0.4  # 40% implied probability
        
        >>> american_to_implied_probability(-110)  # Coin flip with vig
        0.5238  # 52.38% implied probability
    """
    if odds > 0:
        # Positive odds: +150 means bet $100 to win $150
        return 100 / (odds + 100)
    else:
        # Negative odds: -150 means bet $150 to win $100
        return abs(odds) / (abs(odds) + 100)


def implied_probability_to_american(prob: float) -> int:
    """
    Convert implied probability to American odds.
    
    Args:
        prob: Probability as decimal (0 to 1)
        
    Returns:
        American odds (rounded to nearest integer)
        
    Examples:
        >>> implied_probability_to_american(0.6)
        -150
        
        >>> implied_probability_to_american(0.4)
        +150
    """
    if prob <= 0 or prob >= 1:
        raise ValueError("Probability must be between 0 and 1 (exclusive)")
    
    if prob >= 0.5:
        # Favorite (negative odds)
        return int(round(-100 * prob / (1 - prob)))
    else:
        # Underdog (positive odds)
        return int(round(100 * (1 - prob) / prob))


def american_to_decimal(odds: int) -> float:
    """
    Convert American odds to decimal odds.
    
    Decimal odds represent the total payout per $1 wagered (including stake).
    
    Args:
        odds: American odds
        
    Returns:
        Decimal odds (e.g., 2.50 means $2.50 total return per $1 bet)
        
    Examples:
        >>> american_to_decimal(+150)
        2.50
        
        >>> american_to_decimal(-150)
        1.67
    """
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds.
    
    Args:
        decimal_odds: Decimal odds (must be > 1.0)
        
    Returns:
        American odds
    """
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    
    if decimal_odds >= 2.0:
        # Underdog (positive American odds)
        return int(round((decimal_odds - 1) * 100))
    else:
        # Favorite (negative American odds)
        return int(round(-100 / (decimal_odds - 1)))


def calculate_payout(odds: int, stake: float = 100) -> Tuple[float, float]:
    """
    Calculate potential payout for a bet.
    
    Args:
        odds: American odds
        stake: Amount wagered
        
    Returns:
        Tuple of (profit_if_win, total_return_if_win)
        
    Examples:
        >>> calculate_payout(-150, 100)
        (66.67, 166.67)  # Risk $100 to win $66.67
        
        >>> calculate_payout(+150, 100)
        (150.00, 250.00)  # Risk $100 to win $150
    """
    decimal_odds = american_to_decimal(odds)
    total_return = stake * decimal_odds
    profit = total_return - stake
    
    return (round(profit, 2), round(total_return, 2))


def calculate_ev(
    model_probability: float,
    american_odds: int
) -> float:
    """
    Calculate Expected Value for a bet.
    
    EV represents the expected profit/loss per dollar wagered.
    Positive EV (+EV) means the bet is profitable long-term.
    
    Formula:
    EV = (Win Probability × Profit if Win) - (Lose Probability × Stake)
    
    Simplified for unit stake:
    EV = (Model Prob × Decimal Odds) - 1
    
    Args:
        model_probability: Your model's predicted probability (0 to 1)
        american_odds: The betting odds offered
        
    Returns:
        Expected Value as percentage of stake (e.g., 5.0 means +5% EV)
        
    Examples:
        >>> calculate_ev(0.55, -110)  # Model says 55% but odds imply 52.4%
        4.5  # +4.5% EV
        
        >>> calculate_ev(0.50, -110)  # Model agrees with fair odds
        -4.5  # -4.5% EV (the vig)
        
        >>> calculate_ev(0.65, -150)  # Model says 65% but odds imply 60%
        8.3  # +8.3% EV
    """
    if not 0 < model_probability < 1:
        raise ValueError("Model probability must be between 0 and 1")
    
    decimal_odds = american_to_decimal(american_odds)
    
    # EV = (probability × payout) - 1
    # Where payout is decimal odds (total return per unit staked)
    ev = (model_probability * decimal_odds) - 1
    
    return round(ev * 100, 2)  # Return as percentage


def calculate_edge(
    model_probability: float,
    american_odds: int
) -> float:
    """
    Calculate the edge (advantage) over the bookmaker.
    
    Edge = Model Probability - Implied Probability
    
    A positive edge means your model thinks the true probability
    is higher than what the odds suggest.
    
    Args:
        model_probability: Your model's predicted probability
        american_odds: The betting odds offered
        
    Returns:
        Edge as percentage (e.g., 5.0 means 5% edge)
        
    Examples:
        >>> calculate_edge(0.55, -110)  # Model 55%, odds imply 52.4%
        2.6  # 2.6% edge
    """
    implied_prob = american_to_implied_probability(american_odds)
    edge = model_probability - implied_prob
    
    return round(edge * 100, 2)


def calculate_kelly_criterion(
    model_probability: float,
    american_odds: int,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate recommended bet size using Kelly Criterion.
    
    The Kelly Criterion provides the optimal bet size to maximize
    long-term growth while avoiding ruin.
    
    Full Kelly can be aggressive, so we use a fraction (default 25%).
    
    Formula:
    Kelly % = (Edge / (Decimal Odds - 1))
    
    Args:
        model_probability: Your model's predicted probability
        american_odds: The betting odds offered
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        
    Returns:
        Recommended bet size as percentage of bankroll
        
    Examples:
        >>> calculate_kelly_criterion(0.55, -110, kelly_fraction=0.25)
        1.2  # Bet 1.2% of bankroll
    """
    decimal_odds = american_to_decimal(american_odds)
    implied_prob = american_to_implied_probability(american_odds)
    
    edge = model_probability - implied_prob
    
    if edge <= 0:
        return 0.0  # No bet if no edge
    
    # Kelly formula: edge / (decimal_odds - 1)
    full_kelly = edge / (decimal_odds - 1)
    
    # Apply fraction for safety
    recommended = full_kelly * kelly_fraction
    
    return round(recommended * 100, 2)


def calculate_no_vig_probability(
    odds_team_a: int,
    odds_team_b: int
) -> Tuple[float, float]:
    """
    Calculate "fair" probabilities with the vig removed.
    
    Bookmakers add a vig (vigorish) that makes implied probabilities
    sum to more than 100%. This function removes the vig.
    
    Args:
        odds_team_a: American odds for team A
        odds_team_b: American odds for team B
        
    Returns:
        Tuple of (team_a_fair_prob, team_b_fair_prob) that sum to 1.0
        
    Examples:
        >>> calculate_no_vig_probability(-110, -110)
        (0.50, 0.50)  # Fair coin flip despite both being -110
    """
    implied_a = american_to_implied_probability(odds_team_a)
    implied_b = american_to_implied_probability(odds_team_b)
    
    # Total implied probability (includes vig)
    total = implied_a + implied_b
    
    # Normalize to remove vig
    fair_a = implied_a / total
    fair_b = implied_b / total
    
    return (round(fair_a, 4), round(fair_b, 4))


if __name__ == "__main__":
    # Demonstrate EV calculations
    print("=" * 60)
    print("EXPECTED VALUE CALCULATOR - EXAMPLES")
    print("=" * 60)
    
    # Example 1: Model finds edge on underdog
    print("\n📊 Example 1: Underdog with Edge")
    print("-" * 40)
    model_prob = 0.45
    odds = +150
    implied = american_to_implied_probability(odds)
    ev = calculate_ev(model_prob, odds)
    edge = calculate_edge(model_prob, odds)
    kelly = calculate_kelly_criterion(model_prob, odds)
    
    print(f"American Odds: {odds:+d}")
    print(f"Implied Probability: {implied:.1%}")
    print(f"Model Probability: {model_prob:.1%}")
    print(f"Edge: {edge:+.1f}%")
    print(f"Expected Value: {ev:+.1f}%")
    print(f"Recommended Bet (Quarter Kelly): {kelly:.1f}% of bankroll")
    
    # Example 2: Model finds edge on favorite
    print("\n📊 Example 2: Favorite with Edge")
    print("-" * 40)
    model_prob = 0.65
    odds = -150
    implied = american_to_implied_probability(odds)
    ev = calculate_ev(model_prob, odds)
    edge = calculate_edge(model_prob, odds)
    kelly = calculate_kelly_criterion(model_prob, odds)
    
    print(f"American Odds: {odds:+d}")
    print(f"Implied Probability: {implied:.1%}")
    print(f"Model Probability: {model_prob:.1%}")
    print(f"Edge: {edge:+.1f}%")
    print(f"Expected Value: {ev:+.1f}%")
    print(f"Recommended Bet (Quarter Kelly): {kelly:.1f}% of bankroll")
    
    # Example 3: No edge (negative EV)
    print("\n📊 Example 3: No Edge (Negative EV)")
    print("-" * 40)
    model_prob = 0.50
    odds = -110
    implied = american_to_implied_probability(odds)
    ev = calculate_ev(model_prob, odds)
    edge = calculate_edge(model_prob, odds)
    kelly = calculate_kelly_criterion(model_prob, odds)
    
    print(f"American Odds: {odds:+d}")
    print(f"Implied Probability: {implied:.1%}")
    print(f"Model Probability: {model_prob:.1%}")
    print(f"Edge: {edge:+.1f}%")
    print(f"Expected Value: {ev:+.1f}%")
    print(f"Recommended Bet: {kelly:.1f}% (NO BET)")
