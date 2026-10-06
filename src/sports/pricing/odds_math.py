from typing import Iterable


def american_to_decimal(
    american_odds: float,
) -> float:
    """
    Convert American odds to decimal odds.

    Examples:
        +150 -> 2.50
        -200 -> 1.50
    """

    american_odds = float(
        american_odds
    )

    if american_odds == 0:
        raise ValueError(
            "American odds cannot be zero."
        )

    if american_odds > 0:

        return (
            1.0
            +
            american_odds
            /
            100.0
        )

    return (
        1.0
        +
        100.0
        /
        abs(
            american_odds
        )
    )


def decimal_to_implied_probability(
    decimal_odds: float,
) -> float:
    """
    Convert decimal odds to implied probability.

    Example:
        2.00 -> 0.50
    """

    decimal_odds = float(
        decimal_odds
    )

    if decimal_odds <= 1.0:
        raise ValueError(
            "Decimal odds must be greater than 1.0."
        )

    return (
        1.0
        /
        decimal_odds
    )


def american_to_implied_probability(
    american_odds: float,
) -> float:
    """
    Convert American odds directly to implied probability.

    Examples:
        +100 -> 0.50
        -110 -> ~0.5238
        +150 -> 0.40
        -200 -> ~0.6667
    """

    decimal_odds = (
        american_to_decimal(
            american_odds
        )
    )

    return (
        decimal_to_implied_probability(
            decimal_odds
        )
    )


def remove_two_way_vig(
    odds_a: float,
    odds_b: float,
) -> tuple[float, float]:
    """
    Remove sportsbook vig from a two-way market.

    Uses proportional normalization.

    Example:
        Team A -110
        Team B -110

    Raw implied:
        52.38%
        52.38%

    Total:
        104.76%

    No-vig:
        50%
        50%
    """

    probability_a = (
        american_to_implied_probability(
            odds_a
        )
    )

    probability_b = (
        american_to_implied_probability(
            odds_b
        )
    )

    total_probability = (
        probability_a
        +
        probability_b
    )

    if total_probability <= 0:
        raise ValueError(
            "Total implied probability must be positive."
        )

    fair_a = (
        probability_a
        /
        total_probability
    )

    fair_b = (
        probability_b
        /
        total_probability
    )

    return (
        fair_a,
        fair_b,
    )


def sportsbook_hold(
    odds_a: float,
    odds_b: float,
) -> float:
    """
    Calculate sportsbook hold / overround
    for a two-way market.

    Returned as decimal probability.

    Example:
        -110 / -110
        -> ~0.0476
        -> 4.76%
    """

    probability_a = (
        american_to_implied_probability(
            odds_a
        )
    )

    probability_b = (
        american_to_implied_probability(
            odds_b
        )
    )

    return (
        probability_a
        +
        probability_b
        -
        1.0
    )


def consensus_probability(
    probabilities: Iterable[float],
) -> float:
    """
    Simple arithmetic mean of fair probabilities
    from multiple sportsbooks.

    Each input should already be de-vigged.

    Example:
        [0.60, 0.62, 0.61]
        -> 0.61
    """

    probabilities = [
        float(
            probability
        )
        for probability in probabilities
    ]

    if not probabilities:
        raise ValueError(
            "At least one probability is required."
        )

    for probability in probabilities:

        if not (
            0.0
            <=
            probability
            <=
            1.0
        ):

            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

    return (
        sum(
            probabilities
        )
        /
        len(
            probabilities
        )
    )