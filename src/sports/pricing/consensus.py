from dataclasses import dataclass

from src.sports.pricing.odds_math import (
    remove_two_way_vig,
)


@dataclass
class SportsbookQuote:
    sportsbook: str
    team_a_odds: float
    team_b_odds: float


@dataclass
class ConsensusResult:
    team_a_probability: float
    team_b_probability: float
    sportsbook_count: int
    team_a_min_probability: float
    team_a_max_probability: float
    team_a_probability_range: float


def build_moneyline_consensus(
    quotes: list[SportsbookQuote],
) -> ConsensusResult:
    """
    Build an equal-weight no-vig moneyline consensus
    from multiple sportsbooks.

    Each sportsbook contributes one fair probability
    estimate after vig removal.
    """

    if not quotes:

        raise ValueError(
            "At least one sportsbook quote is required."
        )

    team_a_probabilities = []

    team_b_probabilities = []

    for quote in quotes:

        probability_a, probability_b = (
            remove_two_way_vig(
                quote.team_a_odds,
                quote.team_b_odds,
            )
        )

        team_a_probabilities.append(
            probability_a
        )

        team_b_probabilities.append(
            probability_b
        )

    team_a_consensus = (
        sum(
            team_a_probabilities
        )
        /
        len(
            team_a_probabilities
        )
    )

    team_b_consensus = (
        sum(
            team_b_probabilities
        )
        /
        len(
            team_b_probabilities
        )
    )

    team_a_min = min(
        team_a_probabilities
    )

    team_a_max = max(
        team_a_probabilities
    )

    return ConsensusResult(

        team_a_probability=
            team_a_consensus,

        team_b_probability=
            team_b_consensus,

        sportsbook_count=
            len(quotes),

        team_a_min_probability=
            team_a_min,

        team_a_max_probability=
            team_a_max,

        team_a_probability_range=
            team_a_max
            -
            team_a_min,
    )