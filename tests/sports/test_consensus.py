import pytest

from src.sports.pricing.consensus import (
    SportsbookQuote,
    build_moneyline_consensus,
)


def test_consensus_single_book():

    result = (
        build_moneyline_consensus(
            [
                SportsbookQuote(
                    sportsbook=
                        "book_a",

                    team_a_odds=
                        -110,

                    team_b_odds=
                        -110,
                )
            ]
        )
    )

    assert (
        result.team_a_probability
        ==
        pytest.approx(
            0.5
        )
    )

    assert (
        result.team_b_probability
        ==
        pytest.approx(
            0.5
        )
    )

    assert (
        result.sportsbook_count
        ==
        1
    )


def test_consensus_multiple_books():

    result = (
        build_moneyline_consensus(
            [
                SportsbookQuote(
                    sportsbook=
                        "book_a",

                    team_a_odds=
                        -150,

                    team_b_odds=
                        130,
                ),

                SportsbookQuote(
                    sportsbook=
                        "book_b",

                    team_a_odds=
                        -160,

                    team_b_odds=
                        140,
                ),

                SportsbookQuote(
                    sportsbook=
                        "book_c",

                    team_a_odds=
                        -145,

                    team_b_odds=
                        125,
                ),
            ]
        )
    )

    assert (
        result.sportsbook_count
        ==
        3
    )

    assert (
        result.team_a_probability
        >
        0.5
    )

    assert (
        result.team_b_probability
        <
        0.5
    )

    assert (
        result.team_a_probability
        +
        result.team_b_probability
        ==
        pytest.approx(
            1.0
        )
    )


def test_consensus_probability_range():

    result = (
        build_moneyline_consensus(
            [
                SportsbookQuote(
                    sportsbook=
                        "book_a",

                    team_a_odds=
                        -200,

                    team_b_odds=
                        170,
                ),

                SportsbookQuote(
                    sportsbook=
                        "book_b",

                    team_a_odds=
                        -150,

                    team_b_odds=
                        130,
                ),
            ]
        )
    )

    assert (
        result.team_a_max_probability
        >=
        result.team_a_min_probability
    )

    assert (
        result.team_a_probability_range
        ==
        pytest.approx(
            result.team_a_max_probability
            -
            result.team_a_min_probability
        )
    )


def test_consensus_rejects_empty():

    with pytest.raises(
        ValueError
    ):

        build_moneyline_consensus(
            []
        )