import pytest

from src.sports.pricing.odds_math import (
    american_to_decimal,
    american_to_implied_probability,
    consensus_probability,
    decimal_to_implied_probability,
    remove_two_way_vig,
    sportsbook_hold,
)


def test_american_to_decimal_positive():

    assert (
        american_to_decimal(
            150
        )
        ==
        pytest.approx(
            2.5
        )
    )


def test_american_to_decimal_negative():

    assert (
        american_to_decimal(
            -200
        )
        ==
        pytest.approx(
            1.5
        )
    )


def test_decimal_to_implied_probability():

    assert (
        decimal_to_implied_probability(
            2.0
        )
        ==
        pytest.approx(
            0.5
        )
    )


def test_american_to_implied_probability_even():

    assert (
        american_to_implied_probability(
            100
        )
        ==
        pytest.approx(
            0.5
        )
    )


def test_american_to_implied_probability_negative():

    assert (
        american_to_implied_probability(
            -110
        )
        ==
        pytest.approx(
            110 / 210
        )
    )


def test_remove_two_way_vig_equal_market():

    probability_a, probability_b = (
        remove_two_way_vig(
            -110,
            -110,
        )
    )

    assert probability_a == pytest.approx(
        0.5
    )

    assert probability_b == pytest.approx(
        0.5
    )

    assert (
        probability_a
        +
        probability_b
        ==
        pytest.approx(
            1.0
        )
    )


def test_remove_two_way_vig_unequal_market():

    probability_a, probability_b = (
        remove_two_way_vig(
            -200,
            170,
        )
    )

    assert (
        probability_a
        +
        probability_b
        ==
        pytest.approx(
            1.0
        )
    )

    assert (
        probability_a
        >
        probability_b
    )


def test_sportsbook_hold():

    hold = (
        sportsbook_hold(
            -110,
            -110,
        )
    )

    assert hold == pytest.approx(
        0.0476190476
    )


def test_consensus_probability():

    probability = (
        consensus_probability(
            [
                0.60,
                0.62,
                0.61,
            ]
        )
    )

    assert probability == pytest.approx(
        0.61
    )


def test_consensus_probability_rejects_empty():

    with pytest.raises(
        ValueError
    ):

        consensus_probability(
            []
        )