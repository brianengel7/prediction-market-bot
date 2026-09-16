import pytest

from src.weather.distribution import (
    calculate_smoothed_market_probability
)


def make_gefs(highs):
    return {
        "members": [
            {
                "max_temperature": high
            }
            for high in highs
        ]
    }


def test_temperature_buckets_sum_to_one():

    gefs = make_gefs([
        79.0,
        80.0,
        81.0,
        82.0,
        83.0,
        84.0,
        85.0,
        86.0,
        87.0,
        88.0
    ])

    markets = [
        {
            "strike_type": "less",
            "floor_strike": None,
            "cap_strike": 80
        },
        {
            "strike_type": "between",
            "floor_strike": 80,
            "cap_strike": 81
        },
        {
            "strike_type": "between",
            "floor_strike": 82,
            "cap_strike": 83
        },
        {
            "strike_type": "between",
            "floor_strike": 84,
            "cap_strike": 85
        },
        {
            "strike_type": "between",
            "floor_strike": 86,
            "cap_strike": 87
        },
        {
            "strike_type": "greater",
            "floor_strike": 87,
            "cap_strike": None
        }
    ]

    probabilities = [
        calculate_smoothed_market_probability(
            market,
            gefs
        )["probability"]
        for market in markets
    ]

    assert sum(
        probabilities
    ) == pytest.approx(
        1.0,
        abs=1e-9
    )