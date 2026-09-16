import numpy as np

from src.weather.distribution import (
    get_rounded_member_highs,
    calculate_smoothed_market_probability
)


def calculate_market_probability(
    market,
    gefs_results
):

    temperatures = get_rounded_member_highs(
        gefs_results
    )

    strike_type = market["strike_type"].lower()

    floor = market.get("floor_strike")
    cap = market.get("cap_strike")

    if strike_type == "less":

        if cap is None:
            raise ValueError(
                "Less market is missing cap_strike."
            )

        cap = float(cap)

        yes_mask = temperatures < cap

    elif strike_type == "greater":

        if floor is None:
            raise ValueError(
                "Greater market is missing floor_strike."
            )

        floor = float(floor)

        yes_mask = temperatures > floor

    elif strike_type == "between":

        if floor is None or cap is None:
            raise ValueError(
                "Between market is missing a strike."
            )

        floor = float(floor)
        cap = float(cap)

        yes_mask = (
            (temperatures >= floor)
            &
            (temperatures <= cap)
        )

    else:
        raise ValueError(
            f"Unknown strike type: {strike_type}"
        )

    yes_count = int(
        np.sum(yes_mask)
    )

    total_members = len(
        temperatures
    )

    probability = (
        yes_count / total_members
    )

    return {
        "probability": probability,
        "yes_count": yes_count,
        "total_members": total_members
    }


def calculate_market_edge(
    market,
    gefs_results
):

    raw_result = calculate_market_probability(
        market,
        gefs_results
    )

    raw_yes = raw_result[
        "probability"
    ]

    smooth_result = (
        calculate_smoothed_market_probability(
            market,
            gefs_results
        )
    )

    model_yes = smooth_result[
        "probability"
    ]

    model_no = 1.0 - model_yes

    yes_ask = market[
        "yes_ask"
    ]

    no_ask = market[
        "no_ask"
    ]

    yes_edge = (
        model_yes - yes_ask
    )

    no_edge = (
        model_no - no_ask
    )

    if yes_edge >= no_edge:

        best_side = "YES"
        best_edge = yes_edge

    else:

        best_side = "NO"
        best_edge = no_edge

    return {
        "ticker": market[
            "ticker"
        ],

        "title": market[
            "title"
        ],

        "strike_type": market[
            "strike_type"
        ],

        "floor_strike": market[
            "floor_strike"
        ],

        "cap_strike": market[
            "cap_strike"
        ],

        # Raw 31-member GEFS probability
        "raw_gefs_yes": raw_yes,

        # Smoothed probability used for edge
        "model_yes": model_yes,
        "model_no": model_no,

        # KDE smoothing amount
        "bandwidth": smooth_result[
            "bandwidth"
        ],

        # Raw ensemble information
        "yes_count": raw_result[
            "yes_count"
        ],

        "total_members": raw_result[
            "total_members"
        ],

        # Kalshi market prices
        "yes_bid": market[
            "yes_bid"
        ],

        "yes_ask": market[
            "yes_ask"
        ],

        "no_bid": market[
            "no_bid"
        ],

        "no_ask": market[
            "no_ask"
        ],

        "yes_midpoint": market[
            "yes_midpoint"
        ],

        "no_midpoint": market[
            "no_midpoint"
        ],

        "yes_spread": market[
            "yes_spread"
        ],

        "no_spread": market[
            "no_spread"
        ],

        # Model edge
        "yes_edge": yes_edge,
        "no_edge": no_edge,

        "best_side": best_side,
        "best_edge": best_edge
    }


def analyze_weather_markets(
    markets,
    gefs_results
):

    results = []

    for market in markets:

        try:

            result = calculate_market_edge(
                market,
                gefs_results
            )

            results.append(result)

        except ValueError as error:

            print(
                f"Skipping "
                f"{market['ticker']}: "
                f"{error}"
            )

    # Highest raw model edge first.
    results.sort(
        key=lambda result: result[
            "best_edge"
        ],
        reverse=True
    )

    return results