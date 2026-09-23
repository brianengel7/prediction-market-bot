import numpy as np

from src.weather.distribution import (
    get_rounded_member_highs,
    calculate_smoothed_market_probability
)

from src.weather.fair_distribution import (
    probability_for_market
)


PROBABILITY_MODEL_VERSION = (
    "calibrated-multimodel-normal-v1"
)


def calculate_market_probability(
    market,
    gefs_results
):
    """
    Raw GEFS member probability.

    This is retained as a diagnostic.
    It is NOT the primary fair probability.
    """

    temperatures = (
        get_rounded_member_highs(
            gefs_results
        )
    )

    strike_type = (
        market[
            "strike_type"
        ].lower()
    )

    floor = market.get(
        "floor_strike"
    )

    cap = market.get(
        "cap_strike"
    )

    if strike_type == "less":

        if cap is None:

            raise ValueError(
                "Less market is "
                "missing cap_strike."
            )

        cap = float(
            cap
        )

        yes_mask = (
            temperatures
            < cap
        )

    elif strike_type == "greater":

        if floor is None:

            raise ValueError(
                "Greater market is "
                "missing floor_strike."
            )

        floor = float(
            floor
        )

        yes_mask = (
            temperatures
            > floor
        )

    elif strike_type == "between":

        if (
            floor is None
            or cap is None
        ):

            raise ValueError(
                "Between market is "
                "missing a strike."
            )

        floor = float(
            floor
        )

        cap = float(
            cap
        )

        yes_mask = (
            (temperatures >= floor)
            &
            (temperatures <= cap)
        )

    else:

        raise ValueError(
            f"Unknown strike type: "
            f"{strike_type}"
        )

    yes_count = int(
        np.sum(
            yes_mask
        )
    )

    total_members = len(
        temperatures
    )

    if total_members == 0:

        raise ValueError(
            "GEFS ensemble has "
            "no members."
        )

    probability = (
        yes_count
        / total_members
    )

    return {
        "probability":
            probability,

        "yes_count":
            yes_count,

        "total_members":
            total_members
    }


def calculate_market_edge(
    market,
    gefs_results,
    calibration
):
    """
    Calculate market edge using the
    calibrated multi-model probability.

    GEFS raw/KDE values are retained
    only as diagnostic information.
    """

    # --------------------------------------------------
    # RAW GEFS DIAGNOSTIC
    # --------------------------------------------------

    raw_result = (
        calculate_market_probability(
            market,
            gefs_results
        )
    )

    raw_gefs_yes = (
        raw_result[
            "probability"
        ]
    )

    # --------------------------------------------------
    # OLD GEFS KDE DIAGNOSTIC
    # --------------------------------------------------

    smooth_result = (
        calculate_smoothed_market_probability(
            market,
            gefs_results
        )
    )

    gefs_kde_yes = (
        smooth_result[
            "probability"
        ]
    )

    bandwidth = (
        smooth_result[
            "bandwidth"
        ]
    )

    # --------------------------------------------------
    # PRIMARY CALIBRATED FAIR PROBABILITY
    # --------------------------------------------------

    strike_type = (
        market[
            "strike_type"
        ].lower()
    )

    model_yes = (
        probability_for_market(
            strike_type=
                strike_type,

            floor_strike=
                market.get(
                    "floor_strike"
                ),

            cap_strike=
                market.get(
                    "cap_strike"
                ),

            point_forecast=
                calibration[
                    "point_forecast"
                ],

            residual_mean=
                calibration[
                    "residual_mean"
                ],

            residual_std=
                calibration[
                    "residual_std"
                ]
        )
    )

    model_no = (
        1.0
        - model_yes
    )

    # --------------------------------------------------
    # MARKET PRICES
    # --------------------------------------------------

    yes_bid = market.get(
        "yes_bid"
    )

    yes_ask = market.get(
        "yes_ask"
    )

    no_bid = market.get(
        "no_bid"
    )

    no_ask = market.get(
        "no_ask"
    )

    # --------------------------------------------------
    # EDGE
    # --------------------------------------------------

    yes_edge = None
    no_edge = None

    if yes_ask is not None:

        yes_edge = (
            model_yes
            - yes_ask
        )

    if no_ask is not None:

        no_edge = (
            model_no
            - no_ask
        )

    # --------------------------------------------------
    # BEST TRADE SIDE
    # --------------------------------------------------

    if (
        yes_edge is not None
        and no_edge is not None
    ):

        if yes_edge >= no_edge:

            best_side = "YES"
            best_edge = yes_edge

        else:

            best_side = "NO"
            best_edge = no_edge

    elif yes_edge is not None:

        best_side = "YES"
        best_edge = yes_edge

    elif no_edge is not None:

        best_side = "NO"
        best_edge = no_edge

    else:

        best_side = None
        best_edge = None

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    return {
        "ticker":
            market.get(
                "ticker"
            ),

        "title":
            market.get(
                "title"
            ),

        "strike_type":
            market.get(
                "strike_type"
            ),

        "floor_strike":
            market.get(
                "floor_strike"
            ),

        "cap_strike":
            market.get(
                "cap_strike"
            ),

        # ----------------------------------------------
        # GEFS diagnostics
        # ----------------------------------------------

        "raw_gefs_yes":
            raw_gefs_yes,

        "gefs_kde_yes":
            gefs_kde_yes,

        "bandwidth":
            bandwidth,

        "yes_count":
            raw_result[
                "yes_count"
            ],

        "total_members":
            raw_result[
                "total_members"
            ],

        # ----------------------------------------------
        # Primary calibrated model
        # ----------------------------------------------

        "model_yes":
            model_yes,

        "model_no":
            model_no,

        "fair_point_forecast":
            calibration[
                "point_forecast"
            ],

        "residual_mean":
            calibration[
                "residual_mean"
            ],

        "residual_std":
            calibration[
                "residual_std"
            ],

        # ----------------------------------------------
        # Kalshi prices
        # ----------------------------------------------

        "yes_bid":
            yes_bid,

        "yes_ask":
            yes_ask,

        "no_bid":
            no_bid,

        "no_ask":
            no_ask,

        "yes_midpoint":
            market.get(
                "yes_midpoint"
            ),

        "no_midpoint":
            market.get(
                "no_midpoint"
            ),

        "yes_spread":
            market.get(
                "yes_spread"
            ),

        "no_spread":
            market.get(
                "no_spread"
            ),

        # ----------------------------------------------
        # Edge
        # ----------------------------------------------

        "yes_edge":
            yes_edge,

        "no_edge":
            no_edge,

        "best_side":
            best_side,

        "best_edge":
            best_edge,

        "model_version":
            PROBABILITY_MODEL_VERSION
    }


def analyze_weather_markets(
    markets,
    gefs_results,
    calibration
):
    """
    Analyze all Kalshi weather markets
    using the calibrated multi-model
    fair probability.
    """

    results = []

    for market in markets:

        try:

            result = (
                calculate_market_edge(
                    market=
                        market,

                    gefs_results=
                        gefs_results,

                    calibration=
                        calibration
                )
            )

            results.append(
                result
            )

        except ValueError as error:

            print(
                f"Skipping "
                f"{market.get('ticker')}: "
                f"{error}"
            )

    # Highest calibrated edge first.
    results.sort(
        key=lambda result: (
            result[
                "best_edge"
            ]
            if result[
                "best_edge"
            ] is not None
            else float(
                "-inf"
            )
        ),
        reverse=True
    )

    return results