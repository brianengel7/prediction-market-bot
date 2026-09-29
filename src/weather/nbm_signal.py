import pandas as pd

from src.weather.calibrated_ensemble import (
    load_historical_data
)

from src.weather.fair_distribution import (
    probability_for_market
)


MIN_HISTORY_DAYS = 15


def get_nbm_error_std(
    target_date
):
    """
    Calculate NBM historical error standard deviation
    using ONLY dates before the target date.

    This mirrors the v2 source backtest.
    """

    history = (
        load_historical_data(
            before_date=
                target_date
        )
    )

    nbm_history = history[
        history[
            "source"
        ]
        ==
        "NBM"
    ].copy()

    if (
        len(nbm_history)
        <
        MIN_HISTORY_DAYS
    ):
        raise RuntimeError(
            f"Need at least "
            f"{MIN_HISTORY_DAYS} prior NBM dates. "
            f"Found {len(nbm_history)}."
        )

    nbm_history[
        "error"
    ] = (
        nbm_history[
            "forecast_value"
        ]
        -
        nbm_history[
            "actual_high"
        ]
    )

    residual_std = float(
        nbm_history[
            "error"
        ]
        .std(
            ddof=1
        )
    )

    if (
        pd.isna(
            residual_std
        )
        or
        residual_std <= 0
    ):
        raise RuntimeError(
            "Invalid NBM residual standard deviation."
        )

    return residual_std


def build_live_nbm_probabilities(
    markets,
    bundle,
    target_date
):
    """
    Build the NBM-only probability distribution used
    by Weather Model v2.

    Uses:
        - bias-corrected NBM forecast
        - NBM historical raw-error std
        - same probability_for_market() function
          used in the historical v2 test
    """

    calibration = bundle[
        "calibration"
    ]

    corrected_nbm = float(
        calibration[
            "corrected_forecasts"
        ][
            "NBM"
        ]
    )

    residual_std = (
        get_nbm_error_std(
            target_date
        )
    )

    probabilities = []

    for market in markets:

        probability = (
            probability_for_market(
                strike_type=
                    market[
                        "strike_type"
                    ],

                floor_strike=
                    market.get(
                        "floor_strike"
                    ),

                cap_strike=
                    market.get(
                        "cap_strike"
                    ),

                point_forecast=
                    corrected_nbm,

                residual_mean=
                    0.0,

                residual_std=
                    residual_std
            )
        )

        probabilities.append({

            "ticker":
                market[
                    "ticker"
                ],

            "strike_type":
                market[
                    "strike_type"
                ],

            "floor_strike":
                market.get(
                    "floor_strike"
                ),

            "cap_strike":
                market.get(
                    "cap_strike"
                ),

            "nbm_probability":
                float(
                    probability
                ),

            # This is the corrected forecast actually
            # used to generate the probability.
            "nbm_forecast":
                corrected_nbm,

            "residual_std":
                residual_std
        })

    probability_mass = sum(
        row[
            "nbm_probability"
        ]
        for row in probabilities
    )

    print()
    print(
        "NBM V2 SIGNAL"
    )

    print("-" * 60)

    print(
        f"Corrected NBM: "
        f"{corrected_nbm:.2f}°F"
    )

    print(
        f"NBM error std: "
        f"{residual_std:.2f}°F"
    )

    print(
        f"Probability mass: "
        f"{probability_mass:.6f}"
    )

    return probabilities