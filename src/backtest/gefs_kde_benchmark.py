import io
import math
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from src.backtest.gefs import (
    backtest_gefs_date
)

from src.backtest.probability_calibration import (
    MIN_TRAIN_DAYS,
    MIN_RESIDUAL_DAYS,
    settlement_temperature,
    calculate_multiclass_brier,
    calculate_actual_temperature_nll,
    calculate_ranked_probability_score
)

from src.backtest.model_benchmark import (
    run_model_benchmark,
    calculate_model_summary
)

from src.weather.calibrated_ensemble import (
    load_historical_data,
    build_training_matrix
)

from src.weather.distribution import (
    get_rounded_member_highs,
    calculate_smoothed_market_probability
)


MODEL_NAME = "gefs_kde_original_v1"

MIN_TEMPERATURE = 0
MAX_TEMPERATURE = 130


# ============================================================
# BUILD EXACT ORIGINAL KDE DISTRIBUTION
# ============================================================

def build_gefs_kde_distribution(
    gefs_results
):
    """
    Reconstruct the original GEFS KDE probability
    distribution using the exact production function
    already used by the bot.
    """

    probabilities = {}

    bandwidth = None

    for temperature in range(
        MIN_TEMPERATURE,
        MAX_TEMPERATURE + 1
    ):

        # Treat each integer temperature as a
        # one-degree Kalshi-style "between" market.
        market = {
            "strike_type": "between",
            "floor_strike":
                float(
                    temperature
                ),
            "cap_strike":
                float(
                    temperature
                )
        }

        result = (
            calculate_smoothed_market_probability(
                market,
                gefs_results
            )
        )

        probabilities[
            temperature
        ] = float(
            result[
                "probability"
            ]
        )

        if bandwidth is None:

            bandwidth = (
                result.get(
                    "bandwidth"
                )
            )

    # --------------------------------------------------------
    # Normalize tiny numerical tail loss.
    # --------------------------------------------------------

    total_probability = sum(
        probabilities.values()
    )

    if total_probability <= 0:

        raise ValueError(
            "GEFS KDE produced no "
            "probability mass."
        )

    probabilities = {
        temperature:
            probability
            / total_probability

        for (
            temperature,
            probability
        )
        in probabilities.items()
    }

    return {
        "probabilities":
            probabilities,

        "bandwidth":
            bandwidth
    }


# ============================================================
# DISCRETE PREDICTION INTERVAL
# ============================================================

def get_discrete_interval(
    probabilities,
    coverage
):
    """
    Central probability interval from a discrete
    integer-temperature distribution.
    """

    temperatures = sorted(
        probabilities.keys()
    )

    alpha = (
        1.0
        - coverage
    )

    lower_target = (
        alpha / 2.0
    )

    upper_target = (
        1.0
        - alpha / 2.0
    )

    cumulative = 0.0

    lower = temperatures[0]
    upper = temperatures[-1]

    lower_found = False

    for temperature in temperatures:

        cumulative += (
            probabilities[
                temperature
            ]
        )

        if (
            not lower_found
            and cumulative
            >= lower_target
        ):

            lower = temperature
            lower_found = True

        if cumulative >= upper_target:

            upper = temperature
            break

    return (
        lower,
        upper
    )


# ============================================================
# RUN EXACT KDE BACKTEST
# ============================================================

def run_gefs_kde_benchmark():

    historical_data = (
        load_historical_data()
    )

    training = (
        build_training_matrix(
            historical_data
        )
    )

    first_scored_index = (
        MIN_TRAIN_DAYS
        + MIN_RESIDUAL_DAYS
    )

    if len(training) <= first_scored_index:

        raise ValueError(
            "Not enough historical data."
        )

    scored_rows = []

    scored_dates = training.iloc[
        first_scored_index:
    ]

    total_dates = len(
        scored_dates
    )

    print()
    print("=" * 70)
    print(
        "EXACT ORIGINAL GEFS KDE BENCHMARK"
    )
    print("=" * 70)

    print(
        f"Dates to score: "
        f"{total_dates}"
    )

    # --------------------------------------------------------
    # SAME 45 DATES USED BY CURRENT PROBABILITY BENCHMARK
    # --------------------------------------------------------

    for number, (
        target_date,
        current
    ) in enumerate(
        scored_dates.iterrows(),
        start=1
    ):

        target_date_string = (
            pd.Timestamp(
                target_date
            )
            .date()
            .isoformat()
        )

        actual_high = float(
            current[
                "actual_high"
            ]
        )

        actual_temperature = (
            settlement_temperature(
                actual_high
            )
        )

        print(
            f"[{number:02d}/{total_dates}] "
            f"{target_date_string}",
            end=""
        )

        try:

            # ------------------------------------------------
            # The GEFS backtest function is very noisy.
            #
            # Suppress its member-by-member output here.
            # Cache hits still occur normally.
            # ------------------------------------------------

            with redirect_stdout(
                io.StringIO()
            ):

                gefs_results = (
                    backtest_gefs_date(
                        target_date=
                            target_date_string,

                        actual_high=None
                    )
                )

            # ------------------------------------------------
            # EXACT ORIGINAL KDE PROBABILITY MODEL
            # ------------------------------------------------

            kde_result = (
                build_gefs_kde_distribution(
                    gefs_results
                )
            )

            probabilities = (
                kde_result[
                    "probabilities"
                ]
            )

            bandwidth = (
                kde_result[
                    "bandwidth"
                ]
            )

            actual_probability = (
                probabilities.get(
                    actual_temperature,
                    0.0
                )
            )

            brier = (
                calculate_multiclass_brier(
                    probabilities,
                    actual_temperature
                )
            )

            nll = (
                calculate_actual_temperature_nll(
                    probabilities,
                    actual_temperature
                )
            )

            rps = (
                calculate_ranked_probability_score(
                    probabilities,
                    actual_temperature
                )
            )

            # ------------------------------------------------
            # POINT FORECAST METRICS
            #
            # These aren't really the purpose of the KDE
            # benchmark, but useful for comparison.
            # ------------------------------------------------

            member_highs = (
                get_rounded_member_highs(
                    gefs_results
                )
            )

            point_forecast = float(
                np.mean(
                    member_highs
                )
            )

            forecast_error = (
                point_forecast
                - actual_high
            )

            # ------------------------------------------------
            # DISCRETE COVERAGE
            # ------------------------------------------------

            interval_50 = (
                get_discrete_interval(
                    probabilities,
                    0.50
                )
            )

            interval_80 = (
                get_discrete_interval(
                    probabilities,
                    0.80
                )
            )

            interval_90 = (
                get_discrete_interval(
                    probabilities,
                    0.90
                )
            )

            inside_50 = (
                interval_50[0]
                <= actual_temperature
                <= interval_50[1]
            )

            inside_80 = (
                interval_80[0]
                <= actual_temperature
                <= interval_80[1]
            )

            inside_90 = (
                interval_90[0]
                <= actual_temperature
                <= interval_90[1]
            )

            scored_rows.append({
                "target_date":
                    target_date_string,

                "model":
                    MODEL_NAME,

                "actual_high":
                    actual_high,

                "actual_temperature":
                    actual_temperature,

                "point_forecast":
                    point_forecast,

                "forecast_error":
                    forecast_error,

                "absolute_error":
                    abs(
                        forecast_error
                    ),

                "squared_error":
                    forecast_error ** 2,

                "actual_probability":
                    actual_probability,

                "brier":
                    brier,

                "nll":
                    nll,

                "rps":
                    rps,

                "inside_50":
                    inside_50,

                "inside_80":
                    inside_80,

                "inside_90":
                    inside_90,

                "bandwidth":
                    bandwidth
            })

            print(
                f" | "
                f"P(actual)="
                f"{actual_probability:.2%} "
                f"NLL={nll:.3f}"
            )

        except Exception as error:

            print(
                f" | FAILED: "
                f"{error}"
            )

    return pd.DataFrame(
        scored_rows
    )


# ============================================================
# KDE SUMMARY
# ============================================================

def calculate_kde_summary(
    results
):

    if results.empty:

        return None

    mse = float(
        results[
            "squared_error"
        ].mean()
    )

    return {
        "model":
            MODEL_NAME,

        "dates":
            results[
                "target_date"
            ].nunique(),

        "bias":
            results[
                "forecast_error"
            ].mean(),

        "mae":
            results[
                "absolute_error"
            ].mean(),

        "rmse":
            math.sqrt(
                mse
            ),

        "brier":
            results[
                "brier"
            ].mean(),

        "nll":
            results[
                "nll"
            ].mean(),

        "rps":
            results[
                "rps"
            ].mean(),

        "actual_probability":
            results[
                "actual_probability"
            ].mean(),

        "coverage_50":
            results[
                "inside_50"
            ].mean(),

        "coverage_80":
            results[
                "inside_80"
            ].mean(),

        "coverage_90":
            results[
                "inside_90"
            ].mean(),

        "average_bandwidth":
            results[
                "bandwidth"
            ].mean()
    }


# ============================================================
# FINAL COMPARISON
# ============================================================

def print_final_comparison(
    kde_summary
):

    benchmark_results = (
        run_model_benchmark()
    )

    current_summary = (
        calculate_model_summary(
            benchmark_results
        )
    )

    models_to_compare = [
        "weighted_multimodel_v1",
        "equal_multimodel_v1",
        "gefs_corrected_v1",
        "gefs_raw_v1"
    ]

    current_summary = (
        current_summary[
            current_summary[
                "model"
            ].isin(
                models_to_compare
            )
        ]
        .copy()
    )

    kde_row = {
        "model":
            kde_summary[
                "model"
            ],

        "dates":
            kde_summary[
                "dates"
            ],

        "bias":
            kde_summary[
                "bias"
            ],

        "mae":
            kde_summary[
                "mae"
            ],

        "rmse":
            kde_summary[
                "rmse"
            ],

        "brier":
            kde_summary[
                "brier"
            ],

        "nll":
            kde_summary[
                "nll"
            ],

        "rps":
            kde_summary[
                "rps"
            ],

        "actual_probability":
            kde_summary[
                "actual_probability"
            ],

        "coverage_50":
            kde_summary[
                "coverage_50"
            ],

        "coverage_80":
            kde_summary[
                "coverage_80"
            ],

        "coverage_90":
            kde_summary[
                "coverage_90"
            ]
    }

    comparison = pd.concat(
        [
            current_summary,

            pd.DataFrame(
                [
                    kde_row
                ]
            )
        ],
        ignore_index=True
    )

    comparison = (
        comparison.sort_values(
            by="nll",
            ascending=True
        )
    )

    print()
    print("=" * 120)
    print(
        "FINAL WEATHER MODEL COMPARISON"
    )
    print("=" * 120)

    display = comparison.copy()

    display[
        "actual_probability"
    ] *= 100

    display[
        "coverage_50"
    ] *= 100

    display[
        "coverage_80"
    ] *= 100

    display[
        "coverage_90"
    ] *= 100

    print(
        display.to_string(
            index=False,

            formatters={
                "bias":
                    lambda x:
                    f"{x:+.3f}",

                "mae":
                    lambda x:
                    f"{x:.3f}",

                "rmse":
                    lambda x:
                    f"{x:.3f}",

                "brier":
                    lambda x:
                    f"{x:.4f}",

                "nll":
                    lambda x:
                    f"{x:.4f}",

                "rps":
                    lambda x:
                    f"{x:.4f}",

                "actual_probability":
                    lambda x:
                    f"{x:.2f}%",

                "coverage_50":
                    lambda x:
                    f"{x:.1f}%",

                "coverage_80":
                    lambda x:
                    f"{x:.1f}%",

                "coverage_90":
                    lambda x:
                    f"{x:.1f}%"
            }
        )
    )

    print()

    print(
        f"Original GEFS KDE "
        f"average bandwidth: "
        f"{kde_summary['average_bandwidth']:.3f}°F"
    )

if __name__ == "__main__":

    kde_results = (
        run_gefs_kde_benchmark()
    )

    kde_summary = (
        calculate_kde_summary(
            kde_results
        )
    )

    if kde_summary is None:

        print(
            "No successful GEFS KDE "
            "results."
        )

    else:

        print_final_comparison(
            kde_summary
        )