import math

import numpy as np
import pandas as pd

from src.weather.calibrated_ensemble import (
    load_historical_data,
    build_training_matrix,
    calculate_calibration_parameters,
    calibrate_forecasts,
    calculate_weighted_forecast
)

from src.backtest.probability_calibration import (
    MIN_TRAIN_DAYS,
    MIN_RESIDUAL_DAYS,
    SOURCES,
    settlement_temperature,
    build_integer_probabilities,
    calculate_multiclass_brier,
    calculate_actual_temperature_nll,
    calculate_ranked_probability_score,
    calculate_interval
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODEL_NAMES = [
    "weighted_multimodel_v1",
    "equal_multimodel_v1",

    "gefs_corrected_v1",
    "gfs_mos_corrected_v1",
    "hrrr_corrected_v1",
    "nbm_corrected_v1",

    "gefs_raw_v1",
    "gfs_mos_raw_v1",
    "hrrr_raw_v1",
    "nbm_raw_v1"
]


# ============================================================
# MODEL POINT FORECASTS
# ============================================================

def calculate_model_forecasts(
    current,
    parameters
):
    """
    Build every candidate point forecast for one date.
    """

    raw_forecasts = {
        source:
            float(
                current[
                    source
                ]
            )
        for source in SOURCES
    }

    corrected_forecasts = (
        calibrate_forecasts(
            raw_forecasts,
            parameters
        )
    )

    weighted_forecast = (
        calculate_weighted_forecast(
            corrected_forecasts,
            parameters
        )
    )

    equal_forecast = float(
        np.mean(
            list(
                corrected_forecasts.values()
            )
        )
    )

    return {

        # ------------------------------------------
        # Multi-model candidates
        # ------------------------------------------

        "weighted_multimodel_v1":
            weighted_forecast,

        "equal_multimodel_v1":
            equal_forecast,

        # ------------------------------------------
        # Bias-corrected individual models
        # ------------------------------------------

        "gefs_corrected_v1":
            corrected_forecasts[
                "GEFS"
            ],

        "gfs_mos_corrected_v1":
            corrected_forecasts[
                "GFS_MOS"
            ],

        "hrrr_corrected_v1":
            corrected_forecasts[
                "HRRR"
            ],

        "nbm_corrected_v1":
            corrected_forecasts[
                "NBM"
            ],

        # ------------------------------------------
        # Raw model baselines
        # ------------------------------------------

        "gefs_raw_v1":
            raw_forecasts[
                "GEFS"
            ],

        "gfs_mos_raw_v1":
            raw_forecasts[
                "GFS_MOS"
            ],

        "hrrr_raw_v1":
            raw_forecasts[
                "HRRR"
            ],

        "nbm_raw_v1":
            raw_forecasts[
                "NBM"
            ]
    }


# ============================================================
# WALK-FORWARD BENCHMARK
# ============================================================

def run_model_benchmark():

    historical_data = (
        load_historical_data()
    )

    training = (
        build_training_matrix(
            historical_data
        )
    )

    if len(training) <= (
        MIN_TRAIN_DAYS
        + MIN_RESIDUAL_DAYS
    ):
        raise ValueError(
            "Not enough historical data "
            "for model benchmark."
        )

    # Each model gets its OWN historical residuals.
    #
    # This is important:
    # we cannot use the weighted model's uncertainty
    # distribution for every benchmark model.
    residual_histories = {
        model: []
        for model in MODEL_NAMES
    }

    rows = []

    for index in range(
        MIN_TRAIN_DAYS,
        len(training)
    ):

        prior_training = (
            training.iloc[
                :index
            ]
        )

        current = (
            training.iloc[
                index
            ]
        )

        target_date = (
            training.index[
                index
            ]
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

        # ------------------------------------------
        # Learn calibration ONLY from prior dates
        # ------------------------------------------

        parameters = (
            calculate_calibration_parameters(
                prior_training
            )
        )

        model_forecasts = (
            calculate_model_forecasts(
                current,
                parameters
            )
        )

        # ------------------------------------------
        # Score each candidate independently
        # ------------------------------------------

        for (
            model_name,
            point_forecast
        ) in model_forecasts.items():

            history = (
                residual_histories[
                    model_name
                ]
            )

            # Only score probabilities after
            # enough prior residuals exist.
            if (
                len(history)
                >= MIN_RESIDUAL_DAYS
            ):

                residual_mean = float(
                    np.mean(
                        history
                    )
                )

                residual_std = float(
                    np.std(
                        history,
                        ddof=1
                    )
                )

                if residual_std <= 0:
                    continue

                probabilities = (
                    build_integer_probabilities(
                        point_forecast=
                            point_forecast,

                        residual_mean=
                            residual_mean,

                        residual_std=
                            residual_std
                    )
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

                # ----------------------------------
                # Prediction intervals
                # ----------------------------------

                interval_50 = (
                    calculate_interval(
                        point_forecast,
                        residual_mean,
                        residual_std,
                        0.50
                    )
                )

                interval_80 = (
                    calculate_interval(
                        point_forecast,
                        residual_mean,
                        residual_std,
                        0.80
                    )
                )

                interval_90 = (
                    calculate_interval(
                        point_forecast,
                        residual_mean,
                        residual_std,
                        0.90
                    )
                )

                inside_50 = (
                    interval_50[0]
                    <= actual_high
                    <= interval_50[1]
                )

                inside_80 = (
                    interval_80[0]
                    <= actual_high
                    <= interval_80[1]
                )

                inside_90 = (
                    interval_90[0]
                    <= actual_high
                    <= interval_90[1]
                )

                # Forecast minus actual.
                forecast_error = (
                    float(
                        point_forecast
                    )
                    - actual_high
                )

                rows.append({

                    "target_date":
                        target_date,

                    "model":
                        model_name,

                    "actual_high":
                        actual_high,

                    "actual_temperature":
                        actual_temperature,

                    "point_forecast":
                        float(
                            point_forecast
                        ),

                    "forecast_error":
                        forecast_error,

                    "absolute_error":
                        abs(
                            forecast_error
                        ),

                    "squared_error":
                        forecast_error ** 2,

                    "residual_mean":
                        residual_mean,

                    "residual_std":
                        residual_std,

                    "distribution_center":
                        (
                            float(
                                point_forecast
                            )
                            + residual_mean
                        ),

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

                    "residual_count":
                        len(
                            history
                        )
                })

        # ------------------------------------------
        # AFTER scoring, add today's errors.
        #
        # Prevents look-ahead.
        # ------------------------------------------

        for (
            model_name,
            point_forecast
        ) in model_forecasts.items():

            residual = (
                actual_high
                - float(
                    point_forecast
                )
            )

            residual_histories[
                model_name
            ].append(
                residual
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

def calculate_model_summary(
    results
):

    if results.empty:
        return pd.DataFrame()

    summary_rows = []

    for (
        model_name,
        group
    ) in results.groupby(
        "model"
    ):

        mse = float(
            group[
                "squared_error"
            ].mean()
        )

        summary_rows.append({

            "model":
                model_name,

            "dates":
                group[
                    "target_date"
                ].nunique(),

            "bias":
                group[
                    "forecast_error"
                ].mean(),

            "mae":
                group[
                    "absolute_error"
                ].mean(),

            "rmse":
                math.sqrt(
                    mse
                ),

            "brier":
                group[
                    "brier"
                ].mean(),

            "nll":
                group[
                    "nll"
                ].mean(),

            "rps":
                group[
                    "rps"
                ].mean(),

            "actual_probability":
                group[
                    "actual_probability"
                ].mean(),

            "coverage_50":
                group[
                    "inside_50"
                ].mean(),

            "coverage_80":
                group[
                    "inside_80"
                ].mean(),

            "coverage_90":
                group[
                    "inside_90"
                ].mean()
        })

    summary = pd.DataFrame(
        summary_rows
    )

    # NLL is our primary probability-ranking metric
    # for now.
    summary = summary.sort_values(
        by="nll",
        ascending=True
    )

    return summary


# ============================================================
# PRINT RESULTS
# ============================================================

def print_model_summary(
    summary
):

    print()
    print("=" * 120)
    print(
        "WEATHER MODEL PROBABILITY BENCHMARK"
    )
    print("=" * 120)

    if summary.empty:

        print(
            "No benchmark results."
        )

        return

    display = (
        summary.copy()
    )

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

if __name__ == "__main__":

    results = (
        run_model_benchmark()
    )

    summary = (
        calculate_model_summary(
            results
        )
    )

    print_model_summary(
        summary
    )