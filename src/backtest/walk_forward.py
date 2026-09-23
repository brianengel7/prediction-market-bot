import sqlite3

import numpy as np
import pandas as pd

from src.database.db import (
    DATABASE_PATH
)


START_DATE = "2026-06-13"
END_DATE = "2026-09-10"

STATION = "KNYC"
FORECAST_HORIZON = "D-1_12Z"

MIN_TRAIN_DAYS = 30


def load_data():

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        query = """
            SELECT
                target_date,
                source,
                forecast_value,
                actual_high

            FROM weather_model_backtests

            WHERE station = ?
              AND forecast_horizon = ?
              AND target_date >= ?
              AND target_date <= ?

            ORDER BY
                target_date,
                source
        """

        dataframe = pd.read_sql_query(
            query,
            connection,
            params=(
                STATION,
                FORECAST_HORIZON,
                START_DATE,
                END_DATE
            )
        )

    dataframe[
        "target_date"
    ] = pd.to_datetime(
        dataframe["target_date"]
    )

    return dataframe


def build_matrix(
    dataframe
):

    forecasts = dataframe.pivot(
        index="target_date",
        columns="source",
        values="forecast_value"
    )

    actuals = (
        dataframe
        .drop_duplicates(
            "target_date"
        )
        .set_index(
            "target_date"
        )[
            "actual_high"
        ]
    )

    data = forecasts.copy()

    data[
        "actual_high"
    ] = actuals

    # Require all sources for the
    # cleanest apples-to-apples test.
    data = data.dropna()

    return data.sort_index()


def calculate_metrics(
    forecast,
    actual
):

    error = (
        forecast
        - actual
    )

    return {
        "bias":
            error.mean(),

        "mae":
            error.abs().mean(),

        "rmse":
            np.sqrt(
                np.mean(
                    error ** 2
                )
            ),

        "error_std":
            error.std(
                ddof=1
            )
    }


def calculate_training_parameters(
    training,
    sources
):

    biases = {}

    corrected_mae = {}

    for source in sources:

        raw_error = (
            training[source]
            - training[
                "actual_high"
            ]
        )

        bias = (
            raw_error.mean()
        )

        biases[
            source
        ] = bias

        corrected_error = (
            training[source]
            - bias
            - training[
                "actual_high"
            ]
        )

        corrected_mae[
            source
        ] = (
            corrected_error
            .abs()
            .mean()
        )

    inverse_errors = {}

    for source in sources:

        # Avoid divide-by-zero.
        mae = max(
            corrected_mae[source],
            0.000001
        )

        inverse_errors[
            source
        ] = (
            1.0 / mae
        )

    total = sum(
        inverse_errors.values()
    )

    weights = {
        source:
            inverse_errors[source]
            / total

        for source in sources
    }

    return (
        biases,
        weights
    )


def run_walk_forward():

    dataframe = load_data()

    data = build_matrix(
        dataframe
    )

    sources = [
        column
        for column in data.columns
        if column != "actual_high"
    ]

    results = []

    for index in range(
        MIN_TRAIN_DAYS,
        len(data)
    ):

        training = data.iloc[
            :index
        ]

        current = data.iloc[
            index
        ]

        target_date = (
            data.index[index]
        )

        biases, weights = (
            calculate_training_parameters(
                training,
                sources
            )
        )

        corrected_forecasts = {}

        for source in sources:

            corrected_forecasts[
                source
            ] = (
                current[source]
                - biases[source]
            )

        equal_forecast = np.mean(
            list(
                corrected_forecasts.values()
            )
        )

        weighted_forecast = sum(
            corrected_forecasts[source]
            * weights[source]

            for source in sources
        )

        actual_high = current[
            "actual_high"
        ]

        result = {
            "target_date":
                target_date,

            "actual_high":
                actual_high,

            "equal_forecast":
                equal_forecast,

            "weighted_forecast":
                weighted_forecast,

            # Residual convention:
            # positive means actual was
            # warmer than forecast.
            "equal_residual":
                actual_high
                - equal_forecast,

            "weighted_residual":
                actual_high
                - weighted_forecast
        }

        for source in sources:

            result[
                f"{source}_bias"
            ] = biases[source]

            result[
                f"{source}_weight"
            ] = weights[source]

            result[
                f"{source}_corrected"
            ] = corrected_forecasts[
                source
            ]

        results.append(
            result
        )

    results = pd.DataFrame(
        results
    )

    equal_metrics = (
        calculate_metrics(
            results[
                "equal_forecast"
            ],
            results[
                "actual_high"
            ]
        )
    )

    weighted_metrics = (
        calculate_metrics(
            results[
                "weighted_forecast"
            ],
            results[
                "actual_high"
            ]
        )
    )

    print()
    print("=" * 70)
    print(
        "WALK-FORWARD WEATHER CALIBRATION"
    )
    print("=" * 70)

    print(
        f"Historical dates: "
        f"{len(data)}"
    )

    print(
        f"Warm-up dates:    "
        f"{MIN_TRAIN_DAYS}"
    )

    print(
        f"OOS forecasts:    "
        f"{len(results)}"
    )

    print()
    print(
        "EQUAL-WEIGHT ENSEMBLE"
    )
    print("-" * 70)

    print(
        f"Bias:      "
        f"{equal_metrics['bias']:+.3f}°F"
    )

    print(
        f"MAE:       "
        f"{equal_metrics['mae']:.3f}°F"
    )

    print(
        f"RMSE:      "
        f"{equal_metrics['rmse']:.3f}°F"
    )

    print(
        f"Error std: "
        f"{equal_metrics['error_std']:.3f}°F"
    )

    print()
    print(
        "WEIGHTED ENSEMBLE"
    )
    print("-" * 70)

    print(
        f"Bias:      "
        f"{weighted_metrics['bias']:+.3f}°F"
    )

    print(
        f"MAE:       "
        f"{weighted_metrics['mae']:.3f}°F"
    )

    print(
        f"RMSE:      "
        f"{weighted_metrics['rmse']:.3f}°F"
    )

    print(
        f"Error std: "
        f"{weighted_metrics['error_std']:.3f}°F"
    )

    print()
    print(
        "WEIGHTED RESIDUAL DISTRIBUTION"
    )
    print("-" * 70)

    residuals = results[
        "weighted_residual"
    ]

    print(
        f"Mean: "
        f"{residuals.mean():+.3f}°F"
    )

    print(
        f"Std:  "
        f"{residuals.std(ddof=1):.3f}°F"
    )

    print(
        f"10%:  "
        f"{residuals.quantile(0.10):+.3f}°F"
    )

    print(
        f"25%:  "
        f"{residuals.quantile(0.25):+.3f}°F"
    )

    print(
        f"50%:  "
        f"{residuals.quantile(0.50):+.3f}°F"
    )

    print(
        f"75%:  "
        f"{residuals.quantile(0.75):+.3f}°F"
    )

    print(
        f"90%:  "
        f"{residuals.quantile(0.90):+.3f}°F"
    )

    print()
    print(
        "LATEST CALIBRATION PARAMETERS"
    )
    print("-" * 70)

    latest = results.iloc[-1]

    for source in sources:

        print(
            f"{source:10s} "
            f"bias="
            f"{latest[f'{source}_bias']:+.3f}°F "
            f"weight="
            f"{latest[f'{source}_weight']:.3f}"
        )

    return results


if __name__ == "__main__":

    run_walk_forward()