import sqlite3

import numpy as np
import pandas as pd

from src.database.db import (
    DATABASE_PATH
)


START_DATE = "2026-06-13"
SPLIT_DATE = "2026-08-12"
END_DATE = "2026-09-10"

FORECAST_HORIZON = "D-1_12Z"
STATION = "KNYC"


def load_calibration_data():

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
        dataframe[
            "target_date"
        ]
    )

    return dataframe


def build_forecast_matrix(
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

    # Require every source to exist
    # for an apples-to-apples comparison.
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
            )
    }


def run_calibration():

    dataframe = (
        load_calibration_data()
    )

    data = (
        build_forecast_matrix(
            dataframe
        )
    )

    training = data[
        data.index
        < pd.Timestamp(
            SPLIT_DATE
        )
    ].copy()

    testing = data[
        data.index
        >= pd.Timestamp(
            SPLIT_DATE
        )
    ].copy()

    sources = [
        column
        for column in data.columns
        if column != "actual_high"
    ]

    print()
    print("=" * 70)
    print(
        "WEATHER MODEL CALIBRATION"
    )
    print("=" * 70)

    print(
        f"Training dates: "
        f"{len(training)}"
    )

    print(
        f"Testing dates:  "
        f"{len(testing)}"
    )

    print()
    print(
        "TRAINING MODEL BIASES"
    )
    print("-" * 70)

    biases = {}

    corrected_training_mae = {}

    for source in sources:

        training_error = (
            training[source]
            - training["actual_high"]
        )

        bias = (
            training_error.mean()
        )

        biases[
            source
        ] = bias

        corrected_error = (
            training_error
            - bias
        )

        corrected_training_mae[
            source
        ] = (
            corrected_error
            .abs()
            .mean()
        )

        print(
            f"{source:10s}: "
            f"{bias:+.3f}°F"
        )

    print()
    print(
        "OUT-OF-SAMPLE MODEL RESULTS"
    )
    print("-" * 70)

    corrected_test_forecasts = (
        pd.DataFrame(
            index=testing.index
        )
    )

    for source in sources:

        raw_forecast = (
            testing[source]
        )

        corrected_forecast = (
            raw_forecast
            - biases[source]
        )

        corrected_test_forecasts[
            source
        ] = corrected_forecast

        raw_metrics = (
            calculate_metrics(
                raw_forecast,
                testing[
                    "actual_high"
                ]
            )
        )

        corrected_metrics = (
            calculate_metrics(
                corrected_forecast,
                testing[
                    "actual_high"
                ]
            )
        )

        print()
        print(source)

        print(
            f"  Raw bias:       "
            f"{raw_metrics['bias']:+.3f}°F"
        )

        print(
            f"  Raw MAE:        "
            f"{raw_metrics['mae']:.3f}°F"
        )

        print(
            f"  Raw RMSE:       "
            f"{raw_metrics['rmse']:.3f}°F"
        )

        print(
            f"  Corrected bias: "
            f"{corrected_metrics['bias']:+.3f}°F"
        )

        print(
            f"  Corrected MAE:  "
            f"{corrected_metrics['mae']:.3f}°F"
        )

        print(
            f"  Corrected RMSE: "
            f"{corrected_metrics['rmse']:.3f}°F"
        )

    # Equal-weight corrected ensemble
    equal_forecast = (
        corrected_test_forecasts
        .mean(
            axis=1
        )
    )

    equal_metrics = (
        calculate_metrics(
            equal_forecast,
            testing[
                "actual_high"
            ]
        )
    )

    print()
    print("=" * 70)

    print(
        "EQUAL-WEIGHT "
        "BIAS-CORRECTED ENSEMBLE"
    )

    print("-" * 70)

    print(
        f"Bias: "
        f"{equal_metrics['bias']:+.3f}°F"
    )

    print(
        f"MAE:  "
        f"{equal_metrics['mae']:.3f}°F"
    )

    print(
        f"RMSE: "
        f"{equal_metrics['rmse']:.3f}°F"
    )

    # Weight models using inverse
    # corrected training MAE.
    inverse_errors = {
        source:
            1.0
            / corrected_training_mae[
                source
            ]

        for source in sources
    }

    weight_total = sum(
        inverse_errors.values()
    )

    weights = {
        source:
            inverse_errors[source]
            / weight_total

        for source in sources
    }

    print()
    print(
        "TRAINING-DERIVED WEIGHTS"
    )

    print("-" * 70)

    for source in sources:

        print(
            f"{source:10s}: "
            f"{weights[source]:.3f}"
        )

    weighted_forecast = sum(
        corrected_test_forecasts[
            source
        ] * weights[source]

        for source in sources
    )

    weighted_metrics = (
        calculate_metrics(
            weighted_forecast,
            testing[
                "actual_high"
            ]
        )
    )

    print()
    print(
        "WEIGHTED "
        "BIAS-CORRECTED ENSEMBLE"
    )

    print("-" * 70)

    print(
        f"Bias: "
        f"{weighted_metrics['bias']:+.3f}°F"
    )

    print(
        f"MAE:  "
        f"{weighted_metrics['mae']:.3f}°F"
    )

    print(
        f"RMSE: "
        f"{weighted_metrics['rmse']:.3f}°F"
    )


if __name__ == "__main__":

    run_calibration()