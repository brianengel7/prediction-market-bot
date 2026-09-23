import sqlite3

import numpy as np
import pandas as pd

from src.database.db import (
    DATABASE_PATH
)


STATION = "KNYC"
FORECAST_HORIZON = "D-1_12Z"
CALIBRATION_START_DATE = "2026-06-13"

MIN_TRAIN_DAYS = 30

SOURCES = [
    "GEFS",
    "GFS_MOS",
    "HRRR",
    "NBM"
]


def load_historical_data(
    before_date=None
):
    """
    Load historical model forecasts.

    If before_date is supplied, only use
    dates strictly before that target date.
    This prevents look-ahead bias.
    """

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
        """

        params = [
            STATION,
            FORECAST_HORIZON,
            CALIBRATION_START_DATE
        ]

        if before_date is not None:

            query += """
                AND target_date < ?
            """

            params.append(
                str(
                    pd.Timestamp(
                        before_date
                    ).date()
                )
            )

        query += """
            ORDER BY
                target_date,
                source
        """

        dataframe = pd.read_sql_query(
            query,
            connection,
            params=params
        )

    dataframe[
        "target_date"
    ] = pd.to_datetime(
        dataframe[
            "target_date"
        ]
    )

    return dataframe


def build_training_matrix(
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

    required_columns = (
        SOURCES
        + ["actual_high"]
    )

    data = data[
        required_columns
    ].dropna()

    return data.sort_index()


def calculate_calibration_parameters(
    training
):
    """
    Learn bias corrections and model
    weights strictly from historical data.
    """

    if len(training) < MIN_TRAIN_DAYS:

        raise ValueError(
            f"Need at least "
            f"{MIN_TRAIN_DAYS} complete "
            f"training dates. "
            f"Found {len(training)}."
        )

    biases = {}

    corrected_mae = {}

    for source in SOURCES:

        error = (
            training[source]
            - training[
                "actual_high"
            ]
        )

        bias = error.mean()

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

    inverse_errors = {
        source:
            1.0
            / max(
                corrected_mae[source],
                0.000001
            )

        for source in SOURCES
    }

    total = sum(
        inverse_errors.values()
    )

    weights = {
        source:
            inverse_errors[source]
            / total

        for source in SOURCES
    }

    return {
        "biases":
            biases,

        "weights":
            weights,

        "corrected_training_mae":
            corrected_mae,

        "training_days":
            len(training)
    }


def calibrate_forecasts(
    raw_forecasts,
    parameters
):
    corrected = {}

    for source in SOURCES:

        if source not in raw_forecasts:

            raise ValueError(
                f"Missing live forecast "
                f"for {source}."
            )

        corrected[
            source
        ] = (
            float(
                raw_forecasts[source]
            )
            - parameters[
                "biases"
            ][source]
        )

    return corrected


def calculate_weighted_forecast(
    corrected_forecasts,
    parameters
):
    return sum(
        corrected_forecasts[source]
        * parameters[
            "weights"
        ][source]

        for source in SOURCES
    )


def build_walk_forward_residuals(
    historical_data
):
    """
    Reconstruct genuinely out-of-sample
    residuals.

    Each day's prediction uses only
    dates before it.
    """

    results = []

    for index in range(
        MIN_TRAIN_DAYS,
        len(historical_data)
    ):

        training = (
            historical_data.iloc[
                :index
            ]
        )

        current = (
            historical_data.iloc[
                index
            ]
        )

        parameters = (
            calculate_calibration_parameters(
                training
            )
        )

        raw_forecasts = {
            source:
                current[source]

            for source in SOURCES
        }

        corrected = (
            calibrate_forecasts(
                raw_forecasts,
                parameters
            )
        )

        point_forecast = (
            calculate_weighted_forecast(
                corrected,
                parameters
            )
        )

        actual_high = current[
            "actual_high"
        ]

        residual = (
            actual_high
            - point_forecast
        )

        results.append(
            residual
        )

    return np.array(
        results,
        dtype=float
    )


def get_live_calibration(
    raw_forecasts,
    target_date=None
):
    """
    Produce everything needed by the
    fair probability distribution.
    """

    dataframe = load_historical_data(
        before_date=target_date
    )

    historical_data = (
        build_training_matrix(
            dataframe
        )
    )

    parameters = (
        calculate_calibration_parameters(
            historical_data
        )
    )

    corrected_forecasts = (
        calibrate_forecasts(
            raw_forecasts,
            parameters
        )
    )

    point_forecast = (
        calculate_weighted_forecast(
            corrected_forecasts,
            parameters
        )
    )

    residuals = (
        build_walk_forward_residuals(
            historical_data
        )
    )

    residual_mean = (
        residuals.mean()
    )

    residual_std = (
        residuals.std(
            ddof=1
        )
    )

    return {
        "raw_forecasts":
            raw_forecasts,

        "corrected_forecasts":
            corrected_forecasts,

        "biases":
            parameters[
                "biases"
            ],

        "weights":
            parameters[
                "weights"
            ],

        "point_forecast":
            point_forecast,

        "residual_mean":
            residual_mean,

        "residual_std":
            residual_std,

        "training_days":
            len(historical_data),

        "residual_samples":
            len(residuals)
    }

if __name__ == "__main__":

    # Temporary example only.
    raw_forecasts = {
        "GEFS": 87.0,
        "GFS_MOS": 84.0,
        "HRRR": 85.5,
        "NBM": 85.0
    }

    result = get_live_calibration(
        raw_forecasts=raw_forecasts,
        target_date="2026-09-24"
    )

    print()
    print(
        "LIVE CALIBRATED ENSEMBLE"
    )
    print("-" * 60)

    print()
    print("RAW")

    for source, value in (
        result[
            "raw_forecasts"
        ].items()
    ):

        print(
            f"{source:10s}: "
            f"{value:.2f}°F"
        )

    print()
    print("BIAS CORRECTIONS")

    for source, bias in (
        result[
            "biases"
        ].items()
    ):

        print(
            f"{source:10s}: "
            f"{bias:+.3f}°F"
        )

    print()
    print("CORRECTED")

    for source, value in (
        result[
            "corrected_forecasts"
        ].items()
    ):

        weight = result[
            "weights"
        ][source]

        print(
            f"{source:10s}: "
            f"{value:.2f}°F "
            f"weight={weight:.3f}"
        )

    print()

    print(
        f"Fair point forecast: "
        f"{result['point_forecast']:.2f}°F"
    )

    print(
        f"Residual mean:       "
        f"{result['residual_mean']:+.3f}°F"
    )

    print(
        f"Residual std:        "
        f"{result['residual_std']:.3f}°F"
    )

    print(
        f"Training days:       "
        f"{result['training_days']}"
    )

    print(
        f"OOS residuals:       "
        f"{result['residual_samples']}"
    )