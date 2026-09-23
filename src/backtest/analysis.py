import sqlite3

import numpy as np
import pandas as pd

from src.database.db import (
    DATABASE_PATH
)


def load_backtest_data(
    forecast_horizon="D-1_12Z",
    station="KNYC"
):
    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        query = """
            SELECT
                target_date,
                station,
                source,
                product,
                model_run,
                forecast_horizon,
                forecast_value,
                forecast_median,
                forecast_std,
                member_count,
                actual_high,
                error,
                absolute_error,
                squared_error
            FROM weather_model_backtests
            WHERE forecast_horizon = ?
              AND station = ?
            ORDER BY target_date, source
        """

        dataframe = pd.read_sql_query(
            query,
            connection,
            params=(
                forecast_horizon,
                station
            )
        )

    return dataframe


def calculate_source_metrics(
    dataframe
):
    results = []

    grouped = dataframe.groupby(
        [
            "source",
            "product"
        ]
    )

    for (
        source,
        product
    ), group in grouped:

        errors = group[
            "error"
        ].astype(float)

        absolute_errors = group[
            "absolute_error"
        ].astype(float)

        squared_errors = group[
            "squared_error"
        ].astype(float)

        bias = errors.mean()

        mae = absolute_errors.mean()

        rmse = np.sqrt(
            squared_errors.mean()
        )

        error_std = (
            errors.std(
                ddof=1
            )
            if len(errors) > 1
            else 0.0
        )

        results.append(
            {
                "source":
                    source,

                "product":
                    product,

                "samples":
                    len(group),

                "bias":
                    bias,

                "mae":
                    mae,

                "rmse":
                    rmse,

                "error_std":
                    error_std,

                "min_error":
                    errors.min(),

                "max_error":
                    errors.max()
            }
        )

    return pd.DataFrame(
        results
    )


def calculate_pairwise_correlation(
    dataframe
):
    pivot = dataframe.pivot_table(
        index="target_date",
        columns="source",
        values="forecast_value",
        aggfunc="first"
    )

    if pivot.empty:
        return None

    return pivot.corr()


def calculate_pairwise_disagreement(
    dataframe
):
    pivot = dataframe.pivot_table(
        index="target_date",
        columns="source",
        values="forecast_value",
        aggfunc="first"
    )

    sources = list(
        pivot.columns
    )

    results = []

    for i in range(
        len(sources)
    ):
        for j in range(
            i + 1,
            len(sources)
        ):

            source_a = sources[i]
            source_b = sources[j]

            paired = pivot[
                [
                    source_a,
                    source_b
                ]
            ].dropna()

            if paired.empty:
                continue

            difference = (
                paired[source_a]
                - paired[source_b]
            )

            results.append(
                {
                    "source_a":
                        source_a,

                    "source_b":
                        source_b,

                    "samples":
                        len(paired),

                    "mean_difference":
                        difference.mean(),

                    "mean_absolute_difference":
                        difference.abs().mean(),

                    "max_absolute_difference":
                        difference.abs().max()
                }
            )

    return pd.DataFrame(
        results
    )


def calculate_bias_corrected_metrics(
    dataframe
):
    results = []

    for source, group in dataframe.groupby(
        "source"
    ):

        forecast = group[
            "forecast_value"
        ].astype(float)

        actual = group[
            "actual_high"
        ].astype(float)

        errors = (
            forecast
            - actual
        )

        bias = errors.mean()

        corrected_forecast = (
            forecast
            - bias
        )

        corrected_errors = (
            corrected_forecast
            - actual
        )

        results.append(
            {
                "source":
                    source,

                "bias_removed":
                    bias,

                "corrected_mae":
                    corrected_errors
                    .abs()
                    .mean(),

                "corrected_rmse":
                    np.sqrt(
                        np.mean(
                            corrected_errors
                            ** 2
                        )
                    )
            }
        )

    return pd.DataFrame(
        results
    )


def calculate_ensemble_spread_skill(
    dataframe
):
    ensemble_data = dataframe[
        dataframe[
            "forecast_std"
        ].notna()
    ].copy()

    if ensemble_data.empty:
        return pd.DataFrame()

    results = []

    for source, group in ensemble_data.groupby(
        "source"
    ):

        if len(group) < 2:
            correlation = None
        else:
            correlation = (
                group["forecast_std"]
                .corr(
                    group[
                        "absolute_error"
                    ]
                )
            )

        results.append(
            {
                "source":
                    source,

                "samples":
                    len(group),

                "average_spread":
                    group[
                        "forecast_std"
                    ].mean(),

                "average_abs_error":
                    group[
                        "absolute_error"
                    ].mean(),

                "spread_error_correlation":
                    correlation
            }
        )

    return pd.DataFrame(
        results
    )


def print_daily_results(
    dataframe
):
    print()
    print("DAILY RESULTS")
    print("-" * 100)

    output = dataframe[
        [
            "target_date",
            "source",
            "forecast_value",
            "actual_high",
            "error",
            "forecast_std"
        ]
    ].copy()

    for column in [
        "forecast_value",
        "actual_high",
        "error",
        "forecast_std"
    ]:
        output[column] = (
            output[column].round(2)
        )

    print(
        output.to_string(
            index=False
        )
    )


def analyze_backtests(
    forecast_horizon="D-1_12Z",
    station="KNYC"
):
    dataframe = load_backtest_data(
        forecast_horizon=forecast_horizon,
        station=station
    )

    print()
    print("=" * 70)
    print(
        "WEATHER MODEL BACKTEST ANALYSIS"
    )
    print("=" * 70)

    print(
        f"Station: "
        f"{station}"
    )

    print(
        f"Forecast horizon: "
        f"{forecast_horizon}"
    )

    print(
        f"Forecast rows: "
        f"{len(dataframe)}"
    )

    if dataframe.empty:
        print(
            "No backtest data found."
        )
        return

    metrics = (
        calculate_source_metrics(
            dataframe
        )
    )

    print()
    print("MODEL PERFORMANCE")
    print("-" * 70)

    print(
        metrics.round(
            3
        ).to_string(
            index=False
        )
    )

    correlation = (
        calculate_pairwise_correlation(
            dataframe
        )
    )

    print()
    print("MODEL CORRELATION")
    print("-" * 70)

    if correlation is not None:
        print(
            correlation.round(
                3
            ).to_string()
        )

    disagreement = (
        calculate_pairwise_disagreement(
            dataframe
        )
    )

    print()
    print("MODEL DISAGREEMENT")
    print("-" * 70)

    if not disagreement.empty:
        print(
            disagreement.round(
                3
            ).to_string(
                index=False
            )
        )

    spread_skill = (
        calculate_ensemble_spread_skill(
            dataframe
        )
    )

    print()
    print("ENSEMBLE SPREAD SKILL")
    print("-" * 70)

    if not spread_skill.empty:
        print(
            spread_skill.round(
                3
            ).to_string(
                index=False
            )
        )

    corrected = (
        calculate_bias_corrected_metrics(
            dataframe
        )
    )

    print()
    print(
        "BIAS-CORRECTION DIAGNOSTIC"
    )

    print("-" * 70)

    print(
        corrected.round(
            3
        ).to_string(
            index=False
        )
    )

    print_daily_results(
        dataframe
    )

    return {
        "data":
            dataframe,

        "metrics":
            metrics,

        "correlation":
            correlation,

        "disagreement":
            disagreement,

        "spread_skill":
            spread_skill,

        "bias_corrected":
            corrected
    }


if __name__ == "__main__":

    analyze_backtests(
        forecast_horizon="D-1_12Z",
        station="KNYC"
    )