import math

import numpy as np
import pandas as pd

from src.database.db import (
    get_connection,
)


HORIZONS = (
    14,
    10,
    5,
    3,
    2,
    1,
)

BIN_WIDTH = 0.05

EPSILON = 1e-12


# ============================================================
# LOAD
# ============================================================

def load_horizon_data(
    horizon,
):

    with get_connection() as connection:

        rows = (
            connection.execute(
                """
                SELECT
                    mm.ticker,
                    mm.quote_time,
                    mm.minutes_remaining,

                    mm.yes_bid,
                    mm.yes_ask,
                    mm.no_bid,
                    mm.no_ask,

                    mm.midpoint,
                    mm.spread,

                    mm.outcome,

                    m.close_time

                FROM crypto_market_minutes mm

                JOIN crypto_markets m
                    ON m.ticker = mm.ticker

                WHERE
                    mm.actionable = 1

                    AND ABS(
                        mm.minutes_remaining - ?
                    ) < 0.000001

                    AND mm.midpoint IS NOT NULL

                    AND mm.outcome IS NOT NULL

                ORDER BY
                    m.close_time,
                    mm.ticker
                """,
                (
                    float(
                        horizon
                    ),
                ),
            )
            .fetchall()
        )

    columns = [
        "ticker",
        "quote_time",
        "minutes_remaining",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "midpoint",
        "spread",
        "outcome",
        "close_time",
    ]

    dataframe = (
        pd.DataFrame(
            rows,
            columns=columns,
        )
    )

    if dataframe.empty:

        return dataframe

    numeric_columns = [
        "minutes_remaining",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "midpoint",
        "spread",
        "outcome",
    ]

    for column in numeric_columns:

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="raise",
        )

    dataframe[
        "close_time"
    ] = pd.to_datetime(
        dataframe[
            "close_time"
        ],
        utc=True,
        errors="raise",
        format="mixed",
    )

    return dataframe


# ============================================================
# SCORING
# ============================================================

def brier_score(
    probabilities,
    outcomes,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    outcomes = np.asarray(
        outcomes,
        dtype=float,
    )

    return float(
        np.mean(
            (
                probabilities
                -
                outcomes
            )
            ** 2
        )
    )


def log_loss(
    probabilities,
    outcomes,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    outcomes = np.asarray(
        outcomes,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    return float(
        -np.mean(
            (
                outcomes
                *
                np.log(
                    probabilities
                )
            )
            +
            (
                (
                    1.0
                    -
                    outcomes
                )
                *
                np.log(
                    1.0
                    -
                    probabilities
                )
            )
        )
    )


# ============================================================
# CALIBRATION BINS
# ============================================================

def build_calibration_table(
    dataframe,
):

    dataframe = (
        dataframe.copy()
    )

    edges = np.arange(
        0.0,
        1.0 + BIN_WIDTH,
        BIN_WIDTH,
    )

    labels = [
        f"{left:.2f}-{right:.2f}"

        for left, right
        in zip(
            edges[:-1],
            edges[1:],
        )
    ]

    dataframe[
        "probability_bin"
    ] = pd.cut(
        dataframe[
            "midpoint"
        ],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    grouped = (
        dataframe
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            observations=(
                "outcome",
                "size",
            ),

            mean_market_probability=(
                "midpoint",
                "mean",
            ),

            actual_yes_rate=(
                "outcome",
                "mean",
            ),

            mean_spread=(
                "spread",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped[
        "calibration_error"
    ] = (
        grouped[
            "actual_yes_rate"
        ]
        -
        grouped[
            "mean_market_probability"
        ]
    )

    return grouped


# ============================================================
# SPREAD DIAGNOSTICS
# ============================================================

def print_spread_quantiles(
    dataframe,
):

    spreads = (
        dataframe[
            "spread"
        ]
        .dropna()
    )

    print(
        "Spread quantiles:"
    )

    for percentile in (
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ):

        value = float(
            spreads.quantile(
                percentile
            )
        )

        print(
            f"  {percentile:>5.0%}: "
            f"{value:.4f}"
        )


# ============================================================
# ONE HORIZON
# ============================================================

def analyze_horizon(
    horizon,
):

    dataframe = (
        load_horizon_data(
            horizon
        )
    )

    if dataframe.empty:

        print(
            f"\nNo observations at "
            f"{horizon} minutes."
        )

        return None

    brier = (
        brier_score(
            dataframe[
                "midpoint"
            ],
            dataframe[
                "outcome"
            ],
        )
    )

    log = (
        log_loss(
            dataframe[
                "midpoint"
            ],
            dataframe[
                "outcome"
            ],
        )
    )

    mean_probability = float(
        dataframe[
            "midpoint"
        ].mean()
    )

    actual_yes_rate = float(
        dataframe[
            "outcome"
        ].mean()
    )

    table = (
        build_calibration_table(
            dataframe
        )
    )

    print()
    print("=" * 100)
    print(
        f"{horizon} MINUTES REMAINING"
    )
    print("=" * 100)

    print(
        f"Observations:            "
        f"{len(dataframe)}"
    )

    print(
        f"Mean market probability: "
        f"{mean_probability:.4%}"
    )

    print(
        f"Actual YES rate:         "
        f"{actual_yes_rate:.4%}"
    )

    print(
        f"Overall bias:            "
        f"{actual_yes_rate - mean_probability:+.4%}"
    )

    print(
        f"Brier score:             "
        f"{brier:.6f}"
    )

    print(
        f"Log loss:                "
        f"{log:.6f}"
    )

    print()

    print_spread_quantiles(
        dataframe
    )

    print()
    print(
        table.to_string(
            index=False,
            formatters={
                "mean_market_probability":
                    lambda value:
                        f"{value:.2%}",

                "actual_yes_rate":
                    lambda value:
                        f"{value:.2%}",

                "mean_spread":
                    lambda value:
                        f"{value:.2%}",

                "calibration_error":
                    lambda value:
                        f"{value:+.2%}",
            },
        )
    )

    return {
        "horizon":
            horizon,

        "observations":
            len(
                dataframe
            ),

        "brier":
            brier,

        "log_loss":
            log,

        "mean_probability":
            mean_probability,

        "actual_yes_rate":
            actual_yes_rate,

        "bias":
            (
                actual_yes_rate
                -
                mean_probability
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    summaries = []

    for horizon in HORIZONS:

        result = (
            analyze_horizon(
                horizon
            )
        )

        if result is not None:

            summaries.append(
                result
            )

    summary = (
        pd.DataFrame(
            summaries
        )
    )

    print()
    print("=" * 100)
    print(
        "RAW KALSHI CALIBRATION SUMMARY"
    )
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            formatters={
                "brier":
                    lambda value:
                        f"{value:.6f}",

                "log_loss":
                    lambda value:
                        f"{value:.6f}",

                "mean_probability":
                    lambda value:
                        f"{value:.2%}",

                "actual_yes_rate":
                    lambda value:
                        f"{value:.2%}",

                "bias":
                    lambda value:
                        f"{value:+.2%}",
            },
        )
    )


if __name__ == "__main__":

    main()