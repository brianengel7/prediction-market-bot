import numpy as np
import pandas as pd

from src.database.db import (
    get_connection,
)

from src.kalshi.fees import (
    calculate_taker_fee,
)

from src.crypto.backtest.alpha_walkforward import (
    HORIZONS,
    TRAINING_LOOKBACK_DAYS,
    MINIMUM_TRAINING_ROWS,
    apply_alpha,
    fit_alpha,
    get_periods,
)


# ============================================================
# CONFIGURATION
# ============================================================

EDGE_THRESHOLDS = (
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
)

ROI_THRESHOLDS = (
    0.00,
    0.05,
    0.10,
)

# Conservative initial screen.
#
# We are intentionally assuming the full standard multiplier
# here rather than trying to reconstruct any historical
# fee schedule yet.
FEE_MULTIPLIER = 1.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    with get_connection() as connection:

        rows = (
            connection.execute(
                """
                SELECT
                    mm.ticker,
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

                    AND mm.midpoint IS NOT NULL

                    AND mm.yes_ask IS NOT NULL

                    AND mm.no_ask IS NOT NULL

                    AND mm.outcome IS NOT NULL

                ORDER BY
                    m.close_time,
                    mm.ticker
                """
            )
            .fetchall()
        )

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "minutes_remaining",
            "yes_bid",
            "yes_ask",
            "no_bid",
            "no_ask",
            "midpoint",
            "spread",
            "outcome",
            "close_time",
        ],
    )

    numeric_columns = (
        "minutes_remaining",
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "midpoint",
        "spread",
        "outcome",
    )

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
        "outcome"
    ] = (
        dataframe[
            "outcome"
        ]
        .astype(
            int
        )
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

    dataframe[
        "date"
    ] = (
        dataframe[
            "close_time"
        ]
        .dt.normalize()
    )

    dataframe = (
        dataframe[
            dataframe[
                "minutes_remaining"
            ].isin(
                HORIZONS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return dataframe


# ============================================================
# FEE
# ============================================================

def calculate_fee(
    price,
):

    return float(
        calculate_taker_fee(
            price=float(
                price
            ),
            contracts=1,
            multiplier=
                FEE_MULTIPLIER,
        )
    )


# ============================================================
# BUILD EXECUTABLE TRADES
# ============================================================

def build_executable_trades(
    decision_data,
    alpha,
):

    dataframe = (
        decision_data.copy()
    )

    raw_yes = (
        dataframe[
            "midpoint"
        ]
        .to_numpy(
            dtype=float
        )
    )

    adjusted_yes = (
        apply_alpha(
            raw_yes,
            alpha,
        )
    )

    adjusted_no = (
        1.0
        -
        adjusted_yes
    )

    dataframe[
        "adjusted_yes"
    ] = adjusted_yes

    dataframe[
        "adjusted_no"
    ] = adjusted_no

    dataframe[
        "yes_fee"
    ] = (
        dataframe[
            "yes_ask"
        ]
        .apply(
            calculate_fee
        )
    )

    dataframe[
        "no_fee"
    ] = (
        dataframe[
            "no_ask"
        ]
        .apply(
            calculate_fee
        )
    )

    dataframe[
        "yes_cost"
    ] = (
        dataframe[
            "yes_ask"
        ]
        +
        dataframe[
            "yes_fee"
        ]
    )

    dataframe[
        "no_cost"
    ] = (
        dataframe[
            "no_ask"
        ]
        +
        dataframe[
            "no_fee"
        ]
    )

    dataframe[
        "yes_edge"
    ] = (
        dataframe[
            "adjusted_yes"
        ]
        -
        dataframe[
            "yes_cost"
        ]
    )

    dataframe[
        "no_edge"
    ] = (
        dataframe[
            "adjusted_no"
        ]
        -
        dataframe[
            "no_cost"
        ]
    )

    choose_yes = (
        dataframe[
            "yes_edge"
        ]
        >=
        dataframe[
            "no_edge"
        ]
    )

    dataframe[
        "side"
    ] = np.where(
        choose_yes,
        "YES",
        "NO",
    )

    dataframe[
        "fair_probability"
    ] = np.where(
        choose_yes,
        dataframe[
            "adjusted_yes"
        ],
        dataframe[
            "adjusted_no"
        ],
    )

    dataframe[
        "entry_price"
    ] = np.where(
        choose_yes,
        dataframe[
            "yes_ask"
        ],
        dataframe[
            "no_ask"
        ],
    )

    dataframe[
        "fee"
    ] = np.where(
        choose_yes,
        dataframe[
            "yes_fee"
        ],
        dataframe[
            "no_fee"
        ],
    )

    dataframe[
        "cost"
    ] = np.where(
        choose_yes,
        dataframe[
            "yes_cost"
        ],
        dataframe[
            "no_cost"
        ],
    )

    dataframe[
        "edge"
    ] = np.where(
        choose_yes,
        dataframe[
            "yes_edge"
        ],
        dataframe[
            "no_edge"
        ],
    )

    dataframe[
        "expected_roi"
    ] = np.where(
        dataframe[
            "cost"
        ]
        >
        0,

        dataframe[
            "edge"
        ]
        /
        dataframe[
            "cost"
        ],

        float(
            "-inf"
        ),
    )

    dataframe[
        "win"
    ] = np.where(
        dataframe[
            "side"
        ]
        ==
        "YES",

        dataframe[
            "outcome"
        ]
        ==
        1,

        dataframe[
            "outcome"
        ]
        ==
        0,
    )

    dataframe[
        "expected_pnl"
    ] = (
        dataframe[
            "edge"
        ]
    )

    dataframe[
        "actual_pnl"
    ] = np.where(
        dataframe[
            "win"
        ],

        1.0
        -
        dataframe[
            "cost"
        ],

        -
        dataframe[
            "cost"
        ],
    )

    return dataframe


# ============================================================
# CREATE VALIDATION TRADES FOR ONE HORIZON
# ============================================================

def build_validation_trades(
    dataframe,
    horizon,
    periods,
):

    horizon_data = (
        dataframe[
            dataframe[
                "minutes_remaining"
            ]
            ==
            float(
                horizon
            )
        ]
        .copy()
    )

    validation_dates = (
        horizon_data[
            (
                horizon_data[
                    "date"
                ]
                >=
                periods[
                    "validation_start"
                ]
            )
            &
            (
                horizon_data[
                    "date"
                ]
                <=
                periods[
                    "validation_end"
                ]
            )
        ][
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    all_trades = []

    alpha_rows = []

    for decision_date in validation_dates:

        training_start = (
            decision_date
            -
            pd.Timedelta(
                days=
                    TRAINING_LOOKBACK_DAYS
            )
        )

        training_end = (
            decision_date
            -
            pd.Timedelta(
                days=1
            )
        )

        training_data = (
            horizon_data[
                (
                    horizon_data[
                        "date"
                    ]
                    >=
                    training_start
                )
                &
                (
                    horizon_data[
                        "date"
                    ]
                    <=
                    training_end
                )
            ]
            .copy()
        )

        decision_data = (
            horizon_data[
                horizon_data[
                    "date"
                ]
                ==
                decision_date
            ]
            .copy()
        )

        if (
            len(
                training_data
            )
            <
            MINIMUM_TRAINING_ROWS
        ):

            continue

        if decision_data.empty:

            continue

        alpha = (
            fit_alpha(
                training_data
            )
        )

        trades = (
            build_executable_trades(
                decision_data=
                    decision_data,

                alpha=
                    alpha,
            )
        )

        trades[
            "alpha"
        ] = alpha

        trades[
            "decision_date"
        ] = decision_date

        all_trades.append(
            trades
        )

        alpha_rows.append(
            {
                "date":
                    decision_date,

                "training_rows":
                    len(
                        training_data
                    ),

                "evaluation_rows":
                    len(
                        decision_data
                    ),

                "alpha":
                    alpha,
            }
        )

    if not all_trades:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    trade_data = (
        pd.concat(
            all_trades,
            ignore_index=True,
        )
    )

    alpha_data = (
        pd.DataFrame(
            alpha_rows
        )
    )

    return (
        trade_data,
        alpha_data,
    )


# ============================================================
# THRESHOLD PERFORMANCE
# ============================================================

def evaluate_threshold(
    trade_data,
    horizon,
    minimum_edge,
    minimum_roi,
):

    selected = (
        trade_data[
            (
                trade_data[
                    "edge"
                ]
                >=
                minimum_edge
            )
            &
            (
                trade_data[
                    "expected_roi"
                ]
                >=
                minimum_roi
            )
        ]
        .copy()
    )

    trades = len(
        selected
    )

    if trades == 0:

        return {
            "horizon":
                horizon,

            "minimum_edge":
                minimum_edge,

            "minimum_roi":
                minimum_roi,

            "trades":
                0,

            "wins":
                0,

            "losses":
                0,

            "win_rate":
                np.nan,

            "yes_trades":
                0,

            "no_trades":
                0,

            "average_entry":
                np.nan,

            "average_cost":
                np.nan,

            "average_edge":
                np.nan,

            "average_expected_roi":
                np.nan,

            "expected_pnl":
                0.0,

            "actual_pnl":
                0.0,

            "capital":
                0.0,

            "realized_roi":
                np.nan,

            "average_spread":
                np.nan,

            "profitable_days":
                0,

            "losing_days":
                0,

            "active_days":
                0,
        }

    wins = int(
        selected[
            "win"
        ].sum()
    )

    losses = (
        trades
        -
        wins
    )

    yes_trades = int(
        (
            selected[
                "side"
            ]
            ==
            "YES"
        ).sum()
    )

    no_trades = int(
        (
            selected[
                "side"
            ]
            ==
            "NO"
        ).sum()
    )

    expected_pnl = float(
        selected[
            "expected_pnl"
        ].sum()
    )

    actual_pnl = float(
        selected[
            "actual_pnl"
        ].sum()
    )

    capital = float(
        selected[
            "cost"
        ].sum()
    )

    realized_roi = (
        actual_pnl
        /
        capital
        if capital > 0
        else np.nan
    )

    daily = (
        selected
        .groupby(
            "decision_date"
        )
        .agg(
            daily_pnl=(
                "actual_pnl",
                "sum",
            ),

            trades=(
                "ticker",
                "size",
            ),
        )
        .reset_index()
    )

    profitable_days = int(
        (
            daily[
                "daily_pnl"
            ]
            >
            0
        ).sum()
    )

    losing_days = int(
        (
            daily[
                "daily_pnl"
            ]
            <
            0
        ).sum()
    )

    return {
        "horizon":
            horizon,

        "minimum_edge":
            minimum_edge,

        "minimum_roi":
            minimum_roi,

        "trades":
            trades,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            wins
            /
            trades,

        "yes_trades":
            yes_trades,

        "no_trades":
            no_trades,

        "average_entry":
            float(
                selected[
                    "entry_price"
                ].mean()
            ),

        "average_cost":
            float(
                selected[
                    "cost"
                ].mean()
            ),

        "average_edge":
            float(
                selected[
                    "edge"
                ].mean()
            ),

        "average_expected_roi":
            float(
                selected[
                    "expected_roi"
                ].mean()
            ),

        "expected_pnl":
            expected_pnl,

        "actual_pnl":
            actual_pnl,

        "capital":
            capital,

        "realized_roi":
            realized_roi,

        "average_spread":
            float(
                selected[
                    "spread"
                ].mean()
            ),

        "profitable_days":
            profitable_days,

        "losing_days":
            losing_days,

        "active_days":
            len(
                daily
            ),
    }


# ============================================================
# HORIZON DIAGNOSTICS
# ============================================================

def print_horizon_diagnostics(
    horizon,
    trade_data,
    alpha_data,
):

    print()
    print("=" * 110)
    print(
        f"{horizon} MINUTES REMAINING"
    )
    print("=" * 110)

    print(
        f"Validation markets:      "
        f"{len(trade_data)}"
    )

    print(
        f"Validation days:         "
        f"{alpha_data['date'].nunique()}"
    )

    print(
        f"Mean alpha:              "
        f"{alpha_data['alpha'].mean():.4f}"
    )

    print(
        f"Median alpha:            "
        f"{alpha_data['alpha'].median():.4f}"
    )

    print(
        f"Mean spread:             "
        f"{trade_data['spread'].mean():.2%}"
    )

    print(
        f"Median spread:           "
        f"{trade_data['spread'].median():.2%}"
    )

    print(
        f"Mean best edge:          "
        f"{trade_data['edge'].mean():+.2%}"
    )

    print(
        f"Maximum best edge:       "
        f"{trade_data['edge'].max():+.2%}"
    )

    print(
        f"Positive-edge markets:   "
        f"{(trade_data['edge'] > 0).sum()}"
    )

    print(
        f"Edge >= 1%:              "
        f"{(trade_data['edge'] >= 0.01).sum()}"
    )

    print(
        f"Edge >= 2%:              "
        f"{(trade_data['edge'] >= 0.02).sum()}"
    )

    print(
        f"Edge >= 3%:              "
        f"{(trade_data['edge'] >= 0.03).sum()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = (
        load_data()
    )

    periods = (
        get_periods(
            dataframe
        )
    )

    print()
    print("=" * 120)
    print(
        "CRYPTO EXECUTABLE ALPHA VALIDATION"
    )
    print("=" * 120)

    print(
        f"Validation: "
        f"{periods['validation_start'].date()} "
        f"-> "
        f"{periods['validation_end'].date()}"
    )

    print(
        f"RESERVED TEST: "
        f"{periods['test_start'].date()} "
        f"-> "
        f"{periods['test_end'].date()}"
    )

    print(
        f"Training lookback: "
        f"{TRAINING_LOOKBACK_DAYS} days"
    )

    print(
        f"Fee multiplier: "
        f"{FEE_MULTIPLIER:.2f}"
    )

    print()
    print(
        "IMPORTANT: reserved test period is "
        "NOT evaluated by this script."
    )

    all_results = []

    for horizon in HORIZONS:

        (
            trade_data,
            alpha_data,
        ) = (
            build_validation_trades(
                dataframe=
                    dataframe,

                horizon=
                    horizon,

                periods=
                    periods,
            )
        )

        if trade_data.empty:

            print()
            print(
                f"No validation data for "
                f"{horizon}-minute horizon."
            )

            continue

        print_horizon_diagnostics(
            horizon=
                horizon,

            trade_data=
                trade_data,

            alpha_data=
                alpha_data,
        )

        for minimum_edge in EDGE_THRESHOLDS:

            for minimum_roi in ROI_THRESHOLDS:

                result = (
                    evaluate_threshold(
                        trade_data=
                            trade_data,

                        horizon=
                            horizon,

                        minimum_edge=
                            minimum_edge,

                        minimum_roi=
                            minimum_roi,
                    )
                )

                all_results.append(
                    result
                )

    results = (
        pd.DataFrame(
            all_results
        )
    )

    if results.empty:

        raise RuntimeError(
            "No validation results generated."
        )

    # ========================================================
    # FULL THRESHOLD TABLE
    # ========================================================

    print()
    print("=" * 160)
    print(
        "EXECUTABLE VALIDATION THRESHOLD SWEEP"
    )
    print("=" * 160)

    display_columns = [
        "horizon",
        "minimum_edge",
        "minimum_roi",
        "trades",
        "wins",
        "losses",
        "win_rate",
        "yes_trades",
        "no_trades",
        "average_cost",
        "average_edge",
        "average_expected_roi",
        "expected_pnl",
        "actual_pnl",
        "capital",
        "realized_roi",
        "average_spread",
        "active_days",
        "profitable_days",
        "losing_days",
    ]

    print(
        results[
            display_columns
        ]
        .to_string(
            index=False,
            formatters={
                "minimum_edge":
                    lambda value:
                        f"{value:.1%}",

                "minimum_roi":
                    lambda value:
                        f"{value:.0%}",

                "win_rate":
                    lambda value:
                        (
                            f"{value:.2%}"
                            if pd.notna(
                                value
                            )
                            else "-"
                        ),

                "average_cost":
                    lambda value:
                        (
                            f"${value:.4f}"
                            if pd.notna(
                                value
                            )
                            else "-"
                        ),

                "average_edge":
                    lambda value:
                        (
                            f"{value:+.2%}"
                            if pd.notna(
                                value
                            )
                            else "-"
                        ),

                "average_expected_roi":
                    lambda value:
                        (
                            f"{value:+.2%}"
                            if pd.notna(
                                value
                            )
                            else "-"
                        ),

                "expected_pnl":
                    lambda value:
                        f"${value:+.2f}",

                "actual_pnl":
                    lambda value:
                        f"${value:+.2f}",

                "capital":
                    lambda value:
                        f"${value:.2f}",

                "realized_roi":
                    lambda value:
                        (
                            f"{value:+.2%}"
                            if pd.notna(
                                value
                            )
                            else "-"
                        ),

                "average_spread":
                    lambda value:
                        (
                            f"{value:.2%}"
                            if pd.notna(
                                value
                            )
                            else "-"
                        ),
            },
        )
    )

    # ========================================================
    # PROFITABLE CONFIGURATIONS
    # ========================================================

    profitable = (
        results[
            (
                results[
                    "trades"
                ]
                >
                0
            )
            &
            (
                results[
                    "actual_pnl"
                ]
                >
                0
            )
        ]
        .copy()
    )

    print()
    print("=" * 140)
    print(
        "PROFITABLE VALIDATION CONFIGURATIONS"
    )
    print("=" * 140)

    if profitable.empty:

        print(
            "None."
        )

    else:

        profitable = (
            profitable
            .sort_values(
                by=[
                    "realized_roi",
                    "trades",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        print(
            profitable[
                [
                    "horizon",
                    "minimum_edge",
                    "minimum_roi",
                    "trades",
                    "wins",
                    "win_rate",
                    "average_edge",
                    "expected_pnl",
                    "actual_pnl",
                    "capital",
                    "realized_roi",
                    "active_days",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "minimum_edge":
                        lambda value:
                            f"{value:.1%}",

                    "minimum_roi":
                        lambda value:
                            f"{value:.0%}",

                    "win_rate":
                        lambda value:
                            f"{value:.2%}",

                    "average_edge":
                        lambda value:
                            f"{value:+.2%}",

                    "expected_pnl":
                        lambda value:
                            f"${value:+.2f}",

                    "actual_pnl":
                        lambda value:
                            f"${value:+.2f}",

                    "capital":
                        lambda value:
                            f"${value:.2f}",

                    "realized_roi":
                        lambda value:
                            f"{value:+.2%}",
                },
            )
        )

    # ========================================================
    # BEST CONFIGURATION PER HORIZON
    # ========================================================

    print()
    print("=" * 140)
    print(
        "BEST REALIZED ROI PER HORIZON"
    )
    print("=" * 140)

    eligible = (
        results[
            results[
                "trades"
            ]
            >=
            25
        ]
        .copy()
    )

    if eligible.empty:

        print(
            "No configurations had at least "
            "25 validation trades."
        )

    else:

        best_rows = []

        for horizon in HORIZONS:

            horizon_rows = (
                eligible[
                    eligible[
                        "horizon"
                    ]
                    ==
                    horizon
                ]
                .copy()
            )

            if horizon_rows.empty:

                continue

            best_row = (
                horizon_rows
                .sort_values(
                    by=[
                        "realized_roi",
                        "trades",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .iloc[
                    0
                ]
            )

            best_rows.append(
                best_row
            )

        best = (
            pd.DataFrame(
                best_rows
            )
        )

        print(
            best[
                [
                    "horizon",
                    "minimum_edge",
                    "minimum_roi",
                    "trades",
                    "wins",
                    "win_rate",
                    "average_edge",
                    "actual_pnl",
                    "capital",
                    "realized_roi",
                    "active_days",
                    "profitable_days",
                    "losing_days",
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "minimum_edge":
                        lambda value:
                            f"{value:.1%}",

                    "minimum_roi":
                        lambda value:
                            f"{value:.0%}",

                    "win_rate":
                        lambda value:
                            f"{value:.2%}",

                    "average_edge":
                        lambda value:
                            f"{value:+.2%}",

                    "actual_pnl":
                        lambda value:
                            f"${value:+.2f}",

                    "capital":
                        lambda value:
                            f"${value:.2f}",

                    "realized_roi":
                        lambda value:
                            f"{value:+.2%}",
                },
            )
        )


if __name__ == "__main__":

    main()