import numpy as np
import pandas as pd

from src.database.db import (
    get_connection,
)


FEATURE_VERSION = (
    "KXBTC15M_5M_COINBASE_V2"
)

VALIDATION_START = pd.Timestamp(
    "2026-06-09T00:00:00Z"
)

TEST_START = pd.Timestamp(
    "2026-08-08T00:00:00Z"
)

TRAINING_LOOKBACK_DAYS = 60


# ============================================================
# LOAD PRE-TEST DATA ONLY
# ============================================================

def load_data():

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT

                f.ticker,
                f.quote_time,

                f.spot_close,
                f.spot_at_open,

                f.realized_vol_15m,
                f.realized_vol_60m,

                m.floor_strike,
                m.expiration_value,
                m.strike_type,
                m.outcome

            FROM crypto_feature_rows f

            JOIN crypto_markets m
                ON m.ticker = f.ticker

            WHERE

                f.feature_version = ?

                AND f.horizon_minutes = 5

                AND f.quote_time < ?

            ORDER BY
                f.quote_time,
                f.ticker
            """,
            (
                FEATURE_VERSION,
                TEST_START.isoformat(),
            ),
        ).fetchall()

    columns = [
        "ticker",
        "quote_time",

        "spot_close",
        "spot_at_open",

        "realized_vol_15m",
        "realized_vol_60m",

        "floor_strike",
        "expiration_value",
        "strike_type",
        "outcome",
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    dataframe[
        "quote_time"
    ] = pd.to_datetime(
        dataframe[
            "quote_time"
        ],
        utc=True,
        format="mixed",
    )

    dataframe[
        "date"
    ] = (
        dataframe[
            "quote_time"
        ]
        .dt.normalize()
    )

    numeric = [
        "spot_close",
        "spot_at_open",

        "realized_vol_15m",
        "realized_vol_60m",

        "floor_strike",
        "expiration_value",
        "outcome",
    ]

    for column in numeric:

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

    return dataframe


# ============================================================
# DATA INTEGRITY
# ============================================================

def print_integrity(
    dataframe,
):

    validation = (
        dataframe[
            (
                dataframe[
                    "quote_time"
                ]
                >=
                VALIDATION_START
            )
            &
            (
                dataframe[
                    "quote_time"
                ]
                <
                TEST_START
            )
        ]
        .copy()
    )

    missing = (
        validation[
            validation[
                "expiration_value"
            ]
            .isna()
        ]
    )

    print()
    print("=" * 100)
    print(
        "VALIDATION BRTI COVERAGE"
    )
    print("=" * 100)

    print(
        f"Validation rows:           "
        f"{len(validation)}"
    )

    print(
        f"With expiration value:    "
        f"{validation['expiration_value'].notna().sum()}"
    )

    print(
        f"Missing expiration value: "
        f"{len(missing)}"
    )

    if not missing.empty:

        print()
        print(
            "MISSING VALIDATION ROWS"
        )

        print(
            missing[
                [
                    "ticker",
                    "quote_time",
                    "floor_strike",
                    "outcome",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Outcome consistency
    # --------------------------------------------------------

    check = (
        dataframe[
            (
                dataframe[
                    "expiration_value"
                ]
                .notna()
            )
            &
            (
                dataframe[
                    "floor_strike"
                ]
                .notna()
            )
            &
            (
                dataframe[
                    "strike_type"
                ]
                ==
                "greater_or_equal"
            )
            &
            (
                dataframe[
                    "outcome"
                ]
                .notna()
            )
        ]
        .copy()
    )

    check[
        "implied_outcome"
    ] = (
        check[
            "expiration_value"
        ]
        >=
        check[
            "floor_strike"
        ]
    ).astype(
        int
    )

    mismatch = (
        check[
            check[
                "implied_outcome"
            ]
            !=
            check[
                "outcome"
            ]
        ]
    )

    print()
    print("=" * 100)
    print(
        "OUTCOME CONSISTENCY"
    )
    print("=" * 100)

    print(
        f"Checked rows:             "
        f"{len(check)}"
    )

    print(
        f"Mismatches:               "
        f"{len(mismatch)}"
    )

    if not mismatch.empty:

        print()
        print(
            mismatch[
                [
                    "ticker",
                    "quote_time",
                    "floor_strike",
                    "expiration_value",
                    "outcome",
                    "implied_outcome",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# BRTI / COINBASE TRACKING
# ============================================================

def prepare_tracking_data(
    dataframe,
):

    data = (
        dataframe[
            (
                dataframe[
                    "expiration_value"
                ]
                .notna()
            )
            &
            (
                dataframe[
                    "floor_strike"
                ]
                .notna()
            )
            &
            (
                dataframe[
                    "spot_close"
                ]
                .notna()
            )
            &
            (
                dataframe[
                    "expiration_value"
                ]
                >
                0
            )
            &
            (
                dataframe[
                    "floor_strike"
                ]
                >
                0
            )
            &
            (
                dataframe[
                    "spot_close"
                ]
                >
                0
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Position at the 5-minute decision
    #
    # Positive means Coinbase is above the BRTI threshold.
    # --------------------------------------------------------

    data[
        "decision_margin_log"
    ] = np.log(
        data[
            "spot_close"
        ]
        /
        data[
            "floor_strike"
        ]
    )

    # --------------------------------------------------------
    # What happens after the 5-minute decision:
    #
    # current Coinbase -> actual final BRTI
    #
    # This contains BOTH:
    #
    # 1. BTC movement during remaining 5 minutes
    # 2. Coinbase/BRTI settlement basis
    # --------------------------------------------------------

    data[
        "future_settlement_log"
    ] = np.log(
        data[
            "expiration_value"
        ]
        /
        data[
            "spot_close"
        ]
    )

    # --------------------------------------------------------
    # Actual final settlement margin
    # --------------------------------------------------------

    data[
        "final_margin_log"
    ] = np.log(
        data[
            "expiration_value"
        ]
        /
        data[
            "floor_strike"
        ]
    )

    # Identity check:
    #
    # decision margin + future settlement move
    # should equal final BRTI margin.
    # --------------------------------------------------------

    data[
        "identity_error"
    ] = (
        data[
            "decision_margin_log"
        ]
        +
        data[
            "future_settlement_log"
        ]
        -
        data[
            "final_margin_log"
        ]
    )

    return data


# ============================================================
# TRACKING SUMMARY
# ============================================================

def print_tracking_summary(
    data,
):

    validation = (
        data[
            (
                data[
                    "quote_time"
                ]
                >=
                VALIDATION_START
            )
            &
            (
                data[
                    "quote_time"
                ]
                <
                TEST_START
            )
        ]
        .copy()
    )

    print()
    print("=" * 100)
    print(
        "5-MINUTE COINBASE -> FINAL BRTI TRACKING"
    )
    print("=" * 100)

    print(
        f"Validation rows:          "
        f"{len(validation)}"
    )

    identity_max = float(
        validation[
            "identity_error"
        ]
        .abs()
        .max()
    )

    print(
        f"Max identity error:       "
        f"{identity_max:.12f}"
    )

    # --------------------------------------------------------
    # Simple direction predictor:
    #
    # Coinbase above threshold at T-5 => YES
    # Coinbase below threshold at T-5 => NO
    # --------------------------------------------------------

    simple_prediction = (
        validation[
            "spot_close"
        ]
        >=
        validation[
            "floor_strike"
        ]
    ).astype(
        int
    )

    actual = (
        validation[
            "outcome"
        ]
        .astype(
            int
        )
    )

    accuracy = float(
        np.mean(
            simple_prediction
            ==
            actual
        )
    )

    print(
        f"Direction accuracy:       "
        f"{accuracy:.2%}"
    )

    correlation = float(
        np.corrcoef(
            validation[
                "decision_margin_log"
            ],
            validation[
                "final_margin_log"
            ],
        )[
            0,
            1
        ]
    )

    print(
        f"Decision/final corr:      "
        f"{correlation:.4f}"
    )

    # --------------------------------------------------------
    # Future 5-minute settlement residual
    # --------------------------------------------------------

    residual = (
        validation[
            "future_settlement_log"
        ]
        .to_numpy(
            dtype=float
        )
    )

    residual_bps = (
        residual
        *
        10000.0
    )

    quantiles = np.quantile(
        residual_bps,
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
    )

    print()
    print(
        "FUTURE SETTLEMENT MOVE"
    )

    print(
        "Defined as:"
    )

    print(
        "ln(final BRTI / Coinbase spot at T-5)"
    )

    print()

    print(
        f"Mean:                     "
        f"{np.mean(residual_bps):+.3f} bps"
    )

    print(
        f"Std:                      "
        f"{np.std(residual_bps):.3f} bps"
    )

    print(
        f"Median:                   "
        f"{np.median(residual_bps):+.3f} bps"
    )

    print()

    labels = [
        "1%",
        "5%",
        "10%",
        "25%",
        "50%",
        "75%",
        "90%",
        "95%",
        "99%",
    ]

    for (
        label,
        value,
    ) in zip(
        labels,
        quantiles,
    ):

        print(
            f"{label:>4} quantile:             "
            f"{value:+.3f} bps"
        )

    # --------------------------------------------------------
    # Dollar version
    # --------------------------------------------------------

    dollar_gap = (
        validation[
            "expiration_value"
        ]
        -
        validation[
            "spot_close"
        ]
    )

    print()
    print(
        "FINAL BRTI - COINBASE AT T-5"
    )

    print(
        f"Mean:                     "
        f"${dollar_gap.mean():+.2f}"
    )

    print(
        f"Std:                      "
        f"${dollar_gap.std():.2f}"
    )

    print(
        f"Median:                   "
        f"${dollar_gap.median():+.2f}"
    )


# ============================================================
# ROLLING TRAINING COVERAGE
# ============================================================

def print_training_coverage(
    dataframe,
):

    validation_dates = (
        dataframe.loc[
            (
                dataframe[
                    "quote_time"
                ]
                >=
                VALIDATION_START
            )
            &
            (
                dataframe[
                    "quote_time"
                ]
                <
                TEST_START
            ),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    rows = []

    for decision_date in validation_dates:

        start = (
            decision_date
            -
            pd.Timedelta(
                days=
                    TRAINING_LOOKBACK_DAYS
            )
        )

        end = (
            decision_date
            -
            pd.Timedelta(
                days=1
            )
        )

        window = (
            dataframe[
                (
                    dataframe[
                        "date"
                    ]
                    >=
                    start
                )
                &
                (
                    dataframe[
                        "date"
                    ]
                    <=
                    end
                )
            ]
        )

        usable = (
            window[
                (
                    window[
                        "expiration_value"
                    ]
                    .notna()
                )
                &
                (
                    window[
                        "floor_strike"
                    ]
                    .notna()
                )
                &
                (
                    window[
                        "spot_close"
                    ]
                    .notna()
                )
            ]
        )

        rows.append(
            {
                "date":
                    decision_date,

                "total":
                    len(
                        window
                    ),

                "usable":
                    len(
                        usable
                    ),
            }
        )

    coverage = pd.DataFrame(
        rows
    )

    coverage[
        "fraction"
    ] = (
        coverage[
            "usable"
        ]
        /
        coverage[
            "total"
        ]
    )

    print()
    print("=" * 100)
    print(
        "ROLLING 60-DAY BRTI TRAINING COVERAGE"
    )
    print("=" * 100)

    print(
        f"Minimum usable rows:      "
        f"{coverage['usable'].min()}"
    )

    print(
        f"Median usable rows:       "
        f"{coverage['usable'].median():.0f}"
    )

    print(
        f"Maximum usable rows:      "
        f"{coverage['usable'].max()}"
    )

    print()

    print(
        f"Minimum coverage:         "
        f"{coverage['fraction'].min():.2%}"
    )

    print(
        f"Median coverage:          "
        f"{coverage['fraction'].median():.2%}"
    )

    print(
        f"Maximum coverage:         "
        f"{coverage['fraction'].max():.2%}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = (
        load_data()
    )

    print()
    print("=" * 100)
    print(
        "BRTI / COINBASE SETTLEMENT DIAGNOSTIC"
    )
    print("=" * 100)

    print(
        f"Feature version:          "
        f"{FEATURE_VERSION}"
    )

    print(
        f"Validation:               "
        f"{VALIDATION_START.date()} "
        f"-> "
        f"{(TEST_START - pd.Timedelta(days=1)).date()}"
    )

    print()

    print(
        "IMPORTANT: no rows on or after "
        "the reserved test start are loaded."
    )

    print_integrity(
        dataframe
    )

    tracking = (
        prepare_tracking_data(
            dataframe
        )
    )

    print_tracking_summary(
        tracking
    )

    print_training_coverage(
        dataframe
    )


if __name__ == "__main__":

    main()