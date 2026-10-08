import numpy as np
import pandas as pd

from src.database.db import get_connection


# ============================================================
# SETTINGS
# ============================================================

SERIES_TICKER = "KXBTC15M"

PROXY_COLUMNS = {
    "COINBASE": "coinbase_close",
    "BITSTAMP": "bitstamp_close",
    "CRYPTO_COM": "crypto_com_close",
    "MEAN_3": "mean_3",
    "MEDIAN_3": "median_3",
}

NEAR_STRIKE_BPS = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
)


# ============================================================
# LOAD SYNCHRONIZED DATA
# ============================================================

def load_dataset():

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT

                m.ticker,
                m.close_time::timestamptz AS close_time,

                m.floor_strike,
                m.expiration_value,
                m.expiration_value_source,
                m.outcome,

                cb.close AS coinbase_close,
                bs.close AS bitstamp_close,
                cc.close AS crypto_com_close

            FROM crypto_markets m

            JOIN btc_spot_minutes cb

                ON cb.candle_end =
                    m.close_time::timestamptz

                AND cb.product_id =
                    'BTC-USD'

            JOIN cross_exchange_spot_minutes bs

                ON bs.candle_end =
                    m.close_time::timestamptz

                AND bs.exchange =
                    'BITSTAMP'

            JOIN cross_exchange_spot_minutes cc

                ON cc.candle_end =
                    m.close_time::timestamptz

                AND cc.exchange =
                    'CRYPTO_COM'

            WHERE

                m.series_ticker = ?

                AND m.expiration_value
                    IS NOT NULL

            ORDER BY
                m.close_time::timestamptz
            """,
            (
                SERIES_TICKER,
            ),
        )

        rows = cursor.fetchall()

        columns = [
            column[0]
            for column in cursor.description
        ]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    dataframe[
        "close_time"
    ] = pd.to_datetime(
        dataframe[
            "close_time"
        ],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    numeric_columns = [
        "floor_strike",
        "expiration_value",
        "outcome",
        "coinbase_close",
        "bitstamp_close",
        "crypto_com_close",
    ]

    for column in numeric_columns:

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

    dataframe = (
        dataframe.dropna(
            subset=[
                "close_time",
                "expiration_value",
                "coinbase_close",
                "bitstamp_close",
                "crypto_com_close",
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Combined proxies
    # --------------------------------------------------------

    dataframe[
        "mean_3"
    ] = (
        dataframe[
            [
                "coinbase_close",
                "bitstamp_close",
                "crypto_com_close",
            ]
        ]
        .mean(
            axis=1
        )
    )

    dataframe[
        "median_3"
    ] = (
        dataframe[
            [
                "coinbase_close",
                "bitstamp_close",
                "crypto_com_close",
            ]
        ]
        .median(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Actual BRTI denominator for bps errors
    # --------------------------------------------------------

    dataframe = dataframe[
        dataframe[
            "expiration_value"
        ]
        >
        0
    ].copy()

    return dataframe


# ============================================================
# TRACKING METRICS
# ============================================================

def calculate_tracking_metrics(
    dataframe,
    proxy_column,
):

    actual = (
        dataframe[
            "expiration_value"
        ]
        .to_numpy(
            dtype=float
        )
    )

    proxy = (
        dataframe[
            proxy_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    error_dollars = (
        proxy
        -
        actual
    )

    error_bps = (
        error_dollars
        /
        actual
        *
        10000.0
    )

    absolute_dollars = (
        np.abs(
            error_dollars
        )
    )

    absolute_bps = (
        np.abs(
            error_bps
        )
    )

    return {
        "rows":
            len(
                dataframe
            ),

        "mean_error_dollars":
            float(
                np.mean(
                    error_dollars
                )
            ),

        "mae_dollars":
            float(
                np.mean(
                    absolute_dollars
                )
            ),

        "median_ae_dollars":
            float(
                np.median(
                    absolute_dollars
                )
            ),

        "rmse_dollars":
            float(
                np.sqrt(
                    np.mean(
                        error_dollars
                        ** 2
                    )
                )
            ),

        "p95_ae_dollars":
            float(
                np.percentile(
                    absolute_dollars,
                    95,
                )
            ),

        "mean_error_bps":
            float(
                np.mean(
                    error_bps
                )
            ),

        "mae_bps":
            float(
                np.mean(
                    absolute_bps
                )
            ),

        "median_ae_bps":
            float(
                np.median(
                    absolute_bps
                )
            ),

        "rmse_bps":
            float(
                np.sqrt(
                    np.mean(
                        error_bps
                        ** 2
                    )
                )
            ),

        "p95_ae_bps":
            float(
                np.percentile(
                    absolute_bps,
                    95,
                )
            ),
    }


# ============================================================
# VALID STRIKE DATA
# ============================================================

def build_strike_dataset(
    dataframe,
):

    data = (
        dataframe.dropna(
            subset=[
                "floor_strike",
            ]
        )
        .copy()
    )

    data = data[
        data[
            "floor_strike"
        ]
        >
        0
    ].copy()

    # --------------------------------------------------------
    # Exclude corrupted strike metadata.
    #
    # We previously found examples such as a ~$7 strike while
    # BTC was ~$72,000.
    #
    # A legitimate 15-minute KXBTC15M opening strike should be
    # reasonably close to its ending BRTI value. A 5% bound is
    # deliberately very wide and is used only as a data-quality
    # filter.
    # --------------------------------------------------------

    relative_difference = (
        (
            data[
                "floor_strike"
            ]
            -
            data[
                "expiration_value"
            ]
        )
        .abs()
        /
        data[
            "expiration_value"
        ]
    )

    data = (
        data[
            relative_difference
            <=
            0.05
        ]
        .copy()
    )

    # Actual distance from final BRTI settlement to strike.

    data[
        "actual_distance_bps"
    ] = (
        (
            data[
                "expiration_value"
            ]
            -
            data[
                "floor_strike"
            ]
        )
        /
        data[
            "floor_strike"
        ]
        *
        10000.0
    )

    # KXBTC15M resolves YES when expiration >= strike.

    data[
        "actual_yes"
    ] = (
        data[
            "expiration_value"
        ]
        >=
        data[
            "floor_strike"
        ]
    )

    return data


# ============================================================
# DIRECTIONAL ACCURACY
# ============================================================

def calculate_directional_metrics(
    dataframe,
    proxy_column,
):

    if dataframe.empty:

        return {
            "rows": 0,
            "correct": 0,
            "accuracy": np.nan,
            "false_yes": 0,
            "false_no": 0,
        }

    predicted_yes = (
        dataframe[
            proxy_column
        ]
        >=
        dataframe[
            "floor_strike"
        ]
    )

    actual_yes = (
        dataframe[
            "actual_yes"
        ]
    )

    correct = (
        predicted_yes
        ==
        actual_yes
    )

    false_yes = (
        predicted_yes
        &
        ~actual_yes
    )

    false_no = (
        ~predicted_yes
        &
        actual_yes
    )

    return {
        "rows":
            len(
                dataframe
            ),

        "correct":
            int(
                correct.sum()
            ),

        "accuracy":
            float(
                correct.mean()
            ),

        "false_yes":
            int(
                false_yes.sum()
            ),

        "false_no":
            int(
                false_no.sum()
            ),
    }


# ============================================================
# PERIOD COMPARISON
# ============================================================

def period_tracking(
    dataframe,
):

    periods = [
        (
            "2025-12-10 -> 2026-08-07",
            pd.Timestamp(
                "2025-12-10T00:00:00Z"
            ),
            pd.Timestamp(
                "2026-08-08T00:00:00Z"
            ),
        ),
        (
            "2026-08-08 -> 2026-10-06",
            pd.Timestamp(
                "2026-08-08T00:00:00Z"
            ),
            pd.Timestamp(
                "2026-10-07T00:00:00Z"
            ),
        ),
    ]

    rows = []

    for (
        period_name,
        start,
        end,
    ) in periods:

        subset = (
            dataframe[
                (
                    dataframe[
                        "close_time"
                    ]
                    >=
                    start
                )
                &
                (
                    dataframe[
                        "close_time"
                    ]
                    <
                    end
                )
            ]
            .copy()
        )

        for (
            proxy_name,
            proxy_column,
        ) in PROXY_COLUMNS.items():

            metrics = (
                calculate_tracking_metrics(
                    subset,
                    proxy_column,
                )
            )

            rows.append(
                {
                    "period":
                        period_name,

                    "proxy":
                        proxy_name,

                    "rows":
                        metrics[
                            "rows"
                        ],

                    "mae_dollars":
                        metrics[
                            "mae_dollars"
                        ],

                    "mae_bps":
                        metrics[
                            "mae_bps"
                        ],

                    "rmse_bps":
                        metrics[
                            "rmse_bps"
                        ],

                    "p95_ae_bps":
                        metrics[
                            "p95_ae_bps"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT HELPERS
# ============================================================

TRACKING_FORMATTERS = {

    "mean_error_dollars":
        lambda x:
            f"${x:+.2f}",

    "mae_dollars":
        lambda x:
            f"${x:.2f}",

    "median_ae_dollars":
        lambda x:
            f"${x:.2f}",

    "rmse_dollars":
        lambda x:
            f"${x:.2f}",

    "p95_ae_dollars":
        lambda x:
            f"${x:.2f}",

    "mean_error_bps":
        lambda x:
            f"{x:+.3f}",

    "mae_bps":
        lambda x:
            f"{x:.3f}",

    "median_ae_bps":
        lambda x:
            f"{x:.3f}",

    "rmse_bps":
        lambda x:
            f"{x:.3f}",

    "p95_ae_bps":
        lambda x:
            f"{x:.3f}",
}


DIRECTION_FORMATTERS = {

    "accuracy":
        lambda x:
            (
                "n/a"
                if pd.isna(
                    x
                )
                else
                f"{x:.2%}"
            ),
}


PERIOD_FORMATTERS = {

    "mae_dollars":
        lambda x:
            f"${x:.2f}",

    "mae_bps":
        lambda x:
            f"{x:.3f}",

    "rmse_bps":
        lambda x:
            f"{x:.3f}",

    "p95_ae_bps":
        lambda x:
            f"{x:.3f}",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 155
    )

    print(
        "KXBTC15M — CROSS-EXCHANGE BRTI PROXY DIAGNOSTIC"
    )

    print(
        "=" * 155
    )

    print()

    print(
        "Important:"
    )

    print(
        "These are exchange 1-minute CLOSING prices."
    )

    print(
        "They are proxies for BRTI, not reconstructions "
        "of the BRTI methodology."
    )

    dataframe = load_dataset()

    print()

    print(
        f"Synchronized settlements: "
        f"{len(dataframe)}"
    )

    print(
        f"Range: "
        f"{dataframe['close_time'].min()} "
        f"-> "
        f"{dataframe['close_time'].max()}"
    )

    print()

    source_counts = (
        dataframe[
            "expiration_value_source"
        ]
        .fillna(
            "NULL"
        )
        .value_counts(
            dropna=False
        )
    )

    print(
        "Expiration value sources:"
    )

    print(
        source_counts.to_string()
    )

    # --------------------------------------------------------
    # OVERALL TRACKING
    # --------------------------------------------------------

    tracking_rows = []

    for (
        proxy_name,
        proxy_column,
    ) in PROXY_COLUMNS.items():

        metrics = (
            calculate_tracking_metrics(
                dataframe,
                proxy_column,
            )
        )

        tracking_rows.append(
            {
                "proxy":
                    proxy_name,

                **metrics,
            }
        )

    tracking = pd.DataFrame(
        tracking_rows
    )

    print()
    print(
        "=" * 180
    )

    print(
        "OVERALL BRTI TRACKING ERROR"
    )

    print(
        "=" * 180
    )

    print(
        tracking.to_string(
            index=False,
            formatters=
                TRACKING_FORMATTERS,
        )
    )

    # --------------------------------------------------------
    # STRIKE DATA
    # --------------------------------------------------------

    strike_data = (
        build_strike_dataset(
            dataframe
        )
    )

    print()

    print(
        f"Valid strike rows after data-quality filter: "
        f"{len(strike_data)}"
    )

    # --------------------------------------------------------
    # DIRECTIONAL ACCURACY — ALL
    # --------------------------------------------------------

    directional_rows = []

    for (
        proxy_name,
        proxy_column,
    ) in PROXY_COLUMNS.items():

        metrics = (
            calculate_directional_metrics(
                strike_data,
                proxy_column,
            )
        )

        directional_rows.append(
            {
                "proxy":
                    proxy_name,

                **metrics,
            }
        )

    directional = pd.DataFrame(
        directional_rows
    )

    print()
    print(
        "=" * 130
    )

    print(
        "SETTLEMENT-DIRECTION ACCURACY — ALL VALID STRIKES"
    )

    print(
        "=" * 130
    )

    print(
        directional.to_string(
            index=False,
            formatters=
                DIRECTION_FORMATTERS,
        )
    )

    # --------------------------------------------------------
    # NEAR-STRIKE ACCURACY
    # --------------------------------------------------------

    near_rows = []

    for threshold in NEAR_STRIKE_BPS:

        subset = (
            strike_data[
                strike_data[
                    "actual_distance_bps"
                ]
                .abs()
                <=
                threshold
            ]
            .copy()
        )

        for (
            proxy_name,
            proxy_column,
        ) in PROXY_COLUMNS.items():

            metrics = (
                calculate_directional_metrics(
                    subset,
                    proxy_column,
                )
            )

            near_rows.append(
                {
                    "within_bps":
                        threshold,

                    "proxy":
                        proxy_name,

                    **metrics,
                }
            )

    near = pd.DataFrame(
        near_rows
    )

    print()
    print(
        "=" * 145
    )

    print(
        "DIRECTIONAL ACCURACY WHEN ACTUAL BRTI SETTLEMENT IS NEAR STRIKE"
    )

    print(
        "=" * 145
    )

    print(
        near.to_string(
            index=False,
            formatters={
                "within_bps":
                    lambda x:
                        f"{x:.1f}",

                **DIRECTION_FORMATTERS,
            },
        )
    )

    # --------------------------------------------------------
    # REGIME / PERIOD STABILITY
    # --------------------------------------------------------

    periods = (
        period_tracking(
            dataframe
        )
    )

    print()
    print(
        "=" * 145
    )

    print(
        "TRACKING ERROR BY HISTORICAL PERIOD"
    )

    print(
        "=" * 145
    )

    print(
        periods.to_string(
            index=False,
            formatters=
                PERIOD_FORMATTERS,
        )
    )


if __name__ == "__main__":

    main()