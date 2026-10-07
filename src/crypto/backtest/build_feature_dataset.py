import math

import numpy as np
import pandas as pd

from src.database.db import (
    get_connection,
)


# ============================================================
# CONFIGURATION
# ============================================================

SERIES_TICKER = "KXBTC15M"

HORIZON_MINUTES = 5

PRIOR_KALSHI_HORIZON = 10

SPOT_SOURCE = "COINBASE_EXCHANGE"

SPOT_PRODUCT = "BTC-USD"

FEATURE_VERSION = (
    "KXBTC15M_5M_COINBASE_V2"
)

MAX_ABS_OPEN_BASIS_PCT = 0.05

REQUIRED_SPOT_HISTORY_MINUTES = 60

RETURN_WINDOWS = (
    1,
    3,
    5,
    10,
    15,
)

VOLATILITY_WINDOWS = (
    5,
    15,
    30,
    60,
)

VOLUME_WINDOWS = (
    5,
    15,
    60,
)


# ============================================================
# TIME
# ============================================================

def utc_timestamp(
    value,
):

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp


# ============================================================
# FEATURE TABLE
# ============================================================

def ensure_feature_schema():

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
                crypto_feature_rows (

                feature_version TEXT NOT NULL,

                ticker TEXT NOT NULL,

                horizon_minutes INTEGER NOT NULL,

                quote_time TIMESTAMPTZ NOT NULL,

                open_time TIMESTAMPTZ NOT NULL,

                close_time TIMESTAMPTZ NOT NULL,

                floor_strike DOUBLE PRECISION NOT NULL,

                kalshi_yes_bid DOUBLE PRECISION NOT NULL,
                kalshi_yes_ask DOUBLE PRECISION NOT NULL,

                kalshi_no_bid DOUBLE PRECISION NOT NULL,
                kalshi_no_ask DOUBLE PRECISION NOT NULL,

                kalshi_midpoint DOUBLE PRECISION NOT NULL,
                kalshi_spread DOUBLE PRECISION NOT NULL,

                kalshi_midpoint_10m DOUBLE PRECISION NOT NULL,

                kalshi_move_10m_to_5m
                    DOUBLE PRECISION NOT NULL,

                spot_close DOUBLE PRECISION NOT NULL,

                spot_at_open DOUBLE PRECISION NOT NULL,

                distance_to_strike_dollars
                    DOUBLE PRECISION NOT NULL,

                distance_to_strike_pct
                    DOUBLE PRECISION NOT NULL,

                basis_at_open_dollars
                    DOUBLE PRECISION NOT NULL,

                basis_at_open_pct
                    DOUBLE PRECISION NOT NULL,

                return_1m DOUBLE PRECISION NOT NULL,
                return_3m DOUBLE PRECISION NOT NULL,
                return_5m DOUBLE PRECISION NOT NULL,
                return_10m DOUBLE PRECISION NOT NULL,
                return_15m DOUBLE PRECISION NOT NULL,

                realized_vol_5m
                    DOUBLE PRECISION NOT NULL,

                realized_vol_15m
                    DOUBLE PRECISION NOT NULL,

                realized_vol_30m
                    DOUBLE PRECISION NOT NULL,

                realized_vol_60m
                    DOUBLE PRECISION NOT NULL,

                volume_5m DOUBLE PRECISION NOT NULL,
                volume_15m DOUBLE PRECISION NOT NULL,
                volume_60m DOUBLE PRECISION NOT NULL,

                outcome INTEGER NOT NULL,

                created_at TIMESTAMPTZ NOT NULL,

                PRIMARY KEY (
                    feature_version,
                    ticker,
                    horizon_minutes
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_crypto_features_quote_time

            ON crypto_feature_rows (
                quote_time
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_crypto_features_version_time

            ON crypto_feature_rows (
                feature_version,
                quote_time
            )
            """
        )


# ============================================================
# LOAD KALSHI DATA
# ============================================================

def load_kalshi_rows():

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT

                m.ticker,

                m.open_time,
                m.close_time,

                m.strike_type,
                m.floor_strike,
                m.cap_strike,

                mm.quote_time,

                mm.yes_bid,
                mm.yes_ask,

                mm.no_bid,
                mm.no_ask,

                mm.midpoint,
                mm.spread,

                mm.outcome,

                mm10.quote_time,
                mm10.midpoint

            FROM crypto_market_minutes mm

            JOIN crypto_markets m
                ON m.ticker = mm.ticker

            LEFT JOIN crypto_market_minutes mm10

                ON mm10.ticker = mm.ticker

                AND mm10.actionable = 1

                AND ABS(
                    mm10.minutes_remaining
                    -
                    ?
                ) < 0.000001

            WHERE

                m.series_ticker = ?

                AND m.backfill_status = 'COLLECTED'

                AND mm.actionable = 1

                AND ABS(
                    mm.minutes_remaining
                    -
                    ?
                ) < 0.000001

                AND mm.outcome IS NOT NULL

            ORDER BY
                m.close_time,
                m.ticker
            """,
            (
                float(
                    PRIOR_KALSHI_HORIZON
                ),

                SERIES_TICKER,

                float(
                    HORIZON_MINUTES
                ),
            ),
        )

        rows = cursor.fetchall()

    columns = [
        "ticker",

        "open_time",
        "close_time",

        "strike_type",
        "floor_strike",
        "cap_strike",

        "quote_time",

        "yes_bid",
        "yes_ask",

        "no_bid",
        "no_ask",

        "midpoint",
        "spread",

        "outcome",

        "quote_time_10m",
        "midpoint_10m",
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    if dataframe.empty:

        raise RuntimeError(
            "No Kalshi 5-minute rows found."
        )

    for column in (
        "open_time",
        "close_time",
        "quote_time",
        "quote_time_10m",
    ):

        dataframe[
            column
        ] = pd.to_datetime(
            dataframe[
                column
            ],
            utc=True,
            errors="coerce",
            format="mixed",
        )

    for column in (
        "floor_strike",
        "cap_strike",

        "yes_bid",
        "yes_ask",

        "no_bid",
        "no_ask",

        "midpoint",
        "spread",

        "midpoint_10m",
        "outcome",
    ):

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
# LOAD BTC SPOT
# ============================================================

def load_spot_rows(
    earliest_quote,
    latest_quote,
):

    start_time = (
        utc_timestamp(
            earliest_quote
        )
        -
        pd.Timedelta(
            minutes=
                REQUIRED_SPOT_HISTORY_MINUTES
                +
                5
        )
    )

    end_time = (
        utc_timestamp(
            latest_quote
        )
        +
        pd.Timedelta(
            minutes=1
        )
    )

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT

                candle_start,
                candle_end,

                open,
                high,
                low,
                close,

                volume

            FROM btc_spot_minutes

            WHERE

                source = ?

                AND product_id = ?

                AND candle_end >= ?

                AND candle_end <= ?

            ORDER BY
                candle_end
            """,
            (
                SPOT_SOURCE,
                SPOT_PRODUCT,

                start_time.isoformat(),
                end_time.isoformat(),
            ),
        )

        rows = cursor.fetchall()

    columns = [
        "candle_start",
        "candle_end",

        "open",
        "high",
        "low",
        "close",

        "volume",
    ]

    dataframe = pd.DataFrame(
        rows,
        columns=columns,
    )

    if dataframe.empty:

        raise RuntimeError(
            "No BTC spot rows found."
        )

    dataframe[
        "candle_start"
    ] = pd.to_datetime(
        dataframe[
            "candle_start"
        ],
        utc=True,
        errors="raise",
        format="mixed",
    )

    dataframe[
        "candle_end"
    ] = pd.to_datetime(
        dataframe[
            "candle_end"
        ],
        utc=True,
        errors="raise",
        format="mixed",
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="raise",
        )

    return dataframe


# ============================================================
# SPOT LOOKUP
# ============================================================

def build_spot_lookup(
    spot_data,
):

    lookup = {}

    for row in spot_data.itertuples(
        index=False
    ):

        candle_end = utc_timestamp(
            row.candle_end
        )

        if candle_end in lookup:

            raise RuntimeError(
                "Duplicate BTC candle_end "
                f"detected: {candle_end}"
            )

        lookup[
            candle_end
        ] = {
            "candle_start":
                utc_timestamp(
                    row.candle_start
                ),

            "candle_end":
                candle_end,

            "open":
                float(
                    row.open
                ),

            "high":
                float(
                    row.high
                ),

            "low":
                float(
                    row.low
                ),

            "close":
                float(
                    row.close
                ),

            "volume":
                float(
                    row.volume
                ),
        }

    return lookup


# ============================================================
# GET COMPLETE SPOT HISTORY
# ============================================================

def get_spot_history(
    lookup,
    quote_time,
):

    quote_time = utc_timestamp(
        quote_time
    )

    timestamps = [
        (
            quote_time
            -
            pd.Timedelta(
                minutes=offset
            )
        )

        for offset in range(
            REQUIRED_SPOT_HISTORY_MINUTES,
            -1,
            -1,
        )
    ]

    rows = []

    for timestamp in timestamps:

        row = lookup.get(
            timestamp
        )

        if row is None:

            return None

        # Strict anti-lookahead rule.
        #
        # The candle must be fully complete
        # by the Kalshi decision timestamp.
        if (
            row[
                "candle_end"
            ]
            >
            quote_time
        ):

            raise RuntimeError(
                "Future BTC candle entered "
                "feature history."
            )

        rows.append(
            row
        )

    return rows


# ============================================================
# FEATURE CALCULATIONS
# ============================================================

def log_return(
    current_price,
    prior_price,
):

    if (
        current_price <= 0
        or prior_price <= 0
    ):

        raise ValueError(
            "BTC prices must be positive."
        )

    return float(
        math.log(
            current_price
            /
            prior_price
        )
    )


def realized_volatility(
    one_minute_returns,
    window,
):

    returns = np.asarray(
        one_minute_returns[
            -window:
        ],
        dtype=float,
    )

    if len(
        returns
    ) != window:

        raise ValueError(
            "Not enough returns for "
            f"{window}-minute volatility."
        )

    # Non-annualized realized volatility
    # over the requested interval.
    return float(
        math.sqrt(
            float(
                np.sum(
                    returns ** 2
                )
            )
        )
    )


def sum_volume(
    history,
    window,
):

    rows = history[
        -window:
    ]

    if len(
        rows
    ) != window:

        raise ValueError(
            "Not enough candles for "
            f"{window}-minute volume."
        )

    return float(
        sum(
            row[
                "volume"
            ]
            for row in rows
        )
    )


# ============================================================
# BUILD ONE FEATURE ROW
# ============================================================

def build_feature_row(
    market_row,
    spot_lookup,
):

    quote_time = utc_timestamp(
        market_row.quote_time
    )

    open_time = utc_timestamp(
        market_row.open_time
    )

    close_time = utc_timestamp(
        market_row.close_time
    )

    # --------------------------------------------------------
    # Market structure validation
    # --------------------------------------------------------

    strike_type = (
        str(
            market_row.strike_type
        )
        .strip()
        .lower()
    )

    if (
        strike_type
        in (
            "",
            "nan",
            "none",
        )
        or pd.isna(
            market_row.floor_strike
        )
    ):

        return (
            None,
            "MISSING_STRIKE_METADATA",
        )

    if (
        strike_type
        !=
        "greater_or_equal"
    ):

        return (
            None,
            "BAD_STRIKE_TYPE",
        )

    strike = float(
        market_row.floor_strike
    )

    if (
        not math.isfinite(
            strike
        )
        or strike <= 0
    ):

        return (
            None,
            "INVALID_STRIKE",
        )

    if (
        market_row.cap_strike
        is not None
        and
        not pd.isna(
            market_row.cap_strike
        )
    ):

        return (
            None,
            "UNEXPECTED_CAP_STRIKE",
        )

    # --------------------------------------------------------
    # Decision-time validation
    # --------------------------------------------------------

    expected_quote_time = (
        close_time
        -
        pd.Timedelta(
            minutes=
                HORIZON_MINUTES
        )
    )

    if (
        quote_time
        !=
        expected_quote_time
    ):

        return (
            None,
            "BAD_5M_ALIGNMENT",
        )

    expected_open_time = (
        close_time
        -
        pd.Timedelta(
            minutes=15
        )
    )

    if (
        open_time
        !=
        expected_open_time
    ):

        return (
            None,
            "BAD_MARKET_OPEN_ALIGNMENT",
        )

    # --------------------------------------------------------
    # Earlier Kalshi quote
    # --------------------------------------------------------

    if (
        pd.isna(
            market_row.quote_time_10m
        )
        or
        pd.isna(
            market_row.midpoint_10m
        )
    ):

        return (
            None,
            "MISSING_10M_KALSHI",
        )

    quote_time_10m = (
        utc_timestamp(
            market_row.quote_time_10m
        )
    )

    expected_10m_time = (
        close_time
        -
        pd.Timedelta(
            minutes=
                PRIOR_KALSHI_HORIZON
        )
    )

    if (
        quote_time_10m
        !=
        expected_10m_time
    ):

        return (
            None,
            "BAD_10M_ALIGNMENT",
        )

    # --------------------------------------------------------
    # Complete BTC history
    # --------------------------------------------------------

    history = (
        get_spot_history(
            lookup=
                spot_lookup,

            quote_time=
                quote_time,
        )
    )

    if history is None:

        return (
            None,
            "MISSING_SPOT_HISTORY",
        )

    # 61 closing prices:
    #
    # T-60, T-59, ... T-1, T
    closes = np.asarray(
        [
            row[
                "close"
            ]
            for row in history
        ],
        dtype=float,
    )

    if len(
        closes
    ) != 61:

        raise RuntimeError(
            "Expected exactly 61 BTC closes."
        )

    if (
        not np.all(
            np.isfinite(
                closes
            )
        )
        or np.any(
            closes <= 0
        )
    ):

        return (
            None,
            "INVALID_SPOT_PRICE",
        )

    current_spot = float(
        closes[
            -1
        ]
    )

    # --------------------------------------------------------
    # Spot at Kalshi contract open
    #
    # At a 5-minute decision in a 15-minute
    # market, open occurred 10 minutes earlier.
    # --------------------------------------------------------

    open_offset = int(
        (
            quote_time
            -
            open_time
        ).total_seconds()
        /
        60
    )

    if open_offset != 10:

        return (
            None,
            "BAD_OPEN_OFFSET",
        )

    spot_at_open_row = (
        spot_lookup.get(
            open_time
        )
    )

    if spot_at_open_row is None:

        return (
            None,
            "MISSING_SPOT_AT_OPEN",
        )

    if (
        spot_at_open_row[
            "candle_end"
        ]
        >
        quote_time
    ):

        raise RuntimeError(
            "Future open-time BTC candle."
        )

    spot_at_open = float(
        spot_at_open_row[
            "close"
        ]
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    feature_returns = {}

    for window in RETURN_WINDOWS:

        feature_returns[
            window
        ] = (
            log_return(
                current_price=
                    closes[
                        -1
                    ],

                prior_price=
                    closes[
                        -(
                            window
                            +
                            1
                        )
                    ],
            )
        )

    # --------------------------------------------------------
    # One-minute return series
    # --------------------------------------------------------

    one_minute_returns = (
        np.diff(
            np.log(
                closes
            )
        )
    )

    if len(
        one_minute_returns
    ) != 60:

        raise RuntimeError(
            "Expected 60 one-minute BTC returns."
        )

    # --------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------

    volatility = {}

    for window in VOLATILITY_WINDOWS:

        volatility[
            window
        ] = (
            realized_volatility(
                one_minute_returns=
                    one_minute_returns,

                window=
                    window,
            )
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volumes = {}

    for window in VOLUME_WINDOWS:

        volumes[
            window
        ] = (
            sum_volume(
                history=
                    history,

                window=
                    window,
            )
        )

    # --------------------------------------------------------
    # Distance to Kalshi strike
    #
    # IMPORTANT:
    #
    # Coinbase BTC-USD is an external proxy.
    # floor_strike belongs to Kalshi's
    # settlement/index framework.
    #
    # We intentionally keep both raw distance
    # and the opening basis as separate features.
    # --------------------------------------------------------

    distance_dollars = (
        current_spot
        -
        strike
    )

    distance_pct = (
        distance_dollars
        /
        strike
    )

    basis_at_open_dollars = (
        spot_at_open
        -
        strike
    )

    basis_at_open_pct = (
        basis_at_open_dollars
        /
        strike
    )

    if (
        abs(
            basis_at_open_pct
        )
        >
        MAX_ABS_OPEN_BASIS_PCT
    ):

        return (
            None,
            "IMPLAUSIBLE_OPEN_BASIS",
        )

    # --------------------------------------------------------
    # Kalshi movement
    # --------------------------------------------------------

    midpoint = float(
        market_row.midpoint
    )

    midpoint_10m = float(
        market_row.midpoint_10m
    )

    kalshi_move = (
        midpoint
        -
        midpoint_10m
    )

    # --------------------------------------------------------
    # Sanity
    # --------------------------------------------------------

    numeric_values = [
        midpoint,
        midpoint_10m,

        current_spot,
        spot_at_open,

        distance_dollars,
        distance_pct,

        basis_at_open_dollars,
        basis_at_open_pct,

        kalshi_move,

        *feature_returns.values(),
        *volatility.values(),
        *volumes.values(),
    ]

    if not all(
        math.isfinite(
            float(
                value
            )
        )
        for value in numeric_values
    ):

        return (
            None,
            "NONFINITE_FEATURE",
        )

    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    return (
        {
            "feature_version":
                FEATURE_VERSION,

            "ticker":
                str(
                    market_row.ticker
                ),

            "horizon_minutes":
                HORIZON_MINUTES,

            "quote_time":
                quote_time.isoformat(),

            "open_time":
                open_time.isoformat(),

            "close_time":
                close_time.isoformat(),

            "floor_strike":
                strike,

            "kalshi_yes_bid":
                float(
                    market_row.yes_bid
                ),

            "kalshi_yes_ask":
                float(
                    market_row.yes_ask
                ),

            "kalshi_no_bid":
                float(
                    market_row.no_bid
                ),

            "kalshi_no_ask":
                float(
                    market_row.no_ask
                ),

            "kalshi_midpoint":
                midpoint,

            "kalshi_spread":
                float(
                    market_row.spread
                ),

            "kalshi_midpoint_10m":
                midpoint_10m,

            "kalshi_move_10m_to_5m":
                kalshi_move,

            "spot_close":
                current_spot,

            "spot_at_open":
                spot_at_open,

            "distance_to_strike_dollars":
                distance_dollars,

            "distance_to_strike_pct":
                distance_pct,

            "basis_at_open_dollars":
                basis_at_open_dollars,

            "basis_at_open_pct":
                basis_at_open_pct,

            "return_1m":
                feature_returns[
                    1
                ],

            "return_3m":
                feature_returns[
                    3
                ],

            "return_5m":
                feature_returns[
                    5
                ],

            "return_10m":
                feature_returns[
                    10
                ],

            "return_15m":
                feature_returns[
                    15
                ],

            "realized_vol_5m":
                volatility[
                    5
                ],

            "realized_vol_15m":
                volatility[
                    15
                ],

            "realized_vol_30m":
                volatility[
                    30
                ],

            "realized_vol_60m":
                volatility[
                    60
                ],

            "volume_5m":
                volumes[
                    5
                ],

            "volume_15m":
                volumes[
                    15
                ],

            "volume_60m":
                volumes[
                    60
                ],

            "outcome":
                int(
                    market_row.outcome
                ),

            "created_at":
                created_at,
        },
        None,
    )


# ============================================================
# SAVE
# ============================================================

FEATURE_COLUMNS = [
    "feature_version",

    "ticker",

    "horizon_minutes",

    "quote_time",

    "open_time",

    "close_time",

    "floor_strike",

    "kalshi_yes_bid",
    "kalshi_yes_ask",

    "kalshi_no_bid",
    "kalshi_no_ask",

    "kalshi_midpoint",
    "kalshi_spread",

    "kalshi_midpoint_10m",

    "kalshi_move_10m_to_5m",

    "spot_close",

    "spot_at_open",

    "distance_to_strike_dollars",

    "distance_to_strike_pct",

    "basis_at_open_dollars",

    "basis_at_open_pct",

    "return_1m",
    "return_3m",
    "return_5m",
    "return_10m",
    "return_15m",

    "realized_vol_5m",
    "realized_vol_15m",
    "realized_vol_30m",
    "realized_vol_60m",

    "volume_5m",
    "volume_15m",
    "volume_60m",

    "outcome",

    "created_at",
]


def save_feature_rows(
    rows,
    chunk_size=500,
):

    if not rows:

        return 0

    saved = 0

    with get_connection() as connection:

        for start_index in range(
            0,
            len(rows),
            chunk_size,
        ):

            batch = rows[
                start_index:
                start_index
                +
                chunk_size
            ]

            one_row = (
                "("
                +
                ", ".join(
                    "?"
                    for _ in
                    FEATURE_COLUMNS
                )
                +
                ")"
            )

            values_sql = (
                ", ".join(
                    one_row
                    for _ in batch
                )
            )

            parameters = []

            for row in batch:

                parameters.extend(
                    row[
                        column
                    ]
                    for column in
                    FEATURE_COLUMNS
                )

            update_columns = [
                column
                for column in
                FEATURE_COLUMNS
                if column
                not in (
                    "feature_version",
                    "ticker",
                    "horizon_minutes",
                )
            ]

            assignments = (
                ", ".join(
                    f"{column} = "
                    f"excluded.{column}"

                    for column in
                    update_columns
                )
            )

            connection.execute(
                f"""
                INSERT INTO crypto_feature_rows (
                    {", ".join(FEATURE_COLUMNS)}
                )

                VALUES
                    {values_sql}

                ON CONFLICT (
                    feature_version,
                    ticker,
                    horizon_minutes
                )

                DO UPDATE SET
                    {assignments}
                """,
                tuple(
                    parameters
                ),
            )

            saved += len(
                batch
            )

    return saved


# ============================================================
# AUDIT
# ============================================================

def print_feature_audit(
    feature_rows,
    rejections,
):

    dataframe = pd.DataFrame(
        feature_rows
    )

    print()
    print("=" * 100)
    print(
        "5-MINUTE BTC FEATURE DATASET"
    )
    print("=" * 100)

    print(
        f"Feature version:          "
        f"{FEATURE_VERSION}"
    )

    print(
        f"Built rows:               "
        f"{len(dataframe)}"
    )

    print(
        f"Rejected rows:            "
        f"{sum(rejections.values())}"
    )

    print()

    print(
        "Rejection reasons:"
    )

    if not rejections:

        print(
            "  none"
        )

    else:

        for (
            reason,
            count,
        ) in sorted(
            rejections.items(),
            key=lambda item:
                (
                    -item[
                        1
                    ],
                    item[
                        0
                    ],
                ),
        ):

            print(
                f"  {reason:<28s} "
                f"{count}"
            )

    if dataframe.empty:

        return

    dataframe[
        "quote_time"
    ] = pd.to_datetime(
        dataframe[
            "quote_time"
        ],
        utc=True,
    )

    print()

    print(
        f"Earliest feature row:     "
        f"{dataframe['quote_time'].min()}"
    )

    print(
        f"Latest feature row:       "
        f"{dataframe['quote_time'].max()}"
    )

    print(
        f"YES outcomes:             "
        f"{int(dataframe['outcome'].sum())}"
    )

    print(
        f"NO outcomes:              "
        f"{len(dataframe) - int(dataframe['outcome'].sum())}"
    )

    print()

    print(
        "FEATURE SUMMARY"
    )

    summary_columns = [
        "kalshi_midpoint",
        "kalshi_spread",

        "distance_to_strike_dollars",

        "distance_to_strike_pct",

        "basis_at_open_dollars",
        "basis_at_open_pct",

        "kalshi_move_10m_to_5m",

        "return_1m",
        "return_5m",
        "return_10m",

        "realized_vol_5m",
        "realized_vol_15m",
        "realized_vol_60m",

        "volume_5m",
        "volume_15m",
        "volume_60m",
    ]

    summary = (
        dataframe[
            summary_columns
        ]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .T
    )

    print(
        summary.to_string()
    )


# ============================================================
# DATABASE READ-BACK
# ============================================================

def print_database_counts():

    with get_connection() as connection:

        row = (
            connection.execute(
                """
                SELECT

                    COUNT(*),

                    MIN(quote_time),

                    MAX(quote_time)

                FROM crypto_feature_rows

                WHERE
                    feature_version = ?

                    AND horizon_minutes = ?
                """,
                (
                    FEATURE_VERSION,

                    HORIZON_MINUTES,
                ),
            )
            .fetchone()
        )

    print()
    print(
        "Database read-back:"
    )

    print(
        {
            "rows":
                int(
                    row[
                        0
                    ]
                    or 0
                ),

            "earliest":
                row[
                    1
                ],

            "latest":
                row[
                    2
                ],
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_feature_schema()

    kalshi_data = (
        load_kalshi_rows()
    )

    print()
    print("=" * 100)
    print(
        "BUILDING BTC/KALSHI FEATURE DATASET"
    )
    print("=" * 100)

    print(
        f"Kalshi candidate rows:    "
        f"{len(kalshi_data)}"
    )

    print(
        f"Decision horizon:         "
        f"{HORIZON_MINUTES} minutes"
    )

    print(
        f"Prior Kalshi horizon:     "
        f"{PRIOR_KALSHI_HORIZON} minutes"
    )

    print(
        f"Required BTC history:     "
        f"{REQUIRED_SPOT_HISTORY_MINUTES} minutes"
    )

    earliest_quote = (
        kalshi_data[
            "quote_time"
        ].min()
    )

    latest_quote = (
        kalshi_data[
            "quote_time"
        ].max()
    )

    spot_data = (
        load_spot_rows(
            earliest_quote=
                earliest_quote,

            latest_quote=
                latest_quote,
        )
    )

    print(
        f"BTC minute rows loaded:   "
        f"{len(spot_data)}"
    )

    spot_lookup = (
        build_spot_lookup(
            spot_data
        )
    )

    feature_rows = []

    rejections = {}

    for market_row in (
        kalshi_data.itertuples(
            index=False
        )
    ):

        (
            feature_row,
            rejection_reason,
        ) = (
            build_feature_row(
                market_row=
                    market_row,

                spot_lookup=
                    spot_lookup,
            )
        )

        if feature_row is None:

            rejections[
                rejection_reason
            ] = (
                rejections.get(
                    rejection_reason,
                    0,
                )
                +
                1
            )

            continue

        feature_rows.append(
            feature_row
        )

    print_feature_audit(
        feature_rows=
            feature_rows,

        rejections=
            rejections,
    )

    saved = (
        save_feature_rows(
            feature_rows
        )
    )

    print()
    print(
        f"Saved/upserted rows:       "
        f"{saved}"
    )

    print_database_counts()


if __name__ == "__main__":

    main()