import os

import pandas as pd

from src.database.db import (
    get_connection,
)


SOURCE = "COINBASE_EXCHANGE"
PRODUCT_ID = "BTC-USD"


# ============================================================
# ENVIRONMENT
# ============================================================

def require_supabase():

    if not os.environ.get(
        "PREDICTION_DATABASE_URL",
        "",
    ).strip():

        raise RuntimeError(
            "PREDICTION_DATABASE_URL is missing; "
            "Supabase is required for BTC spot storage."
        )


# ============================================================
# SCHEMA
# ============================================================

def ensure_btc_spot_schema():

    require_supabase()

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_spot_minutes (

                source TEXT NOT NULL,
                product_id TEXT NOT NULL,

                candle_start TIMESTAMPTZ NOT NULL,
                candle_end TIMESTAMPTZ NOT NULL,

                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,

                volume DOUBLE PRECISION NOT NULL,

                fetched_at TIMESTAMPTZ NOT NULL,

                PRIMARY KEY (
                    source,
                    product_id,
                    candle_start
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_btc_spot_minutes_time

            ON btc_spot_minutes (
                candle_start
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_btc_spot_minutes_product_time

            ON btc_spot_minutes (
                source,
                product_id,
                candle_start
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_spot_backfill_days (

                source TEXT NOT NULL,
                product_id TEXT NOT NULL,

                utc_date DATE NOT NULL,

                status TEXT NOT NULL,

                row_count INTEGER NOT NULL,
                missing_count INTEGER NOT NULL,

                fetched_at TIMESTAMPTZ NOT NULL,

                error_message TEXT,

                PRIMARY KEY (
                    source,
                    product_id,
                    utc_date
                )
            )
            """
        )


# ============================================================
# BULK UPSERT
# ============================================================

def _bulk_upsert_spot_candles(
    connection,
    rows,
    chunk_size=1000,
):

    if not rows:

        return 0

    columns = [
        "source",
        "product_id",
        "candle_start",
        "candle_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "fetched_at",
    ]

    total_saved = 0

    for start_index in range(
        0,
        len(rows),
        chunk_size,
    ):

        batch = rows[
            start_index:
            start_index + chunk_size
        ]

        one_row_placeholders = (
            "("
            +
            ", ".join(
                "?"
                for _ in columns
            )
            +
            ")"
        )

        values_sql = ", ".join(
            one_row_placeholders
            for _ in batch
        )

        parameters = []

        for row in batch:

            parameters.extend(
                row[column]
                for column in columns
            )

        connection.execute(
            f"""
            INSERT INTO btc_spot_minutes (
                {", ".join(columns)}
            )

            VALUES
                {values_sql}

            ON CONFLICT (
                source,
                product_id,
                candle_start
            )

            DO UPDATE SET

                candle_end =
                    excluded.candle_end,

                open =
                    excluded.open,

                high =
                    excluded.high,

                low =
                    excluded.low,

                close =
                    excluded.close,

                volume =
                    excluded.volume,

                fetched_at =
                    excluded.fetched_at
            """,
            tuple(
                parameters
            ),
        )

        total_saved += len(
            batch
        )

    return total_saved


# ============================================================
# SAVE COMPLETE UTC DAY
# ============================================================

def save_spot_day(
    rows,
    utc_date,
    expected_minutes=1440,
):

    ensure_btc_spot_schema()

    utc_date = (
        pd.Timestamp(
            utc_date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    fetched_at = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    row_count = len(
        rows
    )

    missing_count = max(
        int(
            expected_minutes
        )
        -
        row_count,
        0,
    )

    with get_connection() as connection:

        saved = (
            _bulk_upsert_spot_candles(
                connection=
                    connection,

                rows=
                    rows,
            )
        )

        connection.execute(
            """
            INSERT INTO btc_spot_backfill_days (

                source,
                product_id,
                utc_date,

                status,

                row_count,
                missing_count,

                fetched_at,

                error_message
            )

            VALUES (
                ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )

            ON CONFLICT (
                source,
                product_id,
                utc_date
            )

            DO UPDATE SET

                status =
                    excluded.status,

                row_count =
                    excluded.row_count,

                missing_count =
                    excluded.missing_count,

                fetched_at =
                    excluded.fetched_at,

                error_message =
                    excluded.error_message
            """,
            (
                SOURCE,
                PRODUCT_ID,
                utc_date,

                "COLLECTED",

                row_count,
                missing_count,

                fetched_at,

                None,
            ),
        )

    return {
        "saved":
            saved,

        "row_count":
            row_count,

        "missing_count":
            missing_count,
    }


# ============================================================
# MARK ERROR
# ============================================================

def mark_spot_day_error(
    utc_date,
    error_message,
):

    ensure_btc_spot_schema()

    utc_date = (
        pd.Timestamp(
            utc_date
        )
        .strftime(
            "%Y-%m-%d"
        )
    )

    fetched_at = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO btc_spot_backfill_days (

                source,
                product_id,
                utc_date,

                status,

                row_count,
                missing_count,

                fetched_at,

                error_message
            )

            VALUES (
                ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )

            ON CONFLICT (
                source,
                product_id,
                utc_date
            )

            DO UPDATE SET

                status =
                    excluded.status,

                row_count =
                    excluded.row_count,

                missing_count =
                    excluded.missing_count,

                fetched_at =
                    excluded.fetched_at,

                error_message =
                    excluded.error_message
            """,
            (
                SOURCE,
                PRODUCT_ID,
                utc_date,

                "ERROR",

                0,
                1440,

                fetched_at,

                str(
                    error_message
                ),
            ),
        )


# ============================================================
# COMPLETED DAYS
# ============================================================

def get_completed_spot_days(
    require_complete=False,
):

    ensure_btc_spot_schema()

    query = """
        SELECT
            utc_date

        FROM btc_spot_backfill_days

        WHERE
            source = ?
            AND product_id = ?
            AND status = 'COLLECTED'
    """

    params = [
        SOURCE,
        PRODUCT_ID,
    ]

    if require_complete:

        query += """
            AND missing_count = 0
        """

    query += """
        ORDER BY utc_date
    """

    with get_connection() as connection:

        rows = (
            connection.execute(
                query,
                tuple(
                    params
                ),
            )
            .fetchall()
        )

    return {
        str(
            row[0]
        )
        for row in rows
    }


# ============================================================
# REQUIRED RANGE FROM KALSHI DATA
# ============================================================

def get_required_spot_range():

    require_supabase()

    with get_connection() as connection:

        row = (
            connection.execute(
                """
                SELECT
                    MIN(close_time),
                    MAX(close_time)

                FROM crypto_markets

                WHERE
                    series_ticker = 'KXBTC15M'
                    AND backfill_status = 'COLLECTED'
                """
            )
            .fetchone()
        )

    if (
        row is None
        or row[0] is None
        or row[1] is None
    ):

        raise RuntimeError(
            "No collected KXBTC15M markets "
            "were found."
        )

    earliest = pd.Timestamp(
        row[0]
    )

    latest = pd.Timestamp(
        row[1]
    )

    if earliest.tzinfo is None:

        earliest = (
            earliest.tz_localize(
                "UTC"
            )
        )

    else:

        earliest = (
            earliest.tz_convert(
                "UTC"
            )
        )

    if latest.tzinfo is None:

        latest = (
            latest.tz_localize(
                "UTC"
            )
        )

    else:

        latest = (
            latest.tz_convert(
                "UTC"
            )
        )

    return (
        earliest,
        latest,
    )


# ============================================================
# COUNTS
# ============================================================

def get_spot_counts():

    ensure_btc_spot_schema()

    with get_connection() as connection:

        minute_row = (
            connection.execute(
                """
                SELECT
                    COUNT(*),
                    MIN(candle_start),
                    MAX(candle_start)

                FROM btc_spot_minutes

                WHERE
                    source = ?
                    AND product_id = ?
                """,
                (
                    SOURCE,
                    PRODUCT_ID,
                ),
            )
            .fetchone()
        )

        day_row = (
            connection.execute(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE status = 'COLLECTED'
                    ),

                    COUNT(*) FILTER (
                        WHERE status = 'ERROR'
                    ),

                    COALESCE(
                        SUM(
                            CASE
                                WHEN status = 'COLLECTED'
                                THEN missing_count
                                ELSE 0
                            END
                        ),
                        0
                    )

                FROM btc_spot_backfill_days

                WHERE
                    source = ?
                    AND product_id = ?
                """,
                (
                    SOURCE,
                    PRODUCT_ID,
                ),
            )
            .fetchone()
        )

    return {
        "minute_rows":
            int(
                minute_row[0]
                or 0
            ),

        "earliest":
            minute_row[1],

        "latest":
            minute_row[2],

        "collected_days":
            int(
                day_row[0]
                or 0
            ),

        "error_days":
            int(
                day_row[1]
                or 0
            ),

        "missing_minutes":
            int(
                day_row[2]
                or 0
            ),
    }