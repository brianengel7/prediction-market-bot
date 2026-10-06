from src.database.db import (
    get_connection,
)


# ============================================================
# SCHEMA
# ============================================================

def ensure_crypto_schema():

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS crypto_markets (

                ticker TEXT PRIMARY KEY,

                event_ticker TEXT,
                series_ticker TEXT NOT NULL,

                title TEXT,
                subtitle TEXT,

                strike_type TEXT,
                floor_strike DOUBLE PRECISION,
                cap_strike DOUBLE PRECISION,

                open_time TEXT,
                close_time TEXT NOT NULL,
                settlement_time TEXT,

                market_status TEXT,
                result TEXT,
                outcome INTEGER,

                settlement_value DOUBLE PRECISION,

                market_source TEXT,

                candle_source TEXT,

                raw_candle_count INTEGER,
                valid_quote_count INTEGER,
                actionable_quote_count INTEGER,

                backfill_status TEXT NOT NULL,
                backfill_error TEXT,

                backfilled_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS crypto_market_minutes (

                ticker TEXT NOT NULL,

                quote_time TEXT NOT NULL,

                minutes_remaining DOUBLE PRECISION NOT NULL,

                actionable INTEGER NOT NULL,

                yes_bid DOUBLE PRECISION,
                yes_ask DOUBLE PRECISION,

                no_bid DOUBLE PRECISION,
                no_ask DOUBLE PRECISION,

                midpoint DOUBLE PRECISION,
                spread DOUBLE PRECISION,

                trade_price DOUBLE PRECISION,

                volume DOUBLE PRECISION,
                open_interest DOUBLE PRECISION,

                outcome INTEGER,

                candle_source TEXT,

                PRIMARY KEY (
                    ticker,
                    quote_time
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_crypto_minutes_remaining
            ON crypto_market_minutes (
                minutes_remaining
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_crypto_minutes_quote_time
            ON crypto_market_minutes (
                quote_time
            )
            """
        )


# ============================================================
# BULK UPSERT
# ============================================================

def bulk_upsert(
    connection,
    table,
    rows,
    conflict_columns,
):

    if not rows:

        return 0

    columns = list(
        rows[0]
    )

    for row in rows:

        if list(row) != columns:

            raise ValueError(
                "Bulk-upsert rows have inconsistent columns."
            )

    row_placeholder = (
        "("
        +
        ", ".join(
            "?"
            for _ in columns
        )
        +
        ")"
    )

    values_sql = (
        ", ".join(
            row_placeholder
            for _ in rows
        )
    )

    updates = [
        column
        for column in columns
        if column
        not in conflict_columns
    ]

    update_sql = (
        ", ".join(
            f"{column} = excluded.{column}"
            for column in updates
        )
    )

    params = tuple(
        row[column]
        for row in rows
        for column in columns
    )

    connection.execute(
        f"""
        INSERT INTO {table} (
            {", ".join(columns)}
        )

        VALUES
            {values_sql}

        ON CONFLICT (
            {", ".join(conflict_columns)}
        )

        DO UPDATE SET
            {update_sql}
        """,
        params,
    )

    return len(
        rows
    )


# ============================================================
# SAVE BATCH
# ============================================================

def save_crypto_batch(
    market_rows,
    minute_rows,
):

    if not market_rows:

        return {
            "markets":
                0,

            "minutes":
                0,
        }

    with get_connection() as connection:

        market_count = (
            bulk_upsert(

                connection=
                    connection,

                table=
                    "crypto_markets",

                rows=
                    market_rows,

                conflict_columns=[
                    "ticker",
                ],
            )
        )

        minute_count = (
            bulk_upsert(

                connection=
                    connection,

                table=
                    "crypto_market_minutes",

                rows=
                    minute_rows,

                conflict_columns=[
                    "ticker",
                    "quote_time",
                ],
            )
        )

    return {
        "markets":
            market_count,

        "minutes":
            minute_count,
    }


# ============================================================
# EXISTING BACKFILL STATE
# ============================================================

def get_finished_crypto_tickers(
    retry_missing=False,
):

    with get_connection() as connection:

        if retry_missing:

            rows = (
                connection.execute(
                    """
                    SELECT ticker
                    FROM crypto_markets
                    WHERE backfill_status = 'COLLECTED'
                    """
                )
                .fetchall()
            )

        else:

            rows = (
                connection.execute(
                    """
                    SELECT ticker
                    FROM crypto_markets
                    WHERE backfill_status IN (
                        'COLLECTED',
                        'NO_CANDLES'
                    )
                    """
                )
                .fetchall()
            )

    return {
        row[0]
        for row in rows
    }


# ============================================================
# COUNTS
# ============================================================

def get_crypto_counts():

    with get_connection() as connection:

        markets = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crypto_markets
                """
            )
            .fetchone()[0]
        )

        minutes = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crypto_market_minutes
                """
            )
            .fetchone()[0]
        )

        actionable = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crypto_market_minutes
                WHERE actionable = 1
                """
            )
            .fetchone()[0]
        )

        complete_markets = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crypto_markets
                WHERE backfill_status = 'COLLECTED'
                """
            )
            .fetchone()[0]
        )

    return {
        "markets":
            int(markets),

        "minutes":
            int(minutes),

        "actionable":
            int(actionable),

        "complete_markets":
            int(
                complete_markets
            ),
    }