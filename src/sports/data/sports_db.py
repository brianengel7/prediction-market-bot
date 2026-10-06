from src.database.db import (
    get_connection,
)


# ============================================================
# SCHEMA
# ============================================================

def _ensure_schema(
    connection,
):

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sports_event_snapshots (

            snapshot_time TEXT NOT NULL,
            collected_at TEXT NOT NULL,

            league TEXT NOT NULL,

            sportsbook_event_id TEXT NOT NULL,
            kalshi_event_ticker TEXT NOT NULL,

            kickoff_time TEXT,

            away_team TEXT NOT NULL,
            home_team TEXT NOT NULL,

            away_code TEXT NOT NULL,
            home_code TEXT NOT NULL,

            sportsbook_count INTEGER NOT NULL,

            away_consensus_probability REAL NOT NULL,
            home_consensus_probability REAL NOT NULL,

            away_probability_min REAL,
            away_probability_max REAL,

            home_probability_min REAL,
            home_probability_max REAL,

            probability_range REAL,

            fee_type TEXT,
            fee_multiplier REAL,

            away_market_ticker TEXT NOT NULL,

            away_yes_bid REAL,
            away_yes_ask REAL,
            away_no_bid REAL,
            away_no_ask REAL,

            home_market_ticker TEXT NOT NULL,

            home_yes_bid REAL,
            home_yes_ask REAL,
            home_no_bid REAL,
            home_no_ask REAL,

            PRIMARY KEY (
                snapshot_time,
                sportsbook_event_id
            )
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sportsbook_moneyline_snapshots (

            snapshot_time TEXT NOT NULL,
            collected_at TEXT NOT NULL,

            sportsbook_event_id TEXT NOT NULL,

            sportsbook TEXT NOT NULL,

            away_odds REAL NOT NULL,
            home_odds REAL NOT NULL,

            away_fair_probability REAL NOT NULL,
            home_fair_probability REAL NOT NULL,

            sportsbook_hold REAL NOT NULL,

            PRIMARY KEY (
                snapshot_time,
                sportsbook_event_id,
                sportsbook
            )
        )
        """
    )


def ensure_sports_snapshot_schema():

    with get_connection() as connection:

        _ensure_schema(
            connection
        )


# ============================================================
# GENERIC UPSERT
# ============================================================

def _upsert(
    connection,
    table,
    row,
    conflict_columns,
):

    columns = list(
        row
    )

    placeholders = (
        ", ".join(
            "?"
            for _ in columns
        )
    )

    updates = [
        column
        for column in columns
        if column
        not in
        conflict_columns
    ]

    assignments = (
        ", ".join(
            f"{column} = excluded.{column}"
            for column in updates
        )
    )

    query = f"""
        INSERT INTO {table} (
            {", ".join(columns)}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT (
            {", ".join(conflict_columns)}
        )
        DO UPDATE SET
            {assignments}
    """

    connection.execute(
        query,
        tuple(
            row[column]
            for column in columns
        ),
    )


# ============================================================
# SAVE SNAPSHOT
# ============================================================

def save_sports_snapshot(
    event_row,
    sportsbook_rows,
):

    if not sportsbook_rows:

        raise ValueError(
            "Sportsbook snapshot cannot be empty."
        )

    with get_connection() as connection:

        _ensure_schema(
            connection
        )

        _upsert(

            connection=
                connection,

            table=
                "sports_event_snapshots",

            row=
                event_row,

            conflict_columns=[
                "snapshot_time",
                "sportsbook_event_id",
            ],
        )

        for row in sportsbook_rows:

            _upsert(

                connection=
                    connection,

                table=
                    "sportsbook_moneyline_snapshots",

                row=
                    row,

                conflict_columns=[
                    "snapshot_time",
                    "sportsbook_event_id",
                    "sportsbook",
                ],
            )


# ============================================================
# COUNTS
# ============================================================

def get_sports_snapshot_counts():

    with get_connection() as connection:

        _ensure_schema(
            connection
        )

        event_count = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM sports_event_snapshots
                """
            )
            .fetchone()[0]
        )

        quote_count = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM sportsbook_moneyline_snapshots
                """
            )
            .fetchone()[0]
        )

    return {
        "event_snapshots":
            int(
                event_count
            ),

        "sportsbook_quotes":
            int(
                quote_count
            ),
    }