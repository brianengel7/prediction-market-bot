import math

import pandas as pd

from src.database.db import (
    get_connection,
)

from src.kalshi.client import (
    get_historical_markets,
)

from src.backtest.hourly_request_limits import (
    hourly_backfill_requests,
)


SERIES_TICKER = "KXBTC15M"

REQUEST_INTERVAL_SECONDS = 0.40

FEATURE_VERSION = (
    "KXBTC15M_5M_COINBASE_V2"
)

VALIDATION_START = (
    "2026-06-09T00:00:00+00:00"
)

VALIDATION_END_EXCLUSIVE = (
    "2026-08-08T00:00:00+00:00"
)


# ============================================================
# SCHEMA
# ============================================================

def ensure_schema():

    with get_connection() as connection:

        connection.execute(
            """
            ALTER TABLE crypto_markets

            ADD COLUMN IF NOT EXISTS
                expiration_value
                DOUBLE PRECISION
            """
        )

        connection.execute(
            """
            ALTER TABLE crypto_markets

            ADD COLUMN IF NOT EXISTS
                expiration_value_source
                TEXT
            """
        )

        connection.execute(
            """
            ALTER TABLE crypto_markets

            ADD COLUMN IF NOT EXISTS
                expiration_value_backfilled_at
                TIMESTAMPTZ
            """
        )


# ============================================================
# PARSE
# ============================================================

def parse_expiration_value(
    market,
):

    raw_value = market.get(
        "expiration_value"
    )

    if raw_value in (
        None,
        "",
    ):

        return None

    try:

        value = float(
            raw_value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if (
        not math.isfinite(
            value
        )
        or value <= 0
    ):

        return None

    return value


# ============================================================
# FETCH ARCHIVE
# ============================================================

def fetch_archive():

    print()
    print("=" * 100)
    print(
        "FETCHING KXBTC15M HISTORICAL METADATA"
    )
    print("=" * 100)

    with hourly_backfill_requests(
        interval_seconds=
            REQUEST_INTERVAL_SECONDS
    ):

        data = (
            get_historical_markets(
                series_ticker=
                    SERIES_TICKER
            )
        )

    markets = data.get(
        "markets",
        []
    )

    print(
        f"Historical markets returned: "
        f"{len(markets)}"
    )

    return markets


# ============================================================
# BUILD UPDATE SET
# ============================================================

def prepare_updates(
    markets,
):

    updates = []

    missing_expiration = 0

    bad_expiration = 0

    for market in markets:

        ticker = str(
            market.get(
                "ticker",
                ""
            )
        ).strip()

        if not ticker:

            continue

        raw_value = market.get(
            "expiration_value"
        )

        value = (
            parse_expiration_value(
                market
            )
        )

        if value is None:

            if raw_value in (
                None,
                "",
            ):

                missing_expiration += 1

            else:

                bad_expiration += 1

            continue

        updates.append(
            (
                ticker,
                value,
            )
        )

    print()
    print(
        f"Parseable expiration values: "
        f"{len(updates)}"
    )

    print(
        f"Missing expiration values:   "
        f"{missing_expiration}"
    )

    print(
        f"Invalid expiration values:   "
        f"{bad_expiration}"
    )

    return updates


# ============================================================
# UPDATE DATABASE
# ============================================================

def save_updates(
    updates,
):

    now = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    matched = 0

    with get_connection() as connection:

        for (
            ticker,
            expiration_value,
        ) in updates:

            cursor = connection.execute(
                """
                UPDATE crypto_markets

                SET

                    expiration_value = ?,

                    expiration_value_source =
                        'KALSHI_HISTORICAL_ARCHIVE',

                    expiration_value_backfilled_at = ?

                WHERE
                    ticker = ?

                    AND series_ticker = ?
                """,
                (
                    expiration_value,
                    now,
                    ticker,
                    SERIES_TICKER,
                ),
            )

            matched += int(
                cursor.rowcount
                or 0
            )

    print()
    print(
        f"Database rows updated:       "
        f"{matched}"
    )

    return matched


# ============================================================
# AUDIT
# ============================================================

def audit_database():

    with get_connection() as connection:

        overall = (
            connection.execute(
                """
                SELECT

                    COUNT(*) AS markets,

                    COUNT(expiration_value)
                        AS with_expiration,

                    MIN(expiration_value)
                        AS minimum_expiration,

                    MAX(expiration_value)
                        AS maximum_expiration

                FROM crypto_markets

                WHERE
                    series_ticker = ?
                """,
                (
                    SERIES_TICKER,
                ),
            )
            .fetchone()
        )

        feature_coverage = (
            connection.execute(
                """
                SELECT

                    COUNT(*) AS feature_rows,

                    COUNT(
                        m.expiration_value
                    ) AS with_expiration

                FROM crypto_feature_rows f

                JOIN crypto_markets m
                    ON m.ticker = f.ticker

                WHERE

                    f.feature_version = ?

                    AND f.horizon_minutes = 5
                """,
                (
                    FEATURE_VERSION,
                ),
            )
            .fetchone()
        )

        validation_coverage = (
            connection.execute(
                """
                SELECT

                    COUNT(*) AS feature_rows,

                    COUNT(
                        m.expiration_value
                    ) AS with_expiration

                FROM crypto_feature_rows f

                JOIN crypto_markets m
                    ON m.ticker = f.ticker

                WHERE

                    f.feature_version = ?

                    AND f.horizon_minutes = 5

                    AND f.quote_time >= ?

                    AND f.quote_time < ?
                """,
                (
                    FEATURE_VERSION,

                    VALIDATION_START,

                    VALIDATION_END_EXCLUSIVE,
                ),
            )
            .fetchone()
        )

        outcome_check = (
            connection.execute(
                """
                SELECT

                    COUNT(*) AS checked,

                    SUM(
                        CASE

                            WHEN
                                m.expiration_value
                                >=
                                m.floor_strike

                                AND
                                m.outcome = 1

                            THEN 1

                            WHEN
                                m.expiration_value
                                <
                                m.floor_strike

                                AND
                                m.outcome = 0

                            THEN 1

                            ELSE 0

                        END
                    ) AS consistent

                FROM crypto_markets m

                WHERE

                    m.series_ticker = ?

                    AND m.expiration_value
                        IS NOT NULL

                    AND m.floor_strike
                        IS NOT NULL

                    AND m.outcome
                        IS NOT NULL

                    AND m.strike_type =
                        'greater_or_equal'
                """,
                (
                    SERIES_TICKER,
                ),
            )
            .fetchone()
        )

    print()
    print("=" * 100)
    print(
        "BRTI EXPIRATION VALUE AUDIT"
    )
    print("=" * 100)

    print(
        f"Crypto markets:            "
        f"{int(overall[0])}"
    )

    print(
        f"With expiration value:     "
        f"{int(overall[1])}"
    )

    if overall[0]:

        print(
            f"Expiration coverage:       "
            f"{int(overall[1]) / int(overall[0]):.2%}"
        )

    print(
        f"Expiration value range:    "
        f"{overall[2]} -> {overall[3]}"
    )

    print()

    print(
        f"V2 feature rows:           "
        f"{int(feature_coverage[0])}"
    )

    print(
        f"With BRTI final:           "
        f"{int(feature_coverage[1])}"
    )

    if feature_coverage[0]:

        print(
            f"Feature coverage:          "
            f"{int(feature_coverage[1]) / int(feature_coverage[0]):.2%}"
        )

    print()

    print(
        f"Validation feature rows:   "
        f"{int(validation_coverage[0])}"
    )

    print(
        f"Validation with BRTI final:"
        f" {int(validation_coverage[1])}"
    )

    if validation_coverage[0]:

        print(
            f"Validation coverage:       "
            f"{int(validation_coverage[1]) / int(validation_coverage[0]):.2%}"
        )

    print()

    checked = int(
        outcome_check[0]
        or 0
    )

    consistent = int(
        outcome_check[1]
        or 0
    )

    print(
        f"Outcome consistency rows:  "
        f"{checked}"
    )

    print(
        f"Consistent with strike:    "
        f"{consistent}"
    )

    if checked:

        print(
            f"Consistency rate:          "
            f"{consistent / checked:.4%}"
        )

    if (
        checked
        and consistent != checked
    ):

        print()
        print(
            "WARNING: expiration_value / "
            "floor_strike does not reproduce "
            "every stored outcome."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_schema()

    markets = fetch_archive()

    updates = (
        prepare_updates(
            markets
        )
    )

    save_updates(
        updates
    )

    audit_database()


if __name__ == "__main__":

    main()