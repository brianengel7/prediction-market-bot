import os

from src.database.db import (
    get_connection,
)


HORIZONS = (
    15,
    10,
    5,
    3,
    2,
    1,
)


def print_distribution(
    connection,
    column,
):

    rows = (
        connection.execute(
            f"""
            SELECT
                {column},
                COUNT(*) AS markets
            FROM crypto_markets
            GROUP BY {column}
            ORDER BY {column}
            """
        )
        .fetchall()
    )

    print()

    print(
        f"{column}:"
    )

    for value, count in rows:

        print(
            f"  {str(value):>8s}: "
            f"{count}"
        )


def main():

    if not os.environ.get(
        "PREDICTION_DATABASE_URL",
        "",
    ).strip():

        raise RuntimeError(
            "PREDICTION_DATABASE_URL is missing. "
            "This audit is intended to inspect Supabase."
        )

    with get_connection() as connection:

        # ====================================================
        # MARKET TABLE
        # ====================================================

        market_summary = (
            connection.execute(
                """
                SELECT
                    COUNT(*) AS total_markets,

                    COUNT(*) FILTER (
                        WHERE backfill_status = 'COLLECTED'
                    ) AS collected,

                    COUNT(*) FILTER (
                        WHERE backfill_status = 'NO_CANDLES'
                    ) AS no_candles,

                    COUNT(*) FILTER (
                        WHERE outcome = 1
                    ) AS yes_outcomes,

                    COUNT(*) FILTER (
                        WHERE outcome = 0
                    ) AS no_outcomes,

                    COUNT(*) FILTER (
                        WHERE outcome IS NULL
                    ) AS missing_outcome,

                    MIN(close_time) AS earliest_close,

                    MAX(close_time) AS latest_close

                FROM crypto_markets
                """
            )
            .fetchone()
        )

        (
            total_markets,
            collected,
            no_candles,
            yes_outcomes,
            no_outcomes,
            missing_outcome,
            earliest_close,
            latest_close,
        ) = market_summary

        print()
        print("=" * 90)
        print(
            "CRYPTO DATASET INTEGRITY AUDIT"
        )
        print("=" * 90)

        print(
            f"Markets:                 "
            f"{total_markets}"
        )

        print(
            f"Collected:               "
            f"{collected}"
        )

        print(
            f"No candles:              "
            f"{no_candles}"
        )

        print(
            f"YES outcomes:            "
            f"{yes_outcomes}"
        )

        print(
            f"NO outcomes:             "
            f"{no_outcomes}"
        )

        print(
            f"Missing outcomes:        "
            f"{missing_outcome}"
        )

        print(
            f"Earliest close:          "
            f"{earliest_close}"
        )

        print(
            f"Latest close:            "
            f"{latest_close}"
        )

        print_distribution(
            connection,
            "raw_candle_count",
        )

        print_distribution(
            connection,
            "valid_quote_count",
        )

        print_distribution(
            connection,
            "actionable_quote_count",
        )

        # ====================================================
        # MINUTE TABLE
        # ====================================================

        minute_summary = (
            connection.execute(
                """
                SELECT
                    COUNT(*) AS minute_rows,

                    COUNT(*) FILTER (
                        WHERE actionable = 1
                    ) AS actionable_rows,

                    COUNT(*) FILTER (
                        WHERE actionable = 0
                    ) AS non_actionable_rows,

                    COUNT(*) FILTER (
                        WHERE
                            yes_bid IS NULL
                            OR yes_ask IS NULL
                    ) AS missing_quote_rows,

                    COUNT(*) FILTER (
                        WHERE
                            yes_bid < 0
                            OR yes_ask > 1
                            OR yes_bid > yes_ask
                    ) AS invalid_book_rows,

                    COUNT(*) FILTER (
                        WHERE
                            midpoint IS NOT NULL
                            AND (
                                midpoint < 0
                                OR midpoint > 1
                            )
                    ) AS invalid_midpoints,

                    COUNT(*) FILTER (
                        WHERE
                            spread IS NOT NULL
                            AND spread < 0
                    ) AS negative_spreads,

                    COUNT(*) FILTER (
                        WHERE
                            actionable = 1
                            AND (
                                minutes_remaining <= 0
                                OR minutes_remaining > 15
                            )
                    ) AS bad_actionable_times,

                    MIN(minutes_remaining),

                    MAX(minutes_remaining)

                FROM crypto_market_minutes
                """
            )
            .fetchone()
        )

        (
            minute_rows,
            actionable_rows,
            non_actionable_rows,
            missing_quote_rows,
            invalid_book_rows,
            invalid_midpoints,
            negative_spreads,
            bad_actionable_times,
            minimum_minutes,
            maximum_minutes,
        ) = minute_summary

        print()
        print("=" * 90)
        print(
            "MINUTE DATA"
        )
        print("=" * 90)

        print(
            f"Minute rows:             "
            f"{minute_rows}"
        )

        print(
            f"Actionable rows:         "
            f"{actionable_rows}"
        )

        print(
            f"Non-actionable rows:     "
            f"{non_actionable_rows}"
        )

        print(
            f"Missing bid/ask rows:    "
            f"{missing_quote_rows}"
        )

        print(
            f"Invalid books:           "
            f"{invalid_book_rows}"
        )

        print(
            f"Invalid midpoints:       "
            f"{invalid_midpoints}"
        )

        print(
            f"Negative spreads:        "
            f"{negative_spreads}"
        )

        print(
            f"Bad actionable timing:   "
            f"{bad_actionable_times}"
        )

        print(
            f"Min minutes remaining:   "
            f"{minimum_minutes}"
        )

        print(
            f"Max minutes remaining:   "
            f"{maximum_minutes}"
        )

        # ====================================================
        # PRIMARY KEY / DUPLICATE CHECK
        # ====================================================

        duplicate_groups = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT
                        ticker,
                        quote_time,
                        COUNT(*) AS n
                    FROM crypto_market_minutes
                    GROUP BY
                        ticker,
                        quote_time
                    HAVING COUNT(*) > 1
                ) duplicates
                """
            )
            .fetchone()[0]
        )

        print(
            f"Duplicate ticker/times:  "
            f"{duplicate_groups}"
        )

        # ====================================================
        # MARKET ↔ MINUTE CONSISTENCY
        # ====================================================

        missing_minute_markets = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crypto_markets m
                LEFT JOIN crypto_market_minutes mm
                    ON mm.ticker = m.ticker
                WHERE
                    m.backfill_status = 'COLLECTED'
                    AND mm.ticker IS NULL
                """
            )
            .fetchone()[0]
        )

        outcome_mismatches = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM crypto_market_minutes mm
                JOIN crypto_markets m
                    ON m.ticker = mm.ticker
                WHERE
                    mm.outcome
                    IS DISTINCT FROM
                    m.outcome
                """
            )
            .fetchone()[0]
        )

        print(
            f"Collected w/o minutes:   "
            f"{missing_minute_markets}"
        )

        print(
            f"Outcome mismatches:      "
            f"{outcome_mismatches}"
        )

        # ====================================================
        # PER-MARKET ROW COUNTS
        # ====================================================

        count_distribution = (
            connection.execute(
                """
                SELECT
                    minute_count,
                    actionable_count,
                    COUNT(*) AS markets
                FROM (
                    SELECT
                        ticker,
                        COUNT(*) AS minute_count,
                        SUM(actionable) AS actionable_count
                    FROM crypto_market_minutes
                    GROUP BY ticker
                ) market_counts
                GROUP BY
                    minute_count,
                    actionable_count
                ORDER BY
                    minute_count,
                    actionable_count
                """
            )
            .fetchall()
        )

        print()
        print("=" * 90)
        print(
            "ROWS PER MARKET"
        )
        print("=" * 90)

        for (
            minute_count,
            actionable_count,
            market_count,
        ) in count_distribution:

            print(
                f"stored={minute_count:2d} | "
                f"actionable={int(actionable_count):2d} | "
                f"markets={market_count}"
            )

        # ====================================================
        # EXACT DECISION-HORIZON COVERAGE
        # ====================================================

        print()
        print("=" * 90)
        print(
            "DECISION-HORIZON COVERAGE"
        )
        print("=" * 90)

        for horizon in HORIZONS:

            coverage = (
                connection.execute(
                    """
                    SELECT COUNT(
                        DISTINCT ticker
                    )
                    FROM crypto_market_minutes
                    WHERE
                        actionable = 1
                        AND ABS(
                            minutes_remaining - ?
                        ) < 0.000001
                    """,
                    (
                        float(
                            horizon
                        ),
                    ),
                )
                .fetchone()[0]
            )

            percentage = (
                coverage
                /
                collected
                if collected
                else 0
            )

            print(
                f"{horizon:2d} min remaining: "
                f"{coverage:6d} markets "
                f"({percentage:.2%})"
            )

        # ====================================================
        # SETTLEMENT CANDLE SAFETY
        # ====================================================

        zero_minute_rows = (
            connection.execute(
                """
                SELECT
                    COUNT(*),

                    COUNT(*) FILTER (
                        WHERE actionable = 1
                    )

                FROM crypto_market_minutes

                WHERE ABS(
                    minutes_remaining
                ) < 0.000001
                """
            )
            .fetchone()
        )

        print()
        print("=" * 90)
        print(
            "LOOK-AHEAD SAFETY"
        )
        print("=" * 90)

        print(
            f"0-minute close candles:  "
            f"{zero_minute_rows[0]}"
        )

        print(
            f"0-minute actionable:     "
            f"{zero_minute_rows[1]}"
        )

        # ====================================================
        # SPREAD SUMMARY
        # ====================================================

        spread_summary = (
            connection.execute(
                """
                SELECT
                    AVG(spread),
                    MIN(spread),
                    MAX(spread)

                FROM crypto_market_minutes

                WHERE
                    actionable = 1
                    AND spread IS NOT NULL
                """
            )
            .fetchone()
        )

        print()
        print("=" * 90)
        print(
            "ACTIONABLE SPREADS"
        )
        print("=" * 90)

        print(
            f"Mean spread:             "
            f"{spread_summary[0]:.4f}"
        )

        print(
            f"Minimum spread:          "
            f"{spread_summary[1]:.4f}"
        )

        print(
            f"Maximum spread:          "
            f"{spread_summary[2]:.4f}"
        )


if __name__ == "__main__":

    main()