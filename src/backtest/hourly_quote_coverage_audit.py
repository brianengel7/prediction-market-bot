"""Read-only audit of minute-candle density for KXTEMPNYCHS."""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from src.backtest.hourly_index_market_backtest import load_history
from src.database.db import get_connection, read_dataframe

MINUTE_NS = 60_000_000_000


def main(end, max_age):
    markets, candles = load_history(end)

    stamps = pd.to_datetime(
        candles["stamp"],
        utc=True,
    )

    sec = (
        stamps.dt.second
        + stamps.dt.microsecond / 1e6
    )

    n_duplicate_minutes = int(
        candles.assign(
            m=stamps.dt.floor("min")
        )
        .groupby(["ticker", "m"])
        .size()
        .gt(1)
        .sum()
    )

    print("\nKXTEMPNYCHS DATASET AUDIT (READ ONLY)")
    print("=" * 76)

    print(
        "Dates:",
        markets.day.min(),
        "through",
        markets.day.max(),
    )

    print("Complete events:", markets.event.nunique())
    print("Threshold contracts:", markets.ticker.nunique())
    print("Valid 1-minute quote records:", len(candles))

    print(
        "Rows with sub-minute timestamp:",
        int((sec != 0).sum()),
    )

    print(
        "Contract/minute cells with multiple rows:",
        n_duplicate_minutes,
    )

    print(
        "NOTE: Sub-minute timestamps do NOT prove "
        "second-by-second order books."
    )

    # Explicit nanoseconds conversion fixes the earlier
    # microsecond/nanosecond mismatch.
    quotes = {
        ticker: np.sort(
            g["stamp"]
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
        )
        for ticker, g in candles.groupby(
            "ticker", sort=False
        )
    }

    stats = defaultdict(lambda: [0, 0, 0, 0])
    event_coverage = []

    for event, g in markets.groupby(
        "event", sort=False
    ):
        if len(g) != 10:
            continue

        close = pd.Timestamp(
            g["close"].iloc[0]
        ).value

        num_full = 0

        for h in range(60, 4, -1):
            cutoff = close - (h + 1) * MINUTE_NS
            fresh = 0

            for ticker in g.ticker:
                arr = quotes.get(ticker)

                if arr is None:
                    continue

                i = (
                    np.searchsorted(
                        arr,
                        cutoff,
                        side="right",
                    ) - 1
                )

                if (
                    i >= 0
                    and 0 <= cutoff - arr[i]
                    <= max_age * MINUTE_NS
                ):
                    fresh += 1

            r = stats[h]

            r[0] += 1
            r[1] += int(fresh >= 1)
            r[2] += int(fresh >= 5)
            r[3] += int(fresh == 10)

            num_full += int(fresh == 10)

        event_coverage.append(num_full)

    total = sum(x[0] for x in stats.values())
    any_fresh = sum(x[1] for x in stats.values())
    all_fresh = sum(x[3] for x in stats.values())

    print(
        "\nFINAL-HOUR QUOTE COVERAGE "
        "(60m to 5m, 56 scan minutes/event)"
    )

    print(f"Potential scan instants: {total:,}")

    print(
        f"Instants with >=1 fresh quote: "
        f"{any_fresh:,} ({any_fresh / total:.1%})"
    )

    print(
        f"Instants with 10 fresh quotes: "
        f"{all_fresh:,} ({all_fresh / total:.1%})"
    )

    print(
        f"Quote freshness: maximum {max_age} minutes; "
        "signal cutoff = decision - 1 minute"
    )

    print("\n Horizon   Any quote    >=5 quotes   All 10")

    for h in range(60, 4, -5):
        n, a, b, c = stats[h]

        print(
            f" {h:>3}m      "
            f"{a/n:>7.1%}       "
            f"{b/n:>7.1%}     "
            f"{c/n:>7.1%}"
        )

    print("\nFull-coverage scan minutes per event:")

    print(
        "Median:",
        float(np.median(event_coverage)),
        "of 56",
    )

    print(
        "Events with >=1 fully quoted minute:",
        sum(x > 0 for x in event_coverage),
    )

    print(
        "\nDATABASE TABLES THAT MAY CONTAIN "
        "FINER-GRAINED DATA"
    )

    try:
        with get_connection() as con:
            table_info = read_dataframe(
                """
                SELECT
                    table_schema,
                    table_name,
                    column_name
                FROM information_schema.columns
                WHERE table_schema NOT IN (
                    'pg_catalog',
                    'information_schema'
                )
                AND (
                    table_name ILIKE ?
                    OR table_name ILIKE ?
                    OR table_name ILIKE ?
                    OR table_name ILIKE ?
                )
                ORDER BY
                    table_schema,
                    table_name,
                    ordinal_position
                """,
                con,
                [
                    "%tick%",
                    "%quote%",
                    "%book%",
                    "%candle%",
                ],
            )

        if table_info.empty:
            print("No candidate tables found.")
        else:
            for (
                schema, table
            ), g in table_info.groupby(
                ["table_schema", "table_name"]
            ):
                columns = ", ".join(
                    g.column_name.astype(str)
                )

                print(
                    f"{schema}.{table}: "
                    f"{columns[:220]}"
                )

    except Exception as exc:
        print(
            "Could not inspect table metadata:",
            exc,
        )

    print("\nNo database writes, CSV files, or orders.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--end",
        default="2026-10-07",
    )

    p.add_argument(
        "--quote-max-age",
        type=int,
        default=3,
    )

    a = p.parse_args()

    if not 0 <= a.quote_max_age <= 10:
        p.error("--quote-max-age must be in 0..10")

    main(a.end, a.quote_max_age)
