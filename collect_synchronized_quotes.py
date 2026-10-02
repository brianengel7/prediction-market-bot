import argparse
from pathlib import Path

import pandas as pd

from src.kalshi.historical_market import (
    get_event_markets,
    load_market_candles,
    parse_candle,
)
from src.backtest.hourly_request_limits import hourly_backfill_requests


def collect_date(target_date):
    day = pd.Timestamp(target_date) - pd.Timedelta(days=1)

    decision = (
        day + pd.Timedelta(hours=16, minutes=5)
    ).tz_localize("America/New_York").tz_convert("UTC")

    start = decision - pd.Timedelta(minutes=15)

    event = get_event_markets(target_date)
    markets = sorted(event["markets"], key=lambda m: m["ticker"])

    if len(markets) != 6 or len({m["ticker"] for m in markets}) != 6:
        raise ValueError("Expected six unique contracts.")

    if not all(
        m["ticker"].startswith(event["event_ticker"] + "-")
        for m in markets
    ):
        raise ValueError("Contracts do not belong to the same event.")

    by_ticker = {}
    sources = {}

    for market in markets:
        ticker = market["ticker"]
        data = load_market_candles(ticker, start, decision)

        quotes = {}
        conflicts = set()

        for candle in data["candles"]:
            quote = parse_candle(candle)
            timestamp = quote["timestamp"]
            bid = quote["yes_bid"]
            ask = quote["yes_ask"]

            if (
                start <= timestamp <= decision
                and timestamp == timestamp.floor("min")
                and bid is not None
                and ask is not None
                and 0 <= bid <= ask <= 1
            ):
                if timestamp in quotes and (
                    quotes[timestamp]["yes_bid"],
                    quotes[timestamp]["yes_ask"],
                ) != (bid, ask):
                    conflicts.add(timestamp)

                quotes[timestamp] = quote

        for timestamp in conflicts:
            quotes.pop(timestamp, None)

        by_ticker[ticker] = quotes
        sources[ticker] = data["source"]

        print(
            f"  {ticker}: {len(quotes)} valid minutes",
            flush=True,
        )

    # Keep only timestamps present for every contract.
    common = set.intersection(
        *(set(quotes) for quotes in by_ticker.values())
    )

    if not common:
        return [], {
            "target_date": target_date,
            "status": "NO_COMMON_MINUTE",
        }

    # Earliest qualifying minute, searching 3:50 -> 4:05.
    selected = min(common)
    rows = []

    for market in markets:
        ticker = market["ticker"]
        quote = by_ticker[ticker][selected]

        rows.append({
            "target_date": target_date,
            "event_ticker": event["event_ticker"],
            "ticker": ticker,
            "decision_time": decision.isoformat(),
            "quote_time": selected.isoformat(),
            "quote_age_minutes": (
                decision - selected
            ).total_seconds() / 60,
            "yes_bid": quote["yes_bid"],
            "yes_ask": quote["yes_ask"],
            "no_bid": quote["no_bid"],
            "no_ask": quote["no_ask"],
            "market_status": market.get("status"),
            "result": market.get("result"),
            "settlement_value": market.get(
                "settlement_value_dollars"
            ),
            "candle_source": sources[ticker],
            "selection_policy": "first_common_1550_1605_new_york",
        })

    return rows, {
        "target_date": target_date,
        "status": "COLLECTED",
        "selected_time_et": selected.tz_convert(
            "America/New_York"
        ).isoformat(),
        "common_minutes": len(common),
    }


def main():
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        required=True,
        help="First target date to collect (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end-date",
        required=False,
        help=(
            "Optional final target date (YYYY-MM-DD). "
            "Defaults to yesterday in America/New_York."
        ),
    )

    parser.add_argument(
        "--import-csv",
        help="Import completed quotes from the previous collector run first.",
    )

    args = parser.parse_args()


    start = pd.Timestamp(
        args.start_date
    ).normalize()


    if args.end_date:

        end = pd.Timestamp(
            args.end_date
        ).normalize()

    else:

        today_new_york = (
            pd.Timestamp.now(
                tz="America/New_York"
            )
            .normalize()
        )

        end = (
            today_new_york
            -
            pd.Timedelta(
                days=1
            )
        ).tz_localize(None)


    if start > end:

        parser.error(
            f"Start date {start:%Y-%m-%d} "
            f"must not follow end date "
            f"{end:%Y-%m-%d}."
        )

    load_dotenv(
        Path(__file__).resolve().parent / ".env",
        override=False,
    )

    if not os.environ.get("PREDICTION_DATABASE_URL", "").strip():
        parser.error(
            "PREDICTION_DATABASE_URL is missing; Supabase is required."
        )

    from src.database.db import (
        get_connection,
        ensure_market_calibration_schema,
        save_historical_market_entries,
    )
    from src.kalshi.decision_timing import SYNC_CALIBRATION_PREFIX
    from src.kalshi.historical_market import (
        get_decision_time,
        build_event_ticker,
    )

    with get_connection() as connection:
        ensure_market_calibration_schema(connection)

    policy = "first_common_1550_1605_new_york"

    if SYNC_CALIBRATION_PREFIX != "SYNC_FIRST_1550_1605|":
        raise RuntimeError(
            "Collector and database synchronization policies differ."
        )

    def save_daily(daily):
        entries = []

        for row in daily:
            if row.get("selection_policy") != policy:
                raise ValueError(
                    "Quote selection policy does not match this collector."
                )

            entry = dict(row)
            target = pd.Timestamp(row["target_date"]).strftime("%Y-%m-%d")
            expected_event = build_event_ticker(target)

            if (
                row["event_ticker"] != expected_event
                or not str(row["ticker"]).startswith(expected_event + "-")
            ):
                raise ValueError(
                    f"Wrong event or contract for {target}."
                )

            decision = pd.Timestamp(row["decision_time"])
            quote_time = pd.Timestamp(row["quote_time"])

            if decision.tzinfo is None or quote_time.tzinfo is None:
                raise ValueError(
                    "Quote and decision timestamps must include timezones."
                )

            decision = decision.tz_convert("UTC")
            quote_time = quote_time.tz_convert("UTC")

            if decision != get_decision_time(target):
                raise ValueError(
                    f"Decision time does not match the trader for {target}."
                )

            source = str(row["candle_source"])

            if source not in {"LIVE", "HISTORICAL"}:
                raise ValueError(
                    f"Unexpected candle source: {source}"
                )

            entry["target_date"] = target
            entry["decision_time"] = decision
            entry["entry_time"] = quote_time
            entry["candle_source"] = SYNC_CALIBRATION_PREFIX + source

            entries.append(entry)

        count = save_historical_market_entries(
            entries,
            calibration_quotes=True,
        )

        if count != 6:
            raise RuntimeError(
                f"Expected six saved quotes, received {count}."
            )

        return count

    # One-time recovery of quotes collected by the old CSV version.
    imported = set()

    if args.import_csv:
        try:
            cached = pd.read_csv(
                args.import_csv,
                dtype=str,
                keep_default_na=False,
            )
        except pd.errors.EmptyDataError:
            cached = pd.DataFrame()

        if not cached.empty:
            cached["target_date"] = pd.to_datetime(
                cached["target_date"],
                errors="raise",
            ).dt.strftime("%Y-%m-%d")

            cached = cached.loc[
                cached["target_date"].between(
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                )
            ]

            for target, group in cached.groupby(
                "target_date",
                sort=True,
            ):
                count = save_daily(group.to_dict("records"))
                imported.add(target)

                print(
                    f"{target}: IMPORTED {count} quotes into Supabase",
                    flush=True,
                )
    collection_start = start

    if imported:
        collection_start = (
            pd.Timestamp(max(imported))
            + pd.Timedelta(days=1)
        )

    if collection_start <= end:
        print(
            f"Resuming collection at {collection_start:%Y-%m-%d}",
            flush=True,
        )
    else:
        print(
            "No dates remain after the latest import in the requested range.",
            flush=True,
        )
    saved = skipped = failed = 0

    with hourly_backfill_requests(
        interval_seconds=2,
        max_attempts=6,
    ):
        for date in pd.date_range(collection_start, end):
            target = date.strftime("%Y-%m-%d")

            if target in imported:
                continue

            print(f"Collecting {target}...", flush=True)

            try:
                daily, status = collect_date(target)
            except Exception as error:
                failed += 1
                print(
                    f"{target}: COLLECTION FAILED: {error}",
                    flush=True,
                )
                continue

            if not daily:
                skipped += 1
                print(status, flush=True)
                continue

            # Stop on storage errors so they cannot pass unnoticed.
            count = save_daily(daily)
            saved += 1

            print(
                f"{target}: SAVED {count} quotes to Supabase | "
                f"selected={status.get('selected_time_et')}",
                flush=True,
            )

    print(
        f"Finished: imported_dates={len(imported)}, "
        f"saved_dates={saved}, skipped_dates={skipped}, "
        f"collection_failures={failed}",
        flush=True,
    )

    with get_connection() as connection:
        stored_rows, stored_dates = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT target_date) "
            "FROM historical_market_entries "
            "WHERE calibration_candle_source IN (?, ?)",
            (
                SYNC_CALIBRATION_PREFIX + "LIVE",
                SYNC_CALIBRATION_PREFIX + "HISTORICAL",
            ),
        ).fetchone()

    print(
        f"Requested collection range: "
        f"{start:%Y-%m-%d} -> "
        f"{end:%Y-%m-%d}",
        flush=True,
    )

    print(
        f"Supabase read-back: {stored_rows} synchronized rows "
        f"across {stored_dates} dates."
    )

if __name__ == "__main__":
    main()