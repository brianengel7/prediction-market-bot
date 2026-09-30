import argparse
import hashlib
import json
import os
import time
from datetime import date, timedelta

import pandas as pd

from src.database.db import get_connection
from src.kalshi.historical_market import get_event_markets
from src.backtest.hourly_request_limits import hourly_backfill_requests
from src.kalshi.hourly_market import (
    load_market_candles,
    parse_candle,
)
from src.kalshi.settlements import parse_market_settlement


def encode_payload(value):
    raw = json.dumps(value, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return raw, digest


def initialize_storage():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS weather_market_contract_history (
                ticker TEXT NOT NULL,
                target_date TEXT NOT NULL,
                yes_payout DOUBLE PRECISION,
                payload_hash TEXT NOT NULL,
                raw_market TEXT NOT NULL,
                downloaded_at_utc TEXT NOT NULL,
                PRIMARY KEY (ticker, payload_hash)
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS weather_market_candle_history (
                ticker TEXT NOT NULL,
                target_date TEXT NOT NULL,
                interval_minutes INTEGER NOT NULL,
                candle_end_utc TEXT NOT NULL,
                yes_bid DOUBLE PRECISION,
                yes_ask DOUBLE PRECISION,
                no_bid DOUBLE PRECISION,
                no_ask DOUBLE PRECISION,
                api_tier TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                raw_candle TEXT NOT NULL,
                downloaded_at_utc TEXT NOT NULL,
                PRIMARY KEY (
                    ticker, interval_minutes,
                    candle_end_utc, payload_hash
                )
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_candles_date
            ON weather_market_candle_history (
                target_date, interval_minutes, candle_end_utc
            )
        """)


def insert_rows(connection, table, columns, rows):
    for offset in range(0, len(rows), 50):
        batch = rows[offset:offset + 50]

        single_row = "(" + ",".join(["?"] * len(columns)) + ")"
        placeholders = ",".join([single_row] * len(batch))

        connection.execute(
            f"""
            INSERT INTO {table} ({",".join(columns)})
            VALUES {placeholders}
            ON CONFLICT DO NOTHING
            """,
            tuple(
                value
                for row in batch
                for value in row
            ),
        )


def archive_day(target_date, interval_minutes):
    event = get_event_markets(target_date)
    as_of = pd.Timestamp.now(tz="UTC")

    contract_rows = []
    candle_rows = []
    valid_quotes = 0

    for market in event["markets"]:
        ticker = market["ticker"]
        raw_market, market_hash = encode_payload(market)

        settlement = parse_market_settlement(market)
        payout = (
            settlement["yes_payout"]
            if settlement["is_settled"]
            else None
        )

        contract_rows.append((
            ticker,
            target_date,
            payout,
            market_hash,
            raw_market,
            as_of.isoformat(),
        ))

        start = pd.Timestamp(
            market["open_time"]
        ).tz_convert("UTC")

        end = min(
            pd.Timestamp(
                market["close_time"]
            ).tz_convert("UTC"),
            as_of,
        )

        if end <= start:
            continue

        loaded = load_market_candles(
            ticker=ticker,
            start_time=start,
            end_time=end,
            period_interval=interval_minutes,
        )

        received_at = pd.Timestamp.now(tz="UTC").isoformat()
        count = 0

        for candle in loaded["candles"]:
            quote = parse_candle(candle)
            stamp = quote["timestamp"]

            if not start < stamp <= end:
                continue

            raw_candle, candle_hash = encode_payload(candle)
            bid = quote["yes_bid"]
            ask = quote["yes_ask"]

            if (
                bid is not None
                and ask is not None
                and 0 <= bid <= ask <= 1
            ):
                valid_quotes += 1

            candle_rows.append((
                ticker,
                target_date,
                interval_minutes,
                stamp.isoformat(),
                bid,
                ask,
                quote["no_bid"],
                quote["no_ask"],
                loaded["source"],
                candle_hash,
                raw_candle,
                received_at,
            ))

            count += 1

        print(f"  {ticker}: {count} candles", flush=True)

    with get_connection() as connection:
        insert_rows(
            connection,
            "weather_market_contract_history",
            (
                "ticker",
                "target_date",
                "yes_payout",
                "payload_hash",
                "raw_market",
                "downloaded_at_utc",
            ),
            contract_rows,
        )

        insert_rows(
            connection,
            "weather_market_candle_history",
            (
                "ticker",
                "target_date",
                "interval_minutes",
                "candle_end_utc",
                "yes_bid",
                "yes_ask",
                "no_bid",
                "no_ask",
                "api_tier",
                "payload_hash",
                "raw_candle",
                "downloaded_at_utc",
            ),
            candle_rows,
        )

        stored = connection.execute(
            """
            SELECT COUNT(*) FROM weather_market_candle_history
            WHERE target_date = ? AND interval_minutes = ?
            """,
            (target_date, interval_minutes),
        ).fetchone()[0]

    print(
        f"{target_date}: contracts={len(contract_rows)}, "
        f"fetched={len(candle_rows)}, "
        f"bid/ask pairs={valid_quotes}, stored={stored}",
        flush=True,
    )

    if not candle_rows:
        raise RuntimeError("No candles returned for this date.")

    if not valid_quotes:
        raise RuntimeError(
            "Candles returned without usable bid/ask prices."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--interval-minutes",
        type=int,
        choices=(1, 60),
        default=60,
    )
    args = parser.parse_args()

    if not os.getenv("PREDICTION_DATABASE_URL", "").strip():
        parser.error("PREDICTION_DATABASE_URL is missing.")

    first = date.fromisoformat(args.start_date)
    last = date.fromisoformat(args.end_date)

    if last < first:
        parser.error("end-date must be on or after start-date.")

    initialize_storage()

    day = first
    failed = 0

    while day <= last:
        print(f"Collecting {day.isoformat()}...", flush=True)

        try:
            archive_day(
                day.isoformat(),
                args.interval_minutes,
            )
        except Exception as error:
            failed += 1
            print(
                f"FAILED {day}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

        day += timedelta(days=1)
        time.sleep(0.25)

    if failed:
        raise SystemExit(
            f"{failed} date(s) failed. Review the output above."
        )


if __name__ == "__main__":
    with hourly_backfill_requests():
        main()