import argparse
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from src.database.db import get_connection
from src.backtest.hourly_market_backfill import (
    initialize_storage,
    encode_payload,
    insert_rows,
)
from src.kalshi.hourly_market import parse_candle


BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = "KXTEMPNYCHS"
ET = ZoneInfo("America/New_York")
SESSION = requests.Session()


def fetch(path, params):
    for attempt in range(8):
        response = SESSION.get(
            BASE + path,
            params=params,
            timeout=35,
        )

        if response.status_code == 429 or response.status_code >= 500:
            retry_after = response.headers.get("Retry-After")
            try:
                retry_delay = float(retry_after)
            except (TypeError, ValueError):
                retry_delay = 0

            delay = max(min(2 ** attempt, 30), retry_delay)
            print(f"HTTP {response.status_code}: retry in {delay:.1f}s")
            response.close()
            time.sleep(delay)
            continue

        response.raise_for_status()
        data = response.json()
        response.close()
        time.sleep(0.4)
        return data

    raise RuntimeError(f"API retries exhausted: {path}")


def utc_time(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def load_settled_markets():
    markets = {}
    cursor = None
    seen_cursors = set()

    for _ in range(100):
        params = {
            "series_ticker": SERIES,
            "status": "settled",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        data = fetch("/markets", params)

        for market in data.get("markets", []):
            if market.get("result") in ("yes", "no"):
                markets[market["ticker"]] = market

        next_cursor = data.get("cursor")
        if not next_cursor:
            return list(markets.values())

        if next_cursor in seen_cursors:
            raise RuntimeError("Repeated pagination cursor.")

        seen_cursors.add(next_cursor)
        cursor = next_cursor

    raise RuntimeError("Pagination limit reached.")


def initialize_checkpoint():
    with get_connection() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS
            weather_hourly_index_collection_log (
                event_ticker TEXT NOT NULL,
                lookback_minutes INTEGER NOT NULL,
                contract_count INTEGER NOT NULL,
                candle_count INTEGER NOT NULL,
                completed_at_utc TEXT NOT NULL,
                PRIMARY KEY (event_ticker, lookback_minutes)
            )
        """)


def already_collected(event_ticker, minutes):
    with get_connection() as db:
        row = db.execute("""
            SELECT 1
            FROM weather_hourly_index_collection_log
            WHERE event_ticker = ?
              AND lookback_minutes = ?
        """, (event_ticker, minutes)).fetchone()

    return row is not None


def collect_event(event_ticker, markets, minutes):
    tickers = sorted(m["ticker"] for m in markets)

    if not tickers or len(tickers) > 100:
        raise ValueError("Unexpected contract count.")

    closes = {m["close_time"] for m in markets}
    if len(closes) != 1:
        raise ValueError("Contracts have different close times.")

    close = utc_time(markets[0]["close_time"])
    start = close - timedelta(minutes=minutes)
    target_date = close.astimezone(ET).date().isoformat()

    data = fetch("/markets/candlesticks", {
        "market_tickers": ",".join(tickers),
        "start_ts": int(start.timestamp()),
        "end_ts": int(close.timestamp()),
        "period_interval": 1,
    })

    returned = {}
    for item in data.get("markets", []):
        ticker = item.get("market_ticker")
        if not ticker or ticker in returned:
            raise RuntimeError("Missing or duplicate batch ticker.")
        returned[ticker] = item

    if set(returned) != set(tickers):
        raise RuntimeError(
            f"Batch ticker mismatch for {event_ticker}: "
            f"missing={set(tickers) - set(returned)}, "
            f"unexpected={set(returned) - set(tickers)}"
        )

    now = datetime.now(timezone.utc).isoformat()

    contract_rows = []
    candle_rows = []
    valid_quotes = 0

    for market in markets:
        ticker = market["ticker"]
        payload, digest = encode_payload(market)

        contract_rows.append((
            ticker,
            target_date,
            1.0 if market["result"] == "yes" else 0.0,
            digest,
            payload,
            now,
        ))

        for candle in returned[ticker].get("candlesticks", []):
            quote = parse_candle(candle)
            timestamp = quote["timestamp"]

            if not (
                start.timestamp()
                <= timestamp.timestamp()
                <= close.timestamp()
            ):
                continue

            bid = quote["yes_bid"]
            ask = quote["yes_ask"]

            if bid is not None and ask is not None:
                if 0 <= bid <= ask <= 1:
                    valid_quotes += 1

            raw, candle_hash = encode_payload(candle)

            candle_rows.append((
                ticker,
                target_date,
                1,
                timestamp.isoformat(),
                bid,
                ask,
                quote["no_bid"],
                quote["no_ask"],
                "LIVE_BATCH",
                candle_hash,
                raw,
                now,
            ))

    # Contract data, candles and checkpoint commit together.
    with get_connection() as db:
        insert_rows(
            db,
            "weather_market_contract_history",
            (
                "ticker", "target_date", "yes_payout",
                "payload_hash", "raw_market",
                "downloaded_at_utc",
            ),
            contract_rows,
        )

        insert_rows(
            db,
            "weather_market_candle_history",
            (
                "ticker", "target_date", "interval_minutes",
                "candle_end_utc", "yes_bid", "yes_ask",
                "no_bid", "no_ask", "api_tier",
                "payload_hash", "raw_candle",
                "downloaded_at_utc",
            ),
            candle_rows,
        )

        db.execute("""
            INSERT INTO weather_hourly_index_collection_log
            (
                event_ticker, lookback_minutes,
                contract_count, candle_count, completed_at_utc
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
        """, (
            event_ticker,
            minutes,
            len(contract_rows),
            len(candle_rows),
            now,
        ))

    return len(candle_rows), valid_quotes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--lookback-minutes", type=int, default=60)
    args = parser.parse_args()

    first = date.fromisoformat(args.start_date)
    last = date.fromisoformat(args.end_date)

    if last < first or args.lookback_minutes < 1:
        parser.error("Invalid date range or lookback.")

    if not os.getenv("PREDICTION_DATABASE_URL", "").strip():
        parser.error(
            "PREDICTION_DATABASE_URL missing. "
            "Refusing to use local SQLite."
        )

    initialize_storage()
    initialize_checkpoint()

    markets = load_settled_markets()

    events = defaultdict(list)
    for market in markets:
        close = utc_time(market["close_time"])
        day = close.astimezone(ET).date()

        if first <= day <= last:
            events[market["event_ticker"]].append(market)

    print(
        f"Found {len(events)} settled events "
        f"between {first} and {last}",
        flush=True,
    )

    ordered = sorted(
        events.items(),
        key=lambda pair: pair[1][0]["close_time"],
        reverse=True,
    )

    processed = 0
    skipped = 0
    candles_total = 0

    for event_ticker, event_markets in ordered:
        if already_collected(
            event_ticker, args.lookback_minutes
        ):
            skipped += 1
            continue

        candles, valid = collect_event(
            event_ticker,
            event_markets,
            args.lookback_minutes,
        )

        processed += 1
        candles_total += candles

        print(
            f"{event_ticker}: "
            f"contracts={len(event_markets)}, "
            f"candles={candles}, "
            f"valid_quotes={valid}",
            flush=True,
        )

        if args.max_events and processed >= args.max_events:
            break

    print("\nHOURLY INDEX BACKFILL SUMMARY")
    print("New events:", processed)
    print("Previously collected:", skipped)
    print("Newly fetched candles:", candles_total)
    print("Supabase: prediction_bot schema")


if __name__ == "__main__":
    main()