"""Historical Kalshi BTC15M trades and candles aligned to BRTI."""
import csv
import json
import time
from bisect import bisect_left
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests

API = "https://external-api.kalshi.com/trade-api/v2"

START = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
END = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)

START_TS = int(START.timestamp())
END_TS = int(END.timestamp())

OUT = Path(
    "data/research/btc_cfbenchmarks/kalshi_20261103_14utc"
)
BRTI = OUT.parent / "brti_20261103_14utc.csv"

session = requests.Session()


def get(path, params=None):
    for attempt in range(7):
        try:
            response = session.get(
                API + path,
                params=params,
                timeout=30,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == 6:
                    response.raise_for_status()
                time.sleep(min(15, 1.5 * 2**attempt))
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == 6:
                raise
            time.sleep(min(15, 1.5 * 2**attempt))

    raise RuntimeError(f"Could not fetch {path}")


def dt(value):
    return datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )


def save_json(path, rows):
    path.write_text(json.dumps(rows, indent=2) + "\n")


def candle_value(candle, section, field):
    obj = candle.get(section) or {}
    value = obj.get(field + "_dollars")
    return value if value is not None else obj.get(field)


def discover_markets():
    markets = []
    cursor = None
    seen = set()

    while True:
        params = {
            "series_ticker": "KXBTC15M",
            "min_close_ts": START_TS,
            "max_close_ts": END_TS,
            "limit": 1000,
            "mve_filter": "exclude",
        }

        if cursor:
            params["cursor"] = cursor

        data = get("/markets", params)
        markets.extend(data.get("markets", []))

        cursor = data.get("cursor")
        if not cursor:
            break

        if cursor in seen:
            raise RuntimeError("Repeated market cursor")

        seen.add(cursor)

    result = []

    for market in markets:
        ticker = market.get("ticker", "")
        close = int(dt(market["close_time"]).timestamp())

        if (
            ticker.startswith("KXBTC15M-")
            and START_TS < close <= END_TS
        ):
            result.append(market)

    return sorted(result, key=lambda m: m["close_time"])


def fetch_trades(ticker):
    rows = []
    cursor = None
    seen = set()

    while True:
        params = {
            "ticker": ticker,
            "min_ts": START_TS,
            "max_ts": END_TS,
            "limit": 1000,
        }

        if cursor:
            params["cursor"] = cursor

        data = get("/markets/trades", params)
        rows.extend(data.get("trades", []))

        cursor = data.get("cursor")

        if not cursor:
            break

        if cursor in seen or len(seen) > 1000:
            raise RuntimeError("Trade pagination failure")

        seen.add(cursor)

    unique = {}

    for i, trade in enumerate(rows):
        key = trade.get("trade_id") or f"missing-id-{i}"
        unique[key] = trade

    return sorted(
        unique.values(),
        key=lambda t: (
            t.get("created_time", ""),
            str(t.get("trade_id") or ""),
        ),
    )


def load_brti():
    if not BRTI.exists():
        raise FileNotFoundError(BRTI)

    points = []

    with BRTI.open(newline="") as f:
        for row in csv.DictReader(f):
            points.append((
                int(row["timestamp_ms"]),
                Decimal(row["brti_usd"]),
            ))

    points.sort()
    return [p[0] for p in points], [p[1] for p in points]


def brti_before(timestamp, times, values):
    ms = timestamp.timestamp() * 1000
    index = bisect_left(times, ms) - 1

    if index < 0:
        return None, None

    return values[index], round(ms - times[index], 3)


def save_csv(filename, rows, time_column):
    if not rows:
        print(f"  {filename}: 0 rows")
        return

    rows.sort(key=lambda x: x[time_column])

    with (OUT / filename).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    brti_times, brti_values = load_brti()
    markets = discover_markets()

    if not markets:
        raise RuntimeError("No markets found in requested hour")

    save_json(OUT / "markets.json", markets)

    print("\nKALSHI BTC HISTORICAL BACKFILL")
    print("=" * 70)
    print("UTC window:", START, "to", END)
    print("Markets found:", len(markets))

    all_trades = []
    all_candles = []

    for market in markets:
        ticker = market["ticker"]
        close_ts = int(dt(market["close_time"]).timestamp())
        strike = market.get("floor_strike")

        print("\nDownloading:", ticker, flush=True)

        trades = fetch_trades(ticker)

        candle_response = get(
            f"/series/KXBTC15M/markets/{ticker}/candlesticks",
            {
                "start_ts": START_TS,
                "end_ts": close_ts,
                "period_interval": 1,
            },
        )

        candles = candle_response.get("candlesticks", [])

        save_json(OUT / f"{ticker}_trades.json", trades)
        save_json(OUT / f"{ticker}_candles.json", candles)

        for trade in trades:
            timestamp = dt(trade["created_time"])

            if (
                not START <= timestamp < END
                or timestamp.timestamp() >= close_ts
            ):
                continue

            yes_price = trade.get("yes_price_dollars")

            if yes_price is None and trade.get("yes_price") is not None:
                yes_price = str(
                    Decimal(str(trade["yes_price"])) / 100
                )

            brti, age_ms = brti_before(
                timestamp,
                brti_times,
                brti_values,
            )

            difference = (
                brti - Decimal(str(strike))
                if brti is not None and strike is not None
                else None
            )

            all_trades.append({
                "ticker": ticker,
                "trade_id": trade.get("trade_id"),
                "trade_time_utc": timestamp.isoformat(),
                "yes_price_dollars": yes_price,
                "contracts": trade.get(
                    "count_fp", trade.get("count")
                ),
                "taker_side": trade.get(
                    "taker_outcome_side",
                    trade.get("taker_side"),
                ),
                "taker_book_side": trade.get("taker_book_side"),
                "block_trade": trade.get("is_block_trade"),
                "opening_strike_usd": strike,
                "historical_brti_usd": (
                    str(brti) if brti is not None else None
                ),
                "historical_brti_age_ms": age_ms,
                "brti_minus_strike_usd": (
                    str(difference)
                    if difference is not None else None
                ),
            })

        for candle in candles:
            end_ts = int(candle["end_period_ts"])

            if not START_TS < end_ts <= close_ts:
                continue

            row = {
                "ticker": ticker,
                "candle_end_utc": datetime.fromtimestamp(
                    end_ts, timezone.utc
                ).isoformat(),
                "seconds_to_close": close_ts - end_ts,
                "historically_before_close": end_ts < close_ts,
                "volume": candle.get(
                    "volume_fp", candle.get("volume")
                ),
            }

            for kind in ("open", "high", "low", "close"):
                for part in ("yes_bid", "yes_ask", "price"):
                    row[f"{part}_{kind}"] = candle_value(
                        candle, part, kind
                    )

            all_candles.append(row)

        print(
            f"  Trades: {len(trades)} | "
            f"Minute candles: {len(candles)}",
            flush=True,
        )

    save_csv(
        "trades_brti_aligned.csv",
        all_trades,
        "trade_time_utc",
    )

    save_csv(
        "minute_candles.csv",
        all_candles,
        "candle_end_utc",
    )

    print("\nBACKFILL COMPLETE")
    print("=" * 70)
    print("Total individual trades:", len(all_trades))
    print("Total minute candles:", len(all_candles))
    print("Markets:", len(markets))
    print("Output:", OUT)
    print("No orders submitted.")


if __name__ == "__main__":
    main()
