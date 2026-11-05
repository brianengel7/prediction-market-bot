"""Read-only KXBTC15M penny-reversal backtest using local SQLite."""

import argparse
import json
import math
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

BASE = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_DB = Path(".collector_runs/btc15m_pennies.sqlite3")


def seconds(value):
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).timestamp())
    except (ValueError, TypeError, AttributeError):
        return None


def api(path, params=None):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)

    for attempt in range(6):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "btc15m-penny-research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)

        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise RuntimeError(
                    f"API HTTP {exc.code}: {url}"
                ) from exc

            retry = exc.headers.get("Retry-After", "")
            delay = (
                min(15, float(retry))
                if retry.replace(".", "", 1).isdigit()
                else min(8, 1.5 ** attempt)
            )
            time.sleep(delay)

        except (OSError, TimeoutError):
            if attempt == 5:
                raise
            time.sleep(min(8, 1.5 ** attempt))


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")

    db.executescript("""
    CREATE TABLE IF NOT EXISTS markets (
        ticker TEXT PRIMARY KEY,
        open_ts INTEGER NOT NULL,
        close_ts INTEGER NOT NULL,
        result TEXT NOT NULL,
        raw_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS candles (
        ticker TEXT NOT NULL,
        end_ts INTEGER NOT NULL,
        yes_ask_open REAL,
        yes_ask_close REAL,
        yes_ask_low REAL,
        yes_bid_open REAL,
        yes_bid_close REAL,
        yes_bid_high REAL,
        trade_low REAL,
        raw_json TEXT NOT NULL,
        PRIMARY KEY (ticker, end_ts)
    );

    CREATE INDEX IF NOT EXISTS candle_market
        ON candles(ticker, end_ts);
    """)

    return db


def price(candle, side, field):
    obj = candle.get(side) or {}
    raw = obj.get(
        field + "_dollars",
        obj.get(field)
    )

    if raw is None or raw == "":
        return None

    try:
        value = float(raw)
        return (
            value
            if math.isfinite(value) and 0 <= value <= 1
            else None
        )
    except (TypeError, ValueError):
        return None


def taker_fee(p, qty, multiplier):
    p = Decimal(str(p))
    raw = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * Decimal(qty)
        * p
        * (1 - p)
    )
    return float(
        raw.quantize(
            Decimal("0.01"),
            rounding=ROUND_CEILING
        )
    )


def backfill(db, count):
    print("PUBLIC KALSHI HISTORICAL DATA")
    print("READ ONLY. LOCAL SQLITE. NO ORDERS.", flush=True)

    chosen = []
    cursor = None

    for _ in range(5):
        query = {
            "series_ticker": "KXBTC15M",
            "status": "settled",
            "limit": min(200, max(100, count))
        }
        if cursor:
            query["cursor"] = cursor

        response = api("/markets", query)

        for market in response.get("markets", []):
            if market.get("result") not in ("yes", "no"):
                continue

            opened = seconds(market.get("open_time"))
            closed = seconds(market.get("close_time"))

            if (
                opened is None
                or closed is None
                or opened >= closed
            ):
                continue

            chosen.append((market, opened, closed))

            if len(chosen) >= count:
                break

        if len(chosen) >= count or not response.get("cursor"):
            break

        cursor = response["cursor"]

    if not chosen:
        raise RuntimeError(
            "No settled markets returned by the API."
        )

    print(
        f"Found {len(chosen)} settled markets.",
        flush=True
    )

    failures = []

    for i, (market, opened, closed) in enumerate(chosen, 1):
        ticker = market["ticker"]

        existing = db.execute(
            "SELECT COUNT(*) FROM candles WHERE ticker=?",
            (ticker,)
        ).fetchone()[0]

        if existing:
            continue

        query = {
            "start_ts": opened - 120,
            "end_ts": closed + 120,
            "period_interval": 1
        }

        try:
            data = api(
                "/series/KXBTC15M/markets/"
                + urllib.parse.quote(ticker, safe="")
                + "/candlesticks",
                query
            )

            candles = data.get("candlesticks", [])

            if not candles:
                raise ValueError("No candle data returned")

            with db:
                db.execute(
                    """INSERT OR REPLACE INTO markets
                    VALUES (?,?,?,?,?)""",
                    (
                        ticker,
                        opened,
                        closed,
                        market["result"],
                        json.dumps(market)
                    )
                )

                for candle in candles:
                    end = candle.get("end_period_ts")
                    if end is None:
                        continue

                    db.execute(
                        """INSERT OR REPLACE INTO candles
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (
                            ticker,
                            int(end),
                            price(candle, "yes_ask", "open"),
                            price(candle, "yes_ask", "close"),
                            price(candle, "yes_ask", "low"),
                            price(candle, "yes_bid", "open"),
                            price(candle, "yes_bid", "close"),
                            price(candle, "yes_bid", "high"),
                            price(candle, "price", "low"),
                            json.dumps(candle)
                        )
                    )

        except Exception as exc:
            failures.append((ticker, str(exc)))

        if i % 10 == 0 or i == len(chosen):
            print(
                f"Processed {i}/{len(chosen)} "
                f"| Errors: {len(failures)}",
                flush=True
            )

        time.sleep(0.07)

    print(
        "Stored markets:",
        db.execute(
            "SELECT COUNT(*) FROM markets"
        ).fetchone()[0]
    )

    print(
        "Stored candles:",
        db.execute(
            "SELECT COUNT(*) FROM candles"
        ).fetchone()[0]
    )

    for ticker, error in failures[:5]:
        print("FETCH ERROR:", ticker, error[:150])

    if failures and len(failures) == len(chosen):
        raise RuntimeError(
            "All market candle requests failed."
        )


def ask_quotes(row, mode):
    yes_ask = row["yes_ask_" + mode]
    yes_bid = row["yes_bid_" + mode]

    if (
        yes_ask is None
        or yes_bid is None
        or not 0 < yes_ask <= 1
        or not 0 <= yes_bid < 1
        or yes_ask < yes_bid
    ):
        return []

    return [
        ("yes", yes_ask),
        ("no", round(1 - yes_bid, 6))
    ]


def time_bucket(seconds_left):
    if seconds_left < 60:
        return "<1m"
    if seconds_left < 180:
        return "1-3m"
    if seconds_left < 300:
        return "3-5m"
    return "5m+"


def analyze(db, qty, multiplier, min_left):
    db.row_factory = sqlite3.Row

    markets = db.execute(
        "SELECT * FROM markets ORDER BY close_ts"
    ).fetchall()

    # 0.009999 means strictly LESS THAN 1 cent.
    thresholds = [
        0.0025,
        0.005,
        0.009999,
        0.01,
        0.02,
        0.03
    ]

    methods = [
        "intraminute_low",
        "same_bar_close",
        "next_bar_open"
    ]

    results = {}
    market_count = 0
    examples = []

    for market in markets:
        ticker = market["ticker"]
        winner = market["result"]
        opened = market["open_ts"]
        closes = market["close_ts"]

        rows = db.execute(
            "SELECT * FROM candles "
            "WHERE ticker=? ORDER BY end_ts",
            (ticker,)
        ).fetchall()

        if not rows:
            continue

        market_count += 1

        for method in methods:
            for limit in thresholds:
                candidates = []

                for i, row in enumerate(rows):
                    t = row["end_ts"]

                    if not opened <= t <= closes:
                        continue

                    if method == "intraminute_low":
                        # Best historical ask seen sometime
                        # within the minute. NOT a known fill.
                        quotes = []

                        a = row["yes_ask_low"]
                        b = row["yes_bid_high"]

                        if a is not None and 0 < a <= 1:
                            quotes.append(("yes", a))

                        if b is not None and 0 <= b < 1:
                            quotes.append((
                                "no", round(1 - b, 6)
                            ))

                        eligible = [
                            (s, p, t)
                            for s, p in quotes
                            if 0 < p <= limit
                            and closes - t >= min_left
                        ]

                    elif method == "same_bar_close":
                        eligible = [
                            (s, p, t)
                            for s, p in ask_quotes(row, "close")
                            if 0 < p <= limit
                            and closes - t >= min_left
                        ]

                    else:
                        # Require a cheap closing quote,
                        # then a cheap NEXT-minute open.
                        if i + 1 >= len(rows):
                            continue

                        nxt = rows[i + 1]

                        if not 0 < nxt["end_ts"] - t <= 90:
                            continue

                        entry_t = nxt["end_ts"] - 60

                        if (
                            entry_t < t
                            or closes - entry_t < min_left
                        ):
                            continue

                        prior = {
                            s for s, p in ask_quotes(row, "close")
                            if 0 < p <= limit
                        }

                        eligible = [
                            (s, p, entry_t)
                            for s, p in ask_quotes(nxt, "open")
                            if s in prior and 0 < p <= limit
                        ]

                    candidates.extend(eligible)

                if not candidates:
                    continue

                # At most one entry per 15-minute market.
                candidates.sort(
                    key=lambda x: (x[2], x[1], x[0])
                )

                earliest = candidates[0][2]
                first = [
                    c for c in candidates
                    if c[2] == earliest
                ]

                if len({c[0] for c in first}) != 1:
                    continue

                side, entry_price, entry_time = first[0]

                won = side == winner

                invested = (
                    qty * entry_price
                    + taker_fee(
                        entry_price, qty, multiplier
                    )
                )

                payout = qty if won else 0
                pnl = payout - invested

                key = (method, limit)

                if key not in results:
                    results[key] = {
                        "count": 0,
                        "wins": 0,
                        "invested": 0.0,
                        "payout": 0.0,
                        "pnl": 0.0,
                        "timing": Counter()
                    }

                r = results[key]
                r["count"] += 1
                r["wins"] += int(won)
                r["invested"] += invested
                r["payout"] += payout
                r["pnl"] += pnl
                r["timing"][
                    time_bucket(closes - entry_time)
                ] += 1

                if (
                    method == "next_bar_open"
                    and limit == 0.01
                ):
                    examples.append((
                        ticker, side, entry_price,
                        won, closes - entry_time
                    ))

    print("\n=== BTC 15-MINUTE PENNY BACKTEST ===")
    print("Markets:", market_count)
    print("Contracts per hypothetical entry:", qty)
    print("Assumed fee multiplier:", multiplier)
    print("Minimum seconds remaining:", min_left)
    print("One trade per market, held to settlement.")
    print("All results are quote-based simulations.\n")

    print(
        "Method          Price     Trades  Wins  "
        "Win Rate    Avg Cost   Net P&L      ROI"
    )

    for method in methods:
        for limit in thresholds:
            label = (
                "<1.00c"
                if limit == 0.009999
                else f"<={limit * 100:.2f}c"
            )

            r = results.get((method, limit))

            if not r:
                print(
                    f"{method:15} {label:>7}"
                    f"       0     -         -"
                )
                continue

            n = r["count"]
            win_rate = 100 * r["wins"] / n
            avg_cost = r["invested"] / n
            roi = 100 * r["pnl"] / r["invested"]

            print(
                f"{method:15} {label:>7}"
                f" {n:7} {r['wins']:5}"
                f" {win_rate:8.2f}%"
                f" ${avg_cost:9.3f}"
                f" ${r['pnl']:+9.2f}"
                f" {roi:+8.1f}%"
            )

    key = ("next_bar_open", 0.01)
    if key in results:
        print("\n<=1c next-bar entries by time remaining:")
        print(dict(sorted(results[key]["timing"].items())))

        print("\nExample hypothetical trades:")
        for item in examples[:10]:
            print(item)

    if market_count < 200:
        print(
            "\nWARNING: Small sample. Rare-win estimates "
            "will have high uncertainty."
        )

    print(
        "\nLIMITATIONS:"
        "\n- Intraminute lows have lookahead bias."
        "\n- Candle closes are not verified fills."
        "\n- Next-minute opens are delayed quote proxies."
        "\n- Historical depth and actual fillability unknown."
        "\n- Fees assume cent rounding and specified multiplier."
        "\n- No orders submitted; no Supabase operations."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode", choices=("collect", "analyze")
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB
    )
    parser.add_argument(
        "--markets", type=int, default=96
    )
    parser.add_argument(
        "--qty", type=int, default=100
    )
    parser.add_argument(
        "--fee-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--min-seconds-left", type=int, default=60
    )

    args = parser.parse_args()

    if (
        args.markets < 1
        or args.qty < 1
        or args.fee_multiplier < 0
        or args.min_seconds_left < 0
    ):
        parser.error("Invalid numeric parameter")

    db = connect(args.db)

    try:
        if args.mode == "collect":
            backfill(db, args.markets)
        else:
            analyze(
                db,
                args.qty,
                args.fee_multiplier,
                args.min_seconds_left
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
