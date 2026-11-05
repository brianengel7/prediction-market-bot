"""Expand KXBTC15M SQLite history and report chronological results."""

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from btc_penny_local import (
    api, connect, price, seconds, taker_fee, ask_quotes
)

DEFAULT_DB = Path(".collector_runs/btc15m_pennies.sqlite3")
SERIES = "KXBTC15M"


def date(ts):
    return datetime.fromtimestamp(
        ts, timezone.utc
    ).strftime("%Y-%m-%d")


def list_markets(path, target, found, source):
    cursor = None
    seen = set()

    for page in range(1, 201):
        query = {
            "series_ticker": SERIES,
            "limit": 250
        }

        if source == "live":
            query["status"] = "settled"

        if cursor:
            query["cursor"] = cursor

        data = api(path, query)
        batch = data.get("markets", [])

        for market in batch:
            ticker = market.get("ticker", "")
            opened = seconds(market.get("open_time"))
            closed = seconds(market.get("close_time"))

            if not ticker.startswith(SERIES + "-"):
                continue

            if market.get("result") not in ("yes", "no"):
                continue

            if (
                opened is None
                or closed is None
                or opened >= closed
            ):
                continue

            if ticker not in found:
                found[ticker] = (
                    market, opened, closed, source
                )

        print(
            f"[{source}] Page {page} "
            f"| markets found: {len(found)}",
            flush=True
        )

        if len(found) >= target:
            return

        cursor = data.get("cursor")

        if not cursor:
            return

        if cursor in seen:
            raise RuntimeError("Repeated pagination cursor")

        seen.add(cursor)

        if not batch:
            return

    raise RuntimeError("Pagination safety limit reached")


def collect(db, target):
    existing = db.execute(
        "SELECT COUNT(*) FROM markets"
    ).fetchone()[0]

    print("\n=== EXPANDING BTC PENNY DATABASE ===")
    print("Existing markets:", existing)
    print("Target:", target)
    print("READ ONLY API. NO TRADING.\n")

    cutoff = api("/historical/cutoff")
    print(
        "Historical cutoff:",
        cutoff.get("market_settled_ts")
    )

    found = {}

    list_markets(
        "/markets", target, found, "live"
    )

    if len(found) < target:
        list_markets(
            "/historical/markets",
            target, found, "historical"
        )

    selected = sorted(
        found.values(),
        key=lambda x: x[2],
        reverse=True
    )[:target]

    print("\nSelected:", len(selected))

    if len(selected) < target:
        print("WARNING: Fewer markets than requested.")

    cached = dict(db.execute(
        "SELECT ticker, COUNT(*) "
        "FROM candles GROUP BY ticker"
    ).fetchall())

    skipped = 0
    added = 0
    errors = []

    for i, (market, opened, closed, source) in enumerate(
        selected, 1
    ):
        ticker = market["ticker"]

        if cached.get(ticker, 0) >= 12:
            skipped += 1
            continue

        if source == "historical":
            endpoint = (
                "/historical/markets/"
                + quote(ticker, safe="")
                + "/candlesticks"
            )
        else:
            endpoint = (
                "/series/KXBTC15M/markets/"
                + quote(ticker, safe="")
                + "/candlesticks"
            )

        try:
            data = api(endpoint, {
                "start_ts": opened - 120,
                "end_ts": closed + 120,
                "period_interval": 1
            })

            values = []

            for candle in data.get("candlesticks", []):
                end = candle.get("end_period_ts")

                if end is None:
                    continue

                values.append((
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
                ))

            if not values:
                raise ValueError("Empty candle response")

            with db:
                db.execute(
                    "INSERT OR REPLACE INTO markets "
                    "VALUES (?,?,?,?,?)",
                    (
                        ticker,
                        opened,
                        closed,
                        market["result"],
                        json.dumps(market)
                    )
                )

                if ticker in cached:
                    db.execute(
                        "DELETE FROM candles WHERE ticker=?",
                        (ticker,)
                    )

                db.executemany(
                    "INSERT OR REPLACE INTO candles "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    values
                )

            added += 1

        except Exception as exc:
            errors.append((ticker, str(exc)))

        if i % 50 == 0 or i == len(selected):
            print(
                f"Processed {i}/{len(selected)} "
                f"| reused={skipped} "
                f"| added={added} "
                f"| errors={len(errors)}",
                flush=True
            )

        time.sleep(0.06)

    print("\n=== DATABASE STATUS ===")
    print(
        "Markets:",
        db.execute(
            "SELECT COUNT(*) FROM markets"
        ).fetchone()[0]
    )
    print(
        "Candles:",
        db.execute(
            "SELECT COUNT(*) FROM candles"
        ).fetchone()[0]
    )
    print("Newly downloaded:", added)
    print("Reused:", skipped)
    print("Errors:", len(errors))

    for ticker, error in errors[:10]:
        print("ERROR:", ticker, error[:160])

    if errors:
        print(
            "Rerun the same collection command "
            "to retry missing markets."
        )


def summarize(label, trades, qty):
    n = len(trades)
    cost = sum(t["cost"] for t in trades)
    wins = sum(t["won"] for t in trades)

    pnl = qty * wins - cost
    roi = 100 * pnl / cost if cost > 0 else None

    if roi is None:
        print(f"{label:<25} | no trades")
        return

    print(
        f"{label:<25} | "
        f"trades={n:>4} "
        f"| wins={wins:>3} "
        f"| WR={100 * wins / n:>5.2f}% "
        f"| P&L=${pnl:>+9.2f} "
        f"| ROI={roi:>+7.1f}%"
    )


def report(db, qty, multiplier, block):
    db.row_factory = sqlite3.Row

    markets = db.execute(
        "SELECT * FROM markets "
        "ORDER BY close_ts, ticker"
    ).fetchall()

    print("\n=== CHRONOLOGICAL PENNY STUDY ===")
    print("Markets stored:", len(markets))
    print("Contracts per trade:", qty)
    print("Fee multiplier:", multiplier)

    if not markets:
        return

    print(
        "Date range:",
        date(markets[0]["close_ts"]),
        "to",
        date(markets[-1]["close_ts"])
    )

    trades = []
    complete = []

    for market in markets:
        candles = db.execute(
            "SELECT * FROM candles "
            "WHERE ticker=? ORDER BY end_ts",
            (market["ticker"],)
        ).fetchall()

        if len(candles) < 12:
            continue

        complete.append(market)
        candidates = []

        for row in candles:
            ts = row["end_ts"]

            if not (
                market["open_ts"]
                <= ts
                <= market["close_ts"] - 60
            ):
                continue

            for side, ask in ask_quotes(row, "close"):
                if 0 < ask <= 0.010000001:
                    candidates.append((
                        ts, side, ask
                    ))

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (x[0], x[1])
        )

        first_time = candidates[0][0]
        first_sides = {
            side
            for ts, side, _ in candidates
            if ts == first_time
        }

        if len(first_sides) != 1:
            continue

        entry_ts, side, ask = candidates[0]

        trades.append({
            "ticker": market["ticker"],
            "close": market["close_ts"],
            "side": side,
            "won": int(side == market["result"]),
            "ask": ask,
            "entry_ts": entry_ts,
            "left": market["close_ts"] - entry_ts,
            "cost": (
                qty * ask
                + taker_fee(ask, qty, multiplier)
            )
        })

    print("Markets with sufficient candles:", len(complete))

    print("\n=== OVERALL RESULTS ===")
    summarize("All <=1c trades", trades, qty)

    print("\n=== ENTRY PRICE BANDS ===")
    summarize(
        "<=0.5c",
        [t for t in trades if t["ask"] <= 0.005],
        qty
    )
    summarize(
        ">0.5c to <=1c",
        [t for t in trades if t["ask"] > 0.005],
        qty
    )

    print("\n=== CHRONOLOGICAL BLOCKS ===")

    for start in range(0, len(complete), block):
        subset = complete[start:start + block]

        first = subset[0]["close_ts"]
        last = subset[-1]["close_ts"]

        group = [
            t for t in trades
            if first <= t["close"] <= last
        ]

        summarize(
            date(first) + " to " + date(last),
            group,
            qty
        )

    print("\n=== WINNING TRADES ===")

    for trade in trades:
        if not trade["won"]:
            continue

        print(
            date(trade["close"]),
            trade["ticker"],
            trade["side"].upper(),
            f"ask={trade['ask']*100:.3f}c",
            f"left={trade['left']}s"
        )

    print("\nRESEARCH LIMITATIONS:")
    print("- Candle closing asks are not verified fills.")
    print("- Historical order-book depth is unavailable.")
    print("- Slippage and missed fills are unmodeled.")
    print("- Fee multiplier is assumed.")
    print("- No orders submitted.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("collect", "report")
    )
    parser.add_argument(
        "--markets", type=int, default=3000
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB
    )
    parser.add_argument(
        "--qty", type=int, default=100
    )
    parser.add_argument(
        "--fee-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--block", type=int, default=500
    )

    args = parser.parse_args()

    if (
        args.markets < 1
        or args.qty < 1
        or args.block < 1
        or args.fee_multiplier < 0
    ):
        parser.error("Invalid numeric parameter")

    db = connect(args.db)

    try:
        if args.mode == "collect":
            collect(db, args.markets)
        else:
            report(
                db, args.qty,
                args.fee_multiplier, args.block
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
