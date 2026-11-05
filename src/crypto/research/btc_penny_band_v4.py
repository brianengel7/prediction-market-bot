"""Read-only price-band study using local KXBTC15M SQLite candles."""

import argparse
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

DEFAULT_DB = Path(".collector_runs/btc15m_pennies.sqlite3")

# Non-overlapping price bands and cumulative comparisons.
BANDS = (
    ("<=0.5c", 0, 0.005),
    (">0.5-1c", 0.005, 0.01),
    (">1-1.5c", 0.01, 0.015),
    (">1.5-2c", 0.015, 0.02),
    (">1-2c", 0.01, 0.02),
    (">2-3c", 0.02, 0.03),
    ("<=1c", 0, 0.01),
    ("<=2c", 0, 0.02),
)


def day(ts):
    return datetime.fromtimestamp(
        ts, timezone.utc
    ).strftime("%Y-%m-%d")


def taker_fee(price, contracts, multiplier):
    p = Decimal(str(price))

    raw = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * Decimal(contracts)
        * p * (1 - p)
    )

    return float(raw.quantize(
        Decimal("0.01"),
        rounding=ROUND_CEILING
    ))


def wilson_interval(wins, n):
    if n == 0:
        return None

    z = 1.95996398454
    p = wins / n
    denom = 1 + z * z / n

    mid = (p + z * z / (2 * n)) / denom

    half = (
        z * math.sqrt(
            (p * (1 - p) + z * z / (4 * n)) / n
        ) / denom
    )

    return (
        100 * (mid - half),
        100 * (mid + half)
    )


def metrics(trades, contracts):
    n = len(trades)

    if n == 0:
        return None

    wins = sum(t["won"] for t in trades)
    invested = sum(t["cost"] for t in trades)
    pnl = contracts * wins - invested

    running = 0.0
    peak = 0.0
    max_dd = 0.0

    for trade in trades:
        running += trade["pnl"]
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)

    return {
        "n": n,
        "wins": wins,
        "wr": 100 * wins / n,
        "invested": invested,
        "pnl": pnl,
        "roi": 100 * pnl / invested,
        "breakeven": 100 * invested / (n * contracts),
        "max_dd": max_dd,
        "ci": wilson_interval(wins, n)
    }


def show(label, trades, contracts):
    r = metrics(trades, contracts)

    if r is None:
        print(f"{label:<22} | trades=0")
        return

    print(
        f"{label:<22} "
        f"| trades={r['n']:>4} "
        f"| wins={r['wins']:>3} "
        f"| WR={r['wr']:>5.2f}% "
        f"| BE={r['breakeven']:>5.2f}% "
        f"| PnL=${r['pnl']:>+9.2f} "
        f"| ROI={r['roi']:>+7.1f}% "
        f"| maxDD=${r['max_dd']:>7.2f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB
    )
    parser.add_argument(
        "--markets", type=int, default=5000
    )
    parser.add_argument(
        "--qty", type=int, default=100
    )
    parser.add_argument(
        "--fee-multiplier", type=float, default=1
    )
    parser.add_argument(
        "--min-seconds-left", type=int, default=60
    )
    parser.add_argument(
        "--block", type=int, default=500
    )
    parser.add_argument(
        "--older-holdout", type=int, default=2000
    )

    args = parser.parse_args()

    if (
        args.markets < 1
        or args.qty < 1
        or args.block < 1
        or args.older_holdout < 0
        or args.fee_multiplier < 0
        or args.min_seconds_left < 0
    ):
        parser.error("Invalid numeric parameters")

    if not args.db.is_file():
        parser.error("SQLite database not found.")

    db = sqlite3.connect(
        args.db.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    db.row_factory = sqlite3.Row

    try:
        markets = db.execute(
            "SELECT ticker, open_ts, close_ts, result "
            "FROM markets "
            "WHERE result IN ('yes','no') "
            "ORDER BY close_ts, ticker"
        ).fetchall()

        markets = markets[-args.markets:]

        if not markets:
            raise RuntimeError("No settled markets found.")

        chosen = defaultdict(list)
        incomplete = 0
        invalid_rows = 0

        for market_idx, market in enumerate(markets):
            rows = db.execute(
                "SELECT end_ts, yes_bid_close, yes_ask_close "
                "FROM candles WHERE ticker=? "
                "ORDER BY end_ts",
                (market["ticker"],)
            ).fetchall()

            if len(rows) < 12:
                incomplete += 1
                continue

            candidates = defaultdict(list)

            for row in rows:
                t = int(row["end_ts"])

                if not (
                    market["open_ts"]
                    <= t
                    <= market["close_ts"]
                    - args.min_seconds_left
                ):
                    continue

                bid = row["yes_bid_close"]
                ask = row["yes_ask_close"]

                if bid is None or ask is None:
                    continue

                bid = float(bid)
                ask = float(ask)

                if not 0 <= bid <= ask <= 1:
                    invalid_rows += 1
                    continue

                quotes = (
                    ("yes", ask),
                    ("no", 1 - bid)
                )

                for side, p in quotes:
                    if p <= 0:
                        continue

                    p = round(p, 6)

                    for label, low, high in BANDS:
                        if low < p <= high + 1e-9:
                            candidates[label].append(
                                (t, side, p)
                            )

            for label, _, _ in BANDS:
                possible = candidates[label]

                if not possible:
                    continue

                possible.sort(
                    key=lambda x: (x[0], x[1], x[2])
                )

                entry_time = possible[0][0]

                # Avoid contradictory simultaneous entries.
                sides = {
                    side
                    for t, side, _ in possible
                    if t == entry_time
                }

                if len(sides) > 1:
                    continue

                t, side, p = possible[0]

                cost = (
                    args.qty * p
                    + taker_fee(
                        p, args.qty,
                        args.fee_multiplier
                    )
                )

                won = int(side == market["result"])

                chosen[label].append({
                    "market_idx": market_idx,
                    "ticker": market["ticker"],
                    "close_ts": int(market["close_ts"]),
                    "price": p,
                    "side": side,
                    "left_s": int(
                        market["close_ts"] - t
                    ),
                    "won": won,
                    "cost": cost,
                    "pnl": (
                        args.qty if won else 0
                    ) - cost
                })

        print("\n=== BTC 5000-MARKET PRICE-BAND STUDY ===")
        print("Database:", args.db)
        print("Markets analyzed:", len(markets))
        print(
            "Date range:",
            day(markets[0]["close_ts"]),
            "to",
            day(markets[-1]["close_ts"])
        )
        print("Incomplete markets:", incomplete)
        print("Invalid quote rows:", invalid_rows)
        print("Contracts per trade:", args.qty)
        print("Minimum time remaining:", args.min_seconds_left)
        print("Fee multiplier:", args.fee_multiplier)

        if len(markets) < args.markets:
            print(
                "WARNING: Fewer markets than requested."
            )

        print("\n=== DIRECT PRICE-BAND RESULTS ===")

        for label, _, _ in BANDS:
            show(label, chosen[label], args.qty)

        label = ">1-2c"

        print("\n=== 1-2c CHRONOLOGICAL BLOCKS ===")

        for start in range(0, len(markets), args.block):
            end = min(start + args.block, len(markets))

            group = [
                trade for trade in chosen[label]
                if start <= trade["market_idx"] < end
            ]

            period = (
                day(markets[start]["close_ts"])
                + " to "
                + day(markets[end - 1]["close_ts"])
            )

            show(period, group, args.qty)

        if (
            args.older_holdout > 0
            and len(markets) >= args.older_holdout + 100
        ):
            print(
                "\n=== OLDER HISTORY VS RECENT HISTORY ==="
            )
            print(
                "Retrospective validation, "
                "not forward out-of-sample testing."
            )

            splits = (
                (
                    f"Older {args.older_holdout}",
                    0, args.older_holdout
                ),
                (
                    f"Recent {len(markets)-args.older_holdout}",
                    args.older_holdout, len(markets)
                )
            )

            for name, start, end in splits:
                group = [
                    trade for trade in chosen[label]
                    if start <= trade["market_idx"] < end
                ]

                show(name, group, args.qty)

        print("\n=== 1-2c BY TIME REMAINING ===")

        buckets = (
            ("60-119s", 60, 120),
            ("120-179s", 120, 180),
            ("180-299s", 180, 300),
            ("300-599s", 300, 600),
            ("600s+", 600, 100000)
        )

        for name, low, high in buckets:
            group = [
                trade for trade in chosen[label]
                if low <= trade["left_s"] < high
            ]

            show(name, group, args.qty)

        print("\n=== WINNING 1-2c TRADES ===")

        for trade in chosen[label]:
            if not trade["won"]:
                continue

            print(
                day(trade["close_ts"]),
                trade["ticker"],
                trade["side"].upper(),
                f"ask={trade['price']*100:.3f}c",
                f"left={trade['left_s']}s"
            )

        summary = metrics(chosen[label], args.qty)

        if summary:
            lower, upper = summary["ci"]

            print(
                "\n1-2c win-rate 95% Wilson interval:",
                f"{lower:.2f}% to {upper:.2f}%"
            )

        print("\nREAD-ONLY STUDY.")
        print("Entry quotes are not verified fills.")
        print("Historical order-book depth unavailable.")
        print("Bands are independent and may overlap.")
        print("No live orders or database modifications.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
