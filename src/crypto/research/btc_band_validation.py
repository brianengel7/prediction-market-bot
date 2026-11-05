"""Read-only validation of KXBTC15M 1.5–2.0 cent strategies."""

import argparse
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

DB = Path(".collector_runs/btc15m_pennies.sqlite3")
LOW, HIGH = 0.015, 0.02


def fee(price, qty, multiplier):
    p = Decimal(str(price))
    value = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * Decimal(qty)
        * p * (1 - p)
    )
    return float(value.quantize(
        Decimal("0.01"),
        rounding=ROUND_CEILING
    ))


def quotes(row):
    bid = row["yes_bid_close"]
    ask = row["yes_ask_close"]

    if bid is None or ask is None:
        return []

    bid = round(float(bid), 6)
    ask = round(float(ask), 6)

    if not 0 <= bid <= ask <= 1:
        return []

    return [
        ("yes", ask),
        ("no", round(1 - bid, 6))
    ]


def first_unique(candidates):
    if not candidates:
        return None

    candidates.sort(
        key=lambda v: (v[0], v[1], v[2])
    )

    first_t = candidates[0][0]

    if len({
        x[1] for x in candidates
        if x[0] == first_t
    }) != 1:
        return None

    return candidates[0]


def summarize(title, rows, qty, multiplier, slip=0):
    if not rows:
        print(f"{title:<32} n=0")
        return

    stake = sum(
        qty * (r["price"] + slip)
        + fee(
            r["price"] + slip,
            qty,
            multiplier
        )
        for r in rows
    )

    wins = sum(r["won"] for r in rows)
    pnl = qty * wins - stake

    print(
        f"{title:<32} "
        f"n={len(rows):>4} "
        f"wins={wins:>3} "
        f"WR={100*wins/len(rows):>5.2f}% "
        f"BE={100*stake/(len(rows)*qty):>5.2f}% "
        f"PnL=${pnl:>+8.2f} "
        f"ROI={100*pnl/stake:>+7.1f}%"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--db", type=Path, default=DB
    )
    ap.add_argument(
        "--qty", type=int, default=100
    )
    ap.add_argument(
        "--fee-multiplier",
        type=float, default=1.0
    )
    ap.add_argument(
        "--min-seconds-left",
        type=int, default=60
    )

    args = ap.parse_args()

    if (
        args.qty <= 0
        or args.fee_multiplier < 0
        or args.min_seconds_left < 0
    ):
        ap.error("Invalid parameters")

    if not args.db.exists():
        ap.error("SQLite database not found")

    db = sqlite3.connect(
        args.db.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    db.row_factory = sqlite3.Row

    markets = db.execute(
        "SELECT ticker, open_ts, close_ts, result "
        "FROM markets "
        "WHERE result IN ('yes','no') "
        "ORDER BY close_ts,ticker"
    ).fetchall()

    strategies = defaultdict(list)
    incomplete = 0

    for market_idx, m in enumerate(markets):
        candles = db.execute(
            "SELECT end_ts,yes_bid_close,yes_ask_close "
            "FROM candles WHERE ticker=? "
            "ORDER BY end_ts",
            (m["ticker"],)
        ).fetchall()

        if len(candles) < 12:
            incomplete += 1
            continue

        series = []
        by_time = {}

        for r in candles:
            t = int(r["end_ts"])

            if not (
                m["open_ts"] <= t
                <= m["close_ts"] - args.min_seconds_left
            ):
                continue

            for side, p in quotes(r):
                if 0 < p <= 1:
                    series.append((t, side, p))

            by_time[t] = quotes(r)

        first_band = first_unique([
            x for x in series
            if LOW < x[2] <= HIGH
        ])

        first_below2 = first_unique([
            x for x in series
            if x[2] <= HIGH
        ])

        selections = [
            ("first_in_band", first_band)
        ]

        if (
            first_below2
            and LOW < first_below2[2] <= HIGH
        ):
            selections.append(
                ("first_cheap_in_band", first_below2)
            )

        for kind, entry in selections:
            if not entry:
                continue

            t, side, p = entry

            prior_lower = any(
                ts < t and ss == side and pp <= LOW
                for ts, ss, pp in series
            )

            item = {
                "idx": market_idx,
                "ticker": m["ticker"],
                "time_left": m["close_ts"] - t,
                "won": int(side == m["result"]),
                "price": p,
                "prior_lower": prior_lower,
            }

            strategies[kind].append(item)

            if kind == "first_in_band":
                later_time = t + 60

                if later_time < m["close_ts"]:
                    later_prices = dict(
                        by_time.get(later_time, [])
                    )
                    later_p = later_prices.get(side)

                    if later_p and LOW < later_p <= HIGH:
                        strategies[
                            "stays_in_band_60s"
                        ].append({
                            **item,
                            "price": later_p
                        })

    print("\n=== 1.5-2 CENT ENTRY VALIDATION ===")
    print("Database:", args.db)
    print("Markets:", len(markets))
    print("Incomplete:", incomplete)
    print("READ ONLY - NO ORDERS OR WRITES")

    print("\n=== ENTRY RULE COMPARISON ===")

    for name in (
        "first_in_band",
        "first_cheap_in_band",
        "stays_in_band_60s"
    ):
        summarize(
            name,
            strategies[name],
            args.qty,
            args.fee_multiplier
        )

    original = strategies["first_in_band"]

    summarize(
        "No earlier <=1.5c quote",
        [r for r in original if not r["prior_lower"]],
        args.qty, args.fee_multiplier
    )

    summarize(
        "Earlier <=1.5c quote",
        [r for r in original if r["prior_lower"]],
        args.qty, args.fee_multiplier
    )

    print("\n=== CHRONOLOGICAL 500-MARKET BLOCKS ===")

    for start in range(0, len(markets), 500):
        end = min(start + 500, len(markets))

        d1 = datetime.fromtimestamp(
            markets[start]["close_ts"],
            timezone.utc
        ).strftime("%b %d")

        d2 = datetime.fromtimestamp(
            markets[end - 1]["close_ts"],
            timezone.utc
        ).strftime("%b %d")

        group = [
            r for r in original
            if start <= r["idx"] < end
        ]

        summarize(
            f"{d1} to {d2}",
            group,
            args.qty, args.fee_multiplier
        )

    print("\n=== OLDER VS NEWER MARKETS ===")

    cut = min(2000, len(markets))

    summarize(
        "Older markets",
        [r for r in original if r["idx"] < cut],
        args.qty, args.fee_multiplier
    )

    summarize(
        "Newer markets",
        [r for r in original if r["idx"] >= cut],
        args.qty, args.fee_multiplier
    )

    print("\n=== TIME REMAINING ===")

    buckets = (
        ("60-119s", 60, 120),
        ("120-179s", 120, 180),
        ("180-299s", 180, 300),
        ("300-599s", 300, 600),
        ("600s+", 600, 99999)
    )

    for name, low, high in buckets:
        summarize(
            name,
            [
                r for r in original
                if low <= r["time_left"] < high
            ],
            args.qty, args.fee_multiplier
        )

    print("\n=== EXTRA ENTRY COST STRESS TEST ===")

    for slip in (0, 0.001, 0.002, 0.003, 0.004):
        summarize(
            f"+{slip*100:.1f}c per contract",
            original,
            args.qty,
            args.fee_multiplier,
            slip=slip
        )

    print("\nLIMITATIONS:")
    print("Historical candle-close asks, not verified fills.")
    print("No historical executable depth.")
    print("One-minute persistence is a delayed quote proxy.")
    print("Slippage stress may exceed a 2c buy limit.")
    print("Historical splits are retrospective, not new OOS.")
    print("No orders or database writes.")

    db.close()


if __name__ == "__main__":
    main()
