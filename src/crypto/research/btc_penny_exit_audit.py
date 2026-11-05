"""Read-only penny reversal early-exit audit."""

import argparse
import sqlite3
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


def fee(p, qty, multiplier):
    p = Decimal(str(p))
    raw = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * Decimal(qty)
        * p * (1 - p)
    )
    return float(raw.quantize(
        Decimal("0.01"),
        rounding=ROUND_CEILING
    ))


def entry_asks(row):
    bid = row["yes_bid_close"]
    ask = row["yes_ask_close"]

    if bid is None or ask is None:
        return []

    if not 0 <= bid <= ask <= 1:
        return []

    return [
        ("yes", float(ask)),
        ("no", round(1 - float(bid), 6))
    ]


def bid_value(row, side, mode):
    if side == "yes":
        field = (
            "yes_bid_high"
            if mode == "touch"
            else "yes_bid_close"
        )
        value = row[field]
    else:
        field = (
            "yes_ask_low"
            if mode == "touch"
            else "yes_ask_close"
        )
        raw = row[field]
        value = None if raw is None else 1 - float(raw)

    if value is None:
        return None

    value = float(value)
    return value if 0 <= value <= 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(
            ".collector_runs/btc15m_pennies.sqlite3"
        )
    )
    parser.add_argument("--qty", type=int, default=100)
    parser.add_argument(
        "--fee-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--min-seconds-left", type=int, default=60
    )
    parser.add_argument(
        "--max-ask", type=float, default=0.01
    )
    args = parser.parse_args()

    if (
        args.qty <= 0
        or not 0 < args.max_ask <= 1
        or args.fee_multiplier < 0
        or args.min_seconds_left < 0
    ):
        parser.error("Invalid parameter")

    db = sqlite3.connect(
        args.db.resolve().as_uri() + "?mode=ro",
        uri=True
    )
    db.row_factory = sqlite3.Row

    markets = db.execute(
        "SELECT * FROM markets ORDER BY close_ts"
    ).fetchall()

    targets = (0.05, 0.10, 0.20, 0.50)
    strategies = [("hold", None)] + [
        (mode, target)
        for mode in ("touch", "close")
        for target in targets
    ]

    stats = {
        strategy: {
            "n": 0,
            "wins": 0,
            "early_exits": 0,
            "cost": 0.0,
            "pnl": 0.0
        }
        for strategy in strategies
    }

    for market in markets:
        candles = db.execute(
            "SELECT * FROM candles "
            "WHERE ticker=? ORDER BY end_ts",
            (market["ticker"],)
        ).fetchall()

        candidates = []

        for row in candles:
            ts = int(row["end_ts"])

            if not (
                market["open_ts"] <= ts
                <= market["close_ts"] - args.min_seconds_left
            ):
                continue

            for side, price in entry_asks(row):
                if 0 < price <= args.max_ask + 1e-10:
                    candidates.append((ts, side, price))

        if not candidates:
            continue

        candidates.sort(key=lambda x: (x[0], x[1]))
        ts, side, entry_price = candidates[0]

        # Skip contradictory simultaneous signals.
        if any(
            t == ts and s != side
            for t, s, _ in candidates
        ):
            continue

        entry_cost = (
            args.qty * entry_price
            + fee(
                entry_price,
                args.qty,
                args.fee_multiplier
            )
        )

        won = side == market["result"]

        for mode, target in strategies:
            exit_price = None

            if target is not None:
                for candle in candles:
                    ct = candle["end_ts"]

                    # Future minute only.
                    # No exits at or after market close.
                    if ct <= ts or ct >= market["close_ts"]:
                        continue

                    bid = bid_value(candle, side, mode)

                    if (
                        bid is not None
                        and bid + 1e-10 >= target
                    ):
                        exit_price = target
                        break

            if exit_price is not None:
                payout = (
                    args.qty * exit_price
                    - fee(
                        exit_price,
                        args.qty,
                        args.fee_multiplier
                    )
                )
            else:
                payout = args.qty if won else 0

            r = stats[(mode, target)]
            r["n"] += 1
            r["wins"] += int(won)
            r["early_exits"] += int(
                exit_price is not None
            )
            r["cost"] += entry_cost
            r["pnl"] += payout - entry_cost

    print("\n=== BTC PENNY EXIT AUDIT ===")
    print("Database:", args.db)
    print("Settled markets:", len(markets))
    print("Maximum entry ask:", args.max_ask)
    print("Contracts per trade:", args.qty)
    print("Assumed fee multiplier:", args.fee_multiplier)

    print(
        "\nStrategy     Target  Trades  "
        "Early Exits    Net P&L       ROI"
    )

    for mode, target in strategies:
        r = stats[(mode, target)]

        roi = (
            100 * r["pnl"] / r["cost"]
            if r["cost"] else float("nan")
        )

        label = (
            "--" if target is None
            else f"{target * 100:.0f}c"
        )

        print(
            f"{mode:<12} {label:>5} "
            f"{r['n']:>7} "
            f"{r['early_exits']:>12} "
            f"${r['pnl']:>+10.2f} "
            f"{roi:>+9.1f}%"
        )

    print("\nMETHODS:")
    print("hold: hold every position to settlement.")
    print(
        "touch: assume a resting limit sell fills "
        "when a future minute's bid high reaches target."
    )
    print(
        "close: assume a sell at target when a future "
        "minute's closing bid reaches target."
    )
    print(
        "Both exit methods are optimistic fill proxies."
    )
    print(
        "Final-minute candle exits are excluded."
    )
    print(
        "Fees: assumed taker on entry and exit."
    )
    print(
        "No confirmed liquidity, fills, or live orders."
    )
    db.close()


if __name__ == "__main__":
    main()
