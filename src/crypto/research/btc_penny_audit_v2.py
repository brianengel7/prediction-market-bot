"""Read-only SQLite audit of penny reversal entry rules."""

import argparse
import sqlite3
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


def fee(price, qty, multiplier):
    p = Decimal(str(price))
    raw = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * Decimal(qty)
        * p * (1 - p)
    )
    return float(raw.quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    ))


def asks(row):
    bid = row["yes_bid_close"]
    ask = row["yes_ask_close"]

    if bid is None or ask is None:
        return {}

    if not 0 <= bid <= ask <= 1:
        return {}

    return {
        "yes": float(ask),
        "no": float(1 - bid),
    }


def summarize(trades, qty):
    n = len(trades)
    if not n:
        return "n=0"

    wins = sum(x["won"] for x in trades)
    stake = sum(x["cost"] for x in trades)
    profit = qty * wins - stake
    win_rate = wins / n

    if wins == 0:
        upper = 1 - 0.05 ** (1 / n)
        confidence = (
            f"95% one-sided upper={upper*100:.2f}%"
        )
    else:
        confidence = "Rare wins; enlarge sample"

    return (
        f"n={n} | wins={wins} "
        f"| WR={win_rate*100:.2f}% "
        f"| cost=${stake:.2f} "
        f"| P&L=${profit:+.2f} "
        f"| ROI={100*profit/stake:+.1f}% "
        f"| {confidence}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db", type=Path,
        default=Path(
            ".collector_runs/btc15m_pennies.sqlite3"
        )
    )
    parser.add_argument(
        "--qty", type=int, default=100
    )
    parser.add_argument(
        "--min-seconds-left", type=int, default=60
    )
    parser.add_argument(
        "--fee-multiplier", type=float, default=1.0
    )

    args = parser.parse_args()

    if (
        args.qty < 1
        or args.min_seconds_left < 0
        or args.fee_multiplier < 0
    ):
        parser.error("Invalid parameters")

    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    markets = db.execute(
        "SELECT ticker, open_ts, close_ts, result "
        "FROM markets ORDER BY close_ts"
    ).fetchall()

    print("\n=== BTC PENNY REVERSAL AUDIT V2 ===")
    print("Database:", args.db)
    print("Markets:", len(markets))
    print("Contracts:", args.qty)
    print("Fee multiplier:", args.fee_multiplier)
    print("READ ONLY. NO TRADES.\n")

    for threshold in (0.005, 0.01, 0.02, 0.03):
        first = []
        per_side = []
        delayed_first = []
        delayed_per_side = []

        touched = 0
        winner_touched = 0
        both_touched = 0
        missing = 0

        for market in markets:
            rows = db.execute(
                "SELECT * FROM candles "
                "WHERE ticker=? ORDER BY end_ts",
                (market["ticker"],)
            ).fetchall()

            eligible = [
                row for row in rows
                if (
                    market["open_ts"]
                    <= row["end_ts"]
                    <= market["close_ts"]
                    - args.min_seconds_left
                )
            ]

            if not eligible:
                missing += 1
                continue

            by_time = {
                row["end_ts"]: row
                for row in eligible
            }

            triggers = []

            for row in eligible:
                for side, price in asks(row).items():
                    if 0 < price <= threshold + 1e-10:
                        triggers.append((
                            row["end_ts"],
                            side,
                            price
                        ))

            if not triggers:
                continue

            touched += 1

            sides = {x[1] for x in triggers}

            if market["result"] in sides:
                winner_touched += 1

            if len(sides) == 2:
                both_touched += 1

            triggers.sort(
                key=lambda x: (x[0], x[1])
            )

            first_signal = [triggers[0]]

            side_signals = [
                next(
                    x for x in triggers
                    if x[1] == side
                )
                for side in ("yes", "no")
                if side in sides
            ]

            def add(group, delayed_group, selected):
                for t, side, price in selected:
                    won = side == market["result"]

                    group.append({
                        "won": won,
                        "cost": (
                            args.qty * price
                            + fee(
                                price,
                                args.qty,
                                args.fee_multiplier
                            )
                        )
                    })

                    # One full minute after signal.
                    # Require same side to remain cheap.
                    later = by_time.get(t + 60)

                    if later is None:
                        continue

                    later_price = asks(later).get(side)

                    if later_price is None:
                        continue

                    if not (
                        0 < later_price
                        <= threshold + 1e-10
                    ):
                        continue

                    delayed_group.append({
                        "won": won,
                        "cost": (
                            args.qty * later_price
                            + fee(
                                later_price,
                                args.qty,
                                args.fee_multiplier
                            )
                        )
                    })

            add(
                first,
                delayed_first,
                first_signal
            )

            add(
                per_side,
                delayed_per_side,
                side_signals
            )

        print(
            f"\n=== PRICE <= {threshold*100:.2f}c ==="
        )
        print("Markets with cheap side:", touched)
        print(
            "Markets where eventual winner "
            "was ever cheap:", winner_touched
        )
        print(
            "Markets where both sides "
            "became cheap:", both_touched
        )
        print(
            "Markets without eligible candles:",
            missing
        )

        print("\nFirst cheap side per market:")
        print("  At bar close:", summarize(
            first, args.qty
        ))
        print("  After 60s:", summarize(
            delayed_first, args.qty
        ))

        print("\nFirst cheap quote PER SIDE:")
        print("  At bar close:", summarize(
            per_side, args.qty
        ))
        print("  After 60s:", summarize(
            delayed_per_side, args.qty
        ))

    db.close()

    print("\nLIMITATIONS:")
    print("Historical OHLC quotes, not verified fills.")
    print("No historical executable depth.")
    print("Delayed entry uses a later candle close.")
    print("Winner-ever-cheap uses hindsight.")
    print("Fee multiplier is an assumption.")
    print("No live trading or Supabase writes.")


if __name__ == "__main__":
    main()
