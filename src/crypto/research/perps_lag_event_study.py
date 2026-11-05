"""Read-only BTC lead/lag and executable quote replay."""

import argparse
import bisect
import json
import math
import statistics
from pathlib import Path

THRESHOLDS = (10, 20, 50, 100)
REACTION_MS = (100, 250, 500, 1000, 2000)
DELAYS_MS = (100, 250, 500)
HOLDS_MS = (1000, 3000, 5000)
CONTRACT_BTC = 0.0001


def avg(values):
    return statistics.mean(values) if values else float("nan")


def fmt(value):
    return f"{value:+.2f}" if math.isfinite(value) else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--fee-bps", type=float, default=12.0)
    ap.add_argument("--cooldown-ms", type=float, default=3000)
    ap.add_argument("--max-quote-age-ms", type=float, default=3000)
    args = ap.parse_args()

    if args.run:
        path = (
            args.run / "quotes.jsonl"
            if args.run.is_dir() else args.run
        )
    else:
        found = list(
            Path(".collector_runs").glob(
                "perps_lag_*/quotes.jsonl"
            )
        )
        if not found:
            ap.error("No perps collector data found.")
        path = max(found, key=lambda p: p.stat().st_mtime)

    books = {"coinbase": [], "kalshi": []}
    states = {"coinbase": [], "kalshi": []}

    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
                name = e.get("source")
                if name not in books:
                    continue

                ts = float(e["recv_ms"])

                if e.get("kind") == "connection":
                    states[name].append(
                        (ts, bool(e.get("connected")))
                    )

                elif e.get("kind") == "quote":
                    bid = float(e["bid_btc_usd"])
                    ask = float(e["ask_btc_usd"])

                    if (
                        0 < bid < ask
                        and math.isfinite(bid)
                        and math.isfinite(ask)
                    ):
                        books[name].append(
                            (ts, bid, ask, (bid + ask) / 2)
                        )
            except (
                ValueError, KeyError, TypeError,
                json.JSONDecodeError
            ):
                continue

    times = {}
    connection_times = {}

    for name in books:
        books[name].sort(key=lambda x: x[0])
        states[name].sort(key=lambda x: x[0])
        times[name] = [x[0] for x in books[name]]
        connection_times[name] = [
            x[0] for x in states[name]
        ]

    if not all(books.values()):
        raise RuntimeError("Both feeds need quote data.")

    def quote_at(name, t, max_age=None):
        i = bisect.bisect_right(times[name], t) - 1
        j = bisect.bisect_right(
            connection_times[name], t
        ) - 1

        if i < 0 or j < 0 or not states[name][j][1]:
            return None

        quote = books[name][i]

        # Reject data from a previous connection.
        if quote[0] < states[name][j][0]:
            return None

        if max_age is not None:
            if t - quote[0] > max_age:
                return None

        return quote

    responses = {
        (th, h): []
        for th in THRESHOLDS
        for h in REACTION_MS
    }

    trades = {
        (th, d, h): []
        for th in THRESHOLDS
        for d in DELAYS_MS
        for h in HOLDS_MS
    }

    triggers = {th: 0 for th in THRESHOLDS}
    last_trigger = {
        th: float("-inf") for th in THRESHOLDS
    }

    for t, _, _, cb_mid in books["coinbase"]:
        prior = quote_at("coinbase", t - 1000)

        if prior is None:
            continue

        move = cb_mid - prior[3]

        kalshi_now = quote_at(
            "kalshi", t, args.max_quote_age_ms
        )

        if kalshi_now is None:
            continue

        direction = 1 if move > 0 else -1

        for threshold in THRESHOLDS:
            if abs(move) < threshold:
                continue

            if (
                t - last_trigger[threshold]
                < args.cooldown_ms
            ):
                continue

            triggers[threshold] += 1
            last_trigger[threshold] = t

            # Measure future Kalshi price response.
            for horizon in REACTION_MS:
                target = t + horizon

                if times["kalshi"][-1] < target:
                    continue

                future = quote_at(
                    "kalshi",
                    target,
                    args.max_quote_age_ms,
                )

                if future is None:
                    continue

                aligned_move = direction * (
                    future[3] - kalshi_now[3]
                )

                responses[
                    (threshold, horizon)
                ].append(aligned_move)

            # Replay hypothetical execution.
            for delay in DELAYS_MS:
                entry_t = t + delay

                if times["kalshi"][-1] < entry_t:
                    continue

                entry = quote_at(
                    "kalshi",
                    entry_t,
                    args.max_quote_age_ms,
                )

                if entry is None:
                    continue

                # Long buys ask; short sells bid.
                entry_price = (
                    entry[2] if direction == 1
                    else entry[1]
                )

                for hold in HOLDS_MS:
                    exit_t = entry_t + hold

                    if times["kalshi"][-1] < exit_t:
                        continue

                    exit_quote = quote_at(
                        "kalshi",
                        exit_t,
                        args.max_quote_age_ms,
                    )

                    if exit_quote is None:
                        continue

                    # Long sells bid; short buys ask.
                    exit_price = (
                        exit_quote[1] if direction == 1
                        else exit_quote[2]
                    )

                    gross = direction * (
                        exit_price - entry_price
                    )

                    fees = (
                        entry_price + exit_price
                    ) * args.fee_bps / 10000

                    net = gross - fees

                    trades[
                        (threshold, delay, hold)
                    ].append((gross, net))

    print("\n=== BTC PERPS EVENT STUDY ===")
    print("Data:", path)
    print("Read-only; no orders submitted.")
    print("Quote changes:", {
        n: len(books[n]) for n in books
    })
    print(
        f"Taker fee: {args.fee_bps:.2f} bps per side"
    )

    print("\n=== KALSHI REACTION TO COINBASE ===")
    print(
        "Threshold | Signals | Horizon | N | "
        "Avg aligned K move | Follow %"
    )

    for th in THRESHOLDS:
        for horizon in REACTION_MS:
            values = responses[(th, horizon)]
            count = len(values)
            follow = (
                sum(v > 0 for v in values) / count * 100
                if count else float("nan")
            )

            print(
                f"${th:>3} | "
                f"{triggers[th]:>5} | "
                f"{horizon:>5}ms | "
                f"{count:>4} | "
                f"{fmt(avg(values)):>10} | "
                f"{follow:.1f}%"
            )

    print("\n=== EXECUTABLE QUOTE REPLAY ===")
    print(
        "Jump | Delay | Hold | Trades | "
        "Avg gross $/BTC | Avg net $/BTC | Net wins"
    )

    for th in THRESHOLDS:
        for delay in DELAYS_MS:
            for hold in HOLDS_MS:
                values = trades[(th, delay, hold)]
                count = len(values)

                if count:
                    gross = avg([v[0] for v in values])
                    net = avg([v[1] for v in values])
                    wins = (
                        sum(v[1] > 0 for v in values)
                        / count * 100
                    )
                else:
                    gross = net = wins = float("nan")

                print(
                    f"${th:>3} | "
                    f"{delay:>4}ms | "
                    f"{hold // 1000:>2}s | "
                    f"{count:>5} | "
                    f"{fmt(gross):>10} | "
                    f"{fmt(net):>10} | "
                    f"{wins:.1f}%"
                )

    print(
        "\nP&L shown per BTC equivalent."
        "\nMultiply by 0.0001 for P&L per contract."
        "\nThis assumes the quoted prices are fillable."
        "\nNo slippage, partial fills, or funding modeled."
        "\nOverlapping signals are not independent."
        "\nRESEARCH ONLY. NO ORDERS SUBMITTED."
    )


if __name__ == "__main__":
    main()
