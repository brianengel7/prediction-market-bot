"""NBA sportsbook -> Kalshi strong-signal historical study.

Reads existing nba_historical.py research data.
No API calls, no orders, no paid credits.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from statistics import mean, median

from src.sports.research import nba_historical as base


def valid_quote(q):
    return bool(
        q
        and q["bid"] is not None
        and q["ask"] is not None
        and 0 < q["bid"] <= q["ask"] < 1
    )


def mid(q):
    return (q["bid"] + q["ask"]) / 2


def book_state(rows, t, max_age):
    """Only include bookmaker quotes available at time t."""
    state = {}

    for r in rows:
        try:
            updated = base.dt(r["book_updated"])
            age = (t - updated).total_seconds() / 60
            probability = float(r["home_fair"])

            if 0 <= age <= max_age and 0 < probability < 1:
                state[r["book"]] = (probability, updated)

        except (ValueError, TypeError, KeyError):
            continue

    return state


def trade_pnl(entry, exitq, qty, slip, multiplier):
    """Net profit in cents per contract."""
    buy = entry["ask"] + slip
    sell = exitq["bid"] - slip

    if not (0 < buy < 1 and 0 < sell < 1):
        return None

    entry_fee = base.fee(buy, qty, multiplier)
    exit_fee = base.fee(sell, qty, multiplier)

    return (
        qty * (sell - buy) - entry_fee - exit_fee
    ) / qty * 100


def markout(entry, exitq, qty, slip, multiplier):
    return {
        "midpoint_cents": 100 * (mid(exitq) - mid(entry)),
        "gross_quote_cents": 100 * (
            exitq["bid"] - entry["ask"]
        ),
        "net_cents": trade_pnl(
            entry, exitq, qty, slip, multiplier
        ),
        "spread_cents": 50 * (
            entry["ask"] - entry["bid"]
            + exitq["ask"] - exitq["bid"]
        ),
    }


def build_signals(game, snapshots, candles, args):
    start = base.dt(game["start"])
    events = []
    previous = None

    for t, rows in sorted(snapshots.items()):
        minutes_before = (start - t).total_seconds() / 60

        if not args.min_before <= minutes_before <= args.max_before:
            continue

        state = book_state(
            rows, t, args.max_book_age
        )

        if len(state) < args.min_books:
            continue

        consensus = median(v[0] for v in state.values())

        if (
            previous is None
            or (t - previous[0]).total_seconds() / 60 > args.max_step
        ):
            previous = (t, state, consensus)
            continue

        _, old_state, old_consensus = previous

        common = set(state) & set(old_state)

        # An actual bookmaker probability change, with a
        # newer update timestamp, is required.
        changes = {
            book: state[book][0] - old_state[book][0]
            for book in common
            if state[book][1] > old_state[book][1]
        }

        consensus_change = consensus - old_consensus

        for side, direction in (("home", 1), ("away", -1)):
            aligned = [
                (book, direction * change)
                for book, change in changes.items()
                if direction * change >= args.book_step
            ]

            biggest = max(
                (direction * v for v in changes.values()),
                default=0.0,
            )

            signed_consensus = direction * consensus_change

            prior_kalshi = base.nearest_before(
                candles[side],
                t,
                args.max_quote_age,
            )

            gap = None

            if valid_quote(prior_kalshi):
                fair = (
                    consensus
                    if side == "home"
                    else 1 - consensus
                )
                gap = fair - mid(prior_kalshi)

            rules = []

            if signed_consensus >= 0.01:
                rules.append("consensus_1pp")

            if len(aligned) >= 3 and signed_consensus >= 0.004:
                rules.append("three_books_up")

            if biggest >= 0.015:
                rules.append("single_book_1p5pp")

            if gap is not None and gap >= 0.03:
                rules.append("fair_mid_gap_3pp")

            if (
                args.lead_book
                and args.lead_book in changes
                and direction * changes[args.lead_book] >= 0.01
            ):
                rules.append("chosen_book_1pp")

            for rule in rules:
                events.append({
                    "event_id": game["event_id"],
                    "game": game["away"] + " @ " + game["home"],
                    "rule": rule,
                    "side": side,
                    "signal_time": base.stamp(t),
                    "books": len(state),
                    "changed_books": len(changes),
                    "aligned_books": len(aligned),
                    "median_change_pp": signed_consensus * 100,
                    "largest_book_change_pp": biggest * 100,
                    "gap_pp": (
                        gap * 100 if gap is not None else None
                    ),
                    "ticker": game[side + "_ticker"],
                })

        previous = (t, state, consensus)

    return events


def next_quote(candles, t, deadline, delay, max_lag):
    q = base.nearest_after(
        candles,
        t + timedelta(minutes=delay),
        deadline,
        max_lag,
    )

    return q if valid_quote(q) else None


def evaluate(game, signal, candles, args):
    start = base.dt(game["start"])
    cutoff = start - timedelta(minutes=1)

    # Reserve time for a complete 15-minute exit.
    entry_deadline = cutoff - timedelta(minutes=15)

    series = candles[signal["side"]]
    signal_time = base.dt(signal["signal_time"])

    entry = next_quote(
        series,
        signal_time,
        entry_deadline,
        args.delay,
        args.max_quote_age,
    )

    if not valid_quote(entry):
        return []

    if entry["ask"] - entry["bid"] > args.max_spread:
        return []

    output = []

    # Fixed holding-period experiments.
    for minutes in (1, 3, 5, 10, 15):
        target = entry["t"] + timedelta(minutes=minutes)

        exitq = next_quote(
            series,
            target,
            cutoff,
            0,
            args.max_quote_age,
        )

        if not valid_quote(exitq):
            continue

        metrics = markout(
            entry, exitq,
            args.qty,
            args.slippage,
            args.fee_multiplier,
        )

        if metrics["net_cents"] is None:
            continue

        output.append({
            **signal,
            "exit_rule": f"timed_{minutes}m",
            "entry_time": base.stamp(entry["t"]),
            "exit_time": base.stamp(exitq["t"]),
            **metrics,
        })

    # Immediate profit-taking, maximum 15-minute hold.
    window = [
        q for q in series
        if (
            entry["t"] < q["t"]
            <= entry["t"] + timedelta(minutes=15)
            and valid_quote(q)
            and q["ask"] - q["bid"] <= args.max_spread
        )
    ]

    if window:
        profit_quote = None

        for q in window:
            pnl = trade_pnl(
                entry, q,
                args.qty,
                args.slippage,
                args.fee_multiplier,
            )

            if pnl is not None and pnl > 0:
                profit_quote = q
                break

        if profit_quote is not None:
            exitq = profit_quote
            reason = "profit"
        else:
            exitq = next_quote(
                series,
                entry["t"] + timedelta(minutes=15),
                cutoff,
                0,
                args.max_quote_age,
            )
            reason = "timeout"

        if valid_quote(exitq):
            metrics = markout(
                entry, exitq,
                args.qty,
                args.slippage,
                args.fee_multiplier,
            )

            if metrics["net_cents"] is not None:
                output.append({
                    **signal,
                    "exit_rule": "first_net_profit_or_15m",
                    "reason": reason,
                    "entry_time": base.stamp(entry["t"]),
                    "exit_time": base.stamp(exitq["t"]),
                    **metrics,
                })

    return output


def summarize(trades):
    groups = defaultdict(list)

    for row in trades:
        groups[
            (row["rule"], row["exit_rule"])
        ].append(row)

    results = []

    for (rule, exit_rule), rows in sorted(groups.items()):
        mids = [r["midpoint_cents"] for r in rows]
        nets = [r["net_cents"] for r in rows]

        results.append({
            "rule": rule,
            "exit": exit_rule,
            "trades": len(rows),
            "unique_games": len({
                r["event_id"] for r in rows
            }),
            "favorable_mid": sum(x > 0 for x in mids),
            "avg_mid_cents": round(mean(mids), 3),
            "avg_net_cents": round(mean(nets), 3),
            "net_wins": sum(x > 0 for x in nets),
        })

    return results


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No rows to write:", path)
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    odds = base.read_csv(args.data / "odds.csv")
    games = base.read_json(args.data / "matched_games.json")

    by_game = defaultdict(lambda: defaultdict(list))

    for row in odds:
        by_game[row["event_id"]][
            base.dt(row["timestamp"])
        ].append(row)

    raw = []
    trades = []
    rejected = Counter()

    for game in games:
        paths = {
            side: (
                args.data / "candles" /
                (game[side + "_ticker"] + ".csv")
            )
            for side in ("home", "away")
        }

        if not all(p.exists() for p in paths.values()):
            rejected["missing_candles"] += 1
            continue

        candles = {
            side: base.candle_rows(path)
            for side, path in paths.items()
        }

        signals = build_signals(
            game,
            by_game[game["event_id"]],
            candles,
            args,
        )

        raw.extend(signals)

        # One entry per game per strategy.
        # Different strategies are evaluated separately.
        selected = {}

        for signal in signals:
            selected.setdefault(
                signal["rule"], signal
            )

        for signal in selected.values():
            evaluated = evaluate(
                game, signal, candles, args
            )

            if not evaluated:
                rejected["missing_valid_entry_or_exit"] += 1

            trades.extend(evaluated)

    report = summarize(trades)

    write_csv(args.data / "strong_signals.csv", raw)
    write_csv(args.data / "strong_trades.csv", trades)
    write_csv(args.data / "strong_summary.csv", report)

    print(
        f"Matched games: {len(games)} | "
        f"Candidate signals: {len(raw)} | "
        f"Evaluated scenarios: {len(trades)}"
    )

    print("Rejections:", dict(rejected))
    print(
        "Signals by strategy:",
        dict(Counter(s["rule"] for s in raw)),
    )

    print("\nSTRATEGY RESULTS")
    print("=" * 100)

    for r in report:
        print(
            f"{r['rule']:22s} "
            f"{r['exit']:24s} "
            f"games={r['unique_games']:2d} "
            f"mid={r['avg_mid_cents']:+.2f}c "
            f"net={r['avg_net_cents']:+.2f}c "
            f"wins={r['net_wins']}/{r['trades']}"
        )

    print(
        "\nExploratory results only: historical candle "
        "quotes do not guarantee executable fills."
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/research/sports_nba_lead_lag"),
    )
    p.add_argument("--min-before", type=int, default=20)
    p.add_argument("--max-before", type=int, default=240)
    p.add_argument("--min-books", type=int, default=3)
    p.add_argument("--max-book-age", type=int, default=60)
    p.add_argument("--max-step", type=int, default=10)
    p.add_argument("--max-quote-age", type=int, default=2)
    p.add_argument("--max-spread", type=float, default=.06)
    p.add_argument("--book-step", type=float, default=.002)

    p.add_argument(
        "--lead-book",
        default="",
        help="Exact bookmaker key, if present in the dataset",
    )

    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--qty", type=int, default=10)
    p.add_argument(
        "--fee-multiplier",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--slippage",
        type=float,
        default=.01,
        help="Applied separately to entry and exit",
    )

    args = p.parse_args()

    if (
        args.qty < 1
        or args.delay < 0
        or not 0 <= args.fee_multiplier <= 1
        or args.slippage < 0
    ):
        p.error("Invalid quantity, delay, fee or slippage")

    run(args)


if __name__ == "__main__":
    main()
