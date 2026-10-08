"""Read-only trade diagnostics for the hourly continuous strategy."""

import argparse
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest import hourly_continuous_multi_backtest as model


def worst_event_pnl(positions):
    """Worst possible payout across temperature threshold outcomes."""
    legs = []

    for row in positions.itertuples(index=False):
        if "-T" not in row.ticker:
            raise ValueError(
                f"Invalid threshold ticker: {row.ticker}"
            )

        threshold = float(
            row.ticker.rsplit("-T", 1)[1]
        )

        legs.append((
            threshold,
            str(row.side).upper(),
        ))

    strikes = sorted({
        x[0] for x in legs
    })

    temperatures = (
        [strikes[0] - 1]
        + [
            (a + b) / 2
            for a, b in zip(
                strikes,
                strikes[1:],
            )
        ]
        + [strikes[-1] + 1]
    )

    risked = float(
        positions.risked.sum()
    )

    return min(
        sum(
            (temp > strike)
            if side == "YES"
            else (temp <= strike)
            for strike, side in legs
        ) - risked
        for temp in temperatures
    )


def show_table(data, columns):
    print(
        data[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def analyze(trades, period):
    t = trades[
        trades.period == period
    ].copy()

    print("\n" + "=" * 100)
    print(f"TRADE ANALYSIS: {period}")
    print("=" * 100)

    if t.empty:
        print("No trades in this period.")
        return

    # Include conservative buffer in capital at risk.
    # Fixes earlier ROI accounting inconsistency.
    t["risked"] = (
        t.capital + model.BUFFER
    )

    t["pnl"] = t.buffered_pnl

    t["expected_pnl"] = (
        t.fair - t.risked
    )

    t["winning_roi"] = (
        (1 - t.risked) / t.risked * 100
    )

    t = t.sort_values(
        ["day", "event", "minutes_left"],
        ascending=[
            True,
            True,
            False,
        ],
    )

    capital = float(t.risked.sum())
    pnl = float(t.pnl.sum())
    wins = int(t.win.sum())

    print("\nOVERALL PERFORMANCE")
    print(f"Trades: {len(t)}")
    print(f"Wins: {wins}")
    print(f"Losses: {len(t) - wins}")
    print(f"Win rate: {wins / len(t):.2%}")
    print(
        f"Buffered capital at risk: ${capital:.2f}"
    )
    print(f"Realized profit: ${pnl:+.2f}")
    print(
        f"Realized ROI: {pnl / capital:+.2%}"
    )

    print("\nMODEL ACCURACY ON SELECTED TRADES")

    print(
        f"Average predicted win probability: "
        f"{t.fair.mean():.2%}"
    )

    print(
        f"Actual win frequency: "
        f"{t.win.mean():.2%}"
    )

    print(
        f"Total predicted P&L: "
        f"${t.expected_pnl.sum():+.2f}"
    )

    print(
        f"Total actual P&L: ${pnl:+.2f}"
    )

    print(
        f"Average potential winning ROI: "
        f"{t.winning_roi.mean():.2f}%"
    )

    daily = (
        t.groupby(
            "day",
            as_index=False,
        )
        .agg(
            trades=("win", "size"),
            wins=("win", "sum"),
            risked=("risked", "sum"),
            pnl=("pnl", "sum"),
        )
        .sort_values("day")
    )

    daily["roi_pct"] = (
        daily.pnl / daily.risked * 100
    )

    equity = np.r_[
        0.0,
        np.cumsum(
            daily.pnl.to_numpy()
        ),
    ]

    drawdown = float(
        np.max(
            np.maximum.accumulate(equity)
            - equity
        )
    )

    best = daily.loc[
        daily.pnl.idxmax()
    ]

    worst = daily.loc[
        daily.pnl.idxmin()
    ]

    print("\nPROFIT CONCENTRATION")

    print(
        f"Days with trades: {len(daily)}"
    )

    print(
        f"Profitable days: "
        f"{int((daily.pnl > 0).sum())}"
    )

    print(
        f"Best day: {best['day']} "
        f"(${best.pnl:+.2f})"
    )

    print(
        f"Worst day: {worst['day']} "
        f"(${worst.pnl:+.2f})"
    )

    print(
        f"P&L excluding best day: "
        f"${pnl - best.pnl:+.2f}"
    )

    print(
        f"Maximum daily-equity drawdown: "
        f"${drawdown:.2f}"
    )

    print("\nDAILY PERFORMANCE")

    show_table(
        daily,
        [
            "day",
            "trades",
            "wins",
            "risked",
            "pnl",
            "roi_pct",
        ],
    )

    event_results = []

    for event, positions in t.groupby("event"):
        event_results.append({
            "event": event,
            "entries": len(positions),
            "risked": float(
                positions.risked.sum()
            ),
            "worst_pnl": worst_event_pnl(
                positions
            ),
            "actual_pnl": float(
                positions.pnl.sum()
            ),
        })

    events = pd.DataFrame(event_results)

    worst_event = events.loc[
        events.worst_pnl.idxmin()
    ]

    print("\nEVENT-LEVEL EXPOSURE")

    print(
        f"Events traded: {len(events)}"
    )

    print(
        f"Events with multiple entries: "
        f"{int((events.entries > 1).sum())}"
    )

    print(
        f"Largest single-event capital at risk: "
        f"${events.risked.max():.2f}"
    )

    print(
        f"Worst theoretical single-event P&L: "
        f"${worst_event.worst_pnl:+.2f}"
    )

    print(
        f"Worst event: {worst_event.event}"
    )

    bands = [
        0,
        .50,
        .60,
        .70,
        .80,
        .90,
        1.001,
    ]

    labels = [
        "<50c",
        "50-59c",
        "60-69c",
        "70-79c",
        "80-89c",
        "90c+",
    ]

    t["price_band"] = pd.cut(
        t.entry,
        bins=bands,
        labels=labels,
        right=False,
    )

    t["fair_band"] = pd.cut(
        t.fair,
        bins=bands,
        labels=labels,
        right=False,
    )

    for field, label in (
        ("price_band", "ENTRY PRICE"),
        (
            "fair_band",
            "PREDICTED WIN PROBABILITY",
        ),
    ):
        grouped = (
            t.groupby(
                field,
                observed=True,
            )
            .agg(
                trades=("win", "size"),
                predicted_win=(
                    "fair",
                    "mean",
                ),
                actual_win=(
                    "win",
                    "mean",
                ),
                capital=(
                    "risked",
                    "sum",
                ),
                pnl=("pnl", "sum"),
            )
            .reset_index()
        )

        grouped["roi_pct"] = (
            grouped.pnl
            / grouped.capital
            * 100
        )

        print(
            f"\nPERFORMANCE BY {label}"
        )

        show_table(
            grouped,
            [
                field,
                "trades",
                "predicted_win",
                "actual_win",
                "pnl",
                "roi_pct",
            ],
        )

    print("\nALL INDIVIDUAL TRADES")

    show_table(
        t,
        [
            "day",
            "minutes_left",
            "ticker",
            "side",
            "entry",
            "fair",
            "win",
            "risked",
            "expected_pnl",
            "pnl",
        ],
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--scan-start",
        type=int,
        default=30,
    )

    p.add_argument(
        "--scan-end",
        type=int,
        default=5,
    )

    p.add_argument(
        "--min-roi",
        type=float,
        default=5,
    )

    p.add_argument(
        "--min-winning-roi",
        type=float,
        default=30,
    )

    p.add_argument(
        "--max-entries",
        type=int,
        default=1,
    )

    p.add_argument(
        "--max-event-cost",
        type=float,
        default=2.50,
    )

    p.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2026, 9, 10),
    )

    p.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 10, 7),
    )

    p.add_argument(
        "--verify-baseline",
        action="store_true",
    )

    args = p.parse_args()

    if not 1 <= args.scan_end <= args.scan_start <= 60:
        p.error("Invalid scanning window")

    if not 1 <= args.max_entries <= 10:
        p.error("Invalid maximum entry count")

    if args.max_event_cost <= 0:
        p.error(
            "Event cost limit must be positive"
        )

    if args.start > args.end:
        p.error("Start must not exceed end")

    config = SimpleNamespace(
        **vars(args),
        return_trades=True,
        momentum_mode="none",
        momentum_cents=0,
        momentum_lookback=3,
    )

    summary, trades, events = model.main(
        config
    )

    if args.verify_baseline:
        expected_config = (
            30,
            5,
            5,
            30,
            1,
            date(2026, 9, 10),
            date(2026, 10, 7),
        )

        current_config = (
            args.scan_start,
            args.scan_end,
            args.min_roi,
            args.min_winning_roi,
            args.max_entries,
            args.start,
            args.end,
        )

        if current_config != expected_config:
            p.error(
                "Baseline verification requires "
                "the original 30–5-minute configuration"
            )

        research = trades[
            trades.period == "RESEARCH"
        ]

        viewed = trades[
            trades.period == "VIEWED_OOS"
        ]

        if (
            len(research) != 37
            or abs(
                research.buffered_pnl.sum()
                - 4.24
            ) > 0.015
            or len(viewed) != 1
        ):
            raise RuntimeError(
                "Historical baseline changed. "
                "Investigate before interpreting diagnostics."
            )

        print(
            "\nPASS: 37 research trades, "
            "+$4.24 P&L, "
            "1 previously viewed trade"
        )

    analyze(trades, "RESEARCH")
    analyze(trades, "VIEWED_OOS")

    print("\nDIAGNOSTICS COMPLETE")
    print("No orders submitted.")
    print(
        "No database writes or output files created."
    )
    print(
        "Historical entry prices are not guaranteed fills."
    )
    print(
        "October 3–7 is previously evaluated data, "
        "not a fresh untouched holdout."
    )


if __name__ == "__main__":
    main()
