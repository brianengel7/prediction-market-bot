"""Read-only momentum confirmation/fade sweep for hourly weather."""
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest import hourly_continuous_multi_backtest as model


# Avoid repeatedly downloading the same historical dataset.
_original_history = model.load_history
_original_snapshots = model.make_snapshots

_history = {}
_snapshots = {}


def history_once(end):
    if end not in _history:
        _history[end] = _original_history(end)
    return _history[end]


def snapshots_once(markets, candles, bucket, age, delay, wait):
    key = (bucket, age, delay, wait)

    if key not in _snapshots:
        _snapshots[key] = _original_snapshots(
            markets, candles, bucket, age, delay, wait
        )

    return _snapshots[key]


model.load_history = history_once
model.make_snapshots = snapshots_once


def summarize(trades, period, mode, cents):
    selected = trades[trades.period == period]

    if selected.empty:
        return {
            "period": period,
            "mode": mode,
            "move_c": cents,
            "trades": 0,
            "wins": 0,
            "pnl": 0.0,
            "roi_pct": np.nan,
            "drawdown": 0.0,
            "trading_days": 0,
            "avg_fair_pct": np.nan,
            "avg_momentum_c": np.nan,
        }

    pnl = float(selected.buffered_pnl.sum())

    cash_risked = float(
        (selected.capital + model.BUFFER).sum()
    )

    daily = (
        selected.groupby("day")
        .buffered_pnl.sum()
        .sort_index()
    )

    equity = np.r_[
        0.0,
        daily.cumsum().to_numpy(dtype=float),
    ]

    drawdown = float(
        np.max(np.maximum.accumulate(equity) - equity)
    )

    return {
        "period": period,
        "mode": mode,
        "move_c": cents,
        "trades": len(selected),
        "wins": int(selected.win.sum()),
        "pnl": pnl,
        "roi_pct": 100 * pnl / cash_risked,
        "drawdown": drawdown,
        "trading_days": selected.day.nunique(),
        "avg_fair_pct": float(selected.fair.mean() * 100),
        "avg_momentum_c": float(
            selected.signal_momentum_cents.mean()
        ),
    }


def main():
    configs = [("none", 0)] + [
        (mode, cents)
        for mode in ("follow", "fade")
        for cents in (1, 2, 3, 5)
    ]

    print("\nHOURLY NYC MOMENTUM BACKTEST")
    print("=" * 95)
    print("Entry window: 30–5 minutes")
    print("Expected ROI >= 5%; winning ROI >= 30%")
    print("Momentum lookback: 3 minutes")
    print("Both quotes available before decision; max age 1m")
    print("One entry per event; delayed entry 1–3 minutes")
    print("Rolling alpha and historical settlement outcomes")
    print("No orders, local data output, or database writes\n", flush=True)

    rows = []

    for i, (mode, cents) in enumerate(configs, start=1):
        args = SimpleNamespace(
            start=date(2026, 9, 10),
            end=date(2026, 10, 7),
            scan_start=30,
            scan_end=5,
            max_entries=1,
            max_event_cost=2.50,
            min_roi=5,
            min_winning_roi=30,
            momentum_mode=mode,
            momentum_cents=cents,
            momentum_lookback=3,
            return_trades=True,
        )

        with redirect_stdout(StringIO()):
            _, trades, _ = model.main(args)

        for period in ("RESEARCH", "VIEWED_OOS"):
            rows.append(
                summarize(trades, period, mode, cents)
            )

        r = rows[-2]

        print(
            f"[{i}/{len(configs)}] "
            f"{mode:6s} {cents}c: "
            f"research trades={r['trades']}, "
            f"wins={r['wins']}, "
            f"P&L=${r['pnl']:+.2f}, "
            f"ROI={r['roi_pct']:+.2f}%",
            flush=True,
        )

        if mode == "none":
            if (
                r["trades"] != 37
                or abs(r["pnl"] - 4.24) > 0.02
            ):
                raise RuntimeError(
                    "Original 37-trade research baseline "
                    "did not reproduce. Stop comparison."
                )

            if rows[-1]["trades"] != 1:
                raise RuntimeError(
                    "Original viewed-OOS baseline "
                    "did not reproduce."
                )

    report = pd.DataFrame(rows)

    for period in ("RESEARCH", "VIEWED_OOS"):
        print("\n" + "=" * 100)
        print(period)
        print("=" * 100)

        subset = report[report.period == period]

        print(
            subset[
                [
                    "mode",
                    "move_c",
                    "trades",
                    "wins",
                    "trading_days",
                    "pnl",
                    "roi_pct",
                    "drawdown",
                    "avg_fair_pct",
                    "avg_momentum_c",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    print("\nCOMPLETE")
    print("No orders submitted.")
    print("No local data files or database writes.")
    print(
        "October 3–7 is previously examined data, "
        "not a fresh untouched holdout."
    )


if __name__ == "__main__":
    main()
