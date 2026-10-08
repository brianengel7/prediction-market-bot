"""Read-only expected ROI and winning ROI threshold sweep."""

from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from types import SimpleNamespace

import pandas as pd

import src.backtest.hourly_continuous_multi_backtest as model


# Cache database history and calibration snapshots
# during the sweep, without persisting backtest results.
original_load = model.load_history
history_cache = {}


def load_once(end):
    if end not in history_cache:
        history_cache[end] = original_load(end)

    return history_cache[end]


model.load_history = load_once


original_snapshots = model.make_snapshots
snapshot_cache = {}


def snapshots_once(
    markets,
    candles,
    bucket,
    max_age,
    entry_delay,
    entry_wait,
):
    key = (
        bucket,
        max_age,
        entry_delay,
        entry_wait,
    )

    if key not in snapshot_cache:
        snapshot_cache[key] = original_snapshots(
            markets,
            candles,
            bucket,
            max_age,
            entry_delay,
            entry_wait,
        )

    return snapshot_cache[key]


model.make_snapshots = snapshots_once


def run_one(
    start_min,
    entries,
    min_roi,
    win_roi,
):
    cfg = SimpleNamespace(
        start=date(2026, 9, 10),
        end=date(2026, 10, 7),
        scan_start=start_min,
        scan_end=5,
        max_entries=entries,
        max_event_cost=2.50,
        min_roi=float(min_roi),
        min_winning_roi=float(win_roi),
        momentum_mode="none",
        momentum_cents=0,
        momentum_lookback=3,
    )

    with redirect_stdout(StringIO()):
        result = model.main(cfg)

    if (
        not isinstance(result, dict)
        or "RESEARCH" not in result
        or "VIEWED_OOS" not in result
    ):
        raise RuntimeError(
            "Missing period summaries from research script"
        )

    r = result["RESEARCH"]
    o = result["VIEWED_OOS"]

    return {
        "window": f"{start_min}-5",
        "entries": entries,
        "min_exp_roi": min_roi,
        "min_win_roi": win_roi,
        "r_trades": r["trades"],
        "r_wins": r["wins"],
        "r_pnl": r["pnl"],
        "r_roi": r["roi"],
        "r_drawdown": r["drawdown"],
        "o_trades": o["trades"],
        "o_wins": o["wins"],
        "o_pnl": o["pnl"],
        "o_roi": o["roi"],
    }


def main():
    print("\nKXTEMPNYCHS — ROI FILTER SWEEP")
    print("=" * 100)

    print("Historical data through October 7, 2026")
    print(
        "2 windows x 2 entry limits x "
        "4 expected ROI x 3 winning ROI"
    )
    print(
        "48 configurations, identical calibration "
        "and historical quotes"
    )
    print(
        "Research: through Oct 2 | "
        "Previously viewed: Oct 3–7"
    )
    print("All simulations are read-only.\n", flush=True)

    print(
        "Checking original 45-to-5-minute baseline...",
        flush=True,
    )

    baseline = run_one(45, 1, 5, 0)

    print(
        f"Baseline research: "
        f"{baseline['r_trades']} trades, "
        f"P&L ${baseline['r_pnl']:+.2f}"
    )

    print(
        f"Baseline viewed OOS: "
        f"{baseline['o_trades']} trades, "
        f"P&L ${baseline['o_pnl']:+.2f}\n",
        flush=True,
    )

    if (
        baseline["r_trades"] != 157
        or baseline["o_trades"] != 16
    ):
        raise RuntimeError(
            "Baseline did not reproduce "
            "157 research and 16 viewed trades. "
            "Check the dataset and backtest "
            "before comparing filters."
        )

    records = [baseline]

    for start_min in (45, 30):
        for entries in (1, 3):
            for min_roi in (5, 10, 15, 20):
                for win_roi in (0, 30, 50):

                    if (
                        start_min,
                        entries,
                        min_roi,
                        win_roi,
                    ) == (45, 1, 5, 0):
                        continue

                    row = run_one(
                        start_min,
                        entries,
                        min_roi,
                        win_roi,
                    )

                    records.append(row)

                    print(
                        f"[{len(records):2d}/48] "
                        f"{row['window']}m "
                        f"entries={entries} "
                        f"expected>={min_roi}% "
                        f"winning>={win_roi}% | "
                        f"research trades={row['r_trades']} "
                        f"ROI={row['r_roi']:+.2f}%",
                        flush=True,
                    )

    results = pd.DataFrame(records)

    columns = [
        "min_exp_roi",
        "min_win_roi",
        "r_trades",
        "r_wins",
        "r_pnl",
        "r_roi",
        "r_drawdown",
        "o_trades",
        "o_wins",
        "o_pnl",
        "o_roi",
    ]

    for window in ("45-5", "30-5"):
        for entries in (1, 3):
            subset = results[
                (results.window == window)
                & (results.entries == entries)
            ]

            print("\n" + "=" * 110)

            print(
                f"WINDOW: {window} MINUTES | "
                f"MAX ENTRIES PER EVENT: {entries}"
            )

            print("=" * 110)

            print(
                subset[columns]
                .sort_values([
                    "min_exp_roi",
                    "min_win_roi",
                ])
                .to_string(
                    index=False,
                    float_format=lambda x: f"{x:+.2f}",
                )
            )

    print("\n" + "=" * 100)
    print("SWEEP COMPLETE")
    print("No orders submitted.")
    print("No database writes or CSV files created.")
    print(
        "October 3–7 is already-viewed data, "
        "not a fresh holdout."
    )


if __name__ == "__main__":
    main()
