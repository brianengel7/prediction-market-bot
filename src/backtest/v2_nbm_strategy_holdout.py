import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset
)

from src.backtest.v2_nbm_strategy import (
    run_strategy,
    summarize_strategy,
    print_side_breakdown
)


# ============================================================
# FROZEN STRATEGY
# ============================================================

MINIMUM_EDGE = 0.025

HOLDOUT_START = pd.Timestamp(
    "2026-09-11"
)

HOLDOUT_END = pd.Timestamp(
    "2026-09-26"
)


# ============================================================
# HOLDOUT TEST
# ============================================================

def run_holdout_test():

    data = (
        load_dataset()
    )

    data = data[
        (
            data[
                "target_date"
            ]
            >=
            HOLDOUT_START
        )
        &
        (
            data[
                "target_date"
            ]
            <=
            HOLDOUT_END
        )
    ].copy()

    number_of_dates = (
        data[
            "target_date"
        ]
        .nunique()
    )

    trades = (
        run_strategy(
            data=
                data,

            minimum_edge=
                MINIMUM_EDGE
        )
    )

    summary = (
        summarize_strategy(
            trades=
                trades,

            threshold=
                MINIMUM_EDGE
        )
    )

    print()
    print("=" * 105)

    print(
        "NBM EXECUTABLE STRATEGY - TRUE HOLDOUT"
    )

    print("=" * 105)

    print(
        f"Holdout dates:     "
        f"{number_of_dates}"
    )

    print(
        f"Frozen beta:       "
        f"-0.40"
    )

    print(
        f"Minimum net edge:  "
        f"{MINIMUM_EDGE:.2%}"
    )

    print(
        "Maximum trades/day: 1"
    )

    print()

    print(
        "PERFORMANCE"
    )

    print("-" * 60)

    print(
        f"Trades:             "
        f"{summary['trades']}"
    )

    print(
        f"Wins:               "
        f"{summary['wins']}"
    )

    print(
        f"Losses:             "
        f"{summary['trades'] - summary['wins']}"
    )

    print(
        f"Win rate:           "
        f"{summary['win_rate']:.2%}"
    )

    print(
        f"Average net edge:   "
        f"{summary['avg_edge']:.2%}"
    )

    print(
        f"Capital deployed:   "
        f"${summary['capital']:.2f}"
    )

    print(
        f"Net P&L:            "
        f"${summary['net_pnl']:.2f}"
    )

    print(
        f"ROI:                "
        f"{summary['roi']:.2%}"
    )

    print(
        f"Maximum drawdown:   "
        f"${summary['max_drawdown']:.2f}"
    )

    # ========================================================
    # EXPECTED VS REALIZED P&L
    # ========================================================

    if not trades.empty:

        expected_pnl = float(
            trades[
                "net_edge"
            ]
            .sum()
        )

        actual_pnl = float(
            trades[
                "net_pnl"
            ]
            .sum()
        )

        print()

        print(
            "EXPECTED VS REALIZED"
        )

        print("-" * 60)

        print(
            f"Expected P&L from "
            f"model edge:       "
            f"${expected_pnl:.2f}"
        )

        print(
            f"Actual P&L:        "
            f"${actual_pnl:.2f}"
        )

        print(
            f"Difference:        "
            f"${actual_pnl - expected_pnl:+.2f}"
        )

    # ========================================================
    # SIDE BREAKDOWN
    # ========================================================

    print_side_breakdown(
        trades
    )

    # ========================================================
    # DAILY TRADES
    # ========================================================

    print()
    print("=" * 105)

    print(
        "HOLDOUT TRADES"
    )

    print("=" * 105)

    if trades.empty:

        print(
            "No qualifying trades."
        )

        return

    display = (
        trades.copy()
    )

    display[
        "target_date"
    ] = (
        pd.to_datetime(
            display[
                "target_date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    print(
        display[
            [
                "target_date",
                "ticker",
                "side",
                "fair_probability",
                "entry_price",
                "fee",
                "total_cost",
                "net_edge",
                "won",
                "net_pnl"
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # CUMULATIVE P&L
    # ========================================================

    cumulative = (
        trades.copy()
    )

    cumulative[
        "cumulative_pnl"
    ] = (
        cumulative[
            "net_pnl"
        ]
        .cumsum()
    )

    print()

    print(
        "CUMULATIVE P&L"
    )

    print("-" * 60)

    for _, trade in (
        cumulative.iterrows()
    ):

        target_date = (
            pd.Timestamp(
                trade[
                    "target_date"
                ]
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        print(
            f"{target_date}  "
            f"{trade['side']:<3}  "
            f"{trade['net_pnl']:+.2f}  "
            f"Cum: "
            f"${trade['cumulative_pnl']:+.2f}"
        )


if __name__ == "__main__":

    run_holdout_test()