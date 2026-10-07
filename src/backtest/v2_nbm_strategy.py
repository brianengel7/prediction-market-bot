import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset,
    adjusted_probabilities
)

from src.kalshi.fees import (
    calculate_taker_fee
)

# ============================================================
# CONFIG
# ============================================================

SIGNAL_COLUMN = "NBM_yes"

FROZEN_BETA = -0.40

DEVELOPMENT_START = pd.Timestamp(
    "2026-08-19"
)

DEVELOPMENT_END = pd.Timestamp(
    "2026-09-10"
)

EDGE_THRESHOLDS = [
    0.025,
    0.050,
    0.075,
    0.100
]


# ============================================================
# BUILD TRADE CANDIDATES FOR ONE DAY
# ============================================================

def build_daily_candidates(
    group,
    beta=FROZEN_BETA,
    yes_probabilities=None,
):

    group = (
        group.copy()
        .reset_index(
            drop=True
        )
    )

    if yes_probabilities is None:
        adjusted_yes = adjusted_probabilities(
            group=group,
            signal_column=SIGNAL_COLUMN,
            beta=beta,
        )
    else:
        adjusted_yes = np.asarray(yes_probabilities, dtype=float)

        if (
            adjusted_yes.shape != (len(group),)
            or not np.isfinite(adjusted_yes).all()
            or (adjusted_yes < 0).any()
            or (adjusted_yes > 1).any()
            or not np.isclose(adjusted_yes.sum(), 1.0, atol=1e-6)
        ):
            raise ValueError("Invalid weather probability distribution.")

    group[
        "adjusted_yes"
    ] = adjusted_yes

    group[
        "adjusted_no"
    ] = (
        1.0
        -
        group[
            "adjusted_yes"
        ]
    )

    candidates = []

    for _, row in group.iterrows():

        outcome = int(
            row[
                "outcome"
            ]
        )

        # ----------------------------------------------------
        # YES SIDE
        # ----------------------------------------------------

        yes_ask = row[
            "yes_ask"
        ]

        if not pd.isna(
            yes_ask
        ):

            yes_ask = float(
                yes_ask
            )

            yes_fee = float(
                calculate_taker_fee(
                    price=
                        yes_ask,

                    contracts=
                        1,

                    multiplier=
                        1
                )
            )

            yes_total_cost = (
                yes_ask
                +
                yes_fee
            )

            yes_fair = float(
                row[
                    "adjusted_yes"
                ]
            )

            yes_net_edge = (
                yes_fair
                -
                yes_total_cost
            )

            yes_payout = float(
                outcome
            )

            yes_net_pnl = (
                yes_payout
                -
                yes_total_cost
            )

            candidates.append({

                "target_date":
                    row[
                        "target_date"
                    ],

                "ticker":
                    row[
                        "ticker"
                    ],

                "side":
                    "YES",

                "fair_probability":
                    yes_fair,

                "entry_price":
                    yes_ask,

                "fee":
                    yes_fee,

                "total_cost":
                    yes_total_cost,

                "net_edge":
                    yes_net_edge,

                "payout":
                    yes_payout,

                "won":
                    bool(
                        yes_payout == 1.0
                    ),

                "net_pnl":
                    yes_net_pnl
            })

        # ----------------------------------------------------
        # NO SIDE
        # ----------------------------------------------------

        no_ask = row[
            "no_ask"
        ]

        if not pd.isna(
            no_ask
        ):

            no_ask = float(
                no_ask
            )

            no_fee = float(
                calculate_taker_fee(
                    price=
                        no_ask,

                    contracts=
                        1,

                    multiplier=
                        1
                )
            )

            no_total_cost = (
                no_ask
                +
                no_fee
            )

            no_fair = float(
                row[
                    "adjusted_no"
                ]
            )

            no_net_edge = (
                no_fair
                -
                no_total_cost
            )

            no_payout = float(
                1
                -
                outcome
            )

            no_net_pnl = (
                no_payout
                -
                no_total_cost
            )

            candidates.append({

                "target_date":
                    row[
                        "target_date"
                    ],

                "ticker":
                    row[
                        "ticker"
                    ],

                "side":
                    "NO",

                "fair_probability":
                    no_fair,

                "entry_price":
                    no_ask,

                "fee":
                    no_fee,

                "total_cost":
                    no_total_cost,

                "net_edge":
                    no_net_edge,

                "payout":
                    no_payout,

                "won":
                    bool(
                        no_payout == 1.0
                    ),

                "net_pnl":
                    no_net_pnl
            })

    return pd.DataFrame(
        candidates
    )


# ============================================================
# SELECT ONE TRADE PER DAY
# ============================================================

def select_daily_trade(
    candidates,
    minimum_edge
):

    if candidates.empty:

        return None

    qualifying = candidates[
        candidates[
            "net_edge"
        ]
        >=
        minimum_edge
    ]

    if qualifying.empty:

        return None

    # Maximum one trade per day:
    # select highest expected net edge.
    best_index = (
        qualifying[
            "net_edge"
        ]
        .idxmax()
    )

    return (
        qualifying
        .loc[
            best_index
        ]
        .to_dict()
    )


# ============================================================
# RUN STRATEGY
# ============================================================

def run_strategy(
    data,
    minimum_edge
):

    trades = []

    for target_date, group in (
        data.groupby(
            "target_date"
        )
    ):

        if len(group) != 6:

            continue

        candidates = (
            build_daily_candidates(
                group
            )
        )

        trade = (
            select_daily_trade(
                candidates=
                    candidates,

                minimum_edge=
                    minimum_edge
            )
        )

        if trade is None:

            continue

        trade[
            "minimum_edge"
        ] = minimum_edge

        trades.append(
            trade
        )

    return pd.DataFrame(
        trades
    )


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(
    trades
):

    if trades.empty:

        return 0.0

    cumulative = (
        trades[
            "net_pnl"
        ]
        .cumsum()
        .to_numpy()
    )

    cumulative_with_start = np.concatenate(
        [
            np.array(
                [0.0]
            ),
            cumulative
        ]
    )

    running_peak = (
        np.maximum.accumulate(
            cumulative_with_start
        )
    )

    drawdowns = (
        running_peak
        -
        cumulative_with_start
    )

    return float(
        drawdowns.max()
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_strategy(
    trades,
    threshold
):

    if trades.empty:

        return {

            "threshold":
                threshold,

            "trades":
                0,

            "wins":
                0,

            "win_rate":
                np.nan,

            "avg_edge":
                np.nan,

            "capital":
                0.0,

            "net_pnl":
                0.0,

            "roi":
                np.nan,

            "max_drawdown":
                0.0
        }

    total_capital = float(
        trades[
            "total_cost"
        ].sum()
    )

    net_pnl = float(
        trades[
            "net_pnl"
        ].sum()
    )

    roi = (
        net_pnl
        /
        total_capital
        if total_capital > 0
        else np.nan
    )

    return {

        "threshold":
            threshold,

        "trades":
            len(
                trades
            ),

        "wins":
            int(
                trades[
                    "won"
                ].sum()
            ),

        "win_rate":
            trades[
                "won"
            ].mean(),

        "avg_edge":
            trades[
                "net_edge"
            ].mean(),

        "capital":
            total_capital,

        "net_pnl":
            net_pnl,

        "roi":
            roi,

        "max_drawdown":
            calculate_max_drawdown(
                trades
            )
    }


# ============================================================
# SIDE BREAKDOWN
# ============================================================

def print_side_breakdown(
    trades
):

    if trades.empty:

        return

    print()
    print(
        "SIDE BREAKDOWN"
    )

    print("-" * 75)

    for side, group in (
        trades.groupby(
            "side"
        )
    ):

        capital = (
            group[
                "total_cost"
            ]
            .sum()
        )

        pnl = (
            group[
                "net_pnl"
            ]
            .sum()
        )

        roi = (
            pnl
            /
            capital
            if capital > 0
            else np.nan
        )

        print(
            f"{side:<5} "
            f"Trades: {len(group):<3} "
            f"Wins: {int(group['won'].sum()):<3} "
            f"Win rate: {group['won'].mean():>7.2%} "
            f"P&L: ${pnl:>7.2f} "
            f"ROI: {roi:>7.2%}"
        )


# ============================================================
# DEVELOPMENT TEST
# ============================================================

def run_development_test():

    data = (
        load_dataset()
    )

    data = data[
        (
            data[
                "target_date"
            ]
            >=
            DEVELOPMENT_START
        )
        &
        (
            data[
                "target_date"
            ]
            <=
            DEVELOPMENT_END
        )
    ].copy()

    print()
    print("=" * 105)

    print(
        "NBM EXECUTABLE STRATEGY - DEVELOPMENT PERIOD"
    )

    print("=" * 105)

    print(
        f"Dates available: "
        f"{data['target_date'].nunique()}"
    )

    print(
        f"Signal:          "
        f"{SIGNAL_COLUMN}"
    )

    print(
        f"Frozen beta:     "
        f"{FROZEN_BETA:+.2f}"
    )

    print(
        "One trade maximum per day."
    )

    print()

    summaries = []

    trade_results = {}

    for threshold in EDGE_THRESHOLDS:

        trades = (
            run_strategy(
                data=
                    data,

                minimum_edge=
                    threshold
            )
        )

        trade_results[
            threshold
        ] = trades

        summaries.append(
            summarize_strategy(
                trades=
                    trades,

                threshold=
                    threshold
            )
        )

    summary = pd.DataFrame(
        summaries
    )

    print(
        f"{'Threshold':<12}"
        f"{'Trades':<8}"
        f"{'Wins':<7}"
        f"{'Win Rate':<12}"
        f"{'Avg Edge':<12}"
        f"{'Capital':<12}"
        f"{'Net P&L':<12}"
        f"{'ROI':<12}"
        f"{'Max DD':<10}"
    )

    print("-" * 105)

    for _, row in summary.iterrows():

        print(
            f"{row['threshold']:<12.2%}"
            f"{int(row['trades']):<8}"
            f"{int(row['wins']):<7}"
            f"{row['win_rate']:<12.2%}"
            f"{row['avg_edge']:<12.2%}"
            f"${row['capital']:<11.2f}"
            f"${row['net_pnl']:<11.2f}"
            f"{row['roi']:<12.2%}"
            f"${row['max_drawdown']:<9.2f}"
        )

    # --------------------------------------------------------
    # Show trades for each threshold.
    #
    # We are deliberately NOT selecting a winner automatically.
    # --------------------------------------------------------

    for threshold in EDGE_THRESHOLDS:

        trades = (
            trade_results[
                threshold
            ]
        )

        print()
        print("=" * 105)

        print(
            f"TRADES AT "
            f"{threshold:.1%} "
            f"MINIMUM NET EDGE"
        )

        print("=" * 105)

        if trades.empty:

            print(
                "No trades."
            )

            continue

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

        print_side_breakdown(
            trades
        )


if __name__ == "__main__":

    run_development_test()