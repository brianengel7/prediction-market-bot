import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset
)

from src.kalshi.fees import (
    calculate_taker_fee
)


# ============================================================
# FROZEN MARKET-ONLY STRATEGY
# ============================================================

ALPHA = 1.875
MIN_NET_EDGE = 0.025

DEVELOPMENT_START = "2026-08-19"
DEVELOPMENT_END = "2026-09-10"

HOLDOUT_START = "2026-09-11"
HOLDOUT_END = "2026-09-26"

EPSILON = 1e-6


# ============================================================
# MARKET-ONLY DISTRIBUTION
# ============================================================

def market_only_probabilities(
    probabilities
):
    """
    q_i ∝ p_i ** alpha
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0
    )

    probabilities = (
        probabilities
        /
        probabilities.sum()
    )

    sharpened = (
        probabilities
        ** ALPHA
    )

    return (
        sharpened
        /
        sharpened.sum()
    )


# ============================================================
# BUILD DAILY TRADE CANDIDATES
# ============================================================

def build_daily_candidates(
    daily_data
):
    daily_data = (
        daily_data.copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        daily_data
    ) != 6:

        raise ValueError(
            f"Expected 6 contracts, "
            f"found {len(daily_data)}."
        )

    raw_market = (
        daily_data[
            "market_probability"
        ].to_numpy(
            dtype=float
        )
    )

    adjusted_yes = (
        market_only_probabilities(
            raw_market
        )
    )

    candidates = []

    for index, row in (
        daily_data.iterrows()
    ):

        yes_fair = float(
            adjusted_yes[
                index
            ]
        )

        no_fair = (
            1.0
            -
            yes_fair
        )

        market_probability = float(
            raw_market[
                index
            ]
        )

        outcome = int(
            row[
                "outcome"
            ]
        )

        # ----------------------------------------------------
        # YES CANDIDATE
        # ----------------------------------------------------

        yes_ask = row.get(
            "yes_ask"
        )

        if pd.notna(
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

            yes_net_edge = (
                yes_fair
                -
                yes_total_cost
            )

            yes_won = (
                outcome == 1
            )

            yes_payout = (
                1.0
                if yes_won
                else 0.0
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

                "market_probability":
                    market_probability,

                "adjusted_probability":
                    yes_fair,

                "entry_price":
                    yes_ask,

                "fee":
                    yes_fee,

                "total_cost":
                    yes_total_cost,

                "net_edge":
                    yes_net_edge,

                "won":
                    yes_won,

                "payout":
                    yes_payout,

                "net_pnl":
                    yes_net_pnl
            })

        # ----------------------------------------------------
        # NO CANDIDATE
        # ----------------------------------------------------

        no_ask = row.get(
            "no_ask"
        )

        if pd.notna(
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

            no_net_edge = (
                no_fair
                -
                no_total_cost
            )

            no_won = (
                outcome == 0
            )

            no_payout = (
                1.0
                if no_won
                else 0.0
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

                "market_probability":
                    (
                        1.0
                        -
                        market_probability
                    ),

                "adjusted_probability":
                    no_fair,

                "entry_price":
                    no_ask,

                "fee":
                    no_fee,

                "total_cost":
                    no_total_cost,

                "net_edge":
                    no_net_edge,

                "won":
                    no_won,

                "payout":
                    no_payout,

                "net_pnl":
                    no_net_pnl
            })

    candidates.sort(
        key=lambda candidate:
            candidate[
                "net_edge"
            ],

        reverse=True
    )

    return candidates


# ============================================================
# SELECT ONE TRADE PER DAY
# ============================================================

def select_daily_trade(
    candidates
):
    if not candidates:
        return None

    best = candidates[
        0
    ]

    if (
        best[
            "net_edge"
        ]
        <
        MIN_NET_EDGE
    ):
        return None

    return best


# ============================================================
# RUN STRATEGY
# ============================================================

def run_strategy(
    dataframe,
    start_date,
    end_date
):
    data = (
        dataframe[
            (
                dataframe[
                    "target_date"
                ]
                >=
                start_date
            )
            &
            (
                dataframe[
                    "target_date"
                ]
                <=
                end_date
            )
        ]
        .copy()
    )

    trades = []

    for (
        target_date,
        daily_data
    ) in data.groupby(
        "target_date"
    ):

        if len(
            daily_data
        ) != 6:
            continue

        candidates = (
            build_daily_candidates(
                daily_data
            )
        )

        trade = (
            select_daily_trade(
                candidates
            )
        )

        if trade is not None:

            trades.append(
                trade
            )

    return pd.DataFrame(
        trades
    )


# ============================================================
# DRAWDOWN
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
    )

    cumulative_with_zero = pd.concat(
        [
            pd.Series(
                [0.0]
            ),

            cumulative.reset_index(
                drop=True
            )
        ],

        ignore_index=True
    )

    running_peak = (
        cumulative_with_zero
        .cummax()
    )

    drawdown = (
        running_peak
        -
        cumulative_with_zero
    )

    return float(
        drawdown.max()
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_strategy(
    trades,
    label
):

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    if trades.empty:

        print(
            "No qualifying trades."
        )

        return

    trades = (
        trades.copy()
        .sort_values(
            "target_date"
        )
    )

    trade_count = len(
        trades
    )

    wins = int(
        trades[
            "won"
        ].sum()
    )

    losses = (
        trade_count
        -
        wins
    )

    win_rate = (
        wins
        /
        trade_count
    )

    average_edge = (
        trades[
            "net_edge"
        ].mean()
    )

    capital = (
        trades[
            "total_cost"
        ].sum()
    )

    pnl = (
        trades[
            "net_pnl"
        ].sum()
    )

    roi = (
        pnl
        /
        capital
        if capital > 0
        else 0.0
    )

    max_drawdown = (
        calculate_max_drawdown(
            trades
        )
    )

    expected_pnl = (
        trades[
            "net_edge"
        ].sum()
    )

    print(
        f"Alpha:             "
        f"{ALPHA:.3f}"
    )

    print(
        f"Minimum edge:      "
        f"{MIN_NET_EDGE:.2%}"
    )

    print()

    print(
        f"Trades:            "
        f"{trade_count}"
    )

    print(
        f"Wins:              "
        f"{wins}"
    )

    print(
        f"Losses:            "
        f"{losses}"
    )

    print(
        f"Win rate:          "
        f"{win_rate:.2%}"
    )

    print(
        f"Average net edge:  "
        f"{average_edge:.2%}"
    )

    print()

    print(
        f"Capital:           "
        f"${capital:.2f}"
    )

    print(
        f"Net P&L:           "
        f"${pnl:+.2f}"
    )

    print(
        f"ROI:               "
        f"{roi:+.2%}"
    )

    print(
        f"Max drawdown:      "
        f"${max_drawdown:.2f}"
    )

    print()

    print(
        f"Expected P&L:      "
        f"${expected_pnl:+.2f}"
    )

    print(
        f"Actual P&L:        "
        f"${pnl:+.2f}"
    )

    # --------------------------------------------------------
    # Side breakdown
    # --------------------------------------------------------

    print()
    print(
        "SIDE BREAKDOWN"
    )
    print("-" * 80)

    for side in [
        "YES",
        "NO"
    ]:

        side_trades = trades[
            trades[
                "side"
            ]
            ==
            side
        ]

        if side_trades.empty:
            continue

        side_count = len(
            side_trades
        )

        side_wins = int(
            side_trades[
                "won"
            ].sum()
        )

        side_capital = (
            side_trades[
                "total_cost"
            ].sum()
        )

        side_pnl = (
            side_trades[
                "net_pnl"
            ].sum()
        )

        side_roi = (
            side_pnl
            /
            side_capital
            if side_capital > 0
            else 0.0
        )

        print(
            f"{side}: "
            f"{side_count} trades | "
            f"{side_wins} wins | "
            f"P&L ${side_pnl:+.2f} | "
            f"ROI {side_roi:+.2%}"
        )

    # --------------------------------------------------------
    # Individual trades
    # --------------------------------------------------------

    print()
    print(
        "TRADES"
    )
    print("-" * 80)

    for _, trade in (
        trades.iterrows()
    ):

        result = (
            "WIN"
            if trade[
                "won"
            ]
            else
            "LOSS"
        )

        print(
            f"{trade['target_date']} | "
            f"{trade['ticker']} | "
            f"{trade['side']} | "
            f"Fair "
            f"{trade['adjusted_probability']:.2%} | "
            f"Entry "
            f"{trade['entry_price']:.2%} | "
            f"Fee "
            f"{trade['fee']:.2%} | "
            f"Cost "
            f"{trade['total_cost']:.2%} | "
            f"Edge "
            f"{trade['net_edge']:+.2%} | "
            f"{result} | "
            f"P&L "
            f"${trade['net_pnl']:+.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = (
        load_dataset()
    ).copy()

    dataframe[
        "target_date"
    ] = pd.to_datetime(
        dataframe[
            "target_date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    required_columns = [
        "target_date",
        "ticker",
        "market_probability",
        "yes_ask",
        "no_ask",
        "outcome"
    ]

    missing_columns = [
        column
        for column
        in required_columns
        if column
        not in dataframe.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    print()
    print("=" * 80)

    print(
        "MARKET-ONLY EXECUTABLE STRATEGY"
    )

    print("=" * 80)

    print(
        f"Frozen alpha:      "
        f"{ALPHA:.3f}"
    )

    print(
        f"Minimum net edge:  "
        f"{MIN_NET_EDGE:.2%}"
    )

    print(
        "Max trades/day:    1"
    )

    # --------------------------------------------------------
    # Development
    # --------------------------------------------------------

    development_trades = (
        run_strategy(
            dataframe=
                dataframe,

            start_date=
                DEVELOPMENT_START,

            end_date=
                DEVELOPMENT_END
        )
    )

    summarize_strategy(
        development_trades,
        "DEVELOPMENT PERIOD: AUG 19 - SEP 10"
    )

    # --------------------------------------------------------
    # Frozen holdout
    # --------------------------------------------------------

    holdout_trades = (
        run_strategy(
            dataframe=
                dataframe,

            start_date=
                HOLDOUT_START,

            end_date=
                HOLDOUT_END
        )
    )

    summarize_strategy(
        holdout_trades,
        "HOLDOUT PERIOD: SEP 11 - SEP 26"
    )


if __name__ == "__main__":

    main()