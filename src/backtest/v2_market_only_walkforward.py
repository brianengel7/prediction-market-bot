import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import load_dataset

from src.backtest.v2_market_only_control import (
    fit_market_alpha,
    market_only_probabilities
)

from src.backtest.v2_market_only_strategy import (
    calculate_max_drawdown
)

from src.kalshi.fees import (
    calculate_taker_fee
)


# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

LOOKBACK_DAYS = 20

MIN_NET_EDGE = 0.025

EVALUATION_START = "2026-08-19"
EVALUATION_END = "2026-09-26"

PREVIOUS_HOLDOUT_START = "2026-09-11"
PREVIOUS_HOLDOUT_END = "2026-09-26"


# ============================================================
# TRAINING WINDOW
# ============================================================

def get_training_dates(
    all_dates,
    target_date
):
    """
    Target date = day whose maximum temperature is traded.

    Entry occurs on D-1 at 20:05 UTC.

    Therefore the D-1 daily-high outcome is NOT yet safely
    available for calibration.

    Example:

        target = Sep 11
        decision = Sep 10
        newest allowed training outcome = Sep 9
    """

    target_timestamp = pd.Timestamp(
        target_date
    )

    decision_date = (
        target_timestamp
        -
        pd.Timedelta(
            days=1
        )
    )

    eligible_dates = [
        date
        for date in all_dates
        if pd.Timestamp(
            date
        ) < decision_date
    ]

    if len(
        eligible_dates
    ) < LOOKBACK_DAYS:

        return []

    return eligible_dates[
        -LOOKBACK_DAYS:
    ]


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_daily_candidates(
    daily_data,
    alpha
):
    daily_data = (
        daily_data
        .copy()
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

    adjusted_yes = (
        market_only_probabilities(
            daily_data,
            alpha
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

        market_yes = float(
            row[
                "market_probability"
            ]
        )

        outcome = int(
            row[
                "outcome"
            ]
        )

        # ----------------------------------------------------
        # YES
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

            fee = float(
                calculate_taker_fee(
                    price=
                        yes_ask,

                    contracts=
                        1,

                    multiplier=
                        1
                )
            )

            total_cost = (
                yes_ask
                +
                fee
            )

            net_edge = (
                yes_fair
                -
                total_cost
            )

            won = (
                outcome == 1
            )

            payout = (
                1.0
                if won
                else 0.0
            )

            candidates.append({

                "ticker":
                    row[
                        "ticker"
                    ],

                "side":
                    "YES",

                "kalshi_probability":
                    market_yes,

                "adjusted_probability":
                    yes_fair,

                "entry_price":
                    yes_ask,

                "fee":
                    fee,

                "total_cost":
                    total_cost,

                "net_edge":
                    net_edge,

                "won":
                    won,

                "payout":
                    payout,

                "net_pnl":
                    (
                        payout
                        -
                        total_cost
                    )
            })

        # ----------------------------------------------------
        # NO
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

            fee = float(
                calculate_taker_fee(
                    price=
                        no_ask,

                    contracts=
                        1,

                    multiplier=
                        1
                )
            )

            total_cost = (
                no_ask
                +
                fee
            )

            net_edge = (
                no_fair
                -
                total_cost
            )

            won = (
                outcome == 0
            )

            payout = (
                1.0
                if won
                else 0.0
            )

            candidates.append({

                "ticker":
                    row[
                        "ticker"
                    ],

                "side":
                    "NO",

                "kalshi_probability":
                    (
                        1.0
                        -
                        market_yes
                    ),

                "adjusted_probability":
                    no_fair,

                "entry_price":
                    no_ask,

                "fee":
                    fee,

                "total_cost":
                    total_cost,

                "net_edge":
                    net_edge,

                "won":
                    won,

                "payout":
                    payout,

                "net_pnl":
                    (
                        payout
                        -
                        total_cost
                    )
            })

    candidates.sort(
        key=lambda row:
            row[
                "net_edge"
            ],

        reverse=True
    )

    return candidates


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def run_walkforward(
    dataframe
):
    all_dates = sorted(
        dataframe[
            "target_date"
        ].unique()
    )

    decisions = []

    for target_date in all_dates:

        if (
            target_date
            <
            EVALUATION_START
            or
            target_date
            >
            EVALUATION_END
        ):
            continue

        # ----------------------------------------------------
        # Strict historical training window
        # ----------------------------------------------------

        training_dates = (
            get_training_dates(
                all_dates=
                    all_dates,

                target_date=
                    target_date
            )
        )

        if len(
            training_dates
        ) < LOOKBACK_DAYS:

            continue

        training_data = (
            dataframe[
                dataframe[
                    "target_date"
                ].isin(
                    training_dates
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Fit alpha using PRIOR outcomes only
        # ----------------------------------------------------

        (
            alpha,
            alpha_results
        ) = fit_market_alpha(
            training_data
        )

        best_fit = (
            alpha_results.loc[
                np.isclose(
                    alpha_results[
                        "alpha"
                    ],
                    alpha
                )
            ]
            .iloc[
                0
            ]
        )

        # ----------------------------------------------------
        # Today's six contracts
        # ----------------------------------------------------

        daily_data = (
            dataframe[
                dataframe[
                    "target_date"
                ]
                ==
                target_date
            ]
            .copy()
        )

        if len(
            daily_data
        ) != 6:

            continue

        candidates = (
            build_daily_candidates(
                daily_data=
                    daily_data,

                alpha=
                    alpha
            )
        )

        if not candidates:
            continue

        best = candidates[
            0
        ]

        trade_taken = (
            best[
                "net_edge"
            ]
            >=
            MIN_NET_EDGE
        )

        decision = {

            "target_date":
                target_date,

            "training_start":
                training_dates[
                    0
                ],

            "training_end":
                training_dates[
                    -1
                ],

            "alpha":
                alpha,

            "training_log_loss":
                float(
                    best_fit[
                        "log_loss"
                    ]
                ),

            "training_brier":
                float(
                    best_fit[
                        "brier"
                    ]
                ),

            "ticker":
                best[
                    "ticker"
                ],

            "side":
                best[
                    "side"
                ],

            "kalshi_probability":
                best[
                    "kalshi_probability"
                ],

            "adjusted_probability":
                best[
                    "adjusted_probability"
                ],

            "entry_price":
                best[
                    "entry_price"
                ],

            "fee":
                best[
                    "fee"
                ],

            "total_cost":
                best[
                    "total_cost"
                ],

            "net_edge":
                best[
                    "net_edge"
                ],

            "trade_taken":
                trade_taken,

            "won":
                (
                    best[
                        "won"
                    ]
                    if trade_taken
                    else None
                ),

            "net_pnl":
                (
                    best[
                        "net_pnl"
                    ]
                    if trade_taken
                    else 0.0
                )
        }

        decisions.append(
            decision
        )

    return pd.DataFrame(
        decisions
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_period(
    decisions,
    label,
    start_date=None,
    end_date=None
):
    period = (
        decisions.copy()
    )

    if start_date is not None:

        period = period[
            period[
                "target_date"
            ]
            >=
            start_date
        ]

    if end_date is not None:

        period = period[
            period[
                "target_date"
            ]
            <=
            end_date
        ]

    print()
    print("=" * 90)
    print(label)
    print("=" * 90)

    if period.empty:

        print(
            "No eligible walk-forward dates."
        )

        return

    trades = (
        period[
            period[
                "trade_taken"
            ]
        ]
        .copy()
    )

    print(
        f"Decision days:      "
        f"{len(period)}"
    )

    print(
        f"Trades:             "
        f"{len(trades)}"
    )

    print(
        f"Passes:             "
        f"{len(period) - len(trades)}"
    )

    print()

    print(
        f"Mean alpha:         "
        f"{period['alpha'].mean():.3f}"
    )

    print(
        f"Median alpha:       "
        f"{period['alpha'].median():.3f}"
    )

    print(
        f"Min alpha:          "
        f"{period['alpha'].min():.3f}"
    )

    print(
        f"Max alpha:          "
        f"{period['alpha'].max():.3f}"
    )

    if trades.empty:

        return

    wins = int(
        trades[
            "won"
        ].sum()
    )

    losses = (
        len(trades)
        -
        wins
    )

    capital = float(
        trades[
            "total_cost"
        ].sum()
    )

    pnl = float(
        trades[
            "net_pnl"
        ].sum()
    )

    expected_pnl = float(
        trades[
            "net_edge"
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

    print()
    print(
        f"Wins:               "
        f"{wins}"
    )

    print(
        f"Losses:             "
        f"{losses}"
    )

    print(
        f"Win rate:           "
        f"{wins / len(trades):.2%}"
    )

    print(
        f"Average net edge:   "
        f"{trades['net_edge'].mean():.2%}"
    )

    print()

    print(
        f"Capital:            "
        f"${capital:.2f}"
    )

    print(
        f"Expected P&L:       "
        f"${expected_pnl:+.2f}"
    )

    print(
        f"Actual P&L:         "
        f"${pnl:+.2f}"
    )

    print(
        f"ROI:                "
        f"{roi:+.2%}"
    )

    print(
        f"Max drawdown:       "
        f"${max_drawdown:.2f}"
    )

    print()
    print(
        "SIDE BREAKDOWN"
    )
    print("-" * 90)

    for side in [
        "YES",
        "NO"
    ]:

        side_trades = (
            trades[
                trades[
                    "side"
                ]
                ==
                side
            ]
        )

        if side_trades.empty:
            continue

        side_wins = int(
            side_trades[
                "won"
            ].sum()
        )

        side_capital = float(
            side_trades[
                "total_cost"
            ].sum()
        )

        side_pnl = float(
            side_trades[
                "net_pnl"
            ].sum()
        )

        side_roi = (
            side_pnl
            /
            side_capital
        )

        print(
            f"{side}: "
            f"{len(side_trades)} trades | "
            f"{side_wins} wins | "
            f"P&L ${side_pnl:+.2f} | "
            f"ROI {side_roi:+.2%}"
        )


# ============================================================
# PRINT DAILY PATH
# ============================================================

def print_daily_decisions(
    decisions
):

    print()
    print("=" * 90)
    print(
        "DAILY WALK-FORWARD DECISIONS"
    )
    print("=" * 90)

    for _, row in (
        decisions.iterrows()
    ):

        action = (
            row[
                "side"
            ]
            if row[
                "trade_taken"
            ]
            else
            "PASS"
        )

        pnl_text = (
            f"${row['net_pnl']:+.2f}"
            if row[
                "trade_taken"
            ]
            else
            "-"
        )

        print(
            f"{row['target_date']} | "
            f"train "
            f"{row['training_start']} → "
            f"{row['training_end']} | "
            f"alpha {row['alpha']:.3f} | "
            f"{action:4s} | "
            f"{row['ticker']} | "
            f"edge "
            f"{row['net_edge']:+.2%} | "
            f"P&L {pnl_text}"
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

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: "
            f"{missing}"
        )

    decisions = (
        run_walkforward(
            dataframe
        )
    )

    print()
    print("=" * 90)
    print(
        "MARKET-ONLY WALK-FORWARD STRATEGY"
    )
    print("=" * 90)

    print(
        f"Lookback:           "
        f"{LOOKBACK_DAYS} settled dates"
    )

    print(
        "Outcome embargo:    "
        "1 calendar day"
    )

    print(
        f"Minimum net edge:   "
        f"{MIN_NET_EDGE:.2%}"
    )

    print(
        "Max trades/day:     1"
    )

    summarize_period(
        decisions=
            decisions,

        label=
            "FULL WALK-FORWARD PERIOD"
    )

    summarize_period(
        decisions=
            decisions,

        label=
            "AUG 19 - SEP 10",

        start_date=
            "2026-08-19",

        end_date=
            "2026-09-10"
    )

    summarize_period(
        decisions=
            decisions,

        label=
            "SEP 11 - SEP 26 "
            "(PREVIOUS HOLDOUT; NOW RESEARCHED)",

        start_date=
            PREVIOUS_HOLDOUT_START,

        end_date=
            PREVIOUS_HOLDOUT_END
    )

    print_daily_decisions(
        decisions
    )


if __name__ == "__main__":

    main()