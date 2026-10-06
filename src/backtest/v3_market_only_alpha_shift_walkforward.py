import argparse

import numpy as np
import pandas as pd

from src.backtest.v2_market_only_history import (
    load_synchronized_market_history,
)

from src.backtest.v2_market_only_walkforward import (
    get_training_dates,
)

from src.backtest.v2_market_only_control import (
    ALPHA_VALUES,
    kalshi_probabilities,
    score_model,
)

from src.backtest.v2_market_only_strategy import (
    calculate_max_drawdown,
)

from src.backtest.v3_market_only_shift import (
    SHIFT_VALUES,
    fit_market_alpha_shift,
    market_alpha_shift_probabilities,
)

from src.kalshi.fees import (
    calculate_taker_fee,
)


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK_DAYS = 20

EVALUATION_DATE_COUNT = 30

MIN_NET_EDGE = 0.025

MIN_EXPECTED_RETURN_ON_RISK = 0.10

# ============================================================
# FIT ALPHA + SHIFT
# ============================================================

def fit_rolling_alpha_shift(
    training_data,
):

    (
        _,
        _,
        results,
    ) = fit_market_alpha_shift(
        training_data=
            training_data,

        alpha_values=
            ALPHA_VALUES,

        shift_values=
            SHIFT_VALUES,
    )

    if results.empty:

        raise RuntimeError(
            "Alpha/shift fit returned "
            "no valid results."
        )

    ranked = results.copy()

    ranked[
        "abs_shift"
    ] = ranked[
        "shift_f"
    ].abs()

    ranked[
        "alpha_distance"
    ] = (
        ranked["alpha"]
        -
        1.0
    ).abs()

    # Prefer probability fit first.
    #
    # If effectively tied, prefer the more neutral model:
    # shift closer to zero and alpha closer to one.

    ranked = (
        ranked
        .sort_values(
            [
                "log_loss",
                "brier",
                "abs_shift",
                "alpha_distance",
                "alpha",
                "shift_f",
            ]
        )
        .reset_index(drop=True)
    )

    best = ranked.iloc[0]

    return (
        float(
            best["alpha"]
        ),

        float(
            best["shift_f"]
        ),

        best,
    )


# ============================================================
# DAILY TRADE CANDIDATES
# ============================================================

def build_daily_candidates(
    daily_data,
    adjusted_yes,
):

    daily_data = (
        daily_data
        .copy()
        .reset_index(drop=True)
    )

    adjusted_yes = np.asarray(
        adjusted_yes,
        dtype=float,
    )

    if len(daily_data) != 6:

        raise ValueError(
            f"Expected 6 contracts, found "
            f"{len(daily_data)}."
        )

    candidates = []

    for index, row in daily_data.iterrows():

        yes_fair = float(
            adjusted_yes[index]
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
                    price=yes_ask,
                    contracts=1,
                    multiplier=1,
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

            expected_return_on_risk = (
                net_edge / total_cost
                if total_cost > 0
                else float("-inf")
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
                    row["ticker"],

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

                "expected_return_on_risk":
                    expected_return_on_risk,

                "won":
                    won,

                "net_pnl":
                    payout
                    -
                    total_cost,
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
                    price=no_ask,
                    contracts=1,
                    multiplier=1,
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

            expected_return_on_risk = (
                net_edge / total_cost
                if total_cost > 0
                else float("-inf")
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
                    row["ticker"],

                "side":
                    "NO",

                "kalshi_probability":
                    1.0
                    -
                    market_yes,

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

                "expected_return_on_risk":
                    expected_return_on_risk,

                "won":
                    won,

                "net_pnl":
                    payout
                    -
                    total_cost,
            })

    candidates.sort(
        key=lambda row:
            row["net_edge"],
        reverse=True,
    )

    return candidates


# ============================================================
# WALK-FORWARD
# ============================================================

def run_walkforward(
    dataframe,
):

    all_dates = sorted(
        dataframe[
            "target_date"
        ].unique()
    )

    eligible_evaluation_dates = []

    for target_date in all_dates:

        training_dates = get_training_dates(
            all_dates=all_dates,
            target_date=target_date,
            lookback_days=LOOKBACK_DAYS,
        )

        if (
            len(training_dates)
            ==
            LOOKBACK_DAYS
        ):

            eligible_evaluation_dates.append(
                target_date
            )

    if (
        len(
            eligible_evaluation_dates
        )
        <
        EVALUATION_DATE_COUNT
    ):

        raise RuntimeError(
            f"Only "
            f"{len(eligible_evaluation_dates)} "
            f"eligible walk-forward dates; "
            f"{EVALUATION_DATE_COUNT} required."
        )

    evaluation_dates = (
        eligible_evaluation_dates[
            -EVALUATION_DATE_COUNT:
        ]
    )

    decisions = []

    for target_date in evaluation_dates:

        training_dates = get_training_dates(
            all_dates=all_dates,
            target_date=target_date,
            lookback_days=LOOKBACK_DAYS,
        )

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
        # FIT ALPHA + SHIFT USING PRIOR OUTCOMES ONLY
        # ----------------------------------------------------

        (
            alpha,
            shift_f,
            best_fit,
        ) = fit_rolling_alpha_shift(
            training_data
        )

        daily_data = (
            dataframe[
                dataframe[
                    "target_date"
                ]
                ==
                target_date
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(daily_data) != 6:
            continue

        # ----------------------------------------------------
        # OUT-OF-SAMPLE PROBABILITY SCORING
        # ----------------------------------------------------

        kalshi_scores = (
            score_model(
                daily_data,
                kalshi_probabilities,
            )
        )

        adjusted_scores = (
            score_model(
                daily_data,

                lambda group:
                    market_alpha_shift_probabilities(
                        group,
                        alpha,
                        shift_f,
                    )
            )
        )

        if (
            len(kalshi_scores) != 1
            or
            len(adjusted_scores) != 1
        ):

            raise RuntimeError(
                "Could not score "
                f"{target_date}."
            )

        kalshi_score = (
            kalshi_scores.iloc[0]
        )

        adjusted_score = (
            adjusted_scores.iloc[0]
        )

        adjusted_yes = (
            market_alpha_shift_probabilities(
                daily_data,
                alpha,
                shift_f,
            )
        )

        candidates = (
            build_daily_candidates(
                daily_data=
                    daily_data,

                adjusted_yes=
                    adjusted_yes,
            )
        )

        if not candidates:
            continue

        qualifying_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate[
                    "net_edge"
                ]
                >=
                MIN_NET_EDGE
                and
                candidate[
                    "expected_return_on_risk"
                ]
                >=
                MIN_EXPECTED_RETURN_ON_RISK
            )
        ]

        if qualifying_candidates:

            best = (
                qualifying_candidates[
                    0
                ]
            )

            trade_taken = True

        else:

            best = candidates[0]

            trade_taken = False

        decisions.append({

            "target_date":
                target_date,

            "training_start":
                training_dates[0],

            "training_end":
                training_dates[-1],

            "alpha":
                alpha,

            "shift_f":
                shift_f,

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
                best["ticker"],

            "side":
                best["side"],

            "kalshi_probability":
                best[
                    "kalshi_probability"
                ],

            "adjusted_probability":
                best[
                    "adjusted_probability"
                ],

            "kalshi_brier":
                float(
                    kalshi_score[
                        "brier"
                    ]
                ),

            "adjusted_brier":
                float(
                    adjusted_score[
                        "brier"
                    ]
                ),

            "kalshi_log_loss":
                float(
                    kalshi_score[
                        "log_loss"
                    ]
                ),

            "adjusted_log_loss":
                float(
                    adjusted_score[
                        "log_loss"
                    ]
                ),

            "entry_price":
                best["entry_price"],

            "fee":
                best["fee"],

            "total_cost":
                best["total_cost"],

            "net_edge":
                best["net_edge"],

            "expected_return_on_risk":
                best[
                    "expected_return_on_risk"
                ],

            "trade_taken":
                trade_taken,

            "won":
                (
                    best["won"]
                    if trade_taken
                    else None
                ),

            "net_pnl":
                (
                    best["net_pnl"]
                    if trade_taken
                    else 0.0
                ),
        })

    return pd.DataFrame(
        decisions
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    decisions,
):

    print()
    print("=" * 90)
    print(
        "ALPHA + SHIFT WALK-FORWARD"
    )
    print("=" * 90)

    trades = (
        decisions[
            decisions[
                "trade_taken"
            ]
        ]
        .copy()
    )

    print(
        f"Decision days:      "
        f"{len(decisions)}"
    )

    print(
        f"Trades:             "
        f"{len(trades)}"
    )

    print(
        f"Passes:             "
        f"{len(decisions) - len(trades)}"
    )

    print()

    print(
        f"Mean alpha:         "
        f"{decisions['alpha'].mean():.3f}"
    )

    print(
        f"Median alpha:       "
        f"{decisions['alpha'].median():.3f}"
    )

    print(
        f"Min alpha:          "
        f"{decisions['alpha'].min():.3f}"
    )

    print(
        f"Max alpha:          "
        f"{decisions['alpha'].max():.3f}"
    )

    print()

    print(
        f"Mean shift:         "
        f"{decisions['shift_f'].mean():+.3f}°F"
    )

    print(
        f"Median shift:       "
        f"{decisions['shift_f'].median():+.3f}°F"
    )

    print(
        f"Min shift:          "
        f"{decisions['shift_f'].min():+.3f}°F"
    )

    print(
        f"Max shift:          "
        f"{decisions['shift_f'].max():+.3f}°F"
    )

    shift_boundary_count = int(
        np.isclose(
            decisions["shift_f"],
            SHIFT_VALUES[0],
        ).sum()
        +
        np.isclose(
            decisions["shift_f"],
            SHIFT_VALUES[-1],
        ).sum()
    )

    alpha_boundary_count = int(
        np.isclose(
            decisions["alpha"],
            ALPHA_VALUES[0],
        ).sum()
        +
        np.isclose(
            decisions["alpha"],
            ALPHA_VALUES[-1],
        ).sum()
    )

    print()

    print(
        f"Shift boundary fits:"
        f" {shift_boundary_count}"
    )

    print(
        f"Alpha boundary fits:"
        f" {alpha_boundary_count}"
    )

    if not trades.empty:

        wins = int(
            trades[
                "won"
            ].sum()
        )

        capital = float(
            trades[
                "total_cost"
            ].sum()
        )

        expected_pnl = float(
            trades[
                "net_edge"
            ].sum()
        )

        actual_pnl = float(
            trades[
                "net_pnl"
            ].sum()
        )

        roi = (
            actual_pnl
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
            f"{len(trades) - wins}"
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
            f"${actual_pnl:+.2f}"
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
    print("-" * 90)
    print(
        "OUT-OF-SAMPLE PROBABILITY QUALITY"
    )
    print("-" * 90)

    raw_brier = (
        decisions[
            "kalshi_brier"
        ].mean()
    )

    adjusted_brier = (
        decisions[
            "adjusted_brier"
        ].mean()
    )

    raw_log = (
        decisions[
            "kalshi_log_loss"
        ].mean()
    )

    adjusted_log = (
        decisions[
            "adjusted_log_loss"
        ].mean()
    )

    print(
        f"Brier raw:          "
        f"{raw_brier:.4f}"
    )

    print(
        f"Brier alpha+shift:  "
        f"{adjusted_brier:.4f}"
    )

    print(
        f"Brier improvement:  "
        f"{(raw_brier - adjusted_brier) / raw_brier:+.2%}"
    )

    print()

    print(
        f"Log loss raw:       "
        f"{raw_log:.4f}"
    )

    print(
        f"Log alpha+shift:    "
        f"{adjusted_log:.4f}"
    )

    print(
        f"Log improvement:    "
        f"{(raw_log - adjusted_log) / raw_log:+.2%}"
    )

    brier_days = int(
        (
            decisions[
                "adjusted_brier"
            ]
            <
            decisions[
                "kalshi_brier"
            ]
        ).sum()
    )

    log_days = int(
        (
            decisions[
                "adjusted_log_loss"
            ]
            <
            decisions[
                "kalshi_log_loss"
            ]
        ).sum()
    )

    print()
    print(
        f"Brier improved:     "
        f"{brier_days}/{len(decisions)} days"
    )

    print(
        f"Log loss improved:  "
        f"{log_days}/{len(decisions)} days"
    )

    print()
    print("=" * 90)
    print(
        "DAILY WALK-FORWARD DECISIONS"
    )
    print("=" * 90)

    for _, row in decisions.iterrows():

        action = (
            row["side"]
            if row["trade_taken"]
            else "PASS"
        )

        pnl_text = (
            f"${row['net_pnl']:+.2f}"
            if row["trade_taken"]
            else "-"
        )

        print(
            f"{row['target_date']} | "
            f"train "
            f"{row['training_start']} → "
            f"{row['training_end']} | "
            f"alpha {row['alpha']:.3f} | "
            f"shift {row['shift_f']:+.2f}°F | "
            f"{action:4s} | "
            f"{row['ticker']} | "
            f"edge {row['net_edge']:+.2%} | "
            f"ROI "
            f"{row['expected_return_on_risk']:+.2%} | "
            f"P&L {pnl_text}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--series-ticker",
        default="KXHIGHNY",
    )

    args = parser.parse_args()

    dataframe = (
        load_synchronized_market_history(
            minimum_dates=(
                LOOKBACK_DAYS
                +
                EVALUATION_DATE_COUNT
            ),
            series_ticker=
                args.series_ticker,
        )
        .copy()
    )

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
        "outcome",
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
        "NYC MARKET ALPHA + SHIFT TEST"
    )
    print("=" * 90)

    print(
        f"Series:             "
        f"{args.series_ticker}"
    )

    print(
        f"Lookback:           "
        f"{LOOKBACK_DAYS} settled dates"
    )

    print(
        f"Evaluation:         "
        f"{EVALUATION_DATE_COUNT} dates"
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
        f"Min expected ROI:   "
        f"{MIN_EXPECTED_RETURN_ON_RISK:.2%}"
    )

    print(
        "Max trades/day:     1"
    )

    summarize(
        decisions
    )


if __name__ == "__main__":
    main()