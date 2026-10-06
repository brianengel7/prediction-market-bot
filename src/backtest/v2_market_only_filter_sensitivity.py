import argparse

import numpy as np
import pandas as pd

from src.backtest.v2_market_only_history import (
    load_synchronized_market_history,
)

from src.backtest.v2_market_only_control import (
    fit_market_alpha,
)

from src.backtest.v2_market_only_strategy import (
    calculate_max_drawdown,
)

from src.backtest.v2_market_only_walkforward import (
    LOOKBACK_DAYS,
    EVALUATION_DATE_COUNT,
    get_training_dates,
    build_daily_candidates,
)


# ============================================================
# FILTER SENSITIVITY SETTINGS
# ============================================================

EDGE_THRESHOLDS = [
    0.025,
    0.050,
    0.075,
]

ROI_THRESHOLDS = [
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
]


# Current production/research baseline.
BASELINE_EDGE = 0.025
BASELINE_ROI = 0.100


# ============================================================
# BUILD WALK-FORWARD CANDIDATE HISTORY
# ============================================================

def build_walkforward_candidate_history(
    dataframe,
):
    """
    Fit alpha ONCE per walk-forward date.

    For each evaluation date:

        1. Select the prior 20 eligible settled dates.
        2. Fit alpha using those dates only.
        3. Build every YES/NO candidate for the target date.
        4. Store the full candidate list.

    Execution thresholds are deliberately NOT applied here.

    This allows us to replay the exact same model output through
    many edge / expected-ROI filters without refitting alpha.
    """

    all_dates = sorted(
        dataframe[
            "target_date"
        ].unique()
    )

    # --------------------------------------------------------
    # Determine dates having a complete rolling training set.
    # --------------------------------------------------------

    eligible_evaluation_dates = []

    for target_date in all_dates:

        training_dates = (
            get_training_dates(
                all_dates=
                    all_dates,

                target_date=
                    target_date,

                lookback_days=
                    LOOKBACK_DAYS,
            )
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

    # --------------------------------------------------------
    # Same 30 most recent eligible dates as the main
    # walk-forward strategy.
    # --------------------------------------------------------

    evaluation_dates = (
        eligible_evaluation_dates[
            -EVALUATION_DATE_COUNT:
        ]
    )

    history = []

    # --------------------------------------------------------
    # Fit alpha ONCE for each evaluation date.
    # --------------------------------------------------------

    for target_date in evaluation_dates:

        training_dates = (
            get_training_dates(
                all_dates=
                    all_dates,

                target_date=
                    target_date,

                lookback_days=
                    LOOKBACK_DAYS,
            )
        )

        if (
            len(training_dates)
            !=
            LOOKBACK_DAYS
        ):

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

        (
            alpha,
            alpha_results,
        ) = fit_market_alpha(
            training_data
        )

        best_fit = (
            alpha_results.loc[
                np.isclose(
                    alpha_results[
                        "alpha"
                    ],
                    alpha,
                )
            ]
            .iloc[0]
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
        )

        if (
            len(daily_data)
            !=
            6
        ):

            continue

        candidates = (
            build_daily_candidates(
                daily_data=
                    daily_data,

                alpha=
                    alpha,
            )
        )

        if not candidates:

            continue

        history.append({

            "target_date":
                target_date,

            "training_start":
                training_dates[0],

            "training_end":
                training_dates[-1],

            "alpha":
                float(alpha),

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

            "candidates":
                candidates,
        })

    if (
        len(history)
        !=
        EVALUATION_DATE_COUNT
    ):

        raise RuntimeError(
            f"Expected "
            f"{EVALUATION_DATE_COUNT} "
            f"walk-forward dates, "
            f"built {len(history)}."
        )

    return history


# ============================================================
# SELECT TRADE UNDER ONE FILTER COMBINATION
# ============================================================

def select_trade(
    candidates,
    minimum_edge,
    minimum_expected_roi,
):
    """
    Candidates are already sorted by net edge descending by
    build_daily_candidates().

    This mirrors the main walk-forward strategy exactly:
    choose the first candidate satisfying BOTH filters.
    """

    for candidate in candidates:

        if (
            candidate[
                "net_edge"
            ]
            >=
            minimum_edge
            and
            candidate[
                "expected_return_on_risk"
            ]
            >=
            minimum_expected_roi
        ):

            return candidate

    return None


# ============================================================
# REPLAY ONE FILTER COMBINATION
# ============================================================

def evaluate_filter_combination(
    candidate_history,
    minimum_edge,
    minimum_expected_roi,
):
    trade_rows = []

    for day in candidate_history:

        selected = (
            select_trade(
                candidates=
                    day[
                        "candidates"
                    ],

                minimum_edge=
                    minimum_edge,

                minimum_expected_roi=
                    minimum_expected_roi,
            )
        )

        if selected is None:

            continue

        row = {

            "target_date":
                day[
                    "target_date"
                ],

            "training_start":
                day[
                    "training_start"
                ],

            "training_end":
                day[
                    "training_end"
                ],

            "alpha":
                day[
                    "alpha"
                ],
        }

        row.update(
            selected
        )

        trade_rows.append(
            row
        )

    trades = pd.DataFrame(
        trade_rows
    )

    # --------------------------------------------------------
    # No-trade result
    # --------------------------------------------------------

    if trades.empty:

        return {

            "minimum_edge":
                minimum_edge,

            "minimum_expected_roi":
                minimum_expected_roi,

            "trades":
                0,

            "wins":
                0,

            "losses":
                0,

            "win_rate":
                0.0,

            "yes_trades":
                0,

            "no_trades":
                0,

            "average_net_edge":
                np.nan,

            "average_expected_roi":
                np.nan,

            "capital":
                0.0,

            "expected_pnl":
                0.0,

            "actual_pnl":
                0.0,

            "realized_roi":
                0.0,

            "max_drawdown":
                0.0,
        }, trades

    # --------------------------------------------------------
    # Trading metrics
    # --------------------------------------------------------

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

    realized_roi = (
        actual_pnl
        /
        capital
        if capital > 0
        else 0.0
    )

    max_drawdown = float(
        calculate_max_drawdown(
            trades
        )
    )

    yes_trades = int(
        (
            trades[
                "side"
            ]
            ==
            "YES"
        ).sum()
    )

    no_trades = int(
        (
            trades[
                "side"
            ]
            ==
            "NO"
        ).sum()
    )

    return {

        "minimum_edge":
            minimum_edge,

        "minimum_expected_roi":
            minimum_expected_roi,

        "trades":
            len(trades),

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            wins
            /
            len(trades),

        "yes_trades":
            yes_trades,

        "no_trades":
            no_trades,

        "average_net_edge":
            float(
                trades[
                    "net_edge"
                ].mean()
            ),

        "average_expected_roi":
            float(
                trades[
                    "expected_return_on_risk"
                ].mean()
            ),

        "capital":
            capital,

        "expected_pnl":
            expected_pnl,

        "actual_pnl":
            actual_pnl,

        "realized_roi":
            realized_roi,

        "max_drawdown":
            max_drawdown,

    }, trades


# ============================================================
# RUN COMPLETE GRID
# ============================================================

def run_sensitivity_grid(
    candidate_history,
):
    rows = []

    trade_sets = {}

    for minimum_edge in (
        EDGE_THRESHOLDS
    ):

        for minimum_expected_roi in (
            ROI_THRESHOLDS
        ):

            (
                result,
                trades,
            ) = evaluate_filter_combination(

                candidate_history=
                    candidate_history,

                minimum_edge=
                    minimum_edge,

                minimum_expected_roi=
                    minimum_expected_roi,
            )

            rows.append(
                result
            )

            trade_sets[
                (
                    minimum_edge,
                    minimum_expected_roi,
                )
            ] = trades

    results = pd.DataFrame(
        rows
    )

    return (
        results,
        trade_sets,
    )


# ============================================================
# FORMAT HELPERS
# ============================================================

def pct(
    value,
):
    if pd.isna(
        value
    ):
        return "-"

    return (
        f"{value:.2%}"
    )


def money(
    value,
):
    if pd.isna(
        value
    ):
        return "-"

    return (
        f"${value:+.2f}"
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_full_results(
    results,
):
    print()
    print("=" * 132)
    print(
        "EXECUTION FILTER SENSITIVITY"
    )
    print("=" * 132)

    header = (
        f"{'EDGE':>7} | "
        f"{'ROI FILT':>8} | "
        f"{'TRADES':>6} | "
        f"{'W-L':>7} | "
        f"{'WIN%':>7} | "
        f"{'AVG EDGE':>9} | "
        f"{'AVG EROI':>9} | "
        f"{'CAPITAL':>8} | "
        f"{'EXP P&L':>8} | "
        f"{'P&L':>8} | "
        f"{'ROI':>8} | "
        f"{'MAX DD':>7} | "
        f"{'Y/N':>7}"
    )

    print(
        header
    )

    print(
        "-" * 132
    )

    for _, row in (
        results.iterrows()
    ):

        baseline = (
            np.isclose(
                row[
                    "minimum_edge"
                ],
                BASELINE_EDGE,
            )
            and
            np.isclose(
                row[
                    "minimum_expected_roi"
                ],
                BASELINE_ROI,
            )
        )

        marker = (
            "*"
            if baseline
            else " "
        )

        wl = (
            f"{int(row['wins'])}"
            f"-"
            f"{int(row['losses'])}"
        )

        yn = (
            f"{int(row['yes_trades'])}"
            f"/"
            f"{int(row['no_trades'])}"
        )

        print(
            f"{marker}"
            f"{row['minimum_edge']:>6.1%} | "
            f"{row['minimum_expected_roi']:>8.1%} | "
            f"{int(row['trades']):>6} | "
            f"{wl:>7} | "
            f"{pct(row['win_rate']):>7} | "
            f"{pct(row['average_net_edge']):>9} | "
            f"{pct(row['average_expected_roi']):>9} | "
            f"{row['capital']:>8.2f} | "
            f"{row['expected_pnl']:>+8.2f} | "
            f"{row['actual_pnl']:>+8.2f} | "
            f"{pct(row['realized_roi']):>8} | "
            f"{row['max_drawdown']:>7.2f} | "
            f"{yn:>7}"
        )

    print()
    print(
        "* = current 2.5% edge / 10.0% expected-ROI baseline"
    )


# ============================================================
# MATRIX SUMMARIES
# ============================================================

def print_matrix(
    results,
    value_column,
    title,
    formatter,
):
    matrix = (
        results.pivot(
            index=
                "minimum_edge",

            columns=
                "minimum_expected_roi",

            values=
                value_column,
        )
        .sort_index()
        .sort_index(
            axis=1
        )
    )

    print()
    print("=" * 90)
    print(
        title
    )
    print("=" * 90)

    column_labels = [
        f"{column:.1%}"
        for column in matrix.columns
    ]

    print(
        f"{'EDGE \\ ROI':>12} | "
        +
        " | ".join(
            f"{label:>10}"
            for label in column_labels
        )
    )

    print(
        "-" * (
            15
            +
            13
            *
            len(column_labels)
        )
    )

    for edge, row in (
        matrix.iterrows()
    ):

        values = []

        for value in row:

            values.append(
                f"{formatter(value):>10}"
            )

        print(
            f"{edge:>11.1%} | "
            +
            " | ".join(
                values
            )
        )


# ============================================================
# BASELINE TRADE PATH
# ============================================================

def print_baseline_trades(
    trade_sets,
):
    key = (
        BASELINE_EDGE,
        BASELINE_ROI,
    )

    baseline = (
        trade_sets[
            key
        ]
    )

    print()
    print("=" * 110)
    print(
        "BASELINE TRADE PATH "
        "(2.5% EDGE / 10.0% EXPECTED ROI)"
    )
    print("=" * 110)

    if baseline.empty:

        print(
            "No baseline trades."
        )

        return

    for _, row in (
        baseline.iterrows()
    ):

        result = (
            "WIN"
            if row[
                "won"
            ]
            else
            "LOSS"
        )

        print(
            f"{row['target_date']} | "
            f"alpha {row['alpha']:.3f} | "
            f"{row['side']:3s} | "
            f"{row['ticker']} | "
            f"edge "
            f"{row['net_edge']:+.2%} | "
            f"exp ROI "
            f"{row['expected_return_on_risk']:+.2%} | "
            f"{result:4s} | "
            f"P&L "
            f"${row['net_pnl']:+.2f}"
        )


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

def print_robustness_summary(
    results,
):
    profitable = (
        results[
            results[
                "actual_pnl"
            ]
            >
            0
        ]
    )

    nonnegative = (
        results[
            results[
                "actual_pnl"
            ]
            >=
            0
        ]
    )

    print()
    print("=" * 90)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 90)

    print(
        f"Filter combinations tested: "
        f"{len(results)}"
    )

    print(
        f"Profitable combinations:    "
        f"{len(profitable)}/"
        f"{len(results)} "
        f"({len(profitable) / len(results):.1%})"
    )

    print(
        f"Nonnegative combinations:   "
        f"{len(nonnegative)}/"
        f"{len(results)} "
        f"({len(nonnegative) / len(results):.1%})"
    )

    baseline = (
        results[
            np.isclose(
                results[
                    "minimum_edge"
                ],
                BASELINE_EDGE,
            )
            &
            np.isclose(
                results[
                    "minimum_expected_roi"
                ],
                BASELINE_ROI,
            )
        ]
    )

    if not baseline.empty:

        row = (
            baseline.iloc[0]
        )

        print()
        print(
            "Current baseline:"
        )

        print(
            f"  Trades:       "
            f"{int(row['trades'])}"
        )

        print(
            f"  Wins/Losses:  "
            f"{int(row['wins'])}/"
            f"{int(row['losses'])}"
        )

        print(
            f"  P&L:          "
            f"${row['actual_pnl']:+.2f}"
        )

        print(
            f"  ROI:          "
            f"{row['realized_roi']:+.2%}"
        )

        print(
            f"  Max drawdown: "
            f"${row['max_drawdown']:.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--series-ticker",
        default=
            "KXHIGHNY",
        help=(
            "Kalshi weather series to test. "
            "Defaults to KXHIGHNY."
        ),
    )

    args = (
        parser.parse_args()
    )

    series_ticker = (
        str(
            args.series_ticker
        )
        .strip()
        .upper()
    )

    if not series_ticker:

        parser.error(
            "--series-ticker cannot be empty."
        )

    dataframe = (
        load_synchronized_market_history(
            minimum_dates=
                50,

            series_ticker=
                series_ticker,
        )
        .copy()
    )

    # --------------------------------------------------------
    # Fail fast if the loader mixes series.
    # --------------------------------------------------------

    if not dataframe[
        "ticker"
    ].astype(
        str
    ).str.startswith(
        series_ticker
        +
        "-"
    ).all():

        raise RuntimeError(
            f"History contains contracts "
            f"outside {series_ticker}."
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

    print()
    print("=" * 90)
    print(
        f"ALPHA EXECUTION FILTER SENSITIVITY "
        f"[{series_ticker}]"
    )
    print("=" * 90)

    print(
        f"Series:             "
        f"{series_ticker}"
    )

    print(
        f"Synchronized dates: "
        f"{dataframe['target_date'].nunique()}"
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

    print()
    print(
        "Edge thresholds:    "
        +
        ", ".join(
            f"{value:.1%}"
            for value in EDGE_THRESHOLDS
        )
    )

    print(
        "ROI thresholds:     "
        +
        ", ".join(
            f"{value:.1%}"
            for value in ROI_THRESHOLDS
        )
    )

    print()
    print(
        "Fitting rolling alpha once per "
        "walk-forward date..."
    )

    candidate_history = (
        build_walkforward_candidate_history(
            dataframe
        )
    )

    alpha_values = [
        day[
            "alpha"
        ]
        for day in candidate_history
    ]

    print(
        f"Built candidate sets: "
        f"{len(candidate_history)}"
    )

    print(
        f"Mean alpha:          "
        f"{np.mean(alpha_values):.3f}"
    )

    print(
        f"Median alpha:        "
        f"{np.median(alpha_values):.3f}"
    )

    print(
        f"Min alpha:           "
        f"{np.min(alpha_values):.3f}"
    )

    print(
        f"Max alpha:           "
        f"{np.max(alpha_values):.3f}"
    )

    (
        results,
        trade_sets,
    ) = run_sensitivity_grid(
        candidate_history
    )

    print_full_results(
        results
    )

    print_matrix(
        results=
            results,

        value_column=
            "trades",

        title=
            "TRADE COUNT MATRIX",

        formatter=
            lambda value:
                f"{int(value)}",
    )

    print_matrix(
        results=
            results,

        value_column=
            "actual_pnl",

        title=
            "ACTUAL P&L MATRIX",

        formatter=
            lambda value:
                f"{value:+.2f}",
    )

    print_matrix(
        results=
            results,

        value_column=
            "realized_roi",

        title=
            "REALIZED ROI MATRIX",

        formatter=
            lambda value:
                f"{value:+.1%}",
    )

    print_matrix(
        results=
            results,

        value_column=
            "max_drawdown",

        title=
            "MAX DRAWDOWN MATRIX",

        formatter=
            lambda value:
                f"{value:.2f}",
    )

    print_robustness_summary(
        results
    )

    print_baseline_trades(
        trade_sets
    )


if __name__ == "__main__":

    main()