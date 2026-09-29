import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import load_dataset

from src.backtest.v2_market_only_control import (
    fit_market_alpha,
    score_model,
    kalshi_probabilities,
    market_only_probabilities
)


# ============================================================
# SETTINGS
# ============================================================

FROZEN_ALPHA = 1.875

HOLDOUT_START = "2026-09-11"

MIN_EXPANDING_DAYS = 10
ROLLING_WINDOW_DAYS = 20


# ============================================================
# ALPHA STABILITY
# ============================================================

def run_alpha_stability(dataframe):

    pre_holdout = (
        dataframe[
            dataframe["target_date"] < HOLDOUT_START
        ]
        .copy()
    )

    dates = sorted(
        pre_holdout["target_date"].unique()
    )

    print()
    print("=" * 90)
    print("ALPHA STABILITY")
    print("=" * 90)

    # --------------------------------------------------------
    # Expanding window
    # --------------------------------------------------------

    expanding_results = []

    for end_index in range(
        MIN_EXPANDING_DAYS,
        len(dates) + 1
    ):

        training_dates = dates[:end_index]

        training_data = (
            pre_holdout[
                pre_holdout[
                    "target_date"
                ].isin(
                    training_dates
                )
            ]
        )

        alpha, results = fit_market_alpha(
            training_data
        )

        best_row = (
            results.loc[
                np.isclose(
                    results["alpha"],
                    alpha
                )
            ]
            .iloc[0]
        )

        expanding_results.append({
            "days":
                len(training_dates),

            "start_date":
                training_dates[0],

            "end_date":
                training_dates[-1],

            "alpha":
                alpha,

            "log_loss":
                best_row["log_loss"],

            "brier":
                best_row["brier"]
        })

    expanding_df = pd.DataFrame(
        expanding_results
    )

    print()
    print("EXPANDING WINDOW")
    print("-" * 90)

    print(
        expanding_df.to_string(
            index=False,
            formatters={
                "alpha":
                    "{:.3f}".format,

                "log_loss":
                    "{:.4f}".format,

                "brier":
                    "{:.4f}".format
            }
        )
    )

    # --------------------------------------------------------
    # Rolling 20-day window
    # --------------------------------------------------------

    rolling_results = []

    if len(dates) >= ROLLING_WINDOW_DAYS:

        for end_index in range(
            ROLLING_WINDOW_DAYS,
            len(dates) + 1
        ):

            training_dates = (
                dates[
                    end_index
                    -
                    ROLLING_WINDOW_DAYS
                    :
                    end_index
                ]
            )

            training_data = (
                pre_holdout[
                    pre_holdout[
                        "target_date"
                    ].isin(
                        training_dates
                    )
                ]
            )

            alpha, results = fit_market_alpha(
                training_data
            )

            best_row = (
                results.loc[
                    np.isclose(
                        results["alpha"],
                        alpha
                    )
                ]
                .iloc[0]
            )

            rolling_results.append({
                "start_date":
                    training_dates[0],

                "end_date":
                    training_dates[-1],

                "alpha":
                    alpha,

                "log_loss":
                    best_row["log_loss"],

                "brier":
                    best_row["brier"]
            })

    rolling_df = pd.DataFrame(
        rolling_results
    )

    print()
    print("ROLLING 20-DAY WINDOW")
    print("-" * 90)

    if rolling_df.empty:

        print(
            "Not enough data for rolling window."
        )

    else:

        print(
            rolling_df.to_string(
                index=False,
                formatters={
                    "alpha":
                        "{:.3f}".format,

                    "log_loss":
                        "{:.4f}".format,

                    "brier":
                        "{:.4f}".format
                }
            )
        )

        print()
        print("ROLLING ALPHA SUMMARY")
        print("-" * 90)

        print(
            f"Mean alpha:   "
            f"{rolling_df['alpha'].mean():.3f}"
        )

        print(
            f"Median alpha: "
            f"{rolling_df['alpha'].median():.3f}"
        )

        print(
            f"Std alpha:    "
            f"{rolling_df['alpha'].std(ddof=1):.3f}"
        )

        print(
            f"Min alpha:    "
            f"{rolling_df['alpha'].min():.3f}"
        )

        print(
            f"Max alpha:    "
            f"{rolling_df['alpha'].max():.3f}"
        )

        print(
            f"Frozen alpha: "
            f"{FROZEN_ALPHA:.3f}"
        )


# ============================================================
# RAW MIDPOINT / NORMALIZATION AUDIT
# ============================================================

def run_midpoint_audit(dataframe):

    if "market_mid" not in dataframe.columns:

        raise RuntimeError(
            "Dataset does not contain market_mid."
        )

    daily_rows = []

    for (
        target_date,
        group
    ) in dataframe.groupby(
        "target_date"
    ):

        if len(group) != 6:
            continue

        raw_mid_sum = float(
            group[
                "market_mid"
            ].sum()
        )

        normalized_sum = float(
            group[
                "market_probability"
            ].sum()
        )

        average_spread = (
            float(
                group["spread"].mean()
            )
            if "spread"
            in group.columns
            else np.nan
        )

        total_spread = (
            float(
                group["spread"].sum()
            )
            if "spread"
            in group.columns
            else np.nan
        )

        kalshi_score = (
            score_model(
                group,
                kalshi_probabilities
            )
        )

        sharpened_score = (
            score_model(
                group,

                lambda x:
                    market_only_probabilities(
                        x,
                        FROZEN_ALPHA
                    )
            )
        )

        if (
            kalshi_score.empty
            or
            sharpened_score.empty
        ):
            continue

        kalshi_row = (
            kalshi_score.iloc[0]
        )

        sharpened_row = (
            sharpened_score.iloc[0]
        )

        daily_rows.append({

            "target_date":
                target_date,

            "raw_mid_sum":
                raw_mid_sum,

            "mid_sum_deviation":
                raw_mid_sum - 1.0,

            "absolute_deviation":
                abs(
                    raw_mid_sum - 1.0
                ),

            "normalized_sum":
                normalized_sum,

            "average_spread":
                average_spread,

            "total_spread":
                total_spread,

            "brier_improvement":
                (
                    kalshi_row["brier"]
                    -
                    sharpened_row["brier"]
                ),

            "log_improvement":
                (
                    kalshi_row["log_loss"]
                    -
                    sharpened_row["log_loss"]
                )
        })

    audit = pd.DataFrame(
        daily_rows
    )

    print()
    print("=" * 90)
    print("RAW MIDPOINT / NORMALIZATION AUDIT")
    print("=" * 90)

    print()
    print("MIDPOINT SUM SUMMARY")
    print("-" * 90)

    print(
        f"Days:                  "
        f"{len(audit)}"
    )

    print(
        f"Mean raw midpoint sum: "
        f"{audit['raw_mid_sum'].mean():.4f}"
    )

    print(
        f"Median:                "
        f"{audit['raw_mid_sum'].median():.4f}"
    )

    print(
        f"Std:                   "
        f"{audit['raw_mid_sum'].std(ddof=1):.4f}"
    )

    print(
        f"Minimum:               "
        f"{audit['raw_mid_sum'].min():.4f}"
    )

    print(
        f"Maximum:               "
        f"{audit['raw_mid_sum'].max():.4f}"
    )

    print(
        f"Mean absolute dev:     "
        f"{audit['absolute_deviation'].mean():.4f}"
    )

    print(
        f"Normalized sum mean:   "
        f"{audit['normalized_sum'].mean():.6f}"
    )

    # --------------------------------------------------------
    # Correlation with sharpening performance
    # --------------------------------------------------------

    print()
    print("ARTIFACT CHECK")
    print("-" * 90)

    deviation_brier_corr = (
        audit[
            "absolute_deviation"
        ].corr(
            audit[
                "brier_improvement"
            ]
        )
    )

    deviation_log_corr = (
        audit[
            "absolute_deviation"
        ].corr(
            audit[
                "log_improvement"
            ]
        )
    )

    print(
        "Correlation:"
    )

    print(
        f"|midpoint sum - 1| vs "
        f"Brier improvement: "
        f"{deviation_brier_corr:+.3f}"
    )

    print(
        f"|midpoint sum - 1| vs "
        f"log improvement:   "
        f"{deviation_log_corr:+.3f}"
    )

    if "spread" in dataframe.columns:

        spread_brier_corr = (
            audit[
                "average_spread"
            ].corr(
                audit[
                    "brier_improvement"
                ]
            )
        )

        spread_log_corr = (
            audit[
                "average_spread"
            ].corr(
                audit[
                    "log_improvement"
                ]
            )
        )

        print()

        print(
            f"Average spread vs "
            f"Brier improvement: "
            f"{spread_brier_corr:+.3f}"
        )

        print(
            f"Average spread vs "
            f"log improvement:   "
            f"{spread_log_corr:+.3f}"
        )

    # --------------------------------------------------------
    # Most extreme midpoint sums
    # --------------------------------------------------------

    extremes = (
        audit.sort_values(
            "absolute_deviation",
            ascending=False
        )
        .head(10)
    )

    print()
    print("10 MOST EXTREME RAW MIDPOINT SUMS")
    print("-" * 90)

    columns = [
        "target_date",
        "raw_mid_sum",
        "mid_sum_deviation",
        "brier_improvement",
        "log_improvement"
    ]

    if "spread" in dataframe.columns:

        columns.insert(
            3,
            "average_spread"
        )

    print(
        extremes[
            columns
        ].to_string(
            index=False,

            formatters={
                "raw_mid_sum":
                    "{:.4f}".format,

                "mid_sum_deviation":
                    "{:+.4f}".format,

                "average_spread":
                    "{:.4f}".format,

                "brier_improvement":
                    "{:+.4f}".format,

                "log_improvement":
                    "{:+.4f}".format
            }
        )
    )

    # --------------------------------------------------------
    # Near-1 vs far-from-1 comparison
    # --------------------------------------------------------

    audit[
        "deviation_group"
    ] = pd.cut(

        audit[
            "absolute_deviation"
        ],

        bins=[
            -np.inf,
            0.025,
            0.05,
            0.10,
            np.inf
        ],

        labels=[
            "<=2.5%",
            "2.5-5%",
            "5-10%",
            ">10%"
        ]
    )

    grouped = (
        audit.groupby(
            "deviation_group",
            observed=True
        )
        .agg(
            days=(
                "target_date",
                "count"
            ),

            mean_brier_improvement=(
                "brier_improvement",
                "mean"
            ),

            mean_log_improvement=(
                "log_improvement",
                "mean"
            )
        )
        .reset_index()
    )

    print()
    print("PERFORMANCE BY RAW MIDPOINT DEVIATION")
    print("-" * 90)

    print(
        grouped.to_string(
            index=False,

            formatters={
                "mean_brier_improvement":
                    "{:+.4f}".format,

                "mean_log_improvement":
                    "{:+.4f}".format
            }
        )
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
        "market_probability",
        "market_mid",
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

    run_alpha_stability(
        dataframe
    )

    run_midpoint_audit(
        dataframe
    )


if __name__ == "__main__":

    main()