import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset
)

from src.backtest.v2_market_only_control import (
    score_model,
    market_only_probabilities
)

from src.backtest.v2_market_only_walkforward import (
    get_training_dates
)


# ============================================================
# DIAGNOSTIC SETTINGS
# ============================================================

WIDE_ALPHA_VALUES = np.arange(
    0.50,
    5.0001,
    0.25
)

ORIGINAL_ALPHA_CAP = 3.0

EVALUATION_START = "2026-08-20"
EVALUATION_END = "2026-09-26"


# ============================================================
# SCORE ONE ALPHA
# ============================================================

def score_alpha(
    training_data,
    alpha
):
    scores = score_model(
        training_data,

        lambda group:
            market_only_probabilities(
                group,
                alpha
            )
    )

    if scores.empty:

        return {
            "alpha":
                alpha,

            "log_loss":
                np.nan,

            "brier":
                np.nan
        }

    return {
        "alpha":
            float(
                alpha
            ),

        "log_loss":
            float(
                scores[
                    "log_loss"
                ].mean()
            ),

        "brier":
            float(
                scores[
                    "brier"
                ].mean()
            )
    }


# ============================================================
# FULL ALPHA CURVE
# ============================================================

def build_alpha_curve(
    training_data
):
    rows = []

    for alpha in WIDE_ALPHA_VALUES:

        rows.append(
            score_alpha(
                training_data=
                    training_data,

                alpha=
                    alpha
            )
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DIAGNOSTIC ENGINE
# ============================================================

def run_boundary_diagnostic(
    dataframe
):
    all_dates = sorted(
        dataframe[
            "target_date"
        ].unique()
    )

    summary_rows = []

    detailed_curves = {}

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

        training_dates = (
            get_training_dates(
                all_dates=
                    all_dates,

                target_date=
                    target_date
            )
        )

        if not training_dates:
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

        curve = (
            build_alpha_curve(
                training_data
            )
        )

        best_row = (
            curve.sort_values(
                [
                    "log_loss",
                    "brier"
                ]
            )
            .iloc[
                0
            ]
        )

        best_alpha = float(
            best_row[
                "alpha"
            ]
        )

        original_cap_row = (
            curve.loc[
                np.isclose(
                    curve[
                        "alpha"
                    ],
                    ORIGINAL_ALPHA_CAP
                )
            ]
            .iloc[
                0
            ]
        )

        alpha_5_row = (
            curve.loc[
                np.isclose(
                    curve[
                        "alpha"
                    ],
                    5.0
                )
            ]
            .iloc[
                0
            ]
        )

        hit_old_boundary = (
            best_alpha
            >
            ORIGINAL_ALPHA_CAP
        )

        hit_new_boundary = (
            np.isclose(
                best_alpha,
                WIDE_ALPHA_VALUES[
                    -1
                ]
            )
        )

        summary_rows.append({

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

            "best_alpha_wide":
                best_alpha,

            "best_log_loss":
                float(
                    best_row[
                        "log_loss"
                    ]
                ),

            "log_loss_at_3":
                float(
                    original_cap_row[
                        "log_loss"
                    ]
                ),

            "log_loss_at_5":
                float(
                    alpha_5_row[
                        "log_loss"
                    ]
                ),

            "better_beyond_3":
                bool(
                    hit_old_boundary
                ),

            "hit_alpha_5":
                bool(
                    hit_new_boundary
                )
        })

        if (
            hit_old_boundary
            or
            np.isclose(
                best_alpha,
                ORIGINAL_ALPHA_CAP
            )
        ):

            detailed_curves[
                target_date
            ] = curve.copy()

    return (
        pd.DataFrame(
            summary_rows
        ),
        detailed_curves
    )


# ============================================================
# DISPLAY
# ============================================================

def print_summary(
    summary
):

    print()
    print("=" * 100)
    print(
        "MARKET-ONLY ALPHA BOUNDARY DIAGNOSTIC"
    )
    print("=" * 100)

    print(
        "This is diagnostic only."
    )

    print(
        "No trading parameters are being changed."
    )

    print()

    print(
        summary.to_string(
            index=False,

            formatters={
                "best_alpha_wide":
                    "{:.2f}".format,

                "best_log_loss":
                    "{:.4f}".format,

                "log_loss_at_3":
                    "{:.4f}".format,

                "log_loss_at_5":
                    "{:.4f}".format
            }
        )
    )

    print()
    print("=" * 100)
    print(
        "SUMMARY"
    )
    print("=" * 100)

    print(
        f"Decision dates:             "
        f"{len(summary)}"
    )

    print(
        f"Best alpha > 3.0:           "
        f"{summary['better_beyond_3'].sum()}"
    )

    print(
        f"Best alpha = 5.0 boundary:  "
        f"{summary['hit_alpha_5'].sum()}"
    )

    print(
        f"Mean best alpha:            "
        f"{summary['best_alpha_wide'].mean():.3f}"
    )

    print(
        f"Median best alpha:          "
        f"{summary['best_alpha_wide'].median():.3f}"
    )

    print(
        f"Minimum best alpha:         "
        f"{summary['best_alpha_wide'].min():.3f}"
    )

    print(
        f"Maximum best alpha:         "
        f"{summary['best_alpha_wide'].max():.3f}"
    )


def print_boundary_curves(
    detailed_curves
):

    if not detailed_curves:
        return

    print()
    print("=" * 100)
    print(
        "DETAILED CURVES FOR ORIGINAL BOUNDARY CASES"
    )
    print("=" * 100)

    key_alphas = [
        1.00,
        1.50,
        2.00,
        2.50,
        3.00,
        3.50,
        4.00,
        4.50,
        5.00
    ]

    for (
        target_date,
        curve
    ) in detailed_curves.items():

        filtered = (
            curve[
                curve[
                    "alpha"
                ].isin(
                    key_alphas
                )
            ]
            .copy()
        )

        print()
        print(
            f"TARGET DATE: "
            f"{target_date}"
        )

        print("-" * 100)

        print(
            filtered.to_string(
                index=False,

                formatters={
                    "alpha":
                        "{:.2f}".format,

                    "log_loss":
                        "{:.4f}".format,

                    "brier":
                        "{:.4f}".format
                }
            )
        )

        best_row = (
            curve.sort_values(
                [
                    "log_loss",
                    "brier"
                ]
            )
            .iloc[
                0
            ]
        )

        print()

        print(
            f"Wide-grid optimum: "
            f"alpha="
            f"{best_row['alpha']:.2f}, "
            f"log loss="
            f"{best_row['log_loss']:.4f}, "
            f"Brier="
            f"{best_row['brier']:.4f}"
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

    (
        summary,
        detailed_curves
    ) = run_boundary_diagnostic(
        dataframe
    )

    print_summary(
        summary
    )

    print_boundary_curves(
        detailed_curves
    )


if __name__ == "__main__":

    main()