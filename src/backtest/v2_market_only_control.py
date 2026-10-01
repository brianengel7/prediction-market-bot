import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset
)


# ============================================================
# FROZEN TEST DESIGN
# ============================================================

TRAIN_DATE_COUNT = 20

HOLDOUT_START = "2026-09-01"
HOLDOUT_END = "2026-09-30"

NBM_BETA = -0.40

EPSILON = 1e-6

# Market-only sharpening exponent:
#
# q_i ∝ p_i ** alpha
#
# alpha = 1.0  -> unchanged Kalshi
# alpha > 1.0  -> sharper / more confident Kalshi
# alpha < 1.0  -> flatter / less confident Kalshi
#
ALPHA_VALUES = np.arange(
    0.25,
    3.0001,
    0.025
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_probabilities(
    probabilities
):
    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        None
    )

    total = probabilities.sum()

    if total <= 0:
        raise ValueError(
            "Probability mass must be positive."
        )

    return (
        probabilities
        /
        total
    )


# ============================================================
# MODELS
# ============================================================

def kalshi_probabilities(
    group
):
    return normalize_probabilities(
        group[
            "market_probability"
        ].to_numpy(
            dtype=float
        )
    )


def market_only_probabilities(
    group,
    alpha
):
    market = (
        kalshi_probabilities(
            group
        )
    )

    sharpened = (
        market
        ** float(
            alpha
        )
    )

    return normalize_probabilities(
        sharpened
    )


def nbm_adjusted_probabilities(
    group,
    beta=NBM_BETA
):
    """
    Frozen NBM transformation:

        log(q_i)
          =
        log(market_i)
        +
        beta * (
            log(NBM_i)
            -
            log(market_i)
        )

    With beta = -0.40.
    """

    market = (
        kalshi_probabilities(
            group
        )
    )

    nbm = normalize_probabilities(
        group[
            "NBM_yes"
        ].to_numpy(
            dtype=float
        )
    )

    market = np.clip(
        market,
        EPSILON,
        1.0
    )

    nbm = np.clip(
        nbm,
        EPSILON,
        1.0
    )

    log_scores = (
        np.log(
            market
        )
        +
        float(
            beta
        )
        *
        (
            np.log(
                nbm
            )
            -
            np.log(
                market
            )
        )
    )

    # Numerical stability.
    log_scores = (
        log_scores
        -
        np.max(
            log_scores
        )
    )

    scores = np.exp(
        log_scores
    )

    return normalize_probabilities(
        scores
    )


# ============================================================
# DAILY SCORING
# ============================================================

def score_model(
    dataframe,
    probability_function
):
    daily_rows = []

    for (
        target_date,
        group
    ) in dataframe.groupby(
        "target_date"
    ):

        group = group.copy()

        if len(
            group
        ) != 6:
            continue

        outcomes = (
            group[
                "outcome"
            ].to_numpy(
                dtype=float
            )
        )

        if not np.isclose(
            outcomes.sum(),
            1.0
        ):
            continue

        probabilities = (
            probability_function(
                group
            )
        )

        winner_index = int(
            np.argmax(
                outcomes
            )
        )

        winner_probability = float(
            probabilities[
                winner_index
            ]
        )

        brier = float(
            np.sum(
                (
                    probabilities
                    -
                    outcomes
                )
                ** 2
            )
        )

        log_loss = float(
            -np.log(
                max(
                    winner_probability,
                    EPSILON
                )
            )
        )

        daily_rows.append({
            "target_date":
                target_date,

            "brier":
                brier,

            "log_loss":
                log_loss,

            "winner_probability":
                winner_probability
        })

    return pd.DataFrame(
        daily_rows
    )


# ============================================================
# FIT MARKET-ONLY ALPHA
# ============================================================

def fit_market_alpha(
    training_data,
    alpha_values=None
):
    if alpha_values is None:
        alpha_values = ALPHA_VALUES
    results = []

    for alpha in alpha_values:

        scores = (
            score_model(
                training_data,

                lambda group:
                    market_only_probabilities(
                        group,
                        alpha
                    )
            )
        )

        if scores.empty:
            continue

        results.append({
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
        })

    results = pd.DataFrame(
        results
    )

    if results.empty:
        raise RuntimeError(
            "No valid alpha results."
        )

    best = (
        results.sort_values(
            [
                "log_loss",
                "brier"
            ]
        )
        .iloc[
            0
        ]
    )

    return (
        float(
            best[
                "alpha"
            ]
        ),
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_scores(
    name,
    scores
):
    return {
        "model":
            name,

        "dates":
            len(
                scores
            ),

        "brier":
            scores[
                "brier"
            ].mean(),

        "log_loss":
            scores[
                "log_loss"
            ].mean(),

        "winner_probability":
            scores[
                "winner_probability"
            ].mean()
    }


def compare_models(
    kalshi_scores,
    market_only_scores,
    nbm_scores
):
    comparison = (
        kalshi_scores[
            [
                "target_date",
                "brier",
                "log_loss"
            ]
        ]
        .rename(
            columns={
                "brier":
                    "kalshi_brier",

                "log_loss":
                    "kalshi_log"
            }
        )
        .merge(
            market_only_scores[
                [
                    "target_date",
                    "brier",
                    "log_loss"
                ]
            ].rename(
                columns={
                    "brier":
                        "market_only_brier",

                    "log_loss":
                        "market_only_log"
                }
            ),

            on=
                "target_date"
        )
        .merge(
            nbm_scores[
                [
                    "target_date",
                    "brier",
                    "log_loss"
                ]
            ].rename(
                columns={
                    "brier":
                        "nbm_brier",

                    "log_loss":
                        "nbm_log"
                }
            ),

            on=
                "target_date"
        )
    )

    return comparison


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
        "NBM_yes",
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

    # --------------------------------------------------------
    # Same original pre-holdout chronology.
    # --------------------------------------------------------

    pre_holdout_dates = sorted(
        dataframe.loc[
            dataframe[
                "target_date"
            ]
            <
            HOLDOUT_START,

            "target_date"
        ].unique()
    )

    if (
        len(
            pre_holdout_dates
        )
        <
        TRAIN_DATE_COUNT
    ):

        raise RuntimeError(
            "Not enough pre-holdout dates."
        )

    training_dates = (
        pre_holdout_dates[
            :
            TRAIN_DATE_COUNT
        ]
    )

    validation_dates = (
        pre_holdout_dates[
            TRAIN_DATE_COUNT:
        ]
    )

    holdout_dates = sorted(
        dataframe.loc[
            (
                dataframe[
                    "target_date"
                ]
                >=
                HOLDOUT_START
            )
            &
            (
                dataframe[
                    "target_date"
                ]
                <=
                HOLDOUT_END
            ),

            "target_date"
        ].unique()
    )

    training_data = dataframe[
        dataframe[
            "target_date"
        ].isin(
            training_dates
        )
    ].copy()

    validation_data = dataframe[
        dataframe[
            "target_date"
        ].isin(
            validation_dates
        )
    ].copy()

    holdout_data = dataframe[
        dataframe[
            "target_date"
        ].isin(
            holdout_dates
        )
    ].copy()

    # --------------------------------------------------------
    # Fit market-only exponent ONCE using training dates.
    # --------------------------------------------------------

    (
        frozen_alpha,
        alpha_results
    ) = fit_market_alpha(
        training_data
    )

    print()
    print("=" * 80)
    print(
        "MARKET-ONLY SHARPENING CONTROL"
    )
    print("=" * 80)

    print(
        f"Training dates:   "
        f"{len(training_dates)}"
    )

    print(
        f"Validation dates: "
        f"{len(validation_dates)}"
    )

    print(
        f"Holdout dates:    "
        f"{len(holdout_dates)}"
    )

    print()

    print(
        f"Frozen market-only alpha: "
        f"{frozen_alpha:.3f}"
    )

    print(
        f"Frozen NBM beta:          "
        f"{NBM_BETA:+.2f}"
    )

    if (
        np.isclose(
            frozen_alpha,
            ALPHA_VALUES[
                0
            ]
        )
        or
        np.isclose(
            frozen_alpha,
            ALPHA_VALUES[
                -1
            ]
        )
    ):

        print(
            "WARNING: optimum alpha "
            "hit search-grid boundary."
        )

    # --------------------------------------------------------
    # Show training fit.
    # --------------------------------------------------------

    best_alpha_row = (
        alpha_results[
            np.isclose(
                alpha_results[
                    "alpha"
                ],
                frozen_alpha
            )
        ]
        .iloc[
            0
        ]
    )

    print()
    print(
        "TRAINING FIT"
    )
    print("-" * 80)

    print(
        f"Alpha:     "
        f"{frozen_alpha:.3f}"
    )

    print(
        f"Log loss:  "
        f"{best_alpha_row['log_loss']:.4f}"
    )

    print(
        f"Brier:     "
        f"{best_alpha_row['brier']:.4f}"
    )

    # --------------------------------------------------------
    # Evaluate one period.
    # --------------------------------------------------------

    def evaluate_period(
        label,
        period_data
    ):

        if period_data.empty:
            return

        kalshi_scores = (
            score_model(
                period_data,
                kalshi_probabilities
            )
        )

        market_only_scores = (
            score_model(
                period_data,

                lambda group:
                    market_only_probabilities(
                        group,
                        frozen_alpha
                    )
            )
        )

        nbm_scores = (
            score_model(
                period_data,

                lambda group:
                    nbm_adjusted_probabilities(
                        group,
                        NBM_BETA
                    )
            )
        )

        summaries = pd.DataFrame([
            summarize_scores(
                "Kalshi",
                kalshi_scores
            ),

            summarize_scores(
                "Market-only",
                market_only_scores
            ),

            summarize_scores(
                "NBM adjusted",
                nbm_scores
            )
        ])

        comparison = (
            compare_models(
                kalshi_scores,
                market_only_scores,
                nbm_scores
            )
        )

        print()
        print("=" * 80)
        print(label)
        print("=" * 80)

        print(
            summaries.to_string(
                index=False,
                formatters={
                    "brier":
                        "{:.4f}".format,

                    "log_loss":
                        "{:.4f}".format,

                    "winner_probability":
                        "{:.2%}".format
                }
            )
        )

        market_brier_improvement = (
            comparison[
                "kalshi_brier"
            ].mean()
            -
            comparison[
                "market_only_brier"
            ].mean()
        )

        nbm_brier_improvement = (
            comparison[
                "kalshi_brier"
            ].mean()
            -
            comparison[
                "nbm_brier"
            ].mean()
        )

        market_log_improvement = (
            comparison[
                "kalshi_log"
            ].mean()
            -
            comparison[
                "market_only_log"
            ].mean()
        )

        nbm_log_improvement = (
            comparison[
                "kalshi_log"
            ].mean()
            -
            comparison[
                "nbm_log"
            ].mean()
        )

        market_better_days = int(
            (
                comparison[
                    "market_only_brier"
                ]
                <
                comparison[
                    "kalshi_brier"
                ]
            ).sum()
        )

        nbm_better_days = int(
            (
                comparison[
                    "nbm_brier"
                ]
                <
                comparison[
                    "kalshi_brier"
                ]
            ).sum()
        )

        nbm_beats_market_only = int(
            (
                comparison[
                    "nbm_brier"
                ]
                <
                comparison[
                    "market_only_brier"
                ]
            ).sum()
        )

        total_days = len(
            comparison
        )

        print()
        print(
            "IMPROVEMENT VS KALSHI"
        )
        print("-" * 80)

        print(
            f"Market-only Brier improvement: "
            f"{market_brier_improvement:+.4f}"
        )

        print(
            f"NBM Brier improvement:         "
            f"{nbm_brier_improvement:+.4f}"
        )

        print()

        print(
            f"Market-only log improvement:   "
            f"{market_log_improvement:+.4f}"
        )

        print(
            f"NBM log improvement:           "
            f"{nbm_log_improvement:+.4f}"
        )

        print()
        print(
            "DAILY COMPARISON"
        )
        print("-" * 80)

        print(
            f"Market-only beats Kalshi: "
            f"{market_better_days}/"
            f"{total_days}"
        )

        print(
            f"NBM beats Kalshi:         "
            f"{nbm_better_days}/"
            f"{total_days}"
        )

        print(
            f"NBM beats market-only:    "
            f"{nbm_beats_market_only}/"
            f"{total_days}"
        )

    # --------------------------------------------------------
    # Validation first.
    # --------------------------------------------------------

    evaluate_period(
        "POST-TRAINING VALIDATION",
        validation_data
    )

    # --------------------------------------------------------
    # Untouched holdout last.
    # --------------------------------------------------------

    evaluate_period(
        "SEP 11-26 HOLDOUT",
        holdout_data
    )


if __name__ == "__main__":

    main()