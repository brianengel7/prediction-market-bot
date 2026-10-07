import numpy as np
import pandas as pd

from src.database.db import (
    get_connection,
)


HORIZONS = (
    14,
    10,
    5,
    3,
    2,
    1,
)

TRAINING_LOOKBACK_DAYS = 60

VALIDATION_DAYS = 60

TEST_DAYS = 60

MINIMUM_TRAINING_ROWS = 1000

ALPHA_VALUES = np.arange(
    0.50,
    2.0001,
    0.025,
)

EPSILON = 1e-12


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    with get_connection() as connection:

        rows = (
            connection.execute(
                """
                SELECT
                    mm.ticker,
                    mm.minutes_remaining,
                    mm.midpoint,
                    mm.outcome,
                    m.close_time

                FROM crypto_market_minutes mm

                JOIN crypto_markets m
                    ON m.ticker = mm.ticker

                WHERE
                    mm.actionable = 1
                    AND mm.midpoint IS NOT NULL
                    AND mm.outcome IS NOT NULL

                ORDER BY
                    m.close_time,
                    mm.ticker
                """
            )
            .fetchall()
        )

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "minutes_remaining",
            "midpoint",
            "outcome",
            "close_time",
        ],
    )

    dataframe[
        "minutes_remaining"
    ] = pd.to_numeric(
        dataframe[
            "minutes_remaining"
        ],
        errors="raise",
    )

    dataframe[
        "midpoint"
    ] = pd.to_numeric(
        dataframe[
            "midpoint"
        ],
        errors="raise",
    )

    dataframe[
        "outcome"
    ] = pd.to_numeric(
        dataframe[
            "outcome"
        ],
        errors="raise",
    ).astype(int)

    dataframe[
        "close_time"
    ] = pd.to_datetime(
        dataframe[
            "close_time"
        ],
        utc=True,
        errors="raise",
        format="mixed",
    )

    dataframe[
        "date"
    ] = (
        dataframe[
            "close_time"
        ]
        .dt.normalize()
    )

    dataframe = (
        dataframe[
            dataframe[
                "minutes_remaining"
            ].isin(
                HORIZONS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return dataframe


# ============================================================
# ALPHA TRANSFORM
# ============================================================

def apply_alpha(
    probabilities,
    alpha,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0 - EPSILON,
    )

    logits = np.log(
        probabilities
        /
        (
            1.0
            -
            probabilities
        )
    )

    transformed_logits = (
        float(alpha)
        *
        logits
    )

    transformed_logits = np.clip(
        transformed_logits,
        -50.0,
        50.0,
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -transformed_logits
            )
        )
    )


# ============================================================
# SCORING
# ============================================================

def log_loss(
    probabilities,
    outcomes,
):

    probabilities = np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        EPSILON,
        1.0 - EPSILON,
    )

    outcomes = np.asarray(
        outcomes,
        dtype=float,
    )

    return float(
        -np.mean(
            outcomes
            *
            np.log(
                probabilities
            )
            +
            (
                1.0
                -
                outcomes
            )
            *
            np.log(
                1.0
                -
                probabilities
            )
        )
    )


def brier_score(
    probabilities,
    outcomes,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    outcomes = np.asarray(
        outcomes,
        dtype=float,
    )

    return float(
        np.mean(
            (
                probabilities
                -
                outcomes
            )
            ** 2
        )
    )


# ============================================================
# FIT ALPHA
# ============================================================

def fit_alpha(
    training_data,
):

    probabilities = np.clip(
        training_data[
            "midpoint"
        ].to_numpy(
            dtype=float
        ),
        EPSILON,
        1.0 - EPSILON,
    )

    outcomes = (
        training_data[
            "outcome"
        ].to_numpy(
            dtype=float
        )
    )

    logits = np.log(
        probabilities
        /
        (
            1.0
            -
            probabilities
        )
    )

    # rows × candidate alphas
    transformed_logits = (
        logits[
            :,
            None
        ]
        *
        ALPHA_VALUES[
            None,
            :
        ]
    )

    transformed_logits = np.clip(
        transformed_logits,
        -50.0,
        50.0,
    )

    probabilities_grid = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -transformed_logits
            )
        )
    )

    probabilities_grid = np.clip(
        probabilities_grid,
        EPSILON,
        1.0 - EPSILON,
    )

    outcomes_grid = (
        outcomes[
            :,
            None
        ]
    )

    losses = (
        -np.mean(
            outcomes_grid
            *
            np.log(
                probabilities_grid
            )
            +
            (
                1.0
                -
                outcomes_grid
            )
            *
            np.log(
                1.0
                -
                probabilities_grid
            ),
            axis=0,
        )
    )

    best_index = int(
        np.argmin(
            losses
        )
    )

    return float(
        ALPHA_VALUES[
            best_index
        ]
    )


# ============================================================
# SPLIT DATES
# ============================================================

def get_periods(
    dataframe,
):

    latest_date = (
        dataframe[
            "date"
        ].max()
    )

    test_start = (
        latest_date
        -
        pd.Timedelta(
            days=
                TEST_DAYS
                -
                1
        )
    )

    validation_end = (
        test_start
        -
        pd.Timedelta(
            days=1
        )
    )

    validation_start = (
        validation_end
        -
        pd.Timedelta(
            days=
                VALIDATION_DAYS
                -
                1
        )
    )

    return {
        "latest_date":
            latest_date,

        "validation_start":
            validation_start,

        "validation_end":
            validation_end,

        "test_start":
            test_start,

        "test_end":
            latest_date,
    }


# ============================================================
# ONE HORIZON
# ============================================================

def run_horizon(
    dataframe,
    horizon,
    periods,
):

    horizon_data = (
        dataframe[
            dataframe[
                "minutes_remaining"
            ]
            ==
            float(
                horizon
            )
        ]
        .copy()
    )

    validation_days = (
        horizon_data[
            (
                horizon_data[
                    "date"
                ]
                >=
                periods[
                    "validation_start"
                ]
            )
            &
            (
                horizon_data[
                    "date"
                ]
                <=
                periods[
                    "validation_end"
                ]
            )
        ][
            "date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    rows = []

    all_raw_probabilities = []

    all_adjusted_probabilities = []

    all_outcomes = []

    for decision_date in validation_days:

        training_start = (
            decision_date
            -
            pd.Timedelta(
                days=
                    TRAINING_LOOKBACK_DAYS
            )
        )

        training_end = (
            decision_date
            -
            pd.Timedelta(
                days=1
            )
        )

        training_data = (
            horizon_data[
                (
                    horizon_data[
                        "date"
                    ]
                    >=
                    training_start
                )
                &
                (
                    horizon_data[
                        "date"
                    ]
                    <=
                    training_end
                )
            ]
            .copy()
        )

        decision_data = (
            horizon_data[
                horizon_data[
                    "date"
                ]
                ==
                decision_date
            ]
            .copy()
        )

        if (
            len(
                training_data
            )
            <
            MINIMUM_TRAINING_ROWS
            or
            decision_data.empty
        ):

            continue

        alpha = (
            fit_alpha(
                training_data
            )
        )

        raw_probabilities = (
            decision_data[
                "midpoint"
            ].to_numpy(
                dtype=float
            )
        )

        outcomes = (
            decision_data[
                "outcome"
            ].to_numpy(
                dtype=float
            )
        )

        adjusted_probabilities = (
            apply_alpha(
                raw_probabilities,
                alpha,
            )
        )

        raw_brier = (
            brier_score(
                raw_probabilities,
                outcomes,
            )
        )

        adjusted_brier = (
            brier_score(
                adjusted_probabilities,
                outcomes,
            )
        )

        raw_log = (
            log_loss(
                raw_probabilities,
                outcomes,
            )
        )

        adjusted_log = (
            log_loss(
                adjusted_probabilities,
                outcomes,
            )
        )

        rows.append(
            {
                "date":
                    decision_date,

                "training_rows":
                    len(
                        training_data
                    ),

                "evaluation_rows":
                    len(
                        decision_data
                    ),

                "alpha":
                    alpha,

                "raw_brier":
                    raw_brier,

                "adjusted_brier":
                    adjusted_brier,

                "raw_log_loss":
                    raw_log,

                "adjusted_log_loss":
                    adjusted_log,
            }
        )

        all_raw_probabilities.extend(
            raw_probabilities
        )

        all_adjusted_probabilities.extend(
            adjusted_probabilities
        )

        all_outcomes.extend(
            outcomes
        )

    results = pd.DataFrame(
        rows
    )

    if results.empty:

        return None

    raw_brier = (
        brier_score(
            all_raw_probabilities,
            all_outcomes,
        )
    )

    adjusted_brier = (
        brier_score(
            all_adjusted_probabilities,
            all_outcomes,
        )
    )

    raw_log = (
        log_loss(
            all_raw_probabilities,
            all_outcomes,
        )
    )

    adjusted_log = (
        log_loss(
            all_adjusted_probabilities,
            all_outcomes,
        )
    )

    brier_improvement = (
        (
            raw_brier
            -
            adjusted_brier
        )
        /
        raw_brier
    )

    log_improvement = (
        (
            raw_log
            -
            adjusted_log
        )
        /
        raw_log
    )

    brier_days_improved = int(
        (
            results[
                "adjusted_brier"
            ]
            <
            results[
                "raw_brier"
            ]
        ).sum()
    )

    log_days_improved = int(
        (
            results[
                "adjusted_log_loss"
            ]
            <
            results[
                "raw_log_loss"
            ]
        ).sum()
    )

    both_days_improved = int(
        (
            (
                results[
                    "adjusted_brier"
                ]
                <
                results[
                    "raw_brier"
                ]
            )
            &
            (
                results[
                    "adjusted_log_loss"
                ]
                <
                results[
                    "raw_log_loss"
                ]
            )
        ).sum()
    )

    return {
        "horizon":
            horizon,

        "days":
            len(
                results
            ),

        "rows":
            len(
                all_outcomes
            ),

        "mean_alpha":
            float(
                results[
                    "alpha"
                ].mean()
            ),

        "median_alpha":
            float(
                results[
                    "alpha"
                ].median()
            ),

        "minimum_alpha":
            float(
                results[
                    "alpha"
                ].min()
            ),

        "maximum_alpha":
            float(
                results[
                    "alpha"
                ].max()
            ),

        "alpha_gt_one_days":
            int(
                (
                    results[
                        "alpha"
                    ]
                    >
                    1.0
                ).sum()
            ),

        "raw_brier":
            raw_brier,

        "adjusted_brier":
            adjusted_brier,

        "brier_improvement":
            brier_improvement,

        "raw_log_loss":
            raw_log,

        "adjusted_log_loss":
            adjusted_log,

        "log_improvement":
            log_improvement,

        "brier_days_improved":
            brier_days_improved,

        "log_days_improved":
            log_days_improved,

        "both_days_improved":
            both_days_improved,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = load_data()

    periods = (
        get_periods(
            dataframe
        )
    )

    print()
    print("=" * 100)
    print(
        "CRYPTO ALPHA WALK-FORWARD"
    )
    print("=" * 100)

    print(
        f"Validation: "
        f"{periods['validation_start'].date()} "
        f"-> "
        f"{periods['validation_end'].date()}"
    )

    print(
        f"RESERVED TEST: "
        f"{periods['test_start'].date()} "
        f"-> "
        f"{periods['test_end'].date()}"
    )

    print(
        f"Training lookback: "
        f"{TRAINING_LOOKBACK_DAYS} days"
    )

    print()
    print(
        "IMPORTANT: reserved test period is "
        "NOT evaluated by this script."
    )

    summaries = []

    for horizon in HORIZONS:

        print()
        print(
            f"Running "
            f"{horizon}-minute horizon..."
        )

        result = run_horizon(
            dataframe=
                dataframe,

            horizon=
                horizon,

            periods=
                periods,
        )

        if result is not None:

            summaries.append(
                result
            )

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 120)
    print(
        "OUT-OF-SAMPLE VALIDATION SUMMARY"
    )
    print("=" * 120)

    display_columns = [
        "horizon",
        "days",
        "rows",
        "mean_alpha",
        "median_alpha",
        "minimum_alpha",
        "maximum_alpha",
        "alpha_gt_one_days",
        "raw_brier",
        "adjusted_brier",
        "brier_improvement",
        "raw_log_loss",
        "adjusted_log_loss",
        "log_improvement",
        "both_days_improved",
    ]

    print(
        summary[
            display_columns
        ]
        .to_string(
            index=False,
            formatters={
                "mean_alpha":
                    lambda x:
                        f"{x:.3f}",

                "median_alpha":
                    lambda x:
                        f"{x:.3f}",

                "minimum_alpha":
                    lambda x:
                        f"{x:.3f}",

                "maximum_alpha":
                    lambda x:
                        f"{x:.3f}",

                "raw_brier":
                    lambda x:
                        f"{x:.6f}",

                "adjusted_brier":
                    lambda x:
                        f"{x:.6f}",

                "brier_improvement":
                    lambda x:
                        f"{x:+.2%}",

                "raw_log_loss":
                    lambda x:
                        f"{x:.6f}",

                "adjusted_log_loss":
                    lambda x:
                        f"{x:.6f}",

                "log_improvement":
                    lambda x:
                        f"{x:+.2%}",
            },
        )
    )


if __name__ == "__main__":

    main()