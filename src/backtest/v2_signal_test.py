from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "data"
    / "research"
    / "weather_market_v2.csv"
)

MIN_TRAIN_DAYS = 20

BETA_GRID = np.arange(
    -1.00,
    1.01,
    0.05
)

EPSILON = 1e-6


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    data = pd.read_csv(
        DATA_PATH
    )

    data[
        "target_date"
    ] = pd.to_datetime(
        data[
            "target_date"
        ]
    )

    return data


# ============================================================
# MARKET-CONDITIONED WEATHER ADJUSTMENT
# ============================================================

def adjusted_probabilities(
    group,
    beta
):
    """
    beta = 0:
        pure Kalshi

    beta = 1:
        approximately pure v1 weather model

    beta > 0:
        move toward weather model

    beta < 0:
        move opposite weather-model disagreement

    The adjustment is performed in log-probability space,
    then normalized so the six contracts sum to 1.
    """

    market = (
        group[
            "market_probability"
        ]
        .astype(float)
        .clip(
            EPSILON,
            1.0
        )
        .to_numpy()
    )

    weather = (
        group[
            "v1_model_yes"
        ]
        .astype(float)
        .clip(
            EPSILON,
            1.0
        )
        .to_numpy()
    )

    log_market = np.log(
        market
    )

    disagreement = (
        np.log(
            weather
        )
        -
        log_market
    )

    scores = (
        log_market
        +
        beta
        * disagreement
    )

    # Numerically stable softmax
    scores = (
        scores
        -
        np.max(
            scores
        )
    )

    probabilities = np.exp(
        scores
    )

    probabilities = (
        probabilities
        /
        probabilities.sum()
    )

    return probabilities


# ============================================================
# LOG LOSS FOR ONE DATE
# ============================================================

def date_log_loss(
    group,
    beta
):

    probabilities = (
        adjusted_probabilities(
            group,
            beta
        )
    )

    outcomes = (
        group[
            "outcome"
        ]
        .astype(int)
        .to_numpy()
    )

    if outcomes.sum() != 1:

        raise RuntimeError(
            "Expected exactly one "
            "winning contract per date."
        )

    winner_index = (
        np.argmax(
            outcomes
        )
    )

    winner_probability = (
        probabilities[
            winner_index
        ]
    )

    return (
        -np.log(
            max(
                winner_probability,
                EPSILON
            )
        )
    )


# ============================================================
# FIT BETA USING ONLY PRIOR DATES
# ============================================================

def fit_beta(
    training_data
):

    best_beta = None
    best_loss = None

    for beta in BETA_GRID:

        losses = []

        for _, group in (
            training_data
            .groupby(
                "target_date"
            )
        ):

            losses.append(
                date_log_loss(
                    group,
                    beta
                )
            )

        average_loss = (
            np.mean(
                losses
            )
        )

        if (
            best_loss is None
            or
            average_loss
            <
            best_loss
        ):

            best_loss = (
                average_loss
            )

            best_beta = (
                beta
            )

    return (
        float(
            best_beta
        ),
        float(
            best_loss
        )
    )


# ============================================================
# SCORE ONE TEST DATE
# ============================================================

def score_date(
    group,
    beta
):

    adjusted = (
        adjusted_probabilities(
            group,
            beta
        )
    )

    market = (
        group[
            "market_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    actual = (
        group[
            "outcome"
        ]
        .astype(float)
        .to_numpy()
    )

    winner_index = (
        np.argmax(
            actual
        )
    )

    market_winner_probability = (
        market[
            winner_index
        ]
    )

    adjusted_winner_probability = (
        adjusted[
            winner_index
        ]
    )

    market_brier = np.sum(
        (
            market
            -
            actual
        ) ** 2
    )

    adjusted_brier = np.sum(
        (
            adjusted
            -
            actual
        ) ** 2
    )

    market_log_loss = (
        -np.log(
            max(
                market_winner_probability,
                EPSILON
            )
        )
    )

    adjusted_log_loss = (
        -np.log(
            max(
                adjusted_winner_probability,
                EPSILON
            )
        )
    )

    return {

        "market_brier":
            market_brier,

        "adjusted_brier":
            adjusted_brier,

        "market_log_loss":
            market_log_loss,

        "adjusted_log_loss":
            adjusted_log_loss,

        "market_winner_probability":
            market_winner_probability,

        "adjusted_winner_probability":
            adjusted_winner_probability,

        "adjustment_better":
            adjusted_brier
            <
            market_brier
    }


# ============================================================
# WALK-FORWARD TEST
# ============================================================

def run_walk_forward_test():

    data = (
        load_dataset()
    )

    dates = sorted(
        data[
            "target_date"
        ].unique()
    )

    results = []

    for index in range(
        MIN_TRAIN_DAYS,
        len(dates)
    ):

        test_date = (
            dates[
                index
            ]
        )

        training_dates = (
            dates[
                :index
            ]
        )

        training_data = data[
            data[
                "target_date"
            ].isin(
                training_dates
            )
        ]

        test_data = data[
            data[
                "target_date"
            ]
            ==
            test_date
        ].copy()

        if len(test_data) != 6:
            continue

        beta, training_loss = (
            fit_beta(
                training_data
            )
        )

        metrics = (
            score_date(
                test_data,
                beta
            )
        )

        results.append({

            "target_date":
                test_date,

            "beta":
                beta,

            "training_log_loss":
                training_loss,

            **metrics
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# DISPLAY
# ============================================================

def print_results(
    results
):

    print()
    print("=" * 100)
    print(
        "V2 WALK-FORWARD MARKET-CONDITIONED SIGNAL TEST"
    )
    print("=" * 100)

    print(
        f"Training days required: "
        f"{MIN_TRAIN_DAYS}"
    )

    print(
        f"Out-of-sample days:     "
        f"{len(results)}"
    )

    print()

    print(
        "BRIER SCORE"
    )
    print("-" * 50)

    print(
        f"Kalshi:   "
        f"{results['market_brier'].mean():.4f}"
    )

    print(
        f"Adjusted: "
        f"{results['adjusted_brier'].mean():.4f}"
    )

    print()

    print(
        "LOG LOSS"
    )
    print("-" * 50)

    print(
        f"Kalshi:   "
        f"{results['market_log_loss'].mean():.4f}"
    )

    print(
        f"Adjusted: "
        f"{results['adjusted_log_loss'].mean():.4f}"
    )

    print()

    print(
        "ACTUAL WINNER PROBABILITY"
    )
    print("-" * 50)

    print(
        f"Kalshi:   "
        f"{results['market_winner_probability'].mean():.2%}"
    )

    print(
        f"Adjusted: "
        f"{results['adjusted_winner_probability'].mean():.2%}"
    )

    print()

    print(
        f"Adjustment beats Kalshi: "
        f"{results['adjustment_better'].mean():.2%} "
        f"of OOS days"
    )

    print()

    print(
        "BETA"
    )
    print("-" * 50)

    print(
        f"Mean:   "
        f"{results['beta'].mean():+.3f}"
    )

    print(
        f"Median: "
        f"{results['beta'].median():+.3f}"
    )

    print(
        f"Min:    "
        f"{results['beta'].min():+.3f}"
    )

    print(
        f"Max:    "
        f"{results['beta'].max():+.3f}"
    )

    print()

    print(
        "DAILY RESULTS"
    )
    print("-" * 100)

    display = (
        results.copy()
    )

    display[
        "target_date"
    ] = (
        display[
            "target_date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    print(
        display[
            [
                "target_date",
                "beta",
                "market_brier",
                "adjusted_brier",
                "market_winner_probability",
                "adjusted_winner_probability",
                "adjustment_better"
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":

    results = (
        run_walk_forward_test()
    )

    print_results(
        results
    )