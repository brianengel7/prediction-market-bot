from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.analysis import (
    load_backtest_data
)

from src.weather.fair_distribution import (
    probability_for_market
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "data"
    / "research"
    / "weather_market_v2.csv"
)

SOURCES = [
    "GEFS",
    "GFS_MOS",
    "HRRR",
    "NBM"
]

MIN_TRAIN_DAYS = 20

BETA_GRID = np.arange(
    -1.00,
    1.01,
    0.05
)

EPSILON = 1e-6


# ============================================================
# BUILD INDIVIDUAL MODEL PROBABILITIES
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

    historical = (
        load_backtest_data()
    )

    historical = historical.copy()

    historical[
        "target_date"
    ] = pd.to_datetime(
        historical[
            "target_date"
        ]
    )

    # --------------------------------------------------------
    # Build one probability distribution per weather model.
    #
    # IMPORTANT:
    # residual standard deviation uses ONLY dates before the
    # date being predicted.
    # --------------------------------------------------------

    for source in SOURCES:

        probability_column = (
            f"{source}_yes"
        )

        data[
            probability_column
        ] = np.nan

        source_history = historical[
            historical[
                "source"
            ] == source
        ].copy()

        for target_date in sorted(
            data[
                "target_date"
            ].unique()
        ):

            prior = source_history[
                source_history[
                    "target_date"
                ]
                <
                target_date
            ]

            if len(prior) < 15:
                continue

            residual_std = float(
                prior[
                    "error"
                ].std(
                    ddof=1
                )
            )

            if (
                pd.isna(
                    residual_std
                )
                or
                residual_std <= 0
            ):
                continue

            date_mask = (
                data[
                    "target_date"
                ]
                ==
                target_date
            )

            date_rows = data[
                date_mask
            ]

            probabilities = []

            for _, row in date_rows.iterrows():

                corrected_forecast = float(
                    row[
                        f"{source}_corrected"
                    ]
                )

                probability = (
                    probability_for_market(

                        strike_type=
                            row[
                                "strike_type"
                            ],

                        floor_strike=
                            row[
                                "floor_strike"
                            ],

                        cap_strike=
                            row[
                                "cap_strike"
                            ],

                        point_forecast=
                            corrected_forecast,

                        residual_mean=
                            0.0,

                        residual_std=
                            residual_std
                    )
                )

                probabilities.append(
                    probability
                )

            data.loc[
                date_mask,
                probability_column
            ] = probabilities

    return data


# ============================================================
# MARKET-CONDITIONED ADJUSTMENT
# ============================================================

def adjusted_probabilities(
    group,
    signal_column,
    beta
):

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

    signal = (
        group[
            signal_column
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
            signal
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

    return (
        probabilities
        /
        probabilities.sum()
    )


# ============================================================
# DATE LOSS
# ============================================================

def date_log_loss(
    group,
    signal_column,
    beta
):

    probabilities = (
        adjusted_probabilities(
            group=
                group,
            signal_column=
                signal_column,
            beta=
                beta
        )
    )

    actual = (
        group[
            "outcome"
        ]
        .astype(int)
        .to_numpy()
    )

    winner_index = int(
        np.argmax(
            actual
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
# FIT BETA
# ============================================================

def fit_beta(
    training_data,
    signal_column
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

            if len(group) != 6:
                continue

            losses.append(
                date_log_loss(
                    group=
                        group,
                    signal_column=
                        signal_column,
                    beta=
                        beta
                )
            )

        if not losses:
            continue

        loss = float(
            np.mean(
                losses
            )
        )

        if (
            best_loss is None
            or
            loss < best_loss
        ):

            best_loss = loss
            best_beta = beta

    return (
        float(
            best_beta
        ),
        float(
            best_loss
        )
    )


# ============================================================
# SCORE DATE
# ============================================================

def score_date(
    group,
    signal_column,
    beta
):

    adjusted = (
        adjusted_probabilities(
            group=
                group,
            signal_column=
                signal_column,
            beta=
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

    winner_index = int(
        np.argmax(
            actual
        )
    )

    market_brier = float(
        np.sum(
            (
                market
                -
                actual
            ) ** 2
        )
    )

    adjusted_brier = float(
        np.sum(
            (
                adjusted
                -
                actual
            ) ** 2
        )
    )

    market_winner = float(
        market[
            winner_index
        ]
    )

    adjusted_winner = float(
        adjusted[
            winner_index
        ]
    )

    market_log = (
        -np.log(
            max(
                market_winner,
                EPSILON
            )
        )
    )

    adjusted_log = (
        -np.log(
            max(
                adjusted_winner,
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
            market_log,

        "adjusted_log_loss":
            adjusted_log,

        "market_winner_probability":
            market_winner,

        "adjusted_winner_probability":
            adjusted_winner,

        "better":
            adjusted_brier
            <
            market_brier
    }


# ============================================================
# WALK-FORWARD SOURCE TEST
# ============================================================

def test_source(
    data,
    source
):

    signal_column = (
        f"{source}_yes"
    )

    source_data = (
        data.dropna(
            subset=[
                signal_column
            ]
        )
        .copy()
    )

    dates = sorted(
        source_data[
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

        training = source_data[
            source_data[
                "target_date"
            ].isin(
                training_dates
            )
        ]

        test = source_data[
            source_data[
                "target_date"
            ]
            ==
            test_date
        ]

        if len(test) != 6:
            continue

        beta, training_loss = (
            fit_beta(
                training_data=
                    training,
                signal_column=
                    signal_column
            )
        )

        metrics = (
            score_date(
                group=
                    test,
                signal_column=
                    signal_column,
                beta=
                    beta
            )
        )

        results.append({

            "target_date":
                test_date,

            "source":
                source,

            "beta":
                beta,

            "training_loss":
                training_loss,

            **metrics
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_source(
    results
):

    return {

        "source":
            results[
                "source"
            ].iloc[0],

        "days":
            len(
                results
            ),

        "market_brier":
            results[
                "market_brier"
            ].mean(),

        "adjusted_brier":
            results[
                "adjusted_brier"
            ].mean(),

        "market_log":
            results[
                "market_log_loss"
            ].mean(),

        "adjusted_log":
            results[
                "adjusted_log_loss"
            ].mean(),

        "market_winner":
            results[
                "market_winner_probability"
            ].mean(),

        "adjusted_winner":
            results[
                "adjusted_winner_probability"
            ].mean(),

        "days_better":
            results[
                "better"
            ].mean(),

        "mean_beta":
            results[
                "beta"
            ].mean(),

        "median_beta":
            results[
                "beta"
            ].median()
    }


# ============================================================
# MAIN
# ============================================================

def run_source_tests():

    data = (
        load_dataset()
    )

    summaries = []

    all_results = []

    for source in SOURCES:

        results = (
            test_source(
                data=
                    data,
                source=
                    source
            )
        )

        if results.empty:
            continue

        all_results.append(
            results
        )

        summaries.append(
            summarize_source(
                results
            )
        )

    summary = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 125)
    print(
        "V2 INDIVIDUAL WEATHER SOURCE SIGNAL TEST"
    )
    print("=" * 125)

    print(
        f"{'Source':<12}"
        f"{'Days':<7}"
        f"{'Mkt Brier':<12}"
        f"{'Adj Brier':<12}"
        f"{'Mkt Log':<11}"
        f"{'Adj Log':<11}"
        f"{'Mkt Win P':<12}"
        f"{'Adj Win P':<12}"
        f"{'Better':<10}"
        f"{'Mean Beta':<12}"
    )

    print("-" * 125)

    for _, row in summary.iterrows():

        print(
            f"{row['source']:<12}"
            f"{int(row['days']):<7}"
            f"{row['market_brier']:<12.4f}"
            f"{row['adjusted_brier']:<12.4f}"
            f"{row['market_log']:<11.4f}"
            f"{row['adjusted_log']:<11.4f}"
            f"{row['market_winner']:<12.2%}"
            f"{row['adjusted_winner']:<12.2%}"
            f"{row['days_better']:<10.2%}"
            f"{row['mean_beta']:<+12.3f}"
        )

    return (
        summary,
        pd.concat(
            all_results,
            ignore_index=True
        )
    )


if __name__ == "__main__":

    run_source_tests()