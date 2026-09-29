import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset,
    score_date
)


SIGNAL_COLUMN = "NBM_yes"

FROZEN_BETA = -0.40

HOLDOUT_START = pd.Timestamp(
    "2026-09-11"
)

HOLDOUT_END = pd.Timestamp(
    "2026-09-26"
)


def run_holdout():

    data = (
        load_dataset()
    )

    data = data[
        (
            data[
                "target_date"
            ]
            >=
            HOLDOUT_START
        )
        &
        (
            data[
                "target_date"
            ]
            <=
            HOLDOUT_END
        )
    ].copy()

    dates = sorted(
        data[
            "target_date"
        ].unique()
    )

    results = []

    for target_date in dates:

        group = data[
            data[
                "target_date"
            ]
            ==
            target_date
        ]

        if len(group) != 6:

            print(
                f"Skipping "
                f"{target_date}: "
                f"{len(group)} contracts"
            )

            continue

        metrics = (
            score_date(
                group=
                    group,

                signal_column=
                    SIGNAL_COLUMN,

                beta=
                    FROZEN_BETA
            )
        )

        results.append({

            "target_date":
                target_date,

            **metrics
        })

    results = pd.DataFrame(
        results
    )

    if results.empty:

        print(
            "No complete holdout dates."
        )

        return

    print()
    print("=" * 90)

    print(
        "NBM TRUE HOLDOUT TEST"
    )

    print("=" * 90)

    print(
        f"Dates:       "
        f"{len(results)}"
    )

    print(
        f"Frozen beta: "
        f"{FROZEN_BETA:+.2f}"
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
        "WINNER PROBABILITY"
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
        f"Adjusted beats Kalshi: "
        f"{results['better'].mean():.2%}"
    )

    results[
        "brier_improvement"
    ] = (
        results[
            "market_brier"
        ]
        -
        results[
            "adjusted_brier"
        ]
    )

    results[
        "log_improvement"
    ] = (
        results[
            "market_log_loss"
        ]
        -
        results[
            "adjusted_log_loss"
        ]
    )

    print()

    print(
        f"Mean Brier improvement: "
        f"{results['brier_improvement'].mean():+.4f}"
    )

    print(
        f"Mean log improvement:   "
        f"{results['log_improvement'].mean():+.4f}"
    )

    print()

    print(
        "DAILY RESULTS"
    )

    print("-" * 90)

    display = (
        results.copy()
    )

    display[
        "target_date"
    ] = (
        pd.to_datetime(
            display[
                "target_date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    print(
        display[
            [
                "target_date",
                "market_brier",
                "adjusted_brier",
                "market_log_loss",
                "adjusted_log_loss",
                "better"
            ]
        ]
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":

    run_holdout()