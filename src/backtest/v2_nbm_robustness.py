import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset,
    fit_beta,
    score_date
)


SIGNAL_COLUMN = "NBM_yes"
TRAIN_DAYS = 20


def evaluate_dates(
    data,
    dates,
    beta
):

    results = []

    for target_date in dates:

        test = data[
            data[
                "target_date"
            ]
            ==
            target_date
        ]

        if len(test) != 6:
            continue

        metrics = score_date(
            group=
                test,

            signal_column=
                SIGNAL_COLUMN,

            beta=
                beta
        )

        results.append({

            "target_date":
                target_date,

            "beta":
                beta,

            **metrics
        })

    return pd.DataFrame(
        results
    )


def print_period(
    name,
    results
):

    if results.empty:
        return

    print()
    print(name)
    print("-" * 70)

    print(
        f"Days:               "
        f"{len(results)}"
    )

    print(
        f"Kalshi Brier:       "
        f"{results['market_brier'].mean():.4f}"
    )

    print(
        f"NBM adjusted Brier: "
        f"{results['adjusted_brier'].mean():.4f}"
    )

    print(
        f"Kalshi log loss:    "
        f"{results['market_log_loss'].mean():.4f}"
    )

    print(
        f"NBM adjusted log:   "
        f"{results['adjusted_log_loss'].mean():.4f}"
    )

    print(
        f"Kalshi winner prob: "
        f"{results['market_winner_probability'].mean():.2%}"
    )

    print(
        f"Adjusted winner:    "
        f"{results['adjusted_winner_probability'].mean():.2%}"
    )

    print(
        f"Days better:        "
        f"{results['better'].mean():.2%}"
    )


def run_robustness_test():

    data = (
        load_dataset()
        .dropna(
            subset=[
                SIGNAL_COLUMN
            ]
        )
        .copy()
    )

    dates = sorted(
        data[
            "target_date"
        ].unique()
    )

    # ========================================================
    # ONE training period
    #
    # Fit beta ONCE.
    # Never update it using test results.
    # ========================================================

    training_dates = (
        dates[
            :TRAIN_DAYS
        ]
    )

    test_dates = (
        dates[
            TRAIN_DAYS:
        ]
    )

    training = data[
        data[
            "target_date"
        ].isin(
            training_dates
        )
    ]

    beta, training_loss = (
        fit_beta(
            training_data=
                training,

            signal_column=
                SIGNAL_COLUMN
        )
    )

    # ========================================================
    # Freeze beta for entire test period
    # ========================================================

    results = (
        evaluate_dates(
            data=
                data,

            dates=
                test_dates,

            beta=
                beta
        )
    )

    print()
    print("=" * 80)
    print(
        "NBM FROZEN-BETA ROBUSTNESS TEST"
    )
    print("=" * 80)

    print(
        f"Training days:      "
        f"{len(training_dates)}"
    )

    print(
        f"Test days:          "
        f"{len(test_dates)}"
    )

    print(
        f"Frozen beta:        "
        f"{beta:+.3f}"
    )

    print(
        f"Training log loss:  "
        f"{training_loss:.4f}"
    )

    # ========================================================
    # Entire OOS period
    # ========================================================

    print_period(
        "FULL OUT-OF-SAMPLE PERIOD",
        results
    )

    # ========================================================
    # Stability through time
    # ========================================================

    midpoint = (
        len(results)
        // 2
    )

    first_half = (
        results.iloc[
            :midpoint
        ]
    )

    second_half = (
        results.iloc[
            midpoint:
        ]
    )

    print_period(
        "FIRST HALF",
        first_half
    )

    print_period(
        "SECOND HALF",
        second_half
    )

    # ========================================================
    # Daily results
    # ========================================================

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
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":

    run_robustness_test()