from datetime import timedelta

import pandas as pd

from src.backtest.actuals import (
    get_actual_highs
)

from src.weather.gefs import (
    get_gefs_ensemble
)


BACKTEST_CYCLE_HOUR = 12
BACKTEST_DAYS_BEFORE = 1

GEFS_MEMBERS = list(range(31))


def get_backtest_run_time(
    target_date,
    cycle_hour=BACKTEST_CYCLE_HOUR,
    days_before=BACKTEST_DAYS_BEFORE
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_date = (
        target_date
        - timedelta(
            days=days_before
        )
    )

    return pd.Timestamp(
        year=run_date.year,
        month=run_date.month,
        day=run_date.day,
        hour=cycle_hour,
        tz="UTC"
    )


def backtest_gefs_date(
    target_date,
    actual_high
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_time = get_backtest_run_time(
        target_date
    )

    print()
    print("=" * 60)
    print(f"BACKTESTING {target_date}")
    print(
        f"GEFS run: "
        f"{run_time:%Y-%m-%d %HZ}"
    )

    result = get_gefs_ensemble(
        run_time=run_time,
        target_date=target_date,
        members=GEFS_MEMBERS
    )

    mean_high = result[
        "mean_high"
    ]

    median_high = result[
        "median_high"
    ]

    mean_error = (
        mean_high - actual_high
    )

    median_error = (
        median_high - actual_high
    )

    print(
        f"GEFS mean:   "
        f"{mean_high:.2f}°F"
    )

    print(
        f"GEFS median: "
        f"{median_high:.2f}°F"
    )

    print(
        f"Actual:      "
        f"{actual_high:.2f}°F"
    )

    print(
        f"Mean error:  "
        f"{mean_error:+.2f}°F"
    )

    print(
        f"Spread:      "
        f"{result['std_high']:.2f}°F"
    )

    return {
        "date": target_date,
        "run_time": run_time,

        "mean_high": mean_high,
        "median_high": median_high,
        "std_high": result[
            "std_high"
        ],

        "min_high": result[
            "min_high"
        ],

        "max_high": result[
            "max_high"
        ],

        "actual_high": actual_high,

        "mean_error": mean_error,

        "mean_absolute_error":
            abs(mean_error),

        "mean_squared_error":
            mean_error ** 2,

        "median_error":
            median_error,

        "members": result[
            "members"
        ]
    }


def backtest_gefs_range(
    start_date,
    end_date
):
    actuals = get_actual_highs(
        start_date,
        end_date
    )

    results = []

    for actual in actuals:

        try:
            result = backtest_gefs_date(
                target_date=actual[
                    "date"
                ],
                actual_high=actual[
                    "actual_high"
                ]
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"FAILED "
                f"{actual['date']}: "
                f"{error}"
            )

    return results


def calculate_gefs_metrics(
    results
):
    if not results:
        return None

    errors = [
        result["mean_error"]
        for result in results
    ]

    absolute_errors = [
        result[
            "mean_absolute_error"
        ]
        for result in results
    ]

    squared_errors = [
        result[
            "mean_squared_error"
        ]
        for result in results
    ]

    bias = (
        sum(errors)
        / len(errors)
    )

    mae = (
        sum(absolute_errors)
        / len(absolute_errors)
    )

    mse = (
        sum(squared_errors)
        / len(squared_errors)
    )

    rmse = (
        mse ** 0.5
    )

    average_spread = (
        sum(
            result["std_high"]
            for result in results
        )
        / len(results)
    )

    return {
        "count": len(results),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "average_spread":
            average_spread
    }


if __name__ == "__main__":

    # Keep this tiny at first.
    # GEFS is much heavier than HRRR
    # because we're downloading 31 members.
    results = backtest_gefs_range(
        "2026-09-01",
        "2026-09-02"
    )

    metrics = calculate_gefs_metrics(
        results
    )

    print()
    print("=" * 60)
    print("GEFS BACKTEST SUMMARY")
    print("=" * 60)

    if metrics is None:
        print(
            "No successful results."
        )

    else:
        print(
            f"Dates: "
            f"{metrics['count']}"
        )

        print(
            f"Bias: "
            f"{metrics['bias']:+.2f}°F"
        )

        print(
            f"MAE:  "
            f"{metrics['mae']:.2f}°F"
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.2f}°F"
        )

        print(
            f"Average ensemble spread: "
            f"{metrics['average_spread']:.2f}°F"
        )