from datetime import timedelta

import pandas as pd

from src.backtest.actuals import (
    get_actual_highs
)

from src.weather.hrrr import (
    get_hrrr_run
)


BACKTEST_CYCLE_HOUR = 12
BACKTEST_DAYS_BEFORE = 1


def get_backtest_run_time(
    target_date,
    cycle_hour=BACKTEST_CYCLE_HOUR,
    days_before=BACKTEST_DAYS_BEFORE
):
    """
    Return the fixed HRRR run used for a target date.

    Default:
        target date - 1 day at 12Z

    Example:
        target = 2026-09-17
        run    = 2026-09-16 12Z
    """

    target_date = pd.Timestamp(
        target_date
    ).date()

    run_date = (
        target_date
        - timedelta(
            days=days_before
        )
    )

    run_time = pd.Timestamp(
        year=run_date.year,
        month=run_date.month,
        day=run_date.day,
        hour=cycle_hour,
        tz="UTC"
    )

    return run_time


def backtest_hrrr_date(
    target_date,
    actual_high=None
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_time = get_backtest_run_time(
        target_date
    )

    print()
    print("=" * 60)
    print(
        f"HRRR FORECAST {target_date}"
    )

    print(
        f"HRRR run: "
        f"{run_time:%Y-%m-%d %HZ}"
    )

    result = get_hrrr_run(
        run_time=run_time,
        target_date=target_date
    )

    if not result:

        raise ValueError(
            f"No HRRR result found "
            f"for {target_date}"
        )

    forecast_high = float(
        result[
            "max_temperature"
        ]
    )

    error = None
    absolute_error = None
    squared_error = None

    if actual_high is not None:

        actual_high = float(
            actual_high
        )

        error = (
            forecast_high
            - actual_high
        )

        absolute_error = abs(
            error
        )

        squared_error = (
            error ** 2
        )

    print()
    print(
        f"HRRR high: "
        f"{forecast_high:.2f}°F"
    )

    if actual_high is not None:

        print(
            f"Actual:    "
            f"{actual_high:.2f}°F"
        )

        print(
            f"Error:     "
            f"{error:+.2f}°F"
        )

    return {
        "date":
            target_date,

        "run_time":
            run_time,

        "forecast_high":
            forecast_high,

        "actual_high":
            actual_high,

        "error":
            error,

        "absolute_error":
            absolute_error,

        "squared_error":
            squared_error,

        "hrrr_result":
            result
    }

def backtest_hrrr_range(
    start_date,
    end_date
):
    """
    Backtest HRRR over a range of dates.
    """

    actuals = get_actual_highs(
        start_date,
        end_date
    )

    results = []

    for actual in actuals:

        target_date = actual[
            "date"
        ]

        actual_high = actual[
            "actual_high"
        ]

        try:

            result = backtest_hrrr_date(
                target_date,
                actual_high
            )

            results.append(
                result
            )

        except Exception as error:

            print(
                f"FAILED {target_date}: "
                f"{error}"
            )

    return results


def calculate_hrrr_metrics(
    results
):
    if not results:
        return None

    errors = [
        result["error"]
        for result in results
    ]

    absolute_errors = [
        result["absolute_error"]
        for result in results
    ]

    squared_errors = [
        result["squared_error"]
        for result in results
    ]

    bias = sum(
        errors
    ) / len(
        errors
    )

    mae = sum(
        absolute_errors
    ) / len(
        absolute_errors
    )

    mse = sum(
        squared_errors
    ) / len(
        squared_errors
    )

    rmse = (
        mse ** 0.5
    )

    return {
        "count":
            len(results),

        "bias":
            bias,

        "mae":
            mae,

        "rmse":
            rmse
    }


if __name__ == "__main__":

    results = backtest_hrrr_range(
        "2026-09-01",
        "2026-09-05"
    )

    metrics = calculate_hrrr_metrics(
        results
    )

    print()
    print(
        "=" * 60
    )

    print(
        "HRRR BACKTEST SUMMARY"
    )

    print(
        "=" * 60
    )

    if metrics is None:

        print(
            "No successful results."
        )

    else:

        print(
            f"Dates: {metrics['count']}"
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