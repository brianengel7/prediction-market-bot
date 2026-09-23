from datetime import timedelta

import pandas as pd

from src.backtest.actuals import (
    get_actual_highs
)

from src.database.db import (
    save_weather_actual,
    save_weather_model_backtest
)

from src.weather.nbm import (
    get_nbm_high_forecast
)


BACKTEST_CYCLE_HOUR = 12
BACKTEST_DAYS_BEFORE = 1


def get_backtest_run_time(
    target_date
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_date = (
        target_date
        - timedelta(
            days=BACKTEST_DAYS_BEFORE
        )
    )

    return pd.Timestamp(
        year=run_date.year,
        month=run_date.month,
        day=run_date.day,
        hour=BACKTEST_CYCLE_HOUR,
        tz="UTC"
    )


def backtest_nbm_date(
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

    print(
        f"BACKTESTING NBM "
        f"{target_date}"
    )

    print(
        f"NBM run: "
        f"{run_time:%Y-%m-%d %HZ}"
    )

    result = get_nbm_high_forecast(
        run_time=run_time,
        target_date=target_date
    )

    forecast_high = result[
        "forecast_high"
    ]

    error = (
        forecast_high
        - actual_high
    )

    print(
        f"NBM high: "
        f"{forecast_high:.2f}°F"
    )

    print(
        f"NBM std:  "
        f"{result['forecast_std']:.2f}°F"
    )

    print(
        f"Actual:   "
        f"{actual_high:.2f}°F"
    )

    print(
        f"Error:    "
        f"{error:+.2f}°F"
    )

    return result


def backtest_nbm_range(
    start_date,
    end_date
):
    actuals = get_actual_highs(
        start_date,
        end_date
    )

    successful = 0

    for actual in actuals:

        target_date = actual[
            "date"
        ]

        actual_high = actual[
            "actual_high"
        ]

        save_weather_actual(
            target_date=target_date,
            actual_high=actual_high
        )

        try:

            result = backtest_nbm_date(
                target_date,
                actual_high
            )

            save_weather_model_backtest(
                target_date=target_date,

                source="NBM",
                product="NBS",

                model_run=result[
                    "run_time"
                ],

                forecast_horizon=
                    "D-1_12Z",

                forecast_value=result[
                    "forecast_high"
                ],

                forecast_std=result[
                    "forecast_std"
                ],

                actual_high=actual_high,

                station="KNYC",

                extraction_method=
                    "station_text",

                model_version=
                    "NBM-v5.0-NBS"
            )

            successful += 1

        except Exception as error:

            print(
                f"NBM FAILED "
                f"{target_date}: "
                f"{error}"
            )

    print()
    print("=" * 60)

    print(
        f"NBM completed: "
        f"{successful}/"
        f"{len(actuals)} dates"
    )


if __name__ == "__main__":

    backtest_nbm_range(
        "2026-09-01",
        "2026-09-10"
    )