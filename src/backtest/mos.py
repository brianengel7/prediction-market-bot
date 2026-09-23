from datetime import timedelta

import pandas as pd

from src.backtest.actuals import (
    get_actual_highs
)

from src.database.db import (
    save_weather_actual,
    save_weather_model_backtest
)

from src.weather.mos import (
    get_mos_high_forecast
)


BACKTEST_CYCLE_HOUR = 12
BACKTEST_DAYS_BEFORE = 1

FORECAST_HORIZON = "D-1_12Z"


def get_backtest_run_time(
    target_date
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_date = (
        target_date
        - timedelta(days=1)
    )

    return pd.Timestamp(
        year=run_date.year,
        month=run_date.month,
        day=run_date.day,
        hour=BACKTEST_CYCLE_HOUR,
        tz="UTC"
    )


def backtest_mos_date(
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
        f"BACKTESTING GFS MOS "
        f"{target_date}"
    )

    print(
        f"MOS run: "
        f"{run_time:%Y-%m-%d %HZ}"
    )

    result = get_mos_high_forecast(
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
        f"MOS high: {forecast_high:.2f}°F"
    )

    print(
        f"Actual:   {actual_high:.2f}°F"
    )

    print(
        f"Error:    {error:+.2f}°F"
    )

    return result


def backtest_mos_range(
    start_date,
    end_date
):
    actuals = get_actual_highs(
        start_date,
        end_date
    )

    successful = 0
    failed = 0

    for actual in actuals:

        target_date = actual[
            "date"
        ]

        actual_high = actual[
            "actual_high"
        ]

        save_weather_actual(
            target_date=target_date,
            actual_high=actual_high,
            station="KNYC",
            settlement_source="NWS_CLI",
            product="CLI"
        )

        try:

            result = backtest_mos_date(
                target_date,
                actual_high
            )

            save_weather_model_backtest(
                target_date=target_date,

                source="GFS_MOS",
                product="MAV",

                model_run=result[
                    "run_time"
                ],

                forecast_horizon=
                    FORECAST_HORIZON,

                forecast_value=result[
                    "forecast_high"
                ],

                actual_high=actual_high,

                station="KNYC",

                extraction_method=
                    "station_guidance",

                model_version=
                    "GFS-MOS-MAV"
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"GFS MOS FAILED "
                f"{target_date}: "
                f"{error}"
            )

    print()
    print("=" * 60)
    print("GFS MOS BACKTEST COMPLETE")
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total:      {len(actuals)}"
    )


if __name__ == "__main__":

    backtest_mos_range(
        "2026-09-01",
        "2026-09-10"
    )