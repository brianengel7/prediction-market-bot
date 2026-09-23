import sqlite3

from src.database.db import (
    DATABASE_PATH,
    save_weather_actual,
    save_weather_model_backtest
)


def migrate():
    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        connection.row_factory = (
            sqlite3.Row
        )

        rows = connection.execute(
            """
            SELECT *
            FROM weather_backtests
            ORDER BY target_date
            """
        ).fetchall()

    actual_count = 0
    hrrr_count = 0
    gefs_count = 0

    for row in rows:

        target_date = row[
            "target_date"
        ]

        actual_high = row[
            "actual_high"
        ]

        horizon = row[
            "forecast_horizon"
        ]

        save_weather_actual(
            target_date=target_date,
            actual_high=actual_high
        )

        actual_count += 1

        if row["hrrr_high"] is not None:

            save_weather_model_backtest(
                target_date=target_date,

                source="HRRR",
                product="sfc",

                model_run=row[
                    "hrrr_run_time"
                ],

                forecast_horizon=horizon,

                forecast_value=row[
                    "hrrr_high"
                ],

                actual_high=actual_high,

                extraction_method="nearest",
                model_version="backtest-v1"
            )

            hrrr_count += 1

        if row["gefs_mean"] is not None:

            save_weather_model_backtest(
                target_date=target_date,

                source="GEFS",
                product="atmos.25",

                model_run=row[
                    "gefs_run_time"
                ],

                forecast_horizon=horizon,

                forecast_value=row[
                    "gefs_mean"
                ],

                forecast_median=row[
                    "gefs_median"
                ],

                forecast_std=row[
                    "gefs_std"
                ],

                member_count=31,

                actual_high=actual_high,

                extraction_method="nearest",
                model_version="backtest-v1"
            )

            gefs_count += 1

    print()
    print("MIGRATION COMPLETE")
    print("-" * 40)

    print(
        f"Actuals: {actual_count}"
    )

    print(
        f"HRRR:    {hrrr_count}"
    )

    print(
        f"GEFS:    {gefs_count}"
    )


if __name__ == "__main__":
    migrate()