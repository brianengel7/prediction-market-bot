import math

import pandas as pd

from src.backtest.actuals import (
    get_actual_high
)

from src.backtest.gefs import (
    get_backtest_run_time
)

from src.weather.gefs import (
    CENTRAL_PARK_LAT,
    CENTRAL_PARK_LON,
    get_gefs_forecast_hours,
    load_gefs_dataset
)


GRID_STEP = 0.25


def kelvin_to_fahrenheit(
    temperature_kelvin
):
    return (
        temperature_kelvin
        - 273.15
    ) * 9 / 5 + 32


def get_surrounding_points():

    target_lon = (
        CENTRAL_PARK_LON % 360
    )

    lat_low = (
        math.floor(
            CENTRAL_PARK_LAT
            / GRID_STEP
        )
        * GRID_STEP
    )

    lat_high = (
        math.ceil(
            CENTRAL_PARK_LAT
            / GRID_STEP
        )
        * GRID_STEP
    )

    lon_low = (
        math.floor(
            target_lon
            / GRID_STEP
        )
        * GRID_STEP
    )

    lon_high = (
        math.ceil(
            target_lon
            / GRID_STEP
        )
        * GRID_STEP
    )

    return [
        (
            lat_low,
            lon_low
        ),
        (
            lat_low,
            lon_high
        ),
        (
            lat_high,
            lon_low
        ),
        (
            lat_high,
            lon_high
        )
    ]


def convert_lon_for_display(
    longitude
):
    if longitude > 180:
        return longitude - 360

    return longitude


def diagnose_date(
    target_date,
    member=0
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_time = get_backtest_run_time(
        target_date
    )

    actual = get_actual_high(
        target_date
    )

    if actual is None:
        raise ValueError(
            f"No actual high for "
            f"{target_date}"
        )

    actual_high = actual[
        "actual_high"
    ]

    forecast_hours = (
        get_gefs_forecast_hours(
            run_time,
            target_date
        )
    )

    points = (
        get_surrounding_points()
    )

    point_highs = {
        point: float("-inf")
        for point in points
    }

    interpolated_high = float(
        "-inf"
    )

    print()
    print("=" * 70)

    print(
        f"GEFS GRID DIAGNOSTIC: "
        f"{target_date}"
    )

    print(
        f"Run: "
        f"{run_time:%Y-%m-%d %HZ}"
    )

    print(
        f"Actual KNYC high: "
        f"{actual_high:.2f}°F"
    )

    print("=" * 70)

    target_lon = (
        CENTRAL_PARK_LON % 360
    )

    for forecast_hour in forecast_hours:

        data = load_gefs_dataset(
            run_time=run_time,
            member=member,
            forecast_hour=forecast_hour
        )

        temperature = data[
            "t2m"
        ]

        # Four surrounding grid cells
        for (
            latitude,
            longitude
        ) in points:

            value_k = float(
                temperature.sel(
                    latitude=latitude,
                    longitude=longitude,
                    method="nearest"
                ).values
            )

            value_f = (
                kelvin_to_fahrenheit(
                    value_k
                )
            )

            point_highs[
                (
                    latitude,
                    longitude
                )
            ] = max(
                point_highs[
                    (
                        latitude,
                        longitude
                    )
                ],
                value_f
            )

        # Bilinear interpolation at the
        # actual Central Park coordinates
        interpolated_k = float(
            temperature.interp(
                latitude=CENTRAL_PARK_LAT,
                longitude=target_lon
            ).values
        )

        interpolated_f = (
            kelvin_to_fahrenheit(
                interpolated_k
            )
        )

        interpolated_high = max(
            interpolated_high,
            interpolated_f
        )

    print()
    print(
        "GRID-POINT DAILY HIGHS"
    )

    print("-" * 70)

    for (
        latitude,
        longitude
    ), high in point_highs.items():

        display_lon = (
            convert_lon_for_display(
                longitude
            )
        )

        error = (
            high - actual_high
        )

        print(
            f"{latitude:6.2f}, "
            f"{display_lon:7.2f} | "
            f"{high:6.2f}°F | "
            f"error {error:+6.2f}°F"
        )

    interpolation_error = (
        interpolated_high
        - actual_high
    )

    print()
    print(
        f"Interpolated Central Park: "
        f"{interpolated_high:.2f}°F"
    )

    print(
        f"Interpolation error:       "
        f"{interpolation_error:+.2f}°F"
    )

    return {
        "date": target_date,
        "actual_high": actual_high,
        "point_highs": point_highs,
        "interpolated_high":
            interpolated_high,
        "interpolated_error":
            interpolation_error
    }


if __name__ == "__main__":

    # Three deliberately different days:
    #
    # Sep 2:
    # GEFS was ~6°F too warm.
    #
    # Sep 6:
    # GEFS was nearly correct while
    # HRRR was ~5°F too cold.
    #
    # Sep 10:
    # both models were substantially warm.

    dates = [
        "2026-09-02",
        "2026-09-06",
        "2026-09-10"
    ]

    for date in dates:

        diagnose_date(
            date,
            member=0
        )