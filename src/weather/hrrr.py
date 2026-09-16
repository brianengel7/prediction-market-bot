from herbie import Herbie
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from src.database.db import (
    get_cached_temperature,
    save_temperature
)

CENTRAL_PARK_LAT = 40.77898
CENTRAL_PARK_LON = -73.96925
NYC_TIMEZONE = ZoneInfo("America/New_York")

central_park_grid_index = None


def make_utc_timestamp(timestamp):
    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def get_forecast_hour(run_time, target_date, local_hour):

    run_time = make_utc_timestamp(run_time)
    target_date = pd.Timestamp(target_date).date()

    local_time = pd.Timestamp(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=local_hour,
        tz=NYC_TIMEZONE
    )

    utc_time = local_time.tz_convert("UTC")

    forecast_hour = int(
        (utc_time - run_time).total_seconds() / 3600
    )

    if forecast_hour < 0:
        raise ValueError(
            "Requested forecast time occurs before this HRRR run."
        )

    if forecast_hour > 48:
        raise ValueError(
            f"HRRR run does not cover F{forecast_hour}."
        )

    return forecast_hour


def get_forecast_hours(run_time, target_date):

    local_hours = range(8, 23, 2)

    forecast_hours = []

    for local_hour in local_hours:
        forecast_hour = get_forecast_hour(
            run_time,
            target_date,
            local_hour
        )

        forecast_hours.append(forecast_hour)

    return forecast_hours


def get_hrrr_temperature(run_time, forecast_hour):
    global central_park_grid_index

    run_time_utc = make_utc_timestamp(run_time)

    cached = get_cached_temperature(
        model="hrrr",
        run_time=run_time_utc,
        forecast_hour=forecast_hour,
        latitude=CENTRAL_PARK_LAT,
        longitude=CENTRAL_PARK_LON
    )

    if cached is not None:

        print(
            f"Cache hit: HRRR "
            f"{run_time_utc:%Y-%m-%d %HZ} "
            f"F{forecast_hour:02d}"
        )

        return cached

    print(
        f"Downloading: HRRR "
        f"{run_time_utc:%Y-%m-%d %HZ} "
        f"F{forecast_hour:02d}"
    )

    herbie_run_time = run_time_utc.tz_localize(None)

    hrrr = Herbie(
        herbie_run_time,
        model="hrrr",
        product="sfc",
        fxx=forecast_hour,
        verbose=False
    )

    data = hrrr.xarray("TMP:2 m")

    try:
        if central_park_grid_index is None:

            target_lon = CENTRAL_PARK_LON % 360

            latitudes = data.latitude.values
            longitudes = data.longitude.values

            distance = (
                (latitudes - CENTRAL_PARK_LAT) ** 2
                + (longitudes - target_lon) ** 2
            )

            central_park_grid_index = np.unravel_index(
                np.argmin(distance),
                distance.shape
            )

        y_index, x_index = central_park_grid_index

        temperature_kelvin = float(
            data["t2m"].isel(
                y=y_index,
                x=x_index
            ).values
        )

        temperature_fahrenheit = (
            (temperature_kelvin - 273.15)
            * 9 / 5
            + 32
        )

        valid_time = pd.Timestamp(
            data["valid_time"].values
        )

        if valid_time.tzinfo is None:
            valid_time = valid_time.tz_localize("UTC")
        else:
            valid_time = valid_time.tz_convert("UTC")

        valid_time = valid_time.tz_convert(
            NYC_TIMEZONE
        )

    finally:
        data.close()

    save_temperature(
        model="hrrr",
        run_time=run_time_utc,
        forecast_hour=forecast_hour,
        latitude=CENTRAL_PARK_LAT,
        longitude=CENTRAL_PARK_LON,
        valid_time=valid_time,
        temperature_f=temperature_fahrenheit
    )

    return {
        "forecast_hour": forecast_hour,
        "valid_time": valid_time,
        "temperature": temperature_fahrenheit
    }


def get_hrrr_run(run_time, target_date):

    run_time_utc = make_utc_timestamp(run_time)
    target_date = pd.Timestamp(target_date).date()

    forecast_hours = get_forecast_hours(
        run_time,
        target_date
    )

    readings = {}

    for forecast_hour in forecast_hours:

        reading = get_hrrr_temperature(
            run_time,
            forecast_hour
        )

        readings[forecast_hour] = reading

        print(
            f"{reading['valid_time']:%I:%M %p}: "
            f"{reading['temperature']:.2f}°F"
        )

    coarse_maximum = max(
        readings.values(),
        key=lambda reading: reading["temperature"]
    )

    hottest_forecast_hour = coarse_maximum[
        "forecast_hour"
    ]

    refinement_hours = [
        hottest_forecast_hour - 1,
        hottest_forecast_hour + 1
    ]

    for forecast_hour in refinement_hours:

        if forecast_hour < 0 or forecast_hour > 48:
            continue

        if forecast_hour in readings:
            continue

        valid_time_utc = (
            run_time_utc
            + pd.Timedelta(hours=forecast_hour)
        )

        valid_time_local = valid_time_utc.tz_convert(
            NYC_TIMEZONE
        )

        if valid_time_local.date() != target_date:
            continue

        reading = get_hrrr_temperature(
            run_time,
            forecast_hour
        )

        readings[forecast_hour] = reading

        print(
            f"{reading['valid_time']:%I:%M %p}: "
            f"{reading['temperature']:.2f}°F "
            "(refinement)"
        )

    temperatures = sorted(
        readings.values(),
        key=lambda reading: reading["valid_time"]
    )

    maximum = max(
        temperatures,
        key=lambda reading: reading["temperature"]
    )

    return {
        "run_time": run_time_utc,
        "target_date": target_date,
        "max_temperature": maximum["temperature"],
        "max_time": maximum["valid_time"],
        "temperatures": temperatures
    }


def compare_hrrr_runs(run_times, target_date):

    results = []

    for run_time in run_times:

        print()
        print("=" * 50)
        print(f"HRRR RUN: {run_time}")
        print("=" * 50)

        result = get_hrrr_run(
            run_time,
            target_date
        )

        results.append(result)

        print()
        print(
            f"Predicted high: "
            f"{result['max_temperature']:.2f}°F"
        )

        print(
            f"Predicted high time: "
            f"{result['max_time']:%I:%M %p}"
        )

    for index, result in enumerate(results):

        if index == 0:
            result["change"] = None

        else:
            result["change"] = (
                result["max_temperature"]
                - results[index - 1]["max_temperature"]
            )

    return results

def get_recent_hrrr_runs(
    target_date,
    count=3
):
    now = pd.Timestamp.now(
        tz="UTC"
    )

    candidate = now.floor(
        "6h"
    )

    runs = []

    attempts = 0
    max_attempts = 12

    while (
        len(runs) < count
        and attempts < max_attempts
    ):

        attempts += 1

        print(
            f"Checking HRRR "
            f"{candidate:%Y-%m-%d %HZ}..."
        )

        try:

            forecast_hours = (
                get_forecast_hours(
                    candidate,
                    target_date
                )
            )

        except ValueError as error:

            print(
                f"Skipping HRRR "
                f"{candidate:%Y-%m-%d %HZ}: "
                f"{error}"
            )

            candidate -= pd.Timedelta(
                hours=6
            )

            continue

        required_forecast_hour = max(
            forecast_hours
        )

        herbie_time = (
            candidate.tz_localize(None)
        )

        hrrr = Herbie(
            herbie_time,
            model="hrrr",
            product="sfc",
            fxx=required_forecast_hour,
            verbose=False
        )

        if hrrr.grib is not None:

            print(
                f"Found usable HRRR run: "
                f"{candidate:%Y-%m-%d %HZ} "
                f"(through F"
                f"{required_forecast_hour})"
            )

            runs.append(
                candidate
            )

        else:

            print(
                f"HRRR "
                f"{candidate:%Y-%m-%d %HZ} "
                f"does not yet have F"
                f"{required_forecast_hour}."
            )

        candidate -= pd.Timedelta(
            hours=6
        )

    if not runs:
        raise RuntimeError(
            "No HRRR runs currently cover "
            f"target date {target_date}."
        )

    runs.reverse()

    return runs