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


def make_utc_timestamp(timestamp):
    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


def get_member_name(member):

    if member == 0:
        return "c00"

    return f"p{member:02d}"


def get_latest_gefs_run():

    now = pd.Timestamp.now(tz="UTC")

    candidate = now.floor("6h")

    while True:

        herbie_time = candidate.tz_localize(None)

        print(
            f"Checking GEFS "
            f"{candidate:%Y-%m-%d %HZ}..."
        )

        gefs = Herbie(
            herbie_time,
            model="gefs",
            product="atmos.5",
            member=0,
            fxx=0,
            verbose=False
        )

        if gefs.grib is not None:

            print(
                f"Found GEFS run: "
                f"{candidate:%Y-%m-%d %HZ}"
            )

            return candidate

        candidate -= pd.Timedelta(hours=6)


def get_gefs_forecast_hours(
    run_time,
    target_date
):

    run_time = make_utc_timestamp(run_time)

    target_date = pd.Timestamp(
        target_date
    ).date()

    forecast_hours = []

    # Plenty for the near-term markets we're
    # currently trading.
    for forecast_hour in range(0, 121, 3):

        valid_time_utc = (
            run_time
            + pd.Timedelta(
                hours=forecast_hour
            )
        )

        valid_time_local = (
            valid_time_utc.tz_convert(
                NYC_TIMEZONE
            )
        )

        if (
            valid_time_local.date()
            == target_date
            and 8
            <= valid_time_local.hour
            <= 23
        ):
            forecast_hours.append(
                forecast_hour
            )

        if valid_time_local.date() > target_date:
            break

    if not forecast_hours:
        raise ValueError(
            "GEFS run does not cover "
            f"target date {target_date}."
        )

    return forecast_hours


def get_gefs_temperature(
    run_time,
    forecast_hour,
    member
):
    run_time_utc = make_utc_timestamp(
        run_time
    )

    member_name = get_member_name(
        member
    )

    cached = get_cached_temperature(
        model="gefs",
        member=member_name,
        run_time=run_time_utc,
        forecast_hour=forecast_hour,
        latitude=CENTRAL_PARK_LAT,
        longitude=CENTRAL_PARK_LON
    )

    if cached is not None:

        print(
            f"Cache hit: GEFS "
            f"{member_name} "
            f"F{forecast_hour:03d}"
        )

        return cached

    print(
        f"Downloading: GEFS "
        f"{member_name} "
        f"F{forecast_hour:03d}"
    )

    herbie_run_time = (
        run_time_utc.tz_localize(None)
    )

    gefs = Herbie(
        herbie_run_time,
        model="gefs",
        product="atmos.5",
        member=member,
        fxx=forecast_hour,
        verbose=False
    )

    data = gefs.xarray(
        "TMP:2 m"
    )

    try:

        target_lon = (
            CENTRAL_PARK_LON % 360
        )

        temperature_kelvin = float(
            data["t2m"].sel(
                latitude=CENTRAL_PARK_LAT,
                longitude=target_lon,
                method="nearest"
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
            valid_time = (
                valid_time.tz_localize(
                    "UTC"
                )
            )
        else:
            valid_time = (
                valid_time.tz_convert(
                    "UTC"
                )
            )

        valid_time = (
            valid_time.tz_convert(
                NYC_TIMEZONE
            )
        )

    finally:
        data.close()

    save_temperature(
        model="gefs",
        member=member_name,
        run_time=run_time_utc,
        forecast_hour=forecast_hour,
        latitude=CENTRAL_PARK_LAT,
        longitude=CENTRAL_PARK_LON,
        valid_time=valid_time,
        temperature_f=(
            temperature_fahrenheit
        )
    )

    return {
        "forecast_hour": forecast_hour,
        "valid_time": valid_time,
        "temperature": (
            temperature_fahrenheit
        )
    }


def get_gefs_member_high(
    run_time,
    target_date,
    member
):
    forecast_hours = (
        get_gefs_forecast_hours(
            run_time,
            target_date
        )
    )

    readings = []

    for forecast_hour in forecast_hours:

        reading = get_gefs_temperature(
            run_time,
            forecast_hour,
            member
        )

        readings.append(reading)

    maximum = max(
        readings,
        key=lambda x: x["temperature"]
    )

    return {
        "member": get_member_name(member),
        "max_temperature": (
            maximum["temperature"]
        ),
        "max_time": maximum["valid_time"],
        "temperatures": readings
    }


def get_gefs_ensemble(
    run_time,
    target_date,
    members
):
    results = []

    for member in members:

        member_name = get_member_name(
            member
        )

        print()
        print("=" * 50)
        print(
            f"GEFS MEMBER: {member_name}"
        )
        print("=" * 50)

        result = get_gefs_member_high(
            run_time,
            target_date,
            member
        )

        results.append(result)

        print(
            f"{member_name} predicted high: "
            f"{result['max_temperature']:.2f}°F"
        )

    highs = np.array([
        result["max_temperature"]
        for result in results
    ])

    summary = {
        "run_time": make_utc_timestamp(
            run_time
        ),
        "target_date": (
            pd.Timestamp(
                target_date
            ).date()
        ),
        "members": results,
        "mean_high": float(
            np.mean(highs)
        ),
        "median_high": float(
            np.median(highs)
        ),
        "std_high": float(
            np.std(
                highs,
                ddof=1
            )
        ) if len(highs) > 1 else 0.0,
        "min_high": float(
            np.min(highs)
        ),
        "max_high": float(
            np.max(highs)
        )
    }

    return summary