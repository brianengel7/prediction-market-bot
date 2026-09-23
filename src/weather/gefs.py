from herbie import Herbie
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from src.database.db import (
    get_cached_temperature,
    save_temperature
)
from src.weather.settlement import (
    get_kxhighny_settlement_window
)
import time
import requests


CENTRAL_PARK_LAT = 40.77898
CENTRAL_PARK_LON = -73.96925
NYC_TIMEZONE = ZoneInfo("America/New_York")
GEFS_PRODUCT = "atmos.25"
GEFS_MAX_RETRIES = 5
GEFS_RETRY_BASE_SECONDS = 2
GEFS_REQUEST_PAUSE_SECONDS = 0.10


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
            product=GEFS_PRODUCT,
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
    target_date,
    not_before=None
):
    run_time = make_utc_timestamp(
        run_time
    )

    start_local, end_local = (
        get_kxhighny_settlement_window(
            target_date
        )
    )

    if not_before is not None:

        not_before = pd.Timestamp(
            not_before
        )

        if not_before.tzinfo is None:
            not_before = (
                not_before.tz_localize(
                    NYC_TIMEZONE
                )
            )
        else:
            not_before = (
                not_before.tz_convert(
                    NYC_TIMEZONE
                )
            )

        start_local = max(
            start_local,
            not_before
        )

    forecast_hours = []

    for forecast_hour in range(
        0,
        121,
        3
    ):
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

        if valid_time_local < start_local:
            continue

        if valid_time_local >= end_local:
            break

        forecast_hours.append(
            forecast_hour
        )

    if not forecast_hours:
        raise ValueError(
            "GEFS has no forecast points "
            f"inside the settlement window "
            f"for {target_date}."
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
        product=GEFS_PRODUCT,
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
        product=GEFS_PRODUCT,
        member=member,
        fxx=forecast_hour,
        verbose=False
    )

    data = load_gefs_dataset(
        run_time=run_time,
        member=member,
        forecast_hour=forecast_hour
    )

    nearest_point = data.sel(
        latitude=CENTRAL_PARK_LAT,
        longitude=CENTRAL_PARK_LON % 360,
        method="nearest"
    )

    actual_lat = float(
        nearest_point.latitude.values
    )

    actual_lon = float(
        nearest_point.longitude.values
    )

    actual_lon_west = (
        actual_lon
        if actual_lon <= 180
        else actual_lon - 360
    )

    print(
        f"GEFS grid point: "
        f"{actual_lat:.3f}, "
        f"{actual_lon_west:.3f}"
    )

    try:

        target_lon = (
            CENTRAL_PARK_LON % 360
        )

        temperature_kelvin = float(
            nearest_point["t2m"].values
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
        product=GEFS_PRODUCT,
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
    member,
    not_before=None,
    observed_high=None
):
    forecast_hours = (
        get_gefs_forecast_hours(
            run_time,
            target_date,
            not_before=not_before
        )
    )

    readings = []

    for forecast_hour in forecast_hours:

        reading = get_gefs_temperature(
            run_time,
            forecast_hour,
            member
        )

        readings.append(
            reading
        )

    maximum = max(
        readings,
        key=lambda x:
        x["temperature"]
    )

    forecast_high = maximum[
        "temperature"
    ]

    # If this is today, the final daily high
    # can never be below what has already happened.
    if observed_high is not None:

        effective_high = max(
            forecast_high,
            observed_high
        )

    else:

        effective_high = forecast_high

    return {
        "member": get_member_name(
            member
        ),

        # This is what our probability model uses.
        "max_temperature": effective_high,

        # Keep the raw remaining forecast for debugging.
        "forecast_high": forecast_high,

        "observed_floor": observed_high,

        "max_time": maximum[
            "valid_time"
        ],

        "temperatures": readings
    }


def get_gefs_ensemble(
    run_time,
    target_date,
    members,
    not_before=None,
    observed_high=None
):
    results = []

    for member in members:

        member_name = get_member_name(
            member
        )

        print()
        print(
            f"GEFS MEMBER: {member_name}"
        )

        result = get_gefs_member_high(
            run_time,
            target_date,
            member,
            not_before=not_before,
            observed_high=observed_high
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

def load_gefs_dataset(
    run_time,
    member,
    forecast_hour
):

    run_time = pd.Timestamp(
        run_time
    )

    if run_time.tzinfo is None:
        run_time_utc = run_time.tz_localize(
            "UTC"
        )
    else:
        run_time_utc = run_time.tz_convert(
            "UTC"
        )

    herbie_run_time = (
        run_time_utc.tz_localize(None)
    )

    for attempt in range(
        GEFS_MAX_RETRIES
    ):
        try:
            gefs = Herbie(
                herbie_run_time,
                model="gefs",
                product=GEFS_PRODUCT,
                member=member,
                fxx=forecast_hour
            )

            data = gefs.xarray(
                "TMP:2 m"
            )

            time.sleep(
                GEFS_REQUEST_PAUSE_SECONDS
            )

            return data

        except Exception as error:

            message = str(error)

            retryable = (
                isinstance(
                    error,
                    requests.exceptions.RequestException
                )
                or "503" in message
                or "Slow Down" in message
                or "429" in message
                or "timeout" in message.lower()
            )

            if (
                not retryable
                or attempt
                == GEFS_MAX_RETRIES - 1
            ):
                raise

            delay = (
                GEFS_RETRY_BASE_SECONDS
                * (2 ** attempt)
            )

            print(
                f"GEFS download failed "
                f"(attempt {attempt + 1}/"
                f"{GEFS_MAX_RETRIES}): "
                f"{error}"
            )

            print(
                f"Retrying in "
                f"{delay} seconds..."
            )

            time.sleep(
                delay
            )