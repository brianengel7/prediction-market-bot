import json

import pandas as pd
import requests


IEM_MOS_URL = (
    "https://mesonet.agron.iastate.edu/"
    "api/1/mos.json"
)

MOS_STATION = "KNYC"
MOS_MODEL = "GFS"

REQUEST_TIMEOUT = 30


def normalize_run_time(
    run_time
):
    run_time = pd.Timestamp(
        run_time
    )

    if run_time.tzinfo is None:

        run_time = run_time.tz_localize(
            "UTC"
        )

    else:

        run_time = run_time.tz_convert(
            "UTC"
        )

    return run_time


def get_gfs_mos(
    run_time,
    station=MOS_STATION
):
    run_time = normalize_run_time(
        run_time
    )

    runtime_string = (
        run_time.strftime(
            "%Y-%m-%d %H:%MZ"
        )
    )

    params = {
        "station": station,
        "runtime": runtime_string,
        "model": MOS_MODEL
    }

    response = requests.get(
        IEM_MOS_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return {
        "run_time": run_time,
        "station": station,
        "url": response.url,
        "data": response.json()
    }


def inspect_gfs_mos(
    run_time,
    station=MOS_STATION
):
    result = get_gfs_mos(
        run_time=run_time,
        station=station
    )

    print()
    print("=" * 70)

    print("GFS MOS INSPECTION")

    print(
        f"Run: "
        f"{result['run_time']:%Y-%m-%d %HZ}"
    )

    print(
        f"Station: "
        f"{station}"
    )

    print(
        f"URL: "
        f"{result['url']}"
    )

    print("=" * 70)

    print(
        json.dumps(
            result["data"],
            indent=2
        )
    )

def extract_mos_rows(
    payload
):
    """
    Return the forecast rows from the
    IEM MOS JSON response.

    Handles either a raw list or a
    dictionary containing a list.
    """

    if isinstance(
        payload,
        list
    ):
        return payload

    if isinstance(
        payload,
        dict
    ):

        for key in [
            "data",
            "results",
            "forecasts"
        ]:

            rows = payload.get(
                key
            )

            if isinstance(
                rows,
                list
            ):
                return rows

    raise ValueError(
        "Could not locate MOS "
        "forecast rows in response."
    )


def parse_utc_time(
    value
):
    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp


def get_mos_high_forecast(
    run_time,
    target_date,
    station=MOS_STATION
):
    """
    Get the GFS MOS maximum temperature
    forecast for one target date.

    This function currently supports our
    D-1 12Z backtest configuration.

    For a 12Z MAV cycle, tomorrow's max
    is represented by n_x at 00Z following
    the target date.
    """

    run_time = normalize_run_time(
        run_time
    )

    target_date = pd.Timestamp(
        target_date
    ).date()

    if run_time.hour != 12:

        raise ValueError(
            "MOS high parser currently "
            "expects a 12Z model run."
        )

    result = get_gfs_mos(
        run_time=run_time,
        station=station
    )

    rows = extract_mos_rows(
        result["data"]
    )

    desired_valid_time = pd.Timestamp(
        target_date
    ).tz_localize(
        "UTC"
    ) + pd.Timedelta(
        days=1
    )

    for row in rows:

        n_x = row.get(
            "n_x"
        )

        if n_x is None:
            continue

        time_value = (
            row.get("ftime_utc")
            or row.get("ftime")
        )

        if time_value is None:
            continue

        valid_time = parse_utc_time(
            time_value
        )

        if valid_time != desired_valid_time:
            continue

        return {
            "target_date":
                target_date,

            "forecast_high":
                float(n_x),

            "run_time":
                run_time,

            "valid_time":
                valid_time,

            "station":
                station,

            "model":
                "GFS",

            "product":
                "MAV"
        }

    raise ValueError(
        f"No GFS MOS max temperature "
        f"found for {target_date} "
        f"from run "
        f"{run_time:%Y-%m-%d %HZ}."
    )


if __name__ == "__main__":

    result = get_mos_high_forecast(
        run_time="2026-09-09 12:00",
        target_date="2026-09-10"
    )

    print(result)