import requests
import pandas as pd
from zoneinfo import ZoneInfo


NWS_BASE_URL = "https://api.weather.gov"
CENTRAL_PARK_STATION = "KNYC"

NYC_TIMEZONE = ZoneInfo(
    "America/New_York"
)

HEADERS = {
    "User-Agent": (
        "prediction-market-bot"
    ),
    "Accept": "application/geo+json"
}


def celsius_to_fahrenheit(
    temperature_c
):
    return (
        temperature_c * 9 / 5
        + 32
    )


def get_station_observations(
    station_id=CENTRAL_PARK_STATION
):
    """
    Retrieve recent observations from an
    NWS weather station.
    """

    url = (
        f"{NWS_BASE_URL}/stations/"
        f"{station_id}/observations"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    observations = []

    for feature in data.get(
        "features",
        []
    ):

        properties = feature.get(
            "properties",
            {}
        )

        timestamp = properties.get(
            "timestamp"
        )

        temperature = properties.get(
            "temperature",
            {}
        ).get(
            "value"
        )

        if (
            timestamp is None
            or temperature is None
        ):
            continue

        timestamp = pd.Timestamp(
            timestamp
        )

        timestamp = timestamp.tz_convert(
            NYC_TIMEZONE
        )

        temperature_f = (
            celsius_to_fahrenheit(
                float(temperature)
            )
        )

        observations.append({
            "time": timestamp,
            "temperature": temperature_f
        })

    observations.sort(
        key=lambda observation:
        observation["time"]
    )

    return observations


def get_observed_high(
    target_date,
    station_id=CENTRAL_PARK_STATION
):
    """
    Return the highest observed temperature
    for target_date so far.
    """

    target_date = pd.Timestamp(
        target_date
    ).date()

    observations = (
        get_station_observations(
            station_id
        )
    )

    daily_observations = [
        observation
        for observation in observations
        if observation[
            "time"
        ].date() == target_date
    ]

    if not daily_observations:
        return None

    maximum = max(
        daily_observations,
        key=lambda observation:
        observation["temperature"]
    )

    latest = daily_observations[-1]

    return {
        "station": station_id,

        "observed_high": maximum[
            "temperature"
        ],

        "high_time": maximum[
            "time"
        ],

        "latest_temperature": latest[
            "temperature"
        ],

        "latest_time": latest[
            "time"
        ],

        "observation_count": len(
            daily_observations
        )
    }