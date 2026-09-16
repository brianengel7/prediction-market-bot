import requests
import pandas as pd
from zoneinfo import ZoneInfo


NWS_BASE_URL = "https://api.weather.gov"
NYC_TIMEZONE = ZoneInfo(
    "America/New_York"
)

CENTRAL_PARK_LAT = 40.7789
CENTRAL_PARK_LON = -73.9692

HEADERS = {
    "User-Agent": "prediction-market-bot/0.1"
}

def get_nws_forecast(latitude, longitude):
    point_url = f"{NWS_BASE_URL}/points/{latitude},{longitude}"

    point_response = requests.get(
        point_url,
        headers=HEADERS,
        timeout=20
    )

    point_response.raise_for_status()

    point_data = point_response.json()

    forecast_url = point_data["properties"]["forecast"]

    forecast_response = requests.get(
        forecast_url,
        headers=HEADERS,
        timeout=20
    )

    forecast_response.raise_for_status()

    forecast_data = forecast_response.json()

    return forecast_data["properties"]["periods"]

def get_nws_high_forecast(
    target_date,
    latitude=CENTRAL_PARK_LAT,
    longitude=CENTRAL_PARK_LON
):
    """
    Get the NWS daytime high forecast for
    Central Park on target_date.
    """

    target_date = pd.Timestamp(
        target_date
    ).date()

    periods = get_nws_forecast(
        latitude,
        longitude
    )

    for period in periods:

        start_time = pd.Timestamp(
            period["startTime"]
        ).tz_convert(
            NYC_TIMEZONE
        )

        is_daytime = period.get(
            "isDaytime",
            False
        )

        if (
            start_time.date() == target_date
            and is_daytime
        ):

            temperature = float(
                period["temperature"]
            )

            unit = period.get(
                "temperatureUnit",
                "F"
            )

            if unit == "C":
                temperature = (
                    temperature * 9 / 5
                    + 32
                )

            return {
                "temperature": temperature,
                "name": period.get("name"),
                "start_time": start_time,
                "short_forecast": period.get(
                    "shortForecast"
                ),
                "detailed_forecast": period.get(
                    "detailedForecast"
                )
            }

    return None