import requests


NWS_BASE_URL = "https://api.weather.gov"

CENTRAL_PARK_LAT = 40.7789
CENTRAL_PARK_LON = -73.9692

HEADERS = {
    "User-Agent": "prediction-market-bot/0.1"
}

def get_nws_forecast(latitude, longitude):
    point_url = f"{NWS_BASE_URL}/points/{latitude},{longitude}"

    point_response = requests.get(
        point_url,
        headers=HEADERS
    )

    point_response.raise_for_status()

    point_data = point_response.json()

    forecast_url = point_data["properties"]["forecast"]

    forecast_response = requests.get(
        forecast_url,
        headers=HEADERS
    )

    forecast_response.raise_for_status()

    forecast_data = forecast_response.json()

    return forecast_data["properties"]["periods"]