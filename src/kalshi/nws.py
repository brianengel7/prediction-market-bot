import requests


BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


def get_markets(series_ticker=None, status=None):
    url = f"{BASE_URL}/markets"

    params = {
        "series_ticker": series_ticker,
        "status": status
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    return data

def get_market(ticker):
    url = f"{BASE_URL}/markets/{ticker}"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    return data["market"]

def get_series(series_ticker):
    url = f"{BASE_URL}/series/{series_ticker}"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    return data["series"]