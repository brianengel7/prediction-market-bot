import requests


BASE_URL = (
    "https://external-api.kalshi.com/"
    "trade-api/v2"
)

REQUEST_TIMEOUT = 20


def get_markets(
    series_ticker=None,
    status=None
):
    url = f"{BASE_URL}/markets"

    all_markets = []
    cursor = None

    while True:

        params = {
            "series_ticker":
                series_ticker,
            "status":
                status,
            "limit":
                1000
        }

        if cursor:
            params["cursor"] = cursor

        params = {
            key: value
            for key, value in params.items()
            if value is not None
        }

        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        all_markets.extend(
            data.get(
                "markets",
                []
            )
        )

        cursor = data.get(
            "cursor"
        )

        if not cursor:
            break

    return {
        "markets": all_markets
    }


def get_market(ticker):
    response = requests.get(
        f"{BASE_URL}/markets/{ticker}",
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()[
        "market"
    ]


def get_series(series_ticker):
    response = requests.get(
        f"{BASE_URL}/series/{series_ticker}",
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()[
        "series"
    ]


def get_event(event_ticker):
    response = requests.get(
        f"{BASE_URL}/events/{event_ticker}",
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()[
        "event"
    ]