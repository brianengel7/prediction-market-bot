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

def get_historical_cutoff():

    response = requests.get(
        f"{BASE_URL}/historical/cutoff",
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def get_market_candlesticks(
    series_ticker,
    ticker,
    start_ts,
    end_ts,
    period_interval=60
):

    if period_interval not in (
        1,
        60,
        1440
    ):

        raise ValueError(
            "period_interval must be "
            "1, 60, or 1440."
        )

    response = requests.get(
        (
            f"{BASE_URL}/series/"
            f"{series_ticker}/markets/"
            f"{ticker}/candlesticks"
        ),
        params={
            "start_ts":
                int(start_ts),

            "end_ts":
                int(end_ts),

            "period_interval":
                period_interval
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json().get(
        "candlesticks",
        []
    )


def get_historical_market_candlesticks(
    ticker,
    start_ts,
    end_ts,
    period_interval=60
):

    if period_interval not in (
        1,
        60,
        1440
    ):

        raise ValueError(
            "period_interval must be "
            "1, 60, or 1440."
        )

    response = requests.get(
        (
            f"{BASE_URL}/historical/"
            f"markets/{ticker}/candlesticks"
        ),
        params={
            "start_ts":
                int(start_ts),

            "end_ts":
                int(end_ts),

            "period_interval":
                period_interval
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return response.json().get(
        "candlesticks",
        []
    )

def get_event_with_markets(
    event_ticker
):

    response = requests.get(
        (
            f"{BASE_URL}/events/"
            f"{event_ticker}"
        ),
        params={
            "with_nested_markets":
                "true"
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    event = data[
        "event"
    ]

    markets = (
        event.get(
            "markets"
        )
        or
        data.get(
            "markets",
            []
        )
    )

    return {
        "event":
            event,

        "markets":
            markets
    }


def get_historical_markets(
    event_ticker=None,
    series_ticker=None
):

    url = (
        f"{BASE_URL}/historical/"
        f"markets"
    )

    all_markets = []
    cursor = None

    while True:

        params = {
            "event_ticker":
                event_ticker,

            "series_ticker":
                series_ticker,

            "limit":
                1000
        }

        if cursor:

            params[
                "cursor"
            ] = cursor

        params = {
            key: value
            for key, value
            in params.items()
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
        "markets":
            all_markets
    }