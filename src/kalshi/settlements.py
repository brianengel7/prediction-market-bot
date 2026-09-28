from src.kalshi.client import (
    get_markets,
    get_market
)


SERIES_TICKER = "KXHIGHNY"


# ============================================================
# PARSE OFFICIAL SETTLEMENT
# ============================================================

def parse_market_settlement(
    market
):

    status = market.get(
        "status"
    )

    result = market.get(
        "result"
    )

    settlement_value = market.get(
        "settlement_value_dollars"
    )

    settlement_time = market.get(
        "settlement_ts"
    )

    # --------------------------------------------------------
    # Determine YES payout
    # --------------------------------------------------------

    yes_payout = None

    # Best source: explicit settlement value
    if settlement_value not in (
        None,
        ""
    ):

        yes_payout = float(
            settlement_value
        )

    # Fallback for normal binary markets
    elif result == "yes":

        yes_payout = 1.0

    elif result == "no":

        yes_payout = 0.0

    # --------------------------------------------------------
    # Market is considered settled only when it is finalized
    # and we know the actual payout.
    # --------------------------------------------------------

    is_settled = (
        status in (
            "finalized",
            "settled"
        )
        and
        yes_payout is not None
    )

    if is_settled:

        no_payout = (
            1.0
            - yes_payout
        )

    else:

        no_payout = None

    return {

        "ticker":
            market["ticker"],

        "event_ticker":
            market["event_ticker"],

        "title":
            market.get(
                "title"
            ),

        "status":
            status,

        "result":
            result,

        "is_settled":
            is_settled,

        "settlement_value":
            yes_payout,

        "yes_payout":
            yes_payout,

        "no_payout":
            no_payout,

        "settlement_time":
            settlement_time
    }

# ============================================================
# GET SETTLEMENT FOR ONE MARKET
# ============================================================

def get_market_settlement(
    ticker
):

    market = get_market(
        ticker
    )

    return parse_market_settlement(
        market
    )


# ============================================================
# FIND RECENT SETTLED WEATHER MARKET
# ============================================================

def get_latest_settled_weather_market():

    market_data = get_markets(
        series_ticker=
            SERIES_TICKER,

        status=
            "settled"
    )

    markets = market_data[
        "markets"
    ]

    if not markets:

        raise RuntimeError(
            "No settled KXHIGHNY "
            "markets found."
        )

    markets.sort(
        key=lambda market:
            (
                market.get(
                    "settlement_ts"
                )
                or
                market.get(
                    "close_time"
                )
                or
                ""
            ),
        reverse=True
    )

    ticker = markets[0][
        "ticker"
    ]

    # Fetch it through the individual
    # market endpoint too.
    return get_market_settlement(
        ticker
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    settlement = (
        get_latest_settled_weather_market()
    )

    print()
    print("=" * 72)
    print(
        "LATEST SETTLED KXHIGHNY MARKET"
    )
    print("=" * 72)

    print(
        f"Ticker:          "
        f"{settlement['ticker']}"
    )

    print(
        f"Event:           "
        f"{settlement['event_ticker']}"
    )

    print(
        f"Title:           "
        f"{settlement['title']}"
    )

    print(
        f"Status:          "
        f"{settlement['status']}"
    )

    print(
        f"Result:          "
        f"{settlement['result']}"
    )

    print(
        f"Settled:         "
        f"{settlement['is_settled']}"
    )

    print(
        f"YES payout:      "
        f"{settlement['yes_payout']}"
    )

    print(
        f"NO payout:       "
        f"{settlement['no_payout']}"
    )

    print(
        f"Settlement time: "
        f"{settlement['settlement_time']}"
    )