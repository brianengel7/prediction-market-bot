import pandas as pd
import requests

from src.kalshi.client import (
    get_market_candlesticks,
    get_historical_market_candlesticks
)


SERIES_TICKER = "KXHIGHNY"

TICKER = (
    "KXHIGHNY-26SEP26-T69"
)


def parse_price(
    value
):

    if value in (
        None,
        ""
    ):
        return None

    return float(
        value
    )


def parse_candle(
    candle
):

    yes_bid_data = (
        candle.get("yes_bid")
        or {}
    )

    yes_ask_data = (
        candle.get("yes_ask")
        or {}
    )

    yes_bid = parse_price(
        yes_bid_data.get(
            "close_dollars"
        )
    )

    yes_ask = parse_price(
        yes_ask_data.get(
            "close_dollars"
        )
    )

    # Binary contract identities:
    #
    # NO ask = 1 - YES bid
    # NO bid = 1 - YES ask

    no_ask = (
        1.0 - yes_bid
        if yes_bid is not None
        else None
    )

    no_bid = (
        1.0 - yes_ask
        if yes_ask is not None
        else None
    )

    timestamp = pd.Timestamp(
        candle[
            "end_period_ts"
        ],
        unit="s",
        tz="UTC"
    )

    return {

        "timestamp":
            timestamp,

        "yes_bid":
            yes_bid,

        "yes_ask":
            yes_ask,

        "no_bid":
            no_bid,

        "no_ask":
            no_ask
    }


def main():

    # --------------------------------------------------------
    # Historical strategy decision time
    #
    # Target = Sep 26
    # D-1    = Sep 25
    # --------------------------------------------------------

    decision_time = pd.Timestamp(
        "2026-09-25T20:05:00Z"
    )

    # Give ourselves a 30-minute search window.
    start_time = (
        decision_time
        - pd.Timedelta(
            minutes=5
        )
    )

    end_time = (
        decision_time
        + pd.Timedelta(
            minutes=30
        )
    )

    start_ts = int(
        start_time.timestamp()
    )

    end_ts = int(
        end_time.timestamp()
    )

    # --------------------------------------------------------
    # Fetch 1-minute candles
    # --------------------------------------------------------

    try:

        candles = (
            get_market_candlesticks(
                series_ticker=
                    SERIES_TICKER,

                ticker=
                    TICKER,

                start_ts=
                    start_ts,

                end_ts=
                    end_ts,

                period_interval=
                    1
            )
        )

        source = "LIVE"

    except requests.HTTPError as error:

        if (
            error.response is None
            or
            error.response.status_code
            not in (
                404,
                410
            )
        ):
            raise

        candles = (
            get_historical_market_candlesticks(
                ticker=
                    TICKER,

                start_ts=
                    start_ts,

                end_ts=
                    end_ts,

                period_interval=
                    1
            )
        )

        source = "HISTORICAL"

    print()
    print("=" * 80)
    print(
        "KALSHI HISTORICAL ENTRY TEST"
    )
    print("=" * 80)

    print(
        f"Source:        "
        f"{source}"
    )

    print(
        f"Ticker:        "
        f"{TICKER}"
    )

    print(
        f"Decision time: "
        f"{decision_time}"
    )

    print(
        f"Candles:       "
        f"{len(candles)}"
    )

    # --------------------------------------------------------
    # Parse candles
    # --------------------------------------------------------

    parsed = [
        parse_candle(
            candle
        )
        for candle in candles
    ]

    # --------------------------------------------------------
    # Find FIRST valid quote AFTER decision time
    # --------------------------------------------------------

    valid_quotes = [

        candle

        for candle in parsed

        if (
            candle[
                "timestamp"
            ] > decision_time

            and

            candle[
                "yes_bid"
            ] is not None

            and

            candle[
                "yes_ask"
            ] is not None
        )
    ]

    if not valid_quotes:

        print()
        print(
            "No valid quote found "
            "after decision time."
        )

        return

    entry = valid_quotes[0]

    print()
    print(
        "SELECTED ENTRY QUOTE"
    )
    print("-" * 80)

    print(
        f"Timestamp: "
        f"{entry['timestamp']}"
    )

    print()

    print(
        f"YES bid:   "
        f"${entry['yes_bid']:.2f}"
    )

    print(
        f"YES ask:   "
        f"${entry['yes_ask']:.2f}"
    )

    print()

    print(
        f"NO bid:    "
        f"${entry['no_bid']:.2f}"
    )

    print(
        f"NO ask:    "
        f"${entry['no_ask']:.2f}"
    )


if __name__ == "__main__":

    main()