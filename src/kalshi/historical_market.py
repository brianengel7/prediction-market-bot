import requests
import pandas as pd

from src.kalshi.client import (
    get_event_with_markets,
    get_historical_markets,
    get_market_candlesticks,
    get_historical_market_candlesticks
)


SERIES_TICKER = "KXHIGHNY"

DECISION_HOUR = 20
DECISION_MINUTE = 5

SEARCH_MINUTES = 45


# ============================================================
# EVENT TICKER
# ============================================================

def build_event_ticker(
    target_date
):

    target_date = pd.Timestamp(
        target_date
    )

    date_code = (
        target_date.strftime(
            "%y%b%d"
        ).upper()
    )

    return (
        f"{SERIES_TICKER}-"
        f"{date_code}"
    )


# ============================================================
# DECISION TIME
# ============================================================

def get_decision_time(
    target_date,
    decision_hour=20,
    decision_minute=5
):

    target_date = (
        pd.Timestamp(
            target_date
        )
    )

    decision_date = (
        target_date
        - pd.Timedelta(
            days=1
        )
    )

    decision_time = pd.Timestamp(
        year=
            decision_date.year,

        month=
            decision_date.month,

        day=
            decision_date.day,

        hour=
            decision_hour,

        minute=
            decision_minute,

        tz="UTC"
    )

    return (
        decision_time
    )

# ============================================================
# GET EVENT MARKETS
# ============================================================

def get_event_markets(
    target_date
):

    event_ticker = (
        build_event_ticker(
            target_date
        )
    )

    # --------------------------------------------------------
    # First try normal event endpoint.
    #
    # Recent finalized markets may still be available here.
    # --------------------------------------------------------

    event_data = (
        get_event_with_markets(
            event_ticker
        )
    )

    markets = (
        event_data[
            "markets"
        ]
    )

    source = "LIVE"

    # --------------------------------------------------------
    # Older markets eventually move into historical storage.
    # --------------------------------------------------------

    if not markets:

        historical = (
            get_historical_markets(
                event_ticker=
                    event_ticker
            )
        )

        markets = (
            historical[
                "markets"
            ]
        )

        source = (
            "HISTORICAL"
        )

    if not markets:

        raise RuntimeError(
            f"No Kalshi markets found "
            f"for {event_ticker}."
        )

    return {
        "event_ticker":
            event_ticker,

        "source":
            source,

        "markets":
            markets
    }


# ============================================================
# PARSE PRICE
# ============================================================

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


# ============================================================
# PARSE CANDLE
# ============================================================

def parse_candle(
    candle
):

    yes_bid_data = (
        candle.get(
            "yes_bid"
        )
        or {}
    )

    yes_ask_data = (
        candle.get(
            "yes_ask"
        )
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

    # --------------------------------------------------------
    # Binary contract relationships:
    #
    # NO ask = 1 - YES bid
    # NO bid = 1 - YES ask
    # --------------------------------------------------------

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


# ============================================================
# LOAD CANDLES
# ============================================================

def load_market_candles(
    ticker,
    start_time,
    end_time
):

    start_ts = int(
        start_time.timestamp()
    )

    end_ts = int(
        end_time.timestamp()
    )

    candles = []

    source = None

    # --------------------------------------------------------
    # Try normal candlestick endpoint first.
    # --------------------------------------------------------

    try:

        candles = (
            get_market_candlesticks(
                series_ticker=
                    SERIES_TICKER,

                ticker=
                    ticker,

                start_ts=
                    start_ts,

                end_ts=
                    end_ts,

                period_interval=
                    1
            )
        )

        if candles:

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

    # --------------------------------------------------------
    # Fall back to historical storage.
    # --------------------------------------------------------

    if not candles:

        candles = (
            get_historical_market_candlesticks(
                ticker=
                    ticker,

                start_ts=
                    start_ts,

                end_ts=
                    end_ts,

                period_interval=
                    1
            )
        )

        if candles:

            source = (
                "HISTORICAL"
            )

    return {
        "source":
            source,

        "candles":
            candles
    }


# ============================================================
# FIND EXECUTABLE ENTRY QUOTE
# ============================================================

def get_market_entry_quote(
    ticker,
    target_date,
    decision_hour=20,
    decision_minute=5
):

    decision_time = (
        get_decision_time(
            target_date,
            decision_hour=
                decision_hour,
            decision_minute=
                decision_minute
        )
    )

    end_time = (
        decision_time
        + pd.Timedelta(
            minutes=
                SEARCH_MINUTES
        )
    )

    candle_data = (
        load_market_candles(
            ticker=
                ticker,

            start_time=
                decision_time,

            end_time=
                end_time
        )
    )

    parsed = [

        parse_candle(
            candle
        )

        for candle
        in candle_data[
            "candles"
        ]
    ]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Candle must END strictly after the decision timestamp.
    #
    # A 20:05 candle contains information from BEFORE 20:05,
    # so 20:06 is the earliest acceptable complete candle.
    # --------------------------------------------------------

    valid_quotes = [

        quote

        for quote in parsed

        if (
            quote[
                "timestamp"
            ] > decision_time

            and

            quote[
                "yes_bid"
            ] is not None

            and

            quote[
                "yes_ask"
            ] is not None
        )
    ]

    if not valid_quotes:

        return None

    entry = (
        valid_quotes[0]
    )

    entry[
        "source"
    ] = candle_data[
        "source"
    ]

    entry[
        "decision_time"
    ] = decision_time

    return entry


# ============================================================
# GET ALL CONTRACT ENTRY QUOTES FOR ONE DAY
# ============================================================

def get_event_entry_quotes(
    target_date,
    decision_hour=20,
    decision_minute=5
):

    event_info = (
        get_event_markets(
            target_date
        )
    )

    results = []

    for market in event_info[
        "markets"
    ]:

        ticker = market[
            "ticker"
        ]

        quote = (
            get_market_entry_quote(
                ticker=
                    ticker,

                target_date=
                    target_date,

                decision_hour=
                    decision_hour,

                decision_minute=
                    decision_minute
            )
        )

        if quote is None:

            print(
                f"No entry quote "
                f"found for {ticker}."
            )

            continue

        result = {

            "target_date":
                str(
                    pd.Timestamp(
                        target_date
                    ).date()
                ),

            "event_ticker":
                event_info[
                    "event_ticker"
                ],

            "ticker":
                ticker,

            "title":
                market.get(
                    "title"
                ),

            "strike_type":
                market.get(
                    "strike_type"
                ),

            "floor_strike":
                market.get(
                    "floor_strike"
                ),

            "cap_strike":
                market.get(
                    "cap_strike"
                ),

            "market_status":
                market.get(
                    "status"
                ),

            "result":
                market.get(
                    "result"
                ),

            "settlement_value":
                market.get(
                    "settlement_value_dollars"
                ),

            "decision_time":
                quote[
                    "decision_time"
                ],

            "entry_time":
                quote[
                    "timestamp"
                ],

            "yes_bid":
                quote[
                    "yes_bid"
                ],

            "yes_ask":
                quote[
                    "yes_ask"
                ],

            "no_bid":
                quote[
                    "no_bid"
                ],

            "no_ask":
                quote[
                    "no_ask"
                ],

            "market_source":
                event_info[
                    "source"
                ],

            "candle_source":
                quote[
                    "source"
                ]
        }

        results.append(
            result
        )

    return results


# ============================================================
# DISPLAY
# ============================================================

def print_event_entry_quotes(
    results
):

    print()
    print("=" * 90)
    print(
        "HISTORICAL KALSHI ENTRY QUOTES"
    )
    print("=" * 90)

    if not results:

        print(
            "No entry quotes found."
        )

        return

    print(
        f"Target date:   "
        f"{results[0]['target_date']}"
    )

    print(
        f"Event:         "
        f"{results[0]['event_ticker']}"
    )

    print(
        f"Decision time: "
        f"{results[0]['decision_time']}"
    )

    print(
        f"Contracts:     "
        f"{len(results)}"
    )

    print()

    for result in results:

        print(
            result[
                "title"
            ]
        )

        print(
            f"  Ticker:      "
            f"{result['ticker']}"
        )

        print(
            f"  Entry time:  "
            f"{result['entry_time']}"
        )

        print(
            f"  YES:         "
            f"${result['yes_bid']:.2f} / "
            f"${result['yes_ask']:.2f}"
        )

        print(
            f"  NO:          "
            f"${result['no_bid']:.2f} / "
            f"${result['no_ask']:.2f}"
        )

        print(
            f"  Result:      "
            f"{result['result']}"
        )

        print(
            f"  Candle src:  "
            f"{result['candle_source']}"
        )

        print(
            "-" * 90
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    results = (
        get_event_entry_quotes(
            "2026-06-11"
        )
    )

    print_event_entry_quotes(
        results
    )