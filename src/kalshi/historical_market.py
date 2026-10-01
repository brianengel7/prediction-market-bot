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

def parse_candle(candle):
    def read_close(prices):
        value = prices.get("close_dollars")
        if value in (None, ""):
            value = prices.get("close")
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    "Unexpected numeric close; verify its price units."
                )

        price = parse_price(value)
        if price is not None and not 0.0 <= price <= 1.0:
            raise ValueError(f"Invalid dollar quote: {value!r}")
        return price

    yes_bid = read_close(candle.get("yes_bid") or {})
    yes_ask = read_close(candle.get("yes_ask") or {})

    return {
        "timestamp": pd.Timestamp(
            candle["end_period_ts"], unit="s", tz="UTC"
        ),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": 1.0 - yes_ask if yes_ask is not None else None,
        "no_ask": 1.0 - yes_bid if yes_bid is not None else None,
    }


def load_market_candles(ticker, start_time, end_time):
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())

    endpoints = (
        ("LIVE", get_market_candlesticks, {"series_ticker": SERIES_TICKER}),
        ("HISTORICAL", get_historical_market_candlesticks, {}),
    )

    for source, fetch, extra in endpoints:
        try:
            candles = fetch(
                ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1,
                **extra,
            )
        except requests.HTTPError as error:
            if (
                error.response is None
                or error.response.status_code not in (404, 410)
            ):
                raise
            continue

        if candles:
            return {
                "source": source,
                "candles": sorted(
                    candles,
                    key=lambda candle: candle["end_period_ts"],
                ),
            }

    return {"source": None, "candles": []}

# ============================================================
# FIND EXECUTABLE ENTRY QUOTE
# ============================================================

def get_market_entry_quote(
    ticker, target_date, decision_hour=20, decision_minute=5,
    calibration_quotes=False,
):
    decision = get_decision_time(
        target_date, decision_hour=decision_hour,
        decision_minute=decision_minute,
    )
    if calibration_quotes:
        start = decision - pd.Timedelta(minutes=30)
        end = decision
    else:
        start = decision
        end = decision + pd.Timedelta(minutes=SEARCH_MINUTES)

    data = load_market_candles(
        ticker=ticker, start_time=start, end_time=end,
    )
    valid = []
    for candle in data["candles"]:
        quote = parse_candle(candle)
        bid, ask = quote["yes_bid"], quote["yes_ask"]
        if bid is None or ask is None or not 0 <= bid <= ask <= 1:
            continue

        timestamp = quote["timestamp"]
        if calibration_quotes:
            acceptable = (
                decision - pd.Timedelta(minutes=2)
                <= timestamp <= decision
            )
        else:
            acceptable = decision < timestamp <= end

        if acceptable:
            valid.append(quote)

    if not valid:
        return None

    choose = max if calibration_quotes else min
    quote = choose(valid, key=lambda q: q["timestamp"]).copy()
    quote["source"] = data["source"]
    quote["decision_time"] = decision
    return quote


# ============================================================
# GET ALL CONTRACT ENTRY QUOTES FOR ONE DAY
# ============================================================

def get_event_entry_quotes(
    target_date,
    decision_hour=20,
    decision_minute=5,
    calibration_quotes=False,
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

        quote = get_market_entry_quote(
            ticker=ticker,
            target_date=target_date,
            decision_hour=decision_hour,
            decision_minute=decision_minute,
            calibration_quotes=calibration_quotes,
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