import pandas as pd
import requests

from src.kalshi.client import (
    get_market_candlesticks,
    get_historical_market_candlesticks,
)


def load_market_candles(
    ticker,
    start_time,
    end_time,
    period_interval=60,
):
    if period_interval not in (1, 60):
        raise ValueError("Use 1-minute or 60-minute candles.")

    params = {
        "ticker": ticker,
        "start_ts": int(start_time.timestamp()),
        "end_ts": int(end_time.timestamp()),
        "period_interval": period_interval,
    }

    candles = []

    try:
        candles = get_market_candlesticks(
            series_ticker="KXHIGHNY",
            **params,
        )
    except requests.HTTPError as error:
        if (
            error.response is None
            or error.response.status_code not in (404, 410)
        ):
            raise

    if candles:
        return {
            "source": "LIVE",
            "candles": candles,
        }

    candles = get_historical_market_candlesticks(**params)

    return {
        "source": "HISTORICAL" if candles else None,
        "candles": candles,
    }


def parse_candle(candle):
    def close_price(side):
        data = candle.get(side) or {}
        value = data.get("close_dollars")

        if value in (None, ""):
            value = data.get("close")

            # Legacy integer prices were expressed in cents.
            if isinstance(value, int) and not isinstance(value, bool):
                value = value / 100.0

        price = (
            None
            if value in (None, "")
            else float(value)
        )

        if price is not None and not 0 <= price <= 1:
            raise ValueError(f"Invalid candle price: {value!r}")

        return price

    yes_bid = close_price("yes_bid")
    yes_ask = close_price("yes_ask")

    return {
        "timestamp": pd.Timestamp(
            candle["end_period_ts"],
            unit="s",
            tz="UTC",
        ),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": None if yes_ask is None else 1.0 - yes_ask,
        "no_ask": None if yes_bid is None else 1.0 - yes_bid,
    }