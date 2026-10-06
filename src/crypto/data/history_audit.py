import argparse

import pandas as pd
import requests

from src.backtest.hourly_request_limits import (
    hourly_backfill_requests,
)

from src.kalshi.client import (
    get_market_candlesticks,
    get_historical_market_candlesticks
)

from src.kalshi.historical_market import (
    parse_candle,
)


SERIES_TICKER = "KXBTC15M"

BASE_URL = (
    "https://external-api.kalshi.com"
    "/trade-api/v2"
)


# ============================================================
# MARKET LISTING
# ============================================================

def fetch_market_pages(
    historical,
):
    """
    Walk every Kalshi market page for KXBTC15M.

    Older settled markets live in /historical/markets.
    Newer settled markets remain in /markets.
    """

    if historical:

        url = (
            f"{BASE_URL}"
            "/historical/markets"
        )

        base_params = {
            "series_ticker":
                SERIES_TICKER,

            "limit":
                1000,
        }

    else:

        url = (
            f"{BASE_URL}"
            "/markets"
        )

        base_params = {
            "series_ticker":
                SERIES_TICKER,

            "status":
                "settled",

            "limit":
                1000,

            "mve_filter":
                "exclude",
        }

    markets = []

    cursor = None

    page_number = 0

    while True:

        params = dict(
            base_params
        )

        if cursor:

            params[
                "cursor"
            ] = cursor

        response = (
            requests.get(
                url,
                params=params,
                timeout=30,
            )
        )

        response.raise_for_status()

        payload = (
            response.json()
        )

        page = (
            payload.get(
                "markets",
                []
            )
        )

        page_number += 1

        markets.extend(
            page
        )

        print(
            f"{'HISTORICAL' if historical else 'LIVE SETTLED'} "
            f"page {page_number}: "
            f"{len(page)} markets "
            f"(total {len(markets)})"
        )

        cursor = (
            payload.get(
                "cursor"
            )
        )

        if not cursor:

            break

    return markets


# ============================================================
# TIMESTAMP PARSING
# ============================================================

def parse_timestamp(
    value,
):

    if value in (
        None,
        "",
    ):

        return pd.NaT

    return pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
        format="mixed",
    )


# ============================================================
# COMBINE TIERS
# ============================================================

def build_market_dataframe():

    historical = (
        fetch_market_pages(
            historical=True
        )
    )

    live_settled = (
        fetch_market_pages(
            historical=False
        )
    )

    combined = {}

    for source, markets in (
        (
            "HISTORICAL",
            historical,
        ),
        (
            "LIVE_SETTLED",
            live_settled,
        ),
    ):

        for market in markets:

            ticker = (
                str(
                    market.get(
                        "ticker",
                        ""
                    )
                )
                .strip()
            )

            if not ticker:

                continue

            if not ticker.startswith(
                SERIES_TICKER
                +
                "-"
            ):

                raise RuntimeError(
                    f"Unexpected ticker "
                    f"outside {SERIES_TICKER}: "
                    f"{ticker}"
                )

            row = dict(
                market
            )

            row[
                "data_source"
            ] = source

            combined[
                ticker
            ] = row

    if not combined:

        raise RuntimeError(
            f"No settled {SERIES_TICKER} "
            f"markets found."
        )

    dataframe = (
        pd.DataFrame(
            combined.values()
        )
    )

    for column in (
        "open_time",
        "close_time",
        "settlement_ts",
        "created_time",
        "updated_time",
    ):

        if column in dataframe.columns:

            dataframe[
                column
            ] = dataframe[
                column
            ].apply(
                parse_timestamp
            )

    dataframe = (
        dataframe
        .sort_values(
            [
                "close_time",
                "ticker",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        dataframe,
        historical,
        live_settled,
    )


# ============================================================
# CANDLE SAMPLE
# ============================================================

def choose_sample_markets(
    dataframe,
    count,
):
    usable = (
        dataframe[
            dataframe[
                "close_time"
            ].notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if usable.empty:

        return usable

    count = min(
        int(count),
        len(usable),
    )

    if count <= 0:

        return (
            usable.iloc[
                0:0
            ]
        )

    if count == 1:

        indices = [
            len(usable)
            //
            2
        ]

    else:

        indices = [
            round(
                index
                *
                (
                    len(usable)
                    -
                    1
                )
                /
                (
                    count
                    -
                    1
                )
            )
            for index
            in range(
                count
            )
        ]

    indices = sorted(
        set(
            indices
        )
    )

    return (
        usable.iloc[
            indices
        ]
        .copy()
    )

def load_market_candles_for_market(
    market,
    start_time,
    end_time,
):
    ticker = market["ticker"]

    source_hint = str(
        market.get(
            "data_source",
            ""
        )
    )

    start_ts = int(
        start_time.timestamp()
    )

    end_ts = int(
        end_time.timestamp()
    )

    if source_hint == "LIVE_SETTLED":

        endpoints = [
            (
                "LIVE",
                get_market_candlesticks,
                {
                    "series_ticker":
                        SERIES_TICKER,
                },
            ),
            (
                "HISTORICAL",
                get_historical_market_candlesticks,
                {},
            ),
        ]

    else:

        endpoints = [
            (
                "HISTORICAL",
                get_historical_market_candlesticks,
                {},
            ),
            (
                "LIVE",
                get_market_candlesticks,
                {
                    "series_ticker":
                        SERIES_TICKER,
                },
            ),
        ]

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
                error.response is not None
                and
                error.response.status_code
                in (404, 410)
            ):

                continue

            raise

        if candles:

            return {
                "source":
                    source,

                "candles":
                    sorted(
                        candles,
                        key=lambda candle:
                            candle["end_period_ts"],
                    ),
            }

    return {
        "source":
            None,

        "candles":
            [],
    }


def audit_market_candles(
    market,
):
    ticker = (
        market[
            "ticker"
        ]
    )

    close_time = (
        market[
            "close_time"
        ]
    )

    # We care primarily about the 15-minute trading state
    # immediately preceding settlement. Give ourselves a
    # slightly wider window for boundary candles.
    start_time = (
        close_time
        -
        pd.Timedelta(
            minutes=20
        )
    )

    end_time = (
        close_time
        +
        pd.Timedelta(
            minutes=1
        )
    )

    start_ts = int(
        start_time.timestamp()
    )

    end_ts = int(
        end_time.timestamp()
    )

    source_hint = str(
        market.get(
            "data_source",
            ""
        )
    )

    # --------------------------------------------------------
    # Markets can exist in either Kalshi storage tier.
    #
    # Prefer the tier identified during market discovery,
    # but fall back to the other endpoint because Kalshi can
    # move settled markets between tiers over time.
    # --------------------------------------------------------

    if source_hint == "LIVE_SETTLED":

        endpoints = [

            (
                "LIVE",
                get_market_candlesticks,
                {
                    "series_ticker":
                        SERIES_TICKER,
                },
            ),

            (
                "HISTORICAL",
                get_historical_market_candlesticks,
                {},
            ),
        ]

    else:

        endpoints = [

            (
                "HISTORICAL",
                get_historical_market_candlesticks,
                {},
            ),

            (
                "LIVE",
                get_market_candlesticks,
                {
                    "series_ticker":
                        SERIES_TICKER,
                },
            ),
        ]

    candles = []
    candle_source = None

    for (
        source,
        fetch,
        extra,
    ) in endpoints:

        try:

            candidate = (
                fetch(

                    ticker=
                        ticker,

                    start_ts=
                        start_ts,

                    end_ts=
                        end_ts,

                    period_interval=
                        1,

                    **extra,
                )
            )

        except requests.HTTPError as error:

            if (
                error.response is not None
                and
                error.response.status_code
                in (
                    404,
                    410,
                )
            ):

                continue

            raise

        if candidate:

            candles = (
                candidate
            )

            candle_source = (
                source
            )

            break

    valid_quotes = []

    parse_errors = 0

    for candle in candles:

        try:

            quote = (
                parse_candle(
                    candle
                )
            )

        except Exception:

            parse_errors += 1

            continue

        bid = (
            quote[
                "yes_bid"
            ]
        )

        ask = (
            quote[
                "yes_ask"
            ]
        )

        if (
            bid is not None
            and
            ask is not None
            and
            0.0
            <=
            bid
            <=
            ask
            <=
            1.0
        ):

            valid_quotes.append(
                quote
            )

    return {

        "ticker":
            ticker,

        "close_time":
            close_time,

        "result":
            market.get(
                "result"
            ),
        
        "market_source":
            source_hint,

        "candle_source":
            candle_source,

        "raw_candles":
            len(
                candles
            ),

        "valid_quote_candles":
            len(
                valid_quotes
            ),

        "parse_errors":
            parse_errors,

        "first_quote":
            (
                valid_quotes[0][
                    "timestamp"
                ]
                if valid_quotes
                else pd.NaT
            ),

        "last_quote":
            (
                valid_quotes[-1][
                    "timestamp"
                ]
                if valid_quotes
                else pd.NaT
            ),
    }


def audit_candle_sample(
    dataframe,
    sample_count,
):

    sample = (
        choose_sample_markets(
            dataframe=
                dataframe,

            count=
                sample_count,
        )
    )

    results = []

    if sample.empty:

        return (
            pd.DataFrame()
        )

    print()
    print("=" * 100)
    print(
        "1-MINUTE CANDLE COVERAGE SAMPLE"
    )
    print("=" * 100)

    with hourly_backfill_requests(
        interval_seconds=2
    ):

        for _, market in (
            sample.iterrows()
        ):

            try:

                result = (
                    audit_market_candles(
                        market
                    )
                )

            except Exception as error:

                result = {

                    "ticker":
                        market[
                            "ticker"
                        ],

                    "close_time":
                        market[
                            "close_time"
                        ],

                    "result":
                        market.get(
                            "result"
                        ),

                    "raw_candles":
                        0,

                    "valid_quote_candles":
                        0,

                    "parse_errors":
                        0,

                    "first_quote":
                        pd.NaT,

                    "last_quote":
                        pd.NaT,

                    "error":
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                }

            results.append(
                result
            )

            print(
                f"{result['ticker']} | "
                f"close {result['close_time']} | "
                f"market {result.get('market_source')} | "
                f"candles {result['raw_candles']:2d} | "
                f"valid {result['valid_quote_candles']:2d} | "
                f"source {result.get('candle_source')}"
            )

    return (
        pd.DataFrame(
            results
        )
    )


# ============================================================
# SUMMARY
# ============================================================

def print_market_summary(
    dataframe,
    historical_count,
    live_count,
):

    print()
    print("=" * 100)
    print(
        "KXBTC15M HISTORICAL COVERAGE AUDIT"
    )
    print("=" * 100)

    print(
        f"Historical-tier rows: "
        f"{historical_count}"
    )

    print(
        f"Live settled rows:     "
        f"{live_count}"
    )

    print(
        f"Unique markets:        "
        f"{len(dataframe)}"
    )

    valid_close = (
        dataframe[
            "close_time"
        ]
        .dropna()
    )

    if not valid_close.empty:

        print(
            f"Earliest close:        "
            f"{valid_close.min()}"
        )

        print(
            f"Latest close:          "
            f"{valid_close.max()}"
        )

        span_days = (
            valid_close.max()
            -
            valid_close.min()
        ).total_seconds() / 86400

        print(
            f"Calendar span:         "
            f"{span_days:.1f} days"
        )

    print()

    if (
        "result"
        in
        dataframe.columns
    ):

        result_counts = (
            dataframe[
                "result"
            ]
            .fillna(
                "NULL"
            )
            .astype(str)
            .value_counts(
                dropna=False
            )
        )

        print(
            "Settlement results:"
        )

        for (
            result,
            count,
        ) in (
            result_counts.items()
        ):

            print(
                f"  {result:10s} "
                f"{count}"
            )

    print()

    print(
        "Oldest 5:"
    )

    print(
        dataframe[
            [
                "ticker",
                "close_time",
                "result",
                "data_source",
            ]
        ]
        .head(
            5
        )
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Newest 5:"
    )

    print(
        dataframe[
            [
                "ticker",
                "close_time",
                "result",
                "data_source",
            ]
        ]
        .tail(
            5
        )
        .to_string(
            index=False
        )
    )


def print_candle_summary(
    candle_results,
):

    print()
    print("=" * 100)
    print(
        "CANDLE SAMPLE SUMMARY"
    )
    print("=" * 100)

    if candle_results.empty:

        print(
            "No candle sample results."
        )

        return

    total = (
        len(
            candle_results
        )
    )

    usable = (
        candle_results[
            "valid_quote_candles"
        ]
        >
        0
    )

    near_complete = (
        candle_results[
            "valid_quote_candles"
        ]
        >=
        12
    )

    print(
        f"Sampled markets:       "
        f"{total}"
    )

    print(
        f"Markets with quotes:   "
        f"{usable.sum()} "
        f"({usable.mean():.1%})"
    )

    print(
        f"Markets >=12 minutes:  "
        f"{near_complete.sum()} "
        f"({near_complete.mean():.1%})"
    )

    print(
        f"Median valid candles:  "
        f"{candle_results['valid_quote_candles'].median():.1f}"
    )

    print(
        f"Minimum valid candles: "
        f"{candle_results['valid_quote_candles'].min()}"
    )

    print(
        f"Maximum valid candles: "
        f"{candle_results['valid_quote_candles'].max()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--sample-markets",
        type=int,
        default=12,
        help=(
            "Number of markets spread across history "
            "to inspect for 1-minute candle coverage."
        ),
    )

    args = (
        parser.parse_args()
    )

    if (
        args.sample_markets
        <=
        0
    ):

        parser.error(
            "--sample-markets must be positive."
        )

    (
        dataframe,
        historical,
        live_settled,
    ) = (
        build_market_dataframe()
    )

    print_market_summary(

        dataframe=
            dataframe,

        historical_count=
            len(
                historical
            ),

        live_count=
            len(
                live_settled
            ),
    )

    candle_results = (
        audit_candle_sample(

            dataframe=
                dataframe,

            sample_count=
                args.sample_markets,
        )
    )

    print_candle_summary(
        candle_results
    )


if __name__ == "__main__":

    main()