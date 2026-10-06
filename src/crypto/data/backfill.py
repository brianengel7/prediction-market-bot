import argparse
import math

import pandas as pd

from src.backtest.hourly_request_limits import (
    hourly_backfill_requests,
)

from src.crypto.data.crypto_db import (
    ensure_crypto_schema,
    get_crypto_counts,
    get_finished_crypto_tickers,
    save_crypto_batch,
)

from src.crypto.data.history_audit import (
    SERIES_TICKER,
    build_market_dataframe,
    load_market_candles_for_market,
)


# ============================================================
# HELPERS
# ============================================================

def float_or_none(
    value,
):
    if value in (
        None,
        "",
    ):

        return None

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):

        return None

    return value


def read_dollar_value(
    section,
    field,
):
    """
    Prefer explicit *_dollars fields.

    Plain-string fallback exists for older API responses.
    Numeric non-dollar fields are intentionally not guessed.
    """

    section = (
        section
        or
        {}
    )

    value = (
        section.get(
            f"{field}_dollars"
        )
    )

    if value not in (
        None,
        "",
    ):

        value = float(
            value
        )

    else:

        value = (
            section.get(
                field
            )
        )

        if not isinstance(
            value,
            str,
        ):

            return None

        value = float(
            value
        )

    if not (
        0.0
        <=
        value
        <=
        1.0
    ):

        raise ValueError(
            f"Invalid dollar price: "
            f"{value}"
        )

    return value


def market_outcome(
    market,
):

    result = (
        str(
            market.get(
                "result",
                ""
            )
        )
        .strip()
        .lower()
    )

    if result == "yes":

        return 1

    if result == "no":

        return 0

    return None


# ============================================================
# CANDLE → DATABASE ROW
# ============================================================

def build_minute_row(
    market,
    candle,
    candle_source,
):
    ticker = (
        market[
            "ticker"
        ]
    )

    close_time = (
        pd.Timestamp(
            market[
                "close_time"
            ]
        )
    )

    if close_time.tzinfo is None:

        close_time = (
            close_time.tz_localize(
                "UTC"
            )
        )

    else:

        close_time = (
            close_time.tz_convert(
                "UTC"
            )
        )

    quote_time = (
        pd.Timestamp(
            candle[
                "end_period_ts"
            ],
            unit="s",
            tz="UTC",
        )
    )

    minutes_remaining = (
        (
            close_time
            -
            quote_time
        ).total_seconds()
        /
        60.0
    )

    yes_bid = (
        read_dollar_value(
            candle.get(
                "yes_bid"
            ),
            "close",
        )
    )

    yes_ask = (
        read_dollar_value(
            candle.get(
                "yes_ask"
            ),
            "close",
        )
    )

    if (
        yes_bid is not None
        and
        yes_ask is not None
        and
        not (
            0.0
            <=
            yes_bid
            <=
            yes_ask
            <=
            1.0
        )
    ):

        raise ValueError(
            f"Invalid bid/ask "
            f"{yes_bid}/{yes_ask} "
            f"for {ticker}."
        )

    no_bid = (
        1.0
        -
        yes_ask
        if yes_ask
        is not None
        else None
    )

    no_ask = (
        1.0
        -
        yes_bid
        if yes_bid
        is not None
        else None
    )

    midpoint = (
        (
            yes_bid
            +
            yes_ask
        )
        /
        2.0
        if (
            yes_bid
            is not None
            and
            yes_ask
            is not None
        )
        else None
    )

    spread = (
        yes_ask
        -
        yes_bid
        if (
            yes_bid
            is not None
            and
            yes_ask
            is not None
        )
        else None
    )

    trade_price = (
        read_dollar_value(
            candle.get(
                "price"
            ),
            "close",
        )
    )

    volume = (
        float_or_none(
            candle.get(
                "volume_fp"
            )
            if candle.get(
                "volume_fp"
            )
            is not None
            else
            candle.get(
                "volume"
            )
        )
    )

    open_interest = (
        float_or_none(
            candle.get(
                "open_interest_fp"
            )
            if candle.get(
                "open_interest_fp"
            )
            is not None
            else
            candle.get(
                "open_interest"
            )
        )
    )

    valid_quote = (
        yes_bid is not None
        and
        yes_ask is not None
    )

    # Candle ending exactly at market close is retained
    # for research but is NOT considered executable.
    actionable = (
        valid_quote
        and
        minutes_remaining > 0
        and
        minutes_remaining <= 15
    )

    return {

        "ticker":
            ticker,

        "quote_time":
            quote_time.isoformat(),

        "minutes_remaining":
            float(
                minutes_remaining
            ),

        "actionable":
            int(
                actionable
            ),

        "yes_bid":
            yes_bid,

        "yes_ask":
            yes_ask,

        "no_bid":
            no_bid,

        "no_ask":
            no_ask,

        "midpoint":
            midpoint,

        "spread":
            spread,

        "trade_price":
            trade_price,

        "volume":
            volume,

        "open_interest":
            open_interest,

        "outcome":
            market_outcome(
                market
            ),

        "candle_source":
            candle_source,
    }


# ============================================================
# ONE MARKET
# ============================================================

def backfill_market(
    market,
):

    close_time = (
        pd.Timestamp(
            market[
                "close_time"
            ]
        )
    )

    if close_time.tzinfo is None:

        close_time = (
            close_time.tz_localize(
                "UTC"
            )
        )

    else:

        close_time = (
            close_time.tz_convert(
                "UTC"
            )
        )

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

    data = (
        load_market_candles_for_market(

            market=
                market,

            start_time=
                start_time,

            end_time=
                end_time,
        )
    )

    minute_rows = []

    parse_errors = []

    for candle in data[
        "candles"
    ]:

        try:

            row = (
                build_minute_row(

                    market=
                        market,

                    candle=
                        candle,

                    candle_source=
                        data[
                            "source"
                        ],
                )
            )

        except Exception as error:

            parse_errors.append(
                str(
                    error
                )
            )

            continue

        # Retain only the market's relevant 15-minute
        # window plus the exact close candle.
        if (
            0.0
            <=
            row[
                "minutes_remaining"
            ]
            <=
            15.0
        ):

            minute_rows.append(
                row
            )

    valid_quotes = sum(
        row[
            "yes_bid"
        ]
        is not None
        and
        row[
            "yes_ask"
        ]
        is not None
        for row in minute_rows
    )

    actionable_quotes = sum(
        row[
            "actionable"
        ]
        for row in minute_rows
    )

    status = (
        "COLLECTED"
        if minute_rows
        else
        "NO_CANDLES"
    )

    error_text = (
        "; ".join(
            parse_errors[
                :5
            ]
        )
        if parse_errors
        else None
    )

    outcome = (
        market_outcome(
            market
        )
    )

    market_row = {

        "ticker":
            market[
                "ticker"
            ],

        "event_ticker":
            market.get(
                "event_ticker"
            ),

        "series_ticker":
            SERIES_TICKER,

        "title":
            market.get(
                "title"
            ),

        "subtitle":
            market.get(
                "subtitle"
            ),

        "strike_type":
            market.get(
                "strike_type"
            ),

        "floor_strike":
            float_or_none(
                market.get(
                    "floor_strike"
                )
            ),

        "cap_strike":
            float_or_none(
                market.get(
                    "cap_strike"
                )
            ),

        "open_time":
            (
                str(
                    market.get(
                        "open_time"
                    )
                )
                if market.get(
                    "open_time"
                )
                else None
            ),

        "close_time":
            close_time.isoformat(),

        "settlement_time":
            (
                str(
                    market.get(
                        "settlement_ts"
                    )
                )
                if market.get(
                    "settlement_ts"
                )
                else None
            ),

        "market_status":
            market.get(
                "status"
            ),

        "result":
            market.get(
                "result"
            ),

        "outcome":
            outcome,

        "settlement_value":
            float_or_none(
                market.get(
                    "settlement_value_dollars"
                )
            ),

        "market_source":
            market.get(
                "data_source"
            ),

        "candle_source":
            data[
                "source"
            ],

        "raw_candle_count":
            len(
                data[
                    "candles"
                ]
            ),

        "valid_quote_count":
            int(
                valid_quotes
            ),

        "actionable_quote_count":
            int(
                actionable_quotes
            ),

        "backfill_status":
            status,

        "backfill_error":
            error_text,

        "backfilled_at":
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat(),
    }

    return (
        market_row,
        minute_rows,
    )


# ============================================================
# MAIN BACKFILL
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of unfinished markets "
            "to collect. Useful for pilot runs."
        ),
    )

    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--retry-missing",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if (
        args.limit is not None
        and
        args.limit <= 0
    ):

        parser.error(
            "--limit must be positive."
        )

    if (
        args.interval_seconds
        <= 0
    ):

        parser.error(
            "--interval-seconds must be positive."
        )

    if (
        args.batch_size
        <= 0
    ):

        parser.error(
            "--batch-size must be positive."
        )

    ensure_crypto_schema()

    (
        dataframe,
        historical,
        live_settled,
    ) = (
        build_market_dataframe()
    )

    finished = (
        get_finished_crypto_tickers(
            retry_missing=
                args.retry_missing
        )
    )

    dataframe = (
        dataframe[
            ~dataframe[
                "ticker"
            ].isin(
                finished
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if args.limit is not None:

        dataframe = (
            dataframe
            .head(
                args.limit
            )
            .copy()
        )

    print()
    print("=" * 100)
    print(
        "KXBTC15M SUPABASE BACKFILL"
    )
    print("=" * 100)

    print(
        f"Discovered markets: "
        f"{len(historical) + len(live_settled)}"
    )

    print(
        f"Unique markets:     "
        f"{27993 if len(dataframe) == 0 else 'see discovery above'}"
    )

    print(
        f"Already finished:   "
        f"{len(finished)}"
    )

    print(
        f"Queued this run:     "
        f"{len(dataframe)}"
    )

    print(
        f"Request interval:    "
        f"{args.interval_seconds:.2f}s"
    )

    print()

    market_batch = []
    minute_batch = []

    collected = 0
    missing = 0
    failures = 0

    def flush():

        nonlocal market_batch
        nonlocal minute_batch

        if not market_batch:

            return

        result = (
            save_crypto_batch(
                market_rows=
                    market_batch,

                minute_rows=
                    minute_batch,
            )
        )

        print(
            f"DB BATCH SAVED | "
            f"markets={result['markets']} | "
            f"minutes={result['minutes']}",
            flush=True,
        )

        market_batch = []
        minute_batch = []

    with hourly_backfill_requests(
        interval_seconds=
            args.interval_seconds,

        max_attempts=8,
    ):

        for index, market in (
            dataframe.iterrows()
        ):

            ticker = (
                market[
                    "ticker"
                ]
            )

            try:

                (
                    market_row,
                    minute_rows,
                ) = (
                    backfill_market(
                        market
                    )
                )

            except Exception as error:

                failures += 1

                print(
                    f"FAILED "
                    f"{ticker}: "
                    f"{type(error).__name__}: "
                    f"{error}",
                    flush=True,
                )

                continue

            if (
                market_row[
                    "backfill_status"
                ]
                ==
                "NO_CANDLES"
            ):

                missing += 1

            else:

                collected += 1

            market_batch.append(
                market_row
            )

            minute_batch.extend(
                minute_rows
            )

            print(
                f"{index + 1:5d}/"
                f"{len(dataframe):5d} | "
                f"{ticker} | "
                f"{market_row['candle_source']} | "
                f"candles "
                f"{market_row['raw_candle_count']:2d} | "
                f"stored "
                f"{len(minute_rows):2d} | "
                f"actionable "
                f"{market_row['actionable_quote_count']:2d}",
                flush=True,
            )

            if (
                len(
                    market_batch
                )
                >=
                args.batch_size
            ):

                flush()

    flush()

    counts = (
        get_crypto_counts()
    )

    print()
    print("=" * 100)
    print(
        "BACKFILL SUMMARY"
    )
    print("=" * 100)

    print(
        f"Collected this run:     "
        f"{collected}"
    )

    print(
        f"No candles this run:    "
        f"{missing}"
    )

    print(
        f"Failures this run:      "
        f"{failures}"
    )

    print()

    print(
        f"Database markets:       "
        f"{counts['markets']}"
    )

    print(
        f"Complete markets:       "
        f"{counts['complete_markets']}"
    )

    print(
        f"Market-minute rows:     "
        f"{counts['minutes']}"
    )

    print(
        f"Actionable minute rows: "
        f"{counts['actionable']}"
    )


if __name__ == "__main__":

    main()