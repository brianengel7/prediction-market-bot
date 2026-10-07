import argparse
import time

from pathlib import Path

import pandas as pd
import requests

from dotenv import (
    load_dotenv,
)


load_dotenv(
    dotenv_path=(
        Path(
            __file__
        )
        .resolve()
        .parents[3]
        /
        ".env"
    ),
    override=False,
)


from src.crypto.data.btc_spot_db import (
    SOURCE,
    PRODUCT_ID,
    ensure_btc_spot_schema,
    get_completed_spot_days,
    get_required_spot_range,
    get_spot_counts,
    mark_spot_day_error,
    save_spot_day,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://api.exchange.coinbase.com"
)

GRANULARITY_SECONDS = 60

# Coinbase allows at most 300 candles/request.
# Stay below that limit deliberately.
CHUNK_MINUTES = 240

EXPECTED_MINUTES_PER_DAY = 1440

DEFAULT_INTERVAL_SECONDS = 0.40

MAX_ATTEMPTS = 6

REQUEST_TIMEOUT_SECONDS = 30


# ============================================================
# TIME HELPERS
# ============================================================

def normalize_utc_day(
    value,
):

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return timestamp.normalize()


def to_coinbase_timestamp(
    value,
):

    timestamp = pd.Timestamp(
        value
    )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return (
        timestamp
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


# ============================================================
# HTTP
# ============================================================

def fetch_candle_chunk(
    session,
    start_time,
    end_time,
):

    url = (
        f"{BASE_URL}"
        f"/products/"
        f"{PRODUCT_ID}"
        f"/candles"
    )

    params = {
        "granularity":
            GRANULARITY_SECONDS,

        "start":
            to_coinbase_timestamp(
                start_time
            ),

        "end":
            to_coinbase_timestamp(
                end_time
            ),
    }

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=
                    REQUEST_TIMEOUT_SECONDS,
            )

        except requests.RequestException as error:

            last_error = error

            if attempt >= MAX_ATTEMPTS:

                raise RuntimeError(
                    "Coinbase candle request "
                    "failed after retries."
                ) from error

            delay = min(
                2 ** (
                    attempt - 1
                ),
                30,
            )

            time.sleep(
                delay
            )

            continue

        if response.status_code == 200:

            payload = (
                response.json()
            )

            if not isinstance(
                payload,
                list,
            ):

                raise RuntimeError(
                    "Unexpected Coinbase "
                    "candle response."
                )

            break

        if (
            response.status_code == 429
            or
            response.status_code >= 500
        ):

            last_error = RuntimeError(
                f"HTTP "
                f"{response.status_code}: "
                f"{response.text}"
            )

            if attempt >= MAX_ATTEMPTS:

                raise last_error

            retry_after = (
                response.headers.get(
                    "Retry-After"
                )
            )

            try:

                delay = float(
                    retry_after
                )

            except (
                TypeError,
                ValueError,
            ):

                delay = min(
                    2 ** (
                        attempt - 1
                    ),
                    30,
                )

            time.sleep(
                max(
                    delay,
                    1.0,
                )
            )

            continue

        raise RuntimeError(
            "Coinbase candle request failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    else:

        raise RuntimeError(
            "Coinbase candle request "
            "failed."
        ) from last_error

    fetched_at = (
        pd.Timestamp.now(
            tz="UTC"
        )
        .isoformat()
    )

    start_time = pd.Timestamp(
        start_time
    )

    end_time = pd.Timestamp(
        end_time
    )

    candles = []

    for item in payload:

        if (
            not isinstance(
                item,
                list,
            )
            or len(
                item
            ) < 6
        ):

            raise RuntimeError(
                "Unexpected Coinbase "
                "candle structure."
            )

        (
            timestamp_seconds,
            low,
            high,
            open_price,
            close_price,
            volume,
        ) = item[
            :6
        ]

        candle_start = pd.Timestamp(
            int(
                timestamp_seconds
            ),
            unit="s",
            tz="UTC",
        )

        # Coinbase documents that responses can contain
        # candles preceding the requested start.
        if not (
            start_time
            <=
            candle_start
            <
            end_time
        ):

            continue

        candle_end = (
            candle_start
            +
            pd.Timedelta(
                minutes=1
            )
        )

        candles.append(
            {
                "source":
                    SOURCE,

                "product_id":
                    PRODUCT_ID,

                "candle_start":
                    candle_start.isoformat(),

                "candle_end":
                    candle_end.isoformat(),

                "open":
                    float(
                        open_price
                    ),

                "high":
                    float(
                        high
                    ),

                "low":
                    float(
                        low
                    ),

                "close":
                    float(
                        close_price
                    ),

                "volume":
                    float(
                        volume
                    ),

                "fetched_at":
                    fetched_at,
            }
        )

    return candles


# ============================================================
# FETCH ONE UTC DAY
# ============================================================

def fetch_utc_day(
    session,
    utc_day,
    interval_seconds,
):

    day_start = normalize_utc_day(
        utc_day
    )

    day_end = (
        day_start
        +
        pd.Timedelta(
            days=1
        )
    )

    by_timestamp = {}

    chunk_start = (
        day_start
    )

    while chunk_start < day_end:

        chunk_end = min(
            chunk_start
            +
            pd.Timedelta(
                minutes=
                    CHUNK_MINUTES
            ),

            day_end,
        )

        rows = (
            fetch_candle_chunk(
                session=
                    session,

                start_time=
                    chunk_start,

                end_time=
                    chunk_end,
            )
        )

        for row in rows:

            by_timestamp[
                row[
                    "candle_start"
                ]
            ] = row

        chunk_start = (
            chunk_end
        )

        if chunk_start < day_end:

            time.sleep(
                interval_seconds
            )

    rows = sorted(
        by_timestamp.values(),
        key=lambda row:
            row[
                "candle_start"
            ],
    )

    return rows


# ============================================================
# DEFAULT RANGE
# ============================================================

def get_default_range():

    (
        earliest_market_close,
        latest_market_close,
    ) = (
        get_required_spot_range()
    )

    # Include a full UTC day before the first Kalshi
    # market so lagged 30/60 minute features are
    # available immediately.
    start_day = (
        earliest_market_close
        .normalize()
        -
        pd.Timedelta(
            days=1
        )
    )

    end_day = (
        latest_market_close
        .normalize()
    )

    return (
        start_day,
        end_day,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        help=(
            "Optional first UTC date "
            "(YYYY-MM-DD)."
        ),
    )

    parser.add_argument(
        "--end-date",
        help=(
            "Optional final UTC date "
            "(YYYY-MM-DD)."
        ),
    )

    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=
            DEFAULT_INTERVAL_SECONDS,
        help=(
            "Delay between Coinbase "
            "requests."
        ),
    )

    parser.add_argument(
        "--limit-days",
        type=int,
        help=(
            "Optional number of pending "
            "UTC days to process."
        ),
    )

    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help=(
            "Retry days previously collected "
            "with one or more missing candles."
        ),
    )

    args = parser.parse_args()

    if args.interval_seconds < 0:

        parser.error(
            "--interval-seconds "
            "cannot be negative."
        )

    if (
        args.limit_days is not None
        and args.limit_days <= 0
    ):

        parser.error(
            "--limit-days must be positive."
        )

    ensure_btc_spot_schema()

    (
        default_start,
        default_end,
    ) = (
        get_default_range()
    )

    start_day = (
        normalize_utc_day(
            args.start_date
        )
        if args.start_date
        else default_start
    )

    end_day = (
        normalize_utc_day(
            args.end_date
        )
        if args.end_date
        else default_end
    )

    if start_day > end_day:

        parser.error(
            "Start date must not follow "
            "end date."
        )

    completed_days = (
        get_completed_spot_days(
            require_complete=
                args.retry_incomplete
        )
    )

    requested_days = list(
        pd.date_range(
            start=
                start_day,

            end=
                end_day,

            freq="D",
        )
    )

    pending_days = [
        day
        for day in requested_days
        if (
            day.strftime(
                "%Y-%m-%d"
            )
            not in
            completed_days
        )
    ]

    if args.limit_days is not None:

        pending_days = (
            pending_days[
                :args.limit_days
            ]
        )

    print()
    print("=" * 90)
    print(
        "BTC SPOT 1-MINUTE BACKFILL"
    )
    print("=" * 90)

    print(
        f"Source:          "
        f"{SOURCE}"
    )

    print(
        f"Product:         "
        f"{PRODUCT_ID}"
    )

    print(
        f"Requested range: "
        f"{start_day.date()} "
        f"-> "
        f"{end_day.date()}"
    )

    print(
        f"Total UTC days:  "
        f"{len(requested_days)}"
    )

    print(
        f"Already done:    "
        f"{len(requested_days) - len(pending_days)}"
    )

    print(
        f"Pending this run:"
        f" {len(pending_days)}"
    )

    print()

    if not pending_days:

        print(
            "Nothing to backfill."
        )

        print(
            get_spot_counts()
        )

        return

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent":
                "prediction-market-bot/1.0",

            "Accept":
                "application/json",
        }
    )

    collected = 0
    failed = 0
    total_rows = 0
    total_missing = 0

    for index, utc_day in enumerate(
        pending_days,
        start=1,
    ):

        date_string = (
            utc_day.strftime(
                "%Y-%m-%d"
            )
        )

        print(
            f"[{index:>3}/"
            f"{len(pending_days):>3}] "
            f"{date_string}: ",
            end="",
            flush=True,
        )

        try:

            rows = (
                fetch_utc_day(
                    session=
                        session,

                    utc_day=
                        utc_day,

                    interval_seconds=
                        args.interval_seconds,
                )
            )

            result = (
                save_spot_day(
                    rows=
                        rows,

                    utc_date=
                        utc_day,

                    expected_minutes=
                        EXPECTED_MINUTES_PER_DAY,
                )
            )

            collected += 1

            total_rows += (
                result[
                    "row_count"
                ]
            )

            total_missing += (
                result[
                    "missing_count"
                ]
            )

            print(
                f"COLLECTED "
                f"{result['row_count']}/"
                f"{EXPECTED_MINUTES_PER_DAY}"
                f" candles",
                end="",
            )

            if (
                result[
                    "missing_count"
                ]
                >
                0
            ):

                print(
                    f" "
                    f"(missing "
                    f"{result['missing_count']})",
                    end="",
                )

            print()

        except Exception as error:

            failed += 1

            mark_spot_day_error(
                utc_date=
                    utc_day,

                error_message=
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
            )

            print(
                "FAILED"
            )

            print(
                f"    "
                f"{type(error).__name__}: "
                f"{error}"
            )

        if index < len(
            pending_days
        ):

            time.sleep(
                args.interval_seconds
            )

    print()
    print("=" * 90)
    print(
        "BTC SPOT BACKFILL SUMMARY"
    )
    print("=" * 90)

    print(
        f"Collected days:   "
        f"{collected}"
    )

    print(
        f"Failed days:      "
        f"{failed}"
    )

    print(
        f"Rows this run:    "
        f"{total_rows}"
    )

    print(
        f"Missing minutes:  "
        f"{total_missing}"
    )

    print()

    print(
        "Database:"
    )

    print(
        get_spot_counts()
    )


if __name__ == "__main__":

    main()