import argparse
import time
from datetime import datetime, timedelta, timezone

import requests

from src.database.db import get_connection


BITSTAMP_URL = (
    "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
)

CRYPTO_COM_URL = (
    "https://api.crypto.com/"
    "exchange/v1/public/get-candlestick"
)

CRYPTO_COM_SYMBOL = "BTC_USD"

REQUEST_TIMEOUT = 30

# Keep calls conservative.
REQUEST_SLEEP_SECONDS = 0.20


# ============================================================
# DATABASE
# ============================================================

def initialize_table():

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
                cross_exchange_spot_minutes (

                exchange TEXT NOT NULL,
                product_id TEXT NOT NULL,

                candle_start TIMESTAMPTZ NOT NULL,
                candle_end TIMESTAMPTZ NOT NULL,

                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,

                volume DOUBLE PRECISION,

                collected_at TIMESTAMPTZ NOT NULL,

                PRIMARY KEY (
                    exchange,
                    product_id,
                    candle_start
                )
            )
            """
        )


def save_rows(
    rows,
):

    if not rows:
        return 0

    saved = 0

    with get_connection() as connection:

        for row in rows:

            connection.execute(
                """
                INSERT INTO
                    cross_exchange_spot_minutes (

                        exchange,
                        product_id,

                        candle_start,
                        candle_end,

                        open,
                        high,
                        low,
                        close,

                        volume,

                        collected_at
                    )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )

                ON CONFLICT (
                    exchange,
                    product_id,
                    candle_start
                )

                DO UPDATE SET

                    candle_end =
                        EXCLUDED.candle_end,

                    open =
                        EXCLUDED.open,

                    high =
                        EXCLUDED.high,

                    low =
                        EXCLUDED.low,

                    close =
                        EXCLUDED.close,

                    volume =
                        EXCLUDED.volume,

                    collected_at =
                        EXCLUDED.collected_at
                """,
                (
                    row["exchange"],
                    row["product_id"],

                    row["candle_start"],
                    row["candle_end"],

                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],

                    row["volume"],

                    row["collected_at"],
                ),
            )

            saved += 1

    return saved


# ============================================================
# HELPERS
# ============================================================

def utc_datetime(
    value,
):

    value = value.replace(
        second=0,
        microsecond=0,
    )

    if value.tzinfo is None:

        value = value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


def iso(
    value,
):

    return (
        utc_datetime(value)
        .isoformat()
    )


def float_or_none(
    value,
):

    if value is None:
        return None

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# BITSTAMP
# ============================================================

def fetch_bitstamp(
    start,
    end,
):

    start = utc_datetime(start)
    end = utc_datetime(end)

    params = {
        "step":
            60,

        # Bitstamp may return a surrounding window.
        # Request more than we need, then filter ourselves.
        "limit":
            1000,

        "start":
            int(
                start.timestamp()
            ),

        "end":
            int(
                end.timestamp()
            ),

        "exclude_current_candle":
            "true",
    }

    response = requests.get(
        BITSTAMP_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    candles = (
        payload
        .get("data", {})
        .get("ohlc", [])
    )

    collected_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    rows = []

    for candle in candles:

        candle_start = (
            datetime.fromtimestamp(
                int(
                    candle[
                        "timestamp"
                    ]
                ),
                tz=timezone.utc,
            )
        )

        # Explicit filtering is important because Bitstamp
        # can return candles outside the exact requested range.

        if not (
            start
            <=
            candle_start
            <
            end
        ):

            continue

        candle_end = (
            candle_start
            +
            timedelta(minutes=1)
        )

        rows.append(
            {
                "exchange":
                    "BITSTAMP",

                "product_id":
                    "BTC/USD",

                "candle_start":
                    candle_start.isoformat(),

                "candle_end":
                    candle_end.isoformat(),

                "open":
                    float_or_none(
                        candle.get("open")
                    ),

                "high":
                    float_or_none(
                        candle.get("high")
                    ),

                "low":
                    float_or_none(
                        candle.get("low")
                    ),

                "close":
                    float_or_none(
                        candle.get("close")
                    ),

                "volume":
                    float_or_none(
                        candle.get("volume")
                    ),

                "collected_at":
                    collected_at,
            }
        )

    return rows


# ============================================================
# CRYPTO.COM
# ============================================================

def fetch_crypto_com(
    start,
    end,
):

    start = utc_datetime(start)
    end = utc_datetime(end)

    params = {
        "instrument_name":
            CRYPTO_COM_SYMBOL,

        "timeframe":
            "1m",

        "count":
            300,

        "start_ts":
            int(
                start.timestamp()
                *
                1000
            ),

        "end_ts":
            int(
                end.timestamp()
                *
                1000
            ),
    }

    response = requests.get(
        CRYPTO_COM_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != 0:

        raise RuntimeError(
            f"Crypto.com API error: "
            f"{payload}"
        )

    candles = (
        payload
        .get("result", {})
        .get("data", [])
    )

    collected_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    rows = []

    for candle in candles:

        candle_start = (
            datetime.fromtimestamp(
                int(
                    candle["t"]
                )
                /
                1000.0,
                tz=timezone.utc,
            )
            .replace(
                second=0,
                microsecond=0,
            )
        )

        if not (
            start
            <=
            candle_start
            <
            end
        ):

            continue

        candle_end = (
            candle_start
            +
            timedelta(minutes=1)
        )

        rows.append(
            {
                "exchange":
                    "CRYPTO_COM",

                "product_id":
                    CRYPTO_COM_SYMBOL,

                "candle_start":
                    candle_start.isoformat(),

                "candle_end":
                    candle_end.isoformat(),

                "open":
                    float_or_none(
                        candle.get("o")
                    ),

                "high":
                    float_or_none(
                        candle.get("h")
                    ),

                "low":
                    float_or_none(
                        candle.get("l")
                    ),

                "close":
                    float_or_none(
                        candle.get("c")
                    ),

                "volume":
                    float_or_none(
                        candle.get("v")
                    ),

                "collected_at":
                    collected_at,
            }
        )

    return rows


# ============================================================
# COVERAGE
# ============================================================

def print_coverage():

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT

                exchange,

                COUNT(*) AS rows,

                MIN(candle_start),

                MAX(candle_start),

                COUNT(
                    DISTINCT DATE(candle_start)
                ) AS dates

            FROM
                cross_exchange_spot_minutes

            GROUP BY
                exchange

            ORDER BY
                exchange
            """
        ).fetchall()

    print()
    print(
        "=" * 100
    )

    print(
        "DATABASE COVERAGE"
    )

    print(
        "=" * 100
    )

    for row in rows:

        print(row)


# ============================================================
# BACKFILL
# ============================================================

def backfill(
    start,
    end,
):

    initialize_table()

    current = (
        start
        .replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    )

    total_bitstamp = 0
    total_crypto_com = 0

    while current < end:

        day_end = min(
            current
            +
            timedelta(days=1),

            end,
        )

        print()
        print(
            f"{current.date()}"
        )

        # ----------------------------------------------------
        # BITSTAMP
        #
        # The endpoint truncates large 1-minute requests near
        # ~1000 candles, so fetch the day in 12-hour chunks.
        # ----------------------------------------------------

        bitstamp_rows = []

        chunk_start = current

        while chunk_start < day_end:

            chunk_end = min(
                chunk_start
                +
                timedelta(hours=12),

                day_end,
            )

            try:

                rows = fetch_bitstamp(
                    chunk_start,
                    chunk_end,
                )

                bitstamp_rows.extend(
                    rows
                )

            except Exception as error:

                print(
                    f"  Bitstamp ERROR "
                    f"{chunk_start} -> "
                    f"{chunk_end}: "
                    f"{error}"
                )

            chunk_start = chunk_end

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

        # Deduplicate defensively in case the API includes
        # overlapping boundary candles.

        bitstamp_by_start = {
            row["candle_start"]:
                row

            for row in bitstamp_rows
        }

        bitstamp_rows = list(
            bitstamp_by_start.values()
        )

        bitstamp_rows.sort(
            key=lambda row:
                row["candle_start"]
        )

        saved = save_rows(
            bitstamp_rows
        )

        total_bitstamp += saved

        print(
            f"  Bitstamp:    "
            f"fetched={len(bitstamp_rows):4d} "
            f"saved={saved:4d}"
        )

        # ----------------------------------------------------
        # CRYPTO.COM
        #
        # Request in smaller chunks because this endpoint's
        # count limit is much smaller than one full day.
        # ----------------------------------------------------

        crypto_rows = []

        chunk_start = current

        while chunk_start < day_end:

            chunk_end = min(
                chunk_start
                +
                timedelta(
                    minutes=250
                ),

                day_end,
            )

            try:

                rows = fetch_crypto_com(
                    chunk_start,
                    chunk_end,
                )

                crypto_rows.extend(
                    rows
                )

            except Exception as error:

                print(
                    f"  Crypto.com ERROR "
                    f"{chunk_start} -> "
                    f"{chunk_end}: "
                    f"{error}"
                )

            chunk_start = chunk_end

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

        # Remove duplicates defensively.

        crypto_by_start = {
            row["candle_start"]:
                row

            for row in crypto_rows
        }

        crypto_rows = list(
            crypto_by_start.values()
        )

        saved = save_rows(
            crypto_rows
        )

        total_crypto_com += saved

        print(
            f"  Crypto.com:  "
            f"fetched={len(crypto_rows):4d} "
            f"saved={saved:4d}"
        )

        current = day_end

    print()
    print(
        "=" * 100
    )

    print(
        "BACKFILL COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Bitstamp rows saved:      "
        f"{total_bitstamp}"
    )

    print(
        f"Crypto.com rows saved:    "
        f"{total_crypto_com}"
    )

    print_coverage()


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        required=True,
    )

    parser.add_argument(
        "--end",
        required=True,
    )

    args = parser.parse_args()

    start = datetime.fromisoformat(
        args.start
    )

    end = datetime.fromisoformat(
        args.end
    )

    if start.tzinfo is None:

        start = start.replace(
            tzinfo=timezone.utc
        )

    if end.tzinfo is None:

        end = end.replace(
            tzinfo=timezone.utc
        )

    if end <= start:

        raise ValueError(
            "--end must be after --start"
        )

    backfill(
        start=
            start.astimezone(
                timezone.utc
            ),

        end=
            end.astimezone(
                timezone.utc
            ),
    )


if __name__ == "__main__":

    main()