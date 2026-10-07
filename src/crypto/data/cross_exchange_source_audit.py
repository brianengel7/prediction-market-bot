from datetime import datetime, timezone

import requests


# ============================================================
# TEST WINDOW
# ============================================================

# Test one historical one-minute candle.
#
# Candle:
#   2026-07-01 11:59:00 UTC
#       ->
#   2026-07-01 12:00:00 UTC

START = datetime(
    2026,
    7,
    1,
    11,
    59,
    tzinfo=timezone.utc,
)

END = datetime(
    2026,
    7,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)

START_SECONDS = int(
    START.timestamp()
)

END_SECONDS = int(
    END.timestamp()
)

START_MS = (
    START_SECONDS
    *
    1000
)

END_MS = (
    END_SECONDS
    *
    1000
)


# ============================================================
# BITSTAMP
# ============================================================

def test_bitstamp():

    url = (
        "https://www.bitstamp.net/"
        "api/v2/ohlc/btcusd/"
    )

    params = {
        "step":
            60,

        "limit":
            10,

        "start":
            START_SECONDS,

        "end":
            END_SECONDS,

        "exclude_current_candle":
            "true",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    candles = (
        payload
        .get(
            "data",
            {},
        )
        .get(
            "ohlc",
            [],
        )
    )

    print()
    print(
        "=" * 90
    )

    print(
        "BITSTAMP"
    )

    print(
        "=" * 90
    )

    print(
        f"HTTP status:       "
        f"{response.status_code}"
    )

    print(
        f"Candles returned:  "
        f"{len(candles)}"
    )

    for candle in candles:

        timestamp = int(
            candle[
                "timestamp"
            ]
        )

        candle_time = (
            datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            )
        )

        print(
            candle_time,
            "O=",
            candle.get("open"),
            "H=",
            candle.get("high"),
            "L=",
            candle.get("low"),
            "C=",
            candle.get("close"),
            "V=",
            candle.get("volume"),
        )


# ============================================================
# CRYPTO.COM
# ============================================================

def get_crypto_com_btc_usd_symbol():

    url = (
        "https://api.crypto.com/"
        "exchange/v1/public/get-instruments"
    )

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    instruments = (
        payload
        .get(
            "result",
            {},
        )
        .get(
            "data",
            []
        )
    )

    candidates = []

    for instrument in instruments:

        base = str(
            instrument.get(
                "base_ccy",
                "",
            )
        ).upper()

        quote = str(
            instrument.get(
                "quote_ccy",
                "",
            )
        ).upper()

        symbol = str(
            instrument.get(
                "symbol",
                "",
            )
        )

        instrument_type = str(
            instrument.get(
                "inst_type",
                "",
            )
        ).upper()

        if (
            base == "BTC"
            and
            quote == "USD"
        ):

            candidates.append(
                {
                    "symbol":
                        symbol,

                    "inst_type":
                        instrument_type,

                    "raw":
                        instrument,
                }
            )

    print()
    print(
        "Crypto.com BTC/USD candidates:"
    )

    for candidate in candidates:

        print(
            candidate[
                "symbol"
            ],
            candidate[
                "inst_type"
            ],
        )

    # Prefer an actual spot instrument.

    for candidate in candidates:

        if (
            "SPOT"
            in
            candidate[
                "inst_type"
            ]
        ):

            return candidate[
                "symbol"
            ]

    # Some API versions identify spot differently.
    #
    # Fall back only if there is exactly one non-perpetual
    # BTC/USD candidate.

    non_perpetual = [
        candidate
        for candidate in candidates
        if (
            "PERPETUAL"
            not in
            candidate[
                "inst_type"
            ]
            and
            "FUTURE"
            not in
            candidate[
                "inst_type"
            ]
        )
    ]

    if len(
        non_perpetual
    ) == 1:

        return (
            non_perpetual[
                0
            ][
                "symbol"
            ]
        )

    return None


def test_crypto_com():

    symbol = (
        get_crypto_com_btc_usd_symbol()
    )

    print()
    print(
        "=" * 90
    )

    print(
        "CRYPTO.COM"
    )

    print(
        "=" * 90
    )

    if symbol is None:

        print(
            "Could not uniquely identify "
            "a BTC/USD spot instrument."
        )

        return

    print(
        f"Selected symbol:   "
        f"{symbol}"
    )

    url = (
        "https://api.crypto.com/"
        "exchange/v1/public/get-candlestick"
    )

    params = {
        "instrument_name":
            symbol,

        "timeframe":
            "1m",

        "count":
            10,

        "start_ts":
            START_MS,

        "end_ts":
            END_MS,
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    print(
        f"HTTP status:       "
        f"{response.status_code}"
    )

    print(
        f"API code:          "
        f"{payload.get('code')}"
    )

    candles = (
        payload
        .get(
            "result",
            {},
        )
        .get(
            "data",
            []
        )
    )

    print(
        f"Candles returned:  "
        f"{len(candles)}"
    )

    for candle in candles:

        timestamp = int(
            candle[
                "t"
            ]
        )

        candle_time = (
            datetime.fromtimestamp(
                timestamp
                /
                1000.0,
                tz=timezone.utc,
            )
        )

        print(
            candle_time,
            "O=",
            candle.get("o"),
            "H=",
            candle.get("h"),
            "L=",
            candle.get("l"),
            "C=",
            candle.get("c"),
            "V=",
            candle.get("v"),
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "CROSS-EXCHANGE HISTORICAL SOURCE AUDIT"
    )

    print(
        f"Requested candle: "
        f"{START.isoformat()} "
        f"-> "
        f"{END.isoformat()}"
    )

    test_bitstamp()

    test_crypto_com()


if __name__ == "__main__":

    main()