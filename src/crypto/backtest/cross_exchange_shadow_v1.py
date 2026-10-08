"""Forward-paper test of the frozen KXBTC15M disagreement model.

Never submits real orders.
Stores one decision per contract, including PASS decisions.
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from src.database.db import get_connection

from src.crypto.backtest import (
    cross_exchange_disagreement_walkforward as model,
)

from src.crypto.backtest.cross_exchange_incremental_walkforward import (
    build_trade,
)


KALSHI = "https://external-api.kalshi.com/trade-api/v2"

COINBASE = (
    "https://api.exchange.coinbase.com/"
    "products/BTC-USD/candles"
)

BITSTAMP = (
    "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
)

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "prediction-market-bot-shadow-v1"
})

MAX_AFTER_T5_SECONDS = 35
MIN_TRAIN_ROWS = 1000


# ============================================================
# TIME AND HTTP
# ============================================================

def utcnow():
    return datetime.now(timezone.utc)


def parse_utc(value):

    value = str(value).replace("Z", "+00:00")

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        raise ValueError(
            f"Naive timestamp: {value}"
        )

    return dt.astimezone(timezone.utc)


def get_json(url, params=None):

    response = SESSION.get(
        url,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# KALSHI PUBLIC MARKET DATA
# ============================================================

def market_list():

    response = get_json(
        f"{KALSHI}/markets",
        {
            "series_ticker": "KXBTC15M",
            "status": "open",
            "limit": 200,
        },
    )

    return response.get("markets", [])


def market_detail(ticker):

    response = get_json(
        f"{KALSHI}/markets/{ticker}"
    )

    return response["market"]


def number(market, field):

    dollars = market.get(
        f"{field}_dollars"
    )

    if dollars not in (None, ""):
        return float(dollars)

    cents = market.get(field)

    if cents is None:
        return None

    return float(cents) / 100.0


def market_quote(market):

    yes_bid = number(market, "yes_bid")
    yes_ask = number(market, "yes_ask")

    no_bid = number(market, "no_bid")
    no_ask = number(market, "no_ask")

    if yes_bid is None or no_bid is None:
        raise ValueError("Missing YES/NO bids")

    # Binary contract complement relationships.

    if yes_ask is None:
        yes_ask = 1.0 - no_bid

    if no_ask is None:
        no_ask = 1.0 - yes_bid

    if not (
        0 < yes_bid <= yes_ask < 1
        and
        0 < no_bid <= no_ask < 1
    ):
        raise ValueError(
            "Invalid bid/ask ladder"
        )

    if abs(
        yes_ask + no_bid - 1.0
    ) > 0.011:
        raise ValueError(
            "Inconsistent YES ask / NO bid"
        )

    if abs(
        no_ask + yes_bid - 1.0
    ) > 0.011:
        raise ValueError(
            "Inconsistent NO ask / YES bid"
        )

    return {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "midpoint": (yes_bid + yes_ask) / 2.0,
    }


# ============================================================
# SYNCHRONIZED EXCHANGE CANDLES
# ============================================================

def get_coinbase_candle(start):

    end = start + timedelta(minutes=1)

    response = get_json(
        COINBASE,
        {
            "start": (
                start - timedelta(minutes=1)
            ).isoformat(),
            "end": (
                end + timedelta(minutes=1)
            ).isoformat(),
            "granularity": 60,
        },
    )

    # Coinbase candle schema:
    # [start, low, high, open, close, volume]

    for candle in response:

        if int(candle[0]) == int(
            start.timestamp()
        ):
            return float(candle[4])

    raise ValueError(
        f"Coinbase minute missing: {start}"
    )


def get_bitstamp_candle(start):

    end = start + timedelta(minutes=1)

    response = get_json(
        BITSTAMP,
        {
            "step": 60,
            "limit": 10,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "exclude_current_candle": "true",
        },
    )

    candles = (
        response
        .get("data", {})
        .get("ohlc", [])
    )

    for candle in candles:

        if int(
            candle["timestamp"]
        ) == int(start.timestamp()):

            return float(candle["close"])

    raise ValueError(
        f"Bitstamp minute missing: {start}"
    )


# ============================================================
# SHADOW DATABASE
# ============================================================

def ensure_table():

    with get_connection() as db:

        db.execute(
            """
            CREATE TABLE IF NOT EXISTS
                crypto_cross_exchange_shadow_v1 (

                ticker TEXT PRIMARY KEY,

                created_at TIMESTAMPTZ NOT NULL,
                close_time TIMESTAMPTZ NOT NULL,

                target_decision_time TIMESTAMPTZ NOT NULL,
                quote_received_at TIMESTAMPTZ NOT NULL,
                spot_candle_end TIMESTAMPTZ NOT NULL,

                floor_strike DOUBLE PRECISION NOT NULL,

                coinbase_close DOUBLE PRECISION NOT NULL,
                bitstamp_close DOUBLE PRECISION NOT NULL,

                yes_bid DOUBLE PRECISION NOT NULL,
                yes_ask DOUBLE PRECISION NOT NULL,
                no_bid DOUBLE PRECISION NOT NULL,
                no_ask DOUBLE PRECISION NOT NULL,

                market_probability DOUBLE PRECISION NOT NULL,

                control_probability DOUBLE PRECISION NOT NULL,
                plus_probability DOUBLE PRECISION NOT NULL,

                training_start TIMESTAMPTZ NOT NULL,
                training_end_exclusive TIMESTAMPTZ NOT NULL,
                training_rows INTEGER NOT NULL,

                control_beta DOUBLE PRECISION NOT NULL,
                plus_beta_coinbase DOUBLE PRECISION NOT NULL,
                plus_beta_disagreement DOUBLE PRECISION NOT NULL,

                control_side TEXT,
                control_entry DOUBLE PRECISION,
                control_fee DOUBLE PRECISION,
                control_cost DOUBLE PRECISION,
                control_edge DOUBLE PRECISION,

                plus_side TEXT,
                plus_entry DOUBLE PRECISION,
                plus_fee DOUBLE PRECISION,
                plus_cost DOUBLE PRECISION,
                plus_edge DOUBLE PRECISION,

                settlement_result TEXT,

                control_pnl DOUBLE PRECISION,
                plus_pnl DOUBLE PRECISION,

                settled_at TIMESTAMPTZ
            )
            """
        )


def already_recorded(ticker):

    with get_connection() as db:

        row = db.execute(
            """
            SELECT 1
            FROM crypto_cross_exchange_shadow_v1
            WHERE ticker = ?
            """,
            (ticker,),
        ).fetchone()

    return row is not None


def store_decision(record):

    fields = list(record)

    columns = ", ".join(fields)

    marks = ", ".join(
        "?" for _ in fields
    )

    with get_connection() as db:

        db.execute(
            f"""
            INSERT INTO crypto_cross_exchange_shadow_v1
                ({columns})
            VALUES ({marks})
            ON CONFLICT (ticker) DO NOTHING
            """,
            tuple(record.values()),
        )


# ============================================================
# DAILY RECALIBRATION
# ============================================================

def training_for_day(day):

    # Match historical walk-forward embargo:
    # exclude the entire preceding UTC calendar day.

    end = (
        pd.Timestamp(day).floor("D")
        -
        pd.Timedelta(days=1)
    )

    start = (
        end
        -
        pd.Timedelta(days=30)
    )

    train = model.load_data(
        start=start,
        end=end,
    )

    if len(train) < MIN_TRAIN_ROWS:

        raise ValueError(
            f"Only {len(train)} training rows; "
            f"need {MIN_TRAIN_ROWS}"
        )

    control = model.fit_offset_logistic(
        train,
        ["coinbase_x"],
    )

    plus = model.fit_offset_logistic(
        train,
        [
            "coinbase_x",
            "disagreement_x",
        ],
    )

    return (
        train,
        start,
        end,
        control,
        plus,
    )


# ============================================================
# PAPER TRADE SELECTION
# ============================================================

def paper_candidate(quote, fair_yes):

    # Reuse our existing historical execution rule.
    #
    # build_trade expects an outcome for backtest P&L.
    # The dummy outcome is NOT stored or used here.

    dummy_row = {
        **quote,
        "outcome": 0,
    }

    trade = build_trade(
        dummy_row,
        fair_yes=fair_yes,
    )

    if trade is None:
        return None

    # Explicitly exclude hypothetical won/pnl fields.

    return {
        key: trade[key]
        for key in (
            "side",
            "entry",
            "fee",
            "cost",
            "edge",
        )
    }


# ============================================================
# EVALUATE ONE LIVE CONTRACT
# ============================================================

def handle_market(list_entry):

    ticker = list_entry["ticker"]

    close = parse_utc(
        list_entry["close_time"]
    )

    target = close - timedelta(minutes=5)

    now = utcnow()

    delay = (
        now - target
    ).total_seconds()

    if not (
        0 <= delay <= MAX_AFTER_T5_SECONDS
    ):
        return False

    if already_recorded(ticker):
        return False

    # Take a fresh Kalshi market snapshot.

    market = market_detail(ticker)

    received = utcnow()

    delay = (
        received - target
    ).total_seconds()

    if not (
        0 <= delay <= MAX_AFTER_T5_SECONDS
    ):

        print(
            f"SKIP {ticker}: quote too late "
            f"({delay:.1f}s after T-5)"
        )

        return False

    strike = float(
        market["floor_strike"]
    )

    if not strike > 1000:

        raise ValueError(
            f"Bad/missing live strike: {strike}"
        )

    quote = market_quote(market)

    # Historical backtest used a candle ending
    # exactly one minute before the T-5 quote.
    #
    # If market closes at 12:15:
    # target decision = 12:10
    # required candle end = 12:09
    # candle starts at 12:08.

    candle_end = (
        target - timedelta(minutes=1)
    )

    candle_start = (
        candle_end - timedelta(minutes=1)
    )

    coinbase = get_coinbase_candle(
        candle_start
    )

    bitstamp = get_bitstamp_candle(
        candle_start
    )

    if (
        max(
            abs(coinbase - strike),
            abs(bitstamp - strike),
        )
        /
        strike
    ) > 0.05:

        raise ValueError(
            "Implausible spot/strike deviation"
        )

    (
        train,
        train_start,
        train_end,
        control_beta,
        plus_beta,
    ) = training_for_day(received)

    # Exactly the same feature definitions
    # used in the historical walk-forward.

    coinbase_x = (
        (coinbase - strike)
        /
        strike
        *
        10000.0
        /
        10.0
    )

    disagreement_x = (
        (bitstamp - coinbase)
        /
        strike
        *
        10000.0
        /
        10.0
    )

    market_logit = np.log(
        quote["midpoint"]
        /
        (1.0 - quote["midpoint"])
    )

    control_p = float(
        model.sigmoid(
            np.array([
                market_logit
                +
                control_beta[0] * coinbase_x
            ])
        )[0]
    )

    plus_p = float(
        model.sigmoid(
            np.array([
                market_logit
                +
                plus_beta[0] * coinbase_x
                +
                plus_beta[1] * disagreement_x
            ])
        )[0]
    )

    control_trade = paper_candidate(
        quote,
        control_p,
    )

    plus_trade = paper_candidate(
        quote,
        plus_p,
    )

    # Save inputs, model coefficients,
    # probabilities, and BOTH decisions.
    #
    # No actual settlement is known yet.

    record = {
        "ticker": ticker,
        "created_at": utcnow(),
        "close_time": close,
        "target_decision_time": target,
        "quote_received_at": received,
        "spot_candle_end": candle_end,

        "floor_strike": strike,
        "coinbase_close": coinbase,
        "bitstamp_close": bitstamp,

        "yes_bid": quote["yes_bid"],
        "yes_ask": quote["yes_ask"],
        "no_bid": quote["no_bid"],
        "no_ask": quote["no_ask"],

        "market_probability": quote["midpoint"],

        "control_probability": control_p,
        "plus_probability": plus_p,

        "training_start": (
            train_start.to_pydatetime()
        ),
        "training_end_exclusive": (
            train_end.to_pydatetime()
        ),
        "training_rows": len(train),

        "control_beta": float(
            control_beta[0]
        ),
        "plus_beta_coinbase": float(
            plus_beta[0]
        ),
        "plus_beta_disagreement": float(
            plus_beta[1]
        ),
    }

    for prefix, trade in (
        ("control", control_trade),
        ("plus", plus_trade),
    ):

        record[f"{prefix}_side"] = (
            None if trade is None
            else trade["side"]
        )

        record[f"{prefix}_entry"] = (
            None if trade is None
            else trade["entry"]
        )

        record[f"{prefix}_fee"] = (
            None if trade is None
            else trade["fee"]
        )

        record[f"{prefix}_cost"] = (
            None if trade is None
            else trade["cost"]
        )

        record[f"{prefix}_edge"] = (
            None if trade is None
            else trade["edge"]
        )

    store_decision(record)

    print(
        f"{ticker} "
        f"T-5+{delay:.1f}s | "
        f"Kalshi={quote['midpoint']:.3f} | "
        f"Control={control_p:.3f} "
        f"{record['control_side'] or 'PASS'} | "
        f"Disagreement={plus_p:.3f} "
        f"{record['plus_side'] or 'PASS'}"
    )

    return True


# ============================================================
# POLL OPEN MARKETS
# ============================================================

def poll(quiet=False):

    eligible = []

    now = utcnow()

    for item in market_list():

        close_raw = item.get("close_time")

        if not close_raw:
            continue

        target = (
            parse_utc(close_raw)
            -
            timedelta(minutes=5)
        )

        delay = (
            now - target
        ).total_seconds()

        if (
            -5
            <= delay
            <= MAX_AFTER_T5_SECONDS
        ):
            eligible.append(item)

    for item in eligible:

        try:

            handle_market(item)

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            RuntimeError,
        ) as error:

            print(
                f"ERROR {item.get('ticker')}: "
                f"{type(error).__name__}: {error}"
            )

    if not eligible and not quiet:

        print(
            f"{now.isoformat()} "
            "No T-5 market in window"
        )


# ============================================================
# SETTLEMENT
# ============================================================

def settle():

    with get_connection() as db:

        pending = db.execute(
            """
            SELECT
                ticker,
                control_side,
                control_cost,
                plus_side,
                plus_cost

            FROM crypto_cross_exchange_shadow_v1

            WHERE
                settlement_result IS NULL
                AND close_time < ?

            ORDER BY close_time
            """,
            (utcnow(),),
        ).fetchall()

    for (
        ticker,
        control_side,
        control_cost,
        plus_side,
        plus_cost,
    ) in pending:

        try:

            market = market_detail(ticker)

            result = str(
                market.get("result", "")
            ).lower()

            if result not in ("yes", "no"):
                continue

            def realized(side, cost):

                if side is None:
                    return None

                won = (
                    side.lower() == result
                )

                return float(
                    int(won) - float(cost)
                )

            control_pnl = realized(
                control_side,
                control_cost,
            )

            plus_pnl = realized(
                plus_side,
                plus_cost,
            )

            with get_connection() as db:

                db.execute(
                    """
                    UPDATE crypto_cross_exchange_shadow_v1

                    SET
                        settlement_result = ?,
                        control_pnl = ?,
                        plus_pnl = ?,
                        settled_at = ?

                    WHERE
                        ticker = ?
                        AND settlement_result IS NULL
                    """,
                    (
                        result,
                        control_pnl,
                        plus_pnl,
                        utcnow(),
                        ticker,
                    ),
                )

            print(
                f"SETTLED {ticker}: {result}"
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ) as error:

            print(
                f"SETTLEMENT ERROR {ticker}: "
                f"{error}"
            )


# ============================================================
# READ-ONLY LIVE SOURCE PROBE
# ============================================================

def probe():

    now = utcnow()

    markets = market_list()

    print(
        "UTC:", now.isoformat()
    )

    print(
        "Open KXBTC15M markets:",
        len(markets),
    )

    upcoming = sorted(
        (
            market
            for market in markets
            if market.get("close_time")
            and
            parse_utc(
                market["close_time"]
            ) > now
        ),
        key=lambda market:
            parse_utc(
                market["close_time"]
            ),
    )

    if upcoming:

        market = market_detail(
            upcoming[0]["ticker"]
        )

        print(
            "MARKET",
            market.get("ticker"),
        )

        print(
            "Close:",
            market.get("close_time"),
        )

        print(
            "Strike:",
            market.get("floor_strike"),
        )

        print(
            "QUOTE",
            market_quote(market),
        )

    # Probe a safely completed recent minute.

    candle_start = (
        now.replace(
            second=0,
            microsecond=0,
        )
        -
        timedelta(minutes=2)
    )

    print(
        "CANDLE",
        candle_start.isoformat(),
    )

    print(
        "Coinbase:",
        get_coinbase_candle(
            candle_start
        ),
    )

    print(
        "Bitstamp:",
        get_bitstamp_candle(
            candle_start
        ),
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--probe",
        action="store_true",
    )

    group.add_argument(
        "--once",
        action="store_true",
    )

    group.add_argument(
        "--loop",
        action="store_true",
    )

    group.add_argument(
        "--settle",
        action="store_true",
    )

    args = parser.parse_args()

    if args.probe:

        probe()
        return

    ensure_table()

    if args.settle:

        settle()

    elif args.once:

        poll()

    else:

        print(
            "Forward PAPER logging only. "
            "No orders will be submitted."
        )

        print(
            "Polling every 5 seconds. "
            "Settling every 60 seconds. "
            "Ctrl+C to stop."
        )

        last_settlement_check = (
            time.monotonic() - 60
        )

        try:

            while True:

                try:
                    poll(quiet=True)

                except Exception as error:
                    print(
                        f"POLL ERROR: "
                        f"{type(error).__name__}: {error}"
                    )

                if (
                    time.monotonic()
                    -
                    last_settlement_check
                    >= 60
                ):

                    try:
                        settle()

                    except Exception as error:
                        print(
                            f"SETTLEMENT ERROR: "
                            f"{type(error).__name__}: {error}"
                        )

                    finally:
                        last_settlement_check = (
                            time.monotonic()
                        )

                time.sleep(5)

        except KeyboardInterrupt:
            print("Stopped")


if __name__ == "__main__":
    main()