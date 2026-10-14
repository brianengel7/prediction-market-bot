
"""KXBTC15M live execution with independent risk limits.

Default: dry run.
Real orders require --live and two environment switches.
All orders use one contract and fill-or-kill.
"""

import argparse
import math
import os
import time
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

from src.database.db import get_connection
from src.crypto.backtest import cross_exchange_shadow_v1 as shadow
from src.crypto.backtest.cross_exchange_disagreement_walkforward import sigmoid
from src.kalshi.fees import calculate_taker_fee
from src.kalshi.live_executor import (
    build_client_order_id,
    get_live_fee_multiplier,
    live_trading_enabled,
)
from src.kalshi.trading_client import (
    OrderSubmissionUnknownError,
    find_order_by_client_order_id,
    get_cash_balance,
    get_order_fills,
    place_order,
    summarize_order_fills,
)


STRATEGY = "KXBTC15M_CROSS_EXCHANGE_V1"
MAX_SUBMIT_DELAY_SECONDS = 25
MIN_EDGE = 0.025
MAX_DAILY_ORDERS = 3
MAX_DAILY_REALIZED_LOSS = 3.00
MIN_CASH_BUFFER = 1.00
KILL_SWITCH = Path("STOP_KXBTC15M_LIVE")
POLL_SECONDS = 5


def enabled():
    return (
        live_trading_enabled()
    )


def ensure_table():
    with get_connection() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS crypto_cross_exchange_live_v1 (
                ticker TEXT PRIMARY KEY,
                client_order_id TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL,
                decision_time TIMESTAMPTZ NOT NULL,
                quote_checked_at TIMESTAMPTZ NOT NULL,
                side TEXT NOT NULL,
                limit_price DOUBLE PRECISION NOT NULL,
                fair_probability DOUBLE PRECISION NOT NULL,
                estimated_fee DOUBLE PRECISION NOT NULL,
                net_edge DOUBLE PRECISION NOT NULL,
                contracts INTEGER NOT NULL,
                order_status TEXT NOT NULL,
                kalshi_order_id TEXT,
                fill_count DOUBLE PRECISION,
                average_fill_price DOUBLE PRECISION,
                total_fee DOUBLE PRECISION,
                settlement_result TEXT,
                actual_pnl DOUBLE PRECISION,
                last_error TEXT,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)


def update(ticker, **fields):
    if not fields:
        return

    fields["updated_at"] = shadow.utcnow()
    keys = list(fields)
    sql = ", ".join(f"{name} = ?" for name in keys)

    with get_connection() as db:
        db.execute(
            f"UPDATE crypto_cross_exchange_live_v1 SET {sql} WHERE ticker = ?",
            tuple(fields[key] for key in keys) + (ticker,),
        )


def recent_candidates():
    now = shadow.utcnow()

    with get_connection() as db:
        rows = db.execute("""
            SELECT ticker, close_time, target_decision_time,
                   quote_received_at, spot_candle_end, floor_strike,
                   coinbase_close, bitstamp_close,
                   plus_side, plus_beta_coinbase, plus_beta_disagreement
            FROM crypto_cross_exchange_shadow_v1
            WHERE plus_side IN ('YES', 'NO')
              AND target_decision_time <= ?
              AND target_decision_time >= ?
            ORDER BY target_decision_time DESC
        """, (
            now,
            now - timedelta(seconds=MAX_SUBMIT_DELAY_SECONDS),
        )).fetchall()

    columns = (
        "ticker", "close_time", "target", "quote_time",
        "candle_end", "strike", "coinbase", "bitstamp",
        "side", "beta_c", "beta_d",
    )

    return [dict(zip(columns, row)) for row in rows]


def book_quote(ticker):
    """Compute executable asks using the opposite-side bids."""

    raw = shadow.get_json(
        f"{shadow.KALSHI}/markets/{ticker}/orderbook"
    )

    book = raw.get("orderbook_fp")

    if not isinstance(book, dict):
        raise ValueError("Missing orderbook_fp")

    def best(name):
        levels = book.get(name) or []

        parsed = [
            (Decimal(str(price)), Decimal(str(quantity)))
            for price, quantity in levels
        ]

        valid = [
            (price, quantity)
            for price, quantity in parsed
            if 0 < price < 1 and quantity > 0
        ]

        if not valid:
            raise ValueError(f"Empty {name} order book")

        return max(valid, key=lambda item: item[0])

    yes_bid, yes_size = best("yes_dollars")
    no_bid, no_size = best("no_dollars")

    yes_ask = Decimal("1") - no_bid
    no_ask = Decimal("1") - yes_bid

    if yes_bid >= yes_ask or no_bid >= no_ask:
        raise ValueError("Locked or crossed order book")

    return {
        "midpoint": float((yes_bid + yes_ask) / 2),
        "YES": (float(yes_ask), float(no_size)),
        "NO": (float(no_ask), float(yes_size)),
    }


def check_risk():
    now = shadow.utcnow()
    midnight = now.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    with get_connection() as db:
        count, pnl = db.execute("""
            SELECT COUNT(*), COALESCE(SUM(actual_pnl), 0)
            FROM crypto_cross_exchange_live_v1
            WHERE created_at >= ?
              AND created_at < ?
              AND order_status NOT IN ('REJECTED', 'ABORTED')
        """, (
            midnight,
            midnight + timedelta(days=1),
        )).fetchone()

        unresolved = db.execute("""
            SELECT COUNT(*)
            FROM crypto_cross_exchange_live_v1
            WHERE order_status IN (
                'CLAIMED',
                'SUBMITTING',
                'UNKNOWN',
                'ACK_PENDING',
                'FILLED_UNRECONCILED'
            )
        """).fetchone()[0]

    if count >= MAX_DAILY_ORDERS:
        raise RuntimeError("Daily BTC order cap reached")

    if float(pnl) <= -MAX_DAILY_REALIZED_LOSS:
        raise RuntimeError("Daily BTC realized loss limit reached")

    if unresolved:
        raise RuntimeError(
            "An unresolved order exists; reconcile before trading"
        )

    if KILL_SWITCH.exists():
        raise RuntimeError("BTC kill switch is active")


def evaluate(row):
    """Refresh the Kalshi price and recheck the frozen model."""

    now = shadow.utcnow()
    target = shadow.parse_utc(row["target"])

    delay = (now - target).total_seconds()

    if not 0 <= delay <= MAX_SUBMIT_DELAY_SECONDS:
        return None

    if (
        shadow.parse_utc(row["candle_end"])
        != target - timedelta(minutes=1)
    ):
        raise ValueError("Unexpected exchange candle alignment")

    if (
        shadow.parse_utc(row["quote_time"])
        > target + timedelta(seconds=35)
    ):
        raise ValueError("Original paper quote was late")

    if (
        shadow.parse_utc(row["close_time"])
        != target + timedelta(minutes=5)
    ):
        raise ValueError("Contract timing mismatch")

    ticker = row["ticker"]

    if not ticker.startswith("KXBTC15M-"):
        raise ValueError("Wrong market series")

    details = shadow.market_detail(ticker)

    if (
        shadow.parse_utc(details["close_time"])
        != target + timedelta(minutes=5)
    ):
        raise ValueError("Live market close time changed")

    if str(details.get("result", "")).lower() in ("yes", "no"):
        return None

    strike = float(row["strike"])

    if not (strike > 1000 and math.isfinite(strike)):
        raise ValueError("Invalid strike")

    if not math.isclose(
        float(details["floor_strike"]),
        strike,
        rel_tol=0,
        abs_tol=0.01,
    ):
        raise ValueError("Live strike differs from model strike")

    # Verify the live fee schedule before evaluating an order.
    multiplier = float(get_live_fee_multiplier(ticker))

    if not math.isclose(multiplier, 1.0, abs_tol=1e-9):
        raise RuntimeError(
            "Live fee schedule differs from frozen backtest"
        )

    cash = float(get_cash_balance())

    # Fetch the executable order book after the other API calls.
    quote = book_quote(ticker)
    checked_at = shadow.utcnow()

    if (
        checked_at - target
    ).total_seconds() > MAX_SUBMIT_DELAY_SECONDS:
        return None

    side = str(row["side"]).upper()

    if side not in ("YES", "NO"):
        raise ValueError("Invalid trading side")

    ask, size = quote[side]

    if size < 1:
        raise RuntimeError("Insufficient depth for one contract")

    midpoint = quote["midpoint"]

    if not 0 < midpoint < 1:
        raise ValueError("Invalid Kalshi midpoint")

    # Frozen feature scaling: 10 bps per unit.
    coinbase = float(row["coinbase"])
    bitstamp = float(row["bitstamp"])

    coinbase_x = (
        (coinbase - strike) / strike * 1000.0
    )

    disagreement_x = (
        (bitstamp - coinbase) / strike * 1000.0
    )

    market_logit = math.log(
        midpoint / (1.0 - midpoint)
    )

    adjusted_logit = (
        market_logit
        + float(row["beta_c"]) * coinbase_x
        + float(row["beta_d"]) * disagreement_x
    )

    fair_yes = float(
        sigmoid(np.array([adjusted_logit]))[0]
    )

    fair = fair_yes if side == "YES" else 1.0 - fair_yes

    # Do not allow the existing order client to round an
    # unsupported price to a different executable limit.
    price_decimal = Decimal(str(ask))

    if price_decimal != price_decimal.quantize(
        Decimal("0.0001")
    ):
        raise RuntimeError("Unsupported price precision")

    fee = calculate_taker_fee(
        price=ask,
        contracts=1,
        multiplier=1,
    )

    cost = ask + fee
    edge = fair - cost

    # Preserve the historical rule: trade only the best side.
    other_side = "NO" if side == "YES" else "YES"
    other_ask, _ = quote[other_side]

    other_fee = calculate_taker_fee(
        price=other_ask,
        contracts=1,
        multiplier=1,
    )

    other_edge = (
        (1.0 - fair)
        - other_ask
        - other_fee
    )

    if edge < MIN_EDGE:
        return None

    if edge < other_edge:
        return None

    if cash < cost + MIN_CASH_BUFFER:
        return None

    return {
        "ticker": ticker,
        "target": target,
        "side": side,
        "price": ask,
        "fair": fair,
        "fee": fee,
        "cost": cost,
        "edge": edge,
        "checked_at": checked_at,
    }


def claim(signal):
    """Atomically reserve one submission attempt per contract."""

    ticker = signal["ticker"]

    client_order_id = build_client_order_id(
        strategy=STRATEGY,
        target_date=signal["target"].date().isoformat(),
        decision_time=signal["target"],
        ticker=ticker,
        side=signal["side"],
    )

    now = shadow.utcnow()

    with get_connection() as db:
        row = db.execute("""
            INSERT INTO crypto_cross_exchange_live_v1 (
                ticker,
                client_order_id,
                created_at,
                decision_time,
                quote_checked_at,
                side,
                limit_price,
                fair_probability,
                estimated_fee,
                net_edge,
                contracts,
                order_status,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'CLAIMED', ?)
            ON CONFLICT (ticker) DO NOTHING
            RETURNING ticker
        """, (
            ticker,
            client_order_id,
            now,
            signal["target"],
            signal["checked_at"],
            signal["side"],
            signal["price"],
            signal["fair"],
            signal["fee"],
            signal["edge"],
            now,
        )).fetchone()

    return client_order_id if row else None


def execute(signal):
    check_risk()

    client_order_id = claim(signal)

    if client_order_id is None:
        return

    ticker = signal["ticker"]

    # Never automatically resubmit this claimed order.
    if KILL_SWITCH.exists():
        update(
            ticker,
            order_status="ABORTED",
            last_error="Kill switch activated",
        )
        return

    delay = (
        shadow.utcnow() - signal["target"]
    ).total_seconds()

    if delay > MAX_SUBMIT_DELAY_SECONDS:
        update(
            ticker,
            order_status="ABORTED",
            last_error="Submission deadline exceeded",
        )
        return

    update(ticker, order_status="SUBMITTING")

    try:
        response = place_order(
            ticker=ticker,
            side=signal["side"],
            entry_price=signal["price"],
            contracts=1,
            client_order_id=client_order_id,
            time_in_force="fill_or_kill",
            timeout=5,
        )

    except OrderSubmissionUnknownError as error:
        update(
            ticker,
            order_status="UNKNOWN",
            last_error=str(error)[:500],
        )
        print(f"UNKNOWN {ticker}: do not resubmit")
        return

    except Exception as error:
        update(
            ticker,
            order_status="REJECTED",
            last_error=str(error)[:500],
        )
        print(f"REJECTED {ticker}: {error}")
        return

    order_id = response.get("order_id")

    update(
        ticker,
        kalshi_order_id=order_id,
        order_status="ACK_PENDING",
    )

    print(
        f"ORDER_ACK {ticker}: "
        f"{signal['side']} 1 @ ${signal['price']:.4f}, "
        f"order_id={order_id}"
    )


def reconcile():
    """Recover submitted orders and calculate actual fills."""

    with get_connection() as db:
        pending = db.execute("""
            SELECT
                ticker,
                client_order_id,
                kalshi_order_id,
                side
            FROM crypto_cross_exchange_live_v1
            WHERE order_status IN (
                'CLAIMED',
                'SUBMITTING',
                'UNKNOWN',
                'ACK_PENDING',
                'FILLED_UNRECONCILED'
            )
        """).fetchall()

    for ticker, client_id, order_id, side in pending:
        try:
            if not order_id:
                found = find_order_by_client_order_id(
                    client_id,
                    ticker=ticker,
                )

                if not found or not found.get("order_id"):
                    # Submission state remains ambiguous.
                    # Do not retry it.
                    continue

                order_id = found["order_id"]

                update(
                    ticker,
                    kalshi_order_id=order_id,
                    order_status="ACK_PENDING",
                )

            fills = get_order_fills(order_id)

            summary = summarize_order_fills(
                fills=fills,
                outcome_side=side,
            )

            filled = float(summary["fill_count"])

            if filled >= 1:
                update(
                    ticker,
                    order_status="FILLED",
                    fill_count=filled,
                    average_fill_price=summary["average_fill_price"],
                    total_fee=summary["total_fee"],
                )

                print(
                    f"FILLED {ticker}: "
                    f"{filled:g} @ "
                    f"${summary['average_fill_price']:.4f}"
                )

            else:
                found = find_order_by_client_order_id(
                    client_id,
                    ticker=ticker,
                )

                server_status = str(
                    (found or {}).get("status", "")
                ).lower()

                reported = (found or {}).get("fill_count_fp")

                if reported is None:
                    reported = (found or {}).get("fill_count")

                if (
                    server_status in ("canceled", "cancelled")
                    and reported is not None
                    and float(reported) == 0
                ):
                    update(
                        ticker,
                        order_status="NO_FILL",
                        fill_count=0,
                    )
                    print(f"NO_FILL {ticker}")

                elif (
                    reported is not None
                    and float(reported) > 0
                ):
                    update(
                        ticker,
                        order_status="FILLED_UNRECONCILED",
                    )

                else:
                    update(
                        ticker,
                        order_status="ACK_PENDING",
                    )

        except Exception as error:
            print(
                f"RECONCILE_ERROR {ticker}: "
                f"{type(error).__name__}: {error}"
            )

    # Use real fills and final contract results for P&L.
    with get_connection() as db:
        settled = db.execute("""
            SELECT
                l.ticker,
                l.side,
                l.fill_count,
                l.average_fill_price,
                l.total_fee,
                s.settlement_result
            FROM crypto_cross_exchange_live_v1 l
            JOIN crypto_cross_exchange_shadow_v1 s
              ON s.ticker = l.ticker
            WHERE l.order_status = 'FILLED'
              AND l.actual_pnl IS NULL
              AND s.settlement_result IN ('yes', 'no')
        """).fetchall()

    for ticker, side, qty, price, fees, result in settled:
        if None in (qty, price, fees):
            continue

        payout = (
            float(qty) if side.lower() == result else 0.0
        )

        pnl = (
            payout
            - float(qty) * float(price)
            - float(fees)
        )

        update(
            ticker,
            settlement_result=result,
            actual_pnl=pnl,
            order_status="SETTLED",
        )

        print(
            f"LIVE_SETTLED {ticker}: "
            f"{result} pnl=${pnl:+.2f}"
        )


def poll(live, seen):
    # Reuse the complete existing shadow decision process.
    shadow.poll(quiet=True)

    for row in recent_candidates():
        ticker = row["ticker"]

        if ticker in seen:
            continue

        with get_connection() as db:
            exists = db.execute("""
                SELECT 1
                FROM crypto_cross_exchange_live_v1
                WHERE ticker = ?
            """, (ticker,)).fetchone()

        if exists:
            seen.add(ticker)
            continue

        try:
            signal = evaluate(row)

            if signal is None:
                continue

            seen.add(ticker)

            mode = "LIVE" if live else "DRY RUN"

            print(
                f"ELIGIBLE {ticker}: "
                f"{signal['side']} "
                f"@ ${signal['price']:.4f}, "
                f"edge={signal['edge']:.2%} | {mode}"
            )

            if live:
                execute(signal)

        except Exception as error:
            print(
                f"BTC_BLOCKED {ticker}: "
                f"{type(error).__name__}: {error}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real orders, subject to both environment gates",
    )

    args = parser.parse_args()

    if args.live and not enabled():
        parser.error(
            "Real orders require both "
            "KALSHI_LIVE_TRADING_ENABLED=true"
        )

    shadow.ensure_table()
    ensure_table()

    mode = (
        "REAL ONE-CONTRACT FOK"
        if args.live
        else "DRY RUN"
    )

    print(f"KXBTC15M {mode}")
    print(
        "Polling every 5 seconds. "
        "Settling and reconciling every 60 seconds."
    )
    print(
        "Maximum 3 BTC orders/day. "
        "Daily realized-loss cutoff: $3."
    )

    seen = set()
    last_reconcile = 0.0

    try:
        while True:
            try:
                poll(args.live, seen)

            except Exception as error:
                print(
                    f"POLL_ERROR: "
                    f"{type(error).__name__}: {error}"
                )

            if time.monotonic() - last_reconcile >= 60:
                try:
                    shadow.settle()
                    reconcile()

                except Exception as error:
                    print(
                        f"SETTLE_ERROR: "
                        f"{type(error).__name__}: {error}"
                    )

                last_reconcile = time.monotonic()

            if args.once:
                break

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
