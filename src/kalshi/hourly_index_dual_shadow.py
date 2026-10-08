import argparse
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from src.backtest.hourly_index_market_backtest import (
    load_history,
    make_snapshots,
    fit_alpha,
    select_trades,
    taker_fee,
)
from src.kalshi.hourly_market import parse_candle


BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = "KXTEMPNYCHS"
ET = ZoneInfo("America/New_York")

HORIZONS = (45, 5)
TRAIN_DAYS = 14
MIN_TRAIN_DAYS = 10
MAX_QUOTE_AGE = 3

MIN_EDGE = 0.025
MIN_ROI = 0.05
BUFFER = 0.01
FEE_RATE = 0.07

SESSION = requests.Session()


def utcnow():
    return datetime.now(timezone.utc)


def parse_dt(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def api(path, params=None):
    last_error = None

    for attempt in range(4):
        try:
            response = SESSION.get(
                BASE + path,
                params=params,
                timeout=12,
            )

            if response.status_code in (
                429, 500, 502, 503, 504
            ):
                last_error = RuntimeError(
                    f"HTTP {response.status_code}: {path}"
                )
                response.close()
                time.sleep(min(2 ** attempt, 8))
                continue

            response.raise_for_status()
            result = response.json()
            response.close()
            return result

        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(
        f"Kalshi request failed: {last_error}"
    )


def check_fees():
    series = api(
        f"/series/{SERIES}"
    )["series"]

    fee_type = series.get("fee_type")
    multiplier = series.get("fee_multiplier")

    if (
        fee_type != "quadratic"
        or multiplier is None
        or abs(float(multiplier) - 1.0) > 1e-9
    ):
        raise RuntimeError(
            "Fee assumptions do not match the series. "
            "Paper trading stopped."
        )

    print(
        "Fee check: quadratic, multiplier 1",
        flush=True,
    )


def calculate_alphas(target_day):
    """
    Calculate each horizon's alpha independently,
    using outcomes no later than target_day - 2.
    """
    cutoff = target_day - timedelta(days=2)

    markets, candles = load_history(
        cutoff.isoformat()
    )

    result = {}

    for horizon in HORIZONS:
        snap = make_snapshots(
            markets,
            candles,
            horizon,
            MAX_QUOTE_AGE,
            1,
            2,
        )

        eligible = sorted(
            day for day in snap["day"].unique()
            if day <= cutoff
        )

        training_days = eligible[-TRAIN_DAYS:]

        if len(training_days) < MIN_TRAIN_DAYS:
            raise RuntimeError(
                f"{horizon}m: insufficient training dates"
            )

        newest = training_days[-1]

        if newest < target_day - timedelta(days=3):
            raise RuntimeError(
                f"Historical data too old for {horizon}m. "
                f"Newest date: {newest}. "
                "Run the hourly index backfill."
            )

        train = snap[
            snap["day"].isin(training_days)
        ]

        alpha = fit_alpha(train)
        result[horizon] = alpha

        print(
            f"ALPHA {horizon:2d}m | "
            f"target={target_day} | "
            f"alpha={alpha:.3f} | "
            f"training={training_days[0]} "
            f"to {training_days[-1]} | "
            f"days={len(training_days)}",
            flush=True,
        )

    return result


def get_open_events():
    markets = []
    cursor = None
    seen = set()

    for _ in range(20):
        params = {
            "series_ticker": SERIES,
            "status": "open",
            "limit": 200,
        }

        if cursor:
            params["cursor"] = cursor

        data = api("/markets", params)
        markets.extend(data.get("markets", []))

        cursor = data.get("cursor")

        if not cursor:
            break

        if cursor in seen:
            raise RuntimeError(
                "Repeated market pagination cursor"
            )

        seen.add(cursor)
    else:
        raise RuntimeError(
            "Market pagination limit exceeded"
        )

    events = defaultdict(list)

    for market in markets:
        event = market.get("event_ticker")
        if event:
            events[event].append(market)

    return events


def get_signal_quotes(tickers, decision):
    """
    Use the same cutoff convention as the
    historical backtest.
    """
    cutoff = decision - timedelta(minutes=1)
    earliest = cutoff - timedelta(
        minutes=MAX_QUOTE_AGE
    )

    data = api("/markets/candlesticks", {
        "market_tickers": ",".join(sorted(tickers)),
        "start_ts": int(earliest.timestamp()),
        "end_ts": int(cutoff.timestamp()),
        "period_interval": 1,
    })

    rows = []

    for market in data.get("markets", []):
        ticker = market.get("market_ticker")

        if ticker not in tickers:
            continue

        valid = []

        for candle in market.get("candlesticks", []):
            quote = parse_candle(candle)

            timestamp = quote["timestamp"]
            bid = quote["yes_bid"]
            ask = quote["yes_ask"]

            if (
                earliest <= timestamp <= cutoff
                and bid is not None
                and ask is not None
                and 0 <= bid <= ask <= 1
            ):
                valid.append((
                    timestamp,
                    float(bid),
                    float(ask),
                ))

        if not valid:
            continue

        timestamp, bid, ask = max(
            valid,
            key=lambda x: x[0],
        )

        rows.append({
            "ticker": ticker,
            "mid": (bid + ask) / 2,
            "signal_bid": bid,
            "signal_ask": ask,
            "signal_time": timestamp,
        })

    return pd.DataFrame(rows)


def live_ask(ticker, side):
    """
    An opposing bid implies the executable ask.
    Requires at least one displayed contract.
    """
    data = api(
        f"/markets/{ticker}/orderbook",
        {"depth": 1},
    )

    book = data.get("orderbook_fp") or {}

    if side == "YES":
        levels = book.get("no_dollars") or []
    else:
        levels = book.get("yes_dollars") or []

    if not levels:
        return None

    best = max(
        levels,
        key=lambda level: float(level[0]),
    )

    bid = float(best[0])
    quantity = float(best[1])

    if quantity < 1:
        return None

    return round(1.0 - bid, 4)


def create_signal(
    horizon,
    event,
    markets,
    close,
    alpha,
    pending,
):
    if len(markets) != 10:
        print(
            f"SKIP {horizon}m {event}: "
            f"only {len(markets)} open contracts",
            flush=True,
        )
        return True

    decision = close - timedelta(
        minutes=horizon
    )

    tickers = {m["ticker"] for m in markets}

    quotes = get_signal_quotes(
        tickers,
        decision,
    )

    if quotes.empty:
        print(
            f"NO QUOTES {horizon}m {event}",
            flush=True,
        )
        return False

    quotes["event"] = event

    selected = select_trades(
        quotes,
        alpha,
        FEE_RATE,
        MIN_EDGE,
        MIN_ROI,
        BUFFER,
    )

    if selected.empty:
        print(
            f"PASS {horizon}m {event} "
            f"| alpha={alpha:.3f} "
            f"| fresh contracts={len(quotes)}/10",
            flush=True,
        )
        return True

    row = selected.iloc[0]
    side = row["side"]

    fair = (
        float(row["fair_yes"])
        if side == "YES"
        else 1 - float(row["fair_yes"])
    )

    pending.append({
        "horizon": horizon,
        "event": event,
        "ticker": row["ticker"],
        "side": side,
        "alpha": alpha,
        "fair": fair,
        "close": close,
        "entry_after": (
            decision + timedelta(minutes=1)
        ),
        "entry_deadline": (
            decision + timedelta(minutes=3)
        ),
    })

    print(
        f"SIGNAL {horizon}m {event} "
        f"| {row['ticker']} {side} "
        f"| fair={fair:.2%} "
        f"| signal edge={row['signal_edge']:+.2%} "
        f"| alpha={alpha:.3f}",
        flush=True,
    )

    return True


def process_pending(pending, positions):
    current = utcnow()

    for signal in list(pending):
        if current < signal["entry_after"]:
            continue

        if (
            current > signal["entry_deadline"]
            or current >= signal["close"]
        ):
            print(
                f"EXPIRED {signal['horizon']}m "
                f"{signal['event']}",
                flush=True,
            )
            pending.remove(signal)
            continue

        try:
            price = live_ask(
                signal["ticker"],
                signal["side"],
            )
        except Exception as exc:
            print(
                f"ORDERBOOK ERROR: {exc}",
                flush=True,
            )
            continue

        if price is None:
            continue

        if not 0.05 <= price <= 0.95:
            print(
                f"PASS {signal['horizon']}m "
                f"{signal['event']}: price outside range",
                flush=True,
            )
            pending.remove(signal)
            continue

        fee = taker_fee(price, FEE_RATE)
        capital = price + fee

        edge = signal["fair"] - capital - BUFFER
        roi = edge / (capital + BUFFER)

        if edge < MIN_EDGE or roi < MIN_ROI:
            print(
                f"PASS {signal['horizon']}m "
                f"{signal['event']}: edge faded "
                f"| edge={edge:+.2%} "
                f"| ROI={roi:+.2%}",
                flush=True,
            )
            pending.remove(signal)
            continue

        position = {
            **signal,
            "price": price,
            "fee": fee,
            "capital": capital,
            "entry_time": utcnow(),
            "settled": False,
            "pnl": None,
            "buffered_pnl": None,
        }

        positions.append(position)
        pending.remove(signal)

        print(
            f"PAPER TRADE {signal['horizon']}m "
            f"{signal['event']} "
            f"| {signal['ticker']} {signal['side']} "
            f"| price={price:.2f} "
            f"| fee={fee:.2f} "
            f"| fair={signal['fair']:.2%} "
            f"| net edge={edge:+.2%} "
            f"| expected ROI={roi:+.2%}",
            flush=True,
        )


def settle_positions(positions):
    for position in positions:
        if position["settled"]:
            continue

        if utcnow() < position["close"]:
            continue

        try:
            market = api(
                f"/markets/{position['ticker']}"
            )["market"]
        except Exception as exc:
            print(
                f"SETTLEMENT CHECK ERROR: {exc}",
                flush=True,
            )
            continue

        if (
            market.get("status") not in (
                "finalized", "settled"
            )
            or market.get("result") not in (
                "yes", "no"
            )
        ):
            continue

        won = (
            market["result"]
            == position["side"].lower()
        )

        payout = 1.0 if won else 0.0

        position["pnl"] = (
            payout - position["capital"]
        )
        position["buffered_pnl"] = (
            position["pnl"] - BUFFER
        )
        position["settled"] = True

        print(
            f"SETTLED {position['horizon']}m "
            f"{position['event']} "
            f"| {'WIN' if won else 'LOSS'} "
            f"| P&L={position['pnl']:+.2f} "
            f"| buffered={position['buffered_pnl']:+.2f}",
            flush=True,
        )

        print_summary(positions)


def print_summary(positions):
    print("\nPAPER PERFORMANCE")
    print("-" * 65)

    for horizon in HORIZONS:
        group = [
            p for p in positions
            if p["horizon"] == horizon
            and p["settled"]
        ]

        if not group:
            print(
                f"{horizon}m | settled trades=0"
            )
            continue

        capital = sum(
            p["capital"] for p in group
        )
        pnl = sum(
            p["buffered_pnl"] for p in group
        )
        wins = sum(
            p["pnl"] > 0 for p in group
        )

        roi = pnl / capital if capital else 0

        print(
            f"{horizon}m | trades={len(group)} "
            f"| wins={wins} "
            f"| buffered P&L={pnl:+.2f} "
            f"| buffered ROI={roi:+.2%}"
        )

    print("-" * 65, flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--once",
        action="store_true",
    )

    args = parser.parse_args()

    if args.poll_seconds < 2:
        parser.error(
            "Polling interval must be >= 2 seconds"
        )

    print("\nNYC HOURLY DUAL SHADOW TRADER")
    print("=" * 75)
    print("Strategy A: 45 minutes")
    print("Strategy B: 5 minutes")
    print("Expected ROI requirement: 5%")
    print("Mode: PAPER ONLY")
    print("Supabase: historical reads only")
    print("CSV output: disabled")
    print("Real order submission: disabled\n")

    check_fees()

    alpha_cache = {}
    processed = set()
    pending = []
    positions = []

    events = get_open_events()

    upcoming_days = sorted({
        parse_dt(m["close_time"])
        .astimezone(ET).date()
        for group in events.values()
        for m in group
        if m.get("close_time")
    })

    print("Open events:", len(events))

    for day in upcoming_days[:2]:
        alpha_cache[day] = calculate_alphas(day)

    if args.once:
        print("\nNEXT DECISIONS")

        shown = 0
        for event, group in sorted(
            events.items(),
            key=lambda item: (
                item[1][0].get("close_time") or ""
            ),
        ):
            if not group:
                continue

            close = parse_dt(group[0]["close_time"])

            if close <= utcnow():
                continue

            print(
                event,
                "| close:",
                close.astimezone(ET).isoformat(),
                "| 45m:",
                (close - timedelta(minutes=45))
                .astimezone(ET).strftime("%H:%M"),
                "| 5m:",
                (close - timedelta(minutes=5))
                .astimezone(ET).strftime("%H:%M"),
            )

            shown += 1
            if shown >= 5:
                break

        print("\nSTARTUP CHECK COMPLETE")
        return

    last_discovery = time.monotonic()
    last_settlement_check = 0.0
    last_heartbeat = 0.0

    print("\nMonitoring hourly markets...")
    print("Press Ctrl+C to stop.\n")

    while True:
        current = utcnow()

        # Refresh market discovery every 10 seconds,
        # rather than repeatedly polling all markets
        # on every five-second trading loop.
        if (
            time.monotonic() - last_discovery >= 10
        ):
            try:
                events = get_open_events()
            except Exception as exc:
                print(
                    f"MARKET DISCOVERY ERROR: {exc}",
                    flush=True,
                )

            last_discovery = time.monotonic()

        for event, group in events.items():
            if not group:
                continue

            closes = {
                m.get("close_time") for m in group
            }

            if None in closes or len(closes) != 1:
                continue

            close = parse_dt(next(iter(closes)))
            day = close.astimezone(ET).date()

            for horizon in HORIZONS:
                key = (horizon, event)

                if key in processed:
                    continue

                decision = close - timedelta(
                    minutes=horizon
                )

                lateness = (
                    utcnow() - decision
                ).total_seconds()

                # A bounded grace window accommodates
                # network and scheduling delays.
                if not 0 <= lateness <= 45:
                    continue

                try:
                    if day not in alpha_cache:
                        alpha_cache[day] = (
                            calculate_alphas(day)
                        )

                    alpha = alpha_cache[day][horizon]

                    finished = create_signal(
                        horizon,
                        event,
                        group,
                        close,
                        alpha,
                        pending,
                    )

                    if finished:
                        processed.add(key)

                except Exception as exc:
                    print(
                        f"SIGNAL ERROR {horizon}m "
                        f"{event}: {exc}",
                        flush=True,
                    )

        process_pending(pending, positions)

        if (
            time.monotonic() - last_settlement_check
            >= 60
        ):
            settle_positions(positions)
            last_settlement_check = time.monotonic()

        if time.monotonic() - last_heartbeat >= 300:
            print(
                f"HEARTBEAT {utcnow().isoformat()} "
                f"| pending={len(pending)} "
                f"| paper positions={len(positions)}",
                flush=True,
            )
            last_heartbeat = time.monotonic()

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShadow trader stopped.")