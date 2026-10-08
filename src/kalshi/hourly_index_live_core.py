import argparse
import time

from collections import defaultdict
from datetime import (
    datetime,
    timedelta,
    timezone,
)
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

from src.kalshi.hourly_live_executor import (
    execute_hourly_trade,
    size_and_check,
)

from src.kalshi.live_executor import (
    live_trading_enabled,
)

from src.kalshi.trading_client import (
    get_market_settlement,
)


API = (
    "https://external-api.kalshi.com"
    "/trade-api/v2"
)

SERIES = "KXTEMPNYCHS"
ET = ZoneInfo("America/New_York")

MIN_EDGE = 0.025
MIN_ROI = 0.05
BUFFER = 0.01
FEE_RATE = 0.07

SESSION = requests.Session()


def utcnow():
    return datetime.now(timezone.utc)


def parse_time(value):
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def get(path, params=None):
    for attempt in range(4):
        try:
            response = SESSION.get(
                API + path,
                params=params,
                timeout=12,
            )

            if response.status_code in (
                429, 500, 502, 503, 504
            ):
                response.close()
                time.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            result = response.json()
            response.close()
            return result

        except requests.RequestException:
            if attempt == 3:
                raise

            time.sleep(2 ** attempt)

    raise RuntimeError(
        "Kalshi public API unavailable"
    )


def fee_check():
    info = get(
        f"/series/{SERIES}"
    )["series"]

    if (
        info.get("fee_type") != "quadratic"
        or float(
            info.get("fee_multiplier", -1)
        ) != 1
    ):
        raise RuntimeError(
            "Hourly series fee assumptions "
            "have changed"
        )


def open_events():
    groups = defaultdict(list)
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

        data = get("/markets", params)

        for market in data.get("markets", []):
            if market.get("event_ticker"):
                groups[
                    market["event_ticker"]
                ].append(market)

        cursor = data.get("cursor")

        if not cursor:
            return groups

        if cursor in seen:
            raise RuntimeError(
                "Repeated Kalshi market "
                "pagination cursor"
            )

        seen.add(cursor)

    raise RuntimeError(
        "Open market pagination not exhausted"
    )


def signal_quotes(markets, decision):
    """
    Match the historical backtest's
    decision quote cutoff.
    """
    cutoff = (
        decision - timedelta(minutes=1)
    )

    earliest = (
        cutoff - timedelta(minutes=3)
    )

    wanted = {
        market["ticker"]
        for market in markets
    }

    data = get("/markets/candlesticks", {
        "market_tickers": ",".join(
            sorted(wanted)
        ),
        "start_ts": int(
            earliest.timestamp()
        ),
        "end_ts": int(
            cutoff.timestamp()
        ),
        "period_interval": 1,
    })

    rows = []

    for market in data.get("markets", []):
        ticker = market.get(
            "market_ticker"
        )

        if ticker not in wanted:
            continue

        valid = []

        for candle in market.get(
            "candlesticks", []
        ):
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

        if valid:
            _, bid, ask = max(
                valid,
                key=lambda x: x[0],
            )

            rows.append({
                "ticker": ticker,
                "mid": (bid + ask) / 2,
                "signal_bid": bid,
                "signal_ask": ask,
            })

    return pd.DataFrame(rows)


def book_ask(ticker, side):
    data = get(
        f"/markets/{ticker}/orderbook",
        {"depth": 1},
    )

    book = (
        data.get("orderbook_fp") or {}
    )

    levels = book.get(
        "no_dollars"
        if side == "YES"
        else "yes_dollars"
    ) or []

    if not levels:
        return None

    level = max(
        levels,
        key=lambda x: float(x[0]),
    )

    price = round(
        1 - float(level[0]),
        4,
    )

    depth = float(level[1])

    if (
        not 0 < price < 1
        or depth < 1
    ):
        return None

    return price, depth


def refresh_historical_day(target_day):
    """
    Refresh the last three eligible prior
    settlement dates.

    Only raw historical data is collected.
    """
    from src.backtest.hourly_index_backfill import (
        initialize_storage,
        initialize_checkpoint,
        load_settled_markets,
        already_collected,
        collect_event,
    )

    initialize_storage()
    initialize_checkpoint()

    dates = {
        target_day - timedelta(days=i)
        for i in (2, 3, 4)
    }

    groups = defaultdict(list)

    for market in load_settled_markets():
        day = (
            parse_time(market["close_time"])
            .astimezone(ET)
            .date()
        )

        if day in dates:
            groups[
                market["event_ticker"]
            ].append(market)

    if not groups:
        raise RuntimeError(
            "No previous settled events "
            "available for historical refresh"
        )

    new_count = 0

    for event, markets in groups.items():
        if not already_collected(
            event, 60
        ):
            collect_event(
                event, markets, 60
            )
            new_count += 1

    print(
        f"HISTORY {target_day}: "
        f"checked={len(groups)} "
        f"newly_collected={new_count}",
        flush=True,
    )


def calibrate(day, horizon):
    cutoff = (
        day - timedelta(days=2)
    )

    markets, candles = load_history(
        cutoff.isoformat()
    )

    snapshots = make_snapshots(
        markets,
        candles,
        horizon,
        3,
        1,
        2,
    )

    dates = sorted(
        date
        for date in snapshots["day"].unique()
        if date <= cutoff
    )[-14:]

    if (
        len(dates) < 10
        or dates[-1]
        < day - timedelta(days=3)
    ):
        raise RuntimeError(
            "Insufficient recent hourly "
            f"training data for {day}"
        )

    train = snapshots[
        snapshots["day"].isin(dates)
    ]

    alpha = fit_alpha(train)

    print(
        f"ALPHA {horizon}m "
        f"target={day} "
        f"alpha={alpha:.3f} "
        f"train={dates[0]}..{dates[-1]}",
        flush=True,
    )

    return alpha


def run(horizon):
    if horizon not in (5, 45):
        raise ValueError(
            "Unsupported hourly horizon"
        )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--live",
        action="store_true",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Read-only checks, no orders",
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5,
    )

    args = parser.parse_args()

    if not 2 <= args.poll_seconds <= 10:
        parser.error(
            "--poll-seconds must be 2..10"
        )

    if (
        args.live
        and not live_trading_enabled()
    ):
        raise RuntimeError(
            "Global KALSHI_LIVE_TRADING_ENABLED "
            "switch is off"
        )

    print(
        f"\nHOURLY NYC {horizon}m | "
        f"{'LIVE' if args.live else 'PAPER'}"
    )

    print(
        "2.5c edge, 5% ROI, "
        "1c buffer, one-minute entry delay"
    )

    print(
        "Sizing: existing 10% cash / "
        "$10 minimum / env cap"
    )

    print(
        "Daily executor unchanged. "
        "Backtest results never persisted."
    )

    fee_check()

    events = open_events()

    print(
        "Open events:",
        len(events),
    )

    if args.once:
        for event, markets in list(
            events.items()
        )[:5]:
            close = parse_time(
                markets[0]["close_time"]
            )

            print(
                event,
                "decision=",
                close - timedelta(
                    minutes=horizon
                ),
            )

        days = sorted({
            parse_time(
                markets[0]["close_time"]
            ).astimezone(ET).date()

            for markets in events.values()
            if markets
        })

        if days:
            calibrate(
                days[0],
                horizon,
            )

        print(
            "READ-ONLY CHECK COMPLETE"
        )
        return

    alpha_cache = {}
    attempted = set()
    pending = []
    paper_positions = []

    last_discovery = 0.0
    last_settlements = 0.0
    last_heartbeat = 0.0

    while True:
        tick = time.monotonic()

        if (
            tick - last_discovery >= 10
        ):
            events = open_events()
            last_discovery = tick

        # Prepare the model ahead of
        # its narrow decision window.
        upcoming_days = sorted({
            parse_time(
                markets[0]["close_time"]
            ).astimezone(ET).date()

            for markets in events.values()
            if (
                markets
                and utcnow()
                < parse_time(
                    markets[0]["close_time"]
                )
                <= utcnow()
                + timedelta(hours=3)
            )
        })

        for day in upcoming_days:
            if day not in alpha_cache:
                try:
                    refresh_historical_day(
                        day
                    )

                    alpha_cache[day] = (
                        calibrate(
                            day,
                            horizon,
                        )
                    )

                except Exception as exc:
                    print(
                        "CALIBRATION UNAVAILABLE "
                        f"{day}: {exc}",
                        flush=True,
                    )

                    # Fail closed.
                    alpha_cache[day] = None

        # Evaluate each hourly event exactly once.
        for event, markets in events.items():
            if (
                len(markets) != 10
                or len({
                    market.get("close_time")
                    for market in markets
                }) != 1
            ):
                continue

            close = parse_time(
                markets[0]["close_time"]
            )

            day = (
                close.astimezone(ET)
                .date()
            )

            decision = (
                close
                - timedelta(minutes=horizon)
            )

            late = (
                utcnow() - decision
            ).total_seconds()

            if (
                event in attempted
                or not 0 <= late <= 35
            ):
                continue

            attempted.add(event)

            alpha = alpha_cache.get(day)

            if alpha is None:
                print(
                    f"SKIP {horizon}m {event}: "
                    "no prepared alpha",
                    flush=True,
                )
                continue

            try:
                quotes = signal_quotes(
                    markets,
                    decision,
                )

                if quotes.empty:
                    print(
                        f"PASS {horizon}m {event}: "
                        "no timely quotes",
                        flush=True,
                    )
                    continue

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
                        f"PASS {horizon}m {event}: "
                        "no edge",
                        flush=True,
                    )
                    continue

                row = selected.iloc[0]
                side = row["side"]

                fair = (
                    float(row["fair_yes"])
                    if side == "YES"
                    else 1 - float(
                        row["fair_yes"]
                    )
                )

                pending.append({
                    "event": event,
                    "ticker": row["ticker"],
                    "side": side,
                    "fair": fair,
                    "day": day,
                    "decision": decision,
                    "close": close,
                    "entry_after": (
                        decision
                        + timedelta(minutes=1)
                    ),
                    "deadline": (
                        decision
                        + timedelta(minutes=3)
                    ),
                })

                print(
                    f"SIGNAL {horizon}m {event} "
                    f"{row['ticker']} "
                    f"{side} "
                    f"fair={fair:.3f}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"SIGNAL ERROR {event}: {exc}",
                    flush=True,
                )

        # Delayed entry and order-book recheck.
        for signal in list(pending):
            now = utcnow()

            if now < signal["entry_after"]:
                continue

            if (
                now > signal["deadline"]
                or (
                    signal["close"] - now
                ).total_seconds() <= 15
            ):
                print(
                    f"EXPIRED {horizon}m "
                    f"{signal['event']}",
                    flush=True,
                )

                pending.remove(signal)
                continue

            try:
                ask = book_ask(
                    signal["ticker"],
                    signal["side"],
                )

                if ask is None:
                    continue

                price, depth = ask

                if not 0.05 <= price <= 0.95:
                    pending.remove(signal)
                    continue

                fee = taker_fee(
                    price,
                    FEE_RATE,
                )

                total = price + fee

                edge = (
                    signal["fair"]
                    - total
                    - BUFFER
                )

                roi = (
                    edge / (total + BUFFER)
                )

                if (
                    edge < MIN_EDGE
                    or roi < MIN_ROI
                ):
                    print(
                        f"FADED {horizon}m "
                        f"{signal['event']} "
                        f"edge={edge:.3f} "
                        f"roi={roi:.2%}",
                        flush=True,
                    )

                    pending.remove(signal)
                    continue

                trade = {
                    "event": signal["event"],
                    "ticker": signal["ticker"],
                    "side": signal["side"],
                    "target_date": (
                        signal["day"].isoformat()
                    ),
                    "decision_time": (
                        signal["decision"]
                        .isoformat()
                    ),
                    "market_close_time": (
                        signal["close"]
                        .isoformat()
                    ),
                    "entry_price": price,
                    "adjusted_probability": (
                        signal["fair"]
                    ),
                    "fee": fee,
                    "total_cost": total,
                    "net_edge": (
                        signal["fair"] - total
                    ),
                    "expected_return_on_risk": (
                        signal["fair"] - total
                    ) / total,
                    "minimum_expected_return_on_risk": MIN_ROI,
                    "minimum_edge": MIN_EDGE,
                    "trade_taken": True,
                }

                if args.live:
                    result = execute_hourly_trade(
                        trade,
                        horizon,
                        depth,
                    )

                    print(
                        f"LIVE SUBMISSION {horizon}m "
                        f"{signal['event']} "
                        f"order={result.get('order_id')}",
                        flush=True,
                    )

                else:
                    count, full_fee, full_edge, full_roi, sizing = (
                        size_and_check(
                            trade,
                            depth,
                        )
                    )

                    paper_positions.append({
                        **signal,
                        "count": count,
                        "cost": (
                            count * price
                            + full_fee
                        ),
                        "settled": False,
                        "pnl": 0.0,
                    })

                    print(
                        f"PAPER {horizon}m "
                        f"{signal['event']} "
                        f"count={count} "
                        f"ask={price:.2f} "
                        f"cost=${count * price + full_fee:.2f} "
                        f"ROI={full_roi:.2%}",
                        flush=True,
                    )

                pending.remove(signal)

            except Exception as exc:
                print(
                    f"ENTRY BLOCKED {horizon}m "
                    f"{signal['event']}: {exc}",
                    flush=True,
                )

                pending.remove(signal)

                # Never continue blindly after
                # a potentially ambiguous live order.
                if args.live:
                    raise

        # Paper settlement checks.
        if (
            not args.live
            and tick - last_settlements >= 60
        ):
            last_settlements = tick

            for position in paper_positions:
                if (
                    position["settled"]
                    or utcnow()
                    < position["close"]
                ):
                    continue

                try:
                    market = get_market_settlement(
                        position["ticker"]
                    )

                    result = market.get("result")

                    if (
                        result not in ("YES", "NO")
                        or market.get("status")
                        not in ("finalized", "settled")
                    ):
                        continue

                    won = (
                        result == position["side"]
                    )

                    pnl = (
                        position["count"]
                        if won else 0
                    ) - position["cost"]

                    position["settled"] = True
                    position["pnl"] = pnl

                    print(
                        f"SETTLED {horizon}m "
                        f"{position['event']} "
                        f"{'WIN' if won else 'LOSS'} "
                        f"pnl=${pnl:+.2f}",
                        flush=True,
                    )

                except Exception as exc:
                    print(
                        "SETTLEMENT CHECK ERROR:",
                        exc,
                        flush=True,
                    )

        if (
            tick - last_heartbeat >= 300
        ):
            print(
                f"HEARTBEAT {horizon}m "
                f"UTC={utcnow().isoformat()} "
                f"pending={len(pending)} "
                f"paper_positions={len(paper_positions)}",
                flush=True,
            )

            last_heartbeat = tick

        time.sleep(args.poll_seconds)
