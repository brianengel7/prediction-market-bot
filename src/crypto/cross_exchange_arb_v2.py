
"""Unified Kalshi + Polymarket US arbitrage scanner.

READ ONLY. NO ORDER SUBMISSION.

Scans:
- Cross-exchange complementary positions
- Kalshi same-contract YES/NO
- Polymarket US same-contract YES/NO
- Kalshi ordered-threshold relationships
- Explicitly declared exhaustive outcome sets
- Explicitly declared mutually exclusive outcome sets

Event-driven WebSocket analysis plus a 1-second heartbeat.

All reported spreads are indicative. Execution and
settlement equivalence are not verified.
"""

import argparse
import asyncio
import json
import os
import time

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_EVEN
from pathlib import Path


D = Decimal
ONE = D("1")

WS_URL = (
    "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
)


def dec(value):
    if isinstance(value, dict):
        value = value.get("value")

    try:
        result = D(str(value))
        return result if result.is_finite() else None
    except (TypeError, ValueError, InvalidOperation):
        return None


def utc(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)

    result = datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )

    if result.tzinfo is None:
        raise ValueError("Timezone missing")

    return result.astimezone(timezone.utc)


def timestamp():
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


def best(levels, high=False):
    good = [
        (p, q)
        for p, q in levels.items()
        if D(0) < p < ONE and q > 0
    ]

    if not good:
        return None

    return max(good) if high else min(good)


def poly_levels(items):
    result = {}

    for item in items or []:
        if not isinstance(item, dict):
            continue

        p = dec(item.get("px", item.get("price")))
        q = dec(item.get("qty", item.get("quantity")))

        if (
            p is not None
            and q is not None
            and D(0) < p < ONE
            and q > 0
        ):
            result[p] = result.get(p, D(0)) + q

    return result


@dataclass(frozen=True)
class Leg:
    venue: str
    ticker: str
    side: str
    price: D
    size: D
    inferred: bool = False


class KalshiBook:
    def __init__(self):
        self.bids = {
            "yes": {},
            "no": {},
        }

        self.ready = False
        self.seen = 0.0

    def apply(self, typ, payload):
        if typ == "orderbook_snapshot":
            update = {}

            for side in ("yes", "no"):
                levels = {}

                for price, quantity in payload.get(
                    side + "_dollars_fp", []
                ):
                    p = dec(price)
                    q = dec(quantity)

                    if (
                        p is None
                        or q is None
                        or not D(0) < p < ONE
                        or q < 0
                    ):
                        raise ValueError(
                            "Malformed Kalshi snapshot"
                        )

                    if q:
                        levels[p] = q

                update[side] = levels

            self.bids = update
            self.ready = True

        else:
            if not self.ready:
                raise RuntimeError(
                    "Kalshi delta before snapshot"
                )

            side = payload.get("side")
            p = dec(payload.get("price_dollars"))
            dq = dec(payload.get("delta_fp"))

            if (
                side not in self.bids
                or p is None
                or dq is None
                or not D(0) < p < ONE
            ):
                raise ValueError(
                    "Malformed Kalshi delta"
                )

            quantity = (
                self.bids[side].get(p, D(0)) + dq
            )

            if quantity < 0:
                raise RuntimeError(
                    "Negative Kalshi depth; reconnect"
                )

            if quantity:
                self.bids[side][p] = quantity
            else:
                self.bids[side].pop(p, None)

        self.seen = time.monotonic()

    def ask(self, ticker, side, max_age):
        if (
            not self.ready
            or time.monotonic() - self.seen > max_age
        ):
            return None

        other = (
            "no" if side == "yes" else "yes"
        )

        bid = best(
            self.bids[other],
            high=True
        )

        if bid is None:
            return None

        return Leg(
            "kalshi",
            ticker,
            side,
            ONE - bid[0],
            bid[1]
        )


class PolyBook:
    def __init__(self):
        self.bids = {}
        self.offers = {}
        self.ready = False
        self.seen = 0.0

    def apply(self, payload):
        state = (
            payload.get("state")
            or payload.get("marketState")
        )

        if state not in (
            "MARKET_STATE_OPEN",
            "OPEN",
        ):
            self.ready = False
            return

        if (
            "bids" not in payload
            or "offers" not in payload
        ):
            return

        bids = poly_levels(payload["bids"])
        offers = poly_levels(payload["offers"])

        bb = best(bids, high=True)
        ba = best(offers)

        # A crossed or locked book cannot safely
        # be considered executable arbitrage.
        if bb and ba and bb[0] >= ba[0]:
            self.ready = False
            return

        self.bids = bids
        self.offers = offers
        self.ready = True
        self.seen = time.monotonic()

    def ask(self, slug, side, max_age):
        if (
            not self.ready
            or time.monotonic() - self.seen > max_age
        ):
            return None

        if side == "yes":
            ask = best(self.offers)

            if ask:
                return Leg(
                    "polymarket_us",
                    slug,
                    side,
                    *ask
                )

            return None

        # Polymarket US uses a single instrument.
        # The opposite-side cost is inferred
        # from the best YES bid.
        bid = best(self.bids, high=True)

        if bid:
            return Leg(
                "polymarket_us",
                slug,
                side,
                ONE - bid[0],
                bid[1],
                True
            )

        return None


@dataclass
class State:
    metadata: dict = field(default_factory=dict)
    books: dict = field(default_factory=dict)

    poly_slug: str | None = None
    poly: PolyBook = field(
        default_factory=PolyBook
    )

    kalshi_btc: str | None = None
    close: datetime | None = None

    pair_status: str = "UNVERIFIED"

    seen_signals: dict = field(
        default_factory=dict
    )
    kalshi_messages: int = 0
    poly_messages: int = 0

    updates: int = 0
    alerts: int = 0
    last_error: str = ""



# V2.1_LIVE_PEAK_TRACKER
# Read-only market-data analysis. No order submission.

def estimate_fee(leg, quantity, args):
    """Estimated one-fill taker fee, not an execution quote."""
    p = leg.price

    if leg.venue == "kalshi":
        rate = getattr(args, "kalshi_rate", D(".07"))
        return (
            rate * quantity * p * (ONE - p)
        ).quantize(D("0.0001"), rounding=ROUND_CEILING)

    if leg.venue == "polymarket_us":
        rate = getattr(args, "poly_rate", D(".0695"))
        return (
            rate * quantity * p * (ONE - p)
        ).quantize(D("0.01"), rounding=ROUND_HALF_EVEN)

    raise ValueError("Unknown exchange " + leg.venue)


def priced_spread(kind, legs, args, extra=None):
    if not legs or any(x is None for x in legs):
        return None

    desired = getattr(args, "eval_qty", D("10"))
    quantity = D(int(min(
        desired, *(x.size for x in legs)
    )))

    if quantity <= 0 or quantity < args.min_qty:
        return None

    cost_per = sum(
        (x.price for x in legs), D(0)
    )

    gross_per = ONE - cost_per

    fees = [
        estimate_fee(x, quantity, args)
        for x in legs
    ]

    total_fees = sum(fees, D(0))

    net_per = (
        gross_per
        - total_fees / quantity
        - args.slip
    )

    return {
        "at": timestamp(),
        "kind": kind,
        "legs": [
            {
                "venue": x.venue,
                "market": x.ticker,
                "side": x.side,
                "ask": str(x.price),
                "visible_qty": str(x.size),
                "opposite_bid_inferred": x.inferred,
            }
            for x in legs
        ],
        "max_visible_pairs": str(
            min(x.size for x in legs)
        ),
        "evaluated_pairs": str(quantity),
        "gross_cents_per_pair": str(
            gross_per * 100
        ),
        "buffered_cents_per_pair": str(
            net_per * 100
        ),
        "estimated_total_taker_fees_dollars": str(
            total_fees
        ),
        "estimated_fees_by_leg_dollars": [
            str(f) for f in fees
        ],
        "estimated_total_net_dollars": str(
            net_per * quantity
        ),
        "slippage_buffer_cents_per_pair": str(
            args.slip * 100
        ),
        "fee_model": (
            "Kalshi 0.0001-up; "
            "Poly US 0.01-half-even; "
            "rates configurable"
        ),
        "fees_verified_for_order": False,
        "settlement_equivalence_verified": False,
        "simultaneous_fills_verified": False,
        "orders_submitted": 0,
        **(extra or {}),
    }


def candidate(kind, legs, args, extra=None):
    row = priced_spread(kind, legs, args, extra)

    if row is None:
        return None

    if (
        D(row["buffered_cents_per_pair"])
        < args.min_edge * 100
    ):
        return None

    return row


class Engine:
    def __init__(self, args, state, output_queue):
        self.args = args
        self.state = state
        self.queue = output_queue

        from src.crypto.arb_fillability_audit import (
            PassiveFillabilityAudit,
        )
        self.fillability = PassiveFillabilityAudit(self)

        from src.crypto.arb_readiness_audit import (
            ExecutionReadinessAudit,
        )
        self.readiness = ExecutionReadinessAudit(
            self, priced_spread
        )

        self.peaks = {}
        self.samples = defaultdict(int)

        # Precompute threshold relationships once
        # per discovery window, not on every update.
        self.threshold_pairs = []
        by_event = defaultdict(list)

        for ticker, market in state.metadata.items():
            if not (
                market.get("strike_type") == "greater"
                and market.get("event_ticker")
                and market.get("floor_strike") is not None
            ):
                continue

            strike = dec(market["floor_strike"])

            if strike is None:
                continue

            key = (
                market["event_ticker"],
                market.get("close_time"),
            )

            by_event[key].append(
                (strike, ticker)
            )

        for members in by_event.values():
            members.sort()

            for i, (low, lt) in enumerate(members):
                for high, ht in members[i + 1:]:
                    if high > low:
                        self.threshold_pairs.append(
                            (lt, ht, low, high)
                        )

    def _queue(self, record):
        if not self.args.log_db:
            return

        try:
            self.queue.put_nowait(record)

        except asyncio.QueueFull:
            print(
                "LOGGER QUEUE FULL - record dropped",
                flush=True,
            )

    def emit(self, record):
        signature = (
            record["kind"],
            tuple(
                (
                    x["market"],
                    x["side"],
                    x["ask"],
                )
                for x in record["legs"]
            ),
        )

        now = time.monotonic()

        if (
            now - self.state.seen_signals.get(
                signature, -1e10
            )
            < self.args.cooldown
        ):
            return

        self.state.seen_signals[signature] = now
        self.state.alerts += 1

        print(
            "OPPORTUNITY "
            + json.dumps(
                record,
                separators=(",", ":"),
            ),
            flush=True,
        )

        self._queue(record)

    def consider(
        self,
        kind,
        legs,
        extra=None,
        family=None,
    ):
        # FAST_EXECUTION_ONLY_OPT_V1
        if getattr(self.args, "execution_only", False):
            return
        row = priced_spread(
            kind, legs, self.args, extra
        )

        self.readiness.observe(
            family or kind, kind, legs, row
        )
        self.fillability.observe(
            family or kind, kind, legs, row
        )

        if row is None:
            return

        key = family or kind

        self.samples[key] += 1

        gross = D(row["gross_cents_per_pair"])
        net = D(row["buffered_cents_per_pair"])

        old = self.peaks.get(key)

        improved_gross = (
            old is None
            or gross > old["best_gross"]
        )

        improved_net = (
            old is None
            or net > old["best_net"]
        )

        if old is None:
            self.peaks[key] = {
                "best_gross": gross,
                "best_net": net,
                "gross_at": row["at"],
                "net_at": row["at"],
                "last_gross": gross,
                "last_net": net,
                "observations": 1,
            }

        else:
            old["observations"] += 1
            old["last_gross"] = gross
            old["last_net"] = net

            if improved_gross:
                old["best_gross"] = gross
                old["gross_at"] = row["at"]

            if improved_net:
                old["best_net"] = net
                old["net_at"] = row["at"]

        if improved_gross or improved_net:
            peak = dict(row)

            peak["kind"] = "PEAK_" + key

            peak["peak_type"] = (
                "BOTH"
                if improved_gross and improved_net
                else "GROSS"
                if improved_gross
                else "NET"
            )

            # Every new maximum is sent to the
            # asynchronous logger when enabled.
            self._queue(peak)

            if gross > 0 or net > 0:
                print(
                    f"NEW_PEAK {key} "
                    f"gross={gross:+.3f}c "
                    f"net_est={net:+.3f}c "
                    f"qty={row['evaluated_pairs']} "
                    f"at={row['at']}",
                    flush=True,
                )

        if net >= self.args.min_edge * 100:
            self.emit(row)

    def print_peaks(self):
        self.readiness.print_summary()
        self.fillability.print_summary()
        state = self.state

        print(
            f"FEEDS "
            f"kalshi_messages={state.kalshi_messages} "
            f"poly_messages={state.poly_messages} "
            f"books={len(state.books)} "
            f"threshold_pairs={len(self.threshold_pairs)}",
            flush=True,
        )

        if not self.peaks:
            print(
                "PEAKS no synchronized eligible "
                "two-leg quotes yet",
                flush=True,
            )
            return

        for key, peak in sorted(self.peaks.items()):
            print(
                f"PEAKS {key} "
                f"checks={peak['observations']} "
                f"max_gross="
                f"{peak['best_gross']:+.3f}c "
                f"max_net_est="
                f"{peak['best_net']:+.3f}c "
                f"current_gross="
                f"{peak['last_gross']:+.3f}c "
                f"current_net_est="
                f"{peak['last_net']:+.3f}c",
                flush=True,
            )

    def check(self, changed=None):
        state = self.state
        args = self.args

        state.updates += 1

        if len(state.seen_signals) > 5000:
            state.seen_signals.clear()

        # Same-contract Kalshi:
        # evaluate only the changed ticker.
        for ticker, book in state.books.items():
            if (
                changed is not None
                and changed != ticker
            ):
                continue

            self.consider(
                "KALSHI_SAME_BINARY",
                [
                    book.ask(
                        ticker, "yes", args.max_age
                    ),
                    book.ask(
                        ticker, "no", args.max_age
                    ),
                ],
            )

        # Same-contract Polymarket US.
        if (
            state.poly_slug
            and changed in (None, "POLY")
        ):
            self.consider(
                "POLY_US_SAME_BINARY",
                [
                    state.poly.ask(
                        state.poly_slug,
                        "yes",
                        args.max_age,
                    ),
                    state.poly.ask(
                        state.poly_slug,
                        "no",
                        args.max_age,
                    ),
                ],
            )

        # Cross-exchange BTC:
        # only BTC Kalshi or Poly book updates.
        cross_ready = (
            state.kalshi_btc in state.books
            and state.poly_slug
            and state.close
            and changed in (
                None,
                "POLY",
                state.kalshi_btc,
            )
            and datetime.now(timezone.utc)
            < state.close - timedelta(
                seconds=args.close_buffer
            )
        )

        if cross_ready:
            kb = state.books[state.kalshi_btc]

            for ks, ps in (
                ("yes", "no"),
                ("no", "yes"),
            ):
                self.consider(
                    "CROSS_EXCHANGE",
                    [
                        kb.ask(
                            state.kalshi_btc,
                            ks,
                            args.max_age,
                        ),
                        state.poly.ask(
                            state.poly_slug,
                            ps,
                            args.max_age,
                        ),
                    ],
                    {
                        "rule_match": state.pair_status,
                    },
                    family=(
                        f"CROSS_K_{ks.upper()}"
                        f"_P_{ps.upper()}"
                    ),
                )

        # Related Kalshi threshold contracts.
        for lt, ht, low, high in self.threshold_pairs:
            if (
                changed is not None
                and changed not in (lt, ht)
            ):
                continue

            if (
                lt not in state.books
                or ht not in state.books
            ):
                continue

            self.consider(
                "KALSHI_ORDERED_THRESHOLDS",
                [
                    state.books[lt].ask(
                        lt, "yes", args.max_age
                    ),
                    state.books[ht].ask(
                        ht, "no", args.max_age
                    ),
                ],
                {
                    "low": str(low),
                    "high": str(high),
                },
            )

        # User-declared exhaustive and
        # mutually exclusive contract groups.
        for kind, sets, side in (
            (
                "DECLARED_EXHAUSTIVE",
                args.exhaustive_set,
                "yes",
            ),
            (
                "DECLARED_EXCLUSIVE",
                args.exclusive_set,
                "no",
            ),
        ):
            for tickers in sets:
                if (
                    changed is not None
                    and changed not in tickers
                ):
                    continue

                if not all(
                    t in state.books
                    and t in state.metadata
                    for t in tickers
                ):
                    continue

                events = {
                    state.metadata[t].get(
                        "event_ticker"
                    )
                    for t in tickers
                }

                closes = {
                    state.metadata[t].get(
                        "close_time"
                    )
                    for t in tickers
                }

                if (
                    len(events) != 1
                    or len(closes) != 1
                    or None in events
                    or None in closes
                ):
                    continue

                self.consider(
                    kind,
                    [
                        state.books[t].ask(
                            t,
                            side,
                            args.max_age,
                        )
                        for t in tickers
                    ],
                    {
                        "group": list(tickers),
                        "group_verified_by_exchange": False,
                    },
                )


async def database_worker(queue):
    from src.database.db import get_connection

    while True:
        first = await queue.get()
        batch = [first]

        while len(batch) < 100:
            try:
                batch.append(
                    queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                break

        try:
            def save():
                with get_connection() as db:
                    db.execute("""
                        CREATE TABLE IF NOT EXISTS
                        arb_v2_signals (
                            id TEXT PRIMARY KEY,
                            detected_utc TEXT NOT NULL,
                            kind TEXT NOT NULL,
                            payload_json TEXT NOT NULL
                        )
                    """)

                    import uuid

                    for row in batch:
                        db.execute("""
                            INSERT INTO arb_v2_signals
                            (
                                id,
                                detected_utc,
                                kind,
                                payload_json
                            )
                            VALUES (?,?,?,?)
                        """, (
                            str(uuid.uuid4()),
                            row["at"],
                            row["kind"],
                            json.dumps(row)
                        ))

            await asyncio.to_thread(save)

        except Exception as exc:
            print(
                f"LOGGER ERROR "
                f"{type(exc).__name__}: {exc}",
                flush=True
            )

        finally:
            for _ in batch:
                queue.task_done()


def current_window(now):
    seconds = int(now.timestamp())

    start = datetime.fromtimestamp(
        seconds - seconds % 900,
        tz=timezone.utc
    )

    return (
        start,
        start + timedelta(minutes=15)
    )


def poly_slug(start):
    return (
        "btc-updown-15m-"
        + start.strftime(
            "%Y-%m-%d-%H%Mz"
        ).lower()
    )


def discover_poly(start, verbose=True):
    """Find current Polymarket US BTC market."""
    from polymarket_us import PolymarketUS

    base_slug = poly_slug(start)

    # The cpc- prefix was observed in our live feed.
    candidates = (
        "cpc-" + base_slug,
        base_slug,
    )

    client = PolymarketUS()

    try:
        event_markets = []

        try:
            payload = client.events.retrieve_by_slug(
                base_slug
            )

            event = (
                payload.get("event", payload)
                if isinstance(payload, dict)
                else {}
            )

            if isinstance(event, dict):
                event_markets = (
                    event.get("markets") or []
                )

            if (
                not event_markets
                and isinstance(payload, dict)
            ):
                event_markets = (
                    payload.get("markets") or []
                )

            if isinstance(event_markets, dict):
                event_markets = [event_markets]

        except Exception as exc:
            if verbose:
                print(
                    "POLY event lookup: "
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:100]}",
                    flush=True
                )

        for market in event_markets:
            if not isinstance(market, dict):
                continue

            slug = market.get("slug")

            if (
                slug in candidates
                and market.get("closed") is not True
                and market.get("active") is not False
            ):
                return slug

        # Fallback: directly retrieve the market
        # instead of depending on event.market links.

        errors = []

        for candidate in candidates:
            try:
                payload = (
                    client.markets.retrieve_by_slug(
                        candidate
                    )
                )

                market = (
                    payload.get("market", payload)
                    if isinstance(payload, dict)
                    else {}
                )

                if (
                    isinstance(market, dict)
                    and market.get("slug") == candidate
                    and market.get("closed") is not True
                    and market.get("active") is not False
                ):
                    if verbose:
                        print(
                            "POLY direct lookup matched "
                            + candidate,
                            flush=True
                        )

                    return candidate

            except Exception as exc:
                errors.append(
                    type(exc).__name__
                )

        if verbose:
            print(
                f"POLY NOT LISTED {base_slug}: "
                f"event_markets={len(event_markets)} "
                f"direct_errors="
                f"{','.join(errors) or 'none'}",
                flush=True
            )

        return None

    finally:
        client.close()


def discover(args):
    """Market discovery outside the hot path."""

    from src.kalshi.client import get_markets

    start, close = current_window(
        datetime.now(timezone.utc)
    )

    series = (
        ["KXBTC15M"]
        + args.extra_kalshi_series
    )

    result = {}

    for name in dict.fromkeys(series):
        try:
            response = get_markets(
                series_ticker=name,
                status="open",
            )
        except Exception as exc:
            print(
                f"DISCOVERY skip {name}: "
                f"{type(exc).__name__}: "
                f"{str(exc)[:100]}",
                flush=True,
            )
            continue

        markets = response.get(
            "markets", []
        )

        if name == "KXBTC15M":
            markets = [
                m
                for m in markets
                if (
                    m.get("close_time")
                    and abs(
                        (
                            utc(m["close_time"]) - close
                        ).total_seconds()
                    ) < 1
                )
            ]

        else:
            markets = sorted(
                markets,
                key=lambda m: (
                    m.get("close_time") or ""
                )
            )[:args.max_extra]

        for market in markets:
            ticker = market.get("ticker")

            if (
                ticker
                and market.get("event_ticker")
            ):
                result[ticker] = market

    btc = [
        ticker
        for ticker, market in result.items()
        if (
            ticker.startswith("KXBTC15M-")
            and abs(
                (
                    utc(market["close_time"]) - close
                ).total_seconds()
            ) < 1
        )
    ]

    if len(btc) != 1:
        print(
            f"DISCOVERY BTC markets for "
            f"{close.isoformat()}: {len(btc)} "
            "(cross paused)",
            flush=True
        )

        btc = []

    poly_market = discover_poly(start) if btc else None

    return (
        result,
        btc[0] if btc else None,
        poly_market,
        close
    )


def find_current_btc_ticker(close):
    """Find a BTC contract listed after window rollover."""
    from src.kalshi.client import get_markets

    response = get_markets(
        series_ticker="KXBTC15M",
        status="open",
    )

    matches = []

    for market in response.get("markets", []):
        try:
            ticker = market["ticker"]
            market_close = utc(market["close_time"])
        except (KeyError, TypeError, ValueError):
            continue

        if abs(
            (market_close - close).total_seconds()
        ) < 1:
            matches.append(ticker)

    return matches[0] if len(matches) == 1 else None


async def kalshi_stream(state, engine):
    import websockets

    from src.kalshi.trading_client import (
        build_auth_headers
    )

    tickers = list(state.metadata)

    if not tickers:
        return

    while True:
        state.books = {
            t: KalshiBook()
            for t in tickers
        }

        seq_by_sid = {}

        try:
            headers = build_auth_headers(
                "GET", WS_URL
            )

            async with websockets.connect(
                WS_URL,
                additional_headers=headers,
                ping_interval=15,
                ping_timeout=12,
                max_queue=4096,
            ) as ws:

                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": [
                            "orderbook_delta"
                        ],
                        "market_tickers": tickers
                    }
                }))

                print(
                    f"KALSHI connected; "
                    f"books={len(tickers)}",
                    flush=True
                )

                while True:
                    event = json.loads(
                        await ws.recv()
                    )

                    typ = event.get("type")

                    if typ == "error":
                        raise RuntimeError(
                            str(event.get("msg"))
                        )

                    if typ not in (
                        "orderbook_snapshot",
                        "orderbook_delta"
                    ):
                        continue

                    sid = event.get("sid")
                    seq = event.get("seq")

                    if (
                        sid is not None
                        and seq is not None
                    ):
                        if (
                            typ == "orderbook_delta"
                            and sid in seq_by_sid
                            and seq != seq_by_sid[sid] + 1
                        ):
                            raise RuntimeError(
                                "Sequence gap; "
                                "books invalidated"
                            )

                        seq_by_sid[sid] = seq

                    msg = event["msg"]
                    ticker = msg.get(
                        "market_ticker"
                    )

                    if ticker not in state.books:
                        continue

                    state.books[ticker].apply(
                        typ, msg
                    )

                    state.kalshi_messages += 1

                    engine.check(ticker)

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            state.last_error = (
                f"Kalshi {type(exc).__name__}: "
                f"{str(exc)[:120]}"
            )

            print(
                "KALSHI reconnect: "
                + state.last_error,
                flush=True
            )

            await asyncio.sleep(1)


async def poly_stream(state, engine):
    from polymarket_us import PolymarketUS

    key_id = os.getenv("POLYMARKET_KEY_ID")
    secret_key = os.getenv("POLYMARKET_SECRET_KEY")

    if not key_id or not secret_key:
        state.last_error = "Missing Polymarket US credentials"
        print(
            "POLY DISABLED: API credentials missing",
            flush=True
        )
        return

    loop = asyncio.get_running_loop()

    while True:
        state.poly = PolyBook()
        client = None
        ws = None
        frames = 0

        try:
            client = PolymarketUS(
                key_id=key_id,
                secret_key=secret_key,
            )

            ws = client.ws.markets()

            def handle(frame):
                nonlocal frames

                data = frame.get("marketData") or {}
                slug = data.get("marketSlug")

                frames += 1

                if frames == 1:
                    print(
                        "POLY FIRST FRAME "
                        f"slug={slug} "
                        f"state={data.get('state')} "
                        f"bids={len(data.get('bids') or [])} "
                        f"offers={len(data.get('offers') or [])}",
                        flush=True
                    )

                if slug != state.poly_slug:
                    return

                def update():
                    try:
                        state.poly.apply(data)
                        state.poly_messages += 1
                        engine.check("POLY")
                    except Exception as exc:
                        state.last_error = (
                            f"POLY PARSE: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        print(state.last_error, flush=True)

                loop.call_soon_threadsafe(update)

            def on_error(error):
                state.last_error = (
                    "POLY WS: " + str(error)[:180]
                )
                print(state.last_error, flush=True)

            ws.on("market_data", handle)
            ws.on("error", on_error)

            print(
                f"POLY connecting: {state.poly_slug}",
                flush=True
            )

            await asyncio.wait_for(ws.connect(), timeout=15)

            print(
                "POLY WebSocket connected",
                flush=True
            )

            await asyncio.wait_for(
                ws.subscribe_market_data(
                    "unified-arb",
                    [state.poly_slug]
                ),
                timeout=15
            )

            print(
                "POLY subscribed to live order book",
                flush=True
            )

            subscribed_at = time.monotonic()

            while True:
                await asyncio.sleep(1)

                if (
                    not state.poly.ready
                    and time.monotonic() - subscribed_at > 15
                ):
                    raise TimeoutError(
                        "No usable Polymarket order book "
                        "within 15 seconds"
                    )

                if (
                    state.poly.ready
                    and time.monotonic()
                    - state.poly.seen > 30
                ):
                    raise TimeoutError(
                        "Polymarket order book stale"
                    )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            state.last_error = (
                f"POLY {type(exc).__name__}: "
                f"{str(exc)[:180]}"
            )
            print(
                "POLY reconnect: " + state.last_error,
                flush=True
            )

            await asyncio.sleep(2)

        finally:
            state.poly.ready = False

            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass

            if client is not None:
                client.close()


def print_spread_diagnostics(state, args):
    """Show actual spreads without changing signal logic."""
    kbook = (
        state.books.get(state.kalshi_btc)
        if state.kalshi_btc else None
    )
    pbook = state.poly if state.poly_slug else None

    def get_leg(book, ticker, side):
        if book is None or not ticker:
            return None
        try:
            return book.ask(ticker, side, float("inf"))
        except (ValueError, AttributeError):
            return None

    ky = get_leg(kbook, state.kalshi_btc, "yes")
    kn = get_leg(kbook, state.kalshi_btc, "no")
    py = get_leg(pbook, state.poly_slug, "yes")
    pn = get_leg(pbook, state.poly_slug, "no")

    def show(leg):
        if leg is None:
            return "MISSING"
        return f"{leg.price * 100:.2f}c x{leg.size}"

    now = time.monotonic()
    k_fresh = bool(
        kbook and kbook.ready
        and now - kbook.seen <= args.max_age
    )
    p_fresh = bool(
        pbook and pbook.ready
        and now - pbook.seen <= args.max_age
    )

    print(
        f"BOOKS K_YES={show(ky)} K_NO={show(kn)} "
        f"P_YES={show(py)} "
        f"P_NO(inferred)={show(pn)} "
        f"fresh_K={k_fresh} fresh_P={p_fresh}",
        flush=True,
    )

    def report(name, a, b, fees, fresh):
        if a is None or b is None:
            print(
                f"SPREAD {name} unavailable: missing side/depth",
                flush=True,
            )
            return

        gross = D(1) - a.price - b.price
        buffered = gross - fees - args.slip
        qty = min(a.size, b.size)

        status = (
            "STALE" if not fresh else
            "LOW_DEPTH" if qty < args.min_qty else
            "BELOW_THRESHOLD" if buffered < args.min_edge else
            "PRICE_PASSES_UNVERIFIED"
        )

        print(
            f"SPREAD {name} "
            f"gross={gross * 100:+.2f}c "
            f"buffered={buffered * 100:+.2f}c "
            f"qty={qty} status={status}",
            flush=True,
        )

    report(
        "K_YES+P_NO", ky, pn,
        args.k_fee + args.p_fee,
        k_fresh and p_fresh,
    )
    report(
        "K_NO+P_YES", kn, py,
        args.k_fee + args.p_fee,
        k_fresh and p_fresh,
    )
    report(
        "K_YES+K_NO", ky, kn,
        args.k_fee * 2, k_fresh,
    )
    report(
        "P_YES+P_NO", py, pn,
        args.p_fee * 2, p_fresh,
    )


async def run(args):
    from dotenv import load_dotenv

    load_dotenv(
        dotenv_path=(
            Path(__file__).resolve().parents[2]
            / ".env"
        )
    )

    if (
        args.log_db
        and not os.getenv(
            "PREDICTION_DATABASE_URL"
        )
    ):
        raise RuntimeError(
            "--log-db requires "
            "PREDICTION_DATABASE_URL"
        )

    output_queue = asyncio.Queue(
        maxsize=1000
    )

    deadline = (
        time.monotonic()
        + args.minutes * 60
        if args.minutes
        else float("inf")
    )

    logger = (
        asyncio.create_task(
            database_worker(output_queue)
        )
        if args.log_db
        else None
    )

    print(
        "UNIFIED ARBITRAGE V2 | READ ONLY "
        "| event-driven + 1s heartbeat",
        flush=True
    )

    print(
        "Fees/slippage are USER ASSUMPTIONS, "
        "not live fee quotes",
        flush=True
    )

    try:
        while time.monotonic() < deadline:
            try:
                (
                    metadata,
                    btc,
                    poly,
                    close
                ) = await asyncio.to_thread(
                    discover, args
                )

                if not metadata:
                    print(
                        "No Kalshi books; "
                        "retry in 10s",
                        flush=True
                    )

                    await asyncio.sleep(10)
                    continue

                state = State(
                    metadata=metadata,
                    kalshi_btc=btc,
                    poly_slug=poly,
                    close=close,
                    pair_status=(
                        "WINDOW_MATCH_ONLY_"
                        "TARGET_NOT_VERIFIED"
                    )
                )

                engine = Engine(
                    args, state, output_queue
                )

                print(
                    f"WINDOW close={close.isoformat()} "
                    f"kalshi={btc} "
                    f"poly={poly or 'NONE'} "
                    f"extra_books="
                    f"{len(metadata) - (1 if btc else 0)}",
                    flush=True
                )

                tasks = [
                    asyncio.create_task(
                        kalshi_stream(
                            state, engine
                        )
                    )
                ]

                if poly:
                    tasks.append(
                        asyncio.create_task(
                            poly_stream(
                                state, engine
                            )
                        )
                    )

                next_status = 0.0
                next_poly_retry = 0.0
                next_btc_retry = 0.0
                restart_for_btc = False

                try:
                    while time.monotonic() < deadline:
                        await asyncio.sleep(1)

                        # BTC contracts may appear shortly
                        # after the new 15-minute window starts.
                        # Do not disconnect weather feeds
                        # while checking for late listings.
                        if (
                            not state.kalshi_btc
                            and time.monotonic() >= next_btc_retry
                            and datetime.now(timezone.utc)
                            < close - timedelta(
                                seconds=args.close_buffer
                            )
                        ):
                            next_btc_retry = (
                                time.monotonic() + 5
                            )

                            try:
                                found = await asyncio.to_thread(
                                    find_current_btc_ticker,
                                    close,
                                )
                            except Exception as exc:
                                found = None
                                print(
                                    "BTC RETRY ERROR "
                                    f"{type(exc).__name__}: {exc}",
                                    flush=True,
                                )

                            if found:
                                print(
                                    f"BTC HOT-DISCOVER {found} "
                                    "| rebuilding subscriptions once",
                                    flush=True,
                                )
                                restart_for_btc = True
                                break

                        # Heartbeat analysis supplements
                        # immediate WebSocket-triggered checks.
                        engine.check()

                        # Keep Kalshi connected while waiting
                        # for Polymarket's new market listing.
                        if (
                            state.kalshi_btc
                            and not state.poly_slug
                            and time.monotonic() >= next_poly_retry
                            and datetime.now(timezone.utc)
                            < close - timedelta(
                                seconds=args.close_buffer
                            )
                        ):
                            next_poly_retry = (
                                time.monotonic() + 5
                            )

                            try:
                                found = await asyncio.to_thread(
                                    discover_poly,
                                    close - timedelta(minutes=15),
                                    False,
                                )
                            except Exception as exc:
                                print(
                                    "POLY retry error: "
                                    f"{type(exc).__name__}: {exc}",
                                    flush=True
                                )
                                found = None

                            if found:
                                state.poly_slug = found

                                print(
                                    f"POLY HOT-JOIN: {found}",
                                    flush=True
                                )

                                tasks.append(
                                    asyncio.create_task(
                                        poly_stream(
                                            state, engine
                                        )
                                    )
                                )

                        if (
                            time.monotonic()
                            >= next_status
                        ):
                            next_status = (
                                time.monotonic()
                                + args.status_every
                            )

                            k = (
                                state.books.get(btc)
                                if btc
                                else None
                            )

                            k_age = (
                                round(
                                    time.monotonic()
                                    - k.seen,
                                    2
                                )
                                if k and k.ready
                                else None
                            )

                            p_age = (
                                round(
                                    time.monotonic()
                                    - state.poly.seen,
                                    2
                                )
                                if state.poly.ready
                                else None
                            )

                            print(
                                f"HEARTBEAT {timestamp()} "
                                f"updates={state.updates} "
                                f"alerts={state.alerts} "
                                f"kalshi_age={k_age}s "
                                f"poly_age={p_age}s "
                                f"error="
                                f"{state.last_error or 'none'}",
                                flush=True
                            )
                            print_spread_diagnostics(state, args)

                            engine.print_peaks()
                        if (
                            datetime.now(timezone.utc)
                            >= close
                            - timedelta(
                                seconds=args.close_buffer
                            )
                        ):
                            break

                finally:
                    for task in tasks:
                        task.cancel()

                    await asyncio.gather(
                        *tasks,
                        return_exceptions=True
                    )

                if restart_for_btc:
                    continue

                wait_s = max(
                    0.0,
                    (
                        close - datetime.now(timezone.utc)
                    ).total_seconds() + 0.2,
                )

                if wait_s:
                    print(
                        f"ROLLOVER: waiting {wait_s:.1f}s "
                        "for next window",
                        flush=True
                    )

                    await asyncio.sleep(
                        min(
                            wait_s,
                            max(
                                0.0,
                                deadline - time.monotonic()
                            )
                        )
                    )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print(
                    f"RUNNER ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True
                )

                await asyncio.sleep(5)

    finally:
        if logger:
            try:
                await asyncio.wait_for(
                    output_queue.join(),
                    timeout=8
                )

            except asyncio.TimeoutError:
                print(
                    "LOGGER still pending after 8 seconds",
                    flush=True
                )

            logger.cancel()

            await asyncio.gather(
                logger,
                return_exceptions=True
            )

    print(
        "Stopped. No orders placed.",
        flush=True
    )


def self_test():
    class Args:
        min_qty = D(1)
        min_edge = D(".01")
        k_fee = D(".00")
        p_fee = D(".00")
        slip = D(".00")
        max_age = 10
        cooldown = 1
        exhaustive_set = []
        exclusive_set = []
        log_db = False
        close_buffer = 5

    args = Args()

    k = KalshiBook()

    k.apply(
        "orderbook_snapshot",
        {
            "yes_dollars_fp": [
                [".55", "10"]
            ],
            "no_dollars_fp": [
                [".60", "10"]
            ]
        }
    )

    assert (
        k.ask("T", "yes", 10).price
        == D(".40")
    )

    assert (
        k.ask("T", "no", 10).price
        == D(".45")
    )

    same = candidate(
        "SAME",
        [
            k.ask("T", "yes", 10),
            k.ask("T", "no", 10)
        ],
        args
    )

    assert (
        same["gross_cents_per_pair"]
        == "15.00"
    )

    k.apply(
        "orderbook_delta",
        {
            "side": "no",
            "price_dollars": ".60",
            "delta_fp": "-10"
        }
    )

    assert (
        k.ask("T", "yes", 10)
        is None
    )

    p = PolyBook()

    p.apply({
        "state": "MARKET_STATE_OPEN",
        "bids": [
            {
                "px": {"value": ".95"},
                "qty": "4"
            }
        ],
        "offers": [
            {
                "px": {"value": ".96"},
                "qty": "5"
            }
        ]
    })

    assert (
        p.ask("P", "no", 10).price
        == D(".05")
    )

    assert candidate(
        "POLY_SAME",
        [
            p.ask("P", "yes", 10),
            p.ask("P", "no", 10)
        ],
        args
    ) is None

    ky = Leg(
        "kalshi",
        "T",
        "yes",
        D(".85"),
        D("3")
    )

    cross = candidate(
        "CROSS",
        [
            ky,
            p.ask("P", "no", 10)
        ],
        args
    )

    assert (
        cross["gross_cents_per_pair"]
        == "10.00"
    )

    assert current_window(
        utc("2026-10-12T18:23:00Z")
    )[0] == utc(
        "2026-10-12T18:15:00Z"
    )

    assert poly_slug(
        utc("2026-10-12T18:15:00Z")
    ) == (
        "btc-updown-15m-"
        "2026-10-12-1815z"
    )

    fee_args = type(
        "FeeArgs",
        (),
        {
            "eval_qty": D("10"),
            "min_qty": D("1"),
            "kalshi_rate": D("0.07"),
            "poly_rate": D("0.0695"),
            "slip": D("0"),
            "min_edge": D("0"),
        },
    )()

    row = priced_spread(
        "FEE_TEST",
        [
            Leg(
                "kalshi", "K", "yes",
                D(".50"), D("20")
            ),
            Leg(
                "polymarket_us", "P", "no",
                D(".49"), D("20")
            ),
        ],
        fee_args,
    )

    assert row["evaluated_pairs"] == "10"
    assert row["gross_cents_per_pair"] == "1.00"
    assert (
        D(row["estimated_total_taker_fees_dollars"])
        > 0
    )

    print(
        "PASS: Kalshi book and delta, "
        "Polymarket bid/offer, "
        "same/cross arithmetic, "
        "market-window naming"
    )


def parse_groups(values):
    out = []

    for value in values:
        members = tuple(
            t.strip()
            for t in value.split(",")
            if t.strip()
        )

        if (
            len(members) < 2
            or len(members) != len(set(members))
        ):
            raise ValueError(
                "Group must contain at least "
                "two distinct tickers"
            )

        out.append(members)

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-test",
        action="store_true"
    )

    ap.add_argument(
        "--minutes",
        type=float,
        default=0
    )

    ap.add_argument(
        "--max-age",
        type=float,
        default=1.0
    )

    ap.add_argument(
        "--min-edge",
        type=D,
        default=D(".02")
    )

    ap.add_argument(
        "--k-fee",
        type=D,
        default=D(".02")
    )

    ap.add_argument(
        "--p-fee",
        type=D,
        default=D(".02")
    )

    ap.add_argument(
        "--slip",
        type=D,
        default=D(".01")
    )

    ap.add_argument(
        "--min-qty",
        type=D,
        default=D("1")
    )

    ap.add_argument(
        "--cooldown",
        type=float,
        default=3
    )

    ap.add_argument(
        "--status-every",
        type=float,
        default=10
    )

    ap.add_argument(
        "--close-buffer",
        type=float,
        default=5
    )

    ap.add_argument(
        "--extra-kalshi-series",
        action="append",
        default=[]
    )

    ap.add_argument(
        "--max-extra",
        type=int,
        default=12
    )

    ap.add_argument(
        "--exhaustive-set",
        action="append",
        default=[],
        help=(
            "Comma-separated vetted tickers "
            "covering every outcome"
        )
    )

    ap.add_argument(
        "--exclusive-set",
        action="append",
        default=[],
        help=(
            "Comma-separated vetted "
            "mutually exclusive tickers"
        )
    )

    ap.add_argument(
        "--log-db",
        action="store_true"
    )

    ap.add_argument(
        "--eval-qty",
        type=D,
        default=D("10"),
    )
    ap.add_argument(
        "--kalshi-rate",
        type=D,
        default=D("0.07"),
    )
    ap.add_argument(
        "--poly-rate",
        type=D,
        default=D("0.0695"),
    )
    ap.add_argument(
        "--no-weather",
        action="store_true",
    )
    args = ap.parse_args()
    if not args.no_weather:
        args.extra_kalshi_series = list(
            dict.fromkeys(
                [
                    "KXTEMPNYCHS",
                    "KXTEMPMIAH",
                ]
                + args.extra_kalshi_series
            )
        )


    if args.self_test:
        return self_test()

    args.exhaustive_set = parse_groups(
        args.exhaustive_set
    )

    args.exclusive_set = parse_groups(
        args.exclusive_set
    )

    if (
        args.minutes < 0
        or args.max_age <= 0
        or args.min_qty < 1
        or args.status_every < 1
        or args.cooldown < 0
        or args.close_buffer < 0
        or args.max_extra < 1 or args.eval_qty < 1
        or min(
            args.min_edge,
            args.k_fee,
            args.p_fee,
            args.slip
        ) < 0
    ):
        ap.error("Invalid parameter")

    try:
        asyncio.run(run(args))

    except KeyboardInterrupt:
        print(
            "Stopped. No orders placed.",
            flush=True
        )


if __name__ == "__main__":
    main()
