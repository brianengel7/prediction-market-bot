"""Kalshi multi-city daily-high two-YES basket trader.

Default: continuous scanning and trading with configurable limits.
--one-trade / --one-contract: exactly one contract per leg;
one attempt total, then stop.

KALSHI_LIVE_TRADING_ENABLED=true is required for real orders.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import json
import os
import time
import uuid

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
import websockets

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
    ed25519,
    padding,
)
from dotenv import load_dotenv

from src.backtest.weather_yes_basket_scanner import (
    Book,
    WS_URL,
    scan,
)
from src.backtest.weather_multicity_capacity_scanner import CITY
from src.kalshi.fees import calculate_taker_fee
from src.backtest.weather_loss_management import (
    LossMonitor, held_cities, outstanding_exits,
    recover_loss_exits,
)


ET = ZoneInfo("America/New_York")
BASE = "https://external-api.kalshi.com/trade-api/v2"

STATE_FILE = Path(".weather_multicity_live_state.json")
LOCK_FILE = Path(".weather_multicity_live_process.lock")

TOP_TWO_MIN = D("0.75")
BASKET_CAP = D("0.96")
ONE = D("1")

SAFE_STATUSES = {
    "COMPLETE",
    "FIRST_UNFILLED",
    "UNWOUND",
    "NOT_SUBMITTED",
}


def log(msg):
    timestamp = datetime.now(ET).strftime(
        "%H:%M:%S.%f"
    )[:-3]
    print(f"[{timestamp} ET] {msg}", flush=True)


def usd(value):
    if isinstance(value, dict):
        value = value.get("value")
    return D(str(value))


def fee(ask, qty):
    return usd(
        calculate_taker_fee(
            price=float(ask),
            contracts=qty,
            multiplier=1,
        )
    )


def quotes(book):
    if not getattr(book, "ready", False):
        return None

    yes = [
        (usd(price), usd(qty))
        for price, qty in book.levels["yes"].items()
        if usd(qty) > 0
    ]

    no = [
        (usd(price), usd(qty))
        for price, qty in book.levels["no"].items()
        if usd(qty) > 0
    ]

    if not yes or not no:
        return None

    yes_bid, yes_depth = max(yes)
    no_bid, no_depth = max(no)

    yes_ask = ONE - no_bid

    if not D(0) < yes_bid <= yes_ask < ONE:
        return None

    return yes_bid, yes_ask, yes_depth, no_depth


@dataclass(frozen=True)
class Opportunity:
    city: str
    event: str
    legs: tuple[str, str]
    asks: tuple[D, D]
    mids: D
    qty: int
    allin: D

    @property
    def key(self):
        # One attempted basket per city/event.
        return f"{self.city}:{self.event}"

    @property
    def ask_total(self):
        return sum(self.asks, D(0))


def evaluate(
    city,
    event,
    books,
    tickers,
    *,
    one_trade,
    max_pairs,
    max_cost,
):
    if any(not books[t].ready for t in tickers):
        return None

    a, b, _, valid = scan(books, tickers)

    if (
        not valid
        or not isinstance(a, dict)
        or not isinstance(b, dict)
    ):
        return None

    # Ranking reversals are the same basket.
    legs = tuple(sorted((
        a["ticker"],
        b["ticker"],
    )))

    if (
        legs[0] == legs[1]
        or any(t not in books for t in legs)
    ):
        return None

    first, second = (
        quotes(books[t])
        for t in legs
    )

    if first is None or second is None:
        return None

    # Only the selected two require two-sided quotes.
    mids = (
        first[0] + first[1]
        + second[0] + second[1]
    ) / 2

    asks = (first[1], second[1])
    ask_total = sum(asks, D(0))

    if mids < TOP_TWO_MIN:
        return None

    if ask_total >= BASKET_CAP:
        return None

    # Available pairs at the current best YES asks.
    depth = int(min(first[3], second[3]))

    if depth < 1:
        return None

    largest = min(
        depth,
        1 if one_trade else max_pairs,
    )

    # Choose the largest quantity satisfying every cap.
    for qty in range(largest, 0, -1):
        cost = (
            ask_total * qty
            + fee(asks[0], qty)
            + fee(asks[1], qty)
        )

        if (
            cost <= max_cost
            and cost < D(qty)
        ):
            return Opportunity(
                city=city,
                event=event,
                legs=legs,
                asks=asks,
                mids=mids,
                qty=qty,
                allin=cost,
            )

    return None


class OrderUnknown(RuntimeError):
    pass


class OrderRejected(RuntimeError):
    pass


class Kalshi:
    def __init__(self):
        self.base = os.getenv(
            "KALSHI_BASE_URL",
            BASE,
        ).rstrip("/")

        url = urlparse(self.base)

        if (
            url.scheme != "https"
            or not url.path.endswith("/trade-api/v2")
        ):
            raise RuntimeError(
                "KALSHI_BASE_URL must be an HTTPS V2 endpoint"
            )

        self.key_id = os.environ["KALSHI_API_KEY_ID"]

        self.key = serialization.load_pem_private_key(
            Path(
                os.environ["KALSHI_PRIVATE_KEY_PATH"]
            ).expanduser().read_bytes(),
            password=None,
        )

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(4.0),
        )

        self.cached_balance = None

    def headers(self, method, path):
        timestamp = str(int(time.time() * 1000))

        fullpath = (
            path
            if path.startswith("/trade-api/")
            else urlparse(self.base + path).path
        )

        data = (
            timestamp + method.upper() + fullpath
        ).encode()

        if isinstance(
            self.key,
            ed25519.Ed25519PrivateKey,
        ):
            signature = self.key.sign(data)
        else:
            signature = self.key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": (
                base64.b64encode(signature).decode()
            ),
            "Content-Type": "application/json",
        }

    async def get(self, path):
        response = await self.client.get(
            self.base + path,
            headers=self.headers("GET", path),
        )

        response.raise_for_status()
        return response.json()

    async def balance(self):
        data = await self.get("/portfolio/balance")

        if "balance_dollars" in data:
            cash = usd(data["balance_dollars"])
        else:
            cash = usd(data["balance"]) / 100

        self.cached_balance = cash
        return cash

    async def fok(
        self,
        ticker,
        side,
        price,
        qty,
        client_id,
        *,
        reduce_only=False,
    ):
        if (
            side not in ("bid", "ask")
            or not D(0) < price < ONE
        ):
            raise ValueError(
                "Invalid order side or price"
            )

        if not isinstance(qty, int) or qty < 1:
            raise ValueError(
                "Positive whole-contract count required"
            )

        path = "/portfolio/events/orders"

        body = {
            "ticker": ticker,
            "side": side,
            "count": f"{qty}.00",
            "price": f"{price:.4f}",
            "time_in_force": "fill_or_kill",
            "client_order_id": client_id,
            "self_trade_prevention_type": "taker_at_cross",
            "cancel_order_on_pause": True,
        }

        if reduce_only:
            body["reduce_only"] = True

        log(
            f"POST {side.upper()} {qty} {ticker} "
            f"limit={price:.4f} "
            f"client_id={client_id}"
        )

        started = time.monotonic()

        try:
            response = await self.client.post(
                self.base + path,
                json=body,
                headers=self.headers("POST", path),
            )

        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            raise OrderUnknown(
                f"{ticker}: transport error; "
                f"submission uncertain: {exc}"
            ) from exc

        if (
            response.status_code in (408, 409, 429)
            or response.status_code >= 500
        ):
            raise OrderUnknown(
                f"{ticker}: HTTP {response.status_code}; "
                "submission uncertain"
            )

        if response.status_code != 201:
            raise OrderRejected(
                f"{ticker}: HTTP {response.status_code} "
                f"{response.text[:350]}"
            )

        try:
            data = response.json()
            order = data.get("order", data)

            order_id = order["order_id"]
            filled = usd(order["fill_count"])
            remaining = usd(order["remaining_count"])

        except (
            TypeError,
            KeyError,
            ValueError,
        ) as exc:
            raise OrderUnknown(
                f"{ticker}: accepted response incomplete; "
                "inspect fills"
            ) from exc

        latency = (
            time.monotonic() - started
        ) * 1000

        log(
            f"ACK {ticker} id={order_id} "
            f"filled={filled} remaining={remaining} "
            f"latency={latency:.0f}ms"
        )

        if (
            filled == D(qty)
            and remaining == 0
        ):
            return True, order_id

        if filled == 0 and remaining == 0:
            return False, order_id

        # A nonterminal FOK acknowledgement is uncertain.
        raise OrderUnknown(
            f"{ticker}: ambiguous FOK acknowledgement "
            f"id={order_id} filled={filled} "
            f"remaining={remaining}"
        )

    async def ioc_sell(self, ticker, price, qty, client_id):
        """Submit once; the response is NOT proof of a fill."""
        qty = D(str(qty))

        if (
            not isinstance(ticker, str)
            or not ticker
            or not D(0) < D(str(price)) < ONE
            or not qty.is_finite()
            or qty < D("0.01")
            or qty != qty.quantize(D("0.01"))
        ):
            raise ValueError("Invalid reduce-only IOC sell")

        path = "/portfolio/events/orders"

        body = {
            "ticker": ticker,
            "side": "ask",
            "count": f"{qty:.2f}",
            "price": f"{D(str(price)):.4f}",
            "time_in_force": "immediate_or_cancel",
            "client_order_id": client_id,
            "self_trade_prevention_type": "taker_at_cross",
            "cancel_order_on_pause": True,
            "reduce_only": True,
        }

        log(
            f"POST REDUCE-ONLY IOC SELL {ticker} "
            f"qty={qty} limit={price} client_id={client_id}"
        )

        try:
            response = await self.client.post(
                self.base + path,
                json=body,
                headers=self.headers("POST", path),
            )
        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            raise OrderUnknown(
                f"IOC sell uncertain: {exc}"
            ) from exc

        if (
            response.status_code in (408, 409, 429)
            or response.status_code >= 500
        ):
            raise OrderUnknown(
                f"IOC sell HTTP {response.status_code} uncertain"
            )

        if response.status_code != 201:
            raise OrderRejected(
                f"IOC sell HTTP {response.status_code}: "
                f"{response.text[:350]}"
            )

        try:
            data = response.json()
            order = data.get("order", data)
            order_id = order["order_id"]

            if not isinstance(order_id, str) or not order_id:
                raise ValueError("Missing order ID")

        except (
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise OrderUnknown(
                "IOC sell ACK incomplete; reconcile by client ID"
            ) from exc

        # Order fills must be independently verified.
        return True, order_id

    async def gtc_sell(self, ticker, price, qty, client_id):
        """Submit one persistent YES limit sell, without retries."""
        price = D(str(price))
        if not D(0) < price < ONE or not isinstance(qty, int) or qty < 1:
            raise ValueError("Invalid GTC sell price or quantity")

        path = "/portfolio/events/orders"
        body = {
            "ticker": ticker,
            "side": "ask",
            "count": f"{qty}.00",
            "price": f"{price:.4f}",
            "time_in_force": "good_till_canceled",
            "client_order_id": client_id,
            "self_trade_prevention_type": "maker",
            "cancel_order_on_pause": False,
        }

        log(
            f"POST GTC ASK {qty} {ticker} "
            f"limit={price:.4f} client_id={client_id}"
        )

        try:
            response = await self.client.post(
                self.base + path,
                json=body,
                headers=self.headers("POST", path),
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise OrderUnknown(
                f"GTC unwind submission uncertain: {exc}"
            ) from exc

        if response.status_code in (408, 409, 429) or response.status_code >= 500:
            raise OrderUnknown(
                f"GTC unwind HTTP {response.status_code}: "
                "submission uncertain"
            )

        if response.status_code != 201:
            raise OrderRejected(
                f"GTC unwind HTTP {response.status_code}: "
                f"{response.text[:350]}"
            )

        try:
            data = response.json()
            order = data.get("order", data)
            order_id = order["order_id"]
            filled = usd(order["fill_count"])
            remaining = usd(order["remaining_count"])
            if not order_id or filled < 0 or remaining < 0:
                raise ValueError("Invalid GTC acknowledgement")
            if filled + remaining > D(qty):
                raise ValueError("GTC acknowledgement exceeds quantity")
        except (KeyError, TypeError, ValueError) as exc:
            raise OrderUnknown(
                "GTC accepted but acknowledgement incomplete; "
                "reconcile by client_order_id"
            ) from exc

        log(
            f"GTC ACK {ticker} id={order_id} "
            f"filled={filled} remaining={remaining}"
        )
        return order_id, filled, remaining

    async def close(self):
        await self.client.aclose()


class Journal:
    def __init__(self):
        self.data = {
            "single_attempt_claimed": False,
            "attempts": [],
        }

        if STATE_FILE.exists():
            data = json.loads(
                STATE_FILE.read_text()
            )

            if (
                not isinstance(data, dict)
                or not isinstance(
                    data.get("attempts"),
                    list,
                )
            ):
                raise RuntimeError(
                    "Invalid live state; refusing to trade"
                )

            self.data = data


    def write(self):
        temporary = STATE_FILE.with_suffix(".tmp")

        with temporary.open("w") as handle:
            json.dump(
                self.data,
                handle,
                indent=2,
            )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary, STATE_FILE)
        directory_fd = os.open(
            str(STATE_FILE.parent),
            os.O_DIRECTORY,
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


    def reserve(self, op, *, one_trade, args):
        reason = self.entry_block_reason(op, args)

        if reason is not None:
            raise RuntimeError(
                f"Entry blocked: {reason}"
            )

        rec = {
            "id": str(uuid.uuid4()),
            "key": op.key,
            "timestamp": datetime.now(ET).isoformat(),
            "mode": (
                "one-trade" if one_trade else "normal"
            ),
            "legs": list(op.legs),
            "qty": op.qty,
            "asks": [str(p) for p in op.asks],
            "status": "RESERVED",
        }

        self.data["attempts"].append(rec)

        if one_trade:
            self.data["single_attempt_claimed"] = True

        # Save BEFORE any request reaches Kalshi.
        self.write()
        return rec

    def update(self, record, status, **extras):
        record.update(
            status=status,
            **extras,
        )
        self.write()


    def city_attempts_today(self, city):
        today = datetime.now(ET).date().isoformat()
        prefix = city + ":"
        active = []

        for attempt in self.data["attempts"]:
            stamp = attempt.get("timestamp", "")
            key = attempt.get("key", "")

            if stamp[:10] != today or not key.startswith(prefix):
                continue

            exit_record = self.data.get("loss_exits", {}).get(key)

            if exit_record is not None and exit_record.get("state") == "DONE":
                if exit_record.get("closure_verified") is not True:
                    raise RuntimeError("Unverified city reset: " + key)

                try:
                    placed = datetime.fromisoformat(stamp)
                    closed = datetime.fromisoformat(exit_record["closed_at"])
                    if placed.tzinfo is None or closed.tzinfo is None:
                        raise ValueError("Timezone missing")
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("Invalid reset timestamp: " + key) from exc

                if placed <= closed:
                    continue

            active.append(attempt)

        return active

    def remaining_city_event_pairs(
        self, city, event, limit
    ):
        key = f"{city}:{event}"

        held = sum(
            int(a.get("qty", 0))
            for a in self.data["attempts"]
            if a.get("key") == key
            and a.get("status") == "COMPLETE"
            and a.get("closed_exposure_verified") is not True
        )

        return max(0, limit - held)

    def completed_event_pairs(self, city, event):
        """Completed YES pairs, independent of ranking order."""
        key = f"{city}:{event}"
        pairs = set()

        for record in self.data["attempts"]:
            if (
                record.get("key") != key
                or record.get("status") != "COMPLETE"
                or record.get("closed_exposure_verified") is True
            ):
                continue

            legs = record.get("legs")
            if (
                not isinstance(legs, (list, tuple))
                or len(legs) != 2
                or not all(isinstance(x, str) and x for x in legs)
                or legs[0] == legs[1]
            ):
                raise RuntimeError(
                    f"Invalid completed basket history: {key}; "
                    "trading blocked"
                )

            pairs.add(tuple(sorted(legs)))

        return pairs

    def entry_block_reason(self, op, args):
        if any(
            a.get("key") == op.key
            and a.get("status") not in SAFE_STATUSES
            for a in self.data["attempts"]
        ):
            return "unfinished entry/unwind for this event"
        if op.key in outstanding_exits(self):
            return 'protective exit unfinished for this event'
        stopped = self.data.get("loss_exits", {}).get(op.key)
        if stopped is not None and (
            stopped.get("state") != "DONE"
            or stopped.get("closure_verified") is not True
        ):
            return "loss exit not fully verified; re-entry blocked"
        completed = self.completed_event_pairs(op.city, op.event)

        if len(completed) > 1:
            return (
                "multiple completed pairs in city/event; "
                "manual review required"
            )

        pair = tuple(sorted(op.legs))

        if completed and pair not in completed:
            return (
                f"pair locked to {next(iter(completed))}; "
                f"blocked {pair}"
            )

        if (
            args.one_trade
            and self.data.get("single_attempt_claimed")
        ):
            return "one-trade already used"

        if (
            self.attempts_today()
            >= args.max_daily_attempts
        ):
            return "global daily attempt cap"

        city_attempts = self.city_attempts_today(
            op.city
        )

        if len(city_attempts) >= args.max_city_attempts:
            return "city daily attempt cap"

        remaining = self.remaining_city_event_pairs(
            op.city,
            op.event,
            args.max_city_event_pairs,
        )

        if op.qty > remaining:
            return "city/event cumulative pair cap"

        if city_attempts:
            recent = max(
                datetime.fromisoformat(a["timestamp"])
                for a in city_attempts
            )

            age = (
                datetime.now(ET) - recent
            ).total_seconds()

            if (
                age
                < args.city_cooldown_minutes * 60
            ):
                return "city entry cooldown"

        return None

    def attempts_today(self):
        today = datetime.now(ET).date().isoformat()

        return sum(
            a.get("timestamp", "")[:10] == today
            for a in self.data["attempts"]
        )


from src.backtest.weather_order_reconciliation import (
    ReconciliationError,
    reconcile_order,
    positions_for_event,
    check_positions,
    retry_allowed,
)


async def submit_verified(
    api, journal, record, prefix,
    ticker, side, price, qty, *,
    reduce_only=False,
):
    """Persist identity, POST once, verify against orders/fills."""
    client_id = str(uuid.uuid4())

    journal.update(
        record,
        prefix.upper() + "_POST_PREPARED",
        **{prefix + "_client_id": client_id},
    )

    if reduce_only and side != "ask":
        raise ValueError(
            "reduce_only only permitted for YES sells"
        )

    order_id = None

    try:
        if reduce_only:
            _, order_id = await api.ioc_sell(
                ticker, price, qty, client_id
            )
        else:
            _, order_id = await api.fok(
                ticker, side, price, qty, client_id,
            )

    except OrderRejected as exc:
        journal.update(
            record,
            prefix.upper() + "_REJECTED",
            **{
                prefix + "_rejected": True,
                prefix + "_rejection": str(exc),
            },
        )
        return None

    except OrderUnknown as exc:
        log(
            f"{prefix}: submission uncertain; "
            f"querying Kalshi: {exc}"
        )

    # No second POST until the first outcome is proven.
    verified = await reconcile_order(
        api,
        ticker=ticker,
        client_id=client_id,
        requested=qty,
        order_id=order_id,
        side=side,
        allow_partial=reduce_only,
    )

    journal.update(
        record,
        prefix.upper() + "_VERIFIED",
        **{
            prefix + "_order_id": verified.order_id,
            prefix + "_filled": (
                str(verified.filled)
                if reduce_only
                and verified.filled != int(verified.filled)
                else int(verified.filled)
            ),
        },
    )

    log(
        f"VERIFIED {prefix} {ticker}: "
        f"{verified.filled}/{qty}, "
        f"status={verified.status}"
    )

    return verified


async def unwind_verified(
    api, books, last_seen,
    ticker, qty, journal, record,
):
    """GTC limit sell at original buy price, either leg."""
    if ticker not in record["legs"]:
        raise ReconciliationError(
            "Unwind ticker not part of basket"
        )

    index = record["legs"].index(ticker)
    chosen = "first" if index == 0 else "second"
    other = "second" if index == 0 else "first"

    if D(str(record.get(chosen + "_filled") or 0)) != D(qty):
        raise ReconciliationError(
            "Unwind leg not verified fully filled"
        )

    if D(str(record.get(other + "_filled") or 0)) != 0:
        raise ReconciliationError(
            "Other leg also has exposure"
        )

    if D(str(record.get("second_retry_filled") or 0)) != 0:
        raise ReconciliationError(
            "Second retry exposure exists"
        )

    if record.get("unwind_client_id"):
        raise ReconciliationError(
            "Unwind already submitted or uncertain"
        )

    buy_price = usd(record["asks"][index])

    if not D(0) < buy_price < ONE:
        raise ReconciliationError(
            "Invalid original buy limit"
        )

    event = record["key"].split(":", 1)[1]

    # GTC cannot use reduce_only.
    # Check the position before placing a sell.
    await check_positions(
        api,
        event,
        record["pre_positions"],
        {ticker: qty},
    )

    cid = str(uuid.uuid4())

    journal.update(
        record,
        "UNWIND_POST_PREPARED",
        unwind_client_id=cid,
        unwind_ticker=ticker,
        unwind_limit_price=str(buy_price),
        unwind_time_in_force="good_till_canceled",
    )

    try:
        oid, filled, remaining = await api.gtc_sell(
            ticker, buy_price, qty, cid,
        )

    except OrderRejected as exc:
        journal.update(
            record,
            "EXPOSED_UNWIND_REJECTED",
            error=str(exc),
        )
        log(f"CRITICAL: GTC unwind rejected: {exc}")
        return False

    except Exception as exc:
        journal.update(
            record,
            "EXPOSED_UNWIND_UNKNOWN",
            error=repr(exc),
        )
        log(
            "CRITICAL: GTC submission uncertain. "
            "Do not resubmit blindly."
        )
        return False

    journal.update(
        record,
        "UNWIND_RESTING",
        unwind_order_id=oid,
        unwind_filled=str(filled),
        unwind_remaining=str(remaining),
    )

    if D(str(filled)) != D(qty) or D(str(remaining)) != 0:
        log(
            f"GTC UNWIND RESTING {ticker}: "
            f"{filled}/{qty} sold, remaining={remaining}, "
            f"order={oid}. New trading blocked."
        )
        return False

    verified = await reconcile_order(
        api,
        ticker=ticker,
        client_id=cid,
        requested=qty,
        order_id=oid,
        side="ask",
    )

    if not verified.full:
        raise ReconciliationError(
            "GTC sale not independently verified"
        )

    await check_positions(
        api, event, record["pre_positions"], {},
    )

    journal.update(
        record,
        "UNWOUND",
        unwind_filled=str(qty),
        unwind_remaining="0",
    )

    log(
        f"UNWOUND VERIFIED: {qty} YES on {ticker} "
        f"at minimum {buy_price}"
    )
    return True



def known_gtc_unwind(record):
    return (
        record.get("status") in {
            "UNWIND_RESTING", "RECONCILING",
        }
        and record.get("unwind_time_in_force")
        == "good_till_canceled"
        and bool(record.get("unwind_client_id"))
        and bool(record.get("unwind_order_id"))
    )


async def refresh_gtc_unwind(api, journal, record):
    """Read-only exchange reconciliation; never POST a sell."""
    from src.backtest.weather_unwind_watch import (
        exchange_snapshot, classify, exact_order,
        verify_order,
    )

    probe = dict(record)
    probe["status"] = "UNWIND_RESTING"

    orders, fills, positions = await exchange_snapshot(
        api, probe
    )
    verdict, detail = classify(
        probe, orders, fills, positions
    )

    key = record["key"]

    if verdict == "READY":
        journal.update(
            record,
            "UNWOUND",
            unwind_filled=str(record["qty"]),
            unwind_remaining="0",
            auto_unwind_verified=True,
        )
        log(f"GTC UNWIND VERIFIED {key}; event unlocked")
        return

    if verdict != "PENDING":
        raise RuntimeError(
            f"Unrecognized GTC verdict {key}: {verdict}"
        )

    ticker = record["unwind_ticker"]
    oid = record["unwind_order_id"]
    sell = exact_order(
        orders[ticker],
        ticker,
        order_id=oid,
        client_id=record["unwind_client_id"],
    )

    if sell is None:
        raise RuntimeError(
            f"{key}: tracked GTC order not visible"
        )

    sold, remaining, status = verify_order(
        sell, fills[oid], ticker, "sell",
        D(str(record["qty"])),
    )

    if (
        status not in ("resting", "pending")
        or sold + remaining != D(str(record["qty"]))
    ):
        raise RuntimeError(
            f"{key}: unexpected GTC order state"
        )

    # Verify held exposure matches the exact tracked sale.
    for leg in record["legs"]:
        baseline = D(str(record["pre_positions"][leg]))
        expected = baseline + (
            D(str(record["qty"])) - sold
            if leg == ticker else D(0)
        )
        actual = D(str(positions[leg]))
        if actual != expected:
            raise RuntimeError(
                f"{key}: position mismatch on {leg}: "
                f"expected={expected} actual={actual}"
            )

    if (
        record.get("status") != "UNWIND_RESTING"
        or D(str(record.get("unwind_filled", 0))) != sold
        or D(str(record.get("unwind_remaining", 0)))
        != remaining
    ):
        journal.update(
            record, "UNWIND_RESTING",
            unwind_filled=str(sold),
            unwind_remaining=str(remaining),
        )

    log(
        f"GTC UNWIND PENDING {key}: "
        f"sold={sold}, remaining={remaining}; "
        "this event blocked, other cities continue"
    )


async def poll_open_gtc_unwinds(api, journal):
    for record in journal.data["attempts"]:
        if not known_gtc_unwind(record):
            continue
        try:
            await refresh_gtc_unwind(api, journal, record)
        except Exception as exc:
            log(
                f"GTC UNWIND QUARANTINED "
                f"{record.get('key')}: {exc}; "
                "other cities continue"
            )


async def risk_monitor_tick(
    api, journal, loss_manager, cities, books, last_seen
):
    await loss_manager.tick(api, cities, books, last_seen)

    now = time.monotonic()
    last = getattr(journal, "_gtc_last_check", -1e12)

    if now - last >= 30:
        journal._gtc_last_check = now
        await poll_open_gtc_unwinds(api, journal)


async def recover_startup(api, journal):
    """Reconstruct unfinished trades before starting scanners."""
    unresolved = [
        record
        for record in journal.data["attempts"]
        if record.get("status") not in SAFE_STATUSES
    ]

    if not unresolved:
        return

    log(
        "STARTUP RECONCILIATION: "
        f"{len(unresolved)} unfinished attempt(s)"
    )

    for record in unresolved:
        if known_gtc_unwind(record):
            try:
                await refresh_gtc_unwind(api, journal, record)
            except Exception as exc:
                log(
                    f"GTC STARTUP QUARANTINE "
                    f"{record.get('key')}: {exc}; "
                    "other cities remain eligible"
                )
            continue

        first, second = record["legs"]
        qty = int(record["qty"])

        prefixes = (
            "first",
            "second",
            "second_retry",
            "unwind",
        )

        if (
            record.get("status") == "RESERVED"
            and not any(
                record.get(prefix + "_client_id")
                for prefix in prefixes
            )
        ):
            journal.update(
                record, "NOT_SUBMITTED"
            )
            continue

        results = {}

        for prefix, ticker, side in (
            ("first", first, "bid"),
            ("second", second, "bid"),
            ("second_retry", second, "bid"),
            ("unwind", record.get("unwind_ticker", first), "ask"),
        ):
            cid = record.get(
                prefix + "_client_id"
            )

            if not cid:
                continue

            if record.get(
                prefix + "_rejected"
            ):
                continue

            results[prefix] = (
                await reconcile_order(
                    api,
                    ticker=ticker,
                    client_id=cid,
                    requested=qty,
                    order_id=record.get(
                        prefix + "_order_id"
                    ),
                    side=side,
                )
            )

            journal.update(
                record,
                "RECONCILING",
                **{
                    prefix + "_order_id":
                        results[prefix].order_id,
                    prefix + "_filled":
                        results[prefix].filled,
                },
            )

        a = results.get("first")
        b = results.get("second")
        retry = results.get("second_retry")
        sale = results.get("unwind")

        async def verify_restart_positions(changes):
            # Settled historical contracts may no longer
            # appear as positions. Fills remain authoritative.
            same_day = (
                record.get("timestamp", "")[:10]
                == datetime.now(ET).date().isoformat()
            )

            if (
                record.get("pre_positions")
                and same_day
            ):
                await check_positions(
                    api,
                    record["key"].split(":", 1)[1],
                    record["pre_positions"],
                    changes,
                )

        if (
            record.get("concurrent_buys")
            and not any((sale, retry))
            and (
                record.get("first_rejected")
                or (a and a.unfilled)
            )
            and (
                record.get("second_rejected")
                or (b and b.unfilled)
            )
        ):
            await verify_restart_positions({})
            journal.update(
                record,
                "FIRST_UNFILLED",
                concurrent_both_zero=True,
            )

        elif (
            a is None
            and record.get("first_rejected")
            and not any((b, retry, sale))
        ):
            journal.update(
                record, "NOT_SUBMITTED"
            )

        elif (
            a
            and a.unfilled
            and not any((b, retry, sale))
        ):
            await verify_restart_positions({})
            journal.update(
                record, "FIRST_UNFILLED"
            )

        elif (
            a
            and a.full
            and sum(
                obj.filled
                for obj in (b, retry)
                if obj is not None
            ) == qty
            and (sale is None or sale.unfilled)
        ):
            await verify_restart_positions({
                first: qty,
                second: qty,
            })
            journal.update(
                record, "COMPLETE"
            )

        elif (
            sale and sale.full
            and (
                (
                    record.get("unwind_ticker", first) == first
                    and a and a.full
                    and not any(
                        obj and obj.filled
                        for obj in (b, retry)
                    )
                )
                or (
                    record.get("unwind_ticker") == second
                    and b and b.full
                    and not any(
                        obj and obj.filled
                        for obj in (a, retry)
                    )
                )
            )
        ):
            await verify_restart_positions({})
            journal.update(
                record, "UNWOUND"
            )

        else:
            raise ReconciliationError(
                "Unresolved exposure for "
                f"{record['id']} "
                f"({record['key']}). "
                "Fresh-book/manual recovery required; "
                "new entries prohibited."
            )

    log(
        "STARTUP RECONCILIATION: "
        "safe historical states resolved"
    )


async def execute(
    api, journal, books, last_seen,
    op, tickers, args,
):
    """Concurrent FOK buys; reconcile both afterward."""
    latest = evaluate(
        op.city,
        op.event,
        books,
        tickers,
        one_trade=args.one_trade,
        max_pairs=min(
            args.max_pairs,
            journal.remaining_city_event_pairs(
                op.city,
                op.event,
                args.max_city_event_pairs,
            ),
        ),
        max_cost=args.max_cost,
    )

    if (
        latest is None
        or latest.legs != op.legs
        or latest.qty < op.qty
        or any(
            new > old
            for new, old in zip(latest.asks, op.asks)
        )
        or any(
            time.monotonic() - last_seen.get(t, 0) > 3
            for t in op.legs
        )
    ):
        log(f"{op.city}: preflight failed; no orders sent")
        return False, False

    # Check the persisted pair lock before execution.
    block = journal.entry_block_reason(op, args)

    if block is not None:
        log(
            f"{op.city}: entry blocked: {block}; "
            "no orders submitted"
        )
        return False, False

    cash = getattr(api, "cached_balance", None)

    if cash is None:
        cash = await api.balance()

    if cash < op.allin:
        log(f"{op.city}: insufficient cash; no orders sent")
        return False, False

    # Reject third-outcome exposure already held on Kalshi.
    all_positions = await positions_for_event(
        api, op.event, tickers,
    )

    others = {
        ticker: qty
        for ticker, qty in all_positions.items()
        if ticker not in op.legs and qty != 0
    }

    if others:
        log(
            f"{op.city}: existing third-outcome exposure "
            f"{others}; no orders submitted"
        )
        return False, False

    baseline = {
        ticker: all_positions[ticker]
        for ticker in op.legs
    }

    record = journal.reserve(
        op,
        one_trade=args.one_trade,
        args=args,
    )

    first, second = op.legs

    journal.update(
        record,
        "RESERVED",
        pre_positions={
            ticker: str(value)
            for ticker, value in baseline.items()
        },
    )

    # Persist BOTH identities before either POST.
    identities = {
        "first": str(uuid.uuid4()),
        "second": str(uuid.uuid4()),
    }

    journal.update(
        record,
        "BOTH_POST_PREPARED",
        concurrent_buys=True,
        first_client_id=identities["first"],
        second_client_id=identities["second"],
    )

    log(
        f"LIVE CONCURRENT ATTEMPT {op.city}: "
        f"qty={op.qty}, "
        f"limits=({op.asks[0]:.4f},{op.asks[1]:.4f}), "
        f"estimated=${op.allin:.2f}"
    )

    try:
        specs = (
            ("first", first, op.asks[0]),
            ("second", second, op.asks[1]),
        )

        # Both requests dispatched before either verification.
        post_results = await asyncio.gather(
            *(
                api.fok(
                    ticker,
                    "bid",
                    price,
                    op.qty,
                    identities[prefix],
                )
                for prefix, ticker, price in specs
            ),
            return_exceptions=True,
        )

        known_ids = {}
        rejected = set()

        for (prefix, ticker, price), response in zip(
            specs, post_results
        ):
            if isinstance(response, OrderRejected):
                rejected.add(prefix)

                journal.update(
                    record,
                    prefix.upper() + "_REJECTED",
                    **{
                        prefix + "_rejected": True,
                        prefix + "_rejection": str(response),
                        prefix + "_filled": 0,
                    },
                )

                log(
                    f"{prefix} FOK definitively rejected: "
                    f"{response}"
                )

            elif isinstance(response, BaseException):
                # Never interpret a timeout/409 as unfilled.
                log(
                    f"{prefix} FOK ACK uncertain: {response}"
                )

            else:
                _, oid = response
                known_ids[prefix] = oid

                journal.update(
                    record,
                    prefix.upper() + "_ACK",
                    **{
                        prefix + "_order_id": oid,
                    },
                )

        async def verify_one(prefix, ticker, price):
            if prefix in rejected:
                return None

            return await reconcile_order(
                api,
                ticker=ticker,
                client_id=identities[prefix],
                requested=op.qty,
                order_id=known_ids.get(prefix),
                side="bid",
            )

        # Verification occurs AFTER both POST attempts.
        verifications = await asyncio.gather(
            *(verify_one(*spec) for spec in specs),
            return_exceptions=True,
        )

        for (prefix, ticker, _), result in zip(
            specs, verifications
        ):
            if isinstance(result, BaseException):
                raise ReconciliationError(
                    f"{prefix} {ticker} not proven: {result}"
                ) from result

            if result is not None:
                journal.update(
                    record,
                    prefix.upper() + "_VERIFIED",
                    **{
                        prefix + "_order_id": result.order_id,
                        prefix + "_filled": result.filled,
                    },
                )

                log(
                    f"VERIFIED {prefix} {ticker}: "
                    f"{result.filled}/{op.qty}, "
                    f"status={result.status}"
                )

        a, b = verifications

        first_full = bool(
            a is not None and a.full
        )
        second_full = bool(
            b is not None and b.full
        )

        first_zero = (
            (a is None and "first" in rejected)
            or bool(a and a.unfilled)
        )
        second_zero = (
            (b is None and "second" in rejected)
            or bool(b and b.unfilled)
        )

        if first_full and second_full:
            await check_positions(
                api,
                op.event,
                baseline,
                {
                    first: op.qty,
                    second: op.qty,
                },
            )

            journal.update(record, "COMPLETE")
            log(
                f"BASKET COMPLETE VERIFIED: "
                f"{op.city} {op.qty} pairs"
            )
            return True, False

        if first_zero and second_zero:
            await check_positions(
                api, op.event, baseline, {},
            )

            journal.update(
                record,
                "FIRST_UNFILLED",
                concurrent_both_zero=True,
            )

            log(
                f"BOTH FOK LEGS UNFILLED/REJECTED: "
                f"{op.city}"
            )
            return True, False

        if (
            (first_full and second_zero)
            or (second_full and first_zero)
        ):
            exposed = first if first_full else second

            journal.update(
                record,
                "ONE_LEG_EXPOSED",
                exposed_ticker=exposed,
            )

            log(
                f"ONE LEG FILLED: {exposed}; "
                f"other leg proven zero. "
                f"Submitting GTC unwind at buy limit."
            )

            safe = await unwind_verified(
                api,
                books,
                last_seen,
                exposed,
                op.qty,
                journal,
                record,
            )

            if (
                not safe
                and known_gtc_unwind(record)
            ):
                log(
                    f"GTC UNWIND TRACKING {op.key}; "
                    "other cities continue trading"
                )
                return True, False

            return True, not safe

        raise ReconciliationError(
            "Unexpected partial or unknown FOK execution"
        )

    except Exception as exc:
        journal.update(
            record,
            "UNRESOLVED_NEEDS_RECONCILIATION",
            error=repr(exc),
        )

        log(
            f"CRITICAL: {exc}; "
            "new orders blocked until reconciliation"
        )

        return True, True




async def fetch_event(http, code):
    _, series, timezone_name = CITY[code]

    local = datetime.now(
        ZoneInfo(timezone_name)
    )

    event = (
        f"{series}-{local:%y%b%d}"
    ).upper()

    try:
        response = await http.get(
            BASE + f"/events/{event}",
            params={
                "with_nested_markets": "true",
            },
            timeout=8,
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        data = response.json()
        info = data.get("event", {})

        if info.get("mutually_exclusive") is False:
            log(
                f"{code}: nonexclusive event skipped"
            )
            return None

        markets = (
            data.get("markets")
            or info.get("markets")
            or []
        )

        tickers = tuple(sorted(
            market["ticker"]
            for market in markets
            if market.get("status") in (
                "active",
                "open",
            )
            and market.get("ticker")
        ))

        if len(tickers) < 3:
            return None

        return (
            event,
            tickers,
            local.date(),
        )

    except Exception as exc:
        log(
            f"{code}: event discovery failed: {exc}"
        )
        return None


def time_gate(codes):
    active = []
    waiting = []

    for code in codes:
        local = datetime.now(
            ZoneInfo(CITY[code][2])
        )

        gate = local.replace(
            hour=14,
            minute=0,
            second=0,
            microsecond=0,
        )

        if local >= gate:
            active.append(code)
        else:
            seconds = max(
                0,
                gate.timestamp() - local.timestamp(),
            )
            waiting.append((seconds, code))

    return active, sorted(waiting)


async def ws_session(
    args,
    api,
    journal,
    live,
):
    active, waiting = time_gate(args.cities)
    entry_active = list(active)
    # Loss monitoring is built into the normal bot.
    active = sorted(
        set(active) | held_cities(journal, args.cities)
    )

    # Gate cities BEFORE discovery and subscriptions.
    if not active:
        seconds, next_code = waiting[0]

        log(
            f"WAITING: no city past 2 PM local. "
            f"Next={next_code} in "
            f"{seconds / 60:.1f} minutes"
        )

        await asyncio.sleep(
            max(0.1, seconds)
        )

        return False

    async with httpx.AsyncClient() as http:
        found = await asyncio.gather(*(
            fetch_event(http, city)
            for city in active
        ))

    cities = {
        city: result
        for city, result in zip(active, found)
        if result is not None
    }

    if not cities:
        log(
            "No qualifying open city events. "
            "Retrying discovery in 30 seconds."
        )

        await asyncio.sleep(30)
        return False

    current_active, waiting = time_gate(args.cities)

    if current_active != entry_active:
        return False

    deadline = (
        time.monotonic() + waiting[0][0]
        if waiting
        else float("inf")
    )

    tickers = sorted({
        ticker
        for _, markets, _ in cities.values()
        for ticker in markets
    })

    books = {
        ticker: Book()
        for ticker in tickers
    }

    owners = {
        ticker: code
        for code, (_, markets, _) in cities.items()
        for ticker in markets
    }

    last_seen = {}
    sequence = {}
    signal_keys = {}
    execution = None
    exit_task = None
    loss_manager = LossMonitor(
        journal,
        enabled=live,
        quotes=quotes,
        positions=positions_for_event,
        submit=submit_verified,
        log=log,
        reconcile=reconcile_order,
    )

    updates = 0

    started = time.monotonic()
    maintenance = started
    last_report = started

    log(
        f"ACTIVE={list(cities)} "
        f"WAITING={[c for _, c in waiting]} "
        f"books={len(books)} "
        f"mode={'LIVE' if live else 'SHADOW'} "
        f"{'ONE-TRADE' if args.one_trade else 'CONTINUOUS'}"
    )

    headers = api.headers(
        "GET",
        urlparse(WS_URL).path,
    )

    options = {
        "ping_interval": 20,
        "ping_timeout": 20,
        "max_queue": 4096,
    }

    header_parameter = (
        "additional_headers"
        if (
            "additional_headers"
            in inspect.signature(
                websockets.connect
            ).parameters
        )
        else "extra_headers"
    )

    options[header_parameter] = headers

    async with websockets.connect(
        WS_URL,
        **options,
    ) as ws:

        for index in range(0, len(tickers), 40):
            await ws.send(json.dumps({
                "id": index // 40 + 1,
                "cmd": "subscribe",
                "params": {
                    "channels": [
                        "orderbook_delta",
                    ],
                    "market_tickers": (
                        tickers[index:index + 40]
                    ),
                },
            }))

        try:
            while True:
                now = time.monotonic()

                if (
                    execution is not None
                    and execution.done()
                ):
                    attempted, danger = await execution
                    execution = None

                    if (
                        danger
                        or (args.one_trade and attempted)
                    ):
                        return True

                    if attempted and live:
                        await api.balance()
                if exit_task is not None and exit_task.done():
                    await exit_task
                    exit_task = None

                # Scheduler operates outside the
                # order-book evaluation hot path.
                if (
                    execution is None
                    and now - maintenance >= 1
                ):
                    maintenance = now
                    if (
                        loss_manager is not None
                        and exit_task is None
                    ):
                        exit_task = asyncio.create_task(
                            risk_monitor_tick(
                                api, journal, loss_manager,
                                cities, books, last_seen,
                            )
                        )

                    if (
                        now >= deadline
                        or now - started >= 1200
                    ):
                        log(
                            "Refreshing subscriptions "
                            "for activation or 20m cycle"
                        )
                        return False

                    date_changed = any(
                        datetime.now(
                            ZoneInfo(CITY[code][2])
                        ).date() != event_date
                        for code, (
                            _, _, event_date
                        ) in cities.items()
                    )

                    if date_changed:
                        log(
                            "Local date rollover: refreshing"
                        )
                        return False

                try:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=1,
                    )

                except asyncio.TimeoutError:
                    continue

                message = json.loads(raw)

                if message.get("type") == "error":
                    raise RuntimeError(
                        f"WebSocket error: {message}"
                    )

                message_type = message.get("type")

                if message_type not in (
                    "orderbook_snapshot",
                    "orderbook_delta",
                ):
                    continue

                sid = message.get("sid")
                seq = message.get("seq")

                if (
                    sid is None
                    or not isinstance(seq, int)
                ):
                    raise RuntimeError(
                        "Missing WS orderbook seq/sid"
                    )

                if (
                    sid in sequence
                    and seq != sequence[sid] + 1
                ):
                    raise RuntimeError(
                        f"WS sequence gap sid={sid}: "
                        f"expected={sequence[sid] + 1}, "
                        f"got={seq}"
                    )

                sequence[sid] = seq

                payload = message["msg"]
                ticker = payload.get("market_ticker")

                if ticker not in books:
                    continue

                if message_type == "orderbook_snapshot":
                    books[ticker].snapshot(payload)
                else:
                    books[ticker].delta(payload)

                last_seen[ticker] = time.monotonic()
                updates += 1

                if (
                    time.monotonic()
                    - last_report >= 15
                ):
                    log(
                        f"HEARTBEAT updates={updates} "
                        f"active={list(cities)} "
                        f"attempts_today="
                        f"{journal.attempts_today()} "
                        f"candidate_count="
                        f"{len(signal_keys)}"
                    )

                    last_report = time.monotonic()

                # Keep order books updated while
                # an HTTP execution task runs.
                if execution is not None:
                    continue

                code = owners[ticker]
                if code not in entry_active:
                    continue  # Exit monitoring only before 2 PM.

                event, market_tickers, _ = (
                    cities[code]
                )
                event_key = f'{code}:{event}'
                if event_key in outstanding_exits(journal):
                    continue
                if (
                    event_key in loss_manager.below_since
                    and event_key not in journal.data.get('loss_exits', {})
                ):
                    continue  # Stop confirmation pending.

                if len(journal.city_attempts_today(code)) >= args.max_city_attempts:
                    continue
                if journal.remaining_city_event_pairs(
                    code, event, args.max_city_event_pairs
                ) < 1:
                    continue

                if (
                    journal.attempts_today()
                    >= args.max_daily_attempts
                ):
                    continue

                opportunity = evaluate(
                    code,
                    event,
                    books,
                    market_tickers,
                    one_trade=args.one_trade,
                    max_pairs=min(args.max_pairs, journal.remaining_city_event_pairs(code, event, args.max_city_event_pairs)),
                    max_cost=args.max_cost,
                )

                block = (
                    journal.entry_block_reason(opportunity, args)
                    if opportunity is not None and live
                    else None
                )

                if block is not None and (
                    block.startswith("pair locked to")
                    or block.startswith("multiple completed pairs")
                ):
                    marker = (
                        "PAIR_LOCK_BLOCKED",
                        opportunity.legs,
                        block,
                    )

                    if signal_keys.get(code) != marker:
                        log(
                            f"PAIR LOCK BLOCKED {code}: "
                            f"{block}"
                        )

                    signal_keys[code] = marker
                    continue

                if opportunity and (not live or block is None):
                    identity = (
                        opportunity.legs,
                        opportunity.qty,
                        opportunity.asks,
                    )

                    if signal_keys.get(code) != identity or live:
                        signal_keys[code] = identity

                        log(
                            f"CANDIDATE {code} "
                            f"{opportunity.legs} "
                            f"asks={opportunity.asks} "
                            f"top2_mid="
                            f"{opportunity.mids:.2%} "
                            f"qty={opportunity.qty} "
                            f"estimated_allin="
                            f"${opportunity.allin:.2f}"
                        )

                        if live:
                            execution = asyncio.create_task(
                                execute(
                                    api,
                                    journal,
                                    books,
                                    last_seen,
                                    opportunity,
                                    market_tickers,
                                    args,
                                )
                            )

                        elif args.one_trade:
                            log(
                                "SHADOW ONE-TRADE candidate. "
                                "No order submitted; stopping."
                            )
                            return True

                else:
                    signal_keys.pop(code, None)

        finally:
            if execution is not None:
                await asyncio.shield(execution)
            if exit_task is not None:
                await asyncio.shield(exit_task)


def self_test():
    from types import SimpleNamespace
    from unittest.mock import patch

    def synthetic_book(mid, depth=20):
        low = mid - D(".01")
        high = mid + D(".01")

        return SimpleNamespace(
            ready=True,
            levels={
                "yes": {
                    low: D(depth),
                },
                "no": {
                    ONE - high: D(depth),
                },
            },
        )

    probabilities = (
        D(".45"), D(".37"), D(".10"),
        D(".04"), D(".02"), D(".02"),
    )

    books = {
        f"T{i}": synthetic_book(probability)
        for i, probability
        in enumerate(probabilities)
    }

    assert quotes(books["T0"])[1] == D(".46")

    synthetic_selection = (
        {"ticker": "T0"},
        {"ticker": "T1"},
        D(".84"),
        True,
    )

    with patch(
        __name__ + ".scan",
        return_value=synthetic_selection,
    ):
        one = evaluate(
            "NYC",
            "TEST",
            books,
            tuple(books),
            one_trade=True,
            max_pairs=50,
            max_cost=D("20"),
        )

        assert one is not None
        assert one.qty == 1
        assert one.mids == D(".82")

        normal = evaluate(
            "NYC",
            "TEST",
            books,
            tuple(books),
            one_trade=False,
            max_pairs=7,
            max_cost=D("20"),
        )

        assert normal is not None
        assert normal.qty == 7

    # Check compatibility with installed Book/scan.
    real_books = {}

    try:
        for ticker, probability in zip(
            books,
            probabilities,
        ):
            book = Book()

            book.snapshot({
                "market_ticker": ticker,
                "yes_dollars_fp": [[
                    str(probability - D(".01")),
                    "20.00",
                ]],
                "no_dollars_fp": [[
                    str(ONE - probability - D(".01")),
                    "20.00",
                ]],
            })

            real_books[ticker] = book

        integration = evaluate(
            "NYC",
            "TEST",
            real_books,
            tuple(real_books),
            one_trade=True,
            max_pairs=10,
            max_cost=D("10"),
        )

        assert integration is not None
        assert integration.qty == 1

    except Exception as exc:
        raise RuntimeError(
            "Book/scan integration test failed. "
            "Do not enable live trading."
        ) from exc

    print(
        "PASS: one-trade=1 pair; "
        "normal dynamic sizing; "
        "75% midpoint; cost cap"
    )
    print(
        "PASS: existing Book.snapshot/scan integration"
    )
    print(
        "OFFLINE ONLY: no credentials or orders required"
    )


import fcntl

_LOCK_FD = None


def claim_process_lock():
    """Hold an exclusive Linux process lock until exit."""
    global _LOCK_FD

    if _LOCK_FD is not None:
        raise RuntimeError(
            "Process lock already held in this interpreter"
        )

    fd = os.open(
        LOCK_FILE,
        os.O_CREAT | os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )

    try:
        try:
            fcntl.flock(
                fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another weather live bot is already running"
            ) from exc

        # PID is diagnostic information only.
        # The kernel manages the actual lock.
        message = (
            f"pid={os.getpid()} "
            f"started={datetime.now(ET).isoformat()}\n"
        )

        os.ftruncate(fd, 0)
        os.write(fd, message.encode())
        os.fsync(fd)

    except BaseException:
        os.close(fd)
        raise

    _LOCK_FD = fd


def release_process_lock():
    """Release the lock by closing its file descriptor."""
    global _LOCK_FD

    if _LOCK_FD is not None:
        os.close(_LOCK_FD)
        _LOCK_FD = None


async def main():
    load_dotenv(
        Path.cwd() / ".env"
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--cities",
        default=",".join(CITY),
    )

    parser.add_argument(
        "--one-trade",
        "--one-contract",
        dest="one_trade",
        action="store_true",
        help=(
            "One YES per leg, one basket attempt "
            "total, then stop"
        ),
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help=(
            "Maximum pairs per normal-mode attempt "
            "(default 10)"
        ),
    )

    parser.add_argument(
        "--max-cost",
        type=D,
        default=D("10"),
        help=(
            "Maximum estimated all-in cost "
            "per attempt in dollars"
        ),
    )

    parser.add_argument(
        "--max-daily-attempts",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )


    parser.add_argument(
        "--max-city-attempts",
        type=int,
        default=3,
        help="Maximum attempts per city/day",
    )

    parser.add_argument(
        "--city-cooldown-minutes",
        type=D,
        default=D("10"),
        help="Minutes between entries in same city",
    )

    parser.add_argument(
        "--max-city-event-pairs",
        type=int,
        default=30,
        help="Cumulative filled pairs per city/event",
    )
    args = parser.parse_args()

    if (
        args.max_city_attempts < 1
        or args.max_city_event_pairs < 1
        or not args.city_cooldown_minutes.is_finite()
        or args.city_cooldown_minutes < 0
    ):
        parser.error(
            "Invalid city attempt, cooldown, "
            "or cumulative pair limit"
        )

    args.cities = tuple(
        city.strip().upper()
        for city in args.cities.split(",")
        if city.strip()
    )

    if (
        not args.cities
        or any(
            city not in CITY
            for city in args.cities
        )
    ):
        parser.error(
            "Unknown city in --cities"
        )

    if (
        args.max_pairs < 1
        or not args.max_cost.is_finite()
        or args.max_cost <= 0
        or args.max_daily_attempts < 1
    ):
        parser.error(
            "Pair, cost and daily-attempt limits "
            "must be positive"
        )

    if args.self_test:
        self_test()
        return

    live = os.getenv(
        "KALSHI_LIVE_TRADING_ENABLED",
        "",
    ).strip().lower() in (
        "1",
        "yes",
        "true",
        "on",
    )

    claim_process_lock()

    try:
        legacy_guard = Path(
            ".live_weather_one_pair_guard.json"
        )

        if live and legacy_guard.exists():
            raise RuntimeError(
                "Previous one-pair live guard exists. "
                "Reconcile that attempt before starting "
                "this executor: "
                + str(legacy_guard)
            )

        journal = Journal()

        if (
            args.one_trade
            and journal.data.get(
                "single_attempt_claimed"
            )
        ):
            raise RuntimeError(
                "--one-trade already consumed "
                "in saved state. Inspect first."
            )

        api = Kalshi()

        try:
            if live:
                await recover_startup(api, journal)
                await recover_loss_exits(
                    api, journal, reconcile_order
                )
            cash = await api.balance()

            log(
                f"ACCOUNT cash=${cash:.2f}; "
                f"mode={'LIVE' if live else 'SHADOW'}; "
                f"one_trade={args.one_trade}; "
                f"max_pairs="
                f"{1 if args.one_trade else args.max_pairs}; "
                f"max_cost=${args.max_cost:.2f}; "
                f"max_daily_attempts="
                f"{args.max_daily_attempts}"
            )

            # Keep monitoring owned positions after entry limits.
            while True:
                try:
                    stop = await ws_session(
                        args,
                        api,
                        journal,
                        live,
                    )

                except (
                    KeyboardInterrupt,
                    asyncio.CancelledError,
                ):
                    raise

                except Exception as exc:
                    unresolved = [
                        attempt
                        for attempt
                        in journal.data["attempts"]
                        if (
                            attempt.get("status")
                            not in SAFE_STATUSES
                            and not known_gtc_unwind(attempt)
                        )
                    ]

                    if unresolved:
                        raise RuntimeError(
                            "Unresolved exchange order. "
                            "Stop and reconcile manually."
                        ) from exc

                    log(
                        f"MARKET CONNECTION ERROR: {exc}. "
                        "Reconnecting in 3 seconds."
                    )

                    await asyncio.sleep(3)
                    continue

                if stop:
                    break

                await asyncio.sleep(0.2)

        finally:
            await api.close()

    finally:
        release_process_lock()


if __name__ == "__main__":
    asyncio.run(main())
