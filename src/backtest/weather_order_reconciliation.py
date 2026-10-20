"""Read-only Kalshi verification and bounded retry rules.

This module does not submit orders.
Missing order evidence means UNKNOWN, never unfilled.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


class ReconciliationError(RuntimeError):
    pass


def qty(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ReconciliationError(
            f"Invalid contract quantity {value!r}"
        ) from exc

    if (
        not number.is_finite()
        or number < 0
        or number != number.to_integral_value()
    ):
        raise ReconciliationError(
            f"Non-whole contract quantity {value!r}"
        )
    return int(number)


def count(row, *names):
    for name in names:
        if row.get(name) is not None:
            return qty(row[name])
    raise ReconciliationError(
        f"Missing quantity field {names}"
    )


def fill_count(row):
    """Allow fractional individual fills, with exact decimal accounting."""
    raw = row.get("count_fp")
    if raw is None:
        raw = row.get("count")

    if raw is None:
        raise ReconciliationError("Fill missing contract quantity")

    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ReconciliationError(
            "Invalid fill contract quantity"
        ) from exc

    if (
        not value.is_finite()
        or value <= 0
        or value * 100 != (value * 100).to_integral_value()
    ):
        raise ReconciliationError(
            f"Invalid fixed-point fill quantity {raw!r}"
        )

    return value


async def paged(api, path, kind, params, pages=10):
    result = []
    cursor = None
    seen = set()

    for _ in range(pages):
        query = dict(params)
        if cursor:
            query["cursor"] = cursor

        response = await api.client.get(
            api.base + path,
            params=query,
            headers=api.headers("GET", path),
        )
        response.raise_for_status()
        payload = response.json()

        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get(kind), list)
        ):
            raise ReconciliationError(
                f"Malformed {kind} payload"
            )

        result.extend(payload[kind])
        cursor = payload.get("cursor")

        if not cursor:
            return result

        if cursor in seen:
            raise ReconciliationError(
                "Pagination cursor repeated"
            )
        seen.add(cursor)

    raise ReconciliationError(
        f"{kind}: pagination incomplete"
    )


@dataclass(frozen=True)
class Execution:
    ticker: str
    client_id: str
    order_id: str
    requested: int
    filled: Decimal | int
    remaining: Decimal | int
    status: str

    @property
    def full(self):
        return (
            self.status == "executed"
            and self.filled == self.requested
            and self.remaining == 0
        )

    @property
    def terminal(self):
        return (
            self.remaining == 0
            and (
                self.full
                or self.status in {"canceled", "cancelled"}
            )
        )

    @property
    def unfilled(self):
        return (
            self.status in {"canceled", "cancelled"}
            and self.filled == 0
            and self.remaining == 0
        )


def fixed_qty(value):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ReconciliationError("Invalid fixed-point quantity") from exc

    if (
        not number.is_finite()
        or number < 0
        or number != number.quantize(Decimal("0.01"))
    ):
        raise ReconciliationError("Invalid fixed-point precision")

    return number


def validate_order(
    order, ticker, client_id, requested, side,
    *, allow_partial=False,
):
    if (
        order.get("ticker") != ticker
        or order.get("client_order_id") != client_id
    ):
        raise ReconciliationError(
            "Order identity differs from saved journal"
        )

    book_side = order.get("book_side")
    action = order.get("action")

    if book_side is not None and book_side != side:
        raise ReconciliationError("Order book_side mismatch")

    if (
        action is not None
        and action != ("buy" if side == "bid" else "sell")
    ):
        raise ReconciliationError("Order action mismatch")

    if book_side is None and action is None:
        raise ReconciliationError(
            "Cannot verify order direction"
        )

    if allow_partial:
        filled = fixed_qty(next(
            (order[n] for n in ("fill_count_fp", "fill_count")
             if order.get(n) is not None), None
        ))
        remaining = fixed_qty(next(
            (order[n] for n in ("remaining_count_fp", "remaining_count")
             if order.get(n) is not None), None
        ))
        requested = fixed_qty(requested)
    else:
        filled = count(order, "fill_count_fp", "fill_count")
        remaining = count(order, "remaining_count_fp", "remaining_count")
    status = str(order.get("status", "")).lower()

    if (
        status not in {"executed", "canceled", "cancelled"}
        or remaining != 0
    ):
        raise ReconciliationError(
            f"Nonterminal order: {status}, remaining={remaining}"
        )

    if filled > requested or filled + remaining > requested:
        raise ReconciliationError(
            "Order quantity exceeds requested"
        )

    if 0 < filled < requested and not allow_partial:
        raise ReconciliationError(
            "Unexpected partial FOK fill"
        )

    if status == "executed" and filled != requested:
        raise ReconciliationError(
            "Executed order is not fully filled"
        )

    if (
        status in {"canceled", "cancelled"}
        and filled != 0
        and not allow_partial
    ):
        raise ReconciliationError(
            "Canceled order partially filled"
        )

    return filled, remaining, status


async def reconcile_once(
    api, *,
    ticker,
    client_id,
    requested,
    order_id=None,
    side="bid", allow_partial=False,
):
    if (
        not ticker or not client_id
        or (
            not allow_partial
            and (
                not isinstance(requested, int)
                or requested < 1
            )
        )
        or (
            allow_partial
            and fixed_qty(requested) < Decimal("0.01")
        )
    ):
        raise ReconciliationError(
            "Incomplete saved order identity"
        )

    if side not in {"bid", "ask"}:
        raise ReconciliationError(
            "Invalid saved book side"
        )

    if order_id:
        payload = await api.get(
            f"/portfolio/orders/{order_id}"
        )
        order = (
            payload.get("order")
            if isinstance(payload, dict)
            else None
        )
        if (
            not isinstance(order, dict)
            or order.get("order_id") != order_id
        ):
            raise ReconciliationError(
                "Get Order returned unexpected identity"
            )
    else:
        orders = await paged(
            api,
            "/portfolio/orders",
            "orders",
            {"ticker": ticker, "limit": "200"},
        )
        matches = [
            row for row in orders
            if row.get("client_order_id") == client_id
            and row.get("ticker") == ticker
        ]
        if len(matches) != 1:
            raise ReconciliationError(
                f"Found {len(matches)} matching orders "
                f"for {client_id}; outcome UNKNOWN"
            )
        order = matches[0]

    oid = order.get("order_id")
    if not isinstance(oid, str) or not oid:
        raise ReconciliationError(
            "Missing exchange order_id"
        )

    filled, remaining, status = validate_order(
        order, ticker, client_id, requested, side,
        allow_partial=allow_partial,
    )

    fills = await paged(
        api,
        "/portfolio/fills",
        "fills",
        {"order_id": oid, "limit": "200"},
    )

    seen = set()
    total = Decimal(0)

    for fill in fills:
        if (
            fill.get("order_id") != oid
            or fill.get(
                "ticker", fill.get("market_ticker")
            ) != ticker
        ):
            raise ReconciliationError(
                "Fill identity mismatch"
            )

        if fill.get("side") not in (
            (None, "yes", "no")
            if allow_partial and side == "ask"
            else (None, "yes")
        ):
            raise ReconciliationError(
                "Unexpected NO fill for YES basket"
            )

        action = fill.get("action")
        book_side = fill.get("book_side")

        if (
            action is not None
            and action != (
                "buy" if side == "bid" else "sell"
            )
        ):
            raise ReconciliationError(
                "Fill action mismatch"
            )

        if book_side is not None and book_side != side:
            raise ReconciliationError(
                "Fill book_side mismatch"
            )

        if action is None and book_side is None:
            raise ReconciliationError(
                "Fill direction unavailable"
            )

        fid = fill.get("fill_id")
        if not fid or fid in seen:
            raise ReconciliationError(
                "Duplicate or missing fill ID"
            )
        seen.add(fid)

        total += fill_count(fill)

    if total != filled:
        raise ReconciliationError(
            f"Fill mismatch: order={filled}, "
            f"executions={total}"
        )

    return Execution(
        ticker, client_id, oid,
        requested, filled, remaining, status
    )


async def reconcile_order(
    api, *, ticker, client_id, requested,
    order_id=None, side="bid", book_side=None, tries=12,
    allow_partial=False,
):
    """Read-only verification with order-list fallback."""
    if book_side is not None:
        side = book_side

    if (
        side not in ("bid", "ask")
        or not client_id
        or not ticker
        or (
            allow_partial
            and fixed_qty(requested) < Decimal("0.01")
        )
        or (
            not allow_partial
            and requested < 1
        )
    ):
        raise ReconciliationError(
            "Invalid reconciliation identity"
        )

    if not isinstance(tries, int) or not 1 <= tries <= 12:
        raise ValueError("Invalid retry count")

    last = None

    for attempt in range(tries):
        # Try exact order ID, then client-ID lookup.
        # A 404 is NOT proof that the order failed.
        candidates = (
            (order_id, None) if order_id else (None,)
        )

        for lookup_id in candidates:
            try:
                result = await reconcile_once(
                    api,
                    ticker=ticker,
                    client_id=client_id,
                    requested=requested,
                    order_id=lookup_id,
                    side=side,
                    allow_partial=allow_partial,
                )

                if (
                    result.ticker != ticker
                    or result.client_id != client_id
                    or (
                        order_id
                        and result.order_id != order_id
                    )
                ):
                    raise ReconciliationError(
                        "Reconciliation identity mismatch"
                    )

                return result

            except Exception as exc:
                last = exc

        if attempt + 1 < tries:
            await asyncio.sleep(
                (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.5, 2.0, 2.0, 2.0)[attempt]
            )

    raise ReconciliationError(
        f"Unable to prove outcome of {client_id} "
        f"after {tries} attempts: {last}"
    ) from last


async def positions_for_event(api, event, legs):
    rows = await paged(
        api,
        "/portfolio/positions",
        "market_positions",
        {"event_ticker": event, "limit": "200"},
    )

    result = {
        leg: Decimal(0)
        for leg in legs
    }

    for row in rows:
        ticker = row.get("ticker")
        if ticker in result:
            key = (
                "position_fp"
                if row.get("position_fp") is not None
                else "position"
            )
            if row.get(key) is None:
                raise ReconciliationError(
                    "Position missing count"
                )
            result[ticker] += Decimal(
                str(row[key])
            )

    return result


async def check_positions(api, event, baseline, changes):
    wanted = {
        ticker: (
            Decimal(str(before))
            + Decimal(changes.get(ticker, 0))
        )
        for ticker, before in baseline.items()
    }

    for attempt in range(3):
        now = await positions_for_event(
            api, event, baseline
        )

        if now == wanted:
            return

        if attempt < 2:
            await asyncio.sleep(0.3)

    raise ReconciliationError(
        f"Position mismatch expected={wanted} got={now}"
    )


def retry_allowed(
    *,
    first_price,
    second_price,
    original_second_limit,
    qty_requested,
    depth,
    elapsed,
    max_cost,
    fee,
):
    """Permit one extra FOK only at the original limit or better."""
    if (
        second_price is None
        or depth < qty_requested
        or elapsed > 5.0
    ):
        return False

    if (
        second_price <= 0
        or second_price > original_second_limit
    ):
        return False

    gross = first_price + second_price
    estimated = (
        gross * qty_requested
        + fee(first_price, qty_requested)
        + fee(second_price, qty_requested)
    )

    return (
        gross < Decimal("0.97")
        and estimated < qty_requested
        and estimated <= max_cost
    )


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str


def second_leg_decision(
    first, second, *,
    second_attempts,
    second_ask,
    first_price,
    first_fee,
    available_depth,
    fee_second,
    basket_cap=Decimal("0.97"),
    max_cost=Decimal("10"),
    max_second_ask=None,
    elapsed_seconds=0,
    max_elapsed_seconds=2,
    qty=1,
    exchange_open=True,
):
    """Compatible with the existing standalone offline tests."""
    if not first.full or first.filled != qty:
        return RecoveryDecision(
            "HALT", "First leg not verified"
        )

    if second is None:
        return RecoveryDecision(
            "HALT", "Second leg unknown"
        )

    if second.full and second.filled == qty:
        return RecoveryDecision(
            "COMPLETE", "Both legs verified"
        )

    if not second.unfilled:
        return RecoveryDecision(
            "HALT", "Second leg not unfilled"
        )

    if second_attempts >= 2:
        return RecoveryDecision(
            "UNWIND", "Retry limit reached"
        )

    if (
        not exchange_open
        or second_ask is None
        or available_depth < qty
    ):
        return RecoveryDecision(
            "UNWIND", "No executable quote"
        )

    if elapsed_seconds > max_elapsed_seconds:
        return RecoveryDecision(
            "UNWIND", "Time limit reached"
        )

    if (
        max_second_ask is not None
        and second_ask > max_second_ask
    ):
        return RecoveryDecision(
            "UNWIND", "Price beyond limit"
        )

    if second_ask <= 0 or second_ask >= 1:
        return RecoveryDecision(
            "UNWIND", "Invalid quote"
        )

    cost = (
        qty * (first_price + second_ask)
        + first_fee + fee_second
    )

    if (
        first_price + second_ask >= basket_cap
        or cost >= qty
        or cost > max_cost
    ):
        return RecoveryDecision(
            "UNWIND", "Basket outside limits"
        )

    return RecoveryDecision(
        "RETRY", "One verified retry allowed"
    )
