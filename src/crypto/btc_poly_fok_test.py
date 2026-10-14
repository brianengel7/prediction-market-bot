"""One manually confirmed Polymarket US BTC YES FOK test."""

import argparse
import asyncio
import json
import os
import time

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path

from dotenv import load_dotenv
from src.crypto import cross_exchange_arb_v2 as scan
from src.crypto.backtest.cross_exchange_live_v1 import PolyOrders

LOCK = Path("arb_fast_live_attempt.lock")
STOP = Path("STOP_ARB_FAST")
AUDIT = Path("btc_poly_fok_test.jsonl")

MAX_PRICE = D("0.85")
MAX_ESTIMATED_TOTAL = D("0.95")
TIF = "TIME_IN_FORCE_FILL_OR_KILL"

TERMINAL = {
    "ORDER_STATE_FILLED",
    "ORDER_STATE_CANCELED",
    "ORDER_STATE_CANCELLED",
    "ORDER_STATE_REJECTED",
    "ORDER_STATE_EXPIRED",
}


def log(kind, **values):
    row = {
        "kind": kind,
        "utc": scan.timestamp(),
        **values,
    }

    print(
        "FOK_TEST " + json.dumps(
            row, default=str, separators=(",", ":")
        ),
        flush=True,
    )

    with AUDIT.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def brief(value):
    if not isinstance(value, dict):
        return {"type": type(value).__name__}

    order = value.get("order", value)

    if not isinstance(order, dict):
        return {"type": type(order).__name__}

    return {
        k: order.get(k)
        for k in (
            "id",
            "orderId",
            "marketSlug",
            "state",
            "tif",
            "quantity",
            "cumQuantity",
            "leavesQuantity",
            "avgPx",
            "commissionNotionalTotalCollected",
            "commissionsBasisPoints",
            "createTime",
            "lastTransactTime",
        )
    }


def price_ok(px, tick):
    return (
        D("0.01") <= px <= MAX_PRICE
        and px % tick == 0
    )


def future_window(close):
    return datetime.now(timezone.utc) < (
        close - timedelta(seconds=75)
    )


class Observer:
    def __init__(self):
        self.updated = asyncio.Event()

    def check(self, changed=None):
        if changed == "POLY":
            self.updated.set()


async def recent_ask(
    state, observer, after=0, timeout=12
):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()

        try:
            await asyncio.wait_for(
                observer.updated.wait(),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            break

        observer.updated.clear()
        book = state.poly

        if not book.ready or book.seen <= after:
            continue

        if time.monotonic() - book.seen > 0.09:
            continue

        ask = scan.best(book.offers)

        if ask:
            return ask[0], ask[1], book.seen

    raise TimeoutError(
        "No fresh Polymarket book frame within 12s"
    )


async def reconcile(poly, order_id, slug):
    """Read order status; no new trading orders."""
    last = None

    for i in range(15):
        try:
            result = await poly.order(order_id)
            last = brief(result)

            state = str(last.get("state") or "")
            tif = str(last.get("tif") or "")

            log(
                "ORDER_READ",
                iteration=i,
                order=last,
            )

            if tif and tif != TIF:
                log(
                    "TIF_MISMATCH",
                    returned=tif,
                    expected=TIF,
                )
                break

            if state in TERMINAL:
                break

        except Exception as exc:
            log(
                "READ_ERROR",
                error=type(exc).__name__,
            )

        await asyncio.sleep(0.35)

    if last:
        status = str(last.get("state") or "")
        tif = str(last.get("tif") or "")

        leaves = last.get("leavesQuantity")

        try:
            remaining = (
                D(str(leaves))
                if leaves is not None
                else None
            )
        except Exception:
            remaining = None

        needs_cancel = (
            status not in TERMINAL
            or (
                tif
                and tif != TIF
                and remaining is not None
                and remaining > 0
            )
        )

        if needs_cancel:
            try:
                cancelled = await poly.client.orders.cancel(
                    order_id,
                    {"marketSlug": slug},
                )

                log(
                    "CANCEL_REQUEST",
                    result_type=type(cancelled).__name__,
                )

                check = await poly.order(order_id)
                last = brief(check)

                log("AFTER_CANCEL", order=last)

            except Exception as exc:
                log(
                    "CANCEL_UNCERTAIN",
                    error=type(exc).__name__,
                )

    if not last:
        log(
            "UNKNOWN_HALT",
            instruction=(
                "Inspect Polymarket account "
                "before any other trade"
            ),
        )
        return

    state = str(last.get("state") or "")
    tif = str(last.get("tif") or "")

    try:
        filled = D(str(last.get("cumQuantity")))
    except Exception:
        filled = None

    if (
        tif != TIF
        or filled not in (D(0), D(1))
        or state not in TERMINAL
    ):
        log(
            "MANUAL_REVIEW_REQUIRED",
            order=last,
            instruction="Inspect account; do not resubmit",
        )

    else:
        log(
            "RESULT",
            state=state,
            filled_contracts=str(filled),
            tif=tif,
            possible_unhedged_position=(filled == 1),
            note=(
                "Verify actual fees and transaction "
                "timestamps in account activity"
            ),
        )


async def live():
    load_dotenv(Path.cwd() / ".env")

    if STOP.exists() or LOCK.exists():
        raise SystemExit(
            "BLOCKED: STOP_ARB_FAST or unresolved "
            "arb_fast_live_attempt.lock"
        )

    info = argparse.Namespace(
        extra_kalshi_series=[],
        max_extra=1,
    )

    metadata, k_ticker, slug, close = (
        await asyncio.to_thread(scan.discover, info)
    )

    if (
        not k_ticker
        or not slug
        or not future_window(close)
    ):
        raise SystemExit(
            "No current matched BTC market with "
            "at least 75 seconds remaining. Retry later."
        )

    client = PolyOrders()
    state = scan.State(poly_slug=slug)
    observer = Observer()

    feed = None
    attempted = False

    try:
        # FUNDED_FOK_GUARD_20261013
        funds = await client.open()
        rows = (
            funds.get("balances", [])
            if isinstance(funds, dict)
            else []
        )
        buying = max(
            (
                D(str(r.get("buyingPower") or 0))
                for r in rows
                if isinstance(r, dict)
                and r.get("currency") == "USD"
            ),
            default=D("0"),
        )
        if buying < D("1"):
            raise RuntimeError(
                "Less than $1 verified Polymarket buying power"
            )

        raw = await client.client.markets.retrieve_by_slug(
            slug
        )

        market = (
            raw.get("market", raw)
            if isinstance(raw, dict)
            else {}
        )

        if (
            not isinstance(market, dict)
            or market.get("closed") is True
            or market.get("active") is False
        ):
            raise RuntimeError(
                "Market is not verified open"
            )

        tick = D(str(
            market.get("orderPriceMinTickSize")
            or ".01"
        ))

        min_qty = D(str(
            market.get("minimumTradeQty")
            or "1"
        ))

        if (
            tick <= 0
            or min_qty <= 0
            or min_qty > 1
        ):
            raise RuntimeError(
                "Invalid tick or minimum quantity"
            )

        feed = asyncio.create_task(
            scan.poly_stream(state, observer)
        )

        price, depth, seen = await recent_ask(
            state, observer
        )

        if depth < 1 or not price_ok(price, tick):
            log(
                "NOT_ELIGIBLE",
                market=slug,
                yes_ask=str(price),
                depth=str(depth),
                max_accepted_yes_price=str(MAX_PRICE),
            )
            return

        leg = scan.Leg(
            "polymarket_us",
            slug,
            "yes",
            price,
            D(1),
        )

        est_fee = scan.estimate_fee(
            leg,
            D(1),
            argparse.Namespace(
                poly_rate=D(".0695")
            ),
        )

        if price + est_fee > MAX_ESTIMATED_TOTAL:
            raise RuntimeError(
                "Price plus modeled fee exceeds "
                "95-cent safety threshold"
            )

        log(
            "ARMED_PREVIEW",
            market=slug,
            kalshi_reference=k_ticker,
            side="YES",
            qty=1,
            yes_limit=str(price),
            max_price=str(MAX_PRICE),
            estimated_fee=str(est_fee),
            estimated_total=str(price + est_fee),
            tif=TIF,
            matched_pair_execution=False,
            warning=(
                "REAL unhedged trade. "
                "May lose full cost plus fees."
            ),
        )

        print(
            "\nTHIS IS A REAL POLYMARKET US BUY ORDER.",
            flush=True,
        )
        print(
            f"Exactly 1 BTC YES contract at "
            f"no more than ${MAX_PRICE}, PLUS FEES.",
            flush=True,
        )
        print(
            "Type exactly BUY 1 YES UP TO 0.85 to authorize:",
            flush=True,
        )

        answer = await asyncio.to_thread(
            input, "> "
        )

        if answer.strip() != "BUY 1 YES UP TO 0.85":
            log(
                "ABORTED",
                reason="User did not confirm",
            )
            return

        if (
            STOP.exists()
            or LOCK.exists()
            or not future_window(close)
        ):
            log(
                "ABORTED",
                reason=(
                    "Lock/kill switch/window safety check"
                ),
            )
            return

        fresh, fresh_qty, fresh_seen = await recent_ask(
            state, observer, after=seen, timeout=2
        )

        if STOP.exists() or not future_window(close):
            log(
                "ABORTED",
                reason=(
                    "Kill switch activated or "
                    "market window nearly closed"
                ),
            )
            return

        # BTC_FOK_PRICE_CAP_V2
        # Confirmation authorizes a MAXIMUM price,
        # not the earlier indicative quote.
        final_fee = scan.estimate_fee(
            scan.Leg(
                "polymarket_us", slug,
                "yes", fresh, D(1)
            ),
            D(1),
            argparse.Namespace(poly_rate=D(".0695")),
        )

        if (
            fresh_qty < 1
            or not price_ok(fresh, tick)
            or fresh + final_fee > MAX_ESTIMATED_TOTAL
            or time.monotonic() - fresh_seen > 0.09
        ):
            log(
                "ABORTED",
                reason="Fresh price/depth/budget check failed",
                initial_ask=str(price),
                latest_ask=str(fresh),
                latest_qty=str(fresh_qty),
                estimated_total=str(fresh + final_fee),
                max_price=str(MAX_PRICE),
            )
            return

        log(
            "FINAL_BOOK_CHECK",
            initial_ask=str(price),
            authorized_max=str(MAX_PRICE),
            final_limit=str(fresh),
            final_depth=str(fresh_qty),
            estimated_fee=str(final_fee),
            estimated_total=str(fresh + final_fee),
            price_moved=(fresh != price),
        )

        # Durable, exclusive, one-attempt lock.
        # Deliberately retained after submission.
        fd = os.open(
            LOCK,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )

        with os.fdopen(fd, "w") as f:
            f.write(
                json.dumps({
                    "utc": scan.timestamp(),
                    "test": "POLY_FOK_ONCE",
                    "slug": slug,
                    "limit": str(fresh),
                    "quantity": 1,
                }) + "\n"
            )
            f.flush()
            os.fsync(f.fileno())

        attempted = True

        log(
            "SUBMITTING",
            market=slug,
            yes_limit=str(fresh),
            authorized_max=str(MAX_PRICE),
            qty=1,
            tif=TIF,
        )

        started = time.perf_counter_ns()

        try:
            submitted = await client.fok(
                slug, "yes", fresh, 1
            )

        except Exception as exc:
            log(
                "SUBMISSION_UNKNOWN_OR_REJECTED",
                error=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
                server_message=str(
                    getattr(exc, "message", str(exc))
                )[:500],
                request_id=getattr(exc, "request_id", None),
                duration_ms=round(
                    (
                        time.perf_counter_ns()
                        - started
                    ) / 1e6,
                    2,
                ),
                instruction=(
                    "Do not retry. Inspect account "
                    "orders and activity."
                ),
            )
            return

        latency = (
            time.perf_counter_ns() - started
        ) / 1e6

        log(
            "ACK",
            latency_ms=round(latency, 2),
            response=brief(
                submitted.get("raw")
            ),
        )

        order_id = submitted.get("order_id")

        if not order_id:
            log(
                "UNKNOWN_HALT",
                instruction=(
                    "Missing order ID. "
                    "Do not retry."
                ),
            )
            return

        await reconcile(
            client, order_id, slug
        )

    finally:
        if feed:
            feed.cancel()
            await asyncio.gather(
                feed,
                return_exceptions=True,
            )

        await client.close()

        if attempted:
            log(
                "ONE_ATTEMPT_COMPLETE",
                lock_file=str(LOCK),
                reminder=(
                    "Lock intentionally retained. "
                    "Review order and balance."
                ),
            )


def main():
    ap = argparse.ArgumentParser()

    group = ap.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--self-test",
        action="store_true",
    )
    group.add_argument(
        "--live",
        action="store_true",
    )

    args = ap.parse_args()

    if args.self_test:
        assert price_ok(D(".61"), D(".01"))
        assert not price_ok(D(".91"), D(".01"))
        assert not price_ok(D(".605"), D(".01"))

        request = PolyOrders.payload(
            "example", "yes", D(".61"), 1
        )

        assert request["tif"] == TIF

        print(
            "PASS: price guard, tick and FOK "
            "request construction. NO ORDER."
        )

    else:
        asyncio.run(live())


if __name__ == "__main__":
    main()
