"""Independent Kalshi + Polymarket US BTC arbitrage execution pilot.

PREVIEW by default. LIVE requires explicit verification and approval.
Two separate exchange orders are never atomic.
"""

import argparse
import asyncio
import base64
import json
import os
import time
import uuid

from decimal import Decimal as D
from pathlib import Path
from urllib.parse import urlparse

import httpx

from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, ed25519
from polymarket_us import AsyncPolymarketUS

from src.crypto import cross_exchange_arb_v2 as scan


AUDIT = Path("arb_fast_pilot_orders.jsonl")
ATTEMPT_LOCK = Path("arb_fast_live_attempt.lock")
KILL_SWITCH = Path("STOP_ARB_FAST")

# AUTO_DISCOVERY_ONE_SHOT_V1
AUTO_PAIR_DISCOVERY = (
    os.getenv("ARB_FAST_AUTO_DISCOVER", "").lower()
    == "true"
)


def note(kind, **fields):
    record = {
        "kind": kind,
        "utc": scan.timestamp(),
        **fields,
    }

    print(
        "FAST_" + kind + " "
        + json.dumps(record, default=str),
        flush=True,
    )

    if kind in (
        "PREVIEW", "SUBMITTING", "ACK",
        "FILL_CHECK", "PAIR_FILLED", "NO_FILL",
        "MATCHED_FILLS_FEES_PENDING",
        "UNHEDGED_HALT", "UNKNOWN_HALT",
    ):
        with AUDIT.open("a") as f:
            f.write(
                json.dumps(record, default=str) + "\n"
            )
            f.flush()
            os.fsync(f.fileno())


class UncertainSubmission(RuntimeError):
    pass


class KalshiOrders:
    """Independent authenticated Kalshi order client."""

    def __init__(self):
        self.base = os.getenv(
            "KALSHI_BASE_URL",
            "https://external-api.kalshi.com/trade-api/v2",
        ).rstrip("/")

        self.key_id = os.environ["KALSHI_API_KEY_ID"]

        self.key = serialization.load_pem_private_key(
            Path(
                os.environ["KALSHI_PRIVATE_KEY_PATH"]
            ).read_bytes(),
            password=None,
        )

        self.client = httpx.AsyncClient(
            timeout=3.0,
            limits=httpx.Limits(
                max_keepalive_connections=5
            ),
        )

    def headers(self, method, path):
        stamp = str(int(time.time() * 1000))

        message = (
            stamp
            + method.upper()
            + urlparse(self.base + path).path
        ).encode()

        if isinstance(
            self.key,
            ed25519.Ed25519PrivateKey,
        ):
            signature = self.key.sign(message)
        else:
            signature = self.key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": stamp,
            "KALSHI-ACCESS-SIGNATURE":
                base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }

    async def get(self, path, **params):
        response = await self.client.get(
            self.base + path,
            headers=self.headers("GET", path),
            params=params or None,
        )
        response.raise_for_status()
        return response.json()

    async def prewarm(self):
        return await self.get("/portfolio/balance")

    async def fok(
        self, ticker, side, price, quantity, client_id
    ):
        path = "/portfolio/events/orders"
        p = D(str(price))

        if side not in ("yes", "no") or not D(0) < p < D(1):
            raise ValueError("Invalid Kalshi order")

        yes_price = p if side == "yes" else D(1) - p

        if yes_price != yes_price.quantize(D("0.0001")):
            raise ValueError("Unsupported price precision")

        body = {
            "ticker": ticker,
            "exchange_index": 2,
            "side": "bid" if side == "yes" else "ask",
            "count": f"{D(quantity):.2f}",
            "price": f"{yes_price:.4f}",
            "time_in_force": "fill_or_kill",
            "client_order_id": client_id,
            "self_trade_prevention_type": "taker_at_cross",
            "cancel_order_on_pause": True,
        }

        started = time.perf_counter_ns()

        try:
            response = await self.client.post(
                self.base + path,
                headers=self.headers("POST", path),
                json=body,
            )
        except (
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            raise UncertainSubmission(
                "Kalshi submission may have reached exchange"
            ) from exc

        elapsed = (
            time.perf_counter_ns() - started
        ) / 1e6

        if response.status_code in (408, 409) or (
            response.status_code >= 500
        ):
            raise UncertainSubmission(
                f"Kalshi HTTP {response.status_code}"
            )

        if response.status_code != 201:
            raise RuntimeError(
                f"Kalshi rejected {response.status_code}: "
                f"{response.text[:250]}"
            )

        try:
            result = response.json()
            order = result.get("order", result)
            order_id = order["order_id"]
        except Exception as exc:
            raise UncertainSubmission(
                "Kalshi accepted order but response is uncertain"
            ) from exc

        return {
            "order_id": order_id,
            "ack_ms": round(elapsed, 2),
            "raw": order,
        }

    async def fills(self, order_id):
        data = await self.get(
            "/portfolio/fills",
            order_id=order_id,
            limit=100,
        )
        return data.get("fills", [])

    async def order(self, order_id):
        return await self.get(
            "/portfolio/orders/" + str(order_id)
        )

    async def close(self):
        await self.client.aclose()


class PolyOrders:
    """Separate official Polymarket US SDK adapter."""

    def __init__(self):
        self.client = AsyncPolymarketUS(
            key_id=os.environ["POLYMARKET_KEY_ID"],
            secret_key=os.environ["POLYMARKET_SECRET_KEY"],
            timeout=3,
            max_retries=0,
        )

    async def open(self):
        await self.client.__aenter__()
        return await self.client.account.balances()


    # POLY_YES_PRICE_FIX_20261013
    @staticmethod
    def payload(slug, side, outcome_price, quantity):
        # Polymarket API prices are ALWAYS YES-instrument
        # prices, including ORDER_INTENT_BUY_SHORT.
        p = D(str(outcome_price))

        if (
            side not in ("yes", "no")
            or not D("0") < p < D("1")
        ):
            raise ValueError("Invalid Polymarket outcome limit")

        yes_limit = (
            p if side == "yes" else D("1") - p
        )

        if not D("0.01") <= yes_limit <= D("0.99"):
            raise ValueError(
                "Polymarket YES limit outside permitted range"
            )

        if int(quantity) != quantity or quantity < 1:
            raise ValueError(
                "This pilot requires whole contracts"
            )

        return {
            "marketSlug": slug,
            "intent": (
                "ORDER_INTENT_BUY_LONG"
                if side == "yes"
                else "ORDER_INTENT_BUY_SHORT"
            ),
            "type": "ORDER_TYPE_LIMIT",
            "price": {
                "value": str(yes_limit),
                "currency": "USD",
            },
            "quantity": int(quantity),
            "tif": "TIME_IN_FORCE_FILL_OR_KILL",
            "manualOrderIndicator":
                "MANUAL_ORDER_INDICATOR_AUTOMATIC",
            "participateDontInitiate": False,
        }

    async def preview(
        self, slug, side, price, quantity
    ):
        return await self.client.orders.preview(
            {"request": self.payload(
                slug, side, price, quantity
            )}
        )

    async def fok(
        self, slug, side, price, quantity
    ):
        started = time.perf_counter_ns()

        try:
            # FOK_EXECUTION_AUDIT_V1
            request = self.payload(
                slug, side, price, quantity
            )

            if request["tif"] != (
                "TIME_IN_FORCE_FILL_OR_KILL"
            ):
                raise RuntimeError(
                    "Refusing non-FOK Polymarket order"
                )

            request["synchronousExecution"] = True
            request["maxBlockTime"] = "2"

            raw = await self.client.orders.create(
                request
            )

            # FOK_ACK_DIAGNOSTIC_V1
            if isinstance(raw, dict):
                executions = raw.get("executions") or []
                orders = []
                event_types = []

                if isinstance(raw.get("order"), dict):
                    orders.append(raw["order"])

                if isinstance(executions, list):
                    for event in executions:
                        if not isinstance(event, dict):
                            continue
                        event_types.append(event.get("type"))
                        if isinstance(event.get("order"), dict):
                            orders.append(event["order"])

                print(
                    "POLY_FOK_ACK_DIAGNOSTIC "
                    + json.dumps({
                        "requested_tif": request["tif"],
                        "response_order_id": (
                            raw.get("id")
                            or raw.get("orderId")
                        ),
                        "exchange_reported_tifs": [
                            o.get("tif") for o in orders
                        ],
                        "exchange_order_states": [
                            o.get("state") for o in orders
                        ],
                        "execution_types": event_types,
                    }, default=str),
                    flush=True,
                )
        except Exception as exc:
            if type(exc).__name__ in (
                "APITimeoutError",
                "APIConnectionError",
                "TimeoutError",
            ):
                raise UncertainSubmission(
                    "Polymarket submission state uncertain"
                ) from exc
            raise

        elapsed = (
            time.perf_counter_ns() - started
        ) / 1e6

        order = (
            raw.get("order", raw)
            if isinstance(raw, dict)
            else {}
        )

        order_id = (
            order.get("id")
            or order.get("orderId")
        )

        if not order_id:
            raise UncertainSubmission(
                "Polymarket response missing order ID"
            )

        return {
            "order_id": order_id,
            "ack_ms": round(elapsed, 2),
            "raw": raw,
        }

    async def order(self, order_id):
        return await self.client.orders.retrieve(
            order_id
        )

    async def close(self):
        await self.client.__aexit__(
            None, None, None
        )



# FILL_RECON_V2_20261013
# Reconciliation only: no order submission.

def _money(value):
    if isinstance(value, dict):
        value = value.get("value")
    if value in (None, ""):
        return None

    result = D(str(value))

    if not result.is_finite():
        raise ValueError("Non-finite exchange amount")

    return result


def _kalshi_fills(fills, side, expected_id):
    count = D(0)
    cost = D(0)
    fees = D(0)
    fees_complete = True
    seen = set()

    price_key = (
        "no_price_dollars"
        if side == "no"
        else "yes_price_dollars"
    )

    for fill in fills:
        if not isinstance(fill, dict):
            raise ValueError("Invalid Kalshi fill")

        if fill.get("order_id") not in (
            None, expected_id
        ):
            raise ValueError("Wrong order ID in fill")

        fill_id = fill.get("fill_id")

        if fill_id is not None and fill_id in seen:
            continue

        if fill_id is not None:
            seen.add(fill_id)

        q = _money(
            fill.get("count_fp", fill.get("count"))
        )
        px = _money(fill.get(price_key))

        if (
            q is None
            or px is None
            or q <= 0
            or not 0 < px < 1
        ):
            raise ValueError(
                "Invalid Kalshi fill quantity or price"
            )

        fee = _money(fill.get("fee_cost"))

        if fee is None:
            fees_complete = False
        elif fee < 0:
            raise ValueError("Negative fee")
        else:
            fees += fee

        count += q
        cost += q * px

    return (
        count,
        cost,
        fees if fees_complete else None,
    )


async def reconcile_pair(
    pilot, kalshi, poly,
    k_result, p_result, quantity
):
    """Poll both exchanges without submitting new orders."""

    target = D(str(quantity))

    k_id = (
        k_result.get("order_id")
        if isinstance(k_result, dict)
        else None
    )
    p_id = (
        p_result.get("order_id")
        if isinstance(p_result, dict)
        else None
    )

    if not k_id or not p_id:
        note(
            "UNKNOWN_HALT",
            reason="One or both order IDs missing",
            kalshi_id=k_id,
            poly_id=p_id,
            kalshi_error=(
                repr(k_result)
                if isinstance(k_result, Exception)
                else None
            ),
            poly_error=(
                repr(p_result)
                if isinstance(p_result, Exception)
                else None
            ),
            instruction=(
                "Inspect both accounts. "
                "Do not resubmit."
            ),
        )

    k_terminal = {
        "executed",
        "filled",
        "canceled",
        "cancelled",
        "rejected",
        "expired",
    }

    p_terminal = {
        "ORDER_STATE_FILLED",
        "ORDER_STATE_CANCELED",
        "ORDER_STATE_REJECTED",
        "ORDER_STATE_EXPIRED",
    }

    last = {}
    deadline = time.monotonic() + 12.0

    while time.monotonic() < deadline:

        async def read_k():
            if not k_id:
                return None
            return await pilot.k.order(k_id)

        async def read_fills():
            if not k_id:
                return None
            return await pilot.k.fills(k_id)

        async def read_p():
            if not p_id:
                return None
            return await pilot.p.order(p_id)

        kr, fills, pr = await asyncio.gather(
            read_k(),
            read_fills(),
            read_p(),
            return_exceptions=True,
        )

        errors = {
            key: (
                f"{type(value).__name__}: {value}"
            )
            for key, value in (
                ("kalshi_order", kr),
                ("kalshi_fills", fills),
                ("polymarket_order", pr),
            )
            if isinstance(value, BaseException)
        }

        ko = (
            kr.get("order", kr)
            if isinstance(kr, dict)
            else None
        )

        po = (
            pr.get("order", pr)
            if isinstance(pr, dict)
            else None
        )

        if not isinstance(ko, dict):
            ko = None

        if not isinstance(po, dict):
            po = None

        ks = (
            str(ko.get("status", "")).lower()
            if ko
            else "UNKNOWN"
        )

        ps = (
            str(po.get("state", ""))
            if po
            else "UNKNOWN"
        )

        last = {
            "ko": ko,
            "po": po,
            "fills": fills,
            "errors": errors,
            "kalshi_status": ks,
            "poly_status": ps,
        }

        if (
            k_id
            and p_id
            and isinstance(fills, list)
            and ks in k_terminal
            and ps in p_terminal
        ):
            break

        await asyncio.sleep(0.3)

    po = last.get("po")
    fills = last.get("fills")
    ks = last.get(
        "kalshi_status", "UNKNOWN"
    )
    ps = last.get(
        "poly_status", "UNKNOWN"
    )
    errs = last.get("errors", {})

    k_qty = None
    k_cost = None
    k_fees = None
    p_qty = None
    p_avg = None
    p_cost = None

    try:
        if isinstance(fills, list):
            k_qty, k_cost, k_fees = _kalshi_fills(
                fills, kalshi[2], k_id
            )

        if po is not None:
            p_qty = _money(
                po.get("cumQuantity")
            )

            p_avg = _money(
                po.get("avgPx")
            )

            if (
                p_qty is not None
                and p_avg is not None
            ):
                if (
                    p_qty > 0
                    and not D(0) < p_avg < D(1)
                ):
                    raise ValueError(
                        "Invalid Polymarket execution price"
                    )

                if poly[2] == "no":
                    # BUY_SHORT sells YES.
                    # Effective NO cost = 1 - YES sale price.
                    p_cost = p_qty * (D(1) - p_avg)
                elif poly[2] == "yes":
                    p_cost = p_qty * p_avg
                else:
                    raise ValueError(
                        "Unsupported Polymarket outcome"
                    )

    except (
        ValueError,
        TypeError,
        ArithmeticError,
    ) as exc:
        errs["fill_parse"] = (
            f"{type(exc).__name__}: {exc}"
        )

    tif = (
        po.get("tif") if po else None
    )

    poly_commission_field = (
        po.get(
            "commissionNotionalTotalCollected"
        )
        if po
        else None
    )

    note(
        "FILL_CHECK",
        kalshi_order_id=k_id,
        poly_order_id=p_id,
        kalshi_status=ks,
        polymarket_status=ps,
        polymarket_tif=tif,
        kalshi_quantity=(
            str(k_qty)
            if k_qty is not None else None
        ),
        polymarket_quantity=(
            str(p_qty)
            if p_qty is not None else None
        ),
        kalshi_cost=(
            str(k_cost)
            if k_cost is not None else None
        ),
        kalshi_actual_fees=(
            str(k_fees)
            if k_fees is not None else None
        ),
        polymarket_execution_avg_px=(
            str(p_avg)
            if p_avg is not None else None
        ),
        polymarket_effective_outcome_cost=(
            str(p_cost)
            if p_cost is not None else None
        ),
        polymarket_commission_field_unverified=(
            poly_commission_field
        ),
        errors=errs,
    )

    both_terminal = (
        ks in k_terminal
        and ps in p_terminal
    )

    matched = (
        not errs
        and both_terminal
        and k_qty == target
        and p_qty == target
        and k_cost is not None
        and k_fees is not None
        and p_avg is not None
        and p_avg > 0
        and po.get("intent") == (
            "ORDER_INTENT_BUY_SHORT"
            if poly[2] == "no"
            else "ORDER_INTENT_BUY_LONG"
        )
        and po.get("outcomeSide") == (
            "OUTCOME_SIDE_NO"
            if poly[2] == "no"
            else "OUTCOME_SIDE_YES"
        )
        and tif
            == "TIME_IN_FORCE_FILL_OR_KILL"
    )

    if matched:
        gross_less_k_fee = (
            target
            - k_cost
            - p_cost
            - k_fees
        )

        note(
            "MATCHED_FILLS_FEES_PENDING",
            quantity=str(target),
            gross_less_kalshi_fee=(
                str(gross_less_k_fee)
            ),
            polymarket_actual_fee=(
                "NOT_VERIFIED"
            ),
            warning=(
                "Matched quantities are not "
                "verified profit. Confirm "
                "Polymarket fees and settlement."
            ),
        )

    elif (
        both_terminal
        and k_qty == 0
        and p_qty == 0
        and not errs
    ):
        note(
            "NO_FILL",
            message="Neither exchange filled",
        )

    elif (
        k_qty is not None
        and p_qty is not None
        and k_qty != p_qty
    ):
        note(
            "UNHEDGED_HALT",
            kalshi_quantity=str(k_qty),
            poly_quantity=str(p_qty),
            warning=(
                "Unmatched position. "
                "Inspect both accounts."
            ),
        )

    else:
        note(
            "UNKNOWN_HALT",
            reason=(
                "Order state, FOK policy, "
                "price or fills not verified"
            ),
            kalshi_status=ks,
            poly_status=ps,
            poly_tif=tif,
            errors=errs,
            instruction=(
                "Reconcile manually. "
                "Do not resubmit."
            ),
        )


class Pilot:
    def __init__(self, args):
        self.args = args
        self.k = None
        self.p = None
        self.busy = False
        self.live_attempted = False
        self.preflight_sides = set()
        self.preview_count = 0
        self.last_preview_time = 0.0
        self.pending = set()

    async def open(self):
        self.k = KalshiOrders()
        self.p = PolyOrders()

        try:
            kb, pb = await asyncio.gather(
                self.k.prewarm(),
                self.p.open(),
            )
        except Exception:
            await self.close()
            raise

        note(
            "AUTH_OK",
            kalshi_balance_present=bool(kb),
            poly_balance_present=bool(pb),
            mode=self.args.mode,
        )

    async def close(self):
        if self.k is not None:
            await self.k.close()

        if self.p is not None:
            await self.p.close()

    def observe(self, engine, kind, legs):
        a = self.args
        # ONE_SHOT_DIRECT_LIVE_V1
        # Price the exact quantity the executor will order.
        if a.mode == "live":
            engine.args.eval_qty = D(a.qty)

        if (
            kind != "CROSS_EXCHANGE"
            or self.busy
            or self.live_attempted
        ):
            return

        if len(legs) != 2 or any(
            x is None for x in legs
        ):
            return

        # KYES_PNO_LIVE_V1
        # Allow both complementary cross-exchange directions.
        if a.mode == "live":
            if not (
                legs[0].venue == "kalshi"
                and legs[1].venue == "polymarket_us"
                and (
                    (legs[0].side, legs[1].side)
                    in {("no", "yes"), ("yes", "no")}
                )
            ):
                return

        state = engine.state
        now = time.monotonic()

        kb = (
            state.books.get(state.kalshi_btc)
            if state.kalshi_btc else None
        )

        if not (
            kb
            and kb.ready
            and state.poly.ready
            and state.poly_slug
        ):
            return

        max_age = a.max_age_ms / 1000

        if max(
            now - kb.seen,
            now - state.poly.seen,
        ) > max_age:
            return

        if a.mode == "live":
            if AUTO_PAIR_DISCOVERY:
                # Use the scanner's active market pair.
                if (
                    not state.kalshi_btc
                    or not state.poly_slug
                    or legs[0].ticker != state.kalshi_btc
                    or legs[1].ticker != state.poly_slug
                ):
                    return
            elif (
                state.kalshi_btc != a.approved_kalshi
                or state.poly_slug != a.approved_poly
            ):
                return

        elif self.preview_count >= a.max_previews:
            return

        # FAST_EXECUTION_ONLY_OPT_V1
        if a.mode != "live" and now - self.last_preview_time < 3:
            return

        record = scan.priced_spread(
            "CROSS_EXCHANGE",
            legs,
            engine.args,
            {"rule_match": state.pair_status},
        )

        if record is None:
            return

        if (
            D(record["buffered_cents_per_pair"])
            < a.min_net_cents
        ):
            return

        if (
            D(record["evaluated_pairs"]) < a.qty
            or any(x.size < a.qty for x in legs)
        ):
            return

        pair = [
            (
                x.venue, x.ticker,
                x.side, str(x.price)
            )
            for x in legs
        ]

        if (
            pair[0][0] != "kalshi"
            or pair[1][0] != "polymarket_us"
        ):
            return

        self.busy = True
        self.last_preview_time = now

        task = asyncio.create_task(
            self.act(
                engine,
                pair,
                record,
                time.perf_counter_ns(),
            )
        )

        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def act(
        self, engine, pair, row, detected_ns
    ):
        a = self.args

        try:
            kalshi, poly = pair

            # Preview does not submit real orders.
            if (
                a.mode == "preview"
                or (
                    a.mode != "live"
                    and poly[2] not in self.preflight_sides
                )
            ):
                started = time.perf_counter_ns()

                preview = await self.p.preview(
                    poly[1],
                    poly[2],
                    poly[3],
                    a.qty,
                )

                self.preview_count += 1
                self.preflight_sides.add(poly[2])

                note(
                    "PREVIEW",
                    direction=(
                        kalshi[2], poly[2]
                    ),
                    prices=[
                        kalshi[3], poly[3]
                    ],
                    qty=a.qty,
                    net_est_cents=row[
                        "buffered_cents_per_pair"
                    ],
                    latency_ms=round(
                        (
                            time.perf_counter_ns()
                            - started
                        ) / 1e6,
                        2,
                    ),
                    details=preview,
                )

                if a.mode == "live":
                    note(
                        "PREFLIGHT_READY",
                        message=(
                            "No orders sent on this signal. "
                            "Waiting for another fresh edge."
                        ),
                    )

                return

            # Recheck both order books.
            state = engine.state

            if (
                state.kalshi_btc != kalshi[1]
                or state.poly_slug != poly[1]
            ):
                return

            kb = state.books.get(kalshi[1])

            if not (
                kb and kb.ready and state.poly.ready
            ):
                return

            max_age = a.max_age_ms / 1000
            now = time.monotonic()

            if max(
                now - kb.seen,
                now - state.poly.seen,
            ) > max_age:
                return

            k_leg = kb.ask(
                kalshi[1],
                kalshi[2],
                max_age,
            )

            p_leg = state.poly.ask(
                poly[1],
                poly[2],
                max_age,
            )

            if not k_leg or not p_leg:
                return

            if (
                k_leg.size < a.qty
                or p_leg.size < a.qty
            ):
                return

            # Never chase worse entry prices.
            if (
                k_leg.price > D(kalshi[3])
                or p_leg.price > D(poly[3])
            ):
                return

            latest = scan.priced_spread(
                "CROSS_EXCHANGE",
                [k_leg, p_leg],
                engine.args,
            )

            if (
                latest is None
                or D(
                    latest["buffered_cents_per_pair"]
                ) < a.min_net_cents
            ):
                return

            if (
                k_leg.price + p_leg.price
                > a.max_combined_price
            ):
                return

            if KILL_SWITCH.exists():
                return

            self.live_attempted = True
            client_id = str(uuid.uuid4())

            # Durable lock prevents duplicate attempts.
            try:
                fd = os.open(
                    ATTEMPT_LOCK,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                note(
                    "UNKNOWN_HALT",
                    message="Existing live-attempt lock",
                )
                return

            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({
                    "at": scan.timestamp(),
                    "client_order_id": client_id,
                    "kalshi": kalshi[1],
                    "poly": poly[1],
                }) + "\n")
                f.flush()
                os.fsync(f.fileno())

            dispatch_ns = time.perf_counter_ns()

            # Concurrent FOK is NOT atomic.
            k_result, p_result = await asyncio.gather(
                self.k.fok(
                    kalshi[1], kalshi[2],
                    str(k_leg.price), a.qty, client_id,
                ),
                self.p.fok(
                    poly[1], poly[2],
                    str(p_leg.price), a.qty,
                ),
                return_exceptions=True,
            )

            # Deferred submission logging.
            note(
                "SUBMITTING",
                pair=pair,
                submitted_limit_prices=[
                    str(k_leg.price), str(p_leg.price),
                ],
                client_id=client_id,
                qty=a.qty,
                observed_to_submit_ms=round(
                    (dispatch_ns - detected_ns) / 1e6, 2
                ),
                net_est_cents=latest[
                    "buffered_cents_per_pair"
                ],
                logged_after_order_responses=True,
            )

            note(
                "ACK",
                kalshi=(
                    repr(k_result)
                    if isinstance(k_result, Exception)
                    else k_result
                ),
                poly=(
                    repr(p_result)
                    if isinstance(p_result, Exception)
                    else p_result
                ),
            )

            await reconcile_pair(
                self,
                kalshi,
                poly,
                k_result,
                p_result,
                a.qty,
            )

        except Exception as exc:
            note(
                "UNKNOWN_HALT",
                error=(
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:250]}"
                ),
            )

        finally:
            self.busy = False
            if a.mode == "live" and self.live_attempted:
                note(
                    "ONE_SHOT_ATTEMPT_FINISHED",
                    message=(
                        "Submission attempt and "
                        "reconciliation routine ended"
                    ),
                )




async def preflight_ws_book(slug):
    """Get current top of book from authenticated WS.

    Read-only: never creates, modifies or cancels orders.
    """

    state = scan.State(poly_slug=slug)

    class BookObserver:
        def check(self, changed=None):
            pass

    task = asyncio.create_task(
        scan.poly_stream(state, BookObserver())
    )

    try:
        deadline = time.monotonic() + 12

        while time.monotonic() < deadline:
            book = state.poly

            if (
                book.ready
                and time.monotonic() - book.seen < 1.0
            ):
                bid = scan.best(
                    book.bids, high=True
                )
                ask = scan.best(book.offers)

                if bid and ask and bid[0] < ask[0]:
                    print(
                        "FAST_WS_BOOK_OK "
                        f"slug={slug} "
                        f"YES_bid={bid[0]} "
                        f"YES_ask={ask[0]} "
                        f"bid_qty={bid[1]} "
                        f"ask_qty={ask[1]}",
                        flush=True,
                    )

                    return {
                        "marketData": {
                            "state": "MARKET_STATE_OPEN",
                            "bids": [{
                                "px": {
                                    "value": str(bid[0])
                                },
                                "qty": str(bid[1]),
                            }],
                            "offers": [{
                                "px": {
                                    "value": str(ask[0])
                                },
                                "qty": str(ask[1]),
                            }],
                        }
                    }

            if task.done():
                raise RuntimeError(
                    "Polymarket WebSocket stopped "
                    "before providing a valid book"
                )

            await asyncio.sleep(0.05)

        raise TimeoutError(
            "No valid two-sided Polymarket BTC "
            "book received within 12 seconds"
        )

    finally:
        task.cancel()
        await asyncio.gather(
            task, return_exceptions=True
        )


async def preflight_orders(pilot, approved=None):
    """Authenticated preview. Never submits orders."""

    discover_args = argparse.Namespace(
        extra_kalshi_series=[],
        max_extra=1,
    )

    metadata, kalshi, poly, close = (
        await asyncio.to_thread(
            scan.discover, discover_args
        )
    )

    if not kalshi or not poly:
        raise RuntimeError(
            "Current BTC pair not listed yet; "
            "wait and retry preflight"
        )

    if (
        approved is not None
        and approved != (kalshi, poly)
    ):
        raise RuntimeError(
            "Current BTC market differs from "
            "approved market IDs; no orders"
        )

    raw_market = await pilot.p.client.markets.retrieve_by_slug(
        poly
    )
    raw_book = await preflight_ws_book(poly)

    market = raw_market.get(
        "market", raw_market
    )
    book = raw_book.get(
        "marketData", raw_book
    )

    if (
        not isinstance(market, dict)
        or not isinstance(book, dict)
    ):
        raise RuntimeError(
            "Unexpected Polymarket market/book response"
        )

    if book.get("state") not in (
        None, "OPEN", "MARKET_STATE_OPEN"
    ):
        raise RuntimeError("Polymarket market is not open")

    bids = scan.best(
        scan.poly_levels(book.get("bids")),
        high=True,
    )
    offers = scan.best(
        scan.poly_levels(book.get("offers"))
    )

    if (
        not bids or not offers
        or bids[0] >= offers[0]
    ):
        raise RuntimeError(
            "No valid two-sided Polymarket book"
        )

    tick = D(str(
        market.get("orderPriceMinTickSize") or "0.01"
    ))
    min_qty = D(str(
        market.get("minimumTradeQty") or "1"
    ))

    if (
        min_qty > pilot.args.qty
        or min_qty <= 0
        or tick <= 0
    ):
        raise RuntimeError(
            "Quantity below market minimum "
            "or invalid tick"
        )

    note(
        "CONTRACT_TERMS",
        kalshi=kalshi,
        poly=poly,
        close=str(close),
        kalshi_rule=metadata[kalshi].get("rules_primary"),
        poly_rules={
            k: v for k, v in market.items()
            if any(
                word in k.lower()
                for word in (
                    "rule", "description",
                    "question", "resolution", "settle"
                )
            )
        },
        minimum_qty=str(min_qty),
        price_tick=str(tick),
    )

    for side, outcome_cost in (
        ("yes", offers[0]),
        ("no", D("1") - bids[0]),
    ):
        payload = pilot.p.payload(
            poly, side, outcome_cost, pilot.args.qty
        )

        yes_limit = D(payload["price"]["value"])

        if yes_limit % tick != 0:
            raise RuntimeError(
                f"{side}: limit price not aligned to tick"
            )

        started = time.perf_counter_ns()

        preview = await pilot.p.preview(
            poly, side, outcome_cost, pilot.args.qty
        )

        note(
            "PREFLIGHT",
            kalshi=kalshi,
            poly=poly,
            side=side,
            outcome_cost=str(outcome_cost),
            yes_instrument_limit=str(yes_limit),
            intent=payload["intent"],
            quantity=pilot.args.qty,
            preview_ms=round(
                (
                    time.perf_counter_ns() - started
                ) / 1e6,
                2,
            ),
            response=preview,
        )

    note(
        "PREFLIGHT_OK",
        kalshi=kalshi,
        poly=poly,
        message=(
            "No orders submitted. "
            "Review fees and rules before LIVE."
        ),
    )

    return kalshi, poly


def arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=["preflight", "preview", "live"],
        default="preview",
    )
    ap.add_argument(
        "--minutes", type=float, default=30
    )
    ap.add_argument(
        "--qty", type=int, default=1
    )
    ap.add_argument(
        "--min-net-cents",
        type=D, default=D("5"),
    )
    ap.add_argument(
        "--max-age-ms",
        type=float, default=100,
    )
    ap.add_argument(
        "--max-combined-price",
        type=D, default=D("0.90"),
    )
    ap.add_argument(
        "--max-previews",
        type=int, default=3,
    )
    ap.add_argument("--approved-kalshi")
    ap.add_argument("--approved-poly")
    ap.add_argument(
        "--rules-and-fees-verified",
        action="store_true",
    )
    ap.add_argument(
        "--accept-leg-risk",
        action="store_true",
    )
    ap.add_argument(
        "--balances-confirmed",
        action="store_true",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
    )

    ap.add_argument(
        "--with-weather",
        action="store_true",
        help="Include NYC and Miami Kalshi weather books",
    )
    return ap.parse_args()


async def main():
    args = arguments()

    if args.self_test:
        assert PolyOrders.payload(
            "demo", "no", "0.45", 1
        )["intent"] == "ORDER_INTENT_BUY_SHORT"

        assert PolyOrders.payload(
            "demo", "yes", "0.55", 1
        )["tif"] == "TIME_IN_FORCE_FILL_OR_KILL"

        assert PolyOrders.payload(
            "demo", "no", "0.83", 1
        )["price"]["value"] == "0.17"

        assert PolyOrders.payload(
            "demo", "yes", "0.55", 1
        )["price"]["value"] == "0.55"

        assert PolyOrders.payload(
            "demo", "no", "0.83", 1
        )["manualOrderIndicator"] == (
            "MANUAL_ORDER_INDICATOR_AUTOMATIC"
        )

        print(
            "PASS: Order construction self-test. "
            "No orders submitted."
        )
        return

    if (
        args.qty != 1
        or not D(0)
        < args.max_combined_price
        < D(1)
    ):
        raise SystemExit(
            "Pilot requires qty=1 and combined price below $1"
        )

    if (
        args.minutes <= 0
        or args.max_age_ms <= 0
        or args.min_net_cents < 0
    ):
        raise SystemExit("Invalid settings")

    load_dotenv(
        dotenv_path=(
            Path(__file__).resolve().parents[3]
            / ".env"
        )
    )

    if args.mode == "live":
        if not (
            os.getenv(
                "KALSHI_LIVE_TRADING_ENABLED", ""
            ).lower() == "true"
            and (
                AUTO_PAIR_DISCOVERY
                or (
                    args.approved_kalshi
                    and args.approved_poly
                )
            )
            and args.rules_and_fees_verified
            and args.accept_leg_risk
            and args.balances_confirmed
        ):
            raise SystemExit(
                "LIVE BLOCKED: explicit verification and "
                "approval required"
            )

        if ATTEMPT_LOCK.exists():
            raise SystemExit(
                "LIVE BLOCKED: previous attempt unresolved"
            )

        if KILL_SWITCH.exists():
            raise SystemExit(
                "LIVE BLOCKED: STOP_ARB_FAST exists"
            )

    pilot = Pilot(args)
    await pilot.open()

    if args.mode in ("preflight", "live"):
        try:
            # AUTO_PREFLIGHT_APPROVAL_V2
            # Automatic mode validates the discovered pair.
            # Manual mode still enforces explicit IDs.
            approved = (
                None
                if AUTO_PAIR_DISCOVERY
                else (
                    (
                        args.approved_kalshi,
                        args.approved_poly,
                    )
                    if args.mode == "live"
                    else None
                )
            )

            await preflight_orders(pilot, approved)

        except BaseException:
            await pilot.close()
            raise

        if args.mode == "preflight":
            await pilot.close()
            return

        # Validates API preview only.
        # Does not establish actual fills or matching rules.
        pilot.preflight_sides.update(
            ("yes", "no")
        )

    original_consider = scan.Engine.consider

    def wrapped(
        engine, kind, legs,
        extra=None, family=None
    ):
        if kind == "CROSS_EXCHANGE":
            pilot.observe(engine, kind, legs)

        return original_consider(
            engine, kind, legs, extra, family
        )

    scan.Engine.consider = wrapped

    scan_args = argparse.Namespace(
        minutes=args.minutes,
        log_db=False,
        extra_kalshi_series=(
            ["KXTEMPNYCHS", "KXTEMPMIAH"]
            if args.with_weather else []
        ),
        max_extra=12,
        max_age=args.max_age_ms / 1000,
        min_qty=D("1"),
        eval_qty=D(str(args.qty)),
        min_edge=args.min_net_cents / 100,
        kalshi_rate=D("0.07"),
        poly_rate=D("0.0695"),
        slip=D("0.01"),
        k_fee=D("0.02"),
        p_fee=D("0.02"),
        cooldown=3,
        status_every=10,
        close_buffer=5,
        exhaustive_set=[],
        exclusive_set=[],
        execution_only=(args.mode == "live"),
        no_weather=not args.with_weather,
    )

    note(
        "READY",
        mode=args.mode,
        min_net_cents=str(args.min_net_cents),
        quantity=args.qty,
        max_age_ms=args.max_age_ms,
    )

    try:
        await scan.run(scan_args)

    finally:
        scan.Engine.consider = original_consider

        if pilot.pending:
            await asyncio.gather(
                *tuple(pilot.pending),
                return_exceptions=True,
            )

        await pilot.close()


if __name__ == "__main__":
    asyncio.run(main())
