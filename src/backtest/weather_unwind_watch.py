"""Read-only GTC unwind recovery with automatic systemd restart."""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile

from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

from src.backtest.weather_multicity_live import (
    Kalshi,
    SAFE_STATUSES,
    positions_for_event,
)

STATE = Path(".weather_multicity_live_state.json")
LOCK = Path(".weather_multicity_live_process.lock")
RESTART_MARKER = Path(".weather_unwind_restart_pending")

WATCH_STATUSES = {
    "UNWIND_RESTING",
    "EXPOSED_UNWIND_UNKNOWN",
}


class SafetyHold(Exception):
    pass


def qty(obj, *names):
    """Read fixed-point contract counts without rounding."""
    for name in names:
        value = obj.get(name)

        if value is not None:
            try:
                number = D(str(value))
            except Exception as exc:
                raise SafetyHold(
                    f"Invalid {name}: {value!r}"
                ) from exc

            if not number.is_finite() or number < 0:
                raise SafetyHold(
                    f"Invalid quantity {name}: {number}"
                )

            # Kalshi count_fp has 0.01-contract precision.
            if number != number.quantize(D("0.01")):
                raise SafetyHold(
                    f"Invalid precision in {name}: {number}"
                )

            return number

    raise SafetyHold(
        "Missing quantity: " + ", ".join(names)
    )


def exact_order(
    orders, ticker, *, order_id=None, client_id=None
):
    if not order_id and not client_id:
        raise SafetyHold("Missing order identity")

    matches = [
        order for order in orders
        if order.get("ticker") == ticker
        and (
            not order_id
            or order.get("order_id") == order_id
        )
        and (
            not client_id
            or order.get("client_order_id") == client_id
        )
    ]

    if len(matches) > 1:
        raise SafetyHold(
            f"Ambiguous exchange order for {ticker}"
        )

    return matches[0] if matches else None


def verify_order(
    order, fills, ticker, action, requested
):
    oid = order.get("order_id")

    if not oid or order.get("ticker") != ticker:
        raise SafetyHold("Order identity mismatch")

    if order.get("action") not in (None, action):
        raise SafetyHold("Unexpected order action")

    expected_book_side = (
        "bid" if action == "buy" else "ask"
    )
    expected_outcome_side = (
        "yes" if action == "buy" else "no"
    )

    def check_direction(item, label):
        book_side = item.get("book_side")
        outcome_side = item.get("outcome_side")
        legacy_side = item.get("side")

        # Prefer the new canonical Kalshi fields.
        if (
            book_side is not None
            and book_side != expected_book_side
        ):
            raise SafetyHold(
                f"{label} book_side mismatch: "
                f"{book_side!r} != {expected_book_side!r}"
            )

        if (
            outcome_side is not None
            and outcome_side != expected_outcome_side
        ):
            raise SafetyHold(
                f"{label} outcome_side mismatch: "
                f"{outcome_side!r} != {expected_outcome_side!r}"
            )

        # Legacy API versions can represent the
        # underlying YES contract and transaction
        # direction differently on sell records.
        if legacy_side is not None:
            if legacy_side in ("bid", "ask"):
                if legacy_side != expected_book_side:
                    raise SafetyHold(
                        f"{label} legacy book side mismatch"
                    )
            elif legacy_side in ("yes", "no"):
                if (
                    action == "buy"
                    and legacy_side != "yes"
                ):
                    raise SafetyHold(
                        f"{label} BUY outcome mismatch"
                    )
                # On sells, verify action and the
                # canonical direction where supplied.
            else:
                raise SafetyHold(
                    f"{label} unexpected side: "
                    f"{legacy_side!r}"
                )

    check_direction(order, "Order")

    reported = qty(
        order, "fill_count_fp", "fill_count"
    )
    remaining = qty(
        order, "remaining_count_fp", "remaining_count"
    )

    if reported + remaining > requested:
        raise SafetyHold(
            "Order quantity exceeds requested quantity"
        )

    seen = set()
    actual_filled = 0

    for fill in fills:
        if fill.get("order_id") != oid:
            raise SafetyHold("Fill order ID mismatch")

        fill_ticker = fill.get(
            "ticker", fill.get("market_ticker")
        )

        if fill_ticker != ticker:
            raise SafetyHold("Fill ticker mismatch")

        if fill.get("action") not in (None, action):
            raise SafetyHold("Fill action mismatch")

        check_direction(fill, "Fill")

        fill_id = fill.get("fill_id")

        if not fill_id or fill_id in seen:
            raise SafetyHold(
                "Missing or duplicate fill ID"
            )

        seen.add(fill_id)

        actual_filled += qty(
            fill, "count_fp", "count"
        )

    if reported != actual_filled:
        raise SafetyHold(
            f"Order/fill mismatch on {oid}: "
            f"reported={reported}, actual={actual_filled}"
        )

    return (
        reported,
        remaining,
        str(order.get("status", "")).lower(),
    )


def classify(
    record, orders_by_ticker, fills_by_id, positions
):
    """
    Pure verification logic.
    Missing evidence is never treated as a zero fill.
    """
    if record.get("status") not in WATCH_STATUSES:
        raise SafetyHold(
            "Unsupported recovery status"
        )

    legs = record.get("legs")

    if (
        not isinstance(legs, list)
        or len(legs) != 2
        or legs[0] == legs[1]
    ):
        raise SafetyHold("Invalid basket legs")

    requested = int(record.get("qty", 0))

    if requested < 1:
        raise SafetyHold("Invalid recorded quantity")

    unwind_ticker = record.get("unwind_ticker")

    if unwind_ticker not in legs:
        raise SafetyHold(
            "Unwind ticker missing from basket"
        )

    if not record.get("unwind_client_id"):
        raise SafetyHold("Missing unwind client ID")

    if (
        record.get("second_retry_client_id")
        or record.get("second_retry_order_id")
    ):
        raise SafetyHold(
            "Unexpected second-leg retry"
        )

    baseline = record.get("pre_positions")

    if (
        not isinstance(baseline, dict)
        or any(t not in baseline for t in legs)
    ):
        raise SafetyHold(
            "Missing original position baseline"
        )

    # Verify BOTH purchases independently.
    for index, ticker in enumerate(legs):
        prefix = "first" if index == 0 else "second"

        client_id = record.get(
            prefix + "_client_id"
        )
        order_id = record.get(
            prefix + "_order_id"
        )
        rejected = bool(
            record.get(prefix + "_rejected")
        )

        if not client_id and not rejected:
            raise SafetyHold(
                "Missing BUY client ID: " + prefix
            )

        buy = (
            exact_order(
                orders_by_ticker[ticker],
                ticker,
                order_id=order_id,
                client_id=client_id,
            )
            if client_id else None
        )

        if buy is None:
            if rejected and not order_id:
                filled = 0
            else:
                raise SafetyHold(
                    "Unverified BUY: " + prefix
                )
        else:
            filled, remaining, status = verify_order(
                buy,
                fills_by_id[buy["order_id"]],
                ticker,
                "buy",
                requested,
            )

            if (
                remaining != 0
                or status not in (
                    "executed",
                    "canceled",
                    "cancelled",
                )
            ):
                raise SafetyHold(
                    "BUY not terminal: " + prefix
                )

        expected = (
            requested if ticker == unwind_ticker
            else 0
        )

        if filled != expected:
            raise SafetyHold(
                f"{prefix} filled={filled}, "
                f"expected={expected}"
            )

    # Verify the exact protective GTC sell.
    sell = exact_order(
        orders_by_ticker[unwind_ticker],
        unwind_ticker,
        order_id=record.get("unwind_order_id"),
        client_id=record["unwind_client_id"],
    )

    if sell is None:
        return (
            "PENDING",
            "GTC not yet visible; outcome remains unknown",
        )

    sold, remaining, status = verify_order(
        sell,
        fills_by_id[sell["order_id"]],
        unwind_ticker,
        "sell",
        requested,
    )

    if (
        sold != requested
        or remaining != 0
        or status != "executed"
    ):
        if status in ("resting", "pending"):
            return (
                "PENDING",
                f"GTC sold={sold}/{requested}, "
                f"remaining={remaining}",
            )

        raise SafetyHold(
            f"GTC not fully executed: "
            f"sold={sold}/{requested}, "
            f"remaining={remaining}, status={status}"
        )

    # Verify positions returned to their pre-trade levels.
    for ticker in legs:
        current = D(str(positions.get(ticker, 0)))
        before = D(str(baseline[ticker]))

        if (
            not current.is_finite()
            or current != before
        ):
            raise SafetyHold(
                f"Position not restored on {ticker}: "
                f"before={before}, current={current}"
            )

        # There must be no residual resting orders.
        for order in orders_by_ticker[ticker]:
            order_status = str(
                order.get("status", "")
            ).lower()

            order_remaining = qty(
                order,
                "remaining_count_fp",
                "remaining_count",
            )

            if (
                order_status in ("resting", "pending")
                or order_remaining > 0
            ):
                raise SafetyHold(
                    "Outstanding order on " + ticker
                )

    return "READY", sell["order_id"]


async def paged(
    api, route, kind, params, max_pages=30
):
    rows = []
    cursor = None

    for _ in range(max_pages):
        query = dict(params)

        if cursor:
            query["cursor"] = cursor

        response = await api.client.get(
            api.base + route,
            params=query,
            headers=api.headers("GET", route),
        )

        response.raise_for_status()
        payload = response.json()

        if (
            not isinstance(payload, dict)
            or not isinstance(
                payload.get(kind), list
            )
        ):
            raise SafetyHold(
                "Malformed Kalshi " + kind + " response"
            )

        rows.extend(payload[kind])
        next_cursor = payload.get("cursor")

        if not next_cursor:
            return rows

        if next_cursor == cursor:
            raise SafetyHold(
                "Repeated pagination cursor"
            )

        cursor = next_cursor

    raise SafetyHold(
        "Exchange pagination incomplete: " + kind
    )


async def exchange_snapshot(api, record):
    legs = record["legs"]
    orders_by_ticker = {}

    for ticker in legs:
        orders_by_ticker[ticker] = await paged(
            api,
            "/portfolio/orders",
            "orders",
            {"ticker": ticker, "limit": 200},
        )

    order_ids = {
        order["order_id"]
        for orders in orders_by_ticker.values()
        for order in orders
        if order.get("order_id")
    }

    fills_by_id = {}

    for order_id in order_ids:
        fills_by_id[order_id] = await paged(
            api,
            "/portfolio/fills",
            "fills",
            {
                "order_id": order_id,
                "limit": 200,
            },
        )

    event = record["key"].split(":", 1)[1]

    positions = await positions_for_event(
        api, event, legs
    )

    return (
        orders_by_ticker,
        fills_by_id,
        positions,
    )


def save_resolved(data, original_bytes, updates):
    if STATE.read_bytes() != original_bytes:
        raise SafetyHold(
            "Journal changed during verification"
        )

    for record in data["attempts"]:
        if record.get("id") in updates:
            record.update(updates[record["id"]])

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup = STATE.with_name(
        STATE.name + ".before_auto_unwind." + stamp
    )

    if backup.exists():
        raise SafetyHold(
            "Journal backup name collision"
        )

    shutil.copy2(STATE, backup)
    original_stat = STATE.stat()

    fd, temp_path = tempfile.mkstemp(
        prefix=".weather_unwind_atomic_",
        dir=str(STATE.parent),
    )

    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.chmod(
            temp_path,
            original_stat.st_mode & 0o777,
        )

        if os.geteuid() == 0:
            os.chown(
                temp_path,
                original_stat.st_uid,
                original_stat.st_gid,
            )

        os.replace(temp_path, STATE)

        directory_fd = os.open(
            str(STATE.parent),
            os.O_RDONLY,
        )

        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print(
        "UNWIND VERIFIED — journal set to UNWOUND",
        flush=True,
    )
    print("Backup:", backup, flush=True)


async def check_once():
    lock_fd = os.open(
        LOCK,
        os.O_CREAT | os.O_RDWR
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )

    restart = False

    try:
        try:
            fcntl.flock(
                lock_fd,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            # The trading bot currently owns the journal.
            return False

        if not STATE.is_file():
            raise SafetyHold("Trading journal missing")

        original = STATE.read_bytes()
        data = json.loads(original)

        if (
            not isinstance(data, dict)
            or not isinstance(
                data.get("attempts"), list
            )
        ):
            raise SafetyHold("Invalid trading journal")

        attempts = data["attempts"]

        pending = [
            record for record in attempts
            if record.get("status") in WATCH_STATUSES
        ]

        if not pending:
            # Retry an interrupted systemd restart.
            if RESTART_MARKER.exists() and all(
                record.get("status") in SAFE_STATUSES
                for record in attempts
            ):
                restart = True
            else:
                return False

        other_unresolved = [
            record for record in attempts
            if record.get("status")
            not in SAFE_STATUSES | WATCH_STATUSES
        ]

        if other_unresolved:
            raise SafetyHold(
                "Other unresolved trades exist"
            )

        if pending:
            # The main trader quarantines pending GTC events.
            # Keep its daily startup timer enabled.
            api = Kalshi()

            try:
                updates = {}

                for record in pending:
                    snapshot = await exchange_snapshot(
                        api, record
                    )

                    verdict, detail = classify(
                        record, *snapshot
                    )

                    print(
                        f"WATCH {record['key']}: "
                        f"{verdict} {detail}",
                        flush=True,
                    )

                    if verdict == "READY":
                        updates[record["id"]] = {
                            "status": "UNWOUND",
                            "unwind_order_id": detail,
                            "unwind_filled": int(
                                record["qty"]
                            ),
                            "unwind_remaining": 0,
                            "auto_unwind_verified": True,
                            "auto_unwind_verified_at":
                                datetime.now(
                                    timezone.utc
                                ).isoformat(),
                        }

                if updates:
                    fully_resolved = (
                        len(updates) == len(pending)
                    )

                    # Persist restart intent first, so a
                    # crash after saving the journal cannot
                    # permanently lose the restart request.
                    if (
                        fully_resolved
                        and not RESTART_MARKER.exists()
                    ):
                        marker_fd = os.open(
                            RESTART_MARKER,
                            os.O_CREAT | os.O_EXCL
                            | os.O_WRONLY
                            | getattr(
                                os, "O_NOFOLLOW", 0
                            ),
                            0o600,
                        )
                        os.close(marker_fd)

                    save_resolved(
                        data, original, updates
                    )

                    restart = all(
                        record.get("status")
                        in SAFE_STATUSES
                        for record in attempts
                    )

            finally:
                await api.close()

    finally:
        os.close(lock_fd)

    if restart:
        print(
            "ALL EXPOSURES RESOLVED — "
            "restoring timer and restarting bot",
            flush=True,
        )

        subprocess.run(
            [
                "systemctl",
                "enable",
                "--now",
                "weather-bot.timer",
            ],
            check=True,
        )

        subprocess.run(
            [
                "systemctl",
                "start",
                "weather-bot.service",
            ],
            check=True,
        )

        RESTART_MARKER.unlink(
            missing_ok=True
        )

        print(
            "RESTART REQUESTED SUCCESSFULLY",
            flush=True,
        )

    return restart


def self_test():
    legs = ["FIRST", "SECOND"]

    def order(
        oid, ticker, cid, action, filled,
        remaining=0, status="executed",
    ):
        return {
            "order_id": oid,
            "ticker": ticker,
            "client_order_id": cid,
            "action": action,
            "side": "yes",
            "book_side": (
                "ask" if action == "sell"
                else "bid"
            ),
            "fill_count_fp": str(filled),
            "remaining_count_fp": str(remaining),
            "status": status,
        }

    def fills(oid, ticker, action, number):
        if number == 0:
            return []

        return [{
            "fill_id": "f-" + oid,
            "order_id": oid,
            "ticker": ticker,
            "action": action,
            "side": "yes",
            "count_fp": str(number),
        }]

    def scenario(
        exposed, sold, remaining, status,
        position=None,
    ):
        record = {
            "status": "UNWIND_RESTING",
            "key": "AAA:AAA-26OCT18",
            "legs": legs,
            "qty": 10,
            "pre_positions": {
                "FIRST": "0",
                "SECOND": "0",
            },
            "unwind_ticker": exposed,
            "unwind_client_id": "sell-cid",
            "unwind_order_id": "sale",
            "first_client_id": "cid-1",
            "second_client_id": "cid-2",
            "first_order_id": "buy-1",
            "second_order_id": "buy-2",
        }

        orders = {
            "FIRST": [
                order(
                    "buy-1", "FIRST", "cid-1",
                    "buy",
                    10 if exposed == "FIRST" else 0,
                    0,
                    "executed"
                    if exposed == "FIRST"
                    else "canceled",
                )
            ],
            "SECOND": [
                order(
                    "buy-2", "SECOND", "cid-2",
                    "buy",
                    10 if exposed == "SECOND" else 0,
                    0,
                    "executed"
                    if exposed == "SECOND"
                    else "canceled",
                )
            ],
        }

        orders[exposed].append(
            order(
                "sale", exposed, "sell-cid",
                "sell", sold, remaining, status,
            )
        )

        all_fills = {
            "buy-1": fills(
                "buy-1", "FIRST", "buy",
                10 if exposed == "FIRST" else 0,
            ),
            "buy-2": fills(
                "buy-2", "SECOND", "buy",
                10 if exposed == "SECOND" else 0,
            ),
            "sale": fills(
                "sale", exposed, "sell", sold,
            ),
        }

        positions = {
            exposed: (
                position if position is not None
                else 10 - sold
            )
        }

        return record, orders, all_fills, positions

    for exposed in legs:
        cases = (
            (0, 10, "resting", "PENDING"),
            (4, 6, "resting", "PENDING"),
            (10, 0, "executed", "READY"),
        )

        for sold, remaining, status, expected in cases:
            result = classify(
                *scenario(
                    exposed, sold, remaining, status
                )
            )

            assert result[0] == expected, result

        unsafe = (
            scenario(
                exposed, 10, 0,
                "executed", position=2,
            ),
            scenario(
                exposed, 4, 0, "canceled",
            ),
        )

        for test_case in unsafe:
            try:
                classify(*test_case)
            except SafetyHold:
                pass
            else:
                raise AssertionError(
                    "Unsafe recovery accepted"
                )

        mismatch = list(
            scenario(
                exposed, 10, 0, "executed"
            )
        )

        mismatch[2]["sale"][0]["count_fp"] = "9"

        try:
            classify(*mismatch)
        except SafetyHold:
            pass
        else:
            raise AssertionError(
                "Mismatched fill accepted"
            )

    print(
        "PASS: recovery watcher offline tests — "
        "both exposed legs, partial/full sells, "
        "position mismatch, canceled sell, "
        "and order/fill mismatch"
    )


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        try:
            asyncio.run(check_once())
        except Exception as exc:
            print(
                "WATCHER SAFETY HOLD:",
                type(exc).__name__,
                str(exc),
                flush=True,
            )
            sys.exit(1)
