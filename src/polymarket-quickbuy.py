#!/usr/bin/env python3
"""Polymarket US BTC 5-minute quickbuy: 45c limits on UP and DOWN.

Install: python -m pip install polymarket-us python-dotenv
.env:    POLYMARKET_KEY_ID=... and POLYMARKET_SECRET_KEY=...

Examples:
  python -u src/polymarket-quickbuy.py --dry-run
  python -u src/polymarket-quickbuy.py --5
  python -u src/polymarket-quickbuy.py --10 --loop

Standalone/manual runs intentionally allow another order pair in the same
market every time you launch the script. --loop submits only once per slot.

UTC in event slugs is intentional. 2026-11-17 7:55pm New York is
2026-11-18 00:55 UTC, so the event slug contains 2026-11-18.

US API uses one YES instrument: 45c UP = BUY_LONG at YES 0.45,
45c DOWN = BUY_SHORT at YES 0.55 (equivalent to buying NO at 0.45).

Orders are NOT atomic: one side can fill without the other. All orders are
standard GTD limits that may fill immediately at 45c or better or remain
resting until the 5-minute market expires. They are never FOK or IOC.
A returned order ID does NOT imply exchange acceptance. Verify both states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from zoneinfo import ZoneInfo

PRICE = Decimal("0.45")
ROOT = Path(__file__).resolve().parent.parent
LEDGER_FILE = ROOT / ".collector_runs" / "polymarket_quickbuy_us_attempts.json"
NEW_YORK = ZoneInfo("America/New_York")


def cli():
    parser = argparse.ArgumentParser(description="Polymarket US BTC 5m: standard 45c UP/DOWN limit buys")
    parser.add_argument("--shares", type=int, default=1, help="shares for EACH outcome (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="display orders without placing them")
    parser.add_argument("--loop", action="store_true", help="repeat for each new 5-minute market")
    parser.add_argument("--side", choices=("BOTH", "UP", "DOWN"), default="BOTH", help="submit both outcomes, or only a missing side")
    parser.add_argument("--check-orders", nargs="+", metavar="ORDER_ID", help="READ ONLY: inspect existing order IDs; never submit new orders")
    parser.add_argument("--verify-seconds", type=float, default=8, help="maximum seconds to poll exchange acceptance (default 8)")
    parser.add_argument("--cancel-unpaired", action="store_true", help="if one newly submitted order is definitely rejected, cancel other newly submitted resting order (not previous orders)")
    parser.add_argument("--poll", type=float, default=2, help="seconds between loop checks")
    parser.add_argument("--min-seconds-left", type=int, default=15, help="don't enter close to expiry")
    argv = []
    for s in sys.argv[1:]:
        match = re.fullmatch(r"--([1-9]\d*)", s)
        argv.extend(["--shares", match.group(1)] if match else [s])
    args = parser.parse_args(argv)
    if args.shares < 1 or args.poll <= 0 or args.verify_seconds < 0 or not 0 <= args.min_seconds_left < 300:
        parser.error("shares/poll must be positive, min-seconds-left must be 0..299")
    return args


def current_slot(now):
    now = now.astimezone(timezone.utc)
    epoch = int(now.timestamp()) // 300 * 300
    start = datetime.fromtimestamp(epoch, tz=timezone.utc)
    end = start + timedelta(minutes=5)
    return "btc-updown-5m-" + start.strftime("%Y-%m-%d-%H%Mz").lower(), end


def get_market(client, event_slug):
    # One reliable lookup: BTC 5m *event* slug. The tradable market has a
    # different slug (usually cpc-btc-...), read from the event response.
    print(f"[LOOKUP] Event {event_slug}", flush=True)
    response = client.events.retrieve_by_slug(event_slug)
    event = response.get("event", response) if isinstance(response, dict) else None
    if not isinstance(event, dict) or event.get("slug") != event_slug:
        raise RuntimeError("Event lookup returned the wrong event (no orders sent)")
    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) != 1 or not isinstance(markets[0], dict):
        raise RuntimeError("Event does not contain exactly one tradable market (no orders sent)")
    market = markets[0].get("market", markets[0])
    if not isinstance(market, dict) or not isinstance(market.get("slug"), str) or not market["slug"]:
        raise RuntimeError("Event market is missing its actual trading slug (no orders sent)")
    print(f"[MARKET] {market['slug']}", flush=True)
    return market


def check_market(market, shares, end, min_seconds_left):
    if (end - datetime.now(timezone.utc)).total_seconds() <= min_seconds_left:
        raise RuntimeError("Less than the minimum time remains; skipping this slot")
    if market.get("active") is False or market.get("closed") is True or market.get("archived") is True:
        raise RuntimeError("Market marked closed/inactive; no orders sent")
    if market.get("endDate"):
        actual_end = datetime.fromisoformat(str(market["endDate"]).replace("Z", "+00:00"))
        if actual_end.tzinfo and actual_end <= datetime.now(timezone.utc):
            raise RuntimeError("Market endDate has passed; no orders sent")
    minimum = market.get("minimumTradeQty")
    tick = market.get("orderPriceMinTickSize")
    if minimum is None or tick is None:
        raise RuntimeError("Missing market minimum size or tick metadata; no orders sent")
    minimum = Decimal(str(minimum))
    tick = Decimal(str(tick))
    if minimum <= 0 or tick <= 0 or Decimal(shares) < minimum:
        raise RuntimeError(f"Invalid size {shares}; exchange minimum is {minimum}")
    for px in (PRICE, Decimal("1") - PRICE):
        if px % tick != 0:
            raise RuntimeError(f"Price {px} is not a multiple of the market tick {tick}")
    # Safety check: UP must be the long/YES side if the API supplies side labels.
    for side in market.get("marketSides") or []:
        if isinstance(side, dict) and side.get("long") is True:
            description = str(side.get("description") or "").strip().lower()
            if description in ("down", "no", "bitcoin down", "btc down"):
                raise RuntimeError("YES/long appears to mean DOWN; refusing wrong-side orders")
    print(f"[RULES] minimum={minimum}, price_tick={tick}", flush=True)


def order_for(slug, outcome, shares, expiry):
    # API quotes YES prices even when opening a NO/short position.
    px = PRICE if outcome == "UP" else Decimal("1") - PRICE
    return {
        "marketSlug": slug,
        "intent": "ORDER_INTENT_BUY_LONG" if outcome == "UP" else "ORDER_INTENT_BUY_SHORT",
        "type": "ORDER_TYPE_LIMIT",
        "price": {"value": str(px), "currency": "USD"},
        "quantity": shares,
        "tif": "TIME_IN_FORCE_GOOD_TILL_DATE",
        "goodTillTime": expiry.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "participateDontInitiate": False,  # Standard resting-capable limit, NOT maker-only
        "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC",
    }


def ledger_load():
    if not LEDGER_FILE.exists():
        return {}
    record = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise RuntimeError("Attempt ledger is corrupt; fix it before live trading")
    return record


def ledger_write(ledger):
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = LEDGER_FILE.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(ledger, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, LEDGER_FILE)


def get_order_id(response):
    if not isinstance(response, dict):
        return None
    for obj in (response, response.get("order")):
        if isinstance(obj, dict):
            value = obj.get("id") or obj.get("orderId")
            if value:
                return str(value)
    return None


def classify_order(record):
    """Return (label, remaining, reason) from actual exchange order state."""
    if not isinstance(record, dict):
        return "UNKNOWN", None, "malformed exchange response"
    obj = record.get("order", record)
    if not isinstance(obj, dict):
        return "UNKNOWN", None, "missing order record"
    state = str(obj.get("state") or "UNKNOWN").upper()
    remaining = obj.get("leavesQuantity")
    try:
        remaining = Decimal(str(remaining)) if remaining is not None else None
    except Exception:
        remaining = None
    reason = obj.get("orderRejectReason") or obj.get("rejectReason") or obj.get("text") or ""
    if state == "ORDER_STATE_NEW":
        return ("RESTING" if remaining is None or remaining > 0 else "NO_REMAINING"), remaining, str(reason)
    if state == "ORDER_STATE_PARTIALLY_FILLED":
        return ("PARTIAL_RESTING" if remaining is None or remaining > 0 else "FILLED"), remaining, str(reason)
    mapping = {
        "ORDER_STATE_FILLED": "FILLED", "ORDER_STATE_REJECTED": "REJECTED",
        "ORDER_STATE_CANCELED": "CANCELED", "ORDER_STATE_EXPIRED": "EXPIRED",
        "ORDER_STATE_REPLACED": "REPLACED",
    }
    if state in mapping:
        return mapping[state], remaining, str(reason)
    if state.startswith("ORDER_STATE_PENDING"):
        return "PENDING", remaining, str(reason)
    return "UNKNOWN", remaining, (str(reason) or f"state={state}")


def execution_diagnostics(response):
    """Capture exchange execution codes from the original create response.

    Order GET frequently lacks the original rejection detail, so retain a
    narrow, non-sensitive subset in our local attempt ledger.
    """
    if not isinstance(response, dict):
        return []
    findings = []
    for item in response.get("executions", []) or []:
        if not isinstance(item, dict):
            continue
        fields = ("type", "text", "orderRejectReason", "transactTime", "unsolicitedCancelReason")
        row = {key: str(item[key]) for key in fields if item.get(key) is not None}
        if row:
            findings.append(row)
    return findings


def response_rejection(response):
    """Extract immediate synchronous execution reject details, when provided."""
    if not isinstance(response, dict):
        return None
    for execution in response.get("executions", []) or []:
        if isinstance(execution, dict) and "REJECT" in str(execution.get("type", "")).upper():
            return str(execution.get("orderRejectReason") or execution.get("text") or execution.get("type"))
    return None


def fetch_status(client, oid):
    """Read-only order status; never assume an ID means resting."""
    try:
        response = client.orders.retrieve(oid)
        label, left, reason = classify_order(response)
        return {"status": label, "remaining": str(left) if left is not None else None,
                "reason": reason, "error": None}
    except Exception as exc:
        return {"status": "UNKNOWN", "remaining": None, "reason": "",
                "error": f"{type(exc).__name__}: {exc}"}


def report_status(outcome, oid, snapshot):
    rem = snapshot.get("remaining")
    extra = f" remaining={rem}" if rem is not None else ""
    info = snapshot.get("reason") or snapshot.get("error") or ""
    if info:
        info = f" detail={info}"
    print(f"[STATUS] {outcome}: {snapshot['status']} order_id={oid}{extra}{info}", flush=True)


def verify_orders(client, by_side, seconds):
    """Poll short-lived REST states; unknown/pending NEVER counted as accepted."""
    end = time.monotonic() + seconds
    while True:
        status = {side: fetch_status(client, oid) for side, oid in by_side.items()}
        # Pending order or missing/propagating GET may settle momentarily.
        uncertain = any(v["status"] in ("UNKNOWN", "PENDING") for v in status.values())
        if not uncertain or time.monotonic() >= end:
            break
        time.sleep(min(1.0, max(0.0, end - time.monotonic())))
    for side, oid in by_side.items():
        report_status(side, oid, status[side])
    return status


def check_existing_orders(client, order_ids, seconds):
    print("[READ ONLY] Checking existing order IDs; NO trades or cancellations", flush=True)
    ids = {f"ORDER-{i + 1}": oid for i, oid in enumerate(order_ids)}
    results = verify_orders(client, ids, seconds)
    if any(v["status"] == "UNKNOWN" for v in results.values()):
        print("[UNCONFIRMED] One or more orders could not be inspected. Check account order history.", flush=True)
        return 1
    return 0


def post_pair(client, ledger_key, trade_slug, orders, shares, ledger, *, verify_seconds=8, cancel_unpaired=False):
    # Persist BEFORE the first non-idempotent API call. If a timeout occurs,
    # do NOT submit another pair automatically: the server may have accepted it.
    ledger[ledger_key] = {"status": "submission_started", "market_slug": trade_slug,
                          "shares": shares, "started": datetime.now(timezone.utc).isoformat()}
    ledger_write(ledger)
    by_side = {}
    immediate_rejects = {}
    try:
        for outcome, payload in orders:
            print(f"[SEND] {outcome} 45c x {shares} (GTD limit, {'post-only' if payload['participateDontInitiate'] else 'standard'})", flush=True)
            # IMPORTANT: never retry a POST automatically if timeout / uncertain result.
            response = client.orders.create(payload)
            oid = get_order_id(response)
            if oid is None:
                raise RuntimeError(f"{outcome} API response gave no order ID: {response!r}; inspect your US orders")
            by_side[outcome] = oid
            ledger[ledger_key][outcome] = oid
            diagnostics = execution_diagnostics(response)
            if diagnostics:
                ledger[ledger_key][f"{outcome}_execution_diagnostics"] = diagnostics
            ledger_write(ledger)
            print(f"[ACK] {outcome}: order_id={oid} (NOT YET exchange-confirmed)", flush=True)
            for detail in diagnostics:
                if "REJECT" in detail.get("type", "").upper():
                    print(f"[EXCHANGE REJECTION] {outcome}: {detail}", flush=True)
            reason = response_rejection(response)
            if reason:
                immediate_rejects[outcome] = reason
                print(f"[REJECT] {outcome}: {reason}", flush=True)
    except Exception:
        if by_side:
            print("[WARN] Submission failed/uncertain. Already acknowledged orders may remain live.", flush=True)
            print("[WARN] No automatic retry or cancellation: a first leg might have filled.", flush=True)
        ledger[ledger_key]["status"] = "submission_error_check_us_account_orders"
        ledger_write(ledger)
        raise

    final = verify_orders(client, by_side, verify_seconds)
    for side, reason in immediate_rejects.items():
        # Synchronous rejection is authoritative even if GET is temporarily unavailable.
        if final[side]["status"] in ("PENDING", "UNKNOWN"):
            final[side] = {"status": "REJECTED", "remaining": None, "reason": reason, "error": None}
            print(f"[STATUS] {side}: REJECTED from create response: {reason}", flush=True)

    good = {"RESTING", "PARTIAL_RESTING"}
    if all(v["status"] in good for v in final.values()):
        if len(final) == 2:
            print("[VERIFIED] Both limit orders currently have resting quantity.", flush=True)
            ledger[ledger_key]["status"] = "both_resting_at_verification"
        else:
            print("[VERIFIED] Selected side currently has resting quantity.", flush=True)
            ledger[ledger_key]["status"] = "single_side_resting_at_verification"
    else:
        labels = {side: info["status"] for side, info in final.items()}
        print(f"[WARN] RESTING STATUS NOT CONFIRMED: {labels}", flush=True)
        ledger[ledger_key]["status"] = "unpaired_or_unconfirmed_check_orders"
        if len(final) == 2:
            rejected = [s for s, v in final.items() if v["status"] in ("REJECTED", "CANCELED", "EXPIRED")]
            resting = [s for s, v in final.items() if v["status"] in good]
            if rejected and resting:
                print("[WARN] Only one side appears active. Do not repeat BOTH without considering duplicate exposure.", flush=True)
                if cancel_unpaired:
                    for side in resting:
                        oid = by_side[side]
                        print(f"[CANCEL REQUEST] {side} order={oid} (only this run's order)", flush=True)
                        try:
                            client.orders.cancel(oid, {"marketSlug": trade_slug})
                            result = verify_orders(client, {side: oid}, min(verify_seconds, 5))
                            print(f"[CANCEL CHECK] {side}: {result[side]['status']} (a prior fill cannot be undone)", flush=True)
                        except Exception as exc:
                            print(f"[WARN] Cancel request failed: {exc}", flush=True)
                else:
                    print("[HINT] --cancel-unpaired requests cancellation of this run's lone resting order.", flush=True)
    ledger[ledger_key]["verified"] = final
    ledger_write(ledger)
    print("[DONE] Order IDs and last observed states recorded; fills may change after this check.", flush=True)


def run_slot(client, args, ledger, done):
    event_slug, end = current_slot(datetime.now(timezone.utc))
    if event_slug in done or (args.loop and not args.dry_run and event_slug in ledger):
        done.add(event_slug)
        return
    local_end = end.astimezone(NEW_YORK)
    print(f"[SLOT] {event_slug} | ends {local_end:%b %d %I:%M %p %Z} ({end:%Y-%m-%d %H:%M} UTC)", flush=True)
    market = get_market(client, event_slug)
    check_market(market, args.shares, end, args.min_seconds_left)
    trade_slug = market["slug"]
    sides = ("UP", "DOWN") if args.side == "BOTH" else (args.side,)
    orders = [(side, order_for(trade_slug, side, args.shares, end)) for side in sides]
    print(f"[ORDERS] {args.side}: UP 45c (YES API .45) | DOWN 45c (YES API .55) | {args.shares} shares per selected side", flush=True)
    print(f"[COST] Max planned spend ${len(sides) * PRICE * args.shares:.2f} before fees", flush=True)
    print("[STANDARD LIMIT] Post-only OFF by default: can fill immediately at 45c or better; unfilled quantity rests until expiry. Not FOK/IOC.", flush=True)
    if args.dry_run:
        print("[DRY RUN] Validated event and market. NO orders submitted.", flush=True)
        done.add(event_slug)
        return
    if datetime.now(timezone.utc) >= end - timedelta(seconds=args.min_seconds_left):
        raise RuntimeError("Too close to close after lookup; no orders submitted")
    done.add(event_slug)
    # For --loop, a stable per-market key prevents repeated polling submissions.
    # For a manual run, create a unique audit key; never overwrite earlier order IDs.
    ledger_key = (event_slug if args.loop else f"{event_slug}#{uuid.uuid4().hex}")
    if not args.loop:
        print("[MANUAL] Fresh order request; prior manual orders will NOT be cancelled.", flush=True)
    post_pair(client, ledger_key, trade_slug, orders, args.shares, ledger, verify_seconds=args.verify_seconds, cancel_unpaired=args.cancel_unpaired)


def main():
    args = cli()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    try:
        from polymarket_us import PolymarketUS
    except ImportError as exc:
        raise RuntimeError("Install: python -m pip install polymarket-us python-dotenv") from exc
    mode = "READ ONLY" if args.check_orders else ("DRY RUN" if args.dry_run else "LIVE")
    print(f"[START] Polymarket US BTC 5m | {mode} | {args.shares} shares/side | loop={args.loop}", flush=True)
    print(f"[TIME] {datetime.now(NEW_YORK):%Y-%m-%d %I:%M:%S %p %Z}; market slugs use UTC", flush=True)
    if args.dry_run and not args.check_orders:
        client = PolymarketUS(timeout=8, max_retries=0)
        ledger = {}
    else:
        kid, secret = os.getenv("POLYMARKET_KEY_ID"), os.getenv("POLYMARKET_SECRET_KEY")
        if not kid or not secret:
            raise RuntimeError("Add POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY to .env")
        client = PolymarketUS(key_id=kid, secret_key=secret, timeout=8, max_retries=0)
        ledger = ledger_load()
    if args.check_orders:
        try:
            return check_existing_orders(client, args.check_orders, args.verify_seconds)
        finally:
            client.close()
    done = set()
    try:
        while True:
            slot, _ = current_slot(datetime.now(timezone.utc))
            if slot not in done and (not args.loop or args.dry_run or slot not in ledger):
                try:
                    run_slot(client, args, ledger, done)
                except Exception as exc:
                    print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    if not args.loop:
                        return 1
            elif not args.loop:
                print("[SKIP] This market already attempted in this run.", flush=True)
                return 0
            if not args.loop:
                return 0
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\n[STOPPED] Previous live orders may still be resting.", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"[FATAL] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
