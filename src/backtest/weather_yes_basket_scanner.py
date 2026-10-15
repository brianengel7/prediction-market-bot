"""Read-only event-driven Kalshi NYC daily high top-two YES scanner."""

import argparse
import asyncio
import base64
import inspect
import json
import os
import time
from datetime import datetime
from decimal import Decimal as D
from pathlib import Path
from zoneinfo import ZoneInfo

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

from src.kalshi.client import get_event_with_markets
from src.kalshi.fees import calculate_taker_fee

NY = ZoneInfo("America/New_York")
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
CAP = D("0.97")
ONE = D("1")


def clock():
    return datetime.now(NY).strftime("%H:%M:%S.%f")[:-3] + " ET"


def auth_headers():
    key_id = os.environ["KALSHI_API_KEY_ID"]
    key_path = Path(
        os.environ["KALSHI_PRIVATE_KEY_PATH"]
    ).expanduser()

    if not key_path.is_absolute():
        key_path = Path.cwd() / key_path

    key = serialization.load_pem_private_key(
        key_path.read_bytes(), password=None
    )

    stamp = str(int(time.time() * 1000))
    message = (stamp + "GET" + "/trade-api/ws/v2").encode()

    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(message)
    else:
        sig = key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": stamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
    }


def today_event():
    day = datetime.now(NY).date()
    ticker = f'KXHIGHNY-{day.strftime("%y%b%d").upper()}'

    data = get_event_with_markets(ticker)
    markets = data.get("markets") or []

    names = [
        m["ticker"] for m in markets
        if str(m.get("status", "")).lower() in ("open", "active")
    ]

    if (
        len(names) != 6
        or len(set(names)) != 6
        or any(not s.startswith(ticker + "-") for s in names)
    ):
        raise RuntimeError(
            f"{ticker}: expected six unique active "
            f"outcome markets; found {len(names)}"
        )

    return day, ticker, sorted(names)


class Book:
    def __init__(self):
        self.levels = {"yes": {}, "no": {}}
        self.ready = False

    def snapshot(self, msg):
        self.levels = {"yes": {}, "no": {}}

        # A missing side is a valid empty order book.
        if not any(
            key in msg
            for key in ("yes_dollars_fp", "no_dollars_fp")
        ):
            print(
                f"[{clock()}] EMPTY SNAPSHOT "
                f"{msg.get('market_ticker')} "
                f"fields={sorted(msg.keys())}",
                flush=True,
            )

        for side in ("yes", "no"):
            levels = msg.get(side + "_dollars_fp") or []

            if not isinstance(levels, list):
                raise ValueError(
                    f"Invalid {side} orderbook level format"
                )

            for p, q in levels:
                p, q = D(str(p)), D(str(q))

                if not D(0) <= p <= ONE or q < 0:
                    raise ValueError("Invalid snapshot level")

                if q > 0:
                    self.levels[side][p] = q

        self.ready = True

    def delta(self, msg):
        if not self.ready:
            raise RuntimeError("Delta before initial snapshot")

        side = msg["side"].lower()

        if side not in self.levels:
            raise ValueError("Invalid orderbook side")

        p = D(str(msg["price_dollars"]))
        dq = D(str(msg["delta_fp"]))

        if not D(0) <= p <= ONE:
            raise ValueError("Invalid delta price")

        qty = self.levels[side].get(p, D(0)) + dq

        if qty < 0:
            raise RuntimeError("Negative depth: resync required")

        if qty == 0:
            self.levels[side].pop(p, None)
        else:
            self.levels[side][p] = qty

    def best(self, side):
        levels = self.levels[side]
        return max(levels.items()) if levels else (None, D(0))

    def quote(self, ticker):
        bid, _ = self.best("yes")
        no_bid, no_qty = self.best("no")

        if bid is None and no_bid is None:
            return None

        # YES asks are implied by the opposite NO bids.
        yes_bid = bid if bid is not None else D(0)
        yes_ask = ONE - no_bid if no_bid is not None else ONE

        if yes_bid > yes_ask:
            raise RuntimeError(f"Crossed orderbook: {ticker}")

        return {
            "ticker": ticker,
            "bid": yes_bid,
            "ask": yes_ask,
            "mid": (yes_bid + yes_ask) / 2,
            "ask_qty": no_qty if no_bid is not None else D(0),
            "has_ask": no_bid is not None,
        }


def scan(books, tickers):
    if not all(books[t].ready for t in tickers):
        return None

    quotes = [books[t].quote(t) for t in tickers]

    if any(q is None for q in quotes):
        return None

    quotes.sort(key=lambda q: (-q["mid"], q["ticker"]))

    a, b = quotes[:2]
    cost = a["ask"] + b["ask"]

    liquid = all(
        q["has_ask"] and q["ask_qty"] >= ONE
        for q in (a, b)
    )

    total_mid = sum(q["mid"] for q in quotes)

    for q in (a, b):
        q["weight"] = (
            q["mid"] / total_mid if total_mid > 0 else D(0)
        )

    return a, b, cost, liquid


def display_signal(event, result):
    a, b, cost, _ = result

    fee = sum(
        (
            D(str(calculate_taker_fee(
                price=float(q["ask"]),
                contracts=1,
                multiplier=1,
            )))
            for q in (a, b)
        ),
        D(0),
    )

    all_in = cost + fee

    print("\n" + "=" * 64)
    print(f">>> TOP-TWO YES SIGNAL [{clock()}] {event}")

    for i, q in enumerate((a, b), 1):
        print(
            f'{i}. {q["ticker"]}  '
            f'mid-weight={q["weight"]:.1%}  '
            f'YES ask={q["ask"]*100:.2f}c  '
            f'ask-depth={q["ask_qty"]}'
        )

    print(f"Combined YES asks: {cost*100:.2f}c < 95c")
    print(
        f"Estimated fees: {fee*100:.2f}c  |  "
        f"All-in: {all_in*100:.2f}c"
    )
    print(
        f"If either wins: ${ONE-all_in:+.4f} | "
        f"If neither wins: ${-all_in:+.4f}"
    )
    print("READ ONLY: no orders sent")
    print("=" * 64, flush=True)


async def stream_once(day, event, tickers, heartbeat):
    books = {t: Book() for t in tickers}
    last_seq = None
    last_signature = None
    last_beat = time.monotonic()
    update_count = 0
    last_summary = None

    kwargs = dict(
        ping_interval=20,
        ping_timeout=20,
        max_queue=4096,
    )

    headers = auth_headers()

    header_arg = (
        "additional_headers"
        if "additional_headers" in
        inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )

    ws_context = websockets.connect(
        WS_URL, **{header_arg: headers}, **kwargs
    )

    async with ws_context as ws:
        await ws.send(json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
                "use_yes_price": False,
            },
        }))

        print(
            f"[{clock()}] WS connected; subscribed "
            f"to {len(tickers)} books for {event}",
            flush=True,
        )

        while True:
            if datetime.now(NY).date() != day:
                print("New NYC date: switching events", flush=True)
                return

            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=1.0
                )
            except asyncio.TimeoutError:
                raw = None

            if raw is not None:
                packet = json.loads(raw)
                kind = packet.get("type")

                if kind == "error":
                    raise RuntimeError(
                        f'Kalshi subscription error: {packet.get("msg")}'
                    )

                if kind in ("orderbook_snapshot", "orderbook_delta"):
                    seq = packet.get("seq")

                    if seq is not None:
                        seq = int(seq)

                        if (
                            last_seq is not None
                            and seq != last_seq + 1
                        ):
                            raise RuntimeError(
                                f"Orderbook sequence gap: "
                                f"{last_seq} -> {seq}"
                            )

                        last_seq = seq

                    m = packet["msg"]
                    ticker = m["market_ticker"]

                    if ticker not in books:
                        raise RuntimeError(
                            f"Unexpected ticker: {ticker}"
                        )

                    if kind == "orderbook_snapshot":
                        books[ticker].snapshot(m)
                    else:
                        books[ticker].delta(m)

                    update_count += 1

                    # Evaluate immediately after every book update.
                    last_summary = scan(books, tickers)

                    if last_summary:
                        a, b, cost, liquid = last_summary
                        qualifies = cost < CAP and liquid

                        signature = (
                            a["ticker"], a["ask"],
                            b["ticker"], b["ask"],
                        )

                        if qualifies and signature != last_signature:
                            display_signal(event, last_summary)

                        last_signature = (
                            signature if qualifies else None
                        )

                elif kind == "subscribed":
                    print(
                        f"[{clock()}] Subscription acknowledged",
                        flush=True,
                    )

            # Heartbeat only; this does not poll the API.
            if time.monotonic() - last_beat >= heartbeat:
                ready = sum(b.ready for b in books.values())

                if last_summary:
                    a, b, cost, liquid = last_summary

                    status = (
                        "SIGNAL"
                        if cost < CAP and liquid
                        else "NO DEPTH"
                        if cost < CAP
                        else "PASS"
                    )

                    msg = (
                        f'{cost*100:.2f}c '
                        f'({a["ticker"].split("-")[-1]} + '
                        f'{b["ticker"].split("-")[-1]}) '
                        f'{status}'
                    )
                else:
                    msg = "waiting for complete market snapshots/quotes"

                print(
                    f"[{clock()}] books={ready}/{len(tickers)} "
                    f"updates={update_count} | {msg}",
                    flush=True,
                )

                last_beat = time.monotonic()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", type=float, default=5)
    args = parser.parse_args()

    if args.heartbeat < 1:
        parser.error("--heartbeat must be >= 1 second")

    print("NYC KXHIGHNY HIGH-FREQUENCY TOP-TWO YES SCANNER")
    print("Live WebSocket orderbook; evaluates every book update")
    print("YES basket threshold: strictly below 95c before fees")
    print("READ ONLY: no orders or database writes. Ctrl+C to stop.")

    delay = 1.0

    while True:
        try:
            day, event, tickers = today_event()

            print(
                f'[{clock()}] Watching {event}: '
                f'{", ".join(tickers)}',
                flush=True,
            )

            await stream_once(
                day, event, tickers, args.heartbeat
            )

            delay = 1.0

        except (KeyboardInterrupt, asyncio.CancelledError):
            raise

        except Exception as exc:
            print(
                f"[{clock()}] CONNECTION/SCAN ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print(
                f"Reconnecting in {delay:g}s with fresh books...",
                flush=True,
            )

            await asyncio.sleep(delay)
            delay = min(delay * 2, 20)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScanner stopped.")
