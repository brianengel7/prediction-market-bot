"""Read-only Coinbase vs Kalshi BTC perpetual latency study."""

import argparse
import asyncio
import base64
import inspect
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, ed25519
from dotenv import load_dotenv


COINBASE_WS = "wss://advanced-trade-ws.coinbase.com"
KALSHI_PATH = "/trade-api/ws/v2/margin"
KALSHI_WS = (
    "wss://external-api-margin-ws.kalshi.com" + KALSHI_PATH
)
MARKET = "KXBTCPERP"
CONTRACT_BTC = 0.0001


def now_ms():
    return time.time_ns() / 1_000_000


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def credentials():
    load_dotenv(Path.cwd() / ".env")

    key_id = next((
        os.getenv(k) for k in (
            "KALSHI_API_KEY_ID",
            "KALSHI_KEY_ID",
            "KALSHI_API_KEY",
        ) if os.getenv(k)
    ), None)

    secret = next((
        os.getenv(k) for k in (
            "KALSHI_PRIVATE_KEY_PATH",
            "KALSHI_API_PRIVATE_KEY_PATH",
            "KALSHI_PRIVATE_KEY_FILE",
            "KALSHI_PRIVATE_KEY_PEM",
            "KALSHI_PRIVATE_KEY",
        ) if os.getenv(k)
    ), None)

    if not key_id or not secret:
        raise RuntimeError(
            "Missing Kalshi API key or private key. "
            "Check your .env variable names."
        )

    if "-----BEGIN " in secret:
        pem = secret.replace("\\n", "\n").encode()
    else:
        pem = Path(secret).expanduser().read_bytes()

    key = serialization.load_pem_private_key(
        pem, password=None
    )
    return key_id, key


def auth_headers(key_id, key):
    ts = str(int(now_ms()))
    message = (ts + "GET" + KALSHI_PATH).encode()

    if isinstance(key, ed25519.Ed25519PrivateKey):
        signature = key.sign(message)
    else:
        signature = key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(
            signature
        ).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).timestamp() * 1000
    except (ValueError, TypeError):
        return None


class Collector:
    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

        self.events = (
            folder / "quotes.jsonl"
        ).open("w", buffering=1024 * 1024)

        self.samples = (
            folder / "seconds.jsonl"
        ).open("w", buffering=1024 * 1024)

        self.books = {
            "coinbase": {"bids": {}, "asks": {}},
            "kalshi": {"bids": {}, "asks": {}},
        }

        self.states = {
            name: {
                "connected": False,
                "snapshot": False,
                "bid": None,
                "ask": None,
                "last_book_ms": None,
                "last_msg_ms": None,
                "messages": 0,
                "changes": 0,
            }
            for name in ("coinbase", "kalshi")
        }

        self.reference = None
        self.reference_ts = None
        self.rows = []
        self.kalshi_seq = {}
        self.coinbase_seq = None

    def log(self, record):
        self.events.write(
            json.dumps(record, separators=(",", ":")) + "\n"
        )

    def connection(self, name, connected, error=None):
        state = self.states[name]
        state["connected"] = connected
        state["snapshot"] = False
        state["bid"] = None
        state["ask"] = None
        state["last_book_ms"] = None

        self.books[name] = {"bids": {}, "asks": {}}

        if name == "coinbase":
            self.coinbase_seq = None

        if name == "kalshi":
            self.kalshi_seq.clear()
            self.reference = None
            self.reference_ts = None

        message = str(error)[:200] if error else None

        self.log({
            "kind": "connection",
            "source": name,
            "recv_ms": now_ms(),
            "connected": connected,
            "error": message,
        })

        status = "CONNECTED" if connected else "DISCONNECTED"
        print(f"[{name}] {status} {message or ''}", flush=True)

    def update_top(self, name, received, source_ms=None):
        book = self.books[name]
        state = self.states[name]

        bid = max(book["bids"], default=None)
        ask = min(book["asks"], default=None)

        if bid is None or ask is None or bid >= ask:
            state["bid"] = None
            state["ask"] = None
            return

        if name == "kalshi":
            bid /= CONTRACT_BTC
            ask /= CONTRACT_BTC

        old = (state["bid"], state["ask"])
        state["bid"] = bid
        state["ask"] = ask

        if old != (bid, ask):
            state["changes"] += 1

            self.log({
                "kind": "quote",
                "source": name,
                "recv_ms": received,
                "source_ms": source_ms,
                "bid_btc_usd": bid,
                "ask_btc_usd": ask,
                "mid_btc_usd": (bid + ask) / 2,
            })

    def coinbase_message(self, data, received):
        state = self.states["coinbase"]

        state["messages"] += 1
        state["last_msg_ms"] = received

        if (
            data.get("type") == "error"
            or data.get("channel") == "error"
        ):
            raise RuntimeError(
                "Coinbase error: " + str(data)[:500]
            )

        # Advanced Trade sequences are connection-wide.
        seq = data.get("sequence_num")

        if isinstance(seq, int):
            previous = self.coinbase_seq

            if (
                previous is not None
                and seq != previous + 1
            ):
                raise RuntimeError(
                    f"Coinbase sequence gap: "
                    f"{previous} -> {seq}"
                )

            self.coinbase_seq = seq

        if data.get("channel") != "l2_data":
            return

        events = data.get("events", [])

        if not isinstance(events, list):
            raise RuntimeError(
                "Unexpected Coinbase events format"
            )

        for event in events:
            if event.get("product_id") != "BTC-USD":
                continue

            event_type = event.get("type")

            if event_type == "snapshot":
                self.books["coinbase"] = {
                    "bids": {},
                    "asks": {},
                }
                state["snapshot"] = True

            elif event_type == "update":
                if not state["snapshot"]:
                    continue

            else:
                continue

            book = self.books["coinbase"]
            updates = event.get("updates", [])

            if not isinstance(updates, list):
                raise RuntimeError(
                    "Unexpected Coinbase updates format"
                )

            source_ms = parse_time(
                data.get("timestamp")
            )

            for update in updates:
                side = {
                    "bid": "bids",
                    "buy": "bids",
                    "offer": "asks",
                    "ask": "asks",
                    "sell": "asks",
                }.get(update.get("side"))

                price = num(
                    update.get("price_level")
                )
                quantity = num(
                    update.get("new_quantity")
                )

                if (
                    side is None
                    or price is None
                    or quantity is None
                    or price <= 0
                    or quantity < 0
                ):
                    raise RuntimeError(
                        "Invalid Coinbase book update: "
                        + str(update)[:400]
                    )

                # Advanced Trade sends absolute quantity,
                # NOT a change in quantity.
                if quantity == 0:
                    book[side].pop(price, None)
                else:
                    book[side][price] = quantity

                event_ms = parse_time(
                    update.get("event_time")
                )

                if event_ms is not None:
                    source_ms = event_ms

            state["last_book_ms"] = received

            self.update_top(
                "coinbase",
                received,
                source_ms,
            )

    def kalshi_message(self, data, received):
        typ = data.get("type")
        state = self.states["kalshi"]
        msg = data.get("msg") or {}

        state["messages"] += 1
        state["last_msg_ms"] = received

        if typ == "error":
            raise RuntimeError(str(data)[:400])

        if typ in ("orderbook_snapshot", "orderbook_delta"):
            if msg.get("market_ticker") not in (None, MARKET):
                return

            sid = data.get("sid")
            seq = data.get("seq")

            if sid is not None and seq is not None:
                previous = self.kalshi_seq.get(sid)

                if (
                    typ == "orderbook_delta"
                    and previous is not None
                    and seq != previous + 1
                ):
                    raise RuntimeError(
                        f"Kalshi orderbook sequence gap: "
                        f"{previous} -> {seq}"
                    )

                self.kalshi_seq[sid] = seq

        if typ == "orderbook_snapshot":
            raw = msg.get("orderbook", msg)

            if not isinstance(raw, dict) or not any(
                key in raw for key in (
                    "bid", "ask", "bids", "asks"
                )
            ):
                raise RuntimeError(
                    "Unknown Kalshi snapshot: "
                    + json.dumps(data)[:700]
                )

            self.books["kalshi"] = {
                "bids": {
                    float(p): float(q)
                    for p, q in raw.get(
                        "bid", raw.get("bids", [])
                    )
                    if float(q) > 0
                },
                "asks": {
                    float(p): float(q)
                    for p, q in raw.get(
                        "ask", raw.get("asks", [])
                    )
                    if float(q) > 0
                },
            }

            state["snapshot"] = True
            state["last_book_ms"] = received

            self.update_top(
                "kalshi",
                received,
                msg.get("ts_ms")
                or data.get("sending_ts_ms"),
            )

        elif typ == "orderbook_delta" and state["snapshot"]:
            side = {
                "bid": "bids",
                "ask": "asks",
                "buy": "bids",
                "sell": "asks",
                "bids": "bids",
                "asks": "asks",
            }.get(msg.get("side"))

            price = num(
                msg.get("price", msg.get("price_dollars"))
            )
            delta = num(
                msg.get("delta", msg.get("delta_fp"))
            )

            if side is None or price is None or delta is None:
                raise RuntimeError(
                    "Unknown Kalshi delta: "
                    + json.dumps(data)[:700]
                )

            levels = self.books["kalshi"][side]
            quantity = levels.get(price, 0.0) + delta

            if quantity < -0.00001:
                raise RuntimeError(
                    "Negative Kalshi book quantity; resync needed"
                )

            if quantity <= 0.00001:
                levels.pop(price, None)
            else:
                levels[price] = quantity

            state["last_book_ms"] = received

            self.update_top(
                "kalshi",
                received,
                msg.get("ts_ms")
                or data.get("sending_ts_ms"),
            )

        elif typ == "ticker":
            if msg.get("market_ticker") not in (MARKET, None):
                return

            reference = msg.get("reference_price")

            if isinstance(reference, dict):
                price = num(reference.get("price"))

                if price is not None:
                    normalized = price / CONTRACT_BTC

                    if normalized != self.reference:
                        self.log({
                            "kind": "reference",
                            "source": "kalshi",
                            "recv_ms": received,
                            "source_ms": reference.get("ts_ms"),
                            "price_btc_usd": normalized,
                        })

                    self.reference = normalized
                    self.reference_ts = reference.get("ts_ms")

    def sample(self):
        received = now_ms()
        utc = datetime.fromtimestamp(
            received / 1000, timezone.utc
        ).isoformat()

        row = {
            "kind": "sample",
            "utc": utc,
            "recv_ms": received,
        }

        for name, prefix in (
            ("coinbase", "cb"),
            ("kalshi", "k"),
        ):
            state = self.states[name]

            valid = (
                state["connected"]
                and state["snapshot"]
                and state["bid"] is not None
                and state["ask"] is not None
            )

            bid = state["bid"] if valid else None
            ask = state["ask"] if valid else None

            row[prefix + "_bid"] = bid
            row[prefix + "_ask"] = ask
            row[prefix + "_mid"] = (
                (bid + ask) / 2 if valid else None
            )

            last_book = state["last_book_ms"]
            last_msg = state["last_msg_ms"]

            row[prefix + "_book_age_ms"] = (
                round(received - last_book, 1)
                if last_book else None
            )
            row[prefix + "_msg_age_ms"] = (
                round(received - last_msg, 1)
                if last_msg else None
            )

        row["kalshi_reference_btc_usd"] = self.reference
        row["kalshi_ref_source_age_ms"] = (
            round(received - self.reference_ts, 1)
            if self.reference_ts else None
        )

        if (
            row["cb_mid"] is not None
            and row["k_mid"] is not None
        ):
            row["basis_usd"] = round(
                row["k_mid"] - row["cb_mid"], 2
            )

        self.samples.write(
            json.dumps(row, separators=(",", ":")) + "\n"
        )
        self.rows.append(row)

        def fmt(value):
            return (
                f"{value:,.2f}"
                if value is not None else "WAIT"
            )

        print(
            f"{utc[11:19]}  "
            f"CB {fmt(row['cb_mid'])}  "
            f"K {fmt(row['k_mid'])}  "
            f"K-CB {fmt(row.get('basis_usd'))}  "
            f"REF {fmt(self.reference)}  "
            f"age_ms CB={fmt(row['cb_book_age_ms'])} "
            f"K={fmt(row['k_book_age_ms'])}",
            flush=True,
        )

    def close(self):
        self.events.close()
        self.samples.close()


async def coinbase_stream(monitor, end):
    backoff = 1

    while time.monotonic() < end:
        try:
            async with websockets.connect(
                COINBASE_WS,
                ping_interval=20,
                ping_timeout=20,
                max_size=32 * 1024 * 1024,
                max_queue=64,
            ) as ws:
                monitor.connection("coinbase", True)

                await ws.send(json.dumps({
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channel": "level2",
                }))

                await ws.send(json.dumps({
                    "type": "subscribe",
                    "channel": "heartbeats",
                }))

                backoff = 1

                async for raw in ws:
                    received = now_ms()
                    monitor.coinbase_message(
                        json.loads(raw), received
                    )

                    if time.monotonic() >= end:
                        break

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            monitor.connection("coinbase", False, exc)
            await asyncio.sleep(
                min(backoff, max(0, end - time.monotonic()))
            )
            backoff = min(20, backoff * 2)
        else:
            monitor.connection("coinbase", False)


async def kalshi_stream(monitor, end, key_id, key):
    backoff = 1

    while time.monotonic() < end:
        try:
            header_arg = (
                "additional_headers"
                if "additional_headers" in
                inspect.signature(websockets.connect).parameters
                else "extra_headers"
            )

            async with websockets.connect(
                KALSHI_WS,
                **{header_arg: auth_headers(key_id, key)},
                ping_interval=20,
                ping_timeout=20,
                max_queue=2048,
            ) as ws:
                monitor.connection("kalshi", True)

                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": [
                            "orderbook_delta",
                            "ticker",
                        ],
                        "market_tickers": [MARKET],
                    },
                }))

                backoff = 1

                async for raw in ws:
                    received = now_ms()
                    monitor.kalshi_message(
                        json.loads(raw), received
                    )

                    if time.monotonic() >= end:
                        break

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            monitor.connection("kalshi", False, exc)
            await asyncio.sleep(
                min(backoff, max(0, end - time.monotonic()))
            )
            backoff = min(20, backoff * 2)
        else:
            monitor.connection("kalshi", False)


async def sampler(monitor, end):
    next_tick = math.floor(time.monotonic()) + 1.0

    while time.monotonic() < end:
        await asyncio.sleep(
            max(0, next_tick - time.monotonic())
        )
        monitor.sample()
        next_tick += 1.0


def correlation(x, y):
    if (
        len(x) < 15
        or statistics.pstdev(x) < 1e-12
        or statistics.pstdev(y) < 1e-12
    ):
        return None

    mx = statistics.mean(x)
    my = statistics.mean(y)

    numerator = sum(
        (a - mx) * (b - my)
        for a, b in zip(x, y)
    )

    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x)
        * sum((b - my) ** 2 for b in y)
    )

    return numerator / denominator


def report(monitor):
    print("\n=== COINBASE / KALSHI LAG ANALYSIS ===")

    for name, state in monitor.states.items():
        print(
            f"{name}: {state['messages']} messages, "
            f"{state['changes']} best-quote changes"
        )

    rows = monitor.rows

    for lag in range(-3, 6):
        coin_returns = []
        kalshi_returns = []

        for t in range(1, len(rows)):
            u = t + lag

            if not (0 < u < len(rows)):
                continue

            a0, a1 = rows[t - 1], rows[t]
            b0, b1 = rows[u - 1], rows[u]

            values = (
                a0["cb_mid"], a1["cb_mid"],
                b0["k_mid"], b1["k_mid"],
            )

            if any(value is None for value in values):
                continue

            ages = (
                a0["cb_book_age_ms"],
                a1["cb_book_age_ms"],
                b0["k_book_age_ms"],
                b1["k_book_age_ms"],
            )

            if any(
                age is None or age > 2500
                for age in ages
            ):
                continue

            coin_returns.append(
                (a1["cb_mid"] - a0["cb_mid"])
                / a0["cb_mid"] * 10000
            )

            kalshi_returns.append(
                (b1["k_mid"] - b0["k_mid"])
                / b0["k_mid"] * 10000
            )

        result = correlation(
            coin_returns, kalshi_returns
        )

        label = (
            f"Coinbase leads Kalshi {lag:+d}s | "
            f"n={len(coin_returns)} | "
        )

        if result is None:
            print(label + "insufficient data")
        else:
            print(label + f"correlation={result:.4f}")

    print(
        "\nPositive lag means Coinbase returns are compared "
        "with later Kalshi returns."
    )
    print(
        "This is exploratory correlation, not proof of "
        "an executable or profitable opportunity."
    )
    print("Saved files:", monitor.folder)


async def main(seconds):
    key_id, key = credentials()

    folder = Path(".collector_runs") / (
        "perps_lag_"
        + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    monitor = Collector(folder)
    end = time.monotonic() + seconds

    print("COINBASE / KALSHI BTC LATENCY STUDY")
    print("READ ONLY - NO TRADING")
    print("Market: KXBTCPERP")
    print("Contract size: 0.0001 BTC")
    print(f"Duration: {seconds} seconds")
    print("Ctrl+C to stop.\n")

    try:
        tasks = [
            asyncio.create_task(
                coinbase_stream(monitor, end)
            ),
            asyncio.create_task(
                kalshi_stream(
                    monitor, end, key_id, key
                )
            ),
        ]

        try:
            await sampler(monitor, end)
        finally:
            for task in tasks:
                task.cancel()

            await asyncio.gather(
                *tasks, return_exceptions=True
            )
    finally:
        monitor.close()
        report(monitor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds", type=int, default=300
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.seconds))
    except KeyboardInterrupt:
        pass
