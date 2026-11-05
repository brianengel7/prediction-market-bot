"""Read-only KXBTC15M vs Coinbase quote collector and replay."""

import argparse
import asyncio
import base64
import bisect
import json
import os
import statistics
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

K_WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
K_PATH = "/trade-api/ws/v2"
K_REST = "https://external-api.kalshi.com/trade-api/v2"
C_WS = "wss://advanced-trade-ws.coinbase.com"
SERIES = "KXBTC15M"


def ms():
    return time.time_ns() / 1e6


def iso_ms(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        ).timestamp() * 1000
    except (ValueError, AttributeError):
        return None


def get_json(path, query=None):
    url = K_REST + path
    if query:
        url += "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url, headers={"User-Agent": "btc15m-research/1.0"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def credentials():
    load_dotenv(Path.cwd() / ".env")

    key_id = next((
        os.getenv(k) for k in (
            "KALSHI_API_KEY_ID", "KALSHI_KEY_ID",
            "KALSHI_API_KEY"
        ) if os.getenv(k)
    ), None)

    raw = next((
        os.getenv(k) for k in (
            "KALSHI_PRIVATE_KEY_PATH",
            "KALSHI_API_PRIVATE_KEY_PATH",
            "KALSHI_PRIVATE_KEY_FILE",
            "KALSHI_PRIVATE_KEY_PEM",
            "KALSHI_PRIVATE_KEY"
        ) if os.getenv(k)
    ), None)

    if not key_id or not raw:
        raise RuntimeError("Missing Kalshi credentials in .env")

    pem = (
        raw.replace("\\n", "\n").encode()
        if "-----BEGIN " in raw
        else Path(raw).expanduser().read_bytes()
    )

    return key_id, serialization.load_pem_private_key(
        pem, password=None
    )


def auth_headers(key_id, key):
    timestamp = str(int(ms()))
    message = (timestamp + "GET" + K_PATH).encode()
    signature = key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE":
            base64.b64encode(signature).decode(),
        "KALSHI-ACCESS-TIMESTAMP": timestamp
    }


class State:
    def __init__(self, output):
        self.output = output
        self.out = output.open("w", buffering=1024 * 1024)

        self.coin_book = {"bid": {}, "ask": {}}
        self.coin_ready = False
        self.coin_top = None
        self.coin_seq = None
        self.coin_session = 0

        self.k_book = {"yes": {}, "no": {}}
        self.k_ready = False
        self.k_top = None
        self.k_sid = None
        self.k_seq = None
        self.k_session = 0

        self.market = None
        self.market_meta = None
        self.count = defaultdict(int)

    def emit(self, kind, received=None, **kwargs):
        record = {
            "kind": kind,
            "t_ms": received if received is not None else ms(),
            **kwargs
        }
        self.out.write(json.dumps(record, separators=(",", ":")) + "\n")

    def clear_coin(self):
        self.coin_session += 1
        self.coin_ready = False
        self.coin_book = {"bid": {}, "ask": {}}
        self.coin_top = None
        self.coin_seq = None
        self.emit("invalidate", source="coinbase")

    def clear_k(self):
        self.k_session += 1
        self.k_ready = False
        self.k_book = {"yes": {}, "no": {}}
        self.k_top = None
        self.k_sid = None
        self.k_seq = None
        self.emit("invalidate", source="kalshi")

    def coin_message(self, data, received):
        self.count["coin_messages"] += 1

        if data.get("channel") == "error" or data.get("type") == "error":
            raise RuntimeError("Coinbase error: " + str(data)[:300])

        seq = data.get("sequence_num")
        if isinstance(seq, int):
            if self.coin_seq is not None and seq != self.coin_seq + 1:
                raise RuntimeError("Coinbase sequence gap")
            self.coin_seq = seq

        if data.get("channel") != "l2_data":
            return

        for event in data.get("events", []):
            if event.get("product_id") != "BTC-USD":
                continue

            typ = event.get("type")
            if typ == "snapshot":
                self.coin_book = {"bid": {}, "ask": {}}
                self.coin_ready = True
            elif typ != "update" or not self.coin_ready:
                continue

            for update in event.get("updates", []):
                side = {
                    "bid": "bid",
                    "offer": "ask"
                }.get(update.get("side"))

                if side is None:
                    raise RuntimeError("Unknown Coinbase side")

                price = float(update["price_level"])
                qty = float(update["new_quantity"])

                if price <= 0 or qty < 0:
                    raise RuntimeError("Invalid Coinbase book update")

                if qty == 0:
                    self.coin_book[side].pop(price, None)
                else:
                    self.coin_book[side][price] = qty

            bids = self.coin_book["bid"]
            asks = self.coin_book["ask"]

            if not bids or not asks:
                continue

            bid = max(bids)
            ask = min(asks)

            if bid >= ask:
                raise RuntimeError("Crossed Coinbase book")

            top = (bid, ask)
            if top != self.coin_top:
                self.coin_top = top
                self.count["coin_quotes"] += 1
                self.emit(
                    "quote",
                    received=received,
                    source="coinbase",
                    session=self.coin_session,
                    bid=bid,
                    ask=ask,
                    mid=(bid + ask) / 2,
                    source_ms=iso_ms(data.get("timestamp"))
                )

    def kalshi_message(self, data, received):
        self.count["kalshi_messages"] += 1

        typ = data.get("type")
        body = data.get("msg") or {}

        if typ == "error":
            raise RuntimeError("Kalshi error: " + str(data)[:300])

        if typ == "subscribed" and body.get("channel") == "orderbook_delta":
            self.k_sid = body.get("sid")

        if typ not in ("orderbook_snapshot", "orderbook_delta"):
            return

        if body.get("market_ticker") != self.market:
            return

        if self.k_sid is not None and data.get("sid") != self.k_sid:
            return

        seq = data.get("seq")
        if isinstance(seq, int):
            if (
                typ == "orderbook_delta"
                and self.k_seq is not None
                and seq != self.k_seq + 1
            ):
                raise RuntimeError("Kalshi orderbook sequence gap")
            self.k_seq = seq

        if typ == "orderbook_snapshot":
            self.k_book = {
                "yes": {
                    float(p): float(q)
                    for p, q in body.get("yes_dollars_fp", [])
                    if float(q) > 0
                },
                "no": {
                    float(p): float(q)
                    for p, q in body.get("no_dollars_fp", [])
                    if float(q) > 0
                }
            }
            self.k_ready = True

        elif self.k_ready:
            side = body.get("side")
            if side not in ("yes", "no"):
                raise RuntimeError("Unknown Kalshi side")

            price = float(body["price_dollars"])
            delta = float(body["delta_fp"])

            levels = self.k_book[side]
            new_qty = levels.get(price, 0) + delta

            if new_qty < -0.00001:
                raise RuntimeError("Negative Kalshi book quantity")

            if new_qty <= 0.00001:
                levels.pop(price, None)
            else:
                levels[price] = new_qty
        else:
            return

        yes = self.k_book["yes"]
        no = self.k_book["no"]

        if not yes or not no:
            return

        yes_bid = max(yes)
        no_bid = max(no)

        if yes_bid + no_bid > 1.00001:
            raise RuntimeError("Crossed Kalshi book")

        yes_ask = 1 - no_bid
        no_ask = 1 - yes_bid

        top = (
            yes_bid, yes_ask, no_bid, no_ask,
            yes[yes_bid], no[no_bid]
        )

        if top != self.k_top:
            self.k_top = top
            self.count["kalshi_quotes"] += 1

            self.emit(
                "quote",
                received=received,
                source="kalshi",
                session=self.k_session,
                market=self.market,
                close_ms=self.market_meta["close_ms"],
                strike=self.market_meta.get("floor_strike"),
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                yes_bid_qty=yes[yes_bid],
                yes_ask_qty=no[no_bid],
                no_bid_qty=no[no_bid],
                no_ask_qty=yes[yes_bid],
                source_ms=body.get(
                    "ts_ms", data.get("sending_ts_ms")
                )
            )


def select_market(markets, current_ms):
    active = []

    for market in markets:
        opened = iso_ms(market.get("open_time"))
        closes = iso_ms(market.get("close_time"))

        if opened is None or closes is None:
            continue

        if not opened <= current_ms < closes:
            continue

        if market.get("status") not in ("open", "active"):
            continue

        active.append((closes, market))

    return min(active, key=lambda x: x[0])[1] if active else None


async def discover_markets(state, stop):
    while time.monotonic() < stop:
        try:
            data = await asyncio.to_thread(
                get_json,
                "/markets",
                {
                    "series_ticker": SERIES,
                    "status": "open",
                    "limit": 100
                }
            )

            candidate = select_market(
                data.get("markets", []), ms()
            )
            ticker = candidate.get("ticker") if candidate else None

            if ticker != state.market:
                state.market = ticker
                state.market_meta = (
                    {
                        "close_ms": iso_ms(candidate["close_time"]),
                        "floor_strike": candidate.get("floor_strike")
                    }
                    if candidate else None
                )

                state.clear_k()
                state.emit(
                    "market",
                    market=ticker,
                    meta=state.market_meta
                )
                print(
                    "[market]", ticker or "NO ACTIVE MARKET",
                    "strike:",
                    candidate.get("floor_strike") if candidate else "-",
                    flush=True
                )

            elif candidate and state.market_meta:
                strike = candidate.get("floor_strike")
                if strike != state.market_meta.get("floor_strike"):
                    state.market_meta["floor_strike"] = strike
                    state.emit(
                        "market", market=ticker,
                        meta=state.market_meta
                    )

        except Exception as exc:
            print(
                "[market] discovery error:",
                str(exc)[:200],
                flush=True
            )

        await asyncio.sleep(
            min(10, max(0, stop - time.monotonic()))
        )


async def coinbase_stream(state, stop):
    while time.monotonic() < stop:
        state.clear_coin()

        try:
            async with websockets.connect(
                C_WS,
                max_size=32 * 1024 * 1024,
                max_queue=64,
                ping_interval=20
            ) as ws:
                print("[coinbase] CONNECTED", flush=True)

                await ws.send(json.dumps({
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channel": "level2"
                }))

                await ws.send(json.dumps({
                    "type": "subscribe",
                    "channel": "heartbeats"
                }))

                while time.monotonic() < stop:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=2
                        )
                    except asyncio.TimeoutError:
                        continue

                    received = ms()
                    state.coin_message(
                        json.loads(raw), received
                    )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                "[coinbase] reconnect:",
                str(exc)[:220], flush=True
            )
            await asyncio.sleep(1)
        finally:
            state.clear_coin()


async def kalshi_stream(state, stop, key_id, key):
    while time.monotonic() < stop:
        market = state.market

        if not market:
            await asyncio.sleep(0.4)
            continue

        state.clear_k()

        try:
            async with websockets.connect(
                K_WS,
                additional_headers=auth_headers(key_id, key),
                max_size=8 * 1024 * 1024,
                ping_interval=20
            ) as ws:
                print("[kalshi] CONNECTED", market, flush=True)

                await ws.send(json.dumps({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_ticker": market
                    }
                }))

                while time.monotonic() < stop and market == state.market:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=1
                        )
                    except asyncio.TimeoutError:
                        continue

                    received = ms()
                    state.kalshi_message(
                        json.loads(raw), received
                    )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(
                "[kalshi] reconnect:",
                str(exc)[:220], flush=True
            )
            await asyncio.sleep(1)
        finally:
            state.clear_k()


async def heartbeat(state, stop):
    while time.monotonic() < stop:
        await asyncio.sleep(
            min(5, max(0, stop - time.monotonic()))
        )

        cb = (
            f"{sum(state.coin_top)/2:,.2f}"
            if state.coin_top else "WAIT"
        )
        yes = (
            f"{state.k_top[0]:.3f}/{state.k_top[1]:.3f}"
            if state.k_top else "WAIT"
        )

        print(
            "CB:", cb,
            "| Market:", state.market or "WAIT",
            "| YES bid/ask:", yes,
            "| Recorded:", dict(state.count),
            flush=True
        )


async def collect(seconds):
    key_id, key = credentials()

    folder = Path(".collector_runs") / (
        "btc15m_coinbase_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    folder.mkdir(parents=True, exist_ok=False)

    state = State(folder / "quotes.jsonl")
    stop = time.monotonic() + seconds

    print("READ-ONLY KXBTC15M / COINBASE")
    print("Output:", folder)

    try:
        data = await asyncio.to_thread(
            get_json, "/series/" + SERIES
        )
        series = data["series"]
        state.emit(
            "series",
            fee_type=series.get("fee_type"),
            fee_multiplier=series.get("fee_multiplier")
        )
        print(
            "Fee type:", series.get("fee_type"),
            "Multiplier:", series.get("fee_multiplier")
        )
    except Exception as exc:
        print("Fee lookup unavailable:", str(exc)[:150])

    tasks = [
        asyncio.create_task(discover_markets(state, stop)),
        asyncio.create_task(coinbase_stream(state, stop)),
        asyncio.create_task(kalshi_stream(state, stop, key_id, key)),
        asyncio.create_task(heartbeat(state, stop))
    ]

    try:
        await asyncio.sleep(seconds)
    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        state.out.close()

        print("SAVED:", folder / "quotes.jsonl")
        print("COUNTS:", dict(state.count))


def taker_fee(price, contracts, multiplier):
    p = Decimal(str(price))
    raw = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * Decimal(contracts)
        * p * (1 - p)
    )
    return float(
        (raw * 100).to_integral_value(
            rounding=ROUND_CEILING
        ) / 100
    )


def analyze(path, qty=10, multiplier=1.0):
    rows = [
        json.loads(line)
        for line in path.open()
        if line.strip()
    ]

    cb = []
    markets = defaultdict(list)
    invalids = defaultdict(list)

    series = next(
        (r for r in rows if r.get("kind") == "series"), None
    )

    if series:
        fee_type = series.get("fee_type")
        if fee_type not in ("quadratic", None):
            raise RuntimeError(
                f"Unsupported Kalshi fee type: {fee_type}"
            )
        if series.get("fee_multiplier") is not None:
            if multiplier == 1.0:
                multiplier = float(series["fee_multiplier"])

    for row in rows:
        if row["kind"] == "invalidate":
            invalids[row["source"]].append(row["t_ms"])

        elif row["kind"] == "quote":
            if row["source"] == "coinbase":
                cb.append((
                    row["t_ms"], row["mid"], row["session"]
                ))
            elif row["source"] == "kalshi":
                markets[row["market"]].append(row)

    cb_times = [v[0] for v in cb]
    arrays = {
        ticker: ([r["t_ms"] for r in quotes], quotes)
        for ticker, quotes in markets.items()
    }

    def continuous(source, quote_time, target_time):
        i = bisect.bisect_right(
            invalids[source], target_time
        ) - 1
        return (
            i < 0 or invalids[source][i] <= quote_time
        )

    trials = defaultdict(list)
    seen = defaultdict(int)
    last_signal = defaultdict(lambda: -1e18)

    for t, mid, cb_session in cb:
        previous_i = bisect.bisect_right(
            cb_times, t - 1000
        ) - 1

        if previous_i < 0:
            continue

        previous = cb[previous_i]

        if (
            t - previous[0] > 1500
            or previous[2] != cb_session
            or not continuous("coinbase", previous[0], t)
        ):
            continue

        move = mid - previous[1]

        for threshold in (10, 20, 50):
            if abs(move) < threshold:
                continue

            if t - last_signal[threshold] < 3000:
                continue

            eligible = []

            for ticker, (times, quotes) in arrays.items():
                i = bisect.bisect_right(times, t) - 1
                if i < 0:
                    continue

                q = quotes[i]

                if (
                    0 <= t - times[i] <= 1000
                    and q["close_ms"] > t + 9000
                    and continuous("kalshi", times[i], t)
                ):
                    eligible.append((q["close_ms"], ticker))

            if not eligible:
                continue

            _, ticker = min(eligible)
            times, quotes = arrays[ticker]

            i0 = bisect.bisect_right(times, t) - 1
            initial_session = quotes[i0]["session"]

            direction = "yes" if move > 0 else "no"
            seen[threshold] += 1
            last_signal[threshold] = t

            for delay in (100, 250, 500, 1000, 2000):
                entry_t = t + delay
                i = bisect.bisect_right(times, entry_t) - 1

                if i < 0:
                    continue

                entry = quotes[i]

                if (
                    entry_t - times[i] > 1000
                    or entry["session"] != initial_session
                    or not continuous(
                        "kalshi", times[i], entry_t
                    )
                ):
                    continue

                entry_price = entry[f"{direction}_ask"]
                entry_qty = entry[f"{direction}_ask_qty"]

                if (
                    entry_qty < qty
                    or not 0 < entry_price < 1
                ):
                    continue

                for hold in (1000, 3000, 5000):
                    exit_t = entry_t + hold

                    if exit_t >= entry["close_ms"]:
                        continue

                    j = bisect.bisect_right(
                        times, exit_t
                    ) - 1

                    if j < 0:
                        continue

                    exit_quote = quotes[j]

                    if (
                        exit_t - times[j] > 1000
                        or exit_quote["session"] != initial_session
                        or not continuous(
                            "kalshi", times[j], exit_t
                        )
                    ):
                        continue

                    exit_price = exit_quote[
                        f"{direction}_bid"
                    ]
                    exit_qty = exit_quote[
                        f"{direction}_bid_qty"
                    ]

                    if (
                        exit_qty < qty
                        or not 0 < exit_price < 1
                    ):
                        continue

                    gross = qty * (
                        exit_price - entry_price
                    )
                    fees = (
                        taker_fee(
                            entry_price, qty, multiplier
                        )
                        + taker_fee(
                            exit_price, qty, multiplier
                        )
                    )

                    trials[
                        (threshold, delay, hold)
                    ].append(gross - fees)

    print("\n=== KXBTC15M COINBASE REPLAY ===")
    print("Data:", path)
    print("Coinbase quotes:", len(cb))
    print("Kalshi markets:", len(markets))
    print("Contracts per simulated trade:", qty)
    print("Fee multiplier:", multiplier)
    print("Coinbase jump triggers:", dict(seen))
    print()
    print(
        "Jump | Delay | Hold | Trades | "
        "Avg net $/trade | Net win %"
    )

    for (threshold, delay, hold), values in sorted(trials.items()):
        count = len(values)
        mean = statistics.mean(values)
        wins = sum(v > 0 for v in values) / count * 100

        print(
            f"${threshold:>2} | {delay:>4}ms | "
            f"{hold // 1000:>2}s | {count:>5} | "
            f"{mean:+.4f} | {wins:.1f}%"
        )

    print("\nREAD-ONLY RESEARCH.")
    print("Top-of-book fillability assumed; no actual fills.")
    print("Slippage and adverse selection not modeled.")
    print("No settlement probability model calibrated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(
        dest="mode", required=True
    )

    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument(
        "--seconds", type=int, default=120
    )

    analysis_parser = sub.add_parser("analyze")
    analysis_parser.add_argument(
        "--file", type=Path
    )
    analysis_parser.add_argument(
        "--qty", type=int, default=10
    )
    analysis_parser.add_argument(
        "--fee-multiplier", type=float, default=1.0
    )

    args = parser.parse_args()

    if args.mode == "collect":
        try:
            asyncio.run(collect(args.seconds))
        except KeyboardInterrupt:
            pass

    else:
        path = args.file

        if path is None:
            files = list(
                Path(".collector_runs").glob(
                    "btc15m_coinbase_*/quotes.jsonl"
                )
            )
            if not files:
                parser.error("No collector data found.")
            path = max(
                files, key=lambda p: p.stat().st_mtime
            )

        analyze(
            path,
            qty=args.qty,
            multiplier=args.fee_multiplier
        )
