"""Read-only NYC YES basket execution feasibility audit."""

import argparse
import asyncio
import inspect
import json
import time
from decimal import Decimal as D
from datetime import datetime

import websockets

from src.backtest.weather_yes_basket_scanner import (
    NY, WS_URL, CAP, Book, auth_headers, clock, scan, today_event,
)
from src.kalshi.fees import calculate_taker_fee


DELAYS_MS = (100, 250, 500, 1000, 2000, 5000)


def fill_at_limit(book, max_yes_ask, quantity):
    """Simulate a YES fill from displayed NO bids."""
    left = quantity
    spent = D(0)

    for no_bid, depth in sorted(
        book.levels["no"].items(), reverse=True
    ):
        yes_ask = D(1) - no_bid

        if yes_ask > max_yes_ask:
            break

        take = min(depth, left)

        if take > 0:
            spent += yes_ask * take
            left -= take

        if left <= 0:
            return spent / quantity

    return None


def evaluate_pair(ep, books, top):
    """Check both original limits against current displayed depth."""
    chosen = {q["ticker"] for q in top[:2]} if top else set()

    if chosen != set(ep["tickers"]):
        return False, "top-two pair changed", None

    paid = []

    for ticker in ep["tickers"]:
        p = fill_at_limit(
            books[ticker],
            ep["limits"][ticker],
            ep["pairs"],
        )

        if p is None:
            return False, f"depth/price unavailable: {ticker}", None

        paid.append(p)

    total = sum(paid, D(0))

    if total >= CAP:
        return False, "combined fill price above cap", None

    return True, "both legs available at original limits", paid


def total_fees(prices, pairs):
    return sum(
        (
            D(str(calculate_taker_fee(
                price=float(p),
                contracts=pairs,
                multiplier=1,
            )))
            for p in prices
        ),
        D(0),
    )


class Evaluator:
    def __init__(self, event, tickers, pairs):
        self.event = event
        self.tickers = tickers
        self.pairs = D(pairs)
        self.books = {t: Book() for t in tickers}

        self.active = None
        self.pending = []
        self.count = {d: [0, 0] for d in DELAYS_MS}
        self.sequential = {
            "both": 0,
            "exposed": 0,
            "no_entry": 0,
        }

        self.episodes = 0
        self.shallow_updates = 0
        self.updates = 0
        self.last_top = None
        self.seq = None

    def process_book(self, packet):
        seq = packet.get("seq")

        if seq is not None:
            seq = int(seq)

            if self.seq is not None and seq != self.seq + 1:
                raise RuntimeError(
                    f"WebSocket sequence gap {self.seq} -> {seq}"
                )

            self.seq = seq

        msg = packet["msg"]
        ticker = msg["market_ticker"]

        if ticker not in self.books:
            raise RuntimeError(f"Unexpected market {ticker}")

        if packet["type"] == "orderbook_snapshot":
            self.books[ticker].snapshot(msg)
        else:
            self.books[ticker].delta(msg)

        self.updates += 1

    def updated(self):
        if not all(b.ready for b in self.books.values()):
            return

        state = scan(self.books, self.tickers)
        now = time.monotonic_ns()
        self.last_top = state

        below_cap = (
            state is not None
            and state[2] < CAP
            and state[3]
        )

        enough_depth = (
            below_cap
            and all(
                q["ask_qty"] >= self.pairs
                for q in state[:2]
            )
        )

        if below_cap and not enough_depth:
            self.shallow_updates += 1

        names = (
            tuple(q["ticker"] for q in state[:2])
            if state else ()
        )

        if self.active and (
            not enough_depth
            or names != self.active["tickers"]
        ):
            self.close_active(now)

        if enough_depth and self.active is None:
            self.episodes += 1
            a, b, combined, _ = state

            ep = {
                "id": self.episodes,
                "start": now,
                "tickers": names,
                "limits": {
                    q["ticker"]: q["ask"]
                    for q in (a, b)
                },
                "pairs": self.pairs,
                "samples": {},
                "ended": None,
                "thin": min(
                    (a, b),
                    key=lambda q: q["ask_qty"],
                )["ticker"],
                "seq_leg1": None,
                "seq_done": False,
            }

            self.pending.append(ep)
            self.active = ep

            print(
                f"\n[{clock()}] SIGNAL #{ep['id']} "
                f"pairs={self.pairs}",
                flush=True,
            )

            for q in (a, b):
                print(
                    f"  {q['ticker']} "
                    f"ask={q['ask']*100:.2f}c "
                    f"depth={q['ask_qty']}",
                    flush=True,
                )

            print(
                f"  Combined asks: {combined*100:.2f}c",
                flush=True,
            )

    def close_active(self, now):
        ep = self.active

        if ep is None:
            return

        ep["ended"] = now
        duration = (now - ep["start"]) / 1e6

        print(
            f"[{clock()}] END #{ep['id']} "
            f"duration={duration:.0f}ms",
            flush=True,
        )

        self.active = None

    def check_delays(self):
        now = time.monotonic_ns()
        current_top = (
            self.last_top[:2] if self.last_top else None
        )

        for ep in self.pending[:]:
            age_ms = (now - ep["start"]) / 1e6

            for delay in DELAYS_MS:
                if delay in ep["samples"]:
                    continue

                if age_ms < delay:
                    continue

                if ep["ended"] is not None:
                    ok, reason, paid = (
                        False, "signal expired", None
                    )
                else:
                    ok, reason, paid = evaluate_pair(
                        ep, self.books, current_top
                    )

                ep["samples"][delay] = ok
                self.count[delay][0] += 1
                self.count[delay][1] += int(ok)

                extra = ""

                if paid is not None:
                    paid_total = sum(paid, D(0))
                    fees = total_fees(
                        paid, int(ep["pairs"])
                    )
                    all_in = (
                        paid_total + fees / ep["pairs"]
                    )

                    extra = (
                        f" asks={paid_total*100:.2f}c/pair"
                        f" all-in={all_in*100:.2f}c/pair"
                    )

                print(
                    f"[{clock()}] #{ep['id']} "
                    f"delay={delay}ms "
                    f"observed={age_ms:.0f}ms "
                    f"{'PAPER-OK' if ok else 'FAIL'} "
                    f"{reason}{extra}",
                    flush=True,
                )

            # Sequential fill model:
            # thinner side at +250ms;
            # other side at +500ms.
            if ep["seq_leg1"] is None and age_ms >= 250:
                first = ep["thin"]

                if ep["ended"] is not None:
                    ep["seq_leg1"] = False
                else:
                    ep["seq_leg1"] = (
                        fill_at_limit(
                            self.books[first],
                            ep["limits"][first],
                            ep["pairs"],
                        ) is not None
                    )

                print(
                    f"[{clock()}] #{ep['id']} "
                    f"sequential first leg: "
                    f"{'PAPER-FILL' if ep['seq_leg1'] else 'NO FILL'}",
                    flush=True,
                )

            if not ep["seq_done"] and age_ms >= 500:
                ep["seq_done"] = True

                if not ep["seq_leg1"]:
                    outcome = "no_entry"
                else:
                    second = next(
                        t for t in ep["tickers"]
                        if t != ep["thin"]
                    )

                    p = fill_at_limit(
                        self.books[second],
                        ep["limits"][second],
                        ep["pairs"],
                    )

                    outcome = (
                        "both" if p is not None else "exposed"
                    )

                self.sequential[outcome] += 1

                print(
                    f"[{clock()}] #{ep['id']} "
                    f"SEQUENTIAL RESULT: {outcome.upper()}",
                    flush=True,
                )

            if (
                len(ep["samples"]) == len(DELAYS_MS)
                and ep["seq_done"]
            ):
                self.pending.remove(ep)

    def report(self):
        ready = sum(
            b.ready for b in self.books.values()
        )

        price = (
            f"{self.last_top[2]*100:.2f}c"
            if self.last_top else "unknown"
        )

        print(
            f"\n[{clock()}] HEARTBEAT "
            f"books={ready}/{len(self.books)} "
            f"updates={self.updates} "
            f"price={price} "
            f"episodes={self.episodes} "
            f"shallow_updates={self.shallow_updates}",
            flush=True,
        )

        print("  Delayed pair availability:")

        for delay in DELAYS_MS:
            total, passed = self.count[delay]

            pct = (
                100 * passed / total
                if total else 0
            )

            print(
                f"    {delay:>5}ms: "
                f"{passed}/{total} "
                f"({pct:.1f}%)",
                flush=True,
            )

        print(
            "  Sequential paper outcomes: "
            f"both={self.sequential['both']} "
            f"one_leg_exposed={self.sequential['exposed']} "
            f"no_entry={self.sequential['no_entry']}",
            flush=True,
        )


async def run_event(day, event, tickers, heartbeat, pairs):
    tracker = Evaluator(event, tickers, pairs)

    header_arg = (
        "additional_headers"
        if "additional_headers" in
        inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )

    async with websockets.connect(
        WS_URL,
        **{header_arg: auth_headers()},
        ping_interval=20,
        ping_timeout=20,
        max_queue=4096,
    ) as ws:

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
            f"[{clock()}] Connected to {event}; "
            "awaiting orderbook snapshots",
            flush=True,
        )

        last_heartbeat = time.monotonic()

        while datetime.now(NY).date() == day:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(), timeout=0.02
                )
            except asyncio.TimeoutError:
                raw = None

            if raw is not None:
                packet = json.loads(raw)
                kind = packet.get("type")

                if kind == "error":
                    raise RuntimeError(
                        f"Kalshi WS error: {packet.get('msg')}"
                    )

                if kind in (
                    "orderbook_snapshot",
                    "orderbook_delta",
                ):
                    tracker.process_book(packet)
                    tracker.updated()

                elif kind == "subscribed":
                    print(
                        f"[{clock()}] Subscription acknowledged",
                        flush=True,
                    )

            tracker.check_delays()

            if (
                time.monotonic() - last_heartbeat
                >= heartbeat
            ):
                tracker.report()
                last_heartbeat = time.monotonic()

        tracker.close_active(time.monotonic_ns())
        tracker.report()


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--heartbeat", type=float, default=10
    )
    parser.add_argument(
        "--pairs", type=int, default=1
    )
    args = parser.parse_args()

    if args.heartbeat < 1:
        parser.error("Heartbeat must be >= 1 second")

    if not 1 <= args.pairs <= 100:
        parser.error("--pairs must be between 1 and 100")

    print("NYC WEATHER YES BASKET EXECUTION FEASIBILITY")
    print("High-frequency WebSocket, no fixed market polling")
    print(f"Simulated pairs: {args.pairs}")
    print(f"Entry cap: strictly below {CAP * 100:.0f}c before fees")
    print("READ ONLY: no orders or database writes")

    reconnect_delay = 1

    while True:
        try:
            day, event, tickers = today_event()

            print(
                f"[{clock()}] Watching {event}",
                flush=True,
            )

            await run_event(
                day, event, tickers,
                args.heartbeat, args.pairs
            )

            reconnect_delay = 1

        except (KeyboardInterrupt, asyncio.CancelledError):
            raise

        except Exception as exc:
            print(
                f"[{clock()}] ERROR "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            print(
                f"Reconnecting in {reconnect_delay}s "
                "with fresh books",
                flush=True,
            )

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(
                reconnect_delay * 2, 20
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScanner stopped.")
