"""Kalshi KXBTC15M 1.5-2c reversal research: public REST + authenticated WS; PAPER ONLY.

No order submission code. Uses a separate SQLite file. Run from repository root:
 python -m src.crypto.btc_reversal_shadow_v1 --self-test
 python -m src.crypto.btc_reversal_shadow_v1 --seconds 3600
 python -m src.crypto.btc_reversal_shadow_v1 --report
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal as D, ROUND_CEILING
from pathlib import Path

ONE = D("1")
MIN_PRICE = D("0.015")
MAX_PRICE = D("0.020")
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
REST_URL = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_DB = Path(".collector_runs/btc_reversal_shadow_v3.sqlite3")
SIZES = (10, 50, 100)
PRIMARY_SIZE = 10  # First executable entry must have this much depth
DELAYS_MS = (0, 100, 250, 500, 1000)


def utc(ts: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).isoformat(timespec="milliseconds")


def ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def num(value) -> D:
    d = D(str(value))
    if not d.is_finite():
        raise ValueError("Nonfinite order book number")
    return d


def fee(price: D, qty: int) -> D:
    # Matches the fee assumption used in the historical research; not an official quote.
    return (D("0.07") * qty * price * (ONE - price)).quantize(D("0.01"), rounding=ROUND_CEILING)


class Book:
    def __init__(self):
        self.levels: dict[str, dict[D, D]] = {"yes": {}, "no": {}}
        self.ready = False
        self.last_change = 0.0

    def apply(self, kind: str, payload: dict):
        if kind == "orderbook_snapshot":
            update = {"yes": {}, "no": {}}
            for side in ("yes", "no"):
                for raw_p, raw_q in payload.get(side + "_dollars_fp", []) or []:
                    p, q = num(raw_p), num(raw_q)
                    if not D(0) < p < ONE or q < 0:
                        raise ValueError("Bad book snapshot")
                    if q > 0:
                        update[side][p] = q
            self.levels = update
            self.ready = True
        else:
            if not self.ready:
                raise ValueError("Delta before snapshot")
            side = payload.get("side")
            if side not in self.levels:
                raise ValueError("Unknown book side")
            p, delta = num(payload["price_dollars"]), num(payload["delta_fp"])
            if not D(0) < p < ONE:
                raise ValueError("Bad delta price")
            new_q = self.levels[side].get(p, D(0)) + delta
            if new_q < 0:
                raise ValueError("Negative book qty: resync required")
            if new_q:
                self.levels[side][p] = new_q
            else:
                self.levels[side].pop(p, None)
        self.last_change = time.monotonic()

    def levels_for_buy(self, side: str):
        opposite = "no" if side == "yes" else "yes"
        return sorted(
            ((ONE - bid, qty) for bid, qty in self.levels[opposite].items() if qty > 0),
            key=lambda x: x[0]
        )

    def top(self, side: str):
        levels = self.levels_for_buy(side)
        return levels[0] if self.ready and levels else None

    def cost(self, side: str, qty: int):
        """Acquire full qty with price cap at every consumed level; conservative fee per level."""
        remaining = D(qty)
        gross, estimated_fee, filled = D(0), D(0), D(0)
        for ask, available in self.levels_for_buy(side):
            if ask > MAX_PRICE:
                break
            if ask <= 0:
                continue
            take = min(remaining, available)
            if take <= 0:
                continue
            gross += take * ask
            # Round fee at each level, to avoid understating fees for fragmented fills.
            estimated_fee += (D("0.07") * take * ask * (ONE - ask)).quantize(D("0.01"), rounding=ROUND_CEILING)
            remaining -= take
            filled += take
            if remaining <= 0:
                return {"vwap": gross / filled, "gross": gross, "fee": estimated_fee,
                        "total": gross + estimated_fee, "available": filled}
        return None


@dataclass
class Watch:
    ticker: str
    start: float
    close: float
    book: Book = field(default_factory=Book)
    covered: bool = False
    invalid: bool = False
    first_cheap: bool = False
    last_sample: int = -1


def open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=3000")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS watch (
      ticker TEXT PRIMARY KEY, start_ts REAL NOT NULL, close_ts REAL NOT NULL,
      covered INTEGER NOT NULL DEFAULT 0, invalid_reason TEXT,
      first_cheap_at TEXT, first_side TEXT, first_price TEXT, signal INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS signals (
      ticker TEXT PRIMARY KEY, detected_utc TEXT NOT NULL, side TEXT NOT NULL,
      ask TEXT NOT NULL, seconds_left REAL NOT NULL, best_qty TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS fill_checks (
      ticker TEXT NOT NULL, qty INTEGER NOT NULL, delay_ms INTEGER NOT NULL,
      checked_utc TEXT NOT NULL, outcome TEXT NOT NULL,
      vwap TEXT, gross TEXT, fee TEXT, total TEXT, pnl TEXT,
      PRIMARY KEY (ticker, qty, delay_ms)
    );
    CREATE TABLE IF NOT EXISTS settlements (
      ticker TEXT PRIMARY KEY, outcome TEXT NOT NULL, settled_utc TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS samples (
      ticker TEXT NOT NULL, second INTEGER NOT NULL, seconds_left REAL NOT NULL,
      yes_ask TEXT, no_ask TEXT, yes_qty TEXT, no_qty TEXT,
      PRIMARY KEY(ticker, second)
    );
    """)
    return db


def rest_get(path: str, params=None):
    import requests
    for attempt in range(4):
        try:
            r = requests.get(REST_URL + path, params=params, timeout=12)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Kalshi temporary HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, RuntimeError):
            if attempt == 3:
                raise
            time.sleep(min(7, 1.5 * (attempt + 1)))


def discover():
    # Public endpoint; get near-dated markets only and keep WS subscriptions small.
    response = rest_get("/markets", {"series_ticker": "KXBTC15M", "status": "open", "limit": 1000})
    now = time.time()
    found = []
    for item in response.get("markets", []):
        try:
            ticker = item["ticker"]
            close = ts(item["close_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if ticker.startswith("KXBTC15M-") and now < close < now + 2700:
            found.append((ticker, close))
    return sorted(found, key=lambda x: x[1])


class Shadow:
    def __init__(self, args):
        self.args = args
        self.db = open_db(args.db)
        self.watch: dict[str, Watch] = {}
        self.pending: list[dict] = []
        self.next_id = 0
        self.sequences: dict[int, int] = {}
        self.signals = 0
        self.rejection_seen = {}  # Throttle rejection logs by reason/market
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS entry_rejections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL, observed_utc TEXT NOT NULL,
                seconds_left REAL NOT NULL, side TEXT NOT NULL,
                best_ask TEXT, available_at_cap TEXT,
                reason TEXT NOT NULL
            )""")

    def add_watch(self, ticker, close):
        if ticker in self.watch:
            return

        start = close - 900
        w = Watch(ticker, start, close)

        # Only an actual previous paper signal blocks a repeat.
        row = self.db.execute(
            "SELECT 1 FROM signals WHERE ticker=?",
            (ticker,)
        ).fetchone()

        w.first_cheap = bool(row)
        self.watch[ticker] = w

        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO watch "
                "(ticker,start_ts,close_ts) VALUES (?,?,?)",
                (ticker, start, close)
            )
            self.db.execute(
                "UPDATE watch SET invalid_reason=NULL "
                "WHERE ticker=?",
                (ticker,)
            )

        print(
            f"WATCH {ticker} closes={utc(close)} "
            f"seconds_left={max(0, close-time.time()):.1f} "
            f"already_signaled={w.first_cheap}",
            flush=True
        )

    def invalidate(self, w: Watch, reason: str):
        w.invalid = True
        w.covered = False
        with self.db:
            self.db.execute(
                "UPDATE watch SET covered=0, invalid_reason=COALESCE(invalid_reason,?) WHERE ticker=?",
                (reason, w.ticker)
            )

    def on_book(self, ticker, kind, data):
        w = self.watch.get(ticker)
        if w is None:
            return

        w.book.apply(kind, data)

        # A fresh snapshot is valid even after market start.
        if kind == "orderbook_snapshot":
            w.covered = True

            with self.db:
                self.db.execute(
                    "UPDATE watch SET covered=1, "
                    "invalid_reason=NULL WHERE ticker=?",
                    (ticker,)
                )

    def reject_entry(self, w: Watch, now: float, side: str,
                     price, available, reason: str):
        # At most once per 10s per market/side/reason. The WS scan itself
        # remains continuous; only the debug writes are rate-limited.
        key = (w.ticker, side, reason)
        if now - self.rejection_seen.get(key, float("-inf")) < 10:
            return
        self.rejection_seen[key] = now
        left = w.close - now
        with self.db:
            self.db.execute(
                "INSERT INTO entry_rejections "
                "(ticker,observed_utc,seconds_left,side,best_ask,"
                "available_at_cap,reason) VALUES (?,?,?,?,?,?,?)",
                (w.ticker, utc(now), left, side,
                 str(price) if price is not None else None,
                 str(available) if available is not None else None,
                 reason)
            )
        print(f"ENTRY REJECT {w.ticker} side={side} "
              f"ask={price} depth_at_cap={available} "
              f"left={left:.1f}s reason={reason}", flush=True)

    def check_first_cheap(self, w: Watch, now: float):
        # Legacy name; V3 uses first EXECUTABLE quote, never first cheap quote.
        left = w.close - now
        if w.first_cheap or not 120 <= left < 300:
            return
        if not w.book.ready or not w.covered:
            return
        # An unchanged book remains valid on a healthy WS connection.
        # Connection resets clear w.covered; idle quotes are not "stale".

        eligible = []
        for side in ("yes", "no"):
            quote = w.book.top(side)
            if quote is None:
                continue
            price, best_qty = quote
            if MIN_PRICE < price <= MAX_PRICE:
                eligible.append((side, price, best_qty))

        if len(eligible) > 1:
            self.reject_entry(w, now, "both", None, None, "AMBIGUOUS_BOOK")
            return
        if not eligible:
            # Below 1.5c, above 2c, or absent quotes: keep scanning.
            return

        side, price, best_qty = eligible[0]
        # Aggregate opposite-side bids corresponding to asks <= 2c.
        available = sum(
            (q for ask, q in w.book.levels_for_buy(side)
             if D(0) < ask <= MAX_PRICE), D(0)
        )
        cost = w.book.cost(side, PRIMARY_SIZE)
        if cost is None:
            self.reject_entry(w, now, side, price, available,
                              "INSUFFICIENT_DEPTH")
            return

        # Only NOW lock the market. No lock after a price-only observation.
        detected = utc(now)
        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO signals "
                "(ticker,detected_utc,side,ask,seconds_left,best_qty) "
                "VALUES (?,?,?,?,?,?)",
                (w.ticker, detected, side, str(price), left, str(best_qty))
            )
            self.db.execute(
                "UPDATE watch SET first_cheap_at=?,first_side=?,"
                "first_price=?,signal=1 WHERE ticker=?",
                (detected, side, str(price), w.ticker)
            )
        w.first_cheap = True
        if cursor.rowcount == 0:
            return  # Existing signal persisted from a previous process.
        self.signals += 1
        print(f"PAPER ENTRY FIRST_EXECUTABLE_10_V3 {w.ticker} "
              f"side={side} best_ask={price} "
              f"available_at_cap={available} "
              f"vwap10={cost['vwap']} "
              f"est_total10={cost['total']} "
              f"left={left:.1f}s", flush=True)

        origin = time.monotonic()
        for delay in DELAYS_MS:
            for qty in SIZES:
                self.pending.append({
                    "ticker": w.ticker, "side": side, "qty": qty,
                    "delay_ms": delay, "due": origin + delay / 1000
                })
        # Observe 0ms depth on exactly the same book that triggered entry.
        # Later delays remain independent counterfactuals.
        self.evaluate_pending()

    def evaluate_pending(self):
        now = time.monotonic()
        due = [p for p in self.pending if p["due"] <= now]
        self.pending = [p for p in self.pending if p["due"] > now]
        for p in due:
            w = self.watch.get(p["ticker"])
            value = None
            if not w or not w.book.ready or not w.covered:
                outcome = "NO_BOOK"
            else:
                top = w.book.top(p["side"])
                if top is None or top[0] > MAX_PRICE:
                    outcome = "PRICE_MOVED"
                else:
                    value = w.book.cost(p["side"], p["qty"])
                    outcome = ("PAPER_DEPTH_OK" if value
                               else "INSUFFICIENT_DEPTH")
            with self.db:
                self.db.execute("""INSERT OR IGNORE INTO fill_checks
                    (ticker,qty,delay_ms,checked_utc,outcome,vwap,gross,fee,total)
                    VALUES (?,?,?,?,?,?,?,?,?)""", (
                        p["ticker"], p["qty"], p["delay_ms"], utc(), outcome,
                        str(value["vwap"]) if value else None,
                        str(value["gross"]) if value else None,
                        str(value["fee"]) if value else None,
                        str(value["total"]) if value else None,
                    ))
            if p["qty"] == PRIMARY_SIZE:
                print(f"FILL CHECK {p['ticker']} {p['delay_ms']}ms "
                      f"qty={p['qty']} {outcome} "
                      f"vwap={value['vwap'] if value else '-'}", flush=True)

    def sample(self, w: Watch, now: float):
        remaining = w.close - now
        if not w.book.ready or not 0 < remaining <= 600:
            return
        second = int(now)
        if second == w.last_sample:
            return
        w.last_sample = second
        yes, no = w.book.top("yes"), w.book.top("no")
        with self.db:
            self.db.execute("""INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?)""", (
                w.ticker, second, remaining,
                str(yes[0]) if yes else None, str(no[0]) if no else None,
                str(yes[1]) if yes else None, str(no[1]) if no else None,
            ))

    def tick(self):
        now = time.time()
        for w in self.watch.values():
            if w.start <= now < w.close:
                self.check_first_cheap(w, now)
                self.sample(w, now)
        self.evaluate_pending()

    def lost_connection(self, reason):
        # Disconnect resets the order book,
        # not the market's trading eligibility.
        for w in self.watch.values():
            w.book = Book()
            w.covered = False

        with self.db:
            for w in self.watch.values():
                self.db.execute(
                    "UPDATE watch SET covered=0 "
                    "WHERE ticker=?",
                    (w.ticker,)
                )

            # Never simulate fills on stale data.
            for item in self.pending:
                self.db.execute(
                    """INSERT OR IGNORE INTO fill_checks
                    (ticker,qty,delay_ms,checked_utc,outcome)
                    VALUES (?,?,?,?,?)""",
                    (
                        item["ticker"],
                        item["qty"],
                        item["delay_ms"],
                        utc(),
                        "DISCONNECTED"
                    )
                )

        self.pending.clear()
        self.sequences.clear()

    async def settle(self):
        rows = self.db.execute("""SELECT DISTINCT s.ticker FROM signals s
            LEFT JOIN settlements t ON s.ticker=t.ticker WHERE t.ticker IS NULL""").fetchall()
        for (ticker,) in rows:
            w = self.watch.get(ticker)
            close = w.close if w else self.db.execute(
                "SELECT close_ts FROM watch WHERE ticker=?", (ticker,)
            ).fetchone()[0]
            if time.time() < close + 30:
                continue
            try:
                response = await asyncio.to_thread(rest_get, "/markets/" + ticker)
                market = response.get("market", {})
                result = market.get("result")
                if result not in ("yes", "no") or market.get("status") not in ("settled", "finalized"):
                    continue
                side = self.db.execute("SELECT side FROM signals WHERE ticker=?", (ticker,)).fetchone()[0]
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO settlements VALUES (?,?,?)", (ticker, result, utc()))
                    fill_rows = self.db.execute("""SELECT qty,delay_ms,total FROM fill_checks
                        WHERE ticker=? AND outcome='PAPER_DEPTH_OK'""", (ticker,)).fetchall()
                    for qty, delay, cost in fill_rows:
                        pnl = (D(qty) if side == result else D(0)) - D(cost)
                        self.db.execute("UPDATE fill_checks SET pnl=? WHERE ticker=? AND qty=? AND delay_ms=?",
                                        (str(pnl), ticker, qty, delay))
                print(f"SETTLED {ticker} result={result} paper_side={side}", flush=True)
            except Exception as e:
                print(f"SETTLEMENT RETRY {ticker} {type(e).__name__}: {e}", flush=True)

    async def session(self, deadline):
        import websockets
        from src.kalshi.trading_client import build_auth_headers
        headers = build_auth_headers("GET", WS_URL)
        kwargs = {"ping_interval": 15, "ping_timeout": 15, "max_size": 8 * 1024 * 1024, "max_queue": 4096}
        header_kw = "additional_headers" if "additional_headers" in inspect.signature(websockets.connect).parameters else "extra_headers"
        kwargs[header_kw] = headers
        async with websockets.connect(WS_URL, **kwargs) as ws:
            self.next_id = 0
            self.sequences.clear()
            subscribed = set()
            discover_task = None
            settle_task = None
            last_discover = -1e12
            last_settle = -1e12
            last_status = 0.0
            print("Connected to Kalshi WS. PAPER ONLY. NO ORDER SUBMISSION.", flush=True)
            while time.monotonic() < deadline:
                mono = time.monotonic()
                # REST tasks run concurrently; neither can hold up book message intake.
                if discover_task is None and mono - last_discover >= self.args.discover_seconds:
                    discover_task = asyncio.create_task(asyncio.to_thread(discover))
                    last_discover = mono
                if discover_task is not None and discover_task.done():
                    try:
                        market_list = discover_task.result()
                        for ticker, close in market_list:
                            self.add_watch(ticker, close)
                        additions = [ticker for ticker, _ in market_list if ticker not in subscribed]
                        if additions:
                            self.next_id += 1
                            await ws.send(json.dumps({"id": self.next_id, "cmd": "subscribe",
                                "params": {"channels": ["orderbook_delta"], "market_tickers": additions}}))
                            subscribed.update(additions)
                            print("SUBSCRIBE", additions, flush=True)
                    except Exception as e:
                        print(f"DISCOVERY ERROR {e}", flush=True)
                    finally:
                        discover_task = None
                if settle_task is None and mono - last_settle >= 30:
                    settle_task = asyncio.create_task(self.settle())
                    last_settle = mono
                if settle_task is not None and settle_task.done():
                    try:
                        settle_task.result()
                    except Exception as e:
                        print(f"SETTLEMENT TASK ERROR {e}", flush=True)
                    settle_task = None
                if mono - last_status >= 60:
                    active = sum(w.start <= time.time() < w.close and w.covered for w in self.watch.values())
                    print(f"HEARTBEAT {utc()} active_books={active} signals={self.signals} pending={len(self.pending)}", flush=True)
                    last_status = mono
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                except asyncio.TimeoutError:
                    raw = None
                if raw is not None:
                    packet = json.loads(raw)
                    kind = packet.get("type")
                    if kind == "error":
                        raise RuntimeError(f"WS error: {packet.get('msg')}")
                    if kind in ("orderbook_snapshot", "orderbook_delta"):
                        sid, seq = packet["sid"], packet["seq"]
                        if sid in self.sequences and seq != self.sequences[sid] + 1:
                            raise RuntimeError(f"WS sequence gap sid={sid}: {self.sequences[sid]}->{seq}")
                        self.sequences[sid] = seq
                        body = packet["msg"]
                        self.on_book(body["market_ticker"], kind, body)
                self.tick()

    async def run(self):
        from dotenv import load_dotenv
        load_dotenv()
        print(f"BTC REVERSAL SHADOW V1 db={self.args.db} quantities={SIZES} delays_ms={DELAYS_MS}", flush=True)
        deadline = time.monotonic() + self.args.seconds if self.args.seconds else float("inf")
        try:
            while time.monotonic() < deadline:
                try:
                    await self.session(deadline)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"WS RESET {type(e).__name__}: {e} — resubscribe and refresh order books", flush=True)
                    self.lost_connection("websocket_gap")
                    await asyncio.sleep(5)
        finally:
            self.lost_connection("process_stopped")
            await self.settle()
            self.db.close()
            print("Stopped. Zero orders submitted.", flush=True)


def report(path: Path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    total, signals = db.execute("SELECT COUNT(*) FROM watch").fetchone()[0], db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print(f"WATCHED={total} SIGNALS={signals} (simulation only)")
    for qty in SIZES:
        for delay in DELAYS_MS:
            rows = db.execute("""SELECT f.outcome,f.pnl FROM fill_checks f
                WHERE qty=? AND delay_ms=?""", (qty, delay)).fetchall()
            ok = sum(x[0] == "PAPER_DEPTH_OK" for x in rows)
            realized = [D(pnl) for outcome, pnl in rows if outcome == "PAPER_DEPTH_OK" and pnl is not None]
            print(f"qty={qty:3} delay={delay:4}ms checks={len(rows):4} depth_ok={ok:4} settled_fills={len(realized):4} "
                  f"net_pnl=${sum(realized, D(0)):+.2f}")
    rejects = db.execute(
        "SELECT reason,COUNT(*) FROM entry_rejections "
        "GROUP BY reason ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("REJECTION DIAGNOSTICS (throttled log entries):", rejects)
    print("Unverified paper depth, not confirmed fills. READ ONLY.")
    db.close()


def self_test():
    b = Book()
    b.apply("orderbook_snapshot", {"no_dollars_fp": [["0.982", "40"], ["0.981", "60"]],
                                   "yes_dollars_fp": [["0.50", "1"]]})
    assert b.top("yes") == (D("0.018"), D(40))
    assert b.top("no") == (D("0.50"), D(1))
    x = b.cost("yes", 50)
    assert x and x["gross"] == D("0.91") and x["vwap"] == D("0.0182")
    assert b.cost("yes", 100) and b.cost("yes", 101) is None
    b.apply("orderbook_delta", {"side": "no", "price_dollars": "0.982", "delta_fp": "-40"})
    assert b.top("yes") == (D("0.019"), D(60))
    assert b.cost("yes", 100) is None
    assert fee(D("0.019"), 10) == D("0.02")
    # Regression: join an already-open market with too little depth, then
    # observe more depth in the same market and verify a single entry.
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    with TemporaryDirectory() as tmp:
        probe = Shadow(SimpleNamespace(
            db=Path(tmp) / "probe.sqlite3", max_book_age=5))
        close = time.time() + 215  # Subscribed 11+ minutes after open.
        ticker = "KXBTC15M-SELF-TEST"
        probe.add_watch(ticker, close)
        probe.on_book(ticker, "orderbook_snapshot", {
            "market_ticker": ticker,
            "no_dollars_fp": [["0.982", "3"]]
        })
        w = probe.watch[ticker]
        assert w.covered and not w.first_cheap, "Late snapshot rejected"
        probe.check_first_cheap(w, time.time())
        assert not w.first_cheap, "Insufficient depth locked market"
        assert probe.db.execute(
            "SELECT COUNT(*) FROM signals").fetchone()[0] == 0
        probe.on_book(ticker, "orderbook_delta", {
            "market_ticker": ticker, "side": "no",
            "price_dollars": "0.982", "delta_fp": "7"
        })
        probe.check_first_cheap(w, time.time())
        assert w.first_cheap, "Executable entry not captured"
        assert probe.db.execute(
            "SELECT COUNT(*) FROM signals").fetchone()[0] == 1
        check = probe.db.execute(
            "SELECT outcome FROM fill_checks WHERE ticker=? "
            "AND qty=10 AND delay_ms=0", (ticker,)).fetchone()
        assert check and check[0] == "PAPER_DEPTH_OK", check
        probe.check_first_cheap(w, time.time())
        assert probe.db.execute(
            "SELECT COUNT(*) FROM signals").fetchone()[0] == 1
        probe.db.close()
    print("PASS: late join, insufficient depth, later executable quote, "
          "0ms check and one signal per market")
    print("PASS: book snapshot, delta, opposite-side ask, depth aggregation, price cap, fees")


def main():
    parser = argparse.ArgumentParser(description="KXBTC15M reversal shadow, no orders")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seconds", type=float, default=0, help="0 runs until Ctrl+C")
    parser.add_argument("--discover-seconds", type=float, default=15)
    parser.add_argument("--max-book-age", type=float, default=10)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.report:
        return report(args.db)
    if args.seconds < 0 or args.discover_seconds <= 0 or args.max_book_age <= 0:
        parser.error("Bad numeric argument")
    asyncio.run(Shadow(args).run())


if __name__ == "__main__":
    main()
