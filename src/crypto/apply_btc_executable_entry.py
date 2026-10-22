from pathlib import Path
from datetime import datetime, timezone
import shutil

p = Path('src/crypto/btc_reversal_shadow_v1.py')
if not p.is_file():
    raise SystemExit(f'ABORT: missing {p}')
s = p.read_text()
if 'entry_rejections (' in s or 'FIRST_EXECUTABLE_10_V3' in s:
    raise SystemExit('ABORT: patch already applied; will not apply twice')
# Last live change must be present; do not accidentally reinstate full-history gating.
add = s.split('    def add_watch(self, ticker, close):', 1)[-1].split('    def ', 1)[0]
book = s.split('    def on_book(self, ticker, kind, data):', 1)[-1].split('    def ', 1)[0]
if '"SELECT 1 FROM signals' not in add or 'kind == "orderbook_snapshot"' not in book or 'now <= w.start' in book:
    raise SystemExit('ABORT: late-market-join patch not detected; no files changed')


def substitute_once(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit(f'ABORT: expected one occurrence of {old[:75]!r}, got {s.count(old)}')
    s = s.replace(old, new, 1)


def replace_between(start, end, body):
    global s
    if s.count(start) != 1 or s.count(end) != 1:
        raise SystemExit(f'ABORT: unexpected methods: {start!r}, {end!r}')
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + body.rstrip() + '\n\n' + s[j:]

substitute_once('SIZES = (10, 50, 100)', 'SIZES = (10, 50, 100)\nPRIMARY_SIZE = 10  # First executable entry must have this much depth')
substitute_once('DEFAULT_DB = Path(".collector_runs/btc_reversal_shadow_v2.sqlite3")',
                'DEFAULT_DB = Path(".collector_runs/btc_reversal_shadow_v3.sqlite3")')
substitute_once('        self.signals = 0\n', '''        self.signals = 0
        self.rejection_seen = {}  # Throttle rejection logs by reason/market
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS entry_rejections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL, observed_utc TEXT NOT NULL,
                seconds_left REAL NOT NULL, side TEXT NOT NULL,
                best_ask TEXT, available_at_cap TEXT,
                reason TEXT NOT NULL
            )""")
''')

replacement = '''    def reject_entry(self, w: Watch, now: float, side: str,
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
'''
replace_between('    def check_first_cheap(self, w: Watch, now: float):\n',
                '    def evaluate_pending(self):\n', replacement)

replacement = '''    def evaluate_pending(self):
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
'''
replace_between('    def evaluate_pending(self):\n', '    def sample(self, w: Watch, now: float):\n', replacement)

# The report's existing metrics are preserved. Add a short rejection breakdown.
substitute_once('    print("Unverified paper depth, not confirmed fills. READ ONLY.")', '''    rejects = db.execute(
        "SELECT reason,COUNT(*) FROM entry_rejections "
        "GROUP BY reason ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("REJECTION DIAGNOSTICS (throttled log entries):", rejects)
    print("Unverified paper depth, not confirmed fills. READ ONLY.")''')

substitute_once('    print("PASS: book snapshot, delta, opposite-side ask, depth aggregation, price cap, fees")', '''    # Regression: join an already-open market with too little depth, then
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
    print("PASS: book snapshot, delta, opposite-side ask, depth aggregation, price cap, fees")''')

compile(s, str(p), 'exec')
backup_dir = Path('.collector_runs/backups')
backup_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = backup_dir / f'btc_reversal_shadow_before_executable_v3_{stamp}.py'
if backup.exists():
    raise SystemExit(f'ABORT: backup exists: {backup}')
shutil.copy2(p, backup)
p.write_text(s)
print('PATCHED', p)
print('BACKUP', backup)
print('DEFAULT_DB', '.collector_runs/btc_reversal_shadow_v3.sqlite3')
print('READ-ONLY PAPER TRADER — NO ORDERS')
