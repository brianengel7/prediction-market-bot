#!/usr/bin/env python3
"""Wallet 954b: read-only Binance BTCUSDT spot 1s signal attribution.

Reads existing wallet JSONL under ~/poly_wallet_comparison_20261105/wallet_954b.
Downloads and caches Binance daily public ZIPs ONLY under ~/wallet_954b_spot_research.
Does NOT modify the repository, database, GitHub, or submit trades.
"""
import argparse
import bisect
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import re
import shutil
import statistics
from urllib.request import Request, urlopen
from zipfile import ZipFile
from zoneinfo import ZoneInfo
import json

ET = ZoneInfo("America/New_York")
ROOT = Path.home() / "poly_wallet_comparison_20261105" / "wallet_954b"
CACHE = Path.home() / "wallet_954b_spot_research" / "binance_spot_1s"
URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{day}.zip"
TITLE = re.compile(r"Bitcoin Up or Down\s*-\s*([A-Za-z]+\s+\d{1,2}),\s*(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)\s*ET", re.I)
HORIZONS = (5, 15, 30, 60, 120, 300)


def records(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def market_start(title, entry_ts):
    m = TITLE.search(title)
    if not m:
        return None
    date_text, clock, _ = m.groups()
    try:
        naive = datetime.strptime(f"{date_text} 2026 {clock.replace(' ', '')}", "%B %d %Y %I:%M%p")
    except ValueError:
        return None
    candidates = {int(naive.replace(tzinfo=ET, fold=fold).timestamp()) for fold in (0, 1)}
    valid = [s for s in candidates if 0 <= entry_ts - s < 900]
    return valid[0] if len(valid) == 1 else None


def load_wallet():
    markets = defaultdict(lambda: {"buys": [], "winners": set(), "title": ""})
    for t in records(ROOT / "trades.jsonl"):
        if str(t.get("side") or "").upper() != "BUY":
            continue
        cid = str(t.get("condition_id") or "").lower()
        side = str(t.get("outcome") or "").lower()
        try:
            ts, price, qty = float(t["timestamp"]), float(t["price"]), float(t["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if cid and side in ("up", "down") and ts > 0 and 0 < price < 1 and qty > 0:
            markets[cid]["buys"].append((ts, side, price, qty))
            markets[cid]["title"] = str(t.get("title") or "")
    for a in records(ROOT / "activity.jsonl"):
        if str(a.get("type") or "").upper() != "REDEEM":
            continue
        cid = str(a.get("condition_id") or "").lower()
        if cid not in markets:
            continue
        try:
            cash = float(a.get("usdc_size") or 0)
        except (TypeError, ValueError):
            continue
        side = str(a.get("outcome") or "").lower()
        if cash > 0 and side in ("up", "down"):
            markets[cid]["winners"].add(side)

    result, skipped = [], Counter()
    for cid, m in markets.items():
        trades = m["buys"]
        if not trades:
            continue
        first_ts = min(x[0] for x in trades)
        first = [t for t in trades if t[0] == first_ts]
        sides = {t[1] for t in first}
        if len(sides) != 1:
            skipped["ambiguous_first_second"] += 1
            continue
        start = market_start(m["title"], first_ts)
        if start is None:
            skipped["invalid_market_time_or_DST"] += 1
            continue
        if len(m["winners"]) > 1:
            skipped["ambiguous_settlement"] += 1
            continue
        qty = sum(t[3] for t in first)
        cost = sum(t[3]*t[2] for t in first)
        side = next(iter(sides))
        first_won = side in m["winners"]  # Missing redemptions were resolved in the prior Gamma audit.
        result.append({
            "cid": cid, "ts": int(first_ts), "start": start,
            "side": side, "price": cost/qty, "qty": qty, "cost": cost,
            "won": first_won, "pnl": (qty if first_won else 0)-cost,
            "title": m["title"],
        })
    return sorted(result, key=lambda r: r["ts"]), skipped


def archive(day, download):
    name = f"BTCUSDT-1s-{day}.zip"
    path = CACHE / name
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(f"Missing {path}; rerun without --offline")
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading Binance public spot {day} ...", flush=True)
    temp = path.with_suffix(".part")
    try:
        req = Request(URL.format(day=day), headers={"User-Agent": "wallet-954b-research/1.0"})
        with urlopen(req, timeout=90) as response, temp.open("wb") as file:
            shutil.copyfileobj(response, file)
        with ZipFile(temp) as zf:
            if zf.testzip() is not None:
                raise ValueError(f"Corrupted ZIP: {temp}")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def bars(day, download):
    path = archive(day, download)
    data = {}
    with ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV in {path}")
        with zf.open(names[0]) as raw:
            import io
            for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8")):
                if len(row) < 7:
                    continue
                try:
                    timestamp = int(row[0])
                    close = float(row[4])
                except (ValueError, TypeError):
                    continue  # optional header
                # Spot Binance 2026 archives have microsecond timestamps.
                second = timestamp // (1_000_000 if timestamp >= 10**15 else 1000)
                if close > 0:
                    data[second] = close
    return data


def at(seconds, prices, t, max_age=5):
    i = bisect.bisect_right(seconds, t)-1
    if i < 0 or t-seconds[i] > max_age:
        return None
    return prices[seconds[i]]


def features(r, seconds, prices, safety_lag):
    # Binance 1s candle at second S closes at S+1; never use S >= entry-second.
    entry_sec = r["ts"]-safety_lag
    entry = at(seconds, prices, entry_sec)
    market_open_proxy = at(seconds, prices, r["start"]-safety_lag)
    if entry is None or market_open_proxy is None:
        return None
    x = dict(r)
    x["open_bps"] = (entry/market_open_proxy-1)*10000
    x["remaining"] = r["start"]+900-r["ts"]
    for h in HORIZONS:
        older = at(seconds, prices, entry_sec-h)
        if older is None:
            return None
        x[f"r{h}"] = (entry/older-1)*10000
    return x


def summary(label, sample):
    if not sample:
        return f"{label:<25} N=0"
    cost = sum(r["cost"] for r in sample)
    pnl = sum(r["pnl"] for r in sample)
    n = len(sample)
    return (f"{label:<25} N={n:>4}  wallet_up={sum(r['side']=='up' for r in sample)/n:>6.1%}  "
            f"first_win={sum(r['won'] for r in sample)/n:>6.1%}  "
            f"firstROI={pnl/cost:>+7.1%}  firstP&L=${pnl:>+9,.2f}")


def report(rows, skipped):
    print("\n" + "="*100)
    print("WALLET 954B — BINANCE BTCUSDT 1S SIGNAL ATTRIBUTION (NO LOOKAHEAD)")
    print("="*100)
    print("First entries aligned:", len(rows), "| Not aligned:", dict(skipped))
    print("No-redemption first-side losses use the previously Gamma-verified 215-market check.")
    print("Trade timestamps have second resolution; using completed pre-entry candles only.")
    print("Binance market-open price is a PROXY, not Polymarket's Chainlink TWAP strike.\n")
    print(summary("ALL", rows))
    rows = sorted(rows, key=lambda r: r["ts"])
    mid = len(rows)//2
    print(summary("CHRONO EARLY", rows[:mid]))
    print(summary("CHRONO LATE", rows[mid:]))

    for group, sample in (
        ("20-60c first price", [r for r in rows if .20 <= r["price"] < .60]),
        ("60-80c first price", [r for r in rows if .60 <= r["price"] < .80]),
        ("80c+ first price", [r for r in rows if r["price"] >= .80]),
    ):
        print("\n"+group+":", summary("ALL", sample))
        if not sample: continue
        for h in ("open_bps", "r5", "r15", "r30", "r60", "r120", "r300"):
            usable = [r for r in sample if abs(r[h]) >= 0.5]  # ignore near-zero (<0.5bps) noise
            along = [r for r in usable if (r[h] > 0) == (r["side"] == "up")]
            against = [r for r in usable if (r[h] > 0) != (r["side"] == "up")]
            pct = len(along)/len(usable) if usable else 0
            print(f"  {h:<9}: sign agreement={pct:>6.1%} (N={len(usable):>4})  "
                  f"WITH trend: {summary('', along).strip()} | AGAINST: {summary('', against).strip()}")
    print("\nINTERPRETATION: Compare Binance momentum to the wallet's FIRST side, not to future candles.")
    print("A correlation does NOT identify the data source; test Coinbase, futures and Chainlink separately.")
    print("Gross P&L is at wallet's historical first-second fill prices, before fees and slippage.")
    print("READ ONLY: public archive GETs; ZIP cache outside repo; no trades, DB or GitHub writes.")


def self_test():
    assert market_start("Bitcoin Up or Down - November 3, 2:00PM-2:15PM ET",
                        int(datetime(2026, 11, 3, 19, 7, tzinfo=timezone.utc).timestamp())) == int(datetime(2026, 11, 3, 19, tzinfo=timezone.utc).timestamp())
    start = 1000
    ts = 1070
    price = {s:100+s/100 for s in range(500, 1070)}
    secs=sorted(price)
    r={"start": start, "ts": ts, "side":"up", "cost": 10, "pnl": 5, "won": True}
    x=features(r,secs,price,2)
    assert x and x["open_bps"]>0 and x["r5"]>0
    assert at(secs,price,1076) is None
    print("PASS: timestamp alignment, finished-bar lag, directional features")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-11-03")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--safety-lag-seconds",type=int,default=2)
    parser.add_argument("--offline",action="store_true",help="Use cached ZIP files only")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    start=datetime.strptime(args.start_date,"%Y-%m-%d").date()
    end=datetime.strptime(args.end_date or args.start_date,"%Y-%m-%d").date()
    if end<start or args.safety_lag_seconds < 2:
        parser.error("Date range invalid, or safety lag <2 seconds")
    wallet,pre_skipped=load_wallet()
    groups=defaultdict(list)
    for r in wallet:
        day=datetime.fromtimestamp(r["ts"],timezone.utc).date()
        if start<=day<=end:
            groups[day].append(r)
    print(f"Selected {sum(map(len,groups.values()))} first entries from wallet JSONL. Cache: {CACHE}")
    aligned=[]
    skipped=Counter(pre_skipped)
    for i in range((end-start).days+1):
        day=start+timedelta(days=i)
        batch=groups.get(day, [])
        if not batch:
            continue
        # Start reference and 300-second lookback may fall on previous UTC day.
        try:
            prices={}
            for d in (day-timedelta(days=1), day):
                prices.update(bars(str(d),not args.offline))
            seconds=sorted(prices)
        except Exception as e:
            print(f"FAILED {day}: {type(e).__name__}: {e}")
            skipped["missing_archive_day"]+=len(batch)
            continue
        covered=0
        for r in batch:
            x=features(r,seconds,prices,args.safety_lag_seconds)
            if x:
                aligned.append(x)
                covered+=1
            else:
                skipped["missing_or_stale_spot_seconds"]+=1
        print(f"{day}: {covered}/{len(batch)} first entries aligned ({len(prices):,} 1s candles loaded)")
    report(aligned,skipped)

if __name__=="__main__":
    main()
