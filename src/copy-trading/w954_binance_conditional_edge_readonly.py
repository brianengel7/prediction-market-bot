#!/usr/bin/env python3
"""Read-only wallet 954b conditional first-fill edge diagnostics.

Uses cached Binance BTCUSDT 1s ZIPs from w954_binance_signal_audit_readonly.py.
Does not download, write files, modify GitHub/database, or trade.

Measures whether the historical first-fill gross P&L is explained by
entry price, exposure-weighted win rate, entry time, signed Binance price
movement, and concentration. This cannot identify the trader's exact feed,
unfilled orders, or realizable edge at later executable prices.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import runpy
from statistics import median

BANDS = (("20-40c", .2, .4), ("40-60c", .4, .6))


def in_band(row, name):
    return any(label == name and lo <= row["price"] < hi for label, lo, hi in BANDS)


def describe(label, rows, min_n=1):
    if len(rows) < min_n:
        return None
    n = len(rows)
    q = sum(x["qty"] for x in rows)
    cost = sum(x["cost"] for x in rows)
    pnl = sum(x["pnl"] for x in rows)
    won_qty = sum(x["qty"] for x in rows if x["won"])
    return (f"{label:<25} N={n:>4} win={sum(x['won'] for x in rows)/n:>5.1%} "
            f"paid={100*cost/q:>5.1f}c sizeW={won_qty/q:>5.1%} "
            f"ROI={100*pnl/cost:>+7.1f}% P&L=${pnl:>+9,.2f} "
            f"edge={100*pnl/q:>+5.2f}c/share")


def signed_bucket(x, breaks):
    if x < breaks[0]:
        return f"< {breaks[0]:+g}bp"
    for low, high in zip(breaks, breaks[1:]):
        if low <= x < high:
            return f"[{low:+g},{high:+g})bp"
    return f">= {breaks[-1]:+g}bp"


def classify_lag(x):
    if x < 60: return "00-01m"
    if x < 300: return "01-05m"
    if x < 600: return "05-10m"
    if x < 840: return "10-14m"
    return "14-15m"


def daily_block_ci(rows, nboot=2000, seed=954):
    """Bootstrap observed day blocks, not individual overlapping markets."""
    days = defaultdict(list)
    for r in rows:
        days[datetime.fromtimestamp(r["ts"], timezone.utc).date()].append(r)
    blocks = list(days.values())
    if len(blocks) < 5:
        return None
    random_state = random.Random(seed)
    distribution = []
    for _ in range(nboot):
        sample = [v for _ in blocks for v in random_state.choice(blocks)]
        cost = sum(r["cost"] for r in sample)
        pnl = sum(r["pnl"] for r in sample)
        distribution.append(pnl / cost)
    distribution.sort()
    return (distribution[int(.025*nboot)], distribution[int(.975*nboot)], len(blocks))


def report(rows, warnings):
    rows.sort(key=lambda x: x["ts"])
    print("\n" + "="*112)
    print("WALLET 954B — CONDITIONAL BINANCE SIGNAL / COST / CONCENTRATION AUDIT")
    print("="*112)
    print(f"Aligned first entries: {len(rows)} | exclusions: {dict(warnings)}")
    print("Completed Binance bars at least 2 seconds before entry; no future data in features.")
    print("Binance at start-minus-2s is only a proxy for the actual Chainlink TWAP opening benchmark.")
    print("Win/loss derived from observed redemptions and previously verified no-redemption losses.")

    for label, subset in (
        ("ALL", rows),
        ("20-60c", [r for r in rows if .2 <= r["price"] < .6]),
        ("20-40c", [r for r in rows if in_band(r,"20-40c")]),
        ("40-60c", [r for r in rows if in_band(r,"40-60c")]),
        ("60-80c", [r for r in rows if .6 <= r["price"] < .8]),
        ("80c+", [r for r in rows if r["price"] >= .8]),
    ):
        line = describe(label, subset)
        if line: print(line)

    subset = [r for r in rows if .2 <= r["price"] < .6]
    if not subset:
        print("No 20-60c entries; done.")
        return

    print("\n20-60c: FEE STRESS AT ORIGINAL FILL PRICES (ILLUSTRATIVE, NOT ACTUAL FEE SCHEDULE)")
    q = sum(r["qty"] for r in subset)
    cost = sum(r["cost"] for r in subset)
    pnl = sum(r["pnl"] for r in subset)
    for per_share in (0., .005, .01, .02):
        fee = per_share*q
        print(f"fee={100*per_share:>4.1f}c per bought share | net=${pnl-fee:>+9,.2f} | ROI={(pnl-fee)/(cost+fee):>+7.2%}")

    ci = daily_block_ci(subset)
    if ci:
        print(f"20-60c day-block bootstrap gross ROI (exploratory 95% interval): [{ci[0]:+.1%}, {ci[1]:+.1%}], observed days={ci[2]}")

    mid = len(rows)//2
    midpoint = rows[mid]["ts"]
    for label, ss in (("EARLY", [r for r in subset if r["ts"] < midpoint]),
                      ("LATE", [r for r in subset if r["ts"] >= midpoint])):
        print(describe("20-60c "+label,ss) or "No entries")

    for name, key, breaks in (
        ("SIGNED BINANCE MOVE SINCE MARKET OPEN", "signed_open", (-20,-10,-3,3,10,20)),
        ("SIGNED BINANCE LAST 60s MOVE", "signed_r60", (-5,-2,-.5,.5,2,5)),
        ("SIGNED BINANCE LAST 300s MOVE", "signed_r300", (-15,-5,-1,1,5,15)),
    ):
        print("\n20-60c: "+name)
        groups = defaultdict(list)
        for r in subset:
            groups[signed_bucket(r[key], breaks)].append(r)
        for label in sorted(groups, key=lambda v: sum(r[key] for r in groups[v])/len(groups[v])):
            print(describe(label, groups[label]) or "")

    print("\n20-60c: ENTRY TIMING")
    groups = defaultdict(list)
    for r in subset: groups[classify_lag(r["ts"]-r["start"])].append(r)
    for label in sorted(groups): print(describe(label, groups[label]))

    print("\n20-60c: FIRST-SECOND COST SIZING")
    groups = defaultdict(list)
    for r in subset:
        cost = r["cost"]
        band = ("<=$30" if cost <= 30 else "$30-50" if cost<=50 else "$50-100" if cost<=100 else "$100-250" if cost<=250 else ">$250")
        groups[band].append(r)
    for label in ("<=$30", "$30-50", "$50-100", "$100-250", ">$250"):
        if label in groups: print(describe(label, groups[label]))

    print("\n20-60c: DAILY GROSS P&L")
    days = defaultdict(list)
    for r in subset:
        days[str(datetime.fromtimestamp(r["ts"],timezone.utc).date())].append(r)
    for day, rs in sorted(days.items()): print(describe(day,rs))
    print(f"20-60c positive days: {sum(sum(r['pnl'] for r in rs)>0 for rs in days.values())}/{len(days)}")

    print("\n20-60c: PROFIT CONCENTRATION")
    top = sorted(subset, key=lambda r:r["pnl"], reverse=True)
    for k in (1,3,5,10,20):
        remaining = top[k:]
        print(describe(f"Excluding top {k} winners",remaining))
    print("Largest observed first-second wins:")
    for r in top[:5]:
        dt=datetime.fromtimestamp(r["ts"],timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        print(f" {dt} price={r['price']:.3f} first_cost=${r['cost']:.2f} first_P&L=${r['pnl']:+.2f} "
              f"signed_open={r['signed_open']:+.1f}bp signed_r60={r['signed_r60']:+.1f}bp")

    print("\nIMPORTANT: The entry-price band is chosen using the wallet's realized fill; this is not an")
    print("independently discoverable trading signal. No unfilled orders or contemporaneous book depth.")
    print("Non-overlapping markets can have correlated outcomes. Bootstraps are exploratory, not a")
    print("randomized-control comparison or prospective holdout. No direct evidence of Binance feed use.")
    print("READ ONLY — consumes existing ZIP cache/JSONL; no network, writes, trades, database or GitHub changes.")


def self_test():
    rows=[{"ts":1,"price":.3,"qty":100,"cost":30,"pnl":70,"won":True},
          {"ts":2,"price":.4,"qty":100,"cost":40,"pnl":-40,"won":False}]
    assert "ROI=" in describe("test",rows)
    assert signed_bucket(-8,(-20,-10,-3,3,10,20))=="[-10,-3)bp"
    assert classify_lag(30)=="00-01m" and classify_lag(300)=="05-10m"
    print("PASS: outcome-weighted ROI, signed-price bins, entry timing")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-script",default="src/copy-trading/w954_binance_signal_audit_readonly.py")
    ap.add_argument("--start-date",default="2026-10-06")
    ap.add_argument("--end-date",default="2026-11-05")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:
        self_test()
        return
    source_path=Path(args.source_script)
    if not source_path.is_file():
        ap.error(f"Original first-entry Binance script missing: {source_path}")
    source=runpy.run_path(str(source_path))
    first, skips=source["load_wallet"]()
    selected=defaultdict(list)
    for r in first:
        day=datetime.fromtimestamp(r["ts"],timezone.utc).date()
        if args.start_date<=str(day)<=args.end_date:
            selected[day].append(r)
    rows=[]
    for day in sorted(selected):
        before=day-timedelta(days=1)
        try:
            px=source["bars"](str(before),False)
            px.update(source["bars"](str(day),False))
        except FileNotFoundError as exc:
            raise SystemExit(f"Missing cached file: {exc}. Run original audit for date range first.")
        secs=sorted(px)
        for r in selected[day]:
            x=source["features"](r,secs,px,2)
            if x is None:
                skips["binance_missing_history"]+=1
                continue
            sign=1 if x["side"]=="up" else -1
            x["signed_open"]=sign*x["open_bps"]
            x["signed_r60"]=sign*x["r60"]
            x["signed_r300"]=sign*x["r300"]
            rows.append(x)
        print(f"{day}: {len(selected[day])} wallet markets, {len(px):,} Binance seconds loaded",flush=True)
    report(rows,skips)

if __name__=="__main__":
    main()
