#!/usr/bin/env python3
"""Wallet 954B: ALL_STOP_ON_OPPOSITE cap sweep and untouched November replay.

READ ONLY: local research scripts + public HTTPS GET; no file writes, no POST,
no trading, no modifications to the earlier scripts.

Put in src/copy-trading/ beside the existing w954_* research scripts.

1) --mode caps: compare the original FIRST+CONFIRM+0 and ALL_STOP_ON_OPPOSITE
   with cap 50/100/150/250 on identical OLD wallet first-entry cohort.
2) --mode oos: reconstruct NEW wallet signals directly from Polymarket's
   /v2/trades feed, Nov 6 onward, bypassing the old first-fill loader. Do not
   reuse training signals from w954_kalshi_crossvenue_backtest_readonly.py.

Historical Kalshi buyer-aggressor prints are NOT executable asks/depth.
Polymarket trade timestamps are NOT guaranteed wallet-observation timestamps.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import runpy
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
WALLET = "0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0"
TRADE_URL = "https://data-api.polymarket.com/v2/trades"
NATIVE = HERE / "w954_first_confirm_all_subsequent_readonly.py"
POST = HERE / "w954_post_entry_sequence_audit_readonly.py"
FOLLOW = HERE / "w954_followup_copy_window_audit_readonly.py"


def modules():
    for path in (NATIVE, POST, FOLLOW):
        if not path.is_file():
            raise RuntimeError(f"Missing companion script: {path}")
    m = runpy.run_path(str(NATIVE))
    p, core, af = m["libs"]()
    a = runpy.run_path(str(POST))
    follow = runpy.run_path(str(FOLLOW))
    return m, p, core, af, a, follow


def params(args, cap):
    return SimpleNamespace(
        min_repeat_age=4, max_repeat_age=30,
        min_add_gap=4, add_window=120, max_market_qty=cap,
        sizing="proportional", base_qty=args.base_qty, add_qty=10,
        scale=.1, latencies=args.latencies, max_wait_s=args.max_wait_s,
        max_quote_age_s=2., fee_multiplier=1., merge_gap_s=15,
        trades_file=args.trades_file, offline=args.offline,
    )


def sig_from_training(core, args):
    # ONLY train mode may call the historical first-fill loader.
    s = core["extract_signals"](SimpleNamespace(
        wallet=args.wallet, strategy="first", upper_first_price=.40,
        min_repeat_age=4, max_repeat_age=30))
    return s


def date_epoch(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def get_public(url, query, retries=5):
    address = url + "?" + urlencode(query)
    for attempt in range(retries):
        try:
            request = Request(address, headers={
                "Accept": "application/json", "User-Agent": "w954-historical-replay-readonly/1.0"})
            with urlopen(request, timeout=45) as response:
                body = json.load(response)
            if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                raise RuntimeError("Unrecognized Polymarket v2 feed: expected {data:[...],pagination:{...}}")
            return body
        except (HTTPError, URLError, TimeoutError) as err:
            if isinstance(err, HTTPError) and err.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"GET Polymarket failed with HTTP {err.code}: {err.reason}") from err
            if attempt + 1 == retries:
                raise RuntimeError(f"Polymarket GET failed after retries: {err}") from err
            wait = min(20., 2.**attempt)
            if isinstance(err, HTTPError) and err.headers:
                try: wait = min(30., max(wait, float(err.headers.get("Retry-After", wait))))
                except ValueError: pass
            time.sleep(wait)
    raise AssertionError("unreachable")


def polymarket_trades(start, end, max_pages=100):
    """Fetch ALL trades with maker/taker visibility; fail closed if pagination incomplete."""
    q = dict(user=WALLET, start=start, end=end - 1, limit=500, taker_only="false")
    cursor = None
    seen = set()
    rows = []
    for page in range(1, max_pages + 1):
        query = dict(q)
        if cursor:
            query["cursor"] = cursor
        payload = get_public(TRADE_URL, query)
        got = payload["data"]
        rows.extend(got)
        pg = payload.get("pagination") or {}
        next_cursor = pg.get("next_cursor") or pg.get("nextCursor")
        has_more = pg.get("has_more")
        print(f"  Polymarket trade page {page}: {len(got)} records, cumulative={len(rows)}", flush=True)
        if not next_cursor:
            if has_more is True:
                raise RuntimeError("Feed says has_more=true without next_cursor")
            return rows
        if not got or next_cursor in seen:
            raise RuntimeError("Polymarket pagination stalled; refusing partial data")
        seen.add(next_cursor)
        cursor = next_cursor
    raise RuntimeError("Polymarket pagination exceeded --max-pages; refusing partial backtest")


def decode_wallet(rows, a, begin, end, now):
    """Normalize source trades without disk writes or using the old first-fill cache."""
    grouped = defaultdict(list)
    missed = Counter()
    btc_seen = 0
    duplicates = 0
    unique = set()
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict):
            missed["non_object"] += 1
            continue
        slug = next((str(r.get(k)) for k in ("slug", "event_slug", "eventSlug", "market_slug", "title")
                     if isinstance(r.get(k), str) and "btc-updown-15m-" in r[k]), "")
        if not re.search(r"btc-updown-15m-\d{10}", slug, re.I):
            missed["other_market"] += 1
            continue
        btc_seen += 1
        source_wallet = a["first_present"](r, ("proxy_wallet", "proxyWallet", "user", "wallet"))
        if isinstance(source_wallet, str) and re.fullmatch(r"0x[0-9a-fA-F]{40}", source_wallet):
            if source_wallet.lower() != WALLET:
                raise RuntimeError("Polymarket feed returned a different wallet; refusing mixed-user OOS")
        ts = a["epoch"](a["first_present"](r, ("timestamp", "ts", "time", "created_at")))
        start, confirmed = a["key_from_trade"](r, ts)
        action = str(a["first_present"](r, ("side", "action", "type")) or "").upper()
        side = a["outcome"](a["first_present"](r, ("outcome", "outcome_name", "token_outcome")))
        price = a["fnum"](a["first_present"](r, ("price", "fill_price")))
        qty = a["fnum"](a["first_present"](r, ("size", "shares", "quantity", "qty")))
        if (ts is None or not confirmed or start is None or
            not begin <= ts < end or not begin <= start < end or
            not start <= ts < start + 900 or ts > now or
            action not in ("BUY", "SELL") or side is None or
            price is None or not 0 < price < 1 or qty is None or qty <= 0):
            missed["invalid_btc_record"] += 1
            continue
        # API rows may represent distinct fills of identical size in the same
        # second; ONLY dedupe on a documented unique trade id when present.
        identity = r.get("trade_id") or r.get("tradeId") or r.get("id")
        if identity:
            key = (str(identity), start, action, side)
            if key in unique:
                duplicates += 1
                continue
            unique.add(key)
        grouped[start].append(dict(ts=ts, side=side, action=action,
                                   price=price, size=qty, notional=qty*price,
                                   line=i, start=start))
    for market in grouped.values():
        market.sort(key=lambda x: (x["ts"], x["line"]))
    if missed["invalid_btc_record"]:
        raise RuntimeError(f"{missed['invalid_btc_record']} BTC feed records could not be safely decoded. "
                           f"Other exclusions={dict(missed)}. Check feed schema; no silent truncation.")
    if btc_seen and not grouped:
        raise RuntimeError("BTC records present but none parsed")
    return grouped, missed, duplicates


def derive_first(grouped, follow):
    # Entirely NEW source; the old file's first-fill loader must NOT be used.
    initial = []
    for start, records in sorted(grouped.items()):
        r = next((x for x in records if x["action"] == "BUY"), None)
        if r is None:
            continue
        # The Polymarket result does not enter Kalshi settlement calculations;
        # 'won' is a placeholder needed only by the descriptive join helper.
        initial.append(dict(start=start, ts=r["ts"], price=r["price"],
                            side=r["side"], won=False))
    joined, bad = follow["build"](initial, grouped, {"outcome": lambda x: x})
    filtered = [r for r in joined if .20 <= r["price"] < .40]
    signals = sorted([{**r, "signal_ts": r["ts"], "repeat_price": r["price"],
                       "signal_age": 0} for r in filtered], key=lambda x: x["signal_ts"])
    return signals, initial, bad


def calibrate_old_first(a, follow, core):
    """Optional audit: compare direct-first construction with legacy first-fill loader.

    Do not let this audit contaminate the OOS signals. Used to disclose
    any source-definition disagreement, not to choose OOS winners.
    """
    path = next((f for f in a["DEFAULT_PATHS"] if f.is_file()), None)
    if not path:
        print("  CALIBRATION: old cache unavailable; live OOS first-fill definition is uncalibrated")
        return
    old_grouped, raw, invalid, confirmed, collisions = a["load_trades"](path)
    direct, _, _ = derive_first(old_grouped, follow)
    ref = core["extract_signals"](SimpleNamespace(
        wallet=path, strategy="first", upper_first_price=.40,
        min_repeat_age=4, max_repeat_age=30))
    direct_by = {r["start"]: r for r in direct}
    ref_by = {r["start"]: r for r in ref}
    overlap = set(direct_by) & set(ref_by)
    mismatched = [k for k in overlap if
                  (direct_by[k]["side"] != ref_by[k]["side"] or
                   abs(direct_by[k]["price"] - ref_by[k]["price"]) > .015 or
                   abs(direct_by[k]["ts"] - ref_by[k]["ts"]) > 3)]
    print(f"  CALIBRATION: legacy first-qualified={len(ref)}, direct first-qualified={len(direct)}, "
          f"same market={len(overlap)}, mismatched price/side/clock={len(mismatched)}, "
          f"direct-only={len(set(direct_by)-set(ref_by))}, "
          f"legacy-only={len(set(ref_by)-set(direct_by))}")
    if len(overlap) < .95*len(ref) or mismatched:
        print("  CALIBRATION WARNING: live OOS first-fill selection is NOT fully equivalent "
              "to legacy first-fill loader. Compare cohorts before treating results as directly comparable.")


def markets_and_prints(variants, args, p, core, af):
    all_order = [x for events in variants.values() for x in events]
    if not all_order:
        return {}, None, Counter()
    unique_starts = sorted({e["start"] for e in all_order})
    stub = [{"start": start} for start in unique_starts]
    src = list(core["objrows"](args.markets_file)) if args.markets_file else core["markets_api"](stub)
    by_close = defaultdict(list)
    for m in src:
        close = core["market_close"](m)
        if close is not None and str(m.get("ticker", "")).startswith("KXBTC15M-"):
            by_close[round(close)].append(m)
    paired = {}
    missing = Counter()
    cohorts = None
    for name, events in variants.items():
        mapped, excluded = p["attach_market"](events, by_close)
        paired[name] = mapped
        missing.update(excluded)
        current = {x["ticker"] for x in mapped if x["kind"] == "first"}
        if cohorts is None:
            cohorts = current
        elif current != cohorts:
            raise RuntimeError(f"Non-identical first-entry market coverage: {name}")
        print(f"  {name:<29} mapped={len(mapped):>5}/{len(events):<5} "
              f"markets={len(current):>3} excludes={dict(excluded)}")
    if not any(paired.values()):
        print("NO MATCHED SETTLED KALSHI MARKETS. No P&L to report.")
        return paired, None, missing
    load_cfg = params(args, max(args.caps) if args.mode == "caps" else args.cap)
    print("Loading buyer-aggressor Kalshi trade prints once for all variants...", flush=True)
    all_events = [e for chunk in paired.values() for e in chunk]
    quotes = af["prep_prints"](af["load_trade_prints"](all_events, load_cfg, core))
    return paired, quotes, missing


def evaluate(variants, args, p, core, af):
    paired, pp, missing = markets_and_prints(variants, args, p, core, af)
    if pp is None:
        return
    sim = SimpleNamespace(quantity_mode="wallet", scale=1., qty=1, max_qty=None,
                          max_wait_s=args.max_wait_s, max_quote_age_s=2., fee_multiplier=1.)
    for latency in args.latencies:
        print(f"\nSIMULATED {latency}s latency | actual asks NOT verified")
        print("NAME                           MARKETS FILLS        P&L ($)      COST ($)     ROI     +2c ROI   +5c ROI  TOP5-EXCL")
        print("-"*124)
        outcome = {}
        for name, ev in variants.items():
            mapped = paired[name]
            ex, rows, rejected, dep = p["simulate_dependent"](mapped, sim, latency, af, pp, None)
            pnl = sum(r["pnl"] for r in rows)
            cost = sum(r["buy_cost"] for r in rows)
            stress = p["run_price_stress"](ex, rows, [0., 2., 5.], af["fee"], 1.)
            ranked = sorted(rows, key=lambda r: r["pnl"], reverse=True)
            rest = ranked[5:]
            rp, rc = sum(r["pnl"] for r in rest), sum(r["buy_cost"] for r in rest)
            roi = 100*pnl/cost if cost else float("nan")
            rest_roi = 100*rp/rc if rc else float("nan")
            outcome[name] = rows
            print(f"{name:<29} {len(rows):>3}  {len(ex):>4}/{len(ev):<4}   "
                  f"${pnl:>+10.2f}  ${cost:>11.2f} {roi:>+7.2f}% "
                  f"{stress[1][3]:>+8.2f}% {stress[2][3]:>+8.2f}% {rest_roi:>+8.2f}%")
            if rejected or dep:
                print(f"  proxy rejects={dict(rejected)} dependent skips={dep}")
            print(f"  weeks: {m_weekly(rows)}")
        base = "FIRST+CONFIRM+0"
        if base in outcome:
            for name in outcome:
                if name == base: continue
                try:
                    dp, dc, inc = p["compare"](outcome[base], outcome[name])
                except RuntimeError:
                    continue
                ci = p["bootstrap_increment"](outcome[base], outcome[name], args.bootstrap, 5917)
                ci_txt = f" day-bootstrap CI [${ci[0]:+.2f},${ci[1]:+.2f}]" if ci else ""
                print(f"  {name} vs CONFIRM-ONLY: extra P&L ${dp:+.2f}, "
                      f"extra outlay ${dc:+.2f}, incremental ROI "
                      f"{inc:+.2f}%" if inc is not None else
                      f"  {name} vs CONFIRM-ONLY: extra P&L ${dp:+.2f}", end="")
                print(ci_txt)


def m_weekly(markets):
    week = defaultdict(lambda: [0., 0., 0])
    for x in markets:
        key = datetime.fromtimestamp(x["start"], timezone.utc).strftime("%G-W%V")
        week[key][0] += x["pnl"]
        week[key][1] += x["buy_cost"]
        week[key][2] += 1
    return ", ".join(f"{k}: {v[2]}m {100*v[0]/v[1]:+.1f}%" for k,v in sorted(week.items()) if v[1]) or "none"


def self_test():
    # Only synthetic local logic, no HTTP, no private wallet source needed.
    m,p,core,af,a,follow = modules()
    m["test"]()
    origin = 1793923200  # Nov 6, 2026 00:00 UTC
    start = origin + 900
    wallet = WALLET
    dummy = [
        {"slug":f"btc-updown-15m-{start}", "timestamp":start+5,"price":.3,"size":100,
         "side":"BUY","outcome":"Up","proxy_wallet":wallet,"trade_id":"one"},
        {"slug":f"btc-updown-15m-{start}", "timestamp":start+12,"price":.4,"size":50,
         "side":"BUY","outcome":"Up","proxy_wallet":wallet,"trade_id":"two"},
        {"slug":f"btc-updown-15m-{start}", "timestamp":start+13,"price":.44,"size":25,
         "side":"BUY","outcome":"Up","proxy_wallet":wallet,"trade_id":"three"},
        {"slug":f"btc-updown-15m-{start}", "timestamp":start+20,"price":.5,"size":25,
         "side":"BUY","outcome":"Down","proxy_wallet":wallet,"trade_id":"four"},
        {"slug":f"btc-updown-15m-{start}", "timestamp":start+30,"price":.45,"size":25,
         "side":"BUY","outcome":"Up","proxy_wallet":wallet,"trade_id":"five"},
    ]
    grouped, missed, dup = decode_wallet(dummy, a, origin, origin+86400, origin+2*86400)
    sigs, first, bad = derive_first(grouped, follow)
    assert len(sigs) == 1 and not bad
    assert sigs[0]["first_same"]["ts"] == start + 12
    idx = p["index_wallet"]([r for market in grouped.values() for r in market])
    cfg = SimpleNamespace(min_repeat_age=4,max_repeat_age=30,min_add_gap=4,add_window=120,
                          sizing="proportional",base_qty=10,add_qty=10,scale=.1,max_market_qty=250)
    ev,_,_,obs = m["build_all"](sigs, idx, cfg, p, stop_on_opposite=True)
    assert [(x["kind"], x["size"]) for x in ev] == [("first",10),("repeat",5),("add",3)]
    assert obs["stopped_after_opposite"] == 1
    cfg.max_market_qty=15
    capped,_,_,_ = m["build_all"](sigs,idx,cfg,p,stop_on_opposite=True)
    assert [(x["kind"],x["size"]) for x in capped] == [("first",10),("repeat",5)]
    assert date_epoch("2026-11-06") == origin
    print("PASS: OOS wallet normalization, independent causal first signal, proportional stops/caps, "
          "prior execution-volume/settlement self-test. No HTTP, writes or orders.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mode", choices=("caps","oos"), default="caps")
    ap.add_argument("--wallet", type=Path, help="Original wallet trades.jsonl for caps mode only")
    ap.add_argument("--start", default="2026-11-06", help="OOS UTC inclusive market-start date")
    ap.add_argument("--end", default="2026-11-11", help="OOS UTC exclusive market-start date")
    ap.add_argument("--caps", type=int, nargs="+", default=[50,100,150,250])
    ap.add_argument("--cap", type=int, default=250, help="OOS position cap, select before seeing OOS results")
    ap.add_argument("--base-qty", type=int, default=10)
    ap.add_argument("--latencies", type=int, nargs="+", default=[5,10,20])
    ap.add_argument("--max-wait-s", type=float, default=5.)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--max-pages", type=int, default=100)
    ap.add_argument("--markets-file", type=Path)
    ap.add_argument("--trades-file", type=Path)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    args.caps = sorted(set(args.caps))
    if not args.caps or min(args.caps) < args.base_qty or args.cap < args.base_qty or args.base_qty<1:
        ap.error("All position caps must be >= base qty > 0")
    if not args.latencies or min(args.latencies)<0 or max(args.latencies)>180 or args.max_wait_s<0:
        ap.error("Invalid latency or quote wait")
    if args.offline and not (args.markets_file and args.trades_file):
        ap.error("--offline needs --markets-file and --trades-file")
    m,p,core,af,a,follow = modules()
    if args.mode == "caps":
        first = sig_from_training(core,args)
        wallet, meta = af["prepare_wallet"](args.wallet)
        by_start = p["index_wallet"](wallet)
        # The original 250-contract FIRST+CONFIRM+0 benchmark must NOT be
        # inadvertently replaced by the lowest sweep cap.
        benchmark_cfg = params(args, max(args.caps))
        baseline, _, _ = p["build"](first,by_start,"first",0,benchmark_cfg)
        variants = {"FIRST+CONFIRM+0": baseline}
        for cap in args.caps:
            cfg=params(args,cap)
            full, skipped, _, obs = m["build_all"](first,by_start,cfg,p,stop_on_opposite=True)
            variants[f"ALL_STOP_CAP_{cap}"] = full
            print(f"CAP {cap:>3}: copied orders={len(full)} first={sum(x['kind']=='first' for x in full)}, "
                  f"repeats={sum(x['kind']=='repeat' for x in full)}, "
                  f"additional={sum(x['kind']=='add' for x in full)}, "
                  f"skipped={dict(skipped)}, observations={dict(obs)}")
        base = sorted((e["start"],e["ts"],e["line"],e["size"]) for e in variants["FIRST+CONFIRM+0"]
                      if e["kind"]=="first")
        for name,events in variants.items():
            assert sorted((e["start"],e["ts"],e["line"],e["size"]) for e in events if e["kind"]=="first") == base
        print("\n954B HISTORICAL CAP SENSITIVITY: IN-SAMPLE ONLY")
        print(f"Qualifying old first-entry signals={len(first)}")
    else:
        begin, end = date_epoch(args.start), date_epoch(args.end)
        if not begin < end:
            ap.error("--end must follow --start (exclusive)")
        if begin < date_epoch("2026-11-06"):
            ap.error("OOS must start on/after Nov 6, 2026; original research ended Nov 5")
        if args.wallet is not None:
            ap.error("--wallet (old archive) is forbidden in OOS mode")
        print("954B UNTOUCHED OOS WALLET DOWNLOAD — GET only; no files written", flush=True)
        print(f"UTC range [{args.start}, {args.end}), first-buy market starts only; "
              "unsettled markets excluded by Kalshi result")
        raw = polymarket_trades(begin, end, args.max_pages)
        grouped, missed, dup = decode_wallet(raw,a,begin,end,int(datetime.now(timezone.utc).timestamp()))
        first, initial, excluded = derive_first(grouped,follow)
        print(f"API rows={len(raw)}, BTC 15m markets={len(grouped)}, "
              f"wallet first BUYs={len(initial)}, qualifying 20–40c starts={len(first)}, "
              f"duplicates={dup}, skips={dict(missed)}, join exclusions={dict(excluded)}")
        if not args.skip_calibration:
            calibrate_old_first(a,follow,core)
        if not first:
            print("NO OOS QUALIFYING FIRST ENTRIES; report N=0, not a zero-return strategy.")
            return
        events = [r for market in grouped.values() for r in market]
        if any(r["action"]=="SELL" for r in events):
            raise RuntimeError("New wallet data CONTAINS SELLs; buy-only copy model not faithful. Stop and inspect.")
        by_start=p["index_wallet"](events)
        cfg=params(args,args.cap)
        all_stop, bad, kinds, obs=m["build_all"](first,by_start,cfg,p,stop_on_opposite=True)
        confirmed,cbad,_=p["build"](first,by_start,"first",0,cfg)
        variants={"FIRST_ONLY":[e for e in confirmed if e["kind"]=="first"],
                  "FIRST+CONFIRM+0":confirmed,
                  f"ALL_STOP_CAP_{args.cap}":all_stop}
        print(f"OOS cap={args.cap} (chosen before inspecting OOS results), "
              f"qualifying={len(first)}, confirmed={sum(x['kind']=='repeat' for x in confirmed)}, "
              f"full order count={len(all_stop)}, skips={dict(bad)} observations={dict(obs)}")
        if any(not (begin<=r["start"]<end) for rows in variants.values() for r in rows):
            raise RuntimeError("TRAINING PERIOD LEAK INTO OOS")
    evaluate(variants,args,p,core,af)
    print("\nINTERPRETATION: trade prints are NOT proven executable asks; 5s is a simulated delay "
          "after wallet trade time, not measured public-observation latency. "
          "No orders/writes; GET-only historical research.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError,ValueError,OSError,AssertionError) as exc:
        sys.exit("ABORT: " + str(exc))
