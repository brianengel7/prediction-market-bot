#!/usr/bin/env python3
"""Albert1953 wallet audit, exact desired copy ratio = 0.001 (research only).

Public Polymarket Data API v2 GET requests; NO exchange credentials or orders.
Local writes are restricted to the selected research output directory.

This is a trade-quantity and wallet-fingerprint audit, NOT a copy-trading
profit backtest. Delayed executable bid/ask, Kalshi market equivalence,
contract minimums at the destination, settlement, and fees must be tested
with independent historical data. A block timestamp has 1-second precision.

Python 3.10+; standard library only.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

WALLET = "0x777fae71d2ff9ec48a1213d48ba1d9d91024a1bb"
API = "https://data-api.polymarket.com"
RATIO = Decimal("0.001")


def get_json(path: str, params: dict, attempts: int = 6) -> dict:
    url = API + path + "?" + urlencode(params)
    for i in range(attempts):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "albert1953-readonly-audit/1.0"})
            with urlopen(req, timeout=45) as resp:
                result = json.load(resp)
            if not isinstance(result, dict) or "data" not in result:
                raise RuntimeError(f"Unexpected response shape from {path}; refuse silent empty results")
            return result
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                pause = min(60, max(1.0, float(retry_after))) if retry_after else min(45, 2 ** i)
            except ValueError:
                pause = min(45, 2 ** i)
            time.sleep(pause)
        except (URLError, TimeoutError):
            if i == attempts - 1:
                raise
            time.sleep(min(45, 2 ** i))
    raise AssertionError("unreachable")


def date_epoch(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def download_pages(path: str, base: dict, output: Path, max_pages: int) -> dict:
    partial = output.with_suffix(output.suffix + ".partial")
    n = 0
    cursor = None
    seen = set()
    pages = 0
    with partial.open("w", encoding="utf-8") as f:
        for _ in range(max_pages):
            p = dict(base)
            if cursor:
                p["cursor"] = cursor
            res = get_json(path, p)
            records = res.get("data")
            meta = res.get("pagination")
            if not isinstance(records, list) or not isinstance(meta, dict):
                raise RuntimeError(f"Unexpected paged response from {path}")
            for r in records:
                if not isinstance(r, dict):
                    raise RuntimeError("Non-dictionary record in feed")
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
                n += 1
            pages += 1
            next_cur = meta.get("next_cursor")
            if next_cur is None:
                if meta.get("has_more") is True:
                    raise RuntimeError("Incomplete cursor response: has_more=true and next_cursor=null")
                partial.replace(output)
                return {"complete": True, "rows": n, "pages": pages}
            if not isinstance(next_cur, str) or not next_cur or next_cur in seen:
                raise RuntimeError("Invalid/cycling pagination cursor")
            seen.add(next_cur)
            cursor = next_cur
            if pages % 10 == 0:
                print(f"  {path}: {n:,} rows, {pages} pages", flush=True)
            time.sleep(0.075)
    raise RuntimeError(f"Reached max_pages={max_pages}: partial data only at {partial}; refusing to score")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dec(x) -> Decimal:
    return Decimal(str(x))


def quantile(vals: list[float], f: float):
    if not vals:
        return None
    vals = sorted(vals)
    i = f * (len(vals) - 1)
    a, b = math.floor(i), math.ceil(i)
    return round(vals[a] * (b-i) + vals[b] * (i-a), 5) if a != b else round(vals[a], 5)


def pct(n, d):
    return round(100 * n / d, 3) if d else None


def kind(row: dict) -> str:
    s = " ".join(str(row.get(k) or "") for k in ("title", "slug", "event_slug")).lower()
    if re.search(r"temperature|weather|rainfall|snowfall", s):
        return "weather"
    if re.search(r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|\bxrp\b|dogecoin|hyperliquid|crypto", s):
        return "crypto"
    if re.search(r"\bnfl\b|\bnba\b|\bmlb\b|\bnhl\b|\btennis\b|\bsoccer\b|\bvs\.?\b|\bsports\b", s):
        return "sports"
    if re.search(r"election|president|senate|governor|congress|democrat|republican|vote|trump|harris", s):
        return "politics"
    if re.search(r"\bstock\b|\bshares\b|\bcrude\b|\boil\b|\bgold\b|\bsilver\b|\bwti\b|\bmeta\b|\baapl\b|\bmsft\b|\bnvda\b|\bnasdaq\b|\bs&p\b|\bipo\b|\bmarket cap\b|\bfed\b|\binflation\b|\btreasury\b", s):
        return "finance/economics"
    return "other/unclassified"


def cleaned_rows(records: list[dict], start: int, end: int) -> list[dict]:
    out = []
    for i, r in enumerate(records):
        try:
            ts = int(r["timestamp"])
            side = str(r["side"]).upper()
            price = dec(r["price"])
            q = dec(r["size"])
            token = str(r.get("token_id") or "")
            condition = str(r.get("condition_id") or "")
            address = str(r.get("proxy_wallet") or "").lower()
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise RuntimeError(f"Invalid trade record #{i}: {exc}") from exc
        if not (start <= ts < end):
            raise RuntimeError("API emitted a fill outside the selected window")
        if address and address != WALLET:
            raise RuntimeError("Wrong wallet from wallet-scoped API feed")
        if side not in ("BUY", "SELL") or not (Decimal(0) < price < Decimal(1)) or q <= 0 or not token or not condition:
            raise RuntimeError(f"Malformed trade #{i}; refuse to treat malformed records as successful fills")
        out.append({"ts": ts, "order": i, "side": side, "price": price, "q": q,
                    "notional": price*q, "token": token, "condition": condition,
                    "outcome_index": r.get("outcome_index"), "category": kind(r),
                    "title": str(r.get("title") or r.get("slug") or ""),
                    "slug": str(r.get("slug") or ""),
                    "tx": str(r.get("transaction_hash") or "")})
    # Source has block timestamps only, so same-second ordering is ambiguous.
    return sorted(out, key=lambda r: (r["ts"], r["order"]))


def clusters(rows: list[dict], gap: int) -> list[dict]:
    """Same token + side + <=gap seconds, an approximation, NOT original orders."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["condition"], r["token"], r["side"])].append(r)
    output = []
    for (condition, token, side), items in grouped.items():
        latest = None
        for r in items:
            if latest is None or r["ts"] - latest["last"] > gap:
                latest = {"first": r["ts"], "last": r["ts"], "side": side,
                          "condition": condition, "token": token, "title": r["title"],
                          "category": r["category"], "fills": 0, "q": Decimal(0),
                          "notional": Decimal(0)}
                output.append(latest)
            latest["last"] = r["ts"]
            latest["q"] += r["q"]
            latest["notional"] += r["notional"]
            latest["fills"] += 1
    return sorted(output, key=lambda x: (x["first"], x["condition"], x["token"], x["side"]))


def row_stats(rows: list[dict], name: str) -> dict:
    buys = [r for r in rows if r["side"] == "BUY"]
    sells = [r for r in rows if r["side"] == "SELL"]
    values = [float(r["notional"]) for r in rows]
    copy_shares = [r["q"]*RATIO for r in rows]
    executable_whole_share = [int(q.to_integral_value(rounding=ROUND_FLOOR)) for q in copy_shares]
    return {
        "unit": name, "count": len(rows), "buys": len(buys), "sells": len(sells),
        "median_source_notional_usdc": quantile(values, .5),
        "p90_source_notional_usdc": quantile(values, .9),
        "median_copy_notional_usdc_at_source_price": quantile([v*.001 for v in values], .5),
        "copy_buy_notional_at_source_price_usdc": round(sum((r["notional"]*RATIO for r in buys), Decimal(0)), 4),
        "copy_sell_notional_at_source_price_usdc": round(sum((r["notional"]*RATIO for r in sells), Decimal(0)), 4),
        "source_buy_notional_usdc": round(sum((r["notional"] for r in buys), Decimal(0)), 2),
        "source_sell_notional_usdc": round(sum((r["notional"] for r in sells), Decimal(0)), 2),
        "copy_qty_under_one_whole_contract_pct": pct(sum(q < 1 for q in copy_shares), len(rows)),
        "copy_notional_under_1_usdc_pct": pct(sum(v*.001 < 1 for v in values), len(rows)),
        "copy_notional_under_5_usdc_pct": pct(sum(v*.001 < 5 for v in values), len(rows)),
        "trades_with_at_least_1_full_contract": sum(x >= 1 for x in executable_whole_share),
        "unrounded_target_contracts": round(sum(copy_shares, Decimal(0)), 5),
        "per_trade_whole_contracts_if_floored": sum(executable_whole_share),
        "share_quantity_lost_if_every_trade_floored": round(sum((q - Decimal(int(q.to_integral_value(rounding=ROUND_FLOOR))) for q in copy_shares), Decimal(0)), 5),
        "caveat": "Whole-contract flooring and source-price notionals are feasibility diagnostics, not simulated fills."
    }


def category_table(rows: list[dict]) -> list[dict]:
    sums = defaultdict(lambda: {"fills": 0, "buys": 0, "sells": 0, "buy_notional": Decimal(0), "sell_notional": Decimal(0), "markets": set()})
    for r in rows:
        x = sums[r["category"]]
        x["fills"] += 1
        x[r["side"].lower()+"s"] += 1
        x[r["side"].lower()+"_notional"] += r["notional"]
        x["markets"].add(r["condition"])
    return sorted(({"category": k, "fills": v["fills"], "markets": len(v["markets"]),
                    "buy_fills": v["buys"], "sell_fills": v["sells"],
                    "buy_usdc": round(v["buy_notional"],2), "sell_usdc":round(v["sell_notional"],2),
                    "copy_buy_usdc_at_source_price":round(v["buy_notional"]*RATIO,4)}
                   for k, v in sums.items()),key=lambda r: -r["fills"])


def inventory_diagnostic(rows: list[dict]) -> dict:
    """Within-window gross FIFO matching only, omits fees, settlements and carry-in."""
    inventory = defaultdict(deque)
    unmatched_sell = Decimal(0)
    matched_sale = Decimal(0)
    matched_gross = Decimal(0)
    markets_outcomes = defaultdict(set)
    buy_both = set()
    same_second_both = 0
    by_second = defaultdict(set)
    for r in rows:
        k = (r["condition"],r["token"])
        if r["side"] == "BUY":
            by_second[(r["condition"],r["ts"])].add(r["side"])
            if r["outcome_index"] in (0, 1, "0", "1"):
                markets_outcomes[r["condition"]].add(str(r["outcome_index"]))
            inventory[k].append([r["q"],r["price"]])
        else:
            remaining = r["q"]
            while remaining > 0 and inventory[k]:
                lot = inventory[k][0]
                x = min(remaining,lot[0]); remaining -= x; lot[0] -= x
                matched_sale += x
                matched_gross += x*(r["price"]-lot[1])
                if lot[0] <= 0: inventory[k].popleft()
            unmatched_sell += remaining
    for k,v in markets_outcomes.items():
        if "0" in v and "1" in v: buy_both.add(k)
    return {"markets_with_buys_on_both_outcomes":len(buy_both),
            "markets_with_any_buys":len(markets_outcomes),
            "gross_fifo_sell_gain_from_observed_buys_ex_fees_usdc":round(matched_gross,2),
            "matched_sell_shares":round(matched_sale,5),
            "sell_shares_unmatched_to_observed_buys":round(unmatched_sell,5),
            "warning":"NOT wallet PnL: excludes settled outcomes, carry-in, fees, any ambiguous same-second sequencing, cross-market hedges and open marks."}


def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for x in rows: w.writerow(x)


def self_test():
    rows = [
        {"ts":100,"order":0,"side":"BUY","price":dec("0.5"),"q":dec("50"),"notional":dec("25"),"token":"t0","condition":"a","outcome_index":0,"category":"finance/economics","title":"A","slug":"a","tx":"x"},
        {"ts":101,"order":1,"side":"BUY","price":dec("0.5"),"q":dec("1500"),"notional":dec("750"),"token":"t0","condition":"a","outcome_index":0,"category":"finance/economics","title":"A","slug":"a","tx":"y"},
        {"ts":110,"order":2,"side":"SELL","price":dec("0.6"),"q":dec("500"),"notional":dec("300"),"token":"t0","condition":"a","outcome_index":0,"category":"finance/economics","title":"A","slug":"a","tx":"z"}]
    cl = clusters(rows,2)
    assert len(cl) == 2 and cl[0]["q"] == dec("1550")
    a = row_stats(rows,"fills")
    assert a["trades_with_at_least_1_full_contract"] == 1
    b = row_stats(cl,"clusters")
    assert b["trades_with_at_least_1_full_contract"] == 1
    assert inventory_diagnostic(rows)["gross_fifo_sell_gain_from_observed_buys_ex_fees_usdc"] == dec("50.00")
    assert len(category_table(rows)) == 1
    print("PASS: 0.001 sizing, clustering, partial sell FIFO, category diagnostics (offline)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--start", default="2022-01-01", help="Inclusive UTC date (full history default)")
    parser.add_argument("--end", default=None, help="Exclusive UTC date; default tomorrow UTC")
    parser.add_argument("--output-dir",type=Path,default=Path("data/research/albert1953_ratio001"))
    parser.add_argument("--offline",action="store_true",help="Use existing complete cached trades, no internet")
    parser.add_argument("--max-pages",type=int,default=250)
    parser.add_argument("--cluster-seconds",type=int,default=2)
    args=parser.parse_args()
    if args.self_test: return self_test()
    if args.max_pages < 1 or args.cluster_seconds < 0: parser.error("invalid max-pages/cluster-seconds")
    end_text=args.end or (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    start,end=date_epoch(args.start),date_epoch(end_text)
    if end<=start: parser.error("--end must be later than --start")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    trade_path=out/"albert1953_trades.jsonl"
    meta_path=out/"albert1953_trades_meta.json"
    run_id={"wallet":WALLET,"start":start,"end_exclusive":end,"taker_only":False}
    if not args.offline:
        print(f"Downloading Albert1953 public trades [{args.start}, {end_text}), maker + taker",flush=True)
        params={"user":WALLET,"limit":1000,"taker_only":"false","start":start,"end":end-1}
        meta=download_pages("/v2/trades",params,trade_path,args.max_pages)
        meta.update(run_id)
        meta_path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
        print(f"Downloaded {meta['rows']:,} rows in {meta['pages']} pages")
        for label,endpoint,kwargs in [
            ("stats","/v2/user-stats",{"user":WALLET}),
            ("pnl","/v2/user-pnl",{"user":WALLET,"interval":"max","fidelity":"1d"})]:
            try:
                response=get_json(endpoint,kwargs)
                (out/f"albert1953_{label}.json").write_text(json.dumps(response,indent=2),encoding="utf-8")
            except Exception as exc:
                print(f"WARNING: optional {label} endpoint failed: {exc}",file=sys.stderr)
        for status in ("OPEN","CLOSED"):
            try:
                p=download_pages("/v2/positions",{"user":WALLET,"status":status,"limit":500},out/f"albert1953_positions_{status.lower()}.jsonl",args.max_pages)
                (out/f"albert1953_positions_{status.lower()}_meta.json").write_text(json.dumps(p,indent=2),encoding="utf-8")
            except Exception as exc:
                print(f"WARNING: optional positions {status} failed: {exc}",file=sys.stderr)
    if not trade_path.is_file() or not meta_path.is_file():
        raise RuntimeError("Missing downloaded trades or metadata")
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete") or any(meta.get(k) != v for k,v in run_id.items()):
        raise RuntimeError("Cache incomplete or does not match requested date range / wallet / taker_only")
    source=load_jsonl(trade_path)
    if len(source) != meta.get("rows"): raise RuntimeError("Cached count mismatch")
    rows=cleaned_rows(source,start,end)
    grouped=clusters(rows,args.cluster_seconds)
    result={"wallet":WALLET,"requested_copy_ratio":float(RATIO),
            "coverage_utc":{"start_inclusive":args.start,"end_exclusive":end_text},
            "sample_complete":True,"fills":row_stats(rows,"raw individual public fill rows"),
            "clusters":row_stats(grouped,"same-token+side temporally grouped fills"),
            "cluster_gap_seconds":args.cluster_seconds,
            "categories":category_table(rows),
            "inventory":inventory_diagnostic(rows),
            "market_count":len({r["condition"] for r in rows}),
            "trades_file":str(trade_path),
            "caveats":[
                "No independent delayed bid/ask book; cannot claim executable backtest ROI.",
                "Trading at the source fill price is hypothetical; Kalshi contract prices/market availability can differ.",
                "Market resolution, exact strike/settlement wording, fees, carry-in, and redemptions not replayed.",
                "Clusters are NOT provably the wallet's original submitted orders.",
                "Block timestamps have second precision; within-second ordering is unknown.",
                "Contract minimums and whole-share rounding depend on the execution venue; losing tiny signals may bias selection."]}
    stats_file=out/"albert1953_stats.json"
    if stats_file.exists():
        result["profile_stats_data"] = json.loads(stats_file.read_text(encoding="utf-8")).get("data")
    pnl_file=out/"albert1953_pnl.json"
    if pnl_file.exists():
        pnl=json.loads(pnl_file.read_text(encoding="utf-8")).get("data")
        if isinstance(pnl,dict):
            points=pnl.get("points") or []
            result["pnl_series_meta"]={"points":len(points),"first_timestamp":points[0].get("timestamp") if points else None,
                                       "last_timestamp":points[-1].get("timestamp") if points else None}
    for status in ("open","closed"):
        p=out/f"albert1953_positions_{status}.jsonl"
        meta_p=out/f"albert1953_positions_{status}_meta.json"
        if p.exists() and meta_p.exists():
            pm=json.loads(meta_p.read_text(encoding="utf-8"))
            if pm.get("complete"):
                pos=load_jsonl(p)
                result[f"{status}_positions"]={"count":len(pos),
                   "sum_reported_realized_pnl_usdc":round(sum((dec(x.get("realized_pnl") or 0) for x in pos),Decimal(0)),2),
                   "note":"Position-level realized PnL may not reconcile to wallet lifetime PnL; status is not settlement verification."}
    (out/"albert1953_ratio001_report.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    cluster_csv=[]
    for c in grouped:
        cluster_csv.append({"time_utc":datetime.fromtimestamp(c["first"],tz=timezone.utc).isoformat(),
            "last_utc":datetime.fromtimestamp(c["last"],tz=timezone.utc).isoformat(),"side":c["side"],
            "condition_id":c["condition"],"token_id":c["token"],"category":c["category"],"title":c["title"],
            "fills":c["fills"],"source_shares":str(c["q"]),"source_notional_usdc":str(round(c["notional"],4)),
            "target_shares_001":str(round(c["q"]*RATIO,6)),
            "target_usdc_at_source_price":str(round(c["notional"]*RATIO,5)),
            "whole_contracts_if_floored":int((c["q"]*RATIO).to_integral_value(rounding=ROUND_FLOOR))})
    write_csv(out/"albert1953_ratio001_clusters.csv",cluster_csv)
    print("\nALBERT1953 — EXACT 0.001 TARGET (SOURCE FILLS; NOT FOLLOWER BACKTEST)")
    print(f"Window UTC: {args.start} to {end_text} (exclusive) | markets: {result['market_count']:,}")
    for k in ("fills","clusters"):
        x=result[k]
        print(f"{k}: {x['count']:,} | BUY {x['buys']:,} SELL {x['sells']:,} | median source ${x['median_source_notional_usdc']}")
        print(f"  .001 median target ${x['median_copy_notional_usdc_at_source_price']} | "+
              f"<1 whole contract: {x['copy_qty_under_one_whole_contract_pct']}% | < $1 target: {x['copy_notional_under_1_usdc_pct']}%")
        print(f"  exact target BUY spend at original prices ${x['copy_buy_notional_at_source_price_usdc']} | "+
              f"eligible for >=1 whole contract: {x['trades_with_at_least_1_full_contracts'] if 'trades_with_at_least_1_full_contracts' in x else x['trades_with_at_least_1_full_contract']:,}")
    print("Category fill counts:", ", ".join(f"{v['category']}={v['fills']}" for v in result["categories"]))
    print(f"REPORT: {out/'albert1953_ratio001_report.json'}")
    print(f"CLUSTERS: {out/'albert1953_ratio001_clusters.csv'}")
    print("READ-ONLY TO EXCHANGES. NO ORDERS. Historical executable-copy PnL NOT established.")


if __name__ == "__main__":
    main()
