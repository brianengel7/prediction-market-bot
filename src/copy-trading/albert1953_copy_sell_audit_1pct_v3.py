#!/usr/bin/env python3
"""Albert1953 1% copy-sizing and SELL coverage audit v3 -- offline, no exchange calls/orders.

Reads the COMPLETE cache from albert1953_ratio_audit_v1.py and tests how
0.01 * source share inventory translates into actual position changes.
Scenarios: whole-contract (1.00) vs optional fractional (.01) quantities;
with zero or hypothetical $1 minimum order notional. Neither is a price
or realized-P&L backtest. Source fills are treated as source-price proxies.

Important: Polymarket API trade history excludes settlement/redemption cashflows.
Kalshi fractional quantity is MARKET-DEPENDENT, NOT GUARANTEED. Hypothetical copy trades use source-fill prices as a non-executable proxy.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_FLOOR
import csv
import json
from pathlib import Path

WALLET = "0x777fae71d2ff9ec48a1213d48ba1d9d91024a1bb"
D = Decimal
ZERO = D("0")
EPS = D("0.00000001")


def pct(a, b):
    return round(100 * float(a) / float(b), 3) if b else None


def dec_round(x, places=4):
    return round(float(x), places)


def exact_floor(q: Decimal, granularity: Decimal) -> Decimal:
    return (q / granularity).to_integral_value(rounding=ROUND_FLOOR) * granularity


def read_source(cache: Path):
    meta_file = cache / "albert1953_trades_meta.json"
    trade_file = cache / "albert1953_trades.jsonl"
    if not (meta_file.exists() and trade_file.exists()):
        raise FileNotFoundError("Missing full v1 wallet cache. Run the v1 full-history audit first.")
    m = json.loads(meta_file.read_text())
    if m.get("complete") is not True or m.get("wallet", "").lower() != WALLET or m.get("taker_only") is not False:
        raise ValueError("Source cache must be COMPLETE for this wallet, maker+taker included")
    records=[]
    last=None
    descending_pairs=ascending_pairs=0
    with trade_file.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r=json.loads(line)
            ts=int(r["timestamp"])
            q=D(str(r["size"]))
            p=D(str(r["price"]))
            side=str(r["side"]).upper()
            cond=str(r.get("condition_id") or "")
            tok=str(r.get("token_id") or "")
            address=str(r.get("proxy_wallet") or "").lower()
            if ts < m["start"] or ts >= m["end_exclusive"]:
                raise ValueError(f"Out-of-window source row {i}")
            if address and address != WALLET:
                raise ValueError(f"Wrong wallet in row {i}")
            if not (cond and tok and q>0 and ZERO<p<D(1) and side in ("BUY", "SELL")):
                raise ValueError(f"Malformed fill {i}")
            if last is not None:
                if ts<last: descending_pairs+=1
                if ts>last: ascending_pairs+=1
            last=ts
            records.append({"ts":ts, "file_order":i, "condition":cond,"token":tok,
                            "side":side,"q":q,"price":p, "category": "unclassified",
                            "title":str(r.get("title") or r.get("slug") or ""),
                            "slug":str(r.get("slug") or "")})
    if len(records) != m.get("rows"):
        raise ValueError("Cached row count differs from completeness metadata")
    if not records:
        raise ValueError("Empty cache")
    # The cursor trade feed is generally newest first. Detect the direction
    # rather than assuming the stored feed is chronological. Same-second trade
    # sequencing is still only an approximation and must be flagged.
    desc=descending_pairs>ascending_pairs
    if min(descending_pairs,ascending_pairs)>0:
        raise ValueError("Trade feed chronology changes direction; do not guess within-second order")
    records.sort(key=lambda x: (x["ts"], -x["file_order"] if desc else x["file_order"]))
    return records, m, "newest-first" if desc else "oldest-first"


def market_categories(records):
    # Reuse same v1 heuristic, without importing modules that could make network calls.
    import re
    def categorize(r):
        s=(r["title"]+" "+r["slug"]).lower()
        if re.search(r"temperature|weather|rainfall|snowfall",s):return "weather"
        if re.search(r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|\bxrp\b|dogecoin|hyperliquid|crypto",s):return "crypto"
        if re.search(r"\bnfl\b|\bnba\b|\bmlb\b|\bnhl\b|\btennis\b|\bsoccer\b|\bvs\.?\b|\bsports\b",s):return "sports"
        if re.search(r"election|president|senate|governor|congress|democrat|republican|vote|trump|harris",s):return "politics"
        if re.search(r"\bstock\b|\bshares\b|\bcrude\b|\boil\b|\bgold\b|\bsilver\b|\bwti\b|\bmeta\b|\baapl\b|\bmsft\b|\bnvda\b|\bnasdaq\b|\bs&p\b|\bipo\b|\bmarket cap\b|\bfed\b|\binflation\b|\btreasury\b",s):return "finance/economics"
        return "other/unclassified"
    for r in records:r["category"]=categorize(r)


def validate_token_inventory(rows):
    """Exclude tokens for which trade history implies short/incomplete inventory.

    This catches missing carry-in, ambiguous same-second execution ordering,
    or unrepresented collateral position transfers. It never silently shorts.
    """
    net=defaultdict(lambda: ZERO)
    min_net=defaultdict(lambda:ZERO)
    side_seconds=defaultdict(set)
    for r in rows:
        tok=(r["condition"],r["token"])
        net[tok]+=r["q"]*(1 if r["side"]=="BUY" else -1)
        min_net[tok]=min(min_net[tok],net[tok])
        side_seconds[(tok,r["ts"])].add(r["side"])
    excluded={tok for tok, mn in min_net.items() if mn < -EPS}
    mixed_same_second={tok for (tok,_),sides in side_seconds.items() if len(sides)>1}
    return excluded, mixed_same_second


def evaluate(rows, excluded, step:Decimal, minimum:Decimal, meta, ratio:Decimal, *, sell_trace=None, scenario=""):
    source_net=defaultdict(lambda:ZERO)
    held=defaultdict(lambda:ZERO)
    lots=defaultdict(deque)
    market=defaultdict(lambda: {"source_buy":ZERO,"source_sell":ZERO,"copy_buy":ZERO,"copy_sell":ZERO,
                                "copy_buy_shares":ZERO,"copy_sell_shares":ZERO, "copy_buys":0,"copy_sells":0,
                                "wallet_buys":0,"wallet_sells":0,"fifo_realized":ZERO,
                                "remaining_cost":ZERO,"remaining_shares":ZERO,
                                "title":"", "slug":"", "category":""})
    source_buy=source_sell=copy_buy=copy_sell=realized=ZERO
    # Calendar coverage matters: full-history averages may hide recent strategy shifts.
    end_exclusive=int(meta.get("end_exclusive") or 0)
    bucket=defaultdict(lambda:{"source_buy":ZERO,"source_sell":ZERO,"copy_buy":ZERO,"copy_sell":ZERO,
                              "source_buy_fills":0,"source_sell_fills":0,"copy_buy_events":0,"copy_sell_events":0})
    def periods(ts):
        yield "year_"+datetime.fromtimestamp(ts,timezone.utc).strftime("%Y")
        if end_exclusive and ts>=end_exclusive-90*86400:yield "last_90d"
        if end_exclusive and ts>=end_exclusive-365*86400:yield "last_365d"
    copied_buy_events=copied_sell_events=source_buy_events=source_sell_events=0
    skipped_min_buys=skipped_min_sells=ZERO
    trade_rows_excluded=0
    fee_sensitive_buys=0
    sell_skip_reasons=Counter()
    sell_reason_notional=defaultdict(lambda: ZERO)
    source_sell_shares=ZERO
    copy_sell_shares=ZERO
    for r in rows:
        key=(r["condition"],r["token"])
        if key in excluded:
            trade_rows_excluded+=1
            continue
        m=market[r["condition"]]
        m["title"]=r["title"] or m["title"]
        m["slug"]=r["slug"] or m["slug"]
        m["category"]=r["category"]
        rnot=r["q"]*r["price"]
        if r["side"]=="BUY":
            source_buy+=rnot;source_buy_events+=1
            m["source_buy"]+=rnot;m["wallet_buys"]+=1
            source_net[key]+=r["q"]
            for period in periods(r["ts"]):
                bucket[period]["source_buy"]+=rnot
                bucket[period]["source_buy_fills"]+=1
        else:
            source_sell+=rnot;source_sell_events+=1
            m["source_sell"]+=rnot;m["wallet_sells"]+=1
            source_sell_shares+=r["q"]
            source_net[key]-=r["q"]
            for period in periods(r["ts"]):
                bucket[period]["source_sell"]+=rnot
                bucket[period]["source_sell_fills"]+=1
            if source_net[key]<ZERO and source_net[key] > -EPS:source_net[key]=ZERO
        if source_net[key]<-EPS:
            raise AssertionError("Excluded inventory negative token unexpectedly present")
        desired=exact_floor(source_net[key]*ratio,step)
        diff=desired-held[key]
        if r["side"]=="BUY" and diff>=step:
            notional=diff*r["price"]
            if notional<minimum:
                skipped_min_buys+=1
                continue
            # No up-rounding, no external prices: hypothetically buy at wallet price.
            held[key]+=diff
            lots[key].append([diff,r["price"]])
            copy_buy+=notional;copied_buy_events+=1
            m["copy_buy"]+=notional;m["copy_buy_shares"]+=diff;m["copy_buys"]+=1
            if notional<D("0.01"):fee_sensitive_buys+=1
            for period in periods(r["ts"]):
                bucket[period]["copy_buy"]+=notional
                bucket[period]["copy_buy_events"]+=1
        elif r["side"]=="SELL":
            held_before=held[key]
            def trace(reason, executed=ZERO):
                if sell_trace is not None:
                    sell_trace.append({"scenario":scenario, "utc":datetime.fromtimestamp(r["ts"],timezone.utc).isoformat(),
                        "condition":r["condition"], "token":r["token"], "category":r["category"],
                        "title":r["title"], "slug":r["slug"], "wallet_sell_shares":str(r["q"]),
                        "wallet_sell_price":str(r["price"]), "target_sell_shares":str(r["q"]*ratio),
                        "source_net_after_sell":str(source_net[key]), "desired_inventory":str(desired),
                        "copy_held_before_sell":str(held_before), "copied_sell_shares":str(executed),
                        "copy_held_after_sell":str(held[key]), "decision":reason})
            if held[key] <= ZERO:
                sell_skip_reasons["no_copied_inventory"]+=1
                sell_reason_notional["no_copied_inventory"]+=rnot
                trace("no_copied_inventory")
                continue
            if diff > -step:
                sell_skip_reasons["rounded_target_unchanged"]+=1
                sell_reason_notional["rounded_target_unchanged"]+=rnot
                trace("rounded_target_unchanged")
                continue
            qty=min(held[key],-diff)
            if qty<step:
                sell_skip_reasons["rounded_target_unchanged"]+=1
                sell_reason_notional["rounded_target_unchanged"]+=rnot
                trace("rounded_target_unchanged")
                continue
            notional=qty*r["price"]
            if notional<minimum:
                skipped_min_sells+=1
                sell_skip_reasons["below_hypothetical_min_notional"]+=1
                sell_reason_notional["below_hypothetical_min_notional"]+=rnot
                trace("below_hypothetical_min_notional")
                continue
            held[key]-=qty
            trace("would_sell_if_filled",qty)
            copy_sell_shares+=qty
            copy_sell+=notional;copied_sell_events+=1
            m["copy_sell"]+=notional;m["copy_sell_shares"]+=qty;m["copy_sells"]+=1
            for period in periods(r["ts"]):
                bucket[period]["copy_sell"]+=notional
                bucket[period]["copy_sell_events"]+=1
            rem=qty
            while rem>ZERO and lots[key]:
                item=lots[key][0]
                match=min(rem,item[0]); rem-=match;item[0]-=match
                pnl=match*(r["price"]-item[1])
                realized+=pnl;m["fifo_realized"]+=pnl
                if item[0]<=ZERO:lots[key].popleft()
            if rem>EPS:
                raise AssertionError("Copied sell exceeded copied BUY inventory")
    if sum(sell_skip_reasons.values())+copied_sell_events != source_sell_events:
        raise AssertionError("SELL audit accounting must classify every eligible SELL fill")
    for key,q in held.items():
        if q<=ZERO:continue
        m=market[key[0]]
        m["remaining_shares"]+=q
        m["remaining_cost"]+=sum(qty*p for qty,p in lots[key])
    category=defaultdict(lambda:{"markets":0,"source_buy":ZERO,"copy_buy":ZERO,"copy_sell":ZERO,
                                 "copy_buys":0,"copy_sells":0,"remaining_cost":ZERO})
    for cond,m in market.items():
        k=m["category"]
        cat=category[k];cat["markets"]+=1
        for fld in ("source_buy","copy_buy","copy_sell","remaining_cost"):
            cat[fld]+=m[fld]
        cat["copy_buys"]+=m["copy_buys"]
        cat["copy_sells"]+=m["copy_sells"]
    report={
        "granularity_contracts":str(step),"hypothetical_min_order_notional_usdc":str(minimum),
        "source_buy_notional_in_usable_tokens":dec_round(source_buy,4),
        "copy_target_buy_notional_at_original_price":dec_round(source_buy*ratio,4),
        "copy_target_sell_notional_at_original_price":dec_round(source_sell*ratio,4),
        "copied_buy_notional_at_original_prices":dec_round(copy_buy,4),
        "copied_sell_notional_at_original_prices":dec_round(copy_sell,4),
        "sell_skip_reason_counts":dict(sell_skip_reasons),
        "sell_skip_reason_original_notional_usdc":{k:dec_round(v,4) for k,v in sorted(sell_reason_notional.items())},
        "source_sell_shares_in_usable_tokens":dec_round(source_sell_shares,4),
        "copied_sell_shares_if_filled":dec_round(copy_sell_shares,4),
        "source_buy_fills_in_usable_tokens":source_buy_events,
        "source_sell_fills_in_usable_tokens":source_sell_events,
        "sell_order_event_capture_pct":pct(copied_sell_events,source_sell_events),
        "sell_notional_capture_pct":pct(copy_sell,source_sell*ratio),
        "buy_order_event_capture_pct":pct(copied_buy_events,source_buy_events),
        "buy_order_events_if_filled":copied_buy_events,
        "sell_order_events_if_filled":copied_sell_events,
        "source_buy_spend_capture_pct":pct(copy_buy,source_buy*ratio),
        "source_market_count_in_usable_tokens":len(market),
        "copied_market_count":sum(bool(m["copy_buys"] or m["copy_sells"]) for m in market.values()),
        "copied_buy_market_count":sum(bool(m["copy_buys"]) for m in market.values()),
        "sell_fifo_gross_pnl_before_fees":dec_round(realized,4),
        "unreconciled_inventory_cost_excluding_settlements":dec_round(sum((m["remaining_cost"] for m in market.values()),ZERO),4),
        "unreconciled_inventory_shares_excluding_settlements":dec_round(sum((m["remaining_shares"] for m in market.values()),ZERO),4),
        "min_notional_deferred_buy_fills":int(skipped_min_buys),
        "min_notional_deferred_sell_fills":int(skipped_min_sells),
        "buy_events_under_1_cent":fee_sensitive_buys,
        "source_rows_excluded_for_missing_or_ambiguous_inventory":trade_rows_excluded,
        "caveat":"Optimistic source-price quantity model only: no fees, quotes, fills, resolution, redemptions, Kalshi matching or actual fraction-eligibility; residual cost is NOT open-position capital"
    }
    report["periods"]={k:{"source_buy_usdc":dec_round(v["source_buy"],4),
                           "source_sell_usdc":dec_round(v["source_sell"],4),
                           "sell_target_usdc":dec_round(v["source_sell"]*ratio,4),
                           "sell_notional_capture_pct":pct(v["copy_sell"],v["source_sell"]*ratio),
                           "sell_event_capture_pct":pct(v["copy_sell_events"],v["source_sell_fills"]),
                           "copy_buy_usdc":dec_round(v["copy_buy"],4),
                           "copy_sell_usdc":dec_round(v["copy_sell"],4),
                           "source_buy_fills":v["source_buy_fills"],
                           "source_sell_fills":v["source_sell_fills"],
                           "copy_buy_events":v["copy_buy_events"],
                           "copy_sell_events":v["copy_sell_events"],
                           "buy_spend_capture_pct":pct(v["copy_buy"],v["source_buy"]*ratio)}
                    for k,v in sorted(bucket.items())}
    category_output=[]
    for k,v in sorted(category.items(),key=lambda kv:-kv[1]["source_buy"]):
        category_output.append({"category":k,"source_markets":v["markets"],"wallet_buy_usdc":dec_round(v["source_buy"],2),
            "copy_buy_usdc":dec_round(v["copy_buy"],4),"copy_sell_usdc":dec_round(v["copy_sell"],4),
            "buy_events":v["copy_buys"],"sell_events":v["copy_sells"],
            "copy_buy_capture_pct":pct(v["copy_buy"],v["source_buy"]*ratio),"residual_cost":dec_round(v["remaining_cost"],4)})
    return report, market, category_output


def selftest():
    d={"condition":"a","token":"a-y","category":"finance/economics","title":"A","slug":"a"}
    src=[dict(d,ts=i,file_order=i,side=s,q=D(q),price=D(p)) for i,(s,q,p) in enumerate([
        ("BUY","550","0.5"),("BUY","600","0.5"),("SELL","850","0.6")])]
    bad,amb=validate_token_inventory(src)
    assert not bad and not amb
    a,m,c=evaluate(src,bad,D("1"),ZERO,{"end_exclusive":4},D("0.001"))
    assert a["buy_order_events_if_filled"]==1 and a["sell_order_events_if_filled"]==1
    assert a["sell_fifo_gross_pnl_before_fees"]==0.1 and a["unreconciled_inventory_shares_excluding_settlements"]==0
    b,_,_=evaluate(src,bad,D("0.01"),ZERO,{"end_exclusive":4},D("0.001"))
    assert b["buy_order_events_if_filled"]==2 and b["sell_order_events_if_filled"]==1
    assert b["unreconciled_inventory_shares_excluding_settlements"]==0.3
    c,_,_=evaluate(src,bad,D("1"),D("1"),{"end_exclusive":4},D("0.001"))
    assert c["buy_order_events_if_filled"]==0
    high,_,_=evaluate(src,bad,D("1"),ZERO,{"end_exclusive":4},D("0.01"))
    assert high["buy_order_events_if_filled"]==2 and high["sell_order_events_if_filled"]==1
    assert high["copied_sell_shares_if_filled"]==8
    assert high["sell_fifo_gross_pnl_before_fees"]==0.8
    assert high["sell_order_event_capture_pct"]==100.0
    assert sum(high["sell_skip_reason_counts"].values())+high["sell_order_events_if_filled"]==high["source_sell_fills_in_usable_tokens"]
    print("PASS: ratio .001/.01, source inventory, fractional/whole, BUY/SELL, FIFO, min notional. Offline.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--cache",type=Path,default=Path("data/research/albert1953_ratio001"))
    p.add_argument("--ratio",type=str,default="0.01",help="Share quantity multiplier, default 0.01 (1%%)")
    a=p.parse_args()
    if a.self_test:return selftest()
    try: ratio=D(a.ratio)
    except Exception: p.error("--ratio must be a decimal, e.g. 0.01")
    if not ratio.is_finite() or not (ZERO<ratio<=D("1")):
        p.error("--ratio must be >0 and <=1")
    rows,meta,order=read_source(a.cache)
    market_categories(rows)
    excluded,ambiguous=validate_token_inventory(rows)
    tokens={(r["condition"],r["token"]) for r in rows}
    report={"wallet":WALLET,"ratio":str(ratio),"source_rows":len(rows),
       "source_markets":len({r["condition"] for r in rows}),"source_tokens":len(tokens),
       "api_file_direction":order,"source_min_inventory_negative_tokens_excluded":len(excluded),
       "mixed_buy_sell_same_second_tokens":len(ambiguous),
       "note":"API timestamps have one-second precision; some within-second ordering remains uncertain.",
       "scenarios":{}}
    selected={}
    all_sell_trace=[]
    for step,minimum,label in [(D("1"),ZERO,"whole_no_min"),(D("1"),D("1"),"whole_one_usd_min"),
                               (D("0.01"),ZERO,"fractional_no_min"),(D("0.01"),D("1"),"fractional_one_usd_min")]:
        result,market,cats=evaluate(rows,excluded,step,minimum,meta,ratio,sell_trace=all_sell_trace,scenario=label)
        report["scenarios"][label]=dict(result,categories=cats)
        if label in ("whole_no_min","fractional_no_min"):
            selected[label]=market
    out=a.cache
    suffix="ratio"+format(ratio,"f").replace(".","p")
    rp=out/f"albert1953_sell_audit_v3_{suffix}_report.json"
    rp.write_text(json.dumps(report,indent=2))
    tracepath=out/f"albert1953_sell_audit_v3_{suffix}_sell_decisions.csv"
    with tracepath.open("w",newline="") as fout:
        if all_sell_trace:
            wr=csv.DictWriter(fout,fieldnames=list(all_sell_trace[0]))
            wr.writeheader();wr.writerows(all_sell_trace)
    joined=[]
    ids=set(selected["whole_no_min"])|set(selected["fractional_no_min"])
    for cid in ids:
        w=selected["whole_no_min"].get(cid,{})
        f=selected["fractional_no_min"].get(cid,{})
        v=f or w
        joined.append({"market_condition":cid,"title":v.get("title",""),"slug":v.get("slug",""),
          "category":v.get("category",""),"source_buy_usdc":dec_round(v.get("source_buy",ZERO),2),
          "whole_copy_buy_usdc":dec_round(w.get("copy_buy",ZERO),4),"whole_copy_sell_usdc":dec_round(w.get("copy_sell",ZERO),4),
          "whole_buy_events":w.get("copy_buys",0),"whole_sell_events":w.get("copy_sells",0),
          "fractional_copy_buy_usdc":dec_round(f.get("copy_buy",ZERO),4),
          "fractional_copy_sell_usdc":dec_round(f.get("copy_sell",ZERO),4),
          "fractional_buy_events":f.get("copy_buys",0),"fractional_sell_events":f.get("copy_sells",0)})
    joined.sort(key=lambda r:-r["fractional_copy_buy_usdc"])
    csvpath=out/f"albert1953_sell_audit_v3_{suffix}_markets.csv"
    with csvpath.open("w",newline="") as fout:
        wr=csv.DictWriter(fout,fieldnames=list(joined[0]) if joined else ["market_condition"])
        wr.writeheader();wr.writerows(joined)
    print(f"ALBERT1953 / {ratio} ({ratio*100}%) -- POSITION-AWARE SELL AUDIT (NOT P&L BACKTEST)")
    print(f"Source: {len(rows):,} fills / {report['source_markets']:,} markets / {len(tokens):,} outcome tokens; feed order {order}")
    original_buys=sum(r["side"]=="BUY" for r in rows)
    original_sells=len(rows)-original_buys
    usable_sells=report["scenarios"]["whole_no_min"]["source_sell_fills_in_usable_tokens"]
    print(f"Recorded source executions: {original_buys:,} BUY and {original_sells:,} SELL; usable SELL for replay {usable_sells:,} (excluded {original_sells-usable_sells:,})")
    print(f"Excluded {len(excluded):,} tokens for negative trade-derived inventory; {len(ambiguous):,} tokens have mixed BUY/SELL same-second ambiguity")
    print("SCENARIO                        BUY EVENTS   SELL EVENTS  SELL EVENT %  SELL $ %   MARKETS    PROXY BUY $    BUY $ %  RESIDUAL COST $")
    for name,r in report["scenarios"].items():
        print(f"{name:30} {r['buy_order_events_if_filled']:>10,d} {r['sell_order_events_if_filled']:>12,d} {r['sell_order_event_capture_pct']:>12} {r['sell_notional_capture_pct']:>9} {r['copied_market_count']:>9,d} {r['copied_buy_notional_at_original_prices']:>14,.2f} {r['source_buy_spend_capture_pct']:>10} {r['unreconciled_inventory_cost_excluding_settlements']:>16,.2f}")
        print(f"    SELL miss reasons: {r['sell_skip_reason_counts']}")
    print("\nLast-90-day slice (if history spans that period):")
    for name,r in report["scenarios"].items():
        b=r.get("periods",{}).get("last_90d",{})
        if b:
            print(f"  {name:29} source BUY {b['source_buy_fills']:>5,d}, source SELL {b['source_sell_fills']:>5,d}, copied BUY {b['copy_buy_events']:>5,d}, copied SELL {b['copy_sell_events']:>5,d}, BUY $ capture {b['buy_spend_capture_pct']}%, SELL $ capture {b['sell_notional_capture_pct']}%")
    print("\nExecuted SELL FIFO gross is NOT complete P&L: ignores open/resolved residual inventory, fees, quotes, fills, Kalshi equivalence.")
    print("This replay floors target shares; a sourced SELL only executes up to held shares. Timestamps 1s; within-second sequence uncertain.")
    print("REPORT:",rp)
    print("MARKET RANKING:",csvpath)
    print("SELL DECISIONS:",tracepath)
    print("No exchange calls or orders.")

if __name__=="__main__":main()
