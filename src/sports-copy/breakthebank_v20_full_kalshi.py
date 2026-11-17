#!/usr/bin/env python3
"""All 128 V19-matched sports markets. Historical candle proxy. NO ORDERS."""
import bisect, csv, json, math, runpy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home()/"sports_copy_research"
v14 = runpy.run_path(
    "src/sports-copy/breakthebank_v14_kalshi_crossvenue_readonly.py",
    run_name="library"
)
v15 = runpy.run_path(
    "src/sports-copy/breakthebank_v15_kalshi_settlement_proxy_readonly.py",
    run_name="library"
)
OUT = ROOT/"breakthebank_v20/full_sports_proxy.json"
CACHE = ROOT/"breakthebank_v20/candles"
DELAYS = (5,15,30,60,120,300)
LAG,MAX_AGE,LIMIT = 10,75,.02

def rows(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))

def price(bid,ask,trade,side):
    if side=="YES":
        return ask if trade=="BUY" else bid
    if side=="NO":
        x=bid if trade=="BUY" else ask
        return None if x is None else 1-x
    raise ValueError("Bad Kalshi side")

def test():
    assert price(.4,.42,"BUY","YES")==.42
    assert price(.4,.42,"SELL","YES")==.4
    assert abs(price(.4,.42,"BUY","NO")-.6)<1e-9
    assert abs(price(.4,.42,"SELL","NO")-.58)<1e-9
    assert price(None,.42,"BUY","NO") is None
    assert v15["fee"](100,.5,1)==1.75
    print("PASS V20: YES/NO bid/ask sides, missing quotes, fees; OFFLINE")

def load():
    orig=rows(ROOT/"breakthebank_v8/intents.csv")
    parity=rows(ROOT/"breakthebank_v19/settlement_parity.csv")
    inventory=rows(ROOT/"breakthebank_v17/all_sports_market_audit.csv")
    ids={};all_cids={}
    for o in orig:
        cid=o["market"].lower()
        all_cids.setdefault(cid,[]).append(o)
        if o["intent_id"] in ids:
            raise ValueError("Duplicate intent_id")
        ids[o["intent_id"]]=o
    assert len(orig)==1179 and len(all_cids)==144, \
        "Unexpected historical V8 cohort"
    good={}
    for p in parity:
        cid=p["poly_market"].lower()
        if cid in good or cid not in all_cids:
            raise ValueError("Bad V19 ID "+cid)
        assert p["status"] in (
            "PAYOUT_MATCH_RULES_UNCERTIFIED","PAYOUT_MATCH_REVIEW"
        )
        assert p["side"] in ("YES","NO")
        assert p["ticker"] and \
            {x["token_id"] for x in all_cids[cid]}=={p["token"]}
        assert abs(float(p["poly_payout"])-float(p["kalshi_payout"]))<1e-9
        good[cid]=p
    assert len(good)==128,"V19 count is not 128"
    missing={
        c:set(x["match_status"] for x in inventory
              if x["poly_market"].lower()==c)
        for c in all_cids if c not in good
    }
    by_ticker=defaultdict(list)
    for cid,orders in all_cids.items():
        if cid in good:
            for o in orders:
                by_ticker[good[cid]["ticker"]].append((o,good[cid]))
    return orig,good,missing,by_ticker

def candles(ticker,orders):
    t=[float(o["wallet_trade_ts_utc"]) for o,p in orders]
    start,end=int(min(t))+min(DELAYS),int(max(t))+max(DELAYS)
    path=CACHE/(ticker+".json")
    if path.exists():
        data=json.loads(path.read_text())
        if data["window"]!=[start,end]:
            raise ValueError("Cache window mismatch "+ticker)
        quotes={int(k):tuple(v) for k,v in data["quotes"].items()}
        return sorted(quotes),quotes,"cache"
    ts,quotes=v14["candle_history"](ticker,start,end)
    CACHE.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"window":[start,end],"quotes":quotes}))
    tmp.replace(path)
    return ts,quotes,"download"

def attempt(o,m,delay,times,book,fetch_failed):
    if fetch_failed:
        return None,"FETCH_ERROR"
    target=float(o["wallet_trade_ts_utc"])+delay
    i=bisect.bisect_right(times,target-LAG)-1
    if i<0:
        return None,"NO_ASOF_CANDLE"
    close=times[i]
    if target-close>MAX_AGE:
        return None,"STALE_CANDLE"
    bid,ask=book[close]
    px=price(bid,ask,o["side"],m["side"])
    if px is None or not 0<px<1:
        return None,"MISSING_QUOTE_SIDE"
    ref=float(o["wallet_price"])
    ok=px<=ref+LIMIT+1e-9 if o["side"]=="BUY" \
        else px>=ref-LIMIT-1e-9
    return px,"PASS" if ok else "OUTSIDE_LIMIT"

def replay(events,good,mode):
    states={
        cid:dict(
            cost=0.,proceeds=0.,fee=0.,payout=0.,pnl=0.,
            qty=0.,buys=0,sells=0,count=0,
            complete=True,observed=True,reasons=Counter(),
            sport=p["sport"],kind=p["kind"],ticker=p["ticker"],
            first_ts=None
        ) for cid,p in good.items()
    }
    for o,m,px,status in sorted(
        events,
        key=lambda z:(float(z[0]["wallet_trade_ts_utc"]),
                      z[0]["intent_id"])
    ):
        s=states[o["market"].lower()]
        s["count"]+=1
        ts=float(o["wallet_trade_ts_utc"])
        s["first_ts"]=ts if s["first_ts"] is None \
            else min(ts,s["first_ts"])
        if status not in ("PASS","OUTSIDE_LIMIT"):
            s["observed"]=False
        if status!="PASS":
            s["complete"]=False
            s["reasons"][status]+=1
            continue
        q=float(o["target_copy_shares"])
        if mode=="whole":
            q=math.floor(q+1e-10)
        if q<=0:
            s["reasons"]["TOO_SMALL"]+=1
            s["complete"]=False
            continue
        if o["side"]=="BUY":
            s["qty"]+=q
            s["buys"]+=1
            s["cost"]+=q*px
        else:
            q=min(q,s["qty"])
            if q<=0:
                s["reasons"]["NO_INVENTORY"]+=1
                s["complete"]=False
                continue
            s["qty"]-=q
            s["sells"]+=1
            s["proceeds"]+=q*px
        s["fee"]+=v15["fee"](q,px,1)
    for cid,s in states.items():
        s["payout"]=s["qty"]*float(good[cid]["kalshi_payout"])
        s["pnl"]=s["payout"]+s["proceeds"]-s["cost"]-s["fee"]
    return states

def summary(states):
    s=list(states)
    cost=sum(x["cost"]+x["fee"] for x in s)
    pnl=sum(x["pnl"] for x in s)
    return dict(
        markets=len(s),
        active=sum(x["buys"]>0 for x in s),
        buys=sum(x["buys"] for x in s),
        sells=sum(x["sells"] for x in s),
        cost_plus_fees=round(cost,2),
        pnl=round(pnl,2),
        roi_pct=round(100*pnl/cost,3) if cost else None
    )

def main():
    import sys
    if "--self-test" in sys.argv:
        test()
        return
    if OUT.exists():
        raise SystemExit("Report exists; refusing overwrite: "+str(OUT))
    orig,good,missing,by_ticker=load()
    print(
        "V20 FULL SPORTS KALSHI PROXY | "
        "144 originals | 128 payout-matched | 16 unmapped",
        flush=True
    )
    print("Kinds:",dict(Counter(
        (v["sport"],v["kind"]) for v in good.values()
    )))
    print("Sides:",dict(Counter(
        v["side"] for v in good.values()
    )))
    print("Unmapped:",dict(Counter(
        k for v in missing.values() for k in v
    )))
    events={d:[] for d in DELAYS}
    fetch_errors={}
    for i,(ticker,orders) in enumerate(
        sorted(by_ticker.items()),1
    ):
        try:
            times,book,source=candles(ticker,orders)
            failed=False
            print(
                f"[{i}/{len(by_ticker)}] {ticker}: "
                f"{len(orders)} intents, {len(times)} candles ({source})",
                flush=True
            )
        except Exception as exc:
            times=[];book={};failed=True
            fetch_errors[ticker]=f"{type(exc).__name__}: {exc}"
            print(
                "FETCH ERROR",ticker,fetch_errors[ticker],flush=True
            )
        for d in DELAYS:
            for o,m in orders:
                px,st=attempt(o,m,d,times,book,failed)
                events[d].append((o,m,px,st))
    report=dict(
        source_markets=144,
        source_intents=len(orig),
        mapped_markets=len(good),
        mapped_intents=sum(len(v) for v in by_ticker.values()),
        unmapped={cid:list(status) for cid,status in missing.items()},
        fetch_errors=fetch_errors,
        notes=(
            "CANDLE CLOSE PROXY; no depth or proven fills; "
            "7% fee assumption; source-publication lag not measured"
        ),
        delays={}
    )
    for d in DELAYS:
        scenario={
            "intent_statuses":dict(Counter(
                e[3] for e in events[d]
            ))
        }
        for mode in ("fractional","whole"):
            x=replay(events[d],good,mode)
            october=[
                v for v in x.values()
                if v["first_ts"] is not None
                and datetime.fromtimestamp(
                    v["first_ts"],timezone.utc
                ).month==10
            ]
            november=[
                v for v in x.values()
                if v["first_ts"] is not None
                and datetime.fromtimestamp(
                    v["first_ts"],timezone.utc
                ).month==11
            ]
            scenario[mode]=dict(
                all=summary(x.values()),
                fully_quoted=summary(
                    v for v in x.values() if v["observed"]
                ),
                all_intents_filled=summary(
                    v for v in x.values() if v["complete"]
                ),
                october=summary(october),
                november=summary(november),
                by_kind={
                    f"{league}_{kind}":summary(
                        v for v in x.values()
                        if v["sport"]==league and v["kind"]==kind
                    )
                    for league,kind in sorted({
                        (v["sport"],v["kind"]) for v in x.values()
                    })
                },
                markets={
                    cid:{
                        **{
                            k:v for k,v in s.items()
                            if k!="reasons"
                        },
                        "reasons":dict(s["reasons"])
                    } for cid,s in x.items()
                }
            )
            a=scenario[mode]["all"]
            print(
                f'{d:>3}s {mode:<10} '
                f'PnL={a["pnl"]:+.2f} ROI={a["roi_pct"]}% '
                f'active={a["active"]}/128 '
                f'buys={a["buys"]} sells={a["sells"]} '
                f'OCT={scenario[mode]["october"]["pnl"]:+.2f} '
                f'NOV={scenario[mode]["november"]["pnl"]:+.2f}',
                flush=True
            )
        print("  Statuses:",scenario["intent_statuses"],flush=True)
        report["delays"][str(d)]=scenario
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2))
    print("Report:",OUT)
    print(
        "READ ONLY: public Kalshi candle GETs, no orders. "
        "Not an executable-fill backtest."
    )

if __name__=="__main__":
    main()
