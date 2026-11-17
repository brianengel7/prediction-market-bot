#!/usr/bin/env python3
"""BreakTheBank V15: Kalshi matched-NBA settled-candle proxy (READ ONLY).

Inputs: V14 kalshi_matches_pilot.csv and kalshi_price_screen.csv.
Copies all V14-eligible NBA buy AND sell order intents, proportionally.
Quotes are V14 historical 1-minute CLOSE prices, NOT order-book fills.
Fees are a stated hypothetical Kalshi quadratic taker-fee multiplier.
Kalshi settlement `result` comes only from the public Kalshi market record.
Missing results can optionally be fetched via PUBLIC GET without credentials.

No exchange orders, no private keys, no repository writes.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import quote

BASE='https://external-api.kalshi.com/trade-api/v2'

def read_csv(path):
    with Path(path).expanduser().open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def finite(x):
    y=float(x)
    if not math.isfinite(y): raise ValueError('nonfinite number')
    return y

def fee(shares,price,multiplier):
    """Per-order 7% quadratic taker-fee model, rounded UP to a cent."""
    q,p,m=(Decimal(str(x)) for x in (shares,price,multiplier))
    if not (q>=0 and 0<p<1 and m>=0): raise ValueError('invalid fee inputs')
    return float((Decimal('.07')*m*q*p*(1-p)*100).to_integral_value(rounding=ROUND_CEILING)/100)

def get_result(ticker):
    for prefix in ('/markets/','/historical/markets/'):
        url=BASE+prefix+quote(ticker,safe='')
        try:
            req=Request(url,headers={'User-Agent':'btb-v15-research-readonly/1.0','Accept':'application/json'})
            with urlopen(req,timeout=25) as f:
                data=json.load(f)
        except HTTPError as e:
            if e.code==404:continue
            raise
        m=data.get('market') if isinstance(data,dict) else None
        if isinstance(m,dict) and m.get('ticker')==ticker:
            result=str(m.get('result') or '').lower().strip()
            return result if result in ('yes','no') else None
    return None

def validated_rows(plan,screen):
    byid={}
    for r in plan:
        if r.get('league')!='nba' or r.get('rules_status')!='REVIEW_NBA_RULES':
            continue
        ident=r.get('intent_id')
        if not ident:raise ValueError('missing intent_id')
        if ident in byid: raise ValueError('Duplicate NBA intent_id: '+ident)
        byid[ident]=r
    if not byid:raise RuntimeError('No NBA matched intents in V14 plan')
    data=defaultdict(dict)
    for r in screen:
        ident=r.get('intent_id')
        if ident not in byid:continue
        p=byid[ident]
        for field in ('kalshi_ticker','poly_market','side','shares','wallet_price'):
            if str(r.get(field))!=str(p.get(field)):
                raise ValueError(f'V14 screen/plan mismatch {ident}: {field}')
        delay=int(float(r['delay_s']))
        if delay in data[ident]: raise ValueError(f'Duplicate intent/delay: {ident} {delay}')
        data[ident][delay]=r
    return byid,data

def replay(rows, delay, resultmap, mult):
    # Every eligible side represented: no cherry-picking winning markets.
    inventory=defaultdict(float)
    book=defaultdict(lambda: {'cost':0.,'proceeds':0.,'fees':0.,'payout':0.,'pnl':0.,
                              'selected':0,'intents':0,'missing':0,'filled_buys':0,'filled_sells':0,
                              'buy_qty':0.,'sell_qty':0.,'n_subshare':0})
    count=Counter(); differences=[]; missing_status=Counter()
    byticker=defaultdict(list)
    for p,sc in rows:
        byticker[p['kalshi_ticker']].append((p,sc))
    for ticker,group in sorted(byticker.items()):
        state=book[ticker]
        group.sort(key=lambda v:(finite(v[0]['ts']),str(v[0]['intent_id'])))
        state['intents']=len(group)
        for p,sc in group:
            state['selected']+=1
            if sc is None:
                missing_status['no_candle_row']+=1;state['missing']+=1;continue
            status=sc.get('status','')
            if status!='within_price_limit':
                state['missing']+=1;missing_status[status or 'unknown']+=1;continue
            qty=finite(p['shares']);price=finite(sc['market_price']);wallet=finite(p['wallet_price'])
            if qty<=0 or not (0<price<1 and 0<wallet<1):
                raise ValueError('invalid order quantity or prices')
            if qty<1:state['n_subshare']+=1
            if str(p['side']).upper()=='BUY':
                executed=qty
                inventory[ticker]+=executed
                state['cost']+=executed*price
                state['fees']+=fee(executed,price,mult)
                state['buy_qty']+=executed;state['filled_buys']+=1
                count['buys']+=1
            elif str(p['side']).upper()=='SELL':
                executed=min(qty,inventory[ticker])
                if executed<=1e-10:
                    count['sell_without_inventory']+=1
                    missing_status['sell_without_inventory']+=1
                    state['missing']+=1
                    continue
                inventory[ticker]-=executed
                state['proceeds']+=executed*price
                state['fees']+=fee(executed,price,mult)
                state['sell_qty']+=executed;state['filled_sells']+=1
                count['sells']+=1
            else:raise ValueError('unexpected trade side')
            differences.append((price-wallet if p['side']=='BUY' else wallet-price,executed))
        result=resultmap.get(ticker)
        if result=='yes':state['payout']=inventory[ticker]
        if result in ('yes','no'):
            state['pnl']=state['proceeds']+state['payout']-state['cost']-state['fees']
    score=[v for ticker,v in book.items() if resultmap.get(ticker) in ('yes','no') and v['buy_qty']>0]
    complete=[v for v in score if v['missing']==0]
    def agg(group):
        cost=sum(v['cost'] for v in group);proceeds=sum(v['proceeds'] for v in group)
        fees=sum(v['fees'] for v in group);payout=sum(v['payout'] for v in group)
        pnl=sum(v['pnl'] for v in group)
        return dict(markets=len(group),buys=sum(v['filled_buys'] for v in group),sells=sum(v['filled_sells'] for v in group),
                    cost=round(cost,2),sell_proceeds=round(proceeds,2),settlement=round(payout,2),fees=round(fees,2),
                    pnl=round(pnl,2),roi_pct=round(100*pnl/(cost+fees),3) if cost+fees>0 else None)
    totalvol=sum(q for _,q in differences)
    return dict(delay=delay,scored=agg(score),complete=agg(complete),
                results_missing=sum(1 for k in book if resultmap.get(k) not in ('yes','no')),
                missing_order_status=dict(missing_status),
                measured_worsening_cents=round(100*sum(v*q for v,q in differences)/totalvol,4) if totalvol else None,
                subshare_orders=sum(v['n_subshare'] for v in book.values()),
                unmatched_sells=count['sell_without_inventory'],markets={k:dict(v,kalshi_result=resultmap.get(k)) for k,v in book.items()})

def selftest():
    assert fee(100,.5,1)==1.75
    assert fee(1,.5,1)==.02
    assert fee(10,.1,1)==.07
    def p(i,side,ts,price='0.50'):
        return dict(intent_id=str(i),kalshi_ticker='KXNBAGAME-TEST-ABC',poly_market='x',
                    side=side,shares='2',wallet_price=price,ts=str(ts),league='nba',rules_status='REVIEW_NBA_RULES')
    rows=[(p(1,'BUY',100),dict(status='within_price_limit',market_price='.51')),
          (p(2,'SELL',200),dict(status='within_price_limit',market_price='.60'))]
    r=replay(rows,5,{'KXNBAGAME-TEST-ABC':'yes'},1)
    assert r['scored']['buys']==1 and r['scored']['sells']==1
    assert r['scored']['settlement']==0 and r['scored']['pnl']>0
    rr=replay([(p(1,'BUY',100),dict(status='within_price_limit',market_price='.51'))],5,
              {'KXNBAGAME-TEST-ABC':'no'},1)
    assert rr['scored']['roi_pct']==-100
    rs=replay([(p(2,'SELL',90),dict(status='within_price_limit',market_price='.6'))],5,
              {'KXNBAGAME-TEST-ABC':'no'},1)
    assert rs['unmatched_sells']==1 and rs['complete']['markets']==0
    print('PASS V15: exact intent matching, rounded fee, inventory-capped sells, resolved payouts')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',default=str(Path.home()/'sports_copy_research/breakthebank_v14/kalshi_matches_pilot.csv'))
    ap.add_argument('--screen',default=str(Path.home()/'sports_copy_research/breakthebank_v14/kalshi_price_screen.csv'))
    ap.add_argument('--delays',default='5,15,30,60')
    ap.add_argument('--fee-multiplier',type=float,default=1.)
    ap.add_argument('--fetch-missing-results',action='store_true',help='Public GET for missing Kalshi settlement results')
    ap.add_argument('--out',default=None,help='NEW JSON outside repo (optional)')
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:selftest();return
    if not math.isfinite(a.fee_multiplier) or a.fee_multiplier<0:ap.error('invalid multiplier')
    plan=read_csv(a.plan);screen=read_csv(a.screen)
    byid,bydelay=validated_rows(plan,screen)
    tickers=sorted({r['kalshi_ticker'] for r in byid.values()})
    results={}
    for ticker in tickers:
        vals={str(r.get('kalshi_result') or '').strip().lower() for r in byid.values() if r['kalshi_ticker']==ticker}
        vals.discard('')
        if len(vals)>1:raise ValueError('Conflicting market results for '+ticker)
        result=next(iter(vals),None)
        if result not in ('yes','no') and a.fetch_missing_results:
            result=get_result(ticker)
            time.sleep(.1)
        results[ticker]=result if result in ('yes','no') else None
    delays=[int(x) for x in a.delays.split(',')]
    if not delays or any(x<0 for x in delays):ap.error('invalid delays')
    report={}
    print('BREAKTHEBANK V15 — KALSHI HISTORICAL CANDLE SETTLEMENT PROXY (READ ONLY)')
    print(f'NBA matched intents={len(byid)} | NBA markets={len(tickers)} | official results={sum(v is not None for v in results.values())}/{len(results)}')
    print(f'Fee stress: 7% quadratic taker fee, rounded per order, multiplier={a.fee_multiplier:g}; series overrides NOT certified')
    for delay in delays:
        rows=[(p,bydelay.get(i,{}).get(delay)) for i,p in byid.items()]
        r=replay(rows,delay,results,a.fee_multiplier)
        report[str(delay)]=r
        s,c=r['scored'],r['complete']
        print(f'{delay:>3}s | all scored markets {s["markets"]:>2} PnL={s["pnl"]:>10,.2f} ROI={str(s["roi_pct"]):>8}% | '
              f'complete-intent markets {c["markets"]:>2} PnL={c["pnl"]:>10,.2f} ROI={str(c["roi_pct"]):>8}% '
              f'| filled buys={s["buys"]} sells={s["sells"]} | missing intents={sum(r["missing_order_status"].values())}')
        print('    Order rejections:',r['missing_order_status'],'price difference cents:',r['measured_worsening_cents'],
              'sub-1-share candidate orders:',r['subshare_orders'])
    print('CAUTION: candle CLOSE prices are not executable depth, results rely on V14 outcome mapping, '+
          'contract rules not formally certified, fees hypothetical and fractional share minimums unverified.')
    print('V14 historical date selection and price availability can bias ROI; this is NOT a tradable backtest.')
    if a.out:
        dest=Path(a.out).expanduser().resolve()
        if str(dest).startswith('/workspaces/') or dest.exists():raise ValueError('Use a NEW JSON path outside repository')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(dict(assumptions=dict(fee_multiplier=a.fee_multiplier,venue='Kalshi',source='V14 minute close candles',trade_mode='paper_proxy'),
                                      settlement_results=results,delays=report),indent=2))
        print('Report:',dest)
    print('PUBLIC GET ONLY IF REQUESTED; NO ORDERS/KEYS/REPO WRITES')
if __name__=='__main__':main()
