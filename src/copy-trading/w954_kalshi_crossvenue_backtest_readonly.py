#!/usr/bin/env python3
"""Read-only cross-venue historical replay: wallet 954B -> Kalshi KXBTC15M.

Place beside the existing w954_* scripts under src/copy-trading/ and run from
repository root. No POST, orders, file writes, or GitHub changes. HTTP GET
requests are used ONLY if --source api is selected (default).

In 'trade' mode, a post-signal Kalshi aggressive-buy PRINT is a PRICE PROXY,
not proof that an additional buy could fill. In 'bbo' mode, supply a historical
BBO JSONL/CSV or parquet file containing *as-of* asks and sizes; only its
observed top-of-book is used. Historical BBO with reception timestamps is best.

ALL outcomes are from Kalshi market.result, not from Polymarket/Chainlink.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import json
import math
import runpy
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = Path('src/copy-trading')
BASE_URL = 'https://external-api.kalshi.com/trade-api/v2'


def timestamp(v):
    if v is None or v == '': return None
    if isinstance(v, (float,int)) or (isinstance(v,str) and v.strip().replace('.','',1).isdigit()):
        q=float(v)
        if q >= 1e17: q/=1e9
        elif q >= 1e14: q/=1e6
        elif q >= 1e11: q/=1e3
        return q
    return datetime.fromisoformat(str(v).replace('Z','+00:00')).timestamp()


def objrows(path):
    path=Path(path)
    if path.suffix == '.csv':
        with path.open(newline='') as f:
            yield from csv.DictReader(f)
    elif path.suffix == '.parquet':
        try: import pandas as pd
        except ImportError: raise RuntimeError('To read parquet, install pandas and pyarrow')
        for item in pd.read_parquet(path).to_dict('records'): yield item
    else:
        with path.open() as f:
            if path.suffix == '.json':
                payload=json.load(f)
                if isinstance(payload,list): yield from payload
                else:
                    for field in ('markets','trades','data','rows'):
                        if isinstance(payload.get(field),list):
                            yield from payload[field]; return
                    raise ValueError('JSON must contain list or markets/trades/rows list')
            else:
                for line in f:
                    if line.strip(): yield json.loads(line)


def api_get(route,params=None,retries=5):
    url=BASE_URL+route+('?' + urlencode(params) if params else '')
    for i in range(retries):
        try:
            with urlopen(Request(url,headers={'User-Agent':'w954-readonly-crossvenue-research/1.0',
                                                  'Accept':'application/json'}),timeout=30) as r:
                data=json.load(r)
            if not isinstance(data,dict): raise RuntimeError('Unexpected API JSON type')
            return data
        except (HTTPError, URLError, TimeoutError, ValueError) as e:
            if isinstance(e,HTTPError) and e.code not in (429,500,502,503,504): raise
            if i==retries-1: raise RuntimeError(f'HTTP GET failed: {route}: {e}') from e
            time.sleep(min(16,2**i))
    raise AssertionError('unreachable')


def pages(route,params,key,max_pages=160):
    cursor=None
    seen=set()
    for i in range(max_pages):
        p=dict(params)
        if cursor: p['cursor']=cursor
        data=api_get(route,p)
        items=data.get(key)
        if not isinstance(items,list): raise RuntimeError(f'{route} missing {key} array')
        for it in items: yield it
        next_cursor=data.get('cursor')
        if not next_cursor: return
        if next_cursor in seen: raise RuntimeError('Repeated pagination cursor')
        cursor=next_cursor; seen.add(cursor)
    raise RuntimeError(f'{route} exceeded {max_pages} pages; no partial answer reported')


def market_close(m):
    # Use the actual window close. Never infer from ticker alone.
    return timestamp(m.get('close_time'))


def markets_api(signals):
    earliest=min(s['start'] for s in signals)-86400
    latest=max(s['start']+900 for s in signals)+86400
    out={}
    for m in pages('/historical/markets',{'series_ticker':'KXBTC15M','limit':1000},'markets'):
        if not str(m.get('ticker','')).startswith('KXBTC15M-'): continue
        t=market_close(m)
        if t is not None and earliest <= t <= latest: out[m['ticker']]=m
    # Recently settled markets can remain in the current API. Pull both sources.
    params={'series_ticker':'KXBTC15M','status':'settled','limit':1000,
            'min_settled_ts':int(earliest),'max_settled_ts':int(latest)}
    for m in pages('/markets',params,'markets'):
        if not str(m.get('ticker','')).startswith('KXBTC15M-'): continue
        t=market_close(m)
        if t is not None and earliest<=t<=latest: out[m['ticker']]=m
    return list(out.values())


def match_markets(signals,markets):
    by_close=defaultdict(list)
    for m in markets:
        if not str(m.get('ticker','')).startswith('KXBTC15M-'): continue
        t=market_close(m)
        if t is not None: by_close[round(t)].append(m)
    paired=[]; reasons=Counter()
    for s in signals:
        close=int(s['start'])+900
        candidates=by_close.get(close,[])
        if len(candidates)!=1:
            reasons['no_matching_market' if not candidates else 'ambiguous_market']+=1; continue
        m=candidates[0]
        result=str(m.get('result','')).lower().strip()
        if result not in ('yes','no'):
            reasons['unsettled_or_void']+=1; continue
        # UP -> YES, DOWN -> NO for up/down BTC 15-minute contracts.
        side='yes' if str(s['side']).upper()=='UP' else 'no' if str(s['side']).upper()=='DOWN' else ''
        if not side:
            reasons['unknown_wallet_side']+=1; continue
        if not (s['start'] < s['signal_ts'] < close):
            reasons['invalid_signal_clock']+=1; continue
        paired.append({**s,'ticker':m['ticker'],'kalshi_result':result,'kalshi_side':side,
                       'kalshi_win':int(side==result),'disagree':int(int(s['won'])!=int(side==result))})
    return paired,reasons


def extract_signals(args):
    required=[BASE/'w954_post_entry_sequence_audit_readonly.py',
              BASE/'w954_binance_signal_audit_readonly.py',
              BASE/'w954_followup_copy_window_audit_readonly.py']
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError('Missing existing local wallet loaders: '+', '.join(missing))
    a=runpy.run_path(str(required[0])); b=runpy.run_path(str(required[1])); c=runpy.run_path(str(required[2]))
    path=args.wallet or next((p for p in a['DEFAULT_PATHS'] if p.is_file()),None)
    if not path: raise RuntimeError('Could not find wallet trades.jsonl; use --wallet PATH')
    first,_=b['load_wallet']()
    grouped,raw,invalid,confirmed,collisions=a['load_trades'](path)
    if collisions or confirmed < .75*raw: raise RuntimeError('Wallet raw grouping unsafe')
    rows,excluded=c['build'](first,grouped,a)
    if len(rows)<.95*len(first): raise RuntimeError('Wallet first-fill join coverage too low')
    pool=[r for r in rows if .20<=r['price']<args.upper_first_price]
    signals=[]
    for r in pool:
        if args.strategy == 'first':
            signals.append({**r,'signal_ts':int(r['ts']),'repeat_price':float(r['price']),
                            'signal_age':0})
            continue
        t=r['first_same']
        if not t: continue
        age=t['ts']-r['ts']
        # Genuine *first* repeat after 3 seconds must be inside bounds. This
        # avoids retrospectively selecting a later repeat if one came earlier.
        if not (args.min_repeat_age<=age<=args.max_repeat_age): continue
        if r['prior_opposite'] or (r['first_opp'] and r['first_opp']['ts']<=t['ts']): continue
        signals.append({**r,'signal_ts':int(t['ts']),'repeat_price':float(t['price']),
                        'signal_age':age})
    print(f'Wallet raw={raw} first={len(first)} joined={len(rows)} excluded={dict(excluded)}')
    print(f'Strategy={args.strategy}, first-price cohort={len(pool)}, signals={len(signals)} '
          f'(minAge={args.min_repeat_age}s,maxAge={args.max_repeat_age}s)')
    return sorted(signals,key=lambda r:r['signal_ts'])


def trade_price(t):
    a=t.get('yes_price_dollars')
    if a is not None: return float(a)
    if t.get('yes_price') is not None:
        p=float(t['yes_price']); return p/100 if p>=1 else p
    raise ValueError('No YES price in Kalshi trade')


def public_trades_api(ticker,min_ts,max_ts,cutoff):
    # Official /historical/trades vs /markets/trades split uses cutoffs.
    route='/historical/trades' if max_ts < cutoff else '/markets/trades' if min_ts >= cutoff else None
    if route is None:
        return (public_trades_api(ticker,min_ts,math.floor(cutoff)-1,cutoff)+
                public_trades_api(ticker,math.floor(cutoff),max_ts,cutoff))
    result=[]
    for t in pages(route,{'ticker':ticker,'min_ts':min_ts,'max_ts':max_ts,'limit':1000,
                          'is_block_trade':'false'},'trades',max_pages=200):
        if t.get('ticker')==ticker: result.append(t)
    return result


def make_trade_db(paired,args):
    by_ticker=defaultdict(list)
    if args.trades_file:
        for t in objrows(args.trades_file):
            if t.get('ticker'): by_ticker[t['ticker']].append(t)
    else:
        cutoff_data=api_get('/historical/cutoff')
        cutoff=timestamp(cutoff_data.get('trades_created_ts'))
        if cutoff is None: raise RuntimeError('Historical trade cutoff unavailable')
        groups=defaultdict(list)
        for s in paired: groups[s['ticker']].append(s)
        for idx,(ticker,entries) in enumerate(sorted(groups.items())):
            lo=min(x['signal_ts'] for x in entries)-1
            hi=max(x['signal_ts'] for x in entries)+max(args.latencies)+args.max_wait_s+2
            by_ticker[ticker]=public_trades_api(ticker,int(lo),int(hi),cutoff)
            if (idx+1)%20==0:
                print(f'Fetched trade prints for {idx+1}/{len(groups)} matched Kalshi markets',flush=True)
    prepared=defaultdict(lambda: defaultdict(list))
    for ticker,items in by_ticker.items():
        for t in items:
            if t.get('is_block_trade') in (True,'true','True'): continue
            buyer=str(t.get('taker_outcome_side') or t.get('taker_side') or '').lower()
            if buyer not in ('yes','no'): continue
            try:
                ts=timestamp(t.get('created_time') or t.get('ts'))
                price=trade_price(t)
                if ts is None or not (0 < price < 1): continue
                px=price if buyer=='yes' else 1-price
                qty=float(t.get('count_fp',t.get('count',0)))
            except (TypeError,ValueError): continue
            if qty < args.qty: continue
            prepared[ticker][buyer].append((ts,px))
        for v in prepared[ticker].values(): v.sort()
    return prepared


def make_bbo_db(args):
    if not args.bbo_file: return None
    prepared=defaultdict(list)
    for r in objrows(args.bbo_file):
        ticker=str(r.get('ticker') or r.get('market_ticker') or '')
        if not ticker.startswith('KXBTC15M-'): continue
        v=r.get('ts_recv',r.get('timestamp',r.get('ts')))
        try:
            ts=timestamp(v)
            yes=float(r.get('yes_ask') or r.get('yes_ask_dollars'))
            no_raw=r.get('no_ask') or r.get('no_ask_dollars')
            no=float(no_raw) if no_raw is not None else None
            yes_size=float(r.get('yes_ask_size') or r.get('yes_ask_size_fp') or 0)
            no_size=float(r.get('no_ask_size') or r.get('no_ask_size_fp') or 0)
            # We do not infer missing NO ask without historical YES bid.
            if no is None:
                bid=r.get('yes_bid') or r.get('yes_bid_dollars')
                no=1-float(bid) if bid is not None else None
            if no is None or ts is None or not (0<yes<1 and 0<no<1): continue
            prepared[ticker].append((ts,yes,no,yes_size,no_size))
        except (ValueError,TypeError): continue
    for arr in prepared.values(): arr.sort()
    return prepared


def lookup(paired,latency,trade_db,bbo_db,args):
    successes=[]; reasons=Counter()
    for s in paired:
        when=s['signal_ts']+latency
        if when>=s['start']+900:
            reasons['too_late']+=1;continue
        price=None;at=None
        if bbo_db is None:
            ticks=trade_db[s['ticker']][s['kalshi_side']]
            if not ticks: reasons['no_aggressor_side_trades']+=1;continue
            j=bisect.bisect_left(ticks,(when,-1))
            if j>=len(ticks) or ticks[j][0]>when+args.max_wait_s:
                reasons['no_fresh_trade_print']+=1;continue
            at,price=ticks[j]
        else:
            book=bbo_db.get(s['ticker'],[])
            if not book: reasons['no_bbo_market']+=1;continue
            j=bisect.bisect_right(book,(when,float('inf')))-1
            if j<0 or when-book[j][0]>args.max_quote_age_s:
                reasons['no_fresh_bbo']+=1;continue
            at,yes,no,ysize,nsize=book[j]
            price= yes if s['kalshi_side']=='yes' else no
            available=ysize if s['kalshi_side']=='yes' else nsize
            if available<args.qty:
                reasons['insufficient_top_depth']+=1;continue
        if not (0<price<1): reasons['bad_price']+=1;continue
        if args.max_kalshi_price is not None and price>args.max_kalshi_price:
            reasons['above_kalshi_price_cap']+=1; continue
        successes.append({**s,'kalshi_price':price,'price_ts':at,'latency':latency})
    return successes,reasons


def fee(price,qty,multiplier):
    # General taker formula rounded UP to cent for one fill/order.
    return math.ceil(100*(multiplier*.07*qty*price*(1-price))-1e-12)/100


def summary(label,filled,total,qty,multiplier,stress=0.0):
    if not filled:
        print(f'{label:<29} 0/{total} priced'); return
    costs=fees=payoff=0.
    for r in filled:
        p=min(.9999,r['kalshi_price']+stress)
        costs+=p*qty
        fees+=fee(p,qty,multiplier)
        payoff+=r['kalshi_win']*qty
    pnl=payoff-costs-fees
    wins=sum(r['kalshi_win'] for r in filled)
    print(f'{label:<29} {len(filled):3}/{total:<3} win={wins/len(filled)*100:5.1f}% '
          f'median={median(min(.9999,r["kalshi_price"]+stress) for r in filled):.3f} '
          f'cost=${costs+fees:8.2f} PnL=${pnl:+8.2f} netROI={100*pnl/(costs+fees):+7.2f}%')


def self_test():
    assert timestamp('2026-10-18T14:00:00Z')==1792332000
    assert timestamp(1792332000000000000)==1792332000
    markets=[{'ticker':'KXBTC15M-A','close_time':'2026-10-18T14:15:00Z','result':'no'}]
    s=[dict(start=1792332000,signal_ts=1792332015,side='DOWN',won=1,price=.31,repeat_price=.32)]
    matched,errors=match_markets(s,markets)
    assert len(matched)==1 and not errors and matched[0]['kalshi_win']==1
    trade_db={'KXBTC15M-A':{'no':[(1792332015,.40),(1792332018,.43),(1792332024,.49)]}}
    class A: max_wait_s=5; max_quote_age_s=2; qty=1; max_kalshi_price=None
    xs,rs=lookup(matched,3,trade_db,None,A)
    assert not rs and len(xs)==1 and xs[0]['kalshi_price']==.43
    xs,rs=lookup(matched,15,trade_db,None,A)
    assert not xs and rs['no_fresh_trade_print']==1
    assert fee(.4,10,1)==.17
    print('PASS: market matching, own settlement, second-level trade direction, latency, no-stale-data, fee')


def main():
    p=argparse.ArgumentParser(description='Read-only Kalshi copy-replay of Polymarket wallet 954B')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--wallet',type=Path,default=None)
    p.add_argument('--source',choices=['api','files'],default='api')
    p.add_argument('--markets-file',type=Path,help='Local Kalshi markets JSON or JSONL')
    p.add_argument('--trades-file',type=Path,help='Local Kalshi public trades JSONL/CSV (not BBO)')
    p.add_argument('--bbo-file',type=Path,help='Historical Kalshi BBO CSV/JSONL/parquet with ask sizes')
    p.add_argument('--strategy',choices=['first','repeat'],default='repeat',
                   help='first=copy first wallet buy; repeat=early same-side second buy')
    p.add_argument('--max-kalshi-price',type=float,default=None,
                   help='Optional entry-time Kalshi price cap, e.g. 0.60')
    p.add_argument('--min-repeat-age',type=int,default=4)
    p.add_argument('--max-repeat-age',type=int,default=30)
    p.add_argument('--upper-first-price',type=float,default=.40)
    p.add_argument('--latencies',type=int,nargs='+',default=[0,3,5,10,20])
    p.add_argument('--max-wait-s',type=int,default=5,help='First buyer-aggressor print no later than this delay')
    p.add_argument('--max-quote-age-s',type=float,default=2.,help='BBO snapshot freshness bound')
    p.add_argument('--qty',type=int,default=10)
    p.add_argument('--fee-multiplier',type=float,default=1.)
    args=p.parse_args()
    if args.self_test: self_test(); return
    if args.qty<=0 or args.fee_multiplier<0 or args.min_repeat_age<4 or args.max_repeat_age>300:
        raise SystemExit('Invalid quantity/fee/repeat-age arguments')
    if args.source=='files' and not args.markets_file:
        raise SystemExit('--source files requires --markets-file')
    if args.source=='files' and not (args.trades_file or args.bbo_file):
        raise SystemExit('--source files requires --trades-file or --bbo-file')
    signals=extract_signals(args)
    if not signals: raise SystemExit('No qualifying wallet signals')
    print('Fetching/reading Kalshi market metadata (READ ONLY)...',flush=True)
    mk=list(objrows(args.markets_file)) if args.markets_file else markets_api(signals)
    paired,miss=match_markets(signals,mk)
    print(f'Markets read={len(mk)} paired={len(paired)}/{len(signals)} excludes={dict(miss)}')
    if not paired: raise SystemExit('No exact matching KXBTC15M window & settled result')
    print(f'Polymarket/Kalshi settlement disagreements: {sum(s["disagree"] for s in paired)}/{len(paired)}')
    print('Market side mapping: Polymarket UP->Kalshi YES; DOWN->NO; outcomes = Kalshi official result')
    trade_db=None
    bbo_db=make_bbo_db(args)
    if bbo_db is None:
        print('CAUTION: TRADE-PRINT PROXY MODE. Prints are NOT executable asks or fill proof.')
        trade_db=make_trade_db(paired,args)
    else: print('BBO MODE: using as-of ask with size; queue/crossing slippage still unverified.')
    print('General taker-fee formula with multiplier, rounded up to cents per order; hold to expiry.')
    for latency in args.latencies:
        hit,reasons=lookup(paired,latency,trade_db,bbo_db,args)
        print(f'\nLATENCY: {latency}s   missing={dict(reasons)}')
        summary('price as observed + fee',hit,len(paired),args.qty,args.fee_multiplier)
        for premium in (.01,.03,.05):
            summary(f'+{premium*100:.0f}c entry stress + fee',hit,len(paired),args.qty,args.fee_multiplier,premium)
        if latency in (0,5,10):
            for week in sorted(set(r['week'] for r in paired)):
                sub=[r for r in hit if r['week']==week]
                total=sum(1 for r in paired if r['week']==week)
                summary(f'  {week}',sub,total,args.qty,args.fee_multiplier)
    print('\nRESEARCH WARNINGS:')
    print('- Trade-print mode cannot establish whether the quoted price or quantity was buyable by us.')
    print('- BBO top-of-book size and freshness improve realism; historical latency/market impact remain uncertain.')
    print('- Matching requires exact KXBTC15M close timestamps; unmatched markets never get imputed.')
    print('- All first-price filters refer to historical Polymarket first fills, NOT current Kalshi prices.')
    print('- This is a cross-venue signal replay. All outcomes come from Kalshi, not Polymarket.')
    print('- No market orders, live trading, file writes, or GitHub changes.')


if __name__=='__main__':
    try: main()
    except (RuntimeError,ValueError,HTTPError,URLError) as e:
        sys.exit('ABORT (no partial results to trust): '+str(e))
