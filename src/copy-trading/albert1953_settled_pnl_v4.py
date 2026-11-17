#!/usr/bin/env python3
"""Albert1953: STRICT resolved-market P&L on 0.01 proportional BUY+SELL replay.

READ ONLY: public Polymarket Data API resolution GETs. No trading, private keys,
price replay, book snapshots, or changes to other project files. Reads full v1
trade cache, and uses v3 for chronological input and source-inventory checks.

Only reports P&L for verified binary markets with explicit final per-token
payouts and unambiguous source inventory. A trade-only synthetic follower is
replayed at the source execution prices; it is NOT a Kalshi backtest.

P&L = copied sell proceeds + payout of copied residual tokens - copied buy costs.
Fees, fills, latency, slippage, splits/merges, rebates, FX and cross-market
hedges are excluded. Unresolved/unsupported markets are NOT assigned zero P&L.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json, csv, time, math

import albert1953_copy_sell_audit_1pct_v3 as v3

API = 'https://data-api.polymarket.com/v2/resolutions'
ZERO = D('0')
ONE = D('1')
EPS = D('0.0000001')


def fmt(value):
    return str(value.quantize(D('0.000001'))) if isinstance(value, D) else str(value)


def get_json(url, tries=7):
    for attempt in range(tries):
        try:
            req=Request(url,headers={'User-Agent':'albert1953-research-v4/1.0','Accept':'application/json'})
            with urlopen(req,timeout=45) as res:
                return json.load(res)
        except HTTPError as exc:
            if exc.code not in (429,500,502,503,504) or attempt==tries-1:
                raise
            try: delay=float(exc.headers.get('Retry-After',''))
            except ValueError: delay=min(30,2**attempt)
            time.sleep(min(60,max(0.5,delay)))
        except (URLError,TimeoutError):
            if attempt==tries-1:raise
            time.sleep(min(30,2**attempt))
    raise AssertionError('unreachable')


def fetch_resolution_cache(ids, output, offline=False):
    ids=set(ids)
    cache = {'version':1, 'wallet':v3.WALLET, 'conditions':{}, 'queried':[]}
    if output.exists():
        cache=json.loads(output.read_text())
        if cache.get('version')!=1 or cache.get('wallet')!=v3.WALLET:
            raise ValueError('Resolution cache belongs to a different schema or wallet')
        cache.setdefault('conditions',{})
        cache.setdefault('queried',[])
    queried=set(cache['queried'])
    missing=sorted(ids-queried)
    if offline:
        if missing:
            raise ValueError(f'Offline cache missing {len(missing)} market condition IDs; run without --offline first')
        return cache['conditions']
    for pos in range(0,len(missing),20):
        batch=missing[pos:pos+20]
        payload=get_json(API+'?'+urlencode({'condition':','.join(batch)}))
        data=payload.get('data')
        if not isinstance(data,list):raise ValueError('Invalid /v2/resolutions API envelope')
        for record in data:
            if not isinstance(record,dict):raise ValueError('Non-object resolution record')
            cid=str(record.get('condition_id') or record.get('conditionId') or '').lower()
            if cid in batch:
                # On duplicate rows, accept the latest resolved explicit-payout row.
                existing=cache['conditions'].get(cid)
                if existing is None or valid_payout(record) is not None:
                    cache['conditions'][cid]=record
        queried.update(batch)
        cache['queried']=sorted(queried)
        output.write_text(json.dumps(cache,indent=1))
        if ((pos//20)+1)%20==0 or pos+20>=len(missing):
            print('Resolution lookups: %d/%d conditions checked' % (min(pos+20,len(missing)),len(missing)),flush=True)
        time.sleep(0.05)
    return cache['conditions']


def valid_payout(r):
    if not isinstance(r,dict) or str(r.get('status') or '').lower()!='resolved':return None
    vals=r.get('payouts')
    if not isinstance(vals,list) or len(vals)!=2:return None
    try: arr=[D(str(x)) for x in vals]
    except Exception:return None
    if any(not x.is_finite() or x<ZERO for x in arr):return None
    # V2 per-outcome payout units: micro-USDC/share. Allow actual unit 0..1 as well.
    total=sum(arr,ZERO)
    if abs(total-D('1000000'))<=D('1'): return [x/D('1000000') for x in arr]
    if abs(total-ONE)<D('0.000001'):return arr
    return None


def outcomes_by_token(fp):
    by_token=defaultdict(set)
    with fp.open() as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            by_token[(str(r.get('condition_id') or '').lower(),str(r.get('token_id') or ''))].add(str(r.get('outcome_index','')))
    return by_token


def prepare(rows,token_index,bad,amb,resolutions):
    grouped=defaultdict(list)
    for r in rows:
        cid=r['condition'].lower()
        grouped[cid].append(r)
    states={}
    counts=Counter()
    for cid,market_rows in grouped.items():
        if any((r['condition'],r['token']) in bad for r in market_rows):
            counts['negative_inventory_excluded']+=1;continue
        if any((r['condition'],r['token']) in amb for r in market_rows):
            counts['same_second_ambiguous_excluded']+=1;continue
        tok_map={}
        for r in market_rows:
            key=(cid,r['token'])
            ix=token_index.get(key,set())
            if len(ix)!=1 or next(iter(ix)) not in ('0','1'):
                tok_map=None;break
            tok_map[r['token']]=int(next(iter(ix)))
        if tok_map is None or len(set(tok_map.values()))!=len(tok_map):
            counts['unverified_binary_token_mapping']+=1;continue
        result=resolutions.get(cid)
        payout=valid_payout(result)
        if payout is None:
            counts['no_verified_final_payout']+=1;continue
        states[cid]={'rows':market_rows,'outcome_map':tok_map,'payout':payout,
                     'resolved_at':str(result.get('resolvedAt') or result.get('resolved_at') or '')}
        counts['verified_resolved_markets']+=1
    return states,counts


def simulate_one(market,ratio,step,min_notional):
    """Price proxy only; tracks per-token source inventory, copied inventory."""
    src=defaultdict(lambda:ZERO)
    held=defaultdict(lambda:ZERO)
    pay=market['payout']
    ix=market['outcome_map']
    buy=sell=src_buy=src_sell=ZERO
    n_buy=n_sell=0
    n_copied_buy=n_copied_sell=0
    for r in market['rows']:
        tok=r['token'];q=r['q'];p=r['price']
        if r['side']=='BUY':
            src[tok]+=q
            src_buy+=q*p;n_buy+=1
        else:
            src[tok]-=q
            if src[tok]<-EPS:raise AssertionError('Source inventory negative')
            if -EPS<src[tok]<ZERO:src[tok]=ZERO
            src_sell+=q*p;n_sell+=1
        desired=v3.exact_floor(src[tok]*ratio,step)
        delta=desired-held[tok]
        if r['side']=='BUY' and delta>=step:
            if delta*p>=min_notional:
                held[tok]+=delta;buy+=delta*p;n_copied_buy+=1
        elif r['side']=='SELL' and delta<=-step:
            qsell=min(held[tok],-delta)
            if qsell>=step and qsell*p>=min_notional:
                held[tok]-=qsell;sell+=qsell*p;n_copied_sell+=1
    payout=sum(held[t]*pay[ix[t]] for t in held)
    original_payout=sum(src[t]*pay[ix[t]] for t in src)
    return {'source_pnl':src_sell+original_payout-src_buy,
            'copy_pnl':sell+payout-buy,
            'source_buy_usdc':src_buy,'copy_buy_usdc':buy,
            'copy_sell_usdc':sell,'copy_payout_usdc':payout,
            'source_sell_usdc':src_sell,'source_payout_usdc':original_payout,
            'copy_sell_events':n_copied_sell,'copy_buy_events':n_copied_buy,
            'source_buy_events':n_buy,'source_sell_events':n_sell,
            'net_copy_shares':sum(held.values()),
            'source_net_shares':sum(src.values())}


def self_test():
    def r(i,side,q,p,tok='yes'):
        return {'condition':'a','token':tok,'side':side,'q':D(q),'price':D(p),'ts':i,
                'title':'Test','category':'finance/economics','slug':'test'}
    m={'rows':[r(1,'BUY','120','0.4'),r(2,'BUY','180','0.3'),r(3,'SELL','100','0.6')],
       'outcome_map':{'yes':0},'payout':[ONE,ZERO]}
    v=simulate_one(m,D('0.01'),ONE,ZERO)
    # Copy: 1 @.4, 2 @.3, sell 1 @.6, settle 2 @1 => +1.6
    assert v['copy_pnl']==D('1.6'),v
    assert v['source_pnl']==D('158'),v
    n=simulate_one(m,D('0.01'),ONE,D('1'))
    assert n['copy_pnl']==D('0') and n['copy_buy_usdc']==ZERO
    assert valid_payout({'status':'resolved','payouts':[1000000,0]})==[ONE,ZERO]
    assert valid_payout({'status':'disputed','payouts':[1000000,0]}) is None
    assert valid_payout({'status':'resolved','payouts':[500000,100000]}) is None
    print('PASS: source-price full inventory, partial sale, terminal payout, minimum order, payout state')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    ap.add_argument('--ratio',default='0.01',help='Target share multiplier; default 0.01 (1%%)')
    ap.add_argument('--offline',action='store_true',help='Use existing resolution cache without network')
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:return self_test()
    ratio=D(a.ratio)
    if not ratio.is_finite() or ratio<=ZERO or ratio>ONE:ap.error('0 < ratio <= 1 required')
    rows,meta,order=v3.read_source(a.cache)
    v3.market_categories(rows)
    excluded,ambiguous=v3.validate_token_inventory(rows)
    mapping=outcomes_by_token(a.cache/'albert1953_trades.jsonl')
    ids={r['condition'].lower() for r in rows}
    resolved=fetch_resolution_cache(ids,a.cache/'albert1953_v4_resolutions.json',a.offline)
    states,diagnostics=prepare(rows,mapping,excluded,ambiguous,resolved)
    periods={'all':lambda rs:True, 'last_90d_market_activity':lambda rs:any(r['ts']>=meta['end_exclusive']-90*86400 for r in rs)}
    scenarios=[('whole_no_min',ONE,ZERO),('whole_one_usd_min',ONE,ONE),
               ('fractional_no_min',D('0.01'),ZERO),('fractional_one_usd_min',D('0.01'),ONE)]
    market_data=[]
    results={}
    for name,step,minnot in scenarios:
        dat=[]
        for cid,m in states.items():
            val=simulate_one(m,ratio,step,minnot)
            datum={'condition':cid,'title':m['rows'][0]['title'],
                   'category':m['rows'][0]['category'],'first_trade_utc':datetime.fromtimestamp(m['rows'][0]['ts'],timezone.utc).isoformat(),
                   'last_trade_utc':datetime.fromtimestamp(m['rows'][-1]['ts'],timezone.utc).isoformat(),
                   'has_trade_last_90d':periods['last_90d_market_activity'](m['rows']),
                   'scenario':name,**val}
            dat.append(datum)
        market_data+=dat
        by_period={}
        for period,fn in periods.items():
            group=[d for d in dat if period=='all' or d['has_trade_last_90d']]
            srcp=sum((d['source_pnl'] for d in group),ZERO)
            cp=sum((d['copy_pnl'] for d in group),ZERO)
            spent=sum((d['copy_buy_usdc'] for d in group),ZERO)
            top=sorted((d['copy_pnl'] for d in group if d['copy_pnl']>ZERO),reverse=True)
            bycat=defaultdict(lambda:{'markets':0,'pnl':ZERO,'spend':ZERO})
            for d in group:
                c=bycat[d['category']];c['markets']+=1;c['pnl']+=d['copy_pnl'];c['spend']+=d['copy_buy_usdc']
            by_period[period]={'markets':len(group),'source_pnl_no_fees':fmt(srcp),
                'source_scaled_pnl_no_fees':fmt(srcp*ratio),'copy_pnl_no_fees':fmt(cp),
                'copy_buy_spend':fmt(spent),'copy_pnl_over_gross_buy_spend_pct':fmt(100*cp/spent) if spent else None,
                'positive_copy_pnl_markets':sum(d['copy_pnl']>ZERO for d in group),
                'excluding_top_5_positive_market_wins_pnl':fmt(cp-sum(top[:5],ZERO)),
                'excluding_top_10_positive_market_wins_pnl':fmt(cp-sum(top[:10],ZERO)),
                'category':{k:{'markets':v['markets'],'pnl':fmt(v['pnl']),'spend':fmt(v['spend'])} for k,v in bycat.items()}}
        results[name]=by_period
    suffix='ratio'+format(ratio,'f').replace('.','p')
    out=a.cache/f'albert1953_settled_v4_{suffix}_report.json'
    csv_out=a.cache/f'albert1953_settled_v4_{suffix}_markets.csv'
    report={'wallet':v3.WALLET,'ratio':str(ratio),'source_trade_count':len(rows),'source_market_count':len(ids),
            'trade_feed_order':order,'resolution_rows_returned':len(resolved),
            'resolution_checks':dict(diagnostics),'criteria':'Only verified resolved payout rows, binary token mapping, and nonnegative source inventory. No cross-token transfers/splits/merges modeled.',
            'result_kind':'SOURCE-PRICE GROSS FOLLOWER PROXY, NOT EXECUTABLE KALSHI OR POLYMARKET FOLLOWER',
            'excluded':'fees, historical quotes, slippage, fills, mint/burn, maker rewards, transfer flows, hedges, unresolved markets',
            'scenarios':results}
    out.write_text(json.dumps(report,indent=2))
    fields=['condition','title','category','first_trade_utc','last_trade_utc','has_trade_last_90d','scenario',
            'source_pnl','copy_pnl','source_buy_usdc','copy_buy_usdc','copy_sell_usdc','copy_payout_usdc',
            'source_sell_usdc','source_payout_usdc','copy_sell_events','copy_buy_events','source_buy_events','source_sell_events',
            'net_copy_shares','source_net_shares']
    with csv_out.open('w',newline='') as fh:
        wr=csv.DictWriter(fh,fieldnames=fields);wr.writeheader()
        for d in market_data:wr.writerow({k:fmt(v) if isinstance(v,D) else v for k,v in d.items()})
    print('ALBERT1953: STRICT VERIFIED RESOLVED MARKET SOURCE-PRICE COPY PROXY')
    print('Wallet',v3.WALLET,'ratio',ratio,'source fills',len(rows),'source markets',len(ids))
    print('Resolution diagnostics:',dict(diagnostics))
    print('Resolved-only P&L excludes markets missing final verified payout; it is not a mark-to-market return.')
    for name,res in results.items():
        print('\n'+name)
        for period,d in res.items():
            print('  '+period,'markets',d['markets'],'copied gross PnL $'+d['copy_pnl_no_fees'],
                  'buy spend $'+d['copy_buy_spend'],'ROI on gross buys',d['copy_pnl_over_gross_buy_spend_pct'],
                  'ex-top5 $'+d['excluding_top_5_positive_market_wins_pnl'])
    print('\nREPORT',out,'\nMARKETS',csv_out,'\nREAD ONLY. NO ORDERS.')

if __name__=='__main__':main()
