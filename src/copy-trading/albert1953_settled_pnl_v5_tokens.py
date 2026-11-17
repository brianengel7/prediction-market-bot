#!/usr/bin/env python3
"""Albert1953 1%-ratio settled-profit audit using Data API /v2/tokens.

The V4 resolutions-only join returned zero verified markets. This v5 uses the
venue's token reference endpoint, which supplies resolved and final_price for
both v1 CTF and newer position tokens. Uses existing FULL v1 fill cache.
No orders, no authentication, no imports of trading execution code.

STRICT SCOPE: a market is counted only when two same-module binary tokens are
returned, both are explicitly resolved, both have explicit final prices whose
sum is one, and source trade token/outcome indexes exactly agree with the API.
It excludes ambiguous source inventory, unrelated minted/transferred shares,
and markets without complete evidence. PnL is at the SOURCE wallet's prints,
not a historical executable follower or a Kalshi backtest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal as D
import csv, json, sys, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import albert1953_copy_sell_audit_1pct_v3 as v3
import albert1953_settled_pnl_v4 as v4

BASE = 'https://data-api.polymarket.com/v2/tokens'
ZERO, ONE = D(0), D(1)
VERSION = 5


def get_json(params, tries=6):
    url = BASE+'?'+urlencode(params)
    for n in range(tries):
        try:
            req = Request(url, headers={'User-Agent':'albert1953-settled-token-audit-v5/1.0',
                                        'Accept':'application/json'})
            with urlopen(req,timeout=40) as resp:
                data=json.load(resp)
            if not isinstance(data,dict) or not isinstance(data.get('data'),list):
                raise ValueError('Unexpected token API data envelope; stopping')
            return data['data']
        except HTTPError as e:
            if e.code not in (429,500,502,503,504) or n == tries-1: raise
            try: wait=float(e.headers.get('Retry-After',''))
            except (TypeError,ValueError): wait=min(30,2**n)
            time.sleep(min(60,max(0.5,wait)))
        except (URLError,TimeoutError):
            if n==tries-1: raise
            time.sleep(min(30,2**n))
    raise AssertionError('unreachable')


def cache_fetch(cids, path:Path, *, offline=False, probe=0, refresh_missing=False):
    """Resume one immutable condition-scoped token query cache (up to 10 IDs/call)."""
    cache={'version': VERSION,'wallet':v3.WALLET,'queried':[],'rows_by_condition':{}}
    if path.exists():
        cache=json.loads(path.read_text())
        if cache.get('version')!=VERSION or cache.get('wallet')!=v3.WALLET:
            raise RuntimeError('Wrong token cache version/wallet')
    queried=set(cache.get('queried',[]))
    wanted=list(cids[:probe] if probe else cids)
    if offline:
        missing=[c for c in wanted if c not in queried]
        if missing: raise RuntimeError(f'{len(missing)} conditions missing from token cache')
        return cache
    missing=[]
    for c in wanted:
        if c not in queried or (refresh_missing and not cache['rows_by_condition'].get(c)):
            missing.append(c)
    for start in range(0,len(missing),10):
        batch=missing[start:start+10]
        data=get_json({'condition': ','.join(batch)})
        grouped=defaultdict(list)
        for record in data:
            if not isinstance(record,dict):raise RuntimeError('Bad /v2/tokens record')
            cid=str(record.get('condition_id') or '').lower()
            if cid in batch: grouped[cid].append(record)
        for cid in batch:
            cache['rows_by_condition'][cid]=grouped[cid]
            queried.add(cid)
        cache['queried']=sorted(queried)
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache,separators=(',',':')))
        tmp.replace(path)
        n=min(start+10,len(missing))
        if n%200==0 or n==len(missing):
            rows=sum(bool(cache['rows_by_condition'].get(c)) for c in wanted if c in queried)
            print(f'Tokens lookup: {n:,}/{len(missing):,} new conditions checked; '
                  f'nonempty responses {rows:,}/{len(wanted):,}',flush=True)
        time.sleep(0.08)
    return cache


def dec(x):
    try:
        if x is None or isinstance(x,bool):return None
        d=D(str(x))
        return d if d.is_finite() else None
    except (ValueError,TypeError,ArithmeticError):return None


def verify_market(cid, rows, api_rows, token_index):
    """Return (payout_by_index, reason); strict token and resolution verification."""
    if not api_rows:return None,'no_token_rows'
    source_tokens={r['token'] for r in rows}
    token_records=[]
    for tok in source_tokens:
        matches=[p for p in api_rows if str(p.get('token_id'))==tok and
                 str(p.get('condition_id') or '').lower()==cid]
        if len(matches)!=1:return None,'missing_or_duplicate_source_token'
        token_records.append(matches[0])
    modules={str(r.get('module')) for r in token_records}
    if len(modules)!=1:return None,'mixed_modules'
    module=next(iter(modules))
    if not module or module=='None':return None,'missing_module'
    family=[r for r in api_rows if str(r.get('module'))==module and
            str(r.get('condition_id') or '').lower()==cid]
    if len(family)!=2:return None,'not_exact_binary_family'
    by_index={}
    for r in family:
        ix=r.get('outcome_index')
        try: ix=int(ix)
        except (ValueError,TypeError):return None,'bad_index'
        if ix not in (0,1) or ix in by_index:return None,'bad_or_duplicate_indices'
        if r.get('resolved') is not True:return None,'not_resolved'
        f=dec(r.get('final_price'))
        if f is None or not ZERO<=f<=ONE:return None,'missing_or_invalid_final_price'
        by_index[ix]=f
    if len(by_index)!=2 or by_index[0]+by_index[1]!=ONE:
        return None,'not_complementary_payout'
    for r in rows:
        key=(cid,r['token'])
        indexed=token_index.get(key,set())
        if len(indexed)!=1 or next(iter(indexed)) not in ('0','1'):
            return None,'invalid_cached_trade_outcome_index'
        tr_ix=int(next(iter(indexed)))
        tr_api=[p for p in token_records if str(p['token_id'])==r['token']]
        if len(tr_api)!=1:return None,'source_api_outcome_index_disagreement'
        try: api_ix=int(tr_api[0].get('outcome_index'))
        except (TypeError,ValueError):return None,'invalid_api_trade_token_index'
        if api_ix!=tr_ix:return None,'source_api_outcome_index_disagreement'
    return by_index,None


def make_states(rows, api_by_condition, bad, ambiguous, token_index):
    by_market=defaultdict(list)
    for r in rows:by_market[r['condition'].lower()].append(r)
    states={}
    failures=Counter()
    for cid,rs in by_market.items():
        if any((r['condition'],r['token']) in bad for r in rs):
            failures['negative_inventory']+=1;continue
        if any((r['condition'],r['token']) in ambiguous for r in rs):
            failures['ambiguous_same_second']+=1;continue
        payoff,reason=verify_market(cid,rs,api_by_condition.get(cid,[]),token_index)
        if reason:failures[reason]+=1;continue
        mapping={}
        for r in rs:
            x=token_index[(cid,r['token'])]
            mapping[r['token']]=int(next(iter(x)))
        states[cid]={'rows':rs,'outcome_map':mapping,'payout':[payoff[0],payoff[1]]}
    failures['verified_settled_markets']=len(states)
    return states,failures


def f6(v):return format(v.quantize(D('0.000001')),'f') if isinstance(v,D) else v


def produce(states,meta,ratio):
    scenarios=(('whole_no_min',ONE,ZERO),('whole_one_usd_min',ONE,ONE),
               ('fractional_no_min',D('.01'),ZERO),('fractional_one_usd_min',D('.01'),ONE))
    report={}; market_csv=[]
    for name,step,minimum in scenarios:
        vals=[]
        for cid,m in states.items():
            o=v4.simulate_one(m,ratio,step,minimum)
            recent=any(r['ts']>=meta['end_exclusive']-90*86400 for r in m['rows'])
            rec={'condition_id':cid,'title':m['rows'][0]['title'],
                 'category':m['rows'][0]['category'], 'last_90d_trade_in_market':recent,
                 'first_trade_utc':datetime.fromtimestamp(m['rows'][0]['ts'],timezone.utc).isoformat(),
                 'last_trade_utc':datetime.fromtimestamp(m['rows'][-1]['ts'],timezone.utc).isoformat(),
                 'scenario':name,**o}
            vals.append(rec)
            market_csv.append(rec)
        out={}
        for period in ('all','markets_traded_last_90d'):
            selected=[r for r in vals if period=='all' or r['last_90d_trade_in_market']]
            invested=sum((r['copy_buy_usdc'] for r in selected),ZERO)
            pnl=sum((r['copy_pnl'] for r in selected),ZERO)
            original=sum((r['source_pnl'] for r in selected),ZERO)
            winners=sorted((r['copy_pnl'] for r in selected if r['copy_pnl']>ZERO), reverse=True)
            by_cat=defaultdict(lambda:{'markets':0,'buy_spend':ZERO,'gross_pnl':ZERO})
            for r in selected:
                a=by_cat[r['category']]
                a['markets']+=1;a['buy_spend']+=r['copy_buy_usdc'];a['gross_pnl']+=r['copy_pnl']
            out[period]={'markets':len(selected),'copy_buy_spend':f6(invested),
                         'copy_gross_pnl':f6(pnl),'copy_roi_gross_buy_spend_pct':f6(100*pnl/invested) if invested else None,
                         'source_gross_pnl':f6(original),'source_scaled_gross_pnl':f6(original*ratio),
                         'copy_buy_events':sum(r['copy_buy_events'] for r in selected),
                         'copy_sell_events':sum(r['copy_sell_events'] for r in selected),
                         'positive_pnl_markets':sum(r['copy_pnl']>ZERO for r in selected),
                         'pnl_excluding_top5_wins':f6(pnl-sum(winners[:5],ZERO)),
                         'pnl_excluding_top10_wins':f6(pnl-sum(winners[:10],ZERO)),
                         'category':{cat:{k:f6(v) for k,v in agg.items()} for cat,agg in by_cat.items()}}
        report[name]=out
    return report,market_csv


def self_test():
    cid='0x'+'a'*64
    def t(token,ix,p,resolved=True,module='v1_ctf'):
        return {'token_id':token,'condition_id':cid,'outcome_index':ix,'module':module,
                'resolved':resolved,'final_price':p}
    source=[{'token':'t0','condition':cid,'side':'BUY','ts':1}]
    mapping={(cid,'t0'):{'0'}}
    p,e=verify_market(cid,source,[t('t0',0,1),t('t1',1,0)],mapping)
    assert e is None and p=={0:ONE,1:ZERO},(p,e)
    assert verify_market(cid,source,[t('t0',0,1,False),t('t1',1,0)],mapping)[1]=='not_resolved'
    assert verify_market(cid,source,[t('t0',0,1),t('t1',1,1)],mapping)[1]=='not_complementary_payout'
    assert verify_market(cid,source,[t('t0',0,0.5),t('t1',1,0.5)],mapping)[0]=={0:D('.5'),1:D('.5')}
    assert verify_market(cid,source,[t('t0',0,1),t('t1',1,0,True,'v2')],mapping)[1]=='not_exact_binary_family'
    assert verify_market(cid,source,[t('t0',0,1),t('t1',1,0)],{(cid,'t0'):{'1'}})[1]=='source_api_outcome_index_disagreement'
    m={'rows':[{'token':'t0','side':'BUY','q':D('200'),'price':D('.4')},
               {'token':'t0','side':'SELL','q':D('100'),'price':D('.6')}],
       'outcome_map':{'t0':0},'payout':[ONE,ZERO]}
    z=v4.simulate_one(m,D('.01'),ONE,ZERO)
    assert z['copy_pnl']==D('0.8') and z['copy_sell_events']==1,z
    print('PASS: exact tokens, modules, source outcome mapping, finality, payout vectors, 1% buys + sells')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    a.add_argument('--ratio',default='0.01',help='share multiplier; default 0.01 (1%%)')
    a.add_argument('--offline',action='store_true')
    a.add_argument('--probe',type=int,default=0,help='Look up just N conditions for endpoint inspection; do not score P&L')
    a.add_argument('--refresh-missing',action='store_true',help='Retry previously empty condition lookups')
    a.add_argument('--self-test',action='store_true')
    ns=a.parse_args()
    if ns.self_test:return self_test()
    ratio=dec(ns.ratio)
    if ratio is None or not ZERO<ratio<=ONE:a.error('ratio must be positive and <=1')
    if ns.probe<0:a.error('probe must be >=0')
    rows,meta,order=v3.read_source(ns.cache)
    v3.market_categories(rows)
    bad,amb=v3.validate_token_inventory(rows)
    indexed=v4.outcomes_by_token(ns.cache/'albert1953_trades.jsonl')
    last_ts={}
    for r in rows:
        cid=r['condition'].lower()
        last_ts[cid]=max(last_ts.get(cid,0),r['ts'])
    ids=sorted(last_ts,key=lambda c:(-last_ts[c],c))
    path=ns.cache/'albert1953_v5_token_resolution_cache.json'
    if ns.probe:
        n=min(ns.probe,len(ids))
        ix=[round(i*(len(ids)-1)/max(1,n-1)) for i in range(n)]
        lookup_ids=[ids[i] for i in sorted(set(ix))]
    else:lookup_ids=ids
    token_cache=cache_fetch(lookup_ids,path,offline=ns.offline,probe=0,
                            refresh_missing=ns.refresh_missing)
    api_by=token_cache['rows_by_condition']
    if ns.probe:
        samples=lookup_ids[:min(5,len(lookup_ids))]+lookup_ids[-min(5,len(lookup_ids)):]
        print('PROBE SUMMARY: conditions=',len(lookup_ids),
              'nonempty=',sum(bool(api_by.get(c)) for c in lookup_ids),
              'resolved_token_rows=',sum(1 for c in lookup_ids for r in api_by.get(c,[])
                                        if r.get('resolved') is True and r.get('final_price') is not None))
        for cid in dict.fromkeys(samples):
            rs=api_by.get(cid,[])
            print('PROBE',cid,'tokens',len(rs),
                  'fields', [{k:r.get(k) for k in ('token_id','module','outcome_index','resolved','final_price')}
                            for r in rs[:4]])
        print('PROBE ONLY: no P&L scored. Cached results; now run without --probe.')
        return
    states,checks=make_states(rows,api_by,bad,amb,indexed)
    print(f'ALBERT1953 V5: source fills={len(rows):,} markets={len(ids):,} '
          f'ratio={ratio} feed={order}')
    print('Token validation:',dict(checks))
    print('Verified settled coverage: %.2f%% (%d/%d markets)' % (100*len(states)/len(ids),len(states),len(ids)))
    if len(states)/len(ids)<.5:
        print('WARNING: less than 50%% of markets verified; PnL of verified subset may be severely selection-biased.')
    if not states:
        print('NO VERIFIED MARKETS. STOPPING; no zero-P&L report.',file=sys.stderr)
        raise SystemExit(2)
    report,market_csv=produce(states,meta,ratio)
    suffix='ratio'+format(ratio,'f').replace('.','p')
    outfile=ns.cache/f'albert1953_settled_v5_{suffix}_report.json'
    marketfile=ns.cache/f'albert1953_settled_v5_{suffix}_markets.csv'
    out={'wallet':v3.WALLET,'ratio':str(ratio),'source_fills':len(rows),'source_markets':len(ids),
         'verified_market_count':len(states),'unverified_market_count':len(ids)-len(states),
         'verified_coverage_pct':round(len(states)/len(ids)*100,2),
         'validation':dict(checks),'source':'Polymarket Data API /v2/tokens resolved+final_price',
         'basis':'verified settled-only, wallet-printed-price 1% copy proxy, gross before fees',
         'important_exclusions':'unverified and open markets; settlement transfers/splits, actual depth and latency, fees, Kalshi market semantics',
         'recent_window':'markets with ANY source trade in last 90 days; full recorded market lifetime PnL, NOT a 90-day investment return',
         'scenarios':report}
    outfile.write_text(json.dumps(out,indent=2))
    fields=('condition_id','title','category','last_90d_trade_in_market','first_trade_utc','last_trade_utc','scenario',
            'source_pnl','copy_pnl','source_buy_usdc','copy_buy_usdc','copy_sell_usdc','copy_payout_usdc',
            'source_sell_usdc','source_payout_usdc','source_buy_events','source_sell_events','copy_buy_events','copy_sell_events',
            'net_copy_shares','source_net_shares')
    with marketfile.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for rec in market_csv:
            writer.writerow({k:f6(rec.get(k)) for k in fields})
    for name,periods in report.items():
        print('\n'+name)
        for label,q in periods.items():
            print(' ',label, 'markets',q['markets'], 'gross PnL $'+q['copy_gross_pnl'],
                  'gross BUY spend $'+q['copy_buy_spend'], 'ROI %',q['copy_roi_gross_buy_spend_pct'],
                  'ex-top5 $'+q['pnl_excluding_top5_wins'])
    print('REPORT',outfile,'\nMARKETS',marketfile,'\nREAD ONLY. NO ORDERS. NOT EXECUTABLE FOLLOWER P&L.')

if __name__=='__main__':main()
