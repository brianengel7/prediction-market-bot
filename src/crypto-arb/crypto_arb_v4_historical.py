#!/usr/bin/env python3
"""V4.4 Coinbase dated futures historical study. READ ONLY: GET public market data; no orders.

This backtest is an early-exit CANDLE-OPEN PROXY, not executable or expiry-settled P&L.
Each dated contract contributes at most one trade per configuration. Walk-forward
selection uses contracts that had ALREADY EXPIRED before the next entry decision.
Historical order books are NOT provided by OHLCV endpoints; no fill claim is made.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

BASE = 'https://api.coinbase.com/api/v3/brokerage/market'
ASSETS = {'BTC':'BTC-USD', 'ETH':'ETH-USD', 'SOL':'SOL-USD'}
DATA_DIR = Path('src/crypto-arb/data/v4/historical')
GRANULARITY_SECONDS = {'ONE_HOUR':3600, 'FIVE_MINUTE':300, 'ONE_MINUTE':60}


def utc(value):
    out = pd.Timestamp(value)
    if pd.isna(out): raise ValueError('Missing UTC timestamp')
    if out.tz is None: raise ValueError('Unzoned timestamp rejected')
    return out.tz_convert('UTC')


def api_get(path, params=None, session=None):
    s = session or requests.Session()
    url = BASE + path
    for attempt in range(4):
        try:
            r = s.get(url, params=params, timeout=25, headers={'Accept':'application/json'})
            if r.status_code in (429,500,502,503,504):
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 8)); continue
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, dict) or payload.get('error'):
                raise ValueError(str(payload)[:200])
            return payload
        except (requests.RequestException, ValueError) as exc:
            if attempt == 3 or not isinstance(exc, requests.RequestException):
                raise RuntimeError(f'Coinbase GET failed {path}: {type(exc).__name__}: {str(exc)[:160]}') from exc
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError('unreachable')


def metadata(p):
    if not isinstance(p, dict) or str(p.get('product_type','')).upper() != 'FUTURE': return None
    f = p.get('future_product_details') or {}
    name = str(p.get('display_name') or '')
    if 'perp' in name.lower() or 'perp' in str(f.get('contract_expiry_type','')).lower(): return None
    try:
        exp = utc(f['contract_expiry'])
        size = float(f['contract_size'])
    except (KeyError,ValueError,TypeError): return None
    # Stop fake 2030/2089 perpetual-style expiries.
    if exp > pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=730) or size <= 0: return None
    asset = str(f.get('contract_root_unit') or '').upper()
    if asset not in ASSETS: return None
    pid = str(p.get('product_id') or '')
    if not pid: return None
    return {'product_id':pid, 'asset':asset, 'spot_id':ASSETS[asset], 'expiry_utc':exp.isoformat(),
            'contract_size':size, 'display_name':name, 'status':str(p.get('status') or '')}


def discover(limit_pages=20, session=None):
    found, cursor, seen = [], None, set()
    offset = 0
    seen_raw = set()
    for _ in range(limit_pages):
        params = {'limit':100, 'product_type':'FUTURE','get_all_products':'true'}
        if cursor: params['cursor'] = cursor
        else: params['offset'] = offset
        d = api_get('/products', params, session)
        arr = d.get('products')
        if not isinstance(arr, list): raise RuntimeError('No products array in Coinbase response')
        raw_ids = {str(p.get('product_id')) for p in arr if isinstance(p,dict) and p.get('product_id')}
        if raw_ids and raw_ids.issubset(seen_raw): break  # API ignored pagination
        seen_raw.update(raw_ids)
        for p in arr:
            m = metadata(p)
            if m and m['product_id'] not in seen:
                seen.add(m['product_id']); found.append(m)
        pag = d.get('pagination') or {}
        nxt = pag.get('next_cursor') if isinstance(pag, dict) else None
        if nxt and nxt != cursor:
            cursor = nxt
        elif len(arr) == 100:
            offset += len(arr)
            cursor = None
        else:
            break
    return sorted(found,key=lambda a: (a['asset'],a['expiry_utc'],a['product_id']))


def parse_candles(payload, begin, end, name):
    vals = payload.get('candles', [])
    if not isinstance(vals, list): raise ValueError(f'{name}: candle field absent or not list')
    out=[]
    for c in vals:
        if not isinstance(c,dict): raise ValueError(f'{name}: unexpected candle row type')
        try:
            t = pd.to_datetime(int(c['start']),unit='s',utc=True)
            o,h,l,cl,v=(float(c[k]) for k in ('open','high','low','close','volume'))
        except (ValueError,KeyError,TypeError) as exc:
            raise ValueError(f'{name}: corrupt candle: {str(c)[:180]}') from exc
        if t < begin or t >= end: continue
        if not all(map(math.isfinite,(o,h,l,cl,v))) or min(o,h,l,cl) <=0 or v < 0 or l > min(o,cl) or h < max(o,cl):
            raise ValueError(f'{name}: invalid candle at {t}')
        out.append({'time':t.isoformat(),'open':o,'high':h,'low':l,'close':cl,'volume':v})
    return out


def fetch_candles(product_id, start, end, granularity='ONE_HOUR', session=None):
    step = GRANULARITY_SECONDS[granularity]
    chunk = 250*step # limit below documented 300 candles per request
    s,e = int(utc(start).timestamp()), int(utc(end).timestamp())
    if s >= e: return pd.DataFrame(columns=['time','open','high','low','close','volume'])
    rows=[]
    for a in range(s,e,chunk):
        b=min(e,a+chunk)
        d=api_get('/products/'+quote(product_id,safe='')+'/candles',
                  {'start':str(a),'end':str(b),'granularity':granularity,'limit':300},session)
        rows.extend(parse_candles(d,pd.to_datetime(a,unit='s',utc=True),pd.to_datetime(b,unit='s',utc=True),product_id))
    if not rows: return pd.DataFrame(columns=['time','open','high','low','close','volume'])
    df=pd.DataFrame(rows).sort_values('time')
    if df['time'].duplicated().any():
        # Reject conflicts rather than silently picking one candle.
        for _,g in df.groupby('time'):
            if len(g)>1 and not g.drop(columns='time').eq(g.iloc[0].drop(labels='time')).all().all():
                raise ValueError(f'{product_id}: conflicting duplicate timestamps')
        df=df.drop_duplicates('time')
    return df.reset_index(drop=True)


def save_csv(path, df):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)


def load_manifest(path):
    df=pd.read_csv(path,dtype={'product_id':str,'spot_id':str,'asset':str,'expiry_utc':str})
    required={'product_id','asset','spot_id','expiry_utc','contract_size'}
    if not required.issubset(df.columns): raise ValueError(f'Manifest missing {required-set(df.columns)}')
    if df['product_id'].duplicated().any(): raise ValueError('Duplicate futures contract ID')
    rows=[]
    for r in df.to_dict('records'):
        if r['asset'] not in ASSETS or r['spot_id']!=ASSETS[r['asset']]: raise ValueError('Unsupported or mismatched spot pair')
        if 'perp' in r['product_id'].lower(): raise ValueError('Perpetual not allowed')
        if not (0<float(r['contract_size'])<1e8): raise ValueError('Invalid contract size')
        x=dict(r);x['expiry_utc']=utc(r['expiry_utc']).isoformat();rows.append(x)
    return rows


def run_probe(a):
    start=utc(a.start);end=utc(a.end)
    for id in [a.future_id,a.spot_id]:
        try:
            data=fetch_candles(id,start,end,a.granularity)
            print(json.dumps({'product':id,'bars':len(data),'earliest':data['time'].iloc[0] if len(data) else None,
                              'latest':data['time'].iloc[-1] if len(data) else None,
                              'diagnostic':'historical candles, NOT bid/ask or fills'}))
        except Exception as exc: print(json.dumps({'product':id,'error':str(exc)[:300]}))


def run_backfill(a):
    manifest=load_manifest(a.manifest)
    root=Path(a.output_dir)
    start=utc(a.start);end=min(utc(a.end),pd.Timestamp.now(tz='UTC').floor('h'))
    summary=[]
    for m in manifest:
        exp=utc(m['expiry_utc']);ce=min(end,exp)
        # Futures candles need not exist before listing; missing periods preserved.
        if ce<=start: continue
        pid=m['product_id'];out=root/'bars'/f'{pid}.csv'
        try:
            dat=fetch_candles(pid,start,ce,a.granularity)
            if dat.empty: raise RuntimeError('NO_HISTORICAL_FUTURES_CANDLES (possible public data limitation)')
            save_csv(out,dat)
            print(pid,'futures bars=',len(dat),'from',dat.time.iloc[0],'to',dat.time.iloc[-1])
            summary.append({'product_id':pid,'status':'DATA_SAVED','bars':len(dat),'error':''})
        except (RuntimeError,ValueError) as exc:
            print(pid,'UNAVAILABLE',str(exc)[:240])
            summary.append({'product_id':pid,'status':'UNAVAILABLE','bars':0,'error':str(exc)[:240]})
    for spot in sorted({m['spot_id'] for m in manifest}):
        try:
            out=root/'bars'/f'{spot}.csv'
            dat=fetch_candles(spot,start,end,a.granularity)
            if dat.empty: raise RuntimeError('NO_SPOT_CANDLES')
            save_csv(out,dat)
            print(spot,'spot bars=',len(dat))
        except (RuntimeError,ValueError) as exc:
            print(spot,'UNAVAILABLE',str(exc)[:240])
    save_csv(root/'availability.csv',pd.DataFrame(summary))
    print('Saved availability diagnostics:',root/'availability.csv')
    print('HISTORICAL OHLC ONLY. No historical executable quotes, depths, or final settlements.')


def read_bars(path):
    df=pd.read_csv(path)
    if not {'time','open','close','volume'}.issubset(df.columns): raise ValueError(f'{path}: missing fields')
    df['time']=pd.to_datetime(df['time'],utc=True,format='ISO8601')
    for c in ('open','close','volume'): df[c]=pd.to_numeric(df[c],errors='raise')
    df=df.sort_values('time')
    if df['time'].duplicated().any(): raise ValueError(f'{path}: duplicate hourly timestamps')
    if (~np.isfinite(df[['open','close','volume']])).any().any() or (df[['open','close']]<=0).any().any() or (df['volume']<0).any():
        raise ValueError(f'{path}: invalid prices/volume')
    return df.set_index('time')


def contract_proxy(m, root, a):
    fut_file=root/'bars'/f"{m['product_id']}.csv"
    spot_file=root/'bars'/f"{m['spot_id']}.csv"
    if not fut_file.exists() or not spot_file.exists(): return None,'MISSING_BARS'
    f=read_bars(fut_file);s=read_bars(spot_file)
    df=f[['open','close','volume']].rename(columns={'open':'F','close':'Fclose','volume':'fv'}).join(
        s[['open','close','volume']].rename(columns={'open':'S','close':'Sclose','volume':'sv'}),how='inner')
    exp=utc(m['expiry_utc'])
    wanted=exp-pd.Timedelta(days=a.entry_days)
    stop=exp-pd.Timedelta(hours=a.exit_buffer_hours)
    # Signal is based on previous completed bar; execute on next hourly open.
    eligible=df.loc[(df.index>=wanted)&(df.index<=wanted+pd.Timedelta(hours=3))&(df.index<stop)]
    if len(eligible)==0: return None,'NO_ENTRY_CANDLE_NEAR_TARGET'
    target=eligible.index[0]
    if df.loc[target,'fv']<=0 or df.loc[target,'sv']<=0: return None,'ZERO_VOLUME_DECISION_CANDLE'
    basis_signal_bps=(df.loc[target,'Fclose']/df.loc[target,'Sclose']-1)*10000
    entry=target+pd.Timedelta(hours=1)
    if entry not in df.index or df.loc[entry,'fv']<=0 or df.loc[entry,'sv']<=0:
        return None,'MISSING_NEXT_OPEN_OR_ZERO_VOLUME'
    # Exit on first market-open candle at/after chosen time, before expiration.
    exits=df.index[(df.index>=stop)&(df.index<=stop+pd.Timedelta(hours=3))&(df.index>entry)]
    if len(exits)==0: return None,'NO_PRE_EXPIRY_EXIT_CANDLE'
    et=exits[0]
    if et not in df.index or df.loc[et,'fv']<=0 or df.loc[et,'sv']<=0: return None,'MISSING_EXIT_OR_ZERO_VOLUME'
    sr,fr=df.loc[entry],df.loc[et]
    # No mid-candle hindsight: both entry and early exit priced at BAR OPEN with modeled adverse slippage.
    q=float(m['contract_size']);spot_in=sr.S*(1+a.slippage_bps/10000)
    fut_in=sr.F*(1-a.slippage_bps/10000)
    spot_out=fr.S*(1-a.slippage_bps/10000)
    fut_out=fr.F*(1+a.slippage_bps/10000)
    gross=q*((fr.S-sr.S)+(sr.F-fr.F))
    slippage=q*((spot_in-sr.S)+(sr.F-fut_in)+(fr.S-spot_out)+(fut_out-fr.F))
    spot_fees=q*a.spot_fee_bps/10000*(spot_in+spot_out)
    fut_fees=2*a.futures_fee_per_side
    holding_days=(et-entry).total_seconds()/86400
    financing=q*spot_in*a.financing_apr_pct/100*holding_days/365
    net=gross-slippage-spot_fees-fut_fees-financing
    capital=q*spot_in+a.futures_margin_usd
    return {'asset':m['asset'],'product_id':m['product_id'],'expiry_utc':exp.isoformat(),
            'decision_time':target.isoformat(),'entry_utc':entry.isoformat(),'exit_utc':et.isoformat(),
            'entry_basis_bps':basis_signal_bps,'executed_open_basis_bps':(sr.F/sr.S-1)*10000,'gross_usd':gross,'slippage_usd':slippage,
            'spot_fees_usd':spot_fees,'futures_fees_usd':fut_fees,'financing_usd':financing,
            'net_usd':net,'net_capital_bps':10000*net/capital,'capital_usd':capital,
            'days_held':holding_days,'status':'PROXY_ONLY_NOT_EXECUTABLE'},''


def forward_select(raw, min_train, min_hits):
    # Selecting on previous contracts EXPIRED before the current entry; never on same or future expiry.
    scored=[]
    thresholds=[0,10,25,50]
    for _,r in raw.sort_values('entry_utc').iterrows():
        t=utc(r['entry_utc'])
        older=raw[(raw.asset==r.asset)&(pd.to_datetime(raw.expiry_utc,utc=True)<t)]
        if older.product_id.nunique()<min_train:
            scored.append({'asset':r.asset,'product_id':r.product_id,'status':'INSUFFICIENT_PRIOR_EXPIRED_CONTRACTS',
                           'train_contracts':older.product_id.nunique(),'selected_min_basis_bps':None,'oos_net_usd':None})
            continue
        candidates=[]
        for threshold in thresholds:
            train=older[older.entry_basis_bps>=threshold]
            if train.product_id.nunique()<min_hits: continue
            vals=train.net_capital_bps.to_numpy(dtype=float)
            lower=float(vals.mean()-1.96*vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else -math.inf
            candidates.append((lower,threshold,len(vals)))
        if not candidates or max(candidates)[0]<=0:
            scored.append({'asset':r.asset,'product_id':r.product_id,'status':'NO_POSITIVE_TRAIN_LOWER_BOUND',
                           'train_contracts':len(older),'selected_min_basis_bps':None,'oos_net_usd':None})
            continue
        _,threshold,n=max(candidates)
        selected=r.entry_basis_bps>=threshold
        scored.append({'asset':r.asset,'product_id':r.product_id,
                       'status':'PAPER_PROXY_TRADE' if selected else 'BELOW_FROZEN_THRESHOLD',
                       'train_contracts':len(older),'selected_min_basis_bps':threshold,
                       'oos_net_usd':r.net_usd if selected else None})
    return pd.DataFrame(scored)


def run_evaluate(a):
    manifest=load_manifest(a.manifest);root=Path(a.output_dir)
    rows=[];skipped=[]
    for m in manifest:
        result,reason=contract_proxy(m,root,a)
        if result: rows.append(result)
        else: skipped.append({'product_id':m['product_id'],'reason':reason})
    if not rows:
        print('NO CONTRACT-LEVEL ROUND TRIPS. Need futures and spot history spanning entry and pre-expiry exit. No result invented.')
        if skipped: print(pd.DataFrame(skipped).to_string(index=False))
        return
    raw=pd.DataFrame(rows).sort_values('entry_utc'); wfo=forward_select(raw,a.min_train_contracts,a.min_train_hits)
    save_csv(root/'historical_contract_proxies.csv',raw)
    save_csv(root/'walkforward.csv',wfo)
    save_csv(root/'skipped_contracts.csv',pd.DataFrame(skipped,columns=['product_id','reason']))
    print('HISTORICAL CANDLE-OPEN PROXIES (NOT EXECUTABLE, NOT SETTLEMENT P&L)')
    print(raw[['asset','product_id','entry_utc','exit_utc','entry_basis_bps','net_usd','net_capital_bps']].to_string(index=False))
    print('\nWALK FORWARD: only contracts expired before each entry can train')
    print(wfo.to_string(index=False))
    print(f'Independent contract proxies={len(raw)}; skipped={len(skipped)}; paper decisions={(wfo.status=="PAPER_PROXY_TRADE").sum()}')
    print('NO historical bid/ask/depth, index settlement, guaranteed fill, or margin stress. Not live-tradable P&L.')


def self_test():
    # Fake-perp exclusion, parse, no forward look, fees, chronological OOS test.
    p={'product_id':'BIT-27NOV26-CDE','product_type':'FUTURE','display_name':'BTC 27-Nov',
       'future_product_details':{'contract_root_unit':'BTC','contract_expiry':'2026-11-27T16:00:00Z','contract_size':'0.01','contract_expiry_type':'EXPIRING'}}
    assert metadata(p)['contract_size']==0.01
    pp=json.loads(json.dumps(p));pp['display_name']='BTC Perp';assert metadata(pp) is None
    cand={'candles':[{'start':'1793491200','open':'100','high':'101','low':'99','close':'100','volume':'10'}]}
    assert len(parse_candles(cand,pd.Timestamp('2026-11-01',tz='UTC'),pd.Timestamp('2026-11-02',tz='UTC'),'x'))==1
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        root=Path(tmp);exp=pd.Timestamp('2026-11-27T16:00:00Z');ts=pd.date_range(exp-pd.Timedelta(days=16),exp-pd.Timedelta(hours=20),freq='h',tz='UTC')
        base=pd.DataFrame({'time':ts.astype(str),'open':100.0,'close':100.0,'volume':10.0})
        fut=base.copy();fut['open']=101.0;fut['close']=101.0
        # Only prior completed close triggers; future entry open must not affect signal.
        fut.loc[fut.time==str(exp-pd.Timedelta(days=14)),'close']=102.0
        save_csv(root/'bars'/'BTC-USD.csv',base);save_csv(root/'bars'/'BIT-27NOV26-CDE.csv',fut)
        m=metadata(p); a=argparse.Namespace(entry_days=14,exit_buffer_hours=24,slippage_bps=0,
                                            spot_fee_bps=0,futures_fee_per_side=0,financing_apr_pct=0,
                                            futures_margin_usd=20)
        v,why=contract_proxy(m,root,a)
        assert not why and abs(v['gross_usd'])<1e-9 and abs(v['net_usd'])<1e-9
        assert abs(v['entry_basis_bps']-200)<1e-7 and abs(v['executed_open_basis_bps']-100)<1e-7
        a.spot_fee_bps=100
        v,why=contract_proxy(m,root,a)
        assert v['net_usd']<0 and not why
        records=[]
        for j in range(7):
            x=dict(v);x['product_id']=f'C{j}';x['expiry_utc']=(pd.Timestamp('2026-02-01',tz='UTC')+pd.DateOffset(months=j)).isoformat()
            x['entry_utc']=(pd.Timestamp('2026-01-15',tz='UTC')+pd.DateOffset(months=j)).isoformat()
            x['entry_basis_bps']=30;records.append(x)
        out=forward_select(pd.DataFrame(records),4,4)
        assert (out.iloc[:4].status=='INSUFFICIENT_PRIOR_EXPIRED_CONTRACTS').all()
        assert out.iloc[-1].status=='NO_POSITIVE_TRAIN_LOWER_BOUND'
    print('PASS V4.4: public-API parsing, perpetual exclusion, next-open proxy, fees, contract-level chronological isolation')
    print('OFFLINE self-test; NO API requests; NO orders')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    d=sub.add_parser('discover');d.add_argument('--out',default=str(DATA_DIR/'contracts.csv'));d.add_argument('--limit-pages',type=int,default=20)
    p=sub.add_parser('probe');p.add_argument('--future-id',default='BIT-27NOV26-CDE');p.add_argument('--spot-id',default='BTC-USD')
    p.add_argument('--start',default='2026-11-01T00:00:00Z');p.add_argument('--end',default='2026-11-08T00:00:00Z');p.add_argument('--granularity',choices=GRANULARITY_SECONDS,default='ONE_HOUR')
    b=sub.add_parser('backfill');b.add_argument('--manifest',default=str(DATA_DIR/'contracts.csv'))
    b.add_argument('--start',default='2026-05-01T00:00:00Z');b.add_argument('--end',default='2026-11-13T00:00:00Z');b.add_argument('--granularity',choices=GRANULARITY_SECONDS,default='ONE_HOUR');b.add_argument('--output-dir',default=str(DATA_DIR))
    e=sub.add_parser('evaluate');e.add_argument('--manifest',default=str(DATA_DIR/'contracts.csv'));e.add_argument('--output-dir',default=str(DATA_DIR));
    e.add_argument('--entry-days',type=float,default=14);e.add_argument('--exit-buffer-hours',type=float,default=24)
    e.add_argument('--spot-fee-bps',type=float,default=90);e.add_argument('--slippage-bps',type=float,default=5)
    e.add_argument('--futures-fee-per-side',type=float,default=0.10);e.add_argument('--financing-apr-pct',type=float,default=0)
    e.add_argument('--futures-margin-usd',type=float,default=250);e.add_argument('--min-train-contracts',type=int,default=4);e.add_argument('--min-train-hits',type=int,default=4)
    sub.add_parser('self-test')
    a=ap.parse_args()
    if a.cmd=='self-test':self_test()
    elif a.cmd=='discover':
        rows=discover(a.limit_pages)
        if rows:save_csv(Path(a.out),pd.DataFrame(rows))
        print('Discovered dated BTC/ETH/SOL futures:',len(rows),'(expired contracts may not be listed)')
        for m in rows:print(m['product_id'],m['expiry_utc'],m['status'])
        if rows:print('Manifest:',a.out)
    elif a.cmd=='probe':run_probe(a)
    elif a.cmd=='backfill':run_backfill(a)
    elif a.cmd=='evaluate':
        if min(a.entry_days,a.exit_buffer_hours,a.min_train_contracts,a.min_train_hits)<=0 or min(a.spot_fee_bps,a.slippage_bps,a.futures_fee_per_side,a.financing_apr_pct,a.futures_margin_usd)<0:
            ap.error('Invalid negative cost or non-positive timing/training parameter')
        run_evaluate(a)

if __name__=='__main__':main()
