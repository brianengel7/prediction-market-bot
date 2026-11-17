#!/usr/bin/env python3
"""V4.5: Read-only Binance USD-M dated-futures historical benchmark.

Uses official checksum-verified Binance Vision *expired quarterly* futures and spot
hourly archives. Research proxy only: Binance global is not Coinbase Derivatives,
not a U.S. trading venue, and candles are not executable bid/ask quotes.

No trading/order endpoints, no credentials, no synthetic candles or forward fill.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

DATA_ROOT = Path('src/crypto-arb/data/v45_binance_dated_BENCHMARK_ONLY')
BASE = 'https://data.binance.vision/data'
# Candidate calendar dates; acceptance REQUIRES real, checksum-verified archives.
# The dates alone do not prove that a contract existed or traded on each date.
CANDIDATE_EXPIRIES = (
    '2024-09-27','2024-12-27','2025-03-28','2025-06-27',
    '2025-09-26','2025-12-26','2026-03-27','2026-06-26','2026-09-25',
)
STUDY_ASSETS = ('BTC','ETH')
COLUMNS = ('open_time','open','high','low','close','volume','close_time','qvolume',
           'num_trades','taker_base','taker_quote','ignore')


def utc(t):
    out=pd.Timestamp(t)
    if out.tzinfo is None: raise ValueError('Unzoned time')
    return out.tz_convert('UTC')


def archive_url(kind, symbol, month):
    assert kind in ('spot','futures/um')
    if not re.fullmatch(r'[A-Z0-9_]+', symbol) or not re.fullmatch(r'20\d{2}-(0[1-9]|1[0-2])',month):
        raise ValueError('Invalid archive identifiers')
    return f'{BASE}/{kind}/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip'


def zip_bytes(url, cache, session=None):
    """Return verified raw zip or None for definitive 404. Fail closed on all else."""
    cache.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(url.encode()).hexdigest()[:20]
    zip_file=cache/(key+'.zip')
    sha_file=cache/(key+'.sha256')
    sess=session or requests.Session()
    if zip_file.exists() and sha_file.exists():
        raw=zip_file.read_bytes(); expected=sha_file.read_text().strip()
        if hashlib.sha256(raw).hexdigest()==expected: return raw
        raise ValueError(f'Cached ZIP hash mismatch: {url}')
    resp=sess.get(url+'.CHECKSUM',timeout=25)
    if resp.status_code==404: return None
    resp.raise_for_status()
    check=resp.text
    m=re.search(r'\b([0-9a-fA-F]{64})\b',check)
    if not m: raise ValueError(f'Missing SHA256 checksum: {url}')
    z=sess.get(url,timeout=40)
    if z.status_code==404: raise ValueError(f'Checksum exists but ZIP is missing: {url}')
    z.raise_for_status()
    raw=z.content
    if len(raw)>25_000_000: raise ValueError('Unexpectedly large hourly ZIP')
    h=hashlib.sha256(raw).hexdigest()
    if h.lower()!=m.group(1).lower(): raise ValueError(f'Checksum mismatch: {url}')
    if not zipfile.is_zipfile(io.BytesIO(raw)): raise ValueError(f'Invalid ZIP: {url}')
    zip_file.write_bytes(raw);sha_file.write_text(h)
    return raw


def parse_zip(raw, name):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv') and '/' not in n]
        if len(names)!=1: raise ValueError(f'{name}: expected one archive CSV')
        if z.getinfo(names[0]).file_size>15_000_000: raise ValueError(f'{name}: oversized CSV')
        b=z.read(names[0])
    df=pd.read_csv(io.BytesIO(b),header=None,names=list(COLUMNS),dtype=str,on_bad_lines='error')
    # Some futures archives contain headers; spot archives generally do not.
    df=df.loc[~df['open_time'].astype(str).str.lower().str.strip().isin(['open_time','open time'])].copy()
    if len(df)==0: raise ValueError(f'{name}: empty archive CSV')
    raw_ns=pd.to_numeric(df.open_time,errors='raise').astype('int64')
    digits=int(len(str(abs(int(raw_ns.iloc[0])))))
    units={10:'s',13:'ms',16:'us',19:'ns'}
    if digits not in units: raise ValueError(f'{name}: unknown timestamp precision: {digits}')
    # All timestamp values must have same precision within file.
    if not all(len(str(abs(int(x))))==digits for x in raw_ns): raise ValueError('Mixed time units')
    df['time']=pd.to_datetime(raw_ns,unit=units[digits],utc=True)
    for c in ('open','high','low','close','volume'):
        df[c]=pd.to_numeric(df[c],errors='raise')
    if ((df['open']<=0)|(df['high']<=0)|(df['low']<=0)|(df['close']<=0)|
        (df['volume']<0)|(~np.isfinite(df[['open','high','low','close','volume']])).any(axis=1)|
        (df['high']<df[['open','close','low']].max(axis=1))|
        (df['low']>df[['open','close','high']].min(axis=1))).any():
        raise ValueError(f'{name}: invalid OHLCV')
    if ((df['time'].dt.minute!=0)|(df['time'].dt.second!=0)).any():
        raise ValueError(f'{name}: not hourly aligned')
    return df[['time','open','high','low','close','volume']].copy()


def merged_frames(frames, name):
    if not frames: return pd.DataFrame(columns=['time','open','high','low','close','volume'])
    out=pd.concat(frames,ignore_index=True).sort_values('time')
    if out.time.duplicated().any():
        dups=out[out.time.duplicated(keep=False)]
        for t,g in dups.groupby('time'):
            if not g.drop(columns=['time']).eq(g.drop(columns=['time']).iloc[0]).all().all():
                raise ValueError(f'{name}: conflicting duplicate candle {t}')
        out=out.drop_duplicates('time')
    return out.reset_index(drop=True)


def months_between(start,end):
    s=utc(start);e=utc(end)
    m=pd.Period(year=s.year,month=s.month,freq='M')
    last=pd.Period(year=e.year,month=e.month,freq='M')
    out=[]
    while m<=last:
        out.append(str(m));m+=1
    return out


def symbol_for(asset, expiry_date):
    d=pd.Timestamp(expiry_date)
    if asset not in STUDY_ASSETS:raise ValueError('unsupported asset')
    return asset+'USDT_'+d.strftime('%y%m%d')


def save_csv(path, frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)


def download(a):
    root=Path(a.output_dir)
    expires=[x.strip() for x in a.expiries.split(',') if x.strip()]
    if not expires or len(set(expires))!=len(expires): raise ValueError('No expiries or duplicated dates')
    assets=[x.upper() for x in a.assets]
    if not set(assets).issubset(STUDY_ASSETS): raise ValueError('Only BTC/ETH USD-M dated are supported')
    s=requests.Session()
    status=[]; spot_data={x:[] for x in assets}; found=[]
    for asset in assets:
        for iso in expires:
            d=pd.Timestamp(iso)
            if d>=pd.Timestamp.now(tz='UTC').tz_localize(None).normalize():
                status.append({'asset':asset,'expiry_date':iso,'symbol':symbol_for(asset,iso),'status':'UNEXPIRED_SKIPPED','notes':'Incomplete historical expiry'});continue
            # Binance quarterly deliveries typically occur 08:00 UTC; early exit
            # is 24h *before* delivery, avoiding a settlement-index assumption.
            expiry=pd.Timestamp(iso+'T08:00:00Z')
            first=expiry-pd.Timedelta(days=a.entry_days+4)
            last=expiry-pd.Timedelta(hours=a.exit_buffer_hours)
            sym=symbol_for(asset,iso)
            futures=[];spots=[]; missing_fut=[];missing_spot=[]
            for month in months_between(first,last):
                for kind,target,missing,product in (
                    ('futures/um',futures,missing_fut,sym),
                    ('spot',spots,missing_spot,asset+'USDT'),
                ):
                    url=archive_url(kind,product,month)
                    try:
                        raw=zip_bytes(url,root/'cache',s)
                        if raw is None:missing.append(month);continue
                        target.append(parse_zip(raw,product+' '+month))
                    except Exception as exc:
                        # Network faults / corrupt archives are NOT "missing" data.
                        raise RuntimeError(f'{sym} {kind} {month} download failed: {exc}') from exc
            f=merged_frames(futures,sym);sp=merged_frames(spots,asset+'USDT')
            if f.empty or sp.empty:
                status.append({'asset':asset,'expiry_date':iso,'symbol':sym,'status':'UNAVAILABLE',
                               'notes':f'fut missing={missing_fut}; spot missing={missing_spot}'});continue
            # The contract must have real trading bars at required times.
            wanted=expiry-pd.Timedelta(days=a.entry_days)
            if not (f.time.between(wanted,wanted+pd.Timedelta(hours=4)).any() and
                    f.time.between(last,last+pd.Timedelta(hours=3)).any()):
                status.append({'asset':asset,'expiry_date':iso,'symbol':sym,'status':'INCOMPLETE_ENTRY_EXIT',
                               'notes':f'fut missing={missing_fut}'});continue
            save_csv(root/'bars'/f'{sym}.csv',f)
            spot_data[asset].append(sp)
            found.append({'asset':asset,'symbol':sym,'expiry_utc':expiry.isoformat(),
                          'spot_symbol':asset+'USDT','venue':'binance_global_usdm',
                          'source':'CHECKSUM_VERIFIED_ARCHIVES_NOT_US_EXECUTABLE'})
            status.append({'asset':asset,'expiry_date':iso,'symbol':sym,'status':'ARCHIVE_VERIFIED',
                           'notes':f'{len(f)} future candles; {len(sp)} spot candles'})
            print(f'{sym}: verified {len(f)} future bars and {len(sp)} spot bars')
        if spot_data[asset]:
            save_csv(root/'bars'/f'{asset}USDT.csv',merged_frames(spot_data[asset],asset+'USDT'))
    save_csv(root/'availability.csv',pd.DataFrame(status))
    if found:save_csv(root/'contracts.csv',pd.DataFrame(found))
    print(f'Historical contract archives: {len(found)} verified. Saved {root}/availability.csv')
    print('These are Binance GLOBAL contract histories, NOT Coinbase market data or US executable fills.')


def read_bars(path):
    df=pd.read_csv(path)
    if not {'time','open','close','volume'}.issubset(df):raise ValueError(f'Bad bar schema {path}')
    df.time=pd.to_datetime(df.time,utc=True,format='ISO8601')
    if df.time.duplicated().any():raise ValueError(f'Duplicate time {path}')
    if df[['open','close','volume']].isna().any().any():raise ValueError(f'NaN bars {path}')
    return df.set_index('time').sort_index()


def simulate(m, root, a):
    fut=root/'bars'/f"{m['symbol']}.csv"; spot=root/'bars'/f"{m['spot_symbol']}.csv"
    if not fut.exists() or not spot.exists():return None,'MISSING_PRICE_FILES'
    f=read_bars(fut);s=read_bars(spot)
    both=f[['open','close','volume']].rename(columns={'open':'F','close':'Fc','volume':'Fv'}).join(
        s[['open','close','volume']].rename(columns={'open':'S','close':'Sc','volume':'Sv'}),how='inner')
    exp=utc(m['expiry_utc'])
    d=exp-pd.Timedelta(days=a.entry_days)
    x=exp-pd.Timedelta(hours=a.exit_buffer_hours)
    eligible=both[(both.index>=d)&(both.index<=d+pd.Timedelta(hours=3))&(both.index<x)]
    if eligible.empty:return None,'NO_DECISION_CANDLE'
    decision=eligible.index[0]
    if min(float(both.loc[decision,'Fv']),float(both.loc[decision,'Sv']))<=0:return None,'ZERO_SIGNAL_VOLUME'
    entry=decision+pd.Timedelta(hours=1)
    if entry not in both.index:return None,'MISSING_NEXT_OPEN'
    exits=both.index[(both.index>=x)&(both.index<=x+pd.Timedelta(hours=3))]
    if len(exits)==0:return None,'NO_PRE_EXPIRY_EXIT'
    exit=exits[0]
    if exit<=entry:return None,'EXIT_BEFORE_ENTRY'
    # Require almost-contiguous candles between entry and exit; no imputed bars.
    need=int((exit-entry)/pd.Timedelta(hours=1))+1
    subset=both.loc[entry:exit]
    if len(subset)<need-1:return None,'MISSING_CANDLES_IN_HOLD'
    if (subset[['Fv','Sv']].iloc[[0,-1]]<=0).any().any():return None,'ZERO_EXECUTION_HOUR_VOLUME'
    b0=both.loc[entry];b1=both.loc[exit]
    gross=(b1.S-b0.S)+(b0.F-b1.F)
    fees=(b0.S+b1.S)*a.spot_fee_bps/10000+(b0.F+b1.F)*a.future_fee_bps/10000
    slippage=(b0.S+b1.S+b0.F+b1.F)*a.slippage_bps/10000
    days=float((exit-entry)/pd.Timedelta(days=1))
    financing=b0.S*a.financing_apr_pct/100*days/365
    net=gross-fees-slippage-financing
    cap=b0.S*(1+a.margin_pct/100)
    return {'asset':m['asset'],'symbol':m['symbol'],'expiry_utc':exp.isoformat(),
            'decision_utc':decision.isoformat(),'entry_utc':entry.isoformat(),'exit_utc':exit.isoformat(),
            'decision_basis_bps':(both.loc[decision,'Fc']/both.loc[decision,'Sc']-1)*10000,
            'entry_basis_bps':(b0.F/b0.S-1)*10000,
            'gross_bps':10000*gross/b0.S,'net_capital_bps':10000*net/cap,
            'gross_usd_per_coin':gross,'fees_usd_per_coin':fees,'slippage_usd_per_coin':slippage,
            'financing_usd_per_coin':financing,'net_usd_per_coin':net,'capital_usd_per_coin':cap,
            'days_held':days,'basis_at_exit_bps':(b1.F/b1.S-1)*10000,
            'type':'EARLY_EXIT_CANDLE_OPEN_BENCHMARK_NOT_EXECUTABLE'},''


def walk_forward(data,min_train,min_hits):
    # ONE candidate per expiry per asset. A contract trains only after expiry.
    rows=[]
    for _,r in data.sort_values('entry_utc').iterrows():
        prior=data[(data.asset==r.asset)&(pd.to_datetime(data.expiry_utc,utc=True)<utc(r.entry_utc))]
        n=prior.symbol.nunique()
        out={'asset':r.asset,'symbol':r.symbol,'prior_expired_contracts':n,
             'status':'INSUFFICIENT_PRIOR_EXPIRED','threshold_bps':None,'test_net_capital_bps':None}
        if n<min_train:rows.append(out);continue
        candidate=[]
        for threshold in [0,25,50,100]:
            tr=prior[prior.decision_basis_bps>=threshold]
            if tr.symbol.nunique()<min_hits:continue
            x=tr.net_capital_bps.to_numpy(float)
            if len(x)<2:continue
            lower=float(x.mean()-1.96*x.std(ddof=1)/math.sqrt(len(x)))
            candidate.append((lower,threshold))
        if not candidate or max(candidate)[0]<=0:
            out['status']='NO_POSITIVE_TRAIN_LOWER_BOUND'
        else:
            _,th=max(candidate)
            out['threshold_bps']=th
            if r.decision_basis_bps>=th:
                out['status']='PAPER_SELECTED_PROXY_ONLY';out['test_net_capital_bps']=r.net_capital_bps
            else:out['status']='BELOW_TRAIN_THRESHOLD'
        rows.append(out)
    return pd.DataFrame(rows)


def evaluate(a):
    root=Path(a.output_dir);path=root/'contracts.csv'
    if not path.exists():raise RuntimeError('Missing verified contracts.csv: run download first')
    manifests=pd.read_csv(path).to_dict('records')
    trades=[];skipped=[]
    for m in manifests:
        row,why=simulate(m,root,a)
        if row:trades.append(row)
        else:skipped.append({'symbol':m['symbol'],'reason':why})
    save_csv(root/'eval_skips.csv',pd.DataFrame(skipped,columns=['symbol','reason']))
    if not trades:
        print('NO COMPLETE EARLY-EXIT CONTRACTS. See',root/'eval_skips.csv')
        return
    df=pd.DataFrame(trades).sort_values('entry_utc')
    w=walk_forward(df,a.min_train_contracts,a.min_train_hits)
    save_csv(root/'contract_proxies.csv',df);save_csv(root/'walkforward.csv',w)
    print('CHECKSUM-VERIFIED BINANCE GLOBAL DATED FUTURES: EARLY-EXIT CANDLE PROXIES (NOT TRADABLE)')
    print(df[['asset','symbol','entry_utc','entry_basis_bps','gross_bps','net_capital_bps']].to_string(index=False))
    print('\nWALK FORWARD -- ONLY ALREADY-EXPIRED FUTURES TRAIN EACH DECISION')
    print(w.to_string(index=False))
    selected=w[w.status=='PAPER_SELECTED_PROXY_ONLY']
    print(f'Contracts={len(df)}, missing={len(skipped)}, WFO candidates={len(selected)}, mean WFO net capital bps=',
          round(selected.test_net_capital_bps.mean(),3) if len(selected) else 'N/A')
    print('NOT Coinbase data; NOT actual bid/ask fills, settlement P&L, margin liquidation analysis, or executable US arbitrage.')


def self_test():
    import tempfile
    from argparse import Namespace
    assert 'BTCUSDT_260626'==symbol_for('BTC','2026-06-26')
    assert archive_url('futures/um','BTCUSDT_260626','2026-06').endswith('/BTCUSDT_260626-1h-2026-06.zip')
    assert months_between('2026-05-29T00:00:00Z','2026-06-05T00:00:00Z')==['2026-05','2026-06']
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        times=pd.date_range('2026-06-07T00:00:00Z','2026-06-26T07:00:00Z',freq='h')
        src=pd.DataFrame({'time':times,'open':100.,'high':101.,'low':99.,'close':100.,'volume':10.})
        fut=src.copy();fut['open']=101.;fut['close']=101.
        save_csv(root/'bars'/'BTCUSDT.csv',src)
        save_csv(root/'bars'/'BTCUSDT_260626.csv',fut)
        m={'asset':'BTC','symbol':'BTCUSDT_260626','spot_symbol':'BTCUSDT','expiry_utc':'2026-06-26T08:00:00Z'}
        a=Namespace(entry_days=14,exit_buffer_hours=24,spot_fee_bps=0,future_fee_bps=0,slippage_bps=0,
                    financing_apr_pct=0,margin_pct=20)
        v,why=simulate(m,root,a)
        assert v and not why and abs(v['gross_usd_per_coin'])<1e-7
        a.spot_fee_bps=90
        v,why=simulate(m,root,a)
        assert v['net_usd_per_coin']<0
        # Historical bars parser supports future headers, spot microsecond units.
        line='open_time,open,high,low,close,volume,close_time,qvolume,num_trades,taker_base,taker_quote,ignore\n'
        t=int(pd.Timestamp('2026-06-20T00:00:00Z').timestamp())*1_000_000
        line+=f'{t},100,101,99,100,10,0,0,0,0,0,0\n'
        blob=io.BytesIO()
        with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('file.csv',line)
        parsed=parse_zip(blob.getvalue(),'fixture')
        assert str(parsed.time.iloc[0])=='2026-06-20 00:00:00+00:00'
        records=[]
        for j in range(7):
            r=dict(v);exp=utc('2025-01-26T08:00:00Z')+pd.DateOffset(months=3*j)
            r.update({'asset':'BTC','symbol':f'BTC{j}','expiry_utc':exp.isoformat(),
                      'entry_utc':(exp-pd.Timedelta(days=14)).isoformat(),'decision_basis_bps':40.,'net_capital_bps':-5.})
            records.append(r)
        w=walk_forward(pd.DataFrame(records),4,4)
        assert len(w)==7 and w.iloc[-1].status=='NO_POSITIVE_TRAIN_LOWER_BOUND'
    print('PASS V4.5: archive paths, timestamps, OHLC parsing, no lookahead, 4-leg costs, expired-only WFO')
    print('OFFLINE READ ONLY; NO ORDERS. External historical download still requires Codespaces network.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    d=sub.add_parser('download');d.add_argument('--assets',nargs='+',default=['BTC']);d.add_argument('--expiries',default=','.join(CANDIDATE_EXPIRIES))
    d.add_argument('--output-dir',default=str(DATA_ROOT));d.add_argument('--entry-days',type=float,default=14)
    d.add_argument('--exit-buffer-hours',type=float,default=24)
    e=sub.add_parser('evaluate');e.add_argument('--output-dir',default=str(DATA_ROOT));e.add_argument('--entry-days',type=float,default=14)
    e.add_argument('--exit-buffer-hours',type=float,default=24)
    e.add_argument('--spot-fee-bps',type=float,default=90);e.add_argument('--future-fee-bps',type=float,default=5)
    e.add_argument('--slippage-bps',type=float,default=5);e.add_argument('--financing-apr-pct',type=float,default=0)
    e.add_argument('--margin-pct',type=float,default=20)
    e.add_argument('--min-train-contracts',type=int,default=4);e.add_argument('--min-train-hits',type=int,default=4)
    a=ap.parse_args()
    if a.command=='self-test':self_test()
    elif a.command=='download':
        if a.entry_days<=0 or a.exit_buffer_hours<=0 or a.entry_days*24 <= a.exit_buffer_hours+4:ap.error('entry must precede early exit')
        download(a)
    else:
        if min(a.entry_days,a.exit_buffer_hours,a.min_train_contracts,a.min_train_hits)<=0 or min(
            a.spot_fee_bps,a.future_fee_bps,a.slippage_bps,a.financing_apr_pct,a.margin_pct)<0:ap.error('negative cost or invalid schedule')
        evaluate(a)

if __name__=='__main__':main()
