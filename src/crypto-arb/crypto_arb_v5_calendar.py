#!/usr/bin/env python3
"""V5: Binance Global USD-M historical quarterly futures CALENDAR-SPREAD research.

RESEARCH ONLY. Real checksum-verified hourly archives, not U.S.-venue executable quotes.
Never places orders; never imports credentials. Strict pair overlap and no price filling.
Long NEAR expiry future / short FAR expiry future (equal underlying coin units).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import t as student_t

BASE = 'https://data.binance.vision/data'
ROOT = Path('src/crypto-arb/data/v5_binance_calendar_BENCHMARK_ONLY')
DEFAULT_CACHE = Path('src/crypto-arb/data/v45_binance_dated_BENCHMARK_ONLY/cache')
EXPIRIES = (
    '2024-09-27', '2024-12-27', '2025-03-28', '2025-06-27',
    '2025-09-26', '2025-12-26', '2026-03-27', '2026-06-26',
    '2026-09-25', '2026-12-25',
)
COLS = ('open_time','open','high','low','close','volume','close_time','qvolume',
        'num_trades','taker_base','taker_quote','ignore')
ASSETS = ('BTC','ETH')
LOOKBACKS = (7,14)  # days
THRESHOLDS = (1.5,2.0)
HOLDS = (24,72,168,336)  # hours


def ts(s):
    v = pd.Timestamp(s)
    if v.tzinfo is None: raise ValueError(f'Naive timestamp rejected: {s}')
    return v.tz_convert('UTC')


def mon_range(start,end):
    m = pd.Period(year=ts(start).year,month=ts(start).month,freq='M')
    e = pd.Period(year=ts(end).year,month=ts(end).month,freq='M')
    out=[]
    while m<=e:
        out.append(str(m));m+=1
    return out


def symbol(asset,expiry):
    if asset not in ASSETS:raise ValueError(f'Unsupported asset: {asset}')
    d=pd.Timestamp(expiry)
    if not re.fullmatch(r'20\d\d-\d\d-\d\d',str(expiry)):
        raise ValueError('Invalid expiry')
    return f'{asset}USDT_{d:%y%m%d}'


def url_for(sym,month):
    if not re.fullmatch(r'[A-Z0-9_]+',sym) or not re.fullmatch(r'20\d\d-(?:0[1-9]|1[0-2])',month):
        raise ValueError('Invalid archive name')
    return f'{BASE}/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{month}.zip'


def get_verified(url,cache,session):
    """404 means absent archive. All other unexpected results are fatal."""
    cache.mkdir(parents=True,exist_ok=True)
    tag=hashlib.sha256(url.encode()).hexdigest()[:20]
    target=cache/f'{tag}.zip';sha=cache/f'{tag}.sha256'
    if target.exists() and sha.exists():
        blob=target.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=sha.read_text().strip():
            raise ValueError(f'Invalid cached hash {url}')
        return blob
    r=session.get(url+'.CHECKSUM',timeout=30)
    if r.status_code==404:return None
    r.raise_for_status()
    m=re.search(r'\b[0-9a-fA-F]{64}\b',r.text)
    if not m:raise ValueError(f'Missing upstream checksum for {url}')
    r=session.get(url,timeout=60)
    if r.status_code==404:raise ValueError('ZIP missing despite published checksum')
    r.raise_for_status()
    blob=r.content
    if len(blob)>30_000_000 or not zipfile.is_zipfile(io.BytesIO(blob)):
        raise ValueError(f'Invalid or oversized archive {url}')
    digest=hashlib.sha256(blob).hexdigest()
    if digest.lower()!=m.group(0).lower():raise ValueError(f'Checksum mismatch {url}')
    target.write_bytes(blob);sha.write_text(digest)
    return blob


def parse_zip(blob,name):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        files=[f for f in z.namelist() if f.endswith('.csv') and '/' not in f]
        if len(files)!=1 or z.getinfo(files[0]).file_size>20_000_000:
            raise ValueError(f'{name}: unsafe or unexpected ZIP layout')
        content=z.read(files[0])
    df=pd.read_csv(io.BytesIO(content),header=None,names=COLS,dtype=str,on_bad_lines='error')
    df=df[~df.open_time.str.strip().str.lower().isin(['open_time','open time'])].copy()
    if df.empty:raise ValueError(f'{name}: empty CSV')
    raw=pd.to_numeric(df.open_time,errors='raise').astype('int64')
    lengths={len(str(abs(int(x)))) for x in raw}
    if len(lengths)!=1 or next(iter(lengths)) not in (10,13,16,19):
        raise ValueError(f'{name}: unsupported/mixed time units {lengths}')
    unit={10:'s',13:'ms',16:'us',19:'ns'}[next(iter(lengths))]
    df['time']=pd.to_datetime(raw,unit=unit,utc=True)
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c],errors='raise')
    prices=df[['open','high','low','close']]
    if ((~np.isfinite(df[['open','high','low','close','volume']])).any(axis=1)|
        (prices<=0).any(axis=1)|(df.volume<0)|
        (df.high<df[['open','close','low']].max(axis=1))|
        (df.low>df[['open','close','high']].min(axis=1))|
        (df.time.dt.floor('h')!=df.time)).any():
        raise ValueError(f'{name}: bad OHLC/volume/timestamp')
    return df[['time','open','high','low','close','volume']]


def merge(frames,label):
    if not frames:return pd.DataFrame(columns=['time','open','high','low','close','volume'])
    out=pd.concat(frames).sort_values('time')
    for t,group in out[out.time.duplicated(keep=False)].groupby('time'):
        vals=group.drop(columns='time')
        if not vals.eq(vals.iloc[0]).all().all():
            raise ValueError(f'{label}: conflicting candle {t}')
    return out.drop_duplicates('time').reset_index(drop=True)


def save(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False)


def pair_defs(assets,expiries):
    ds=list(expiries)
    if len(ds)<2 or ds!=sorted(set(ds)):raise ValueError('Expiries must be sorted unique')
    for asset in assets:
        for i in range(len(ds)-1):
            near=ds[i];far=ds[i+1]
            yield {'asset':asset,'near_symbol':symbol(asset,near),'far_symbol':symbol(asset,far),
                   'near_expiry_utc':f'{near}T08:00:00+00:00',
                   'far_expiry_utc':f'{far}T08:00:00+00:00',
                   'venue':'binance_global_usdm_benchmark_only'}


def download(a):
    root=Path(a.output_dir);cache=Path(a.cache_dir)
    assets=list(dict.fromkeys(x.upper() for x in a.assets))
    if not set(assets).issubset(ASSETS):raise ValueError('BTC and ETH USD-M only')
    ds=[x.strip() for x in a.expiries.split(',') if x.strip()]
    pairs=list(pair_defs(assets,ds))
    needed={}
    # Only download actual, historically completed near-term trading windows.
    now=ts(datetime.now(timezone.utc))
    for p in pairs:
        near=ts(p['near_expiry_utc'])
        if near>=now:continue
        for sym in (p['near_symbol'],p['far_symbol']):
            for m in mon_range(near-pd.Timedelta(days=a.history_days),near-pd.Timedelta(hours=24)):
                needed.setdefault(sym,set()).add(m)
    sess=requests.Session();bars={};availability=[]
    for sym in sorted(needed):
        parts=[]; missing=[]
        for m in sorted(needed[sym]):
            u=url_for(sym,m)
            blob=get_verified(u,cache,sess)
            if blob is None:
                missing.append(m);continue
            parts.append(parse_zip(blob,f'{sym} {m}'))
            print(f'{sym} {m}: verified {len(parts[-1])} candles',flush=True)
        frame=merge(parts,sym)
        if not frame.empty:save(frame,root/'bars'/f'{sym}.csv')
        availability.append({'symbol':sym,'bars':len(frame),
                             'missing_months':';'.join(missing),'archive_status':'PARTIAL' if missing else 'COMPLETE',
                             'first':frame.time.min() if len(frame) else None,
                             'last':frame.time.max() if len(frame) else None})
        bars[sym]=frame
    rows=[]
    for p in pairs:
        near=ts(p['near_expiry_utc']);far=ts(p['far_expiry_utc'])
        if near>=now:continue
        n=bars.get(p['near_symbol'],pd.DataFrame());f=bars.get(p['far_symbol'],pd.DataFrame())
        if n.empty or f.empty:
            status='NO_ARCHIVE_OVERLAP';common=0
        else:
            overlap=n[['time']].merge(f[['time']],on='time')
            overlap=overlap[(overlap.time>=near-pd.Timedelta(days=a.history_days)) &
                            (overlap.time<=near-pd.Timedelta(hours=24))]
            common=len(overlap)
            status='READY_FOR_EVALUATION' if common>=a.min_overlap_hours else 'INSUFFICIENT_OVERLAP'
        p.update({'overlap_hours':common,'status':status});rows.append(p)
        print(f"{p['near_symbol']} / {p['far_symbol']}: {status}, overlap_hours={common}",flush=True)
    save(pd.DataFrame(availability),root/'archive_availability.csv')
    save(pd.DataFrame(rows),root/'calendar_pairs.csv')
    print(f'Pairs ready: {sum(p["status"]=="READY_FOR_EVALUATION" for p in rows)}/{len(rows)}; no orders')


def read_bars(path):
    df=pd.read_csv(path)
    required={'time','open','high','low','close','volume'}
    if not required.issubset(df.columns):raise ValueError(f'Bad columns in {path}')
    df.time=pd.to_datetime(df.time,utc=True,format='ISO8601')
    if df.time.duplicated().any():raise ValueError(f'Duplicate hours {path}')
    if (df.time.dt.floor('h')!=df.time).any():raise ValueError('Misaligned times')
    if (df[['open','high','low','close']]<=0).any().any() or (df.volume<0).any() or \
       not np.isfinite(df[['open','high','low','close','volume']]).all().all():
        raise ValueError(f'Invalid price/volume {path}')
    return df.set_index('time').sort_index()


def joined(p,root):
    npath=root/'bars'/f"{p['near_symbol']}.csv"
    fpath=root/'bars'/f"{p['far_symbol']}.csv"
    if not npath.exists() or not fpath.exists():return None
    n=read_bars(npath).add_prefix('n_')
    f=read_bars(fpath).add_prefix('f_')
    j=n.join(f,how='inner')
    until=ts(p['near_expiry_utc'])-pd.Timedelta(hours=24)
    return j[j.index<=until].copy()


def add_zscore(j,lookback_days):
    j=j.copy()
    j['spread_bps']=(j.f_close/j.n_close-1)*10000
    lb=lookback_days*24
    # Strict hourly rolling history: incomplete windows never create a signal.
    stable=(j.index.to_series().diff()==pd.Timedelta(hours=1))
    valid_run=stable.astype(int).groupby((~stable).cumsum()).cumsum()+1
    past=j.spread_bps.shift(1)
    mu=past.rolling(lb,min_periods=lb).mean()
    sd=past.rolling(lb,min_periods=lb).std(ddof=1)
    j['z']=(j.spread_bps-mu)/sd.where(sd>1e-8)
    j.loc[valid_run<lb+1,'z']=float('nan')
    return j


def valid_hourly(j,start,end):
    if start not in j.index or end not in j.index:return False
    if end<start:return False
    expected=int((end-start)/pd.Timedelta(hours=1))+1
    return len(j.loc[start:end])==expected


def trade(p,j,decision,lookback,zthreshold,hold_h,a,mode='SIGNAL'):
    near_exp=ts(p['near_expiry_utc'])
    forced=near_exp-pd.Timedelta(hours=24)
    entry=decision+pd.Timedelta(hours=1)
    if entry not in j.index or entry>=forced:return None,'NO_NEXT_OPEN'
    if not valid_hourly(j,decision,entry):return None,'ENTRY_GAP'
    if min(j.loc[entry,['n_volume','f_volume']])<=0:return None,'ZERO_EXEC_VOLUME'
    end=min(entry+pd.Timedelta(hours=hold_h),forced)
    if end not in j.index or end<=entry:return None,'NO_EXIT_OPEN'
    # Use a completed hourly candle to decide whether to exit NEXT hour.
    if mode=='SIGNAL':
        for clock in j.loc[entry:end-pd.Timedelta(hours=1)].index:
            if pd.notna(j.loc[clock,'z']) and j.loc[clock,'z']<=0:
                end=min(end,clock+pd.Timedelta(hours=1));break
    if not valid_hourly(j,entry,end):return None,'GAP_DURING_HOLD'
    if min(j.loc[end,['n_volume','f_volume']])<=0:return None,'ZERO_EXIT_VOLUME'
    b=j.loc[entry];e=j.loc[end]
    # Equal coin units: long near / short far. Spread compression is profitable.
    gross=(e.n_open-b.n_open)+(b.f_open-e.f_open)
    notional=float(b.n_open)
    fees=(b.n_open+e.n_open+b.f_open+e.f_open)*a.future_fee_bps/10000
    slip=(b.n_open+e.n_open+b.f_open+e.f_open)*a.slippage_bps/10000
    cap=a.margin_pct/100*(b.n_open+b.f_open)
    days=(end-entry)/pd.Timedelta(days=1)
    capital_cost=cap*a.capital_apr_pct/100*days/365
    net=gross-fees-slip-capital_cost
    # Conservative intrabar bound: the two extremes may NOT be simultaneous.
    # Position closes at end OPEN: never include high/low/close of exit hour.
    hold=j.loc[entry:end-pd.Timedelta(hours=1)]
    close_pnl=((hold.n_close-b.n_open)+(b.f_open-hold.f_close))
    end_open_pnl=float((e.n_open-b.n_open)+(b.f_open-e.f_open))
    min_close=min(float(close_pnl.min()),end_open_pnl)
    extreme_pnl=((hold.n_low-b.n_open)+(b.f_open-hold.f_high))
    min_extreme=min(float(extreme_pnl.min()),end_open_pnl)
    breach=bool(min_close-fees-slip < -cap)
    stress_breach=bool(min_extreme-fees-slip < -cap)
    row={'asset':p['asset'],'near_symbol':p['near_symbol'],'far_symbol':p['far_symbol'],
         'near_expiry_utc':p['near_expiry_utc'],'decision_utc':decision.isoformat(),
         'entry_utc':entry.isoformat(),'exit_utc':end.isoformat(),'hold_hours':float((end-entry)/pd.Timedelta(hours=1)),
         'lookback_days':lookback,'entry_threshold_z':zthreshold,
         'decision_z':float(j.loc[decision,'z']) if mode=='SIGNAL' else None,
         'decision_spread_bps':float((j.loc[decision,'f_close']/j.loc[decision,'n_close']-1)*10000),
         'entry_spread_bps':float((b.f_open/b.n_open-1)*10000),
         'exit_spread_bps':float((e.f_open/e.n_open-1)*10000),
         'gross_bps':float(10000*gross/notional),'fee_bps':float(10000*fees/notional),
         'slippage_bps':float(10000*slip/notional),
         'capital_cost_bps':float(10000*capital_cost/notional),'net_bps':float(10000*net/notional),
         'net_capital_bps':float(10000*net/cap),'min_intratrade_close_pnl_bps':float(10000*min_close/notional),
         'min_intrabar_bound_pnl_bps':float(10000*min_extreme/notional),
         'close_margin_breach':breach,'conservative_extreme_margin_flag':stress_breach,
         'mode':mode,'proxy_warning':'HOURLY_OPEN_NOT_EXECUTABLE_DEPTH_OR_REAL_MARGIN'}
    return row,''


def simulate_config(p,joined_df,lb,threshold,hold,a):
    j=add_zscore(joined_df,lb)
    expiry=ts(p['near_expiry_utc'])
    start=expiry-pd.Timedelta(days=a.entry_window_days)
    latest=expiry-pd.Timedelta(hours=max(48,a.exit_buffer_hours))
    candidates=j[(j.index>=start)&(j.index<=latest)&(j.z>=threshold)&
                 (j.n_volume>0)&(j.f_volume>0)]
    errors=[]
    for t in candidates.index:
        result,error=trade(p,j,t,lb,threshold,hold,a)
        if result:return result,''
        errors.append(error)
    return None,('NO_SIGNAL' if candidates.empty else '|'.join(sorted(set(errors))))


def benchmark(p,joined_df,a):
    """Fixed 14d-before-to-24h-before near expiry for every pair, no optimization."""
    exp=ts(p['near_expiry_utc']);d=exp-pd.Timedelta(days=14)
    j=joined_df
    close=j[(j.index>=d)&(j.index<=d+pd.Timedelta(hours=3))]
    if close.empty:return None,'NO_BENCHMARK_ENTRY'
    return trade(p,j,close.index[0],None,None,14*24,a,mode='FIXED_BENCHMARK')


def forward_select(trades,min_train=4,min_hits=4):
    """One pair is one independent out-of-sample episode, no future-leak training."""
    if trades.empty:return pd.DataFrame()
    out=[]
    keys=['lookback_days','entry_threshold_z','max_hold_hours']
    grouped=trades.groupby('near_symbol',sort=False)
    for near,g in grouped:
        first=g.iloc[0]
        entry=ts(first['target_window_start_utc'])
        prior=trades[(pd.to_datetime(trades.near_expiry_utc,utc=True)<entry)&
                     (trades.asset==first.asset)]
        prior_count=prior.near_symbol.nunique()
        row={'asset':first.asset,'near_symbol':near,'far_symbol':first.far_symbol,
             'prior_expired_pairs':prior_count,'status':'INSUFFICIENT_PRIOR_PAIRS',
             'chosen_config':None,'test_net_bps':None,'test_net_capital_bps':None}
        if prior_count<min_train:out.append(row);continue
        possible=[]
        for params,pg in prior.groupby(keys):
            if pg.net_bps.notna().sum()<min_hits:continue
            # Require MIN_HITS distinct pair episodes with a signal; missing episodes
            # do not silently become profitable or zero-return positions.
            hits=pg[pg.net_bps.notna()]
            if hits.near_symbol.nunique()<min_hits:continue
            wins=hits[~hits.close_margin_breach.astype(bool) & ~hits.conservative_extreme_margin_flag.astype(bool)]
            if len(wins)!=len(hits):continue
            values=hits.net_bps.to_numpy(float)
            if len(values)<2:continue
            # 16 attempted configurations: Bonferroni familywise 95% lower bound.
            critical=student_t.ppf(1-0.05/(len(LOOKBACKS)*len(THRESHOLDS)*len(HOLDS)),len(values)-1)
            lower=values.mean()-critical*values.std(ddof=1)/np.sqrt(len(values))
            possible.append((float(lower),tuple(params)))
        if not possible or max(possible)[0]<=0:
            row['status']='NO_POSITIVE_TRAIN_LOWER_BOUND'
        else:
            best=max(possible,key=lambda x:x[0])[1]
            row['chosen_config']=f'{best[0]}d_z{best[1]}_{best[2]}h'
            selected=g[(g.lookback_days==best[0]) &
                       (g.entry_threshold_z==best[1]) & (g.max_hold_hours==best[2])]
            if len(selected)!=1:raise ValueError('Non-unique parameter row')
            candidate=selected.iloc[0]
            if pd.isna(candidate.net_bps):row['status']='NO_SIGNAL_OOS'
            elif bool(candidate.close_margin_breach) or bool(candidate.conservative_extreme_margin_flag):
                row['status']='MARGIN_STRESS_REJECT'
            else:
                row['status']='PAPER_SELECTED_PROXY_ONLY'
                row['test_net_bps']=float(candidate.net_bps)
                row['test_net_capital_bps']=float(candidate.net_capital_bps)
        out.append(row)
    return pd.DataFrame(out)


def evaluate(a):
    root=Path(a.output_dir)
    suffix='' if a.scenario=='baseline' else '_'+a.scenario
    def result_path(name):return root/(name+suffix+'.csv')
    manifest=root/'calendar_pairs.csv'
    if not manifest.exists():raise ValueError(f'No manifest at {manifest}; run download')
    p=pd.read_csv(manifest)
    if p.empty:print('No pairs downloaded');return
    diagnostics=[];rows=[];bench=[]
    for _,pair in p.iterrows():
        if pair.status!='READY_FOR_EVALUATION':
            diagnostics.append({'pair':pair.near_symbol+'/'+pair.far_symbol,'reason':pair.status});continue
        j=joined(pair,root)
        if j is None or j.empty:
            diagnostics.append({'pair':pair.near_symbol+'/'+pair.far_symbol,'reason':'NO_OVERLAP'});continue
        expiry=ts(pair.near_expiry_utc)
        b,reason=benchmark(pair,j,a)
        if b:bench.append(b)
        else:diagnostics.append({'pair':pair.near_symbol+'/'+pair.far_symbol,'reason':'BENCH_'+reason})
        for lb in LOOKBACKS:
            for z in THRESHOLDS:
                for h in HOLDS:
                    v,why=simulate_config(pair,j,lb,z,h,a)
                    common={'asset':pair.asset,'near_symbol':pair.near_symbol,'far_symbol':pair.far_symbol,
                            'near_expiry_utc':pair.near_expiry_utc,
                            'target_window_start_utc':(expiry-pd.Timedelta(days=a.entry_window_days)).isoformat(),
                            'lookback_days':lb,'entry_threshold_z':z,'max_hold_hours':h,
                            'failure_reason':why if not v else ''}
                    if v:common.update(v)
                    else:common.update({'net_bps':float('nan'),'net_capital_bps':float('nan'),
                                        'close_margin_breach':False,'conservative_extreme_margin_flag':False})
                    rows.append(common)
        print(f'{pair.near_symbol} vs {pair.far_symbol}: completed {len(LOOKBACKS)*len(THRESHOLDS)*len(HOLDS)} settings')
    save(pd.DataFrame(diagnostics,columns=['pair','reason']),result_path('eval_diagnostics'))
    if not rows:
        print('NO VERIFIED OVERLAPPING FUTURES PAIRS. See eval_diagnostics.csv; no proxy results fabricated.')
        return
    r=pd.DataFrame(rows)
    save(r,result_path('all_config_diagnostics'))
    if bench:
        b=pd.DataFrame(bench)
        save(b,result_path('fixed_benchmark'))
        print('\nFIXED 14D TO 24H PRE-EXPIRY BENCHMARK (HOURLY OPEN PROXIES; NOT EXECUTABLE)')
        print(b[['asset','near_symbol','far_symbol','gross_bps','net_bps','close_margin_breach']].to_string(index=False))
    w=forward_select(r,a.min_train_pairs,a.min_train_hits)
    save(w,result_path('walk_forward'))
    print('\nSTRICT WALK-FORWARD (SELECTION ONLY FROM PREVIOUS EXPIRED NEAR CONTRACTS)')
    print(w.to_string(index=False))
    s=w[w.status=='PAPER_SELECTED_PROXY_ONLY']
    print(f'Pairs={r.near_symbol.nunique()} configs/pair={len(LOOKBACKS)*len(THRESHOLDS)*len(HOLDS)} '
          f'WFO selected={len(s)}, net bps mean=',f'{s.test_net_bps.mean():.2f}' if len(s) else 'N/A')
    print('BINANCE GLOBAL BENCHMARK; NOT U.S. ACCESSIBLE, NO EXECUTABLE QUOTES, NO LIVE ORDERS')


def self_test():
    import tempfile
    from argparse import Namespace
    assert symbol('BTC','2026-09-25')=='BTCUSDT_260925'
    assert mon_range('2026-08-15T00:00:00Z','2026-09-24T23:00:00Z')==['2026-08','2026-09']
    assert url_for('BTCUSDT_260925','2026-09').endswith('BTCUSDT_260925-1h-2026-09.zip')
    # Regression: futures archives can have headers and millisecond/microsecond times.
    for mul in (1000,1_000_000):
        t=int(ts('2026-06-01T00:00:00Z').timestamp())*mul
        rows=('open_time,open,high,low,close,volume,close_time,qvolume,num_trades,taker_base,taker_quote,ignore\n'
              f'{t},100,101,99,100,10,0,0,0,0,0,0\n')
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('test.csv',rows)
        got=parse_zip(buf.getvalue(),'test_archive')
        assert len(got)==1 and got.time.iloc[0]==ts('2026-06-01T00:00:00Z')
    d=list(pair_defs(['BTC'],['2026-06-26','2026-09-25']))
    assert len(d)==1 and d[0]['far_symbol']=='BTCUSDT_260925'
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);p=d[0]
        idx=pd.date_range('2026-05-01T00:00:00Z','2026-06-25T08:00:00Z',freq='h')
        # Quoted far-near spread rises sharply 18d before near expiry, then reverts.
        base=np.full(len(idx),10000.)
        far=base+50.+np.sin(np.arange(len(idx))/11.)*2.
        impulse=ts('2026-06-08T00:00:00Z')
        far[idx>=impulse]+=200.
        far[idx>=impulse+pd.Timedelta(days=2)]-=200.
        n=pd.DataFrame({'time':idx,'open':base,'high':base+1,'low':base-1,'close':base,'volume':10.})
        f=pd.DataFrame({'time':idx,'open':far,'high':far+1,'low':far-1,'close':far,'volume':10.})
        save(n,root/'bars'/f"{p['near_symbol']}.csv")
        save(f,root/'bars'/f"{p['far_symbol']}.csv")
        j=joined(p,root);assert len(j)>800
        a=Namespace(future_fee_bps=0.,slippage_bps=0.,margin_pct=20.,capital_apr_pct=0.,
                    entry_window_days=21,exit_buffer_hours=24)
        z=add_zscore(j,7)
        assert pd.isna(z.z.iloc[50]) and pd.notna(z.z.loc[impulse])
        v,why=simulate_config(p,j,7,1.5,72,a)
        assert v is not None and v['gross_bps']>0,(v,why)
        assert ts(v['entry_utc'])>ts(v['decision_utc'])
        assert v['exit_spread_bps']<v['entry_spread_bps']
        a.future_fee_bps=200.
        costly,_=simulate_config(p,j,7,1.5,72,a)
        assert costly['net_bps']<v['net_bps']
        # Missing candle in holding interval must prevent a fabricated fill.
        missing=add_zscore(j,7).drop(index=ts(v['entry_utc'])+pd.Timedelta(hours=3))
        trade_row,error=trade(p,missing,ts(v['decision_utc']),7,1.5,72,a)
        assert trade_row is None and error=='GAP_DURING_HOLD',(trade_row,error)
        # Strict prior expiry embargo, and no hindsight if all earlier trades lose.
        allrows=[]
        for i in range(8):
            expiry=ts('2024-03-29T08:00:00Z')+pd.DateOffset(months=3*i)
            row={'asset':'BTC','near_symbol':f'N{i}','far_symbol':f'F{i}',
                 'near_expiry_utc':expiry.isoformat(),
                 'target_window_start_utc':(expiry-pd.Timedelta(days=21)).isoformat(),
                 'lookback_days':7,'entry_threshold_z':1.5,'max_hold_hours':24,
                 'net_bps':-20.,'net_capital_bps':-50.,
                 'close_margin_breach':False,'conservative_extreme_margin_flag':False}
            allrows.append(row)
        walk=forward_select(pd.DataFrame(allrows))
        assert (walk.iloc[:4].status=='INSUFFICIENT_PRIOR_PAIRS').all()
        assert walk.iloc[-1].status=='NO_POSITIVE_TRAIN_LOWER_BOUND'
        # No lookahead in entry signal: spike at instant does not enter at same instant.
        assert ts(v['entry_utc'])==ts(v['decision_utc'])+pd.Timedelta(hours=1)
        # Strict missing history: all-NaN z-score never triggers.
        assert simulate_config(p,j.iloc[:100],7,2.,24,a)[0] is None
    print('PASS V5: archived pair identifiers, causal 7d/14d z-score, next-open fills, four legs,')
    print('         strict overlap, spread compression, missing-bar rejection, expired-pair WFO')
    print('OFFLINE READ-ONLY; NO ORDERS, CREDENTIALS, LIVE EXCHANGE ACCESS')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    dl=sub.add_parser('download')
    dl.add_argument('--assets',nargs='+',default=['BTC'])
    dl.add_argument('--expiries',default=','.join(EXPIRIES))
    dl.add_argument('--output-dir',default=str(ROOT))
    dl.add_argument('--cache-dir',default=str(DEFAULT_CACHE))
    dl.add_argument('--history-days',type=int,default=60)
    dl.add_argument('--min-overlap-hours',type=int,default=450)
    e=sub.add_parser('evaluate')
    e.add_argument('--output-dir',default=str(ROOT))
    e.add_argument('--scenario',default='baseline',help='Safe suffix for multiple cost runs')
    e.add_argument('--future-fee-bps',type=float,default=5.)
    e.add_argument('--slippage-bps',type=float,default=5.)
    e.add_argument('--margin-pct',type=float,default=20.)
    e.add_argument('--capital-apr-pct',type=float,default=0.)
    e.add_argument('--entry-window-days',type=int,default=21)
    e.add_argument('--exit-buffer-hours',type=int,default=24)
    e.add_argument('--min-train-pairs',type=int,default=4)
    e.add_argument('--min-train-hits',type=int,default=4)
    a=ap.parse_args()
    if a.command=='self-test':self_test()
    elif a.command=='download':
        if not(25<=a.history_days<=120) or a.min_overlap_hours<300:ap.error('invalid data window')
        download(a)
    else:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,30}',a.scenario):ap.error('Invalid scenario label')
        if (min(a.future_fee_bps,a.slippage_bps,a.margin_pct,a.capital_apr_pct)<0 or
            a.margin_pct<=0 or a.entry_window_days<15 or a.exit_buffer_hours<24 or
            a.min_train_pairs<2 or a.min_train_hits<2):ap.error('invalid costs/window')
        evaluate(a)

if __name__=='__main__':main()
