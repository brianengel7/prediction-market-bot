#!/usr/bin/env python3
"""BreakTheBank V9: local market/day L2 request plan + verified BBO export for V8.

No network, no orders, no credentials. Uses the exact V8 `intents.csv` and
Kalshi Eye `polymarket_l2` raw CSV/Parquet files (NOT minute candle or trades).

Workflow:
  python3 -B breakthebank_v9_l2_prepare_readonly.py --self-test
  python3 -B breakthebank_v9_l2_prepare_readonly.py --manifest .../intents.csv --plan-out ~/.../l2_plan.csv
  python3 -B breakthebank_v9_l2_prepare_readonly.py --manifest .../intents.csv \
    --l2-root ~/.../kalshieye-data --out ~/.../bbo.csv

Parquet needs `python -m pip install pyarrow`.

L2 rows (documented Kalshi Eye schema): ts_recv, ticker, series,
    is_snapshot, side, price, size.
State is cleared at the first row of each new snapshot timestamp per token;
price-change sizes are ABSOLUTE resting shares, not deltas. No state is assumed
until a full snapshot appears. Reads only relevant local market/day files.
Quotes are exported at their real last receive timestamps, as observed before
requested order-intent times + delays. Missing snapshot is NOT filled in.
V8 subsequently applies stale-book filters and conservative top-of-book IOC.

Limitations: This data converter assumes raw L2 files are recorder-order within
 each token, does not establish historical message drop/coverage quality,
 does not measure wallet public feed lag or guarantee actual IOC fills.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import re

RAW_FILE_RE = re.compile(r'^polymarket_l2-(0x[a-f0-9]{64})-(\d{4}-\d{2}-\d{2})\.(?:parquet|csv)$', re.I)
FIELDS = ['ts_recv','token_id','bid','bid_size','ask','ask_size']
RAW_COLS = ['ts_recv','ticker','series','is_snapshot','side','price','size']

def parse_time(x):
    if isinstance(x,(int,float)):
        return float(x)
    if hasattr(x,'timestamp'):
        if x.tzinfo is None: x=x.replace(tzinfo=timezone.utc)
        return x.timestamp()
    v=str(x).strip()
    if v.isdigit():
        # Parquet converted to CSV may represent timestamp in ns or ms.
        iv=int(v)
        if iv>10**17:return iv/10**9
        if iv>10**14:return iv/10**6
        if iv>10**11:return iv/10**3
        return float(iv)
    try:
        return float(v)
    except ValueError:
        dt=datetime.fromisoformat(v.replace('Z','+00:00'))
        if dt.tzinfo is None:
            dt=dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

def read_intents(path,delays):
    requests=defaultdict(set)
    needed=defaultdict(set)
    market_titles={}
    count=0
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        required={'market','token_id','wallet_trade_ts_utc','title'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Intents missing {sorted(required-set(reader.fieldnames or []))}')
        for line,r in enumerate(reader,2):
            market=str(r['market']).lower().strip()
            token=str(r['token_id']).strip()
            if not re.fullmatch('0x[a-f0-9]{64}',market) or not token.isdigit():
                raise ValueError(f'Unexpected market/token at manifest line {line}')
            ts=parse_time(r['wallet_trade_ts_utc'])
            if not math.isfinite(ts):raise ValueError('Invalid intent time')
            needed[market].add(token)
            market_titles[market]=r['title']
            for d in delays:
                t=ts+d
                requests[(market,token)].add(t)
            count+=1
    if not count:raise ValueError('Manifest has no intents')
    return requests,needed,market_titles,count

def required_files(requests):
    pairs=set()
    for (market,_), times in requests.items():
        for t in times:
            date=datetime.fromtimestamp(t,timezone.utc).date().isoformat()
            pairs.add((market,date))
    return pairs

def write_plan(path,requests,titles):
    pairs=sorted(required_files(requests))
    queries=defaultdict(int)
    for (market,_),times in requests.items():
        for t in times:queries[(market,datetime.fromtimestamp(t,timezone.utc).date().isoformat())]+=1
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['date','market','target_times','filename','kalshieye_relative_path','title'])
        w.writeheader()
        for market,date in pairs:
            name=f'polymarket_l2-{market}-{date}.parquet'
            rel=f'flat/v1/polymarket_l2/date={date}/series={market}/{name}'
            w.writerow({'date':date,'market':market,'target_times':queries[(market,date)],
                        'filename':name,'kalshieye_relative_path':rel,'title':titles[market]})
    return pairs

def iter_file_rows(path):
    if path.suffix.lower()=='.csv':
        with path.open(newline='',encoding='utf-8') as f:
            r=csv.DictReader(f)
            if not set(RAW_COLS).issubset(r.fieldnames or []):
                raise ValueError(f'L2 file missing required columns: {path}')
            yield from r
    else:
        try:import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError('Install pyarrow: python3 -m pip install pyarrow') from e
        pf=pq.ParquetFile(path)
        if not set(RAW_COLS).issubset(pf.schema.names):
            raise ValueError(f'L2 parquet missing required columns: {path}')
        for batch in pf.iter_batches(batch_size=50000,columns=RAW_COLS):
            obj=batch.to_pydict()
            for values in zip(*(obj[k] for k in RAW_COLS)):
                yield dict(zip(RAW_COLS,values))

class Book:
    def __init__(self):
        self.levels={'bid':{},'ask':{}}
        self.known=False
        self.ts=None
        self.last_snapshot=None
        self.prev_ts=None

    def update(self,r):
        t=parse_time(r['ts_recv'])
        if not math.isfinite(t) or t<1e9:raise ValueError('Invalid L2 receive timestamp')
        if self.prev_ts is not None and t+1e-8<self.prev_ts:
            raise ValueError('L2 rows not chronological per token; do NOT silently sort ties')
        self.prev_ts=t
        snap=r['is_snapshot']
        snap = snap if isinstance(snap,bool) else str(snap).lower() in ('true','t','1')
        if snap:
            if self.last_snapshot != t:
                self.levels={'bid':{},'ask':{}}
                self.known=True
                self.last_snapshot=t
        else:
            self.last_snapshot=None
        side=str(r['side'] or '').strip().lower()
        if side in ('','none','null'):
            if not snap:raise ValueError('A null-side delta is not a valid update')
        else:
            if side not in ('bid','ask'):raise ValueError(f'Unexpected side {side}')
            try:
                p=float(r['price']); size=float(r['size'])
            except (ValueError,TypeError):raise ValueError('Invalid L2 level')
            if not math.isfinite(p) or not 0<p<1 or not math.isfinite(size) or size<0:
                raise ValueError('Invalid L2 price/size')
            if not self.known:
                # Never infer the full book from an incremental level update.
                self.ts=t
                return
            if size==0: self.levels[side].pop(p,None)
            else:self.levels[side][p]=size
        self.ts=t

    def quote(self,token):
        if not self.known or self.ts is None:return None
        bids=self.levels['bid'];asks=self.levels['ask']
        bp=max(bids) if bids else None
        ap=min(asks) if asks else None
        if bp is not None and ap is not None and bp>ap+1e-8:
            raise ValueError('Crossed reconstructed book: archive rows malformed/order mismatched')
        return (self.ts,token,bp,bids.get(bp) if bp is not None else None,
                ap,asks.get(ap) if ap is not None else None)

def normalize_to_targets(rows,targets):
    # rows are original recorder order; timestamps for each token must be nondecreasing.
    books=defaultdict(Book)
    pointers={t:0 for t in targets}
    results=[]
    missing=Counter()
    def drain(tok,upto, inclusive):
        ix=pointers[tok]
        arr=targets[tok]
        book=books[tok]
        while ix<len(arr) and (arr[ix]<=upto if inclusive else arr[ix]<upto):
            quote=book.quote(tok)
            if quote is None:missing['no_initial_snapshot']+=1
            else:results.append(quote)
            ix+=1
        pointers[tok]=ix
    for r in rows:
        tok=str(r['ticker']).strip()
        if tok not in targets:continue
        stamp=parse_time(r['ts_recv'])
        drain(tok,stamp,False)
        books[tok].update(r)
    for tok in targets:
        drain(tok,float('inf'),True)
    return results,missing

def discover(root):
    index=defaultdict(list)
    for path in root.rglob('*'):
        if not path.is_file():continue
        m=RAW_FILE_RE.fullmatch(path.name)
        if m:index[(m.group(1).lower(),m.group(2))].append(path)
    for pair,options in index.items():
        if len(options)>1:
            raise ValueError(f'Duplicate L2 files for {pair}: {options}')
    return {pair:paths[0] for pair,paths in index.items()}

def export_local(requests,root,out):
    dates=required_files(requests)
    files=discover(root)
    found=dates&set(files)
    missing=dates-set(files)
    outrows=[]
    stats=Counter()
    for market,date in sorted(found):
        needed=defaultdict(list)
        for (m,tok),ts in requests.items():
            if m==market:
                # Request all target times on or after the beginning of this date.
                day=date
                needed[tok]=sorted(x for x in ts if datetime.fromtimestamp(x,timezone.utc).date().isoformat()==day)
        needed={k:v for k,v in needed.items() if v}
        if not needed:continue
        path=files[(market,date)]
        # Require actual observed snapshots within downloaded file (fail closed).
        rows, miss=normalize_to_targets(iter_file_rows(path),needed)
        outrows.extend(rows)
        stats.update(miss)
        stats['L2_market_days_parsed']+=1
    # V8 requires nonconflicting duplicate token+time quote snapshots.
    uniq={}
    for r in outrows:
        k=(r[1],r[0]); old=uniq.get(k)
        if old is not None and old!=r:raise ValueError('Conflicting book states at same token + receive time')
        uniq[k]=r
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(FIELDS)
        for r in sorted(uniq.values(),key=lambda x:(x[0],x[1])):
            w.writerow([f'{r[0]:.6f}',r[1]]+[f'{v:.10f}' if v is not None else '' for v in r[2:]])
    stats['market_days_requested']=len(dates)
    stats['market_days_missing_file']=len(missing)
    stats['quote_rows_output']=len(uniq)
    return stats,missing

def self_test():
    def row(t,tok,snap,side,p=None,n=None):
        return {'ts_recv':t,'ticker':tok,'series':'c','is_snapshot':snap,'side':side,'price':p,'size':n}
    a=1790000000
    rows=[row(a,'1',True,'bid',.4,20),row(a,'1',True,'ask',.5,10),
          row(a+1,'1',False,'ask',.5,0),row(a+1,'1',False,'ask',.52,14),
          row(a+3,'1',True,'bid',.42,10),row(a+3,'1',True,'ask',.45,30),
          row(a+3,'2',False,'bid',.4,10)]
    q,missing=normalize_to_targets(rows,{'1':[a+.5,a+1.5,a+3.5],'2':[a+4]})
    assert [(r[2],r[4]) for r in q]==[(.4,.5),(.4,.52),(.42,.45)],q
    assert q[0][0]==a and q[1][0]==a+1
    assert missing['no_initial_snapshot']==1,missing
    b=Book();b.update(row(a,'1',True,None));assert b.quote('1')[2] is None
    print('PASS V9: exact snapshot reset, absolute size changes, no pre-snapshot quotes,')
    print('         as-received as-of BBO, null levels, no lookahead and multiple targets')
    print('OFFLINE; no downloads, credentials, orders or project edits')

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--manifest',type=Path)
    p.add_argument('--delays',default='5,15,30,60')
    p.add_argument('--plan-out',type=Path)
    p.add_argument('--l2-root',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.self_test:self_test();return
    if not a.manifest:p.error('--manifest required')
    delays=[float(v) for v in a.delays.split(',')]
    if not delays or any(not math.isfinite(x) or x<0 for x in delays):p.error('Invalid delays')
    requests,_,titles,n=read_intents(a.manifest.expanduser(),delays)
    print(f'V9 L2 plan: {n:,} intents, {len(requests)} market/token pairs, delays {delays}')
    print(f'Market/day files: {len(required_files(requests))}')
    for output in (a.plan_out,a.out):
        if output:
            path=output.expanduser().resolve()
            if path.exists() or str(path).startswith('/workspaces/'):
                p.error('Refusing overwrite or output under /workspaces/')
    if a.plan_out:
        dest=a.plan_out.expanduser().resolve();dest.parent.mkdir(parents=True,exist_ok=True)
        write_plan(dest,requests,titles)
        print('Plan:',dest)
    if a.l2_root and a.out:
        root=a.l2_root.expanduser()
        if not root.is_dir():p.error('L2 directory not found')
        dest=a.out.expanduser().resolve();dest.parent.mkdir(parents=True,exist_ok=True)
        stats,missing=export_local(requests,root,dest)
        print('BBO:',dest)
        print('Coverage:',json.dumps(dict(stats),sort_keys=True))
        print('Missing market/day inputs:',len(missing))
        for market,day in sorted(missing)[:15]:print('  MISSING',day,market)
        print('WARNING: missing archive days and pre-snapshot periods are not inferred; V8 rejects them')
    elif a.l2_root or a.out:
        p.error('--l2-root and --out must be provided together')
    if not a.plan_out and not a.l2_root:p.error('Specify --plan-out or --l2-root and --out')
    print('READ ONLY — no network, orders, keys, GitHub edits or repo writes')

if __name__=='__main__':main()
