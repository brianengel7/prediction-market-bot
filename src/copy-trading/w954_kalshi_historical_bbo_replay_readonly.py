#!/usr/bin/env python3
"""W954 -> Kalshi real historical BBO/snapshot replay. READ ONLY.

Run from repository root; put this file in src/copy-trading/ beside
w954_kalshi_crossvenue_backtest_readonly.py. Requires locally-downloaded
historical top-of-book (bbo) or top-10 snapshots (book_snap_1s) Parquet/CSV.
Does not download files, place orders, modify files, or change repository.

Crucial: Wallet public trade timestamps are not verified OBSERVATION timestamps.
Add plausible publication/processing latency with --latencies. Historical quotes
show hypothetical taker liquidity, not a proven fill (queue, trade-through,
exchange latency, and cancellations remain unverified).
"""
from __future__ import annotations
import argparse
import bisect
import csv
import math
import random
import runpy
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from types import SimpleNamespace

CORE = Path('src/copy-trading/w954_kalshi_crossvenue_backtest_readonly.py')
FIELDS_BBO = ['ts_recv','ticker','yes_bid','yes_bid_size','yes_ask','yes_ask_size']
FIELDS_SNAP = ['ts','ticker'] + [f'{p}_{item}_{i}' for p in ('bid','ask') for item in ('px','sz') for i in range(10)]


def to_seconds(x):
    if x is None: return None
    if isinstance(x, datetime):
        if x.tzinfo is None: x=x.replace(tzinfo=timezone.utc)
        return x.timestamp()
    s=str(x).strip()
    if not s or s.lower() in {'nan','none','nat'}: return None
    try:
        y=float(s)
        if not math.isfinite(y):return None
        if abs(y)>=1e17: return y/1e9
        if abs(y)>=1e14: return y/1e6
        if abs(y)>=1e11: return y/1e3
        return y
    except ValueError:
        z=datetime.fromisoformat(s.replace('Z','+00:00'))
        if z.tzinfo is None:z=z.replace(tzinfo=timezone.utc)
        return z.timestamp()


def numeric(x):
    try:
        a=float(x)
        return a if math.isfinite(a) else None
    except (ValueError,TypeError):return None


def ensure_levels(row,kind):
    """Both return YES and NO *ask* ladders, in ascending cost order."""
    if kind=='bbo':
        a=numeric(row.get('yes_ask')); a_qty=numeric(row.get('yes_ask_size'))
        b=numeric(row.get('yes_bid')); b_qty=numeric(row.get('yes_bid_size'))
        yes=[(a,a_qty)] if a is not None and 0<a<1 and a_qty and a_qty>0 else []
        no=[(1-b,b_qty)] if b is not None and 0<b<1 and b_qty and b_qty>0 else []
        return yes,no
    yes=[]; no=[]
    for i in range(10):
        ask=numeric(row.get(f'ask_px_{i}')); ask_sz=numeric(row.get(f'ask_sz_{i}'))
        bid=numeric(row.get(f'bid_px_{i}')); bid_sz=numeric(row.get(f'bid_sz_{i}'))
        if ask is not None and 0<ask<1 and ask_sz and ask_sz>0:yes.append((ask,ask_sz))
        if bid is not None and 0<bid<1 and bid_sz and bid_sz>0:no.append((1-bid,bid_sz))
    return sorted(yes),sorted(no)


def sweep(levels, qty):
    remaining=float(qty); total=0.0
    for px, sz in sorted(levels):
        take=min(remaining,sz)
        total+=take*px
        remaining-=take
        if remaining<=1e-9:return total/qty
    return None


def files_in(root,kind):
    if root.is_file():return [root]
    if not root.is_dir():raise RuntimeError(f'Book path not found: {root}')
    paths=[]
    for ext in ('*.parquet','*.csv'):
        for p in root.rglob(ext):
            if f'series=KXBTC15M' not in str(p) and 'KXBTC15M' not in p.name: continue
            # Prevent accidentally reading unrelated data within a full lake folder.
            parts=set(p.parts)
            if kind=='bbo' and 'bbo' not in parts and not p.name.startswith('bbo-'):continue
            if kind=='book_snap_1s' and 'book_snap_1s' not in parts and not p.name.startswith('book_snap_1s-'):continue
            paths.append(p)
    return sorted(set(paths))


def source_rows(path,kind,tickers):
    cols=FIELDS_BBO if kind=='bbo' else FIELDS_SNAP
    if path.suffix=='.csv':
        with path.open(newline='') as f:
            r=csv.DictReader(f)
            if not set(cols).issubset(r.fieldnames or []):
                raise ValueError(f'Missing {set(cols)-set(r.fieldnames or [])} in {path}')
            for row in r:
                if row.get('ticker') in tickers:yield row
    elif path.suffix=='.parquet':
        try:
            import pyarrow.dataset as ds
        except ImportError:
            raise RuntimeError('Reading Parquet requires pyarrow: python3 -m pip install pyarrow')
        data=ds.dataset(str(path),format='parquet')
        missing=set(cols)-set(data.schema.names)
        if missing:raise ValueError(f'Missing {sorted(missing)} in {path}')
        scanner=data.scanner(columns=cols,filter=ds.field('ticker').isin(sorted(tickers)),batch_size=8192)
        for batch in scanner.to_batches():
            yield from batch.to_pylist()
    else:raise RuntimeError('Historical book data must be .parquet or .csv')


def ingest(root,kind,tickers):
    out=defaultdict(list)
    paths=files_in(root,kind)
    if not paths:raise RuntimeError(f'No {kind} KXBTC15M Parquet/CSV files under {root}')
    n=0; bad=0
    for path in paths:
        matched=0
        for r in source_rows(path,kind,tickers):
            ts=to_seconds(r.get('ts_recv' if kind=='bbo' else 'ts'))
            if ts is None:bad+=1;continue
            # Second-level snapshots are end-of-second aggregates: only use
            # the observation from the following second (avoid lookahead).
            if kind=='book_snap_1s':ts+=1.0
            yes,no=ensure_levels(r,kind)
            out[str(r['ticker'])].append((ts,yes,no))
            n+=1;matched+=1
        print(f'  Loaded {matched:,} relevant {kind} rows from {path.name}',flush=True)
    for t in out:out[t].sort(key=lambda x:x[0])
    return out,len(paths),n,bad


def evaluate(signals,db,kind,qty,latency,quote_age,fee_mult,max_price=None):
    rows=[]; rejected=Counter()
    for s in signals:
        entry=float(s['signal_ts'])+latency
        if entry>=float(s['start'])+900:
            rejected['after_market_close']+=1;continue
        books=db.get(s['ticker'],[])
        if not books:rejected['no_market_book']+=1;continue
        times=[x[0] for x in books]
        j=bisect.bisect_right(times,entry)-1
        if j<0:rejected['no_prior_quote']+=1;continue
        ts,yes,no=books[j]
        age=entry-ts
        if age < -1e-8: raise AssertionError('Lookahead quote detected')
        if age>quote_age:rejected['quote_too_old']+=1;continue
        ladder=yes if s['kalshi_side']=='yes' else no
        price=sweep(ladder,qty)
        if price is None:rejected['insufficient_ask_depth']+=1;continue
        if max_price is not None and price>max_price:
            rejected['over_price_cap']+=1;continue
        fee=math.ceil(100*fee_mult*.07*qty*price*(1-price)-1e-12)/100
        payout=qty*int(s['kalshi_win'])
        pnl=payout-qty*price-fee
        rows.append({**s,'quote_ts':ts,'quote_age':age,'vwap':price,'fee':fee,
                     'cost':qty*price+fee,'pnl':pnl,'per_share':pnl/qty})
    return rows,rejected


def summary(name,rows,total,qty):
    if not rows:
        print(f'  {name}: 0/{total} simulated fills');return
    cost=sum(r['cost'] for r in rows);pnl=sum(r['pnl'] for r in rows)
    print(f'  {name:<26} fills={len(rows):3}/{total:<3} wins={sum(r["kalshi_win"] for r in rows)/len(rows)*100:5.1f}% '
          f'medVwap={median(r["vwap"]):.3f} medQuoteAge={median(r["quote_age"]):.2f}s '
          f'netPnL=${pnl:+7.2f} netROI={100*pnl/cost:+7.2f}% net/share={pnl/(qty*len(rows)):+.4f}')


def boot_ci(rows,qty,iters=3000):
    byday=defaultdict(list)
    for r in rows:
        d=datetime.fromtimestamp(r['start'],timezone.utc).date().isoformat()
        byday[d].append(r)
    keys=list(byday)
    if len(keys)<3:return None
    rng=random.Random(954)
    values=[]
    for _ in range(iters):
        sample=[r for _ in keys for r in byday[rng.choice(keys)]]
        values.append(sum(r['pnl'] for r in sample)/(qty*len(sample)))
    values.sort()
    return values[int(.025*iters)],values[int(.975*iters)],len(keys)


def self_test():
    q=10
    y,n=ensure_levels({'yes_bid':.38,'yes_bid_size':15,'yes_ask':.42,'yes_ask_size':20},'bbo')
    assert abs(sweep(y,q)-.42)<1e-10 and abs(sweep(n,q)-.62)<1e-10
    y,n=ensure_levels({'ask_px_0':.40,'ask_sz_0':4,'ask_px_1':.43,'ask_sz_1':8,
                       'bid_px_0':.60,'bid_sz_0':9,'bid_px_1':.58,'bid_sz_1':9},'book_snap_1s')
    assert abs(sweep(y,q)-.418)<1e-10 and abs(sweep(n,q)-.402)<1e-10
    s={'ticker':'X','signal_ts':101,'start':90,'kalshi_side':'yes','kalshi_win':1}
    db={'X':[(101,[ (.42,20) ],[(.58,20)]),(106,[ (.93,20) ],[(.07,20)])]}
    out,reason=evaluate([s],db,'bbo',10,0,2,1)
    assert len(out)==1 and abs(out[0]['vwap']-.42)<1e-10 and out[0]['pnl']>0 and not reason
    out,reason=evaluate([s],db,'bbo',10,3,2,1)
    assert not out and reason['quote_too_old']==1
    out,reason=evaluate([s],db,'bbo',10,5,2,1)
    assert len(out)==1 and abs(out[0]['vwap']-.93)<1e-10
    assert to_seconds(1792332000000000000)==1792332000
    print('PASS: read-only as-of quote (no lookahead), NO conversion, ladder depth, fees, latency and age checks')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--book-root',type=Path,help='Local file or downloaded lake folder, required for real replay')
    parser.add_argument('--book-kind',choices=['bbo','book_snap_1s'],default='bbo')
    parser.add_argument('--markets-file',type=Path,help='Optional local JSON/JSONL of Kalshi markets; otherwise public GET')
    parser.add_argument('--wallet',type=Path)
    parser.add_argument('--qty',type=int,default=10)
    parser.add_argument('--fee-multiplier',type=float,default=1.)
    parser.add_argument('--latencies',type=float,nargs='+',default=[0,3,5,10,20,30])
    parser.add_argument('--quote-age',type=float,default=5.,help='Maximum age of the last observed quote in seconds')
    parser.add_argument('--min-repeat-age',type=int,default=4)
    parser.add_argument('--max-repeat-age',type=int,default=30)
    parser.add_argument('--first-price-max',type=float,default=.40)
    parser.add_argument('--max-kalshi-price',type=float,default=None)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.book_root is None:parser.error('--book-root must point to downloaded historical BBO/snapshots')
    if args.qty<=0 or args.quote_age<0 or args.fee_multiplier<0 or min(args.latencies)<0:
        parser.error('Invalid quantity, quote age, fee, or latency')
    if not CORE.is_file():raise RuntimeError(f'Missing existing local research loader: {CORE}')
    core=runpy.run_path(str(CORE))
    params=dict(wallet=args.wallet,min_repeat_age=args.min_repeat_age,
                max_repeat_age=args.max_repeat_age,upper_first_price=args.first_price_max)
    repeat=core['extract_signals'](SimpleNamespace(strategy='repeat',**params))
    first=core['extract_signals'](SimpleNamespace(strategy='first',**params))
    print('Matching Kalshi settlement from historical markets (GET only unless --markets-file) ...',flush=True)
    market_records=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](repeat)
    rm,errs=core['match_markets'](repeat,market_records)
    fm,_=core['match_markets'](first,market_records)
    keys={(r['start'],r['side']) for r in rm}
    first_matched=[r for r in fm if (r['start'],r['side']) in keys]
    tickers={r['ticker'] for r in rm}
    print(f'BBO test cohort: {len(rm)} matched repeat markets, first matched={len(first_matched)}, errors={dict(errs)}')
    db,num_files,n,bad=ingest(args.book_root,args.book_kind,tickers)
    print(f'Loaded files={num_files} relevant rows={n:,} invalid timestamps={bad}, book markets={len(db)}; no local writes')
    if not db:
        print('No overlap between archived quote files and wallet repeat markets. Try different dates; no P&L calculated.')
        return
    print('Price = as-of historical ASK (VWAP if 10-level snapshot); no future quote, enforce depth and quote age.')
    print('Wallet publication lag UNKNOWN. Latency scenarios are hypothetical; quoted liquidity may cancel.')
    for delay in args.latencies:
        repeat_f,miss=evaluate(rm,db,args.book_kind,args.qty,delay,args.quote_age,args.fee_multiplier,args.max_kalshi_price)
        print(f'\nDelay {delay:g}s · repeat: rejected={dict(miss)}')
        summary('Repeat BUY',repeat_f,len(rm),args.qty)
        if repeat_f:
            ci=boot_ci(repeat_f,args.qty)
            if ci:print(f'  Day-bootstrap 95% net/share [{ci[0]:+.4f}, {ci[1]:+.4f}], days={ci[2]}')
            for w in sorted({r['week'] for r in repeat_f}):
                summary(f'  {w}',[r for r in repeat_f if r['week']==w],len([r for r in rm if r['week']==w]),args.qty)
        first_f,miss_f=evaluate(first_matched,db,args.book_kind,args.qty,delay,args.quote_age,args.fee_multiplier,args.max_kalshi_price)
        summary('First on same cohort',first_f,len(first_matched),args.qty)
        if first_f and miss_f:print(f'  First rejected={dict(miss_f)}')
    print('\nLIMITATIONS: BBO reflects recorded public market conditions, not queue priority or assured fills. '
          'As-of timestamps are recorder timestamps for BBO; one-second snapshots are delayed one second; '
          'Polymarket signal receipt latency is unknown. Never treat missing-book markets as losses or zero-cost fills. '
          'No writes, orders, or GitHub changes.')


if __name__=='__main__':
    try:main()
    except Exception as e:sys.exit('ABORT: '+str(e))
