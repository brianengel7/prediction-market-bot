#!/usr/bin/env python3
"""Wallet 954B -> Kalshi 1-minute BID/ASK candle stress audit. READ ONLY.

Place in src/copy-trading/ next to w954_kalshi_crossvenue_backtest_readonly.py.
Uses locally cached wallet 954B fills; downloads only public, historical
Kalshi markets/candlesticks with HTTP GET; does not save files or place orders.

IMPORTANT: A minute's ask HIGH can reflect quotes before OR after the decision.
It is a hindsight, conservative price envelope, NOT a proven achievable ask.
The most recent COMPLETED minute's ask close is available at signal time,
but too stale to claim execution. Neither has ask depth or wallet publication lag.
"""
from __future__ import annotations

import argparse
import bisect
import math
import random
import runpy
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from urllib.error import HTTPError

ROOT = Path('src/copy-trading')
CORE_PATH = ROOT / 'w954_kalshi_crossvenue_backtest_readonly.py'


def val(block, field):
    if not isinstance(block, dict):
        return None
    for key in (field+'_dollars', field):
        x = block.get(key)
        try:
            if x is not None and x != '':
                z = float(x)
                return z if math.isfinite(z) and 0 < z < 1 else None
        except (TypeError, ValueError):
            pass
    return None


def quote(row, side, stat):
    """For NO ask, use 1 - YES bid, with min/max reversed."""
    if side == 'yes':
        x = val(row.get('yes_ask'), stat)
    else:
        counterpart = {'high':'low','low':'high','open':'open','close':'close'}[stat]
        bid = val(row.get('yes_bid'), counterpart)
        x = 1. - bid if bid is not None else None
    return x if x is not None and 0 < x < 1 else None


def find_candles(entry, candles):
    """At decision ts, identify most recent completed minute and containing minute.

    Assumes candle end_period_ts marks the end of a 60s interval. The
    containing candle is used only for hindsight envelope diagnostics.
    """
    if not candles:
        return None, None
    ts = float(entry)
    secs = [float(c['end_period_ts']) for c in candles]
    j = bisect.bisect_right(secs, ts)
    completed = candles[j-1] if j > 0 else None
    # Never use a completed close older than 120 seconds.
    if completed and ts-float(completed['end_period_ts']) > 120:
        completed = None
    containing = candles[j] if j < len(candles) else None
    if containing and not (float(containing['end_period_ts'])-60 <= ts < float(containing['end_period_ts'])):
        containing = None
    return completed, containing


def candidate_prices(s, candle_rows, delay):
    when = float(s['signal_ts']) + delay
    if when >= float(s['start'])+900:
        return None, 'too_late'
    prev, current = find_candles(when, candle_rows)
    result = {'when':when}
    if prev:
        p = quote(prev,s['kalshi_side'],'close')
        if p is not None: result['previous_close']=p
    if current:
        pmin=quote(current,s['kalshi_side'],'low')
        pmax=quote(current,s['kalshi_side'],'high')
        if pmin is not None and pmax is not None and pmin-1e-8 <= pmax:
            result.update(minute_low=pmin, minute_high=pmax,
                          end_period_ts=current['end_period_ts'])
    if 'minute_high' not in result:
        return result, 'no_containing_minute_quote'
    return result, None


def stats(rows, price_field, qty, fee_multiplier, stress=0.0):
    use=[r for r in rows if r.get(price_field) is not None]
    if not use: return None
    prices=[min(.9999,float(r[price_field])+stress) for r in use]
    paid=sum(qty*p + math.ceil(100*fee_multiplier*.07*qty*p*(1-p)-1e-12)/100 for p in prices)
    payout=sum(int(r['kalshi_win'])*qty for r in use)
    return {'n':len(use),'win':sum(int(r['kalshi_win']) for r in use),
            'median':median(prices),'pnl':payout-paid,
            'cost':paid,'roi':100*(payout-paid)/paid if paid else float('nan'),
            'unit':(payout-paid)/(qty*len(use))}


def ci_by_day(rows, price_field, qty, mult, iters=3000):
    byday=defaultdict(list)
    for r in rows:
        if price_field not in r: continue
        d=datetime.fromtimestamp(r['start'],timezone.utc).date().isoformat()
        byday[d].append(r)
    days=sorted(byday)
    if len(days)<3: return None
    rng=random.Random(954)
    estimates=[]
    for _ in range(iters):
        sample=[r for _ in days for r in byday[rng.choice(days)]]
        estimates.append(stats(sample,price_field,qty,mult)['unit'])
    estimates.sort()
    return estimates[int(.025*iters)],estimates[int(.975*iters)],len(days)


def print_summary(label, rows, field, qty, fee_multiplier):
    st=stats(rows,field,qty,fee_multiplier)
    if not st:
        print(f'  {label:<30} N=0');return
    print(f'  {label:<30} N={st["n"]:3} win={st["win"]/st["n"]*100:5.1f}% '
          f'med_price={st["median"]:.3f} netPnL=${st["pnl"]:+8.2f} '
          f'netROI={st["roi"]:+7.2f}% pnl/share={st["unit"]:+.4f}')


def candle_get(core, ticker, start, close):
    """Try archival route first, then current route only on 404."""
    params={'start_ts':int(start)-120,'end_ts':int(close)+60,'period_interval':1}
    historical=f'/historical/markets/{ticker}/candlesticks'
    live=f'/series/KXBTC15M/markets/{ticker}/candlesticks'
    try:
        payload=core['api_get'](historical,params)
    except HTTPError as exc:
        if exc.code != 404: raise
        payload=core['api_get'](live,params)
    rows=payload.get('candlesticks')
    if not isinstance(rows,list):
        raise RuntimeError('Kalshi candle result missing candlesticks array for '+ticker)
    clean=[];seen=set()
    for row in rows:
        try:
            t=int(row['end_period_ts'])
        except (ValueError, KeyError, TypeError): continue
        if t not in seen:
            seen.add(t); clean.append(row)
    return sorted(clean,key=lambda x:int(x['end_period_ts']))


def candles_from_file(core, path):
    db=defaultdict(list)
    for item in core['objrows'](path):
        ticker=item.get('ticker')
        if ticker and 'end_period_ts' in item: db[ticker].append(item)
        elif ticker and isinstance(item.get('candlesticks'),list):
            db[ticker].extend(item['candlesticks'])
    for ticker in db: db[ticker].sort(key=lambda x:float(x['end_period_ts']))
    return db


def self_test():
    candles=[{'end_period_ts':60,'yes_bid':{'close':'0.30','low':'0.25','high':'0.35'},
              'yes_ask':{'close':'0.40','low':'0.33','high':'0.46'}},
             {'end_period_ts':120,'yes_bid':{'close':'0.20','low':'0.15','high':'0.30'},
              'yes_ask':{'close':'0.50','low':'0.35','high':'0.65'}}]
    prev,cur=find_candles(87,candles)
    assert prev['end_period_ts']==60 and cur['end_period_ts']==120
    assert quote(cur,'yes','high')==.65
    assert abs(quote(cur,'no','high')-.85)<1e-9
    assert abs(quote(cur,'no','low')-.70)<1e-9
    row={'start':0,'signal_ts':87,'kalshi_side':'yes','kalshi_win':1}
    q,err=candidate_prices(row,candles,5)
    assert err is None and q['previous_close']==.4 and q['minute_high']==.65
    row2=dict(row,kalshi_side='no',kalshi_win=0)
    q2,err2=candidate_prices(row2,candles,5)
    assert err2 is None and abs(q2['minute_high']-.85)<1e-9
    batch=[dict(row,minute_high=.65),dict(row2,minute_high=.85)]
    assert stats(batch,'minute_high',10,1)['pnl']<0
    print('PASS: candlestick timestamp, historical schema, NO-side quote conversion, fee, outcome')


def main():
    a=argparse.ArgumentParser(description='954B -> Kalshi historical minute-ask envelope study (READ ONLY)')
    a.add_argument('--self-test',action='store_true')
    a.add_argument('--wallet',type=Path)
    a.add_argument('--markets-file',type=Path)
    a.add_argument('--candles-file',type=Path, help='Optional existing candlesticks JSONL/CSV/JSON; no downloads')
    a.add_argument('--min-repeat-age',type=int,default=4)
    a.add_argument('--max-repeat-age',type=int,default=30)
    a.add_argument('--first-price-max',type=float,default=.40)
    a.add_argument('--qty',type=int,default=10)
    a.add_argument('--fee-multiplier',type=float,default=1.0)
    a.add_argument('--latencies',type=int,nargs='+',default=[0,5,10,20,30])
    args=a.parse_args()
    if args.self_test:
        self_test();return
    if args.qty<1 or args.fee_multiplier<0 or not args.latencies or min(args.latencies)<0:
        raise SystemExit('Invalid quantity / fee / latencies')
    if not CORE_PATH.is_file(): raise RuntimeError(f'Missing shared loader: {CORE_PATH}')
    core=runpy.run_path(str(CORE_PATH))
    params=dict(wallet=args.wallet,min_repeat_age=args.min_repeat_age,
                max_repeat_age=args.max_repeat_age,upper_first_price=args.first_price_max)
    signals=core['extract_signals'](SimpleNamespace(strategy='repeat',**params))
    print('Reading Kalshi settled markets (public GET only unless --markets-file)...',flush=True)
    markets=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](signals)
    paired,missing=core['match_markets'](signals,markets)
    print(f'Exact markets={len(paired)}/{len(signals)} excluded={dict(missing)}')
    if not paired: raise RuntimeError('No Kalshi settled matches')
    grouped={r['ticker']:r for r in paired}
    if args.candles_file:
        db=candles_from_file(core,args.candles_file)
    else:
        db={}
        for i,(ticker,s) in enumerate(sorted(grouped.items())):
            db[ticker]=candle_get(core,ticker,s['start'],s['start']+900)
            if (i+1)%15==0: print(f'Read candlesticks {i+1}/{len(grouped)}',flush=True)
    print('Quote data: ONE-MINUTE yes_bid/yes_ask OHLC, not time-exact executable orderbook')
    print('Current-minute HIGH looks ahead within minute; preceding minute CLOSE is stale.')
    print('Note: both outcomes and filter were established by historical data; future wallet visibility unknown.')
    for lag in args.latencies:
        out=[]; errors=Counter(); prev_count=0; low_count=0
        for s in paired:
            q,err=candidate_prices(s,db.get(s['ticker'],[]),lag)
            if err: errors[err]+=1
            if q and 'previous_close' in q: prev_count+=1
            if q and 'minute_low' in q: low_count+=1
            if q:
                out.append({**s,**q})
        quote_rows=[r for r in out if 'minute_high' in r]
        print(f'\nLATENCY {lag}s minute-high coverage={len(quote_rows)}/{len(paired)} '
              f'prev-close coverage={prev_count} missing={dict(errors)}')
        print_summary('current minute ASK high',quote_rows,'minute_high',args.qty,args.fee_multiplier)
        print_summary('current minute ASK low (floor)',quote_rows,'minute_low',args.qty,args.fee_multiplier)
        print_summary('prior completed minute ASK close',out,'previous_close',args.qty,args.fee_multiplier)
        if quote_rows:
            widths=[r['minute_high']-r['minute_low'] for r in quote_rows]
            print(f'  Within-minute ask range: median={median(widths)*100:.2f}c, '
                  f'>=5c={sum(x>=.05 for x in widths)}/{len(widths)}')
            ci=ci_by_day(quote_rows,'minute_high',args.qty,args.fee_multiplier)
            if ci:
                print(f'  Minute-HIGH day bootstrap net PnL/share 95% '
                      f'[{ci[0]:+.4f},{ci[1]:+.4f}] over {ci[2]} distinct days')
            if lag in (5,20):
                for w in sorted({r['week'] for r in quote_rows}):
                    subset=[r for r in quote_rows if r['week']==w]
                    print_summary('  '+w+' minute ASK high',subset,'minute_high',args.qty,args.fee_multiplier)
    print('\nGUARDRAILS: The candle HIGH is not known at entry time, may occur before/after the signal,')
    print('and is NOT a guaranteed buyable price. One-minute OHLC has no instantaneous timestamps')
    print('or depth; a positive or negative result cannot establish executable P&L.')
    print('No local writes, order submissions, or repo changes; only Kalshi public historical GET requests.')


if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,HTTPError) as exc:
        sys.exit('ABORT (no partial results to trust): '+str(exc))
