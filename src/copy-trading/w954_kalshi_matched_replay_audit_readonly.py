#!/usr/bin/env python3
"""Read-only matched-cohort follow-up to wallet 954B Kalshi trade-print backtest.

Place alongside w954_kalshi_crossvenue_backtest_readonly.py in src/copy-trading/.
Run from repository root. Uses existing Codespace wallet cache. Makes Kalshi
public API GET calls (unless --source files). Never writes files or orders.

This is NOT an execution backtest. It uses buyer-aggressor trade prints at or
shortly AFTER simulated entry time, not guaranteed asks or executable depth.
Polymarket wallet timestamps are execution timestamps, not proven copier
receipt timestamps. Delay scenarios are sensitivities, not measured latency.
"""
from __future__ import annotations

import argparse
import math
import random
import runpy
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from types import SimpleNamespace

CORE_PATH = Path('src/copy-trading/w954_kalshi_crossvenue_backtest_readonly.py')


def key(r):
    return (int(r['start']), str(r['side']).upper())


def ex_stats(rows, qty, fee_multiplier, premium):
    if not rows:
        return None
    prices = [min(.9999, r['kalshi_price'] + premium) for r in rows]
    spent = sum(qty*p + fee_at(p, qty, fee_multiplier) for p in prices)
    paid = sum(qty * r['kalshi_win'] for r in rows)
    pnl = paid - spent
    return dict(n=len(rows), paid=spent, pnl=pnl, roi=pnl/spent*100 if spent>0 else float('nan'),
                unit_pnl=pnl/(qty*len(rows)), med=median(prices),
                win=sum(int(r['kalshi_win']) for r in rows)/len(rows)*100)


def fee_at(p, qty, mult):
    return math.ceil(100*mult*.07*qty*p*(1-p)-1e-12)/100


def show(label, rows, qty, mult, premium=0):
    x=ex_stats(rows,qty,mult,premium)
    if x is None:
        print(f'  {label:<30} N=0'); return
    print(f'  {label:<30} N={x["n"]:3} win={x["win"]:5.1f}% '
          f'med_price={x["med"]:.3f} netPnL=${x["pnl"]:+8.2f} '
          f'netROI={x["roi"]:+7.2f}% pnl/share={x["unit_pnl"]:+.4f}')


def boot_day_ci(rows, qty, multiplier, premium, iterations=4000, seed=954):
    if len(rows)<4: return None
    byday=defaultdict(list)
    for r in rows:
        d=datetime.fromtimestamp(r['start'],timezone.utc).date().isoformat()
        byday[d].append(r)
    days=sorted(byday)
    rng=random.Random(seed)
    trials=[]
    for _ in range(iterations):
        sampled=[r for _ in days for r in byday[rng.choice(days)]]
        v=ex_stats(sampled,qty,multiplier,premium)
        trials.append(v['unit_pnl'])
    trials.sort()
    return trials[int(.025*iterations)],trials[int(.975*iterations)],len(days)


def max_extra_premium(rows, qty, mult):
    if not rows: return None
    if ex_stats(rows,qty,mult,0)['pnl']<=0: return 0.
    lo,hi=0.,.40
    for _ in range(45):
        mid=(lo+hi)/2
        if ex_stats(rows,qty,mult,mid)['pnl']>0: lo=mid
        else: hi=mid
    return lo


def stats_trade_wait(rows, latency):
    if not rows:return ''
    delays=[r['price_ts']-(r['signal_ts']+latency) for r in rows]
    return f'median_next_print_wait={median(delays):.2f}s >2s={sum(d>2 for d in delays)}/{len(delays)}'


def boot_paired_delta(a,b,qty,mult,premium,iterations=4000):
    """Per-share matched difference: REPEAT net P&L minus FIRST net P&L.

    Both entries have identical resolution by definition, so positive delta
    means the second entry was cheaper inclusive of fees, not more predictive.
    """
    ka={key(r):r for r in a}; kb={key(r):r for r in b}
    common=sorted(ka.keys() & kb.keys())
    if not common: return None
    deltas=[]
    byday=defaultdict(list)
    for k in common:
        ra,rb=ka[k],kb[k]
        p0=min(.9999,ra['kalshi_price']+premium)
        p1=min(.9999,rb['kalshi_price']+premium)
        first=qty*(ra['kalshi_win']-p0)-fee_at(p0,qty,mult)
        repeat=qty*(rb['kalshi_win']-p1)-fee_at(p1,qty,mult)
        delta=(repeat-first)/qty
        deltas.append(delta)
        d=datetime.fromtimestamp(rb['start'],timezone.utc).date().isoformat()
        byday[d].append(delta)
    days=sorted(byday)
    rng=random.Random(2026)
    trials=[]
    for _ in range(iterations):
        sampled=[x for _ in days for x in byday[rng.choice(days)]]
        trials.append(sum(sampled)/len(sampled))
    trials.sort()
    return len(common), sum(deltas)/len(deltas), (trials[int(.025*iterations)],trials[int(.975*iterations)])


def self_test():
    r1=[dict(start=1792332000,side='UP',kalshi_price=.30,kalshi_win=1,price_ts=1792332006,signal_ts=1792332005),
        dict(start=1792332900,side='DOWN',kalshi_price=.25,kalshi_win=0,price_ts=1792332906,signal_ts=1792332905)]
    r2=[dict(start=r['start'],side=r['side'],kalshi_price=r['kalshi_price']+.05,
             kalshi_win=r['kalshi_win'],price_ts=r['price_ts']+8,signal_ts=r['signal_ts']+8) for r in r1]
    a=ex_stats(r1,10,1,0); b=ex_stats(r1,10,1,.05)
    assert a['n']==2 and b['pnl']<a['pnl']
    assert fee_at(.4,10,1)==.17
    n,avg,ci=boot_paired_delta(r1,r2,10,1,0,200)
    assert n==2 and -.06<avg<-.04,(n,avg,ci)
    z=boot_day_ci(r1+r1,10,1,.05,200)
    assert z is not None and len(z)==3
    assert 0 <= max_extra_premium(r1,10,1) <= .4
    print('PASS: matched cohort accounting, direction-independent payoff, fees, bootstraps, price stress')


def main():
    p=argparse.ArgumentParser(description='Wallet 954B -> Kalshi matched first-vs-repeat historical proxy audit')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--wallet',type=Path,default=None)
    p.add_argument('--source',choices=['api','files'],default='api')
    p.add_argument('--markets-file',type=Path)
    p.add_argument('--trades-file',type=Path)
    p.add_argument('--bbo-file',type=Path)
    p.add_argument('--qty',type=int,default=10)
    p.add_argument('--fee-multiplier',type=float,default=1.)
    p.add_argument('--first-price-max',type=float,default=.40)
    p.add_argument('--min-repeat-age',type=int,default=4)
    p.add_argument('--max-repeat-age',type=int,default=30)
    p.add_argument('--latencies',type=int,nargs='+',default=[0,5,10,20,30,60,120])
    p.add_argument('--max-wait-s',type=int,default=5)
    p.add_argument('--max-quote-age-s',type=float,default=2)
    p.add_argument('--max-kalshi-price',type=float,default=None)
    args=p.parse_args()
    if args.self_test: self_test();return
    if args.qty<=0 or args.fee_multiplier<0 or min(args.latencies)<0 or args.max_wait_s<0:
        raise SystemExit('Invalid quantity, latency, fee or wait args')
    if args.source=='files' and not args.markets_file: raise SystemExit('Files mode requires --markets-file')
    if args.source=='files' and not (args.trades_file or args.bbo_file): raise SystemExit('Files mode requires --trades-file or --bbo-file')
    if not CORE_PATH.is_file(): raise SystemExit(f'Missing previous audit file: {CORE_PATH}')
    core=runpy.run_path(str(CORE_PATH))
    defaults=dict(wallet=args.wallet,min_repeat_age=args.min_repeat_age,
                  max_repeat_age=args.max_repeat_age,upper_first_price=args.first_price_max)
    fs=core['extract_signals'](SimpleNamespace(strategy='first',**defaults))
    rs=core['extract_signals'](SimpleNamespace(strategy='repeat',**defaults))
    if len(fs)!=len({key(r) for r in fs}) or len(rs)!=len({key(r) for r in rs}):
        raise RuntimeError('Duplicate wallet market/side entries; abort matched comparison')
    print('Reading Kalshi settled markets once (GET only)...',flush=True)
    mk=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](fs)
    mf,missing_f=core['match_markets'](fs,mk)
    mr,missing_r=core['match_markets'](rs,mk)
    fm={key(r):r for r in mf}; rm={key(r):r for r in mr}
    keys=sorted(fm.keys() & rm.keys())
    paired_first=[fm[k] for k in keys]
    paired_repeat=[rm[k] for k in keys]
    first_only=[r for k,r in fm.items() if k not in rm]
    print(f'Settled Kalshi matches: first={len(mf)}/{len(fs)} repeat={len(mr)}/{len(rs)} '
          f'paired_same_market={len(keys)} first_without_repeat={len(first_only)}')
    print(f'Missing: first={dict(missing_f)} repeat={dict(missing_r)}')
    print(f'Kalshi settlement disagreements on repeat subset={sum(r["disagree"] for r in paired_repeat)}/{len(paired_repeat)}')
    if len(keys)<15: raise RuntimeError('Insufficient exact paired market coverage')
    # Build price database once for all signals and all latency scenarios.
    # The previous module fetches public trade prints within signal->max(latencies)+wait.
    all_signal_rows=mf+mr
    bbo_db=core['make_bbo_db'](args)
    tr_db=None if bbo_db is not None else core['make_trade_db'](all_signal_rows,args)
    mode='historical BBO as-of asks with top-of-book size' if bbo_db is not None else 'post-signal buyer-aggressor trade-print PROXY'
    print(f'Entry-price source: {mode}; fees=general taker formula, per 10 share order')
    print('NO price after receipt time is guaranteed executable. Polymarket wallet publication lag unknown.')
    for delay in args.latencies:
        pf,err_pf=core['lookup'](paired_first,delay,tr_db,bbo_db,args)
        pr,err_pr=core['lookup'](paired_repeat,delay,tr_db,bbo_db,args)
        control,err_ctl=core['lookup'](first_only,delay,tr_db,bbo_db,args)
        mapf={key(r):r for r in pf}; mapr={key(r):r for r in pr}
        common=mapf.keys() & mapr.keys()
        pf_same=[mapf[k] for k in sorted(common)]
        pr_same=[mapr[k] for k in sorted(common)]
        print(f'\nLATENCY {delay}s | paired at both prices={len(common)}/{len(keys)} '
              f'firstonly={len(control)}/{len(first_only)} '
              f'price_fail_repeat={dict(err_pr)} firstonly_fail={dict(err_ctl)}')
        show('SAME MARKETS: first buy + fee',pf_same,args.qty,args.fee_multiplier,0)
        show('SAME MARKETS: repeat buy + fee',pr_same,args.qty,args.fee_multiplier,0)
        show('SAME MARKETS: repeat +3c + fee',pr_same,args.qty,args.fee_multiplier,.03)
        show('SAME MARKETS: repeat +5c + fee',pr_same,args.qty,args.fee_multiplier,.05)
        show('CONTROL: no-repeat first + fee',control,args.qty,args.fee_multiplier,0)
        show('CONTROL: no-repeat first +5c',control,args.qty,args.fee_multiplier,.05)
        if delay in (5,10,20,60,120) and pr_same:
            ci=boot_day_ci(pr_same,args.qty,args.fee_multiplier,.05)
            print(f'  REPEAT +5c daily-bootstrap 95% pnl/share [{ci[0]:+.4f},{ci[1]:+.4f}] (days={ci[2]})')
            d=boot_paired_delta(pf_same,pr_same,args.qty,args.fee_multiplier,0)
            print(f'  REPEAT - FIRST on same {d[0]} markets: mean net difference/share={d[1]:+.4f} '
                  f'daily-bootstrap 95% [{d[2][0]:+.4f},{d[2][1]:+.4f}]')
            cushion=max_extra_premium(pr_same,args.qty,args.fee_multiplier)
            print(f'  Approx EXTRA price premium to break-even after fee: {cushion*100:.2f}c '
                  f'(proxy only; not a slippage estimate)')
            if bbo_db is None:
                print('  REPEAT next-print latency:',stats_trade_wait(pr_same,delay))
        if delay in (5,20):
            for w in sorted({r['week'] for r in pr_same}):
                sub=[r for r in pr_same if r['week']==w]
                show('    '+w+' repeat +5c',sub,args.qty,args.fee_multiplier,.05)
    print('\nWARNINGS: This is NOT an executable ask-price backtest; 100% trade-print coverage')
    print('does not prove a copier could buy at those prices or receive the wallet signal in time.')
    print('Same-market first comparison is retrospective: one cannot know a repeat will occur at first entry.')
    print('No network actions except public historical GETs; no writes, no trades, no GitHub changes.')


if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError) as e: sys.exit('ABORT: '+str(e))
