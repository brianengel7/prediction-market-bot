#!/usr/bin/env python3
"""954B BTC 15m: does removing 4-30s confirmation improve same-side copying?

READ ONLY. Uses existing 954B/Kalshi research modules. No POST, orders,
SQLite writes, or repo changes. Public HTTPS GETs in normal modes; with
--offline uses existing market/print files.

Strategies (identical 20<=FIRST wallet BUY<40c first-entry cohort):
  FIRST_ONLY: initial 10 Kalshi contracts only.
  CURRENT_CONFIRM_THEN_ALL: initial 10, require a genuine qualifying same-side
      wallet repeat 4-30s after first before adding; then copy ALL later same-side
      buys until first opposite buy/SELL/market end, capped by planned position.
  NO_CONFIRM_ALL: initial 10, then copy ALL subsequent same-side wallet BUYs
      immediately even if the first repeat is at +0, +93 or +400 seconds;
      stop on opposite buy/SELL/market end, capped by planned position.

Both full strategies use ROUND_HALF_UP(base_qty * cumulative_wallet_shares /
first_wallet_shares) and the same historical Kalshi buyer-aggressor trade-print
simulation. These prints are NOT historical executable asks or proof of fills.
Wallet timestamps are not live-copy first observation times. The +2/+5 cent
stress is a post-fill price penalty, not a new execution liquidity backtest.
Historical order/day spending limits and real Kalshi ask drift are NOT modeled.

Install beside current research companions in src/copy-trading and run:
  python3 -B src/copy-trading/w954_remove_confirmation_historical_compare_readonly.py --self-test
  python3 -u src/copy-trading/w954_remove_confirmation_historical_compare_readonly.py --mode archive
  python3 -u src/copy-trading/w954_remove_confirmation_historical_compare_readonly.py --mode recent --start 2026-11-06 --end 2026-11-11
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import runpy
import math
import sys

HERE=Path(__file__).resolve().parent
OOS=HERE/'w954_all_subsequent_caps_oos_readonly.py'


def build_unrestricted(signals, wallet_index, first_events, cap, base_qty, round_target):
    """Only change vs existing full builder: removes early-confirm prerequisite.

    Place no orders in this routine; produce hypothetical chronological events.
    Every wallet source row is acted on at most once in stream order; target
    uses cumulative shares (not separately rounded wallet fill quantities).
    """
    starts={s['start']:s for s in signals}
    initial={e['start']:e for e in first_events}
    if len(initial)!=len(first_events):
        raise RuntimeError('Multiple initial positions in one market')
    result=list(first_events)
    stats=Counter()
    new_market_starts=set()
    late_starts=set()
    for start, e in sorted(initial.items()):
        s=starts[start]
        first_candidates=[r for r in wallet_index.get(start, ())
                          if r['action']=='BUY' and r['side']==s['side']
                          and abs(r['ts']-s['ts'])<=3
                          and abs(r['price']-s['price'])<=.015
                          and r['size']>0 and math.isfinite(r['size'])]
        if not first_candidates:
            raise RuntimeError(f'Cannot find wallet anchor in market {start}')
        first=min(first_candidates,key=lambda r:(abs(r['ts']-s['ts']),
                                                   abs(r['price']-s['price']),r['line']))
        anchor=first['size']
        cumulative=anchor
        planned=base_qty
        first_clock=(first['ts'],first['line'])
        first_followup=None
        early_qualified=False
        for r in sorted(wallet_index.get(start,()),key=lambda row:(row['ts'],row['line'])):
            clock=(r['ts'],r['line'])
            if clock<=first_clock:continue
            if r['ts']>=start+900:break
            if r['action']=='SELL':
                stats['stopped_on_wallet_sell']+=1
                break
            if r['action']=='BUY' and r['side']!=s['side']:
                stats['stopped_on_opposite_buy']+=1
                break
            if r['action']!='BUY' or r['side']!=s['side']:
                continue
            if first_followup is None:
                first_followup=r['ts']-first['ts']
                early_qualified=(4<=first_followup<=30)
                if first_followup>30:
                    stats['first_followup_after_30s']+=1
                if first_followup<4:
                    stats['first_followup_under_4s']+=1
            stats['subsequent_same_side_wallet_fills']+=1
            cumulative+=r['size']
            target=round_target(base_qty,cumulative,anchor)
            if target>cap:
                stats['stopped_at_market_quantity_cap']+=1
                break
            qty=target-planned
            if qty<0:raise RuntimeError('Wallet add reduced cumulative copy target')
            if qty==0:
                stats['rounded_to_zero_pending']+=1
                continue
            new_market_starts.add(start)
            if not early_qualified:
                late_starts.add(start)
            new=dict(e)
            new.update(kind='add',ts=r['ts'],line=r['line'],size=qty,
                       wallet_fill_price=r['price'],wallet_anchor_shares=anchor,
                       wallet_cumulative_shares=cumulative,
                       wallet_fills_aggregated=1)
            result.append(new)
            planned=target
            stats['add_orders']+=1
            stats['additional_contracts']+=qty
        if first_followup is None:stats['no_same_side_followup_before_stop']+=1
    stats['markets_with_adds']=len(new_market_starts)
    stats['markets_with_adds_without_early_confirm']=len(late_starts)
    return sorted(result,key=lambda e:(e['ts'],e['start'],e['line'])),stats


def self_test():
    def target(base,cumulative,anchor):
        return int(math.floor(base*cumulative/anchor+.5+1e-10))
    def row(start,ts,line,side='UP',qty=50,price=.31,action='BUY'):
        return dict(start=start,ts=ts,line=line,side=side,size=float(qty),price=price,action=action)
    cases={
      900:[row(900,910,1,qty=100),row(900,1003,2,qty=50),row(900,1004,3,qty=20)],
      1800:[row(1800,1810,1,qty=100),row(1800,1816,2,qty=40),row(1800,1817,3,qty=10)],
      2700:[row(2700,2710,1,qty=100),row(2700,2712,2,qty=20),row(2700,2713,3,qty=10),
            row(2700,2720,4,side='DOWN'),row(2700,2730,5,qty=70)],
      3600:[row(3600,3610,1,qty=100),row(3600,3700,2,qty=300)],
      4500:[row(4500,4510,1,qty=100)]
    }
    sig=[dict(start=k,ts=v[0]['ts'],price=v[0]['price'],side='UP') for k,v in cases.items()]
    first=[dict(start=k,ts=v[0]['ts'],line=1,kind='first',side='UP',action='BUY',
                price=.31,close=k+900,size=10,signal_id=k) for k,v in cases.items()]
    ev,stats=build_unrestricted(sig,cases,first,250,10,target)
    one=lambda s:[(e['ts'],e['size']) for e in ev if e['start']==s]
    assert one(900)==[(910,10),(1003,5),(1004,2)],one(900)
    assert one(1800)==[(1810,10),(1816,4),(1817,1)],one(1800)
    assert one(2700)==[(2710,10),(2712,2),(2713,1)],one(2700)
    assert one(3600)==[(3610,10),(3700,30)],one(3600)
    assert one(4500)==[(4510,10)],one(4500)
    assert stats['markets_with_adds_without_early_confirm']==3,stats
    # A quantity cap stops (not clips) an excessive add while preserving starter.
    small,bad=build_unrestricted(sig,cases,first,25,10,target)
    assert [(e['ts'],e['size']) for e in small if e['start']==3600]==[(3610,10)]
    # Identical records with same source sequence are not a new event.
    assert len({(e['start'],e['ts'],e['line']) for e in ev})==len(ev)
    print('PASS: late +93s confirmation, early +6s confirmation, same-second fills, '
          'opposite stop, cumulative rounding, no-followup, cap stop, matched first entries.')


def load_source(args, oos, core, af, a, follow):
    if args.mode=='archive':
        signals=core['extract_signals'](SimpleNamespace(
            wallet=args.wallet,strategy='first',upper_first_price=.40,
            min_repeat_age=4,max_repeat_age=30))
        wallet, meta=af['prepare_wallet'](args.wallet)
        print(f'ARCHIVE wallet source rows={meta["raw"]} 20-40c first starts={len(signals)}',flush=True)
        return signals,wallet
    begin,end=oos['date_epoch'](args.start),oos['date_epoch'](args.end)
    if not begin<end or begin<oos['date_epoch']('2026-11-06'):
        raise ValueError('RECENT range must be [start,end), start on/after Nov 6, 2026')
    print(f'RECENT source [UTC {args.start}, {args.end}) direct Polymarket GET',flush=True)
    data=oos['polymarket_trades'](begin,end,args.max_pages)
    grouped,errors,duplicates=oos['decode_wallet'](
        data,a,begin,end,int(datetime.now(timezone.utc).timestamp()))
    signals,_,excluded=oos['derive_first'](grouped,follow)
    print(f'API records={len(data)} BTC markets={len(grouped)} qualifying={len(signals)} '
          f'native-id duplicates={duplicates} decode={dict(errors)} join={dict(excluded)}',flush=True)
    return signals,[r for market in grouped.values() for r in market]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--mode',choices=['archive','recent'],default='archive')
    ap.add_argument('--wallet',type=Path,default=None)
    ap.add_argument('--start',default='2026-11-06')
    ap.add_argument('--end',default='2026-11-11')
    ap.add_argument('--max-pages',type=int,default=100)
    ap.add_argument('--base-qty',type=int,default=10)
    ap.add_argument('--cap',type=int,default=250)
    ap.add_argument('--latencies',nargs='+',type=int,default=[5,10,20])
    ap.add_argument('--max-wait-s',type=float,default=5.)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--markets-file',type=Path)
    ap.add_argument('--trades-file',type=Path)
    ap.add_argument('--offline',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    if args.base_qty<1 or args.cap<args.base_qty or args.max_wait_s<0 or args.bootstrap<0:
        ap.error('Invalid base, cap, wait or bootstrap')
    if not args.latencies or min(args.latencies)<0 or max(args.latencies)>180:
        ap.error('Invalid latency list')
    if args.offline and not(args.markets_file and args.trades_file):
        ap.error('--offline requires --markets-file and --trades-file')
    if not OOS.is_file():
        raise RuntimeError('Missing companion '+str(OOS))
    oos=runpy.run_path(str(OOS))
    m,p,core,af,a,follow=oos['modules']()
    signals,wallet=load_source(args,oos,core,af,a,follow)
    if not signals:
        print('No qualifying entries in source date range');return
    # This historical source lacks a guaranteed per-execution ID in all rows.
    # Report economic fingerprint collisions, but do not silently delete them:
    # multiple economically identical fills may genuinely have occurred.
    fingerprint=Counter((r['start'],r['ts'],r['action'],r['side'],
                         round(r['price'],8),round(r['size'],8)) for r in wallet)
    overlap=sum(v-1 for v in fingerprint.values() if v>1)
    if overlap:
        print(f'SOURCE IDENTITY WARNING: {overlap} extra rows share a same-second '
              'side/price/size fingerprint. They may be separate fills or API '
              'duplicates; replay preserves all of them.',flush=True)
    idx=p['index_wallet'](wallet)
    cfg=oos['params'](args,args.cap)
    current,curbad,curkinds,curobs=m['build_all'](
        signals,idx,cfg,p,stop_on_opposite=True)
    first=[dict(e) for e in current if e['kind']=='first']
    unrestricted,unresobs=build_unrestricted(
        signals,idx,first,args.cap,args.base_qty,p['rounded_target'])
    if len(first)!=len({e['start'] for e in first}):
        raise RuntimeError('Duplicate first entries; cannot pair variants')
    variants={
      'FIRST_ONLY':first,
      'CURRENT_CONFIRM_THEN_ALL':current,
      'NO_CONFIRM_ALL':unrestricted,
    }
    control=sorted((x['start'],x['ts'],x['line'],x['size']) for x in first)
    for name,events in variants.items():
        other=sorted((x['start'],x['ts'],x['line'],x['size']) for x in events
                     if x['kind']=='first')
        if other!=control:raise RuntimeError(f'First-entry mismatch: {name}')
    print('\n954B CONFIRMATION RULE REMOVAL: PAIRED HISTORICAL BACKTEST',flush=True)
    print(f'Period={args.mode}; first entry first wallet BUY 20<=p<40c; '
          f'base={args.base_qty}; position cap={args.cap}; simulated delays={args.latencies}',flush=True)
    print('Old rule: 4-30s first same-side repeat required to begin additions.',flush=True)
    print('New rule: add on every later same-side BUY, regardless of gap.',flush=True)
    print(f'CURRENT orders={len(current)} kinds={dict(curkinds)} skips={dict(curbad)} '
          f'observations={dict(curobs)}',flush=True)
    print(f'NO_CONFIRM orders={len(unrestricted)} observations={dict(unresobs)}',flush=True)
    stub=[{'start':start} for start in sorted({e['start'] for e in first})]
    markets=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](stub)
    byclose=defaultdict(list)
    for r in markets:
        close=core['market_close'](r)
        if close is not None and str(r.get('ticker','')).startswith('KXBTC15M-'):
            byclose[round(close)].append(r)
    paired={}
    tickers=None
    for name,ev in variants.items():
        mapped,excluded=p['attach_market'](ev,byclose)
        paired[name]=mapped
        s={x['ticker'] for x in mapped if x['kind']=='first'}
        if tickers is None:tickers=s
        elif s!=tickers:raise RuntimeError('Different mapped first markets')
        print(f'  {name:<28} matched={len(mapped)}/{len(ev)} '
              f'markets={len(s)} excludes={dict(excluded)}',flush=True)
    if not tickers:
        print('No settled Kalshi markets matched; cannot calculate P&L');return
    union=[e for events in paired.values() for e in events]
    print('Loading Kalshi historical prints (public GET, no orders)...',flush=True)
    pp=af['prep_prints'](af['load_trade_prints'](union,cfg,core))
    sim=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
        max_wait_s=args.max_wait_s,max_quote_age_s=2.,fee_multiplier=1.)
    for latency in args.latencies:
        print(f'\n=== {latency}s latency | historical prints NOT executable asks ===')
        print('STRATEGY                     MARKETS   FILLS       PNL      COST     ROI   +2c ROI  +5c ROI')
        print('-'*101)
        results={}
        for name,events in paired.items():
            executions,markets,failed,dep=p['simulate_dependent'](
                events,sim,latency,af,pp,None)
            results[name]=(executions,markets)
            cost=sum(x['buy_cost'] for x in markets)
            pnl=sum(x['pnl'] for x in markets)
            roi=100*pnl/cost if cost else float('nan')
            stress=p['run_price_stress'](executions,markets,[0,2,5],af['fee'],1.)
            print(f'{name:<28} {len(markets):>3}  {len(executions):>4}/{len(events):<4}'
                  f'  ${pnl:>+9.2f}  ${cost:>9.2f} {roi:>+7.2f}%  '
                  f'{stress[1][3]:>+7.2f}%  {stress[2][3]:>+7.2f}%',flush=True)
            if failed or dep:print(f'  execution rejects={dict(failed)} dependent skips={dep}',flush=True)
        base=results['CURRENT_CONFIRM_THEN_ALL'][1]
        alt=results['NO_CONFIRM_ALL'][1]
        delta,cost_inc,inc_roi=p['compare'](base,alt)
        ci=p['bootstrap_increment'](base,alt,args.bootstrap,954306)
        print(f'NO_CONFIRM vs CURRENT: ΔPnL=${delta:+.2f}, '
              f'Δcommitted cost=${cost_inc:+.2f}, incremental ROI='
              f'{inc_roi:+.2f}%' if inc_roi is not None else
              f'NO_CONFIRM vs CURRENT: ΔPnL=${delta:+.2f}, Δcost=${cost_inc:+.2f} (no extra cost)')
        if ci:print(f'Paired day-bootstrap 95% ΔPnL range ${ci[0]:+.2f} to ${ci[1]:+.2f}',flush=True)
        print('Weeks current:',m['calendar'](base),flush=True)
        print('Weeks no-confirm:',m['calendar'](alt),flush=True)
    print('\nCAUTION: No executable historical Kalshi ask/depth, no true wallet publication time, '
          'and no $20/order, $50/market, $100/day live risk-budget or 5c max drift simulation. '
          'Not a live profitability estimate. Existing training + previously reviewed Nov6-10 '
          'holdout must not be called untouched out-of-sample.',flush=True)

if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,OSError,AssertionError) as exc:
        sys.exit('ABORT: '+str(exc))
