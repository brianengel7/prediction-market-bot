#!/usr/bin/env python3
"""Wallet 954B: FIRST+CONFIRM, then copy ALL later same-side BUYs (READ ONLY).

Requires these EXISTING scripts in the same src/copy-trading/ directory:
  w954_repeat_proportional_scale_in_readonly.py
  w954_kalshi_crossvenue_backtest_readonly.py
  w954_kalshi_all_fills_backtest_readonly.py
  w954_post_entry_sequence_audit_readonly.py
  w954_binance_signal_audit_readonly.py
  w954_followup_copy_window_audit_readonly.py

The qualifying first-BUY cohort is unchanged (20c <= first price < 40c).
Buy 10 at the first BUY, add at the first same-side repeat 4-30s later,
then buy the delta in the *rounded cumulative proportional target* on EVERY
additional same-side wallet BUY. No extra 4-second spacing requirement, no
120-second window, and no five-add limit. Sub-share increments are carried
forward until they total at least one copied Kalshi contract. A quantity cap
is never silently clipped: when exceeded, further adds are stopped.

Variants:
  FIRST_ONLY                 copy first BUY in all qualifying markets
  FIRST+CONFIRM+0            first BUY + first qualifying repeat (old benchmark)
  FIRST+CONFIRM+5_OLD       previous 4s-bucket/120s-window, at most 5 adds
  ALL_STOP_ON_OPPOSITE       copy every later same-side BUY before first opposite BUY
  ALL_THROUGH_CLOSE         copy every later same-side BUY through market end;
                            ignore opposite-side buys, without copying them

Print data remains a TRADE-PRICE PROXY, not executable ask/depth, and wallet
signal publication delay is unknown. No file writes, trades, or POSTs;
public Kalshi GETs only, unless local market/trade files are supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys

HERE = Path(__file__).resolve().parent
PROP_PATH = HERE / 'w954_repeat_proportional_scale_in_readonly.py'


def libs():
    if not PROP_PATH.is_file():
        raise RuntimeError(f'Missing companion: {PROP_PATH}')
    p = runpy.run_path(str(PROP_PATH))
    core, af = p['refs']()
    return p, core, af


def build_all(sigs, wallet_index, cfg, p, *, stop_on_opposite):
    """Append causal individual wallet follow-on BUYs to identical old starter+repeat orders."""
    events, exclusions, kinds = p['build'](sigs, wallet_index, 'first', 0, cfg)
    exclusions = Counter(exclusions)
    kinds = Counter(kinds)
    by_market = defaultdict(list)
    for e in events:
        by_market[e['start']].append(e)
    added = []
    observations = Counter()
    for s in sigs:
        rows = by_market[s['start']]
        repeats = [r for r in rows if r['kind'] == 'repeat']
        if not repeats:
            # No qualifying confirmation, or it was rejected by the old cap.
            continue
        if len(repeats) != 1:
            raise RuntimeError('Multiple confirmation orders for a market')
        if not p['repeat_confirmed'](s, cfg.min_repeat_age, cfg.max_repeat_age):
            raise RuntimeError('Noncausal confirmation passed original builder')
        first = p['wallet_first_fill'](s, wallet_index)
        if first is None or not first['size'] > 0:
            raise RuntimeError('Missing wallet anchor in confirmed market')
        rep = s['first_same']
        # At confirmation we already copied the change in the wallet's
        # cumulative same-side position since the first fill.
        cumulative = p['wallet_at_repeat'](s, first, wallet_index)
        anchor = first['size']
        planned = sum(r['size'] for r in rows if r['kind'] in ('first', 'repeat'))
        target = p['rounded_target'](cfg.base_qty, cumulative, anchor)
        if planned != target:
            raise RuntimeError(f'Confirmation quantity mismatch in market {s["start"]}: '
                               f'planned={planned} wallet target={target}')
        market_fills = sorted(wallet_index.get(s['start'], ()),
                              key=lambda r: (r['ts'], r['line']))
        last_observation = (rep['ts'], rep['line'])
        for r in market_fills:
            clock = (r['ts'], r['line'])
            if clock <= last_observation:
                continue
            if r['ts'] >= s['start'] + 900:
                break
            if r['action'] == 'BUY' and r['side'] != s['side']:
                observations['opposite_buys_observed'] += 1
                if stop_on_opposite:
                    observations['stopped_after_opposite'] += 1
                    break
                continue
            if r['action'] == 'SELL':
                # A true sell would invalidate the buy-only inventory model.
                # None were seen in the audited 9,358-record wallet archive.
                observations['wallet_sells_observed'] += 1
                raise RuntimeError('Source contains SELL; inventory tracking must be updated')
            if r['action'] != 'BUY' or r['side'] != s['side']:
                continue
            observations['subsequent_same_side_wallet_fills'] += 1
            cumulative += r['size']
            target = p['rounded_target'](cfg.base_qty, cumulative, anchor)
            if target > cfg.max_market_qty:
                exclusions['market_qty_cap_all_followups'] += 1
                observations['stopped_after_cap'] += 1
                break
            qty = target - planned
            if qty < 0:
                raise RuntimeError('Copy target decreased after a BUY')
            if qty == 0:
                observations['pending_rounding_small_fill'] += 1
                continue
            e = dict(repeats[0])
            e.update(ts=r['ts'], line=r['line'], size=qty, kind='add',
                     wallet_fills_aggregated=1, wallet_anchor_shares=anchor,
                     wallet_cumulative_shares=cumulative)
            added.append(e)
            kinds['add'] += 1
            planned = target
    final = sorted(events + added, key=lambda x: (x['ts'], x['start'], x['line']))
    return final, exclusions, kinds, observations


def summary(markets):
    pnl = sum(r['pnl'] for r in markets)
    cost = sum(r['buy_cost'] for r in markets)
    return pnl, cost, 100*pnl/cost if cost else float('nan')


def calendar(markets):
    weeks = defaultdict(lambda: [0., 0., 0])
    for m in markets:
        w = datetime.fromtimestamp(m['start'], timezone.utc).strftime('%G-W%V')
        weeks[w][0] += m['pnl']
        weeks[w][1] += m['buy_cost']
        weeks[w][2] += 1
    return ', '.join(f'{w} {int(v[2])}m {100*v[0]/v[1]:+.2f}%'
                     for w, v in sorted(weeks.items()) if v[1])


def test():
    p, core, af = libs()
    p['self_test']()  # old benchmark regression
    cfg = SimpleNamespace(min_repeat_age=4, max_repeat_age=30,
                          min_add_gap=4, add_window=120,
                          max_market_qty=250, sizing='proportional',
                          base_qty=10, add_qty=10, scale=.1)
    first = {'start':900, 'ts':1000, 'side':'UP', 'price':.30,
             'line':1, 'action':'BUY', 'size':100.}
    rep = {'start':900, 'ts':1008, 'side':'UP', 'price':.32,
           'line':2, 'action':'BUY', 'size':50.}
    # Seven post-confirm BUYs: first two at same second, one after 120s,
    # one opposing buy, then more supporting trades (through-close variant).
    extra = [(1009,3,'UP',10.),(1009,4,'UP',15.),(1011,5,'UP',20.),
             (1015,6,'UP',25.),(1018,7,'UP',50.),(1200,8,'UP',20.),
             (1400,9,'UP',30.),(1450,10,'DOWN',50.),
             (1460,11,'UP',10.),(1500,12,'UP',15.)]
    wallet = [first,rep] + [dict(start=900,ts=t,line=l,side=s,
                                 action='BUY',size=q,price=.33)
                            for t,l,s,q in extra]
    signal = dict(start=900,ts=1000,price=.30,side='UP',
                  first_same=rep,first_opp=wallet[-3],
                  prior_opposite=False,
                  buys=[x for x in wallet if x['side']=='UP' and x['ts']>1003])
    idx = p['index_wallet'](wallet)
    stop, bad, counts, obs = build_all([signal],idx,cfg,p,stop_on_opposite=True)
    through, _, _, _ = build_all([signal],idx,cfg,p,stop_on_opposite=False)
    baseline = p['build']([signal],idx,'first',0,cfg)[0]
    assert [x['kind'] for x in baseline] == ['first','repeat']
    assert len([x for x in stop if x['kind']=='add']) == 7, stop
    assert len([x for x in through if x['kind']=='add']) == 9, through
    assert sum(e['size'] for e in stop) == p['rounded_target'](10,320.,100.), stop
    assert sum(e['size'] for e in through) == p['rounded_target'](10,345.,100.), through
    assert any(e['ts'] == 1200 for e in stop), 'Post-120s add omitted'
    assert len({e['line'] for e in stop if e['kind']=='add'}) == 7
    assert [e for e in stop if e['ts']==1009 and e['kind']=='add'], 'Same-second copies omitted'
    assert obs['stopped_after_opposite'] == 1
    # Proportional rounding: 100 initial to 10; +50 -> 15;
    # +10 -> 16; +15 -> 18 (round half up); other sizes accumulate.
    assert [(e['ts'],e['size']) for e in stop[:4]] == [
        (1000,10),(1008,5),(1009,1),(1009,2)], stop[:4]
    # Decrease cap and verify it prevents further orders rather than clipping.
    cfg.max_market_qty=20
    capped,bad,_,obs=build_all([signal],idx,cfg,p,stop_on_opposite=True)
    assert len([e for e in capped if e['kind']=='add']) < 7 and bad['market_qty_cap_all_followups']==1
    # No-confirmation markets must still enter at first BUY (no hindsight bias).
    no_repeat = dict(signal,start=1800,ts=1900,first_same=None,first_opp=None,
                     prior_opposite=False, buys=[])
    no_first = dict(first,start=1800,ts=1900)
    noidx=p['index_wallet'](wallet+[no_first])
    joined,_,_,_=build_all([signal,no_repeat],noidx,cfg,p,stop_on_opposite=True)
    assert len([e for e in joined if e['start']==1800])==1
    # Offline market settlement + print-consumption regression for extra fills.
    cfg.max_market_qty=250
    mapped=[{**e,'ticker':'KXBTC15M-TEST','kalshi_side':'yes','result':'yes'} for e in stop]
    pp=af['prep_prints']({'KXBTC15M-TEST':[
        {'t':float(e['ts']+5),'yes':.35,'taker':'yes','qty':200.,'id':str(i)}
        for i,e in enumerate(mapped)]})
    scfg=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
                         max_wait_s=0.,max_quote_age_s=2.,fee_multiplier=1.)
    ex,ms,failed,dep=p['simulate_dependent'](mapped,scfg,5,af,pp,None)
    assert len(ex)==len(mapped) and not failed and not dep, (ex,failed,dep)
    assert abs(ms[0]['payout']-sum(e['size'] for e in mapped))<1e-9
    print('PASS: unlimited fill-by-fill additions, post-120s trades, same-second trades, '
          'opposite policy, cumulative proportional sizing, market cap, no-hindsight, '
          'execution volume/settlement (offline)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--wallet',type=Path)
    ap.add_argument('--markets-file',type=Path)
    ap.add_argument('--trades-file',type=Path)
    ap.add_argument('--offline',action='store_true')
    ap.add_argument('--latencies',nargs='+',type=int,default=[5,10,20])
    ap.add_argument('--base-qty',type=int,default=10)
    ap.add_argument('--max-market-qty',type=int,default=250)
    ap.add_argument('--max-wait-s',type=float,default=5.)
    ap.add_argument('--penalties-cents',nargs='+',type=float,default=[0,1,2,5])
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--include-through-opposite',action='store_true',
                    help='Also copy original side after an opposing wallet BUY')
    args=ap.parse_args()
    if args.self_test:
        test();return
    if (args.base_qty < 1 or args.max_market_qty < args.base_qty or
        not args.latencies or min(args.latencies)<0 or max(args.latencies)>180 or
        args.max_wait_s < 0 or args.bootstrap < 0 or
        not args.penalties_cents or any(x<0 for x in args.penalties_cents)):
        raise ValueError('Invalid sizing, time, bootstrap or penalty parameters')
    if args.offline and not (args.markets_file and args.trades_file):
        raise ValueError('Offline mode needs --markets-file and --trades-file')
    p, core, af = libs()
    cfg=SimpleNamespace(wallet=args.wallet, upper_first_price=.40,
        min_repeat_age=4,max_repeat_age=30,min_add_gap=4,add_window=120,
        max_market_qty=args.max_market_qty,sizing='proportional',
        base_qty=args.base_qty,add_qty=10,scale=.1,
        latencies=args.latencies,max_wait_s=args.max_wait_s,
        max_quote_age_s=2.,fee_multiplier=1.,merge_gap_s=15,
        trades_file=args.trades_file,offline=args.offline)
    signals=core['extract_signals'](SimpleNamespace(
        wallet=args.wallet, strategy='first',upper_first_price=.40,
        min_repeat_age=4,max_repeat_age=30))
    wallet, meta = af['prepare_wallet'](args.wallet)
    by_start=p['index_wallet'](wallet)
    baseline, baseline_bad, _=p['build'](signals,by_start,'first',0,cfg)
    old5, old5_bad, _=p['build'](signals,by_start,'first',5,cfg)
    full, full_bad, full_kind, full_obs=build_all(signals,by_start,cfg,p,stop_on_opposite=True)
    variants={
        'FIRST_ONLY':[e for e in baseline if e['kind']=='first'],
        'FIRST+CONFIRM+0':baseline,
        'FIRST+CONFIRM+5_OLD':old5,
        'ALL_STOP_ON_OPPOSITE':full,
    }
    if args.include_through_opposite:
        extra,extra_bad,extra_kind,extra_obs=build_all(
            signals,by_start,cfg,p,stop_on_opposite=False)
        variants['ALL_THROUGH_CLOSE']=extra
    cf=lambda rows:sorted((e['start'],e['ts'],e['line'],e['size'],e['side'])
                          for e in rows if e['kind']=='first')
    check=cf(baseline)
    for name,rows in variants.items():
        if cf(rows) != check:
            raise RuntimeError(f'First orders are not identical in {name}')
    print('\n954B — FIRST + CONFIRM + EVERY SUBSEQUENT BUY (READ ONLY)',flush=True)
    print('='*108)
    print(f'Wallet raw={meta["raw"]:,}; qualifying first={len(signals)}; '
          f'base={args.base_qty}, cap={args.max_market_qty}, '
          f'latencies={args.latencies}',flush=True)
    print('EVERY means no count/time/4-second spacing limit. Cumulative quantity targets; '
          'each recorded same-side BUY may produce an add, except rounded zero/cap.',flush=True)
    print(f'Original baseline skips={dict(baseline_bad)}; original +5 skips={dict(old5_bad)}')
    print(f'ALL until opposing BUY: orders={len(full)}, kinds={dict(full_kind)}, '
          f'skips={dict(full_bad)}, observations={dict(full_obs)}',flush=True)
    if args.include_through_opposite:
        print(f'ALL through close: orders={len(extra)}, kinds={dict(extra_kind)}, '
              f'skips={dict(extra_bad)}, observations={dict(extra_obs)}',flush=True)
    mock=[{'start':s} for s in sorted({e['start'] for e in baseline})]
    markets=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](mock)
    lookup=defaultdict(list)
    for m in markets:
        close=core['market_close'](m)
        if close is not None and str(m.get('ticker','')).startswith('KXBTC15M-'):
            lookup[round(close)].append(m)
    paired={}
    for name,ev in variants.items():
        mapped,skipped=p['attach_market'](ev,lookup)
        paired[name]=mapped
        print(f'  {name:<26} matched={len(mapped):>4}/{len(ev):<4} '
              f'markets={len({e["ticker"] for e in mapped}):3} '
              f'kinds={dict(Counter(e["kind"] for e in mapped))} '
              f'no-market={dict(skipped)}',flush=True)
    base_ticks={e['ticker'] for e in paired['FIRST_ONLY']}
    if not base_ticks:
        raise RuntimeError('No matching settled Kalshi markets')
    for name,rows in paired.items():
        if {e['ticker'] for e in rows} != base_ticks:
            raise RuntimeError(f'Market coverage differs from baseline for {name}')
    # Download once for every planned order across all variants.
    print('Loading Kalshi buyer-aggressor trade prints once for every scenario (GET only)...',flush=True)
    union=[e for ev in paired.values() for e in ev]
    pp=af['prep_prints'](af['load_trade_prints'](union,cfg,core))
    sim=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
        max_wait_s=args.max_wait_s,max_quote_age_s=2.,fee_multiplier=1.)
    for latency in args.latencies:
        print(f'\n=== SIMULATED LATENCY {latency}s — PRINT PRICE PROXY ONLY ===')
        results={}
        for name,ev in paired.items():
            ex,rows,fail,dep=p['simulate_dependent'](ev,sim,latency,af,pp,None)
            pnl,cost,roi=summary(rows)
            stress=p['run_price_stress'](ex,rows,args.penalties_cents,af['fee'],1.)
            top=sorted(rows,key=lambda x:x['pnl'],reverse=True)
            top5=summary(top[5:])[2] if len(top)>5 else float('nan')
            mx=max((m['buy_cost'] for m in rows),default=0.)
            print(f'{name:<26} N={len(rows):3} fills={len(ex):>4}/{len(ev):<4} '
                  f'add-fills={sum(r["kind"]=="add" for r in ex):3} '
                  f'PnL=${pnl:+9.2f} cost=${cost:>9.2f} ROI={roi:+7.2f}% '
                  f'top5-excluded={top5:+7.2f}% max-market-cost=${mx:>7.2f}')
            print('    +c penalty ROI: '+', '.join(f'+{c:g}c {r:+.2f}%'
                   for c,_,_,r in stress))
            if fail or dep:
                print(f'    print-proxy rejects={dict(fail)}; dependent skips={dep}')
            print(f'    weekly {calendar(rows)}')
            results[name]=(ex,rows)
        print('PAIRED INCREMENTAL BENEFIT vs FIRST+CONFIRM+0')
        base=results['FIRST+CONFIRM+0'][1]
        for name,(ex,rows) in results.items():
            if name in ('FIRST_ONLY','FIRST+CONFIRM+0'):
                continue
            dp,dc,incr=p['compare'](base,rows)
            ci=p['bootstrap_increment'](base,rows,args.bootstrap,11235)
            cistr=(f' day-bootstrap 95% PnL interval [${ci[0]:+.2f}, ${ci[1]:+.2f}]'
                   if ci else '')
            print(f'  {name:<26} ΔPnL=${dp:+9.2f} Δcapital=${dc:+9.2f} '
                  f'incremental ROI={incr:+7.2f}%' if incr is not None else
                  f'  {name} ΔPnL=${dp:+.2f} (no extra committed capital)', end='')
            print(cistr)
    print('\nWARNING: Results are in-sample and use trade prints, not executable asks/depth. '
          'Wallet-publication delay and skipped/partial-copy effects remain unknown.\n'
          'Read-only: no orders, no writes, public historical GETs only.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError,ValueError,OSError,AssertionError) as err:
        sys.exit('ABORT: '+str(err))
