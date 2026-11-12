#!/usr/bin/env python3
"""Read-only FIRST+CONFIRM+0 audit for wallet 954B -> Kalshi BTC15M.

Uses the SAME causal first/repeat eligibility, proportional order builder,
Kalshi trade-print proxy, official market outcomes, and taker fees as
w954_repeat_proportional_scale_in_readonly.py.

Focus: paired FIRST-only counterfactual using IDENTICAL opening orders,
profit concentration, sizes, capped confirmations, drawdown on settled P&L,
maximum concurrently committed entry costs, and sensitivity to worse prices.

No file writes, trades, POSTs, or GitHub modifications. May use historical
public Kalshi GETs unless --offline with local market and trade files.

IMPORTANT: Kalshi trade prints are NOT evidence of executable asks/depth.
Wallet archive timestamps are NOT verified public-detection timestamps.
"""
from __future__ import annotations

import argparse
import math
import random
import runpy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROP = HERE / 'w954_repeat_proportional_scale_in_readonly.py'


def pct(x, y):
    return 100*x/y if y else float('nan')


def money(x):
    return f'${x:+,.2f}'


def percentile(values, p):
    if not values:
        return float('nan')
    a = sorted(values)
    i = (len(a)-1)*p
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo]*(hi-i)+a[hi]*(i-lo)


def summary(markets):
    return (sum(m['pnl'] for m in markets),
            sum(m['buy_cost'] for m in markets))


def by_ticker(rows):
    out = {m['ticker']:m for m in rows}
    if len(out) != len(rows):
        raise RuntimeError('Duplicate market rows in replay')
    return out


def matched_pair(base_markets, full_markets):
    b, f = by_ticker(base_markets), by_ticker(full_markets)
    if set(b) != set(f):
        raise RuntimeError('Unpaired market populations; refusing causal attribution')
    return b, f


def day_bootstrap(base_markets, full_markets, n, seed=271828):
    if n <= 0: return None
    b, f = matched_pair(base_markets, full_markets)
    daily = defaultdict(lambda: [0.,0.,0.,0.])
    for ticker in sorted(b):
        m = f[ticker]; old = b[ticker]
        day = datetime.fromtimestamp(m['start'],timezone.utc).strftime('%Y-%m-%d')
        z = daily[day]
        z[0] += m['pnl']; z[1] += m['buy_cost']
        z[2] += m['pnl']-old['pnl']; z[3] += m['buy_cost']-old['buy_cost']
    days = list(daily.values())
    if len(days) < 3: return None
    rng = random.Random(seed)
    vals = [[], [], []]
    for _ in range(n):
        p=c=dp=0.
        for _ in days:
            x = days[rng.randrange(len(days))]
            p+=x[0];c+=x[1];dp+=x[2]
        vals[0].append(p);vals[1].append(pct(p,c));vals[2].append(dp)
    return [(percentile(z,.025),percentile(z,.975)) for z in vals], len(days)


def exposure_curve(executed, market_rows):
    """Reserve purchases until resolution; optimistic release at scheduled close.

    Reports funds committed, not margin requirement/actual cash flow or a
    liquidatable account drawdown. Trades in a market can span the window.
    """
    close_map = {}
    for m in market_rows:
        close_map[m['ticker']] = m['start'] + 900
    flows = []
    costs = defaultdict(float)
    for e in executed:
        if e['action'] != 'BUY': continue
        cost = e['copy_px']*e['copy_qty'] + e['fee']
        costs[e['ticker']] += cost
        flows.append((e['copy_at'], 1, cost))
    for ticker, cost in costs.items():
        flows.append((close_map[ticker], 0, -cost))
    locked=peak=0.
    peak_at=None
    for t, kind, cash in sorted(flows):
        locked += cash
        if locked > peak:
            peak=locked;peak_at=t
        if locked < -1e-6:
            raise RuntimeError('Negative reserved-capital amount')
    if abs(locked)>1e-5:
        raise RuntimeError('Capital reserve did not return to zero')
    return peak,peak_at, max(costs.values(),default=0.)


def settlement_curve(rows):
    """Cumulative P&L at market close, not mark-to-market pre-settlement."""
    running=peak=draw=0.
    low_at=None
    worst_streak=streak=0
    for m in sorted(rows,key=lambda x:(x['start']+900,x['ticker'])):
        running += m['pnl']
        if m['pnl'] < 0:
            streak+=1;worst_streak=max(worst_streak,streak)
        else:
            streak=0
        peak=max(peak,running)
        if peak-running>draw:
            draw=peak-running;low_at=m['start']+900
    return draw,low_at,worst_streak,running


def print_top(label, rows, n):
    print(f'  {label}:')
    if not rows:
        print('    (none)')
    for m in rows[:n]:
        when=datetime.fromtimestamp(m['start'],timezone.utc).strftime('%Y-%m-%d %H:%M')
        print(f'    {when}Z {m["ticker"]} pnl={money(m["pnl"])} '
              f'cost=${m["buy_cost"]:,.2f} ROI={pct(m["pnl"],m["buy_cost"]):+.1f}% '
              f'filled={m["filled"]} paid={m["payout"]}')


def assert_paired_first_orders(first, full):
    """Same initial orders in both simulations, no future data in first entry."""
    canonical=lambda e:(e['ticker'],e['ts'],e['line'],e['size'],e['side'])
    a=sorted(map(canonical,first))
    b=sorted(canonical(e) for e in full if e['kind']=='first')
    if a != b or len(a) != len(b):
        raise RuntimeError('Counterfactual first orders differ from full strategy')


def audit(args):
    if not PROP.is_file():
        raise SystemExit(f'Missing companion: {PROP}')
    p = runpy.run_path(str(PROP))
    core, af = p['refs']()
    cfg=SimpleNamespace(
        wallet=args.wallet, upper_first_price=.40,
        min_repeat_age=4,max_repeat_age=30,
        min_add_gap=4,add_window=120,
        max_market_qty=args.max_market_qty,
        sizing='proportional',base_qty=args.base_qty,add_qty=10,scale=.1,
        max_wait_s=args.max_wait_s,max_quote_age_s=2.,fee_multiplier=1.,
        latencies=[args.latency],merge_gap_s=15,
        trades_file=args.trades_file,offline=args.offline
    )
    print('WALLET 954B — FIRST BUY + PROPORTIONAL CONFIRMATION AUDIT (READ ONLY)')
    print('='*98,flush=True)
    print('Causal qualifying first buys 20c to <40c, 4–30s repeat; '
          'FIRST+CONFIRM+0, cap='+str(args.max_market_qty))
    print(f'Copied starter={args.base_qty}, latency={args.latency}s, '
          f'price stresses={args.penalties_cents}c; trade prints NOT executable asks.')
    signals=core['extract_signals'](SimpleNamespace(wallet=args.wallet,strategy='first',
              upper_first_price=.4,min_repeat_age=4,max_repeat_age=30))
    repeats=core['extract_signals'](SimpleNamespace(wallet=args.wallet,strategy='repeat',
              upper_first_price=.4,min_repeat_age=4,max_repeat_age=30))
    wallet,meta=af['prepare_wallet'](args.wallet)
    indexed=p['index_wallet'](wallet)
    planned,skipped,kinds=p['build'](signals,indexed,'first',0,cfg)
    base_planned=[e for e in planned if e['kind']=='first']
    assert len(base_planned)==len(signals), ('First signals omitted unexpectedly', skipped)
    print(f'Wallet archive raw={meta["raw"]:,}, qualifying first={len(signals)}, '
          f'qualifying repeat={len(repeats)}')
    print(f'Planned initial orders={len(base_planned)}, '
          f'planned confirmation orders={sum(e["kind"]=="repeat" for e in planned)}, '
          f'planning exclusions={dict(skipped)}')
    stubs=[{'start':s} for s in sorted({e['start'] for e in planned})]
    print('Loading historical Kalshi market metadata (public GET unless local file)...',flush=True)
    markets=list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](stubs)
    lookup=defaultdict(list)
    for m in markets:
        close=core['market_close'](m)
        if close is not None and str(m.get('ticker','')).startswith('KXBTC15M-'):
            lookup[round(close)].append(m)
    full,invalid=p['attach_market'](planned,lookup)
    first=[e for e in full if e['kind']=='first']
    assert_paired_first_orders(first,full)
    if len({e['ticker'] for e in first}) != len(first):
        raise RuntimeError('Duplicate initial orders in paired population')
    print(f'Kalshi paired markets={len(first)}, mapped all orders={len(full)}, '
          f'market exclusions={dict(invalid)}')
    if not first: raise RuntimeError('No matched markets')
    if args.offline and not args.trades_file:
        raise RuntimeError('--offline requires both local --markets-file and --trades-file')
    print('Loading historical Kalshi aggressive-buy trade prints (GET unless local file)...',flush=True)
    pp=af['prep_prints'](af['load_trade_prints'](full,cfg,core))
    replay_cfg=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
                  max_wait_s=args.max_wait_s,max_quote_age_s=2.,fee_multiplier=1.)
    first_ex,first_ms,first_reject,first_dep=p['simulate_dependent'](
        first,replay_cfg,args.latency,af,pp,None)
    full_ex,full_ms,full_reject,full_dep=p['simulate_dependent'](
        full,replay_cfg,args.latency,af,pp,None)
    b,f=matched_pair(first_ms,full_ms)
    base_pnl,base_cost=summary(first_ms)
    full_pnl,full_cost=summary(full_ms)
    inc_pnl=full_pnl-base_pnl
    inc_cost=full_cost-base_cost
    print('\nPAIRED RESULTS (identical opening order/timestamp for every market)')
    print('-'*98)
    print(f'  First-only:     N={len(first_ms)} fills={len(first_ex)}/{len(first)} '
          f'PnL={money(base_pnl)} cost=${base_cost:,.2f} ROI={pct(base_pnl,base_cost):+.2f}% '
          f'reject={dict(first_reject)}')
    print(f'  First+confirm:  N={len(full_ms)} fills={len(full_ex)}/{len(full)} '
          f'PnL={money(full_pnl)} cost=${full_cost:,.2f} ROI={pct(full_pnl,full_cost):+.2f}% '
          f'reject={dict(full_reject)} dependent_skips={full_dep}')
    print(f'  Confirmation incremental: PnL={money(inc_pnl)}, '
          f'extra outlay=${inc_cost:,.2f}, incremental ROI={pct(inc_pnl,inc_cost):+.2f}%')
    print(f'  Fees: first-only=${sum(m["fees"] for m in first_ms):,.2f}; '
          f'confirmed=${sum(m["fees"] for m in full_ms):,.2f}')
    print('\nADVERSE ENTRY PRICE STRESS — volume matching held fixed (NOT a fillability test)')
    ba=p['run_price_stress'](first_ex,first_ms,args.penalties_cents,af['fee'],1.)
    fa=p['run_price_stress'](full_ex,full_ms,args.penalties_cents,af['fee'],1.)
    for (c,bp,bc,br),(d,fp,fc,fr) in zip(ba,fa):
        assert c==d
        print(f'  +{c:g}c: FIRST={br:+7.2f}% ({money(bp)}), '
              f'FIRST+CONFIRM={fr:+7.2f}% ({money(fp)}), '
              f'confirmation incremental PnL={money(fp-bp)}, '
              f'incremental ROI={pct(fp-bp,fc-bc):+.2f}%')
    boots=day_bootstrap(first_ms,full_ms,args.bootstrap)
    if boots:
        cis,days=boots
        print(f'\nDAY-CLUSTER BOOTSTRAP (n={args.bootstrap}, independent UTC days={days})')
        print(f'  Confirmed total PnL: 95% CI [{money(cis[0][0])}, {money(cis[0][1])}]')
        print(f'  Confirmed ROI:       95% CI [{cis[1][0]:+.2f}%, {cis[1][1]:+.2f}%]')
        print(f'  Incremental PnL:     95% CI [{money(cis[2][0])}, {money(cis[2][1])}]')
    print('\nWEEKLY AND LEAVE-ONE-WEEK-OUT ROBUSTNESS')
    weeks=defaultdict(list)
    for m in full_ms:
        weeks[datetime.fromtimestamp(m['start'],timezone.utc).strftime('%G-W%V')].append(m)
    for w in sorted(weeks):
        subset=weeks[w];pnl,cost=summary(subset)
        rest=[m for m in full_ms if m not in subset]
        rp,rc=summary(rest)
        print(f'  {w}: N={len(subset):3} PnL={money(pnl):>12} cost=${cost:>8,.2f} '
              f'ROI={pct(pnl,cost):+7.2f}% | EXCLUDING week ROI={pct(rp,rc):+7.2f}%')
    print('\nPROFIT CONCENTRATION')
    ranked=sorted(full_ms,key=lambda m:m['pnl'],reverse=True)
    winners=[m for m in ranked if m['pnl']>0]
    losses=[m for m in ranked if m['pnl']<0]
    print(f'  Winning markets={len(winners)}, losing markets={len(losses)}, '
          f'zero={len(ranked)-len(winners)-len(losses)}')
    print(f'  Gross winners={money(sum(m["pnl"] for m in winners))}, '
          f'gross losers={money(sum(m["pnl"] for m in losses))}')
    for count in (1,3,5,10,20):
        top=ranked[:count]
        remaining=ranked[count:]
        remaining_pnl,remaining_cost=summary(remaining)
        print(f'  Top {count:2} winners: profit={money(sum(m["pnl"] for m in top))} '
              f'({pct(sum(m["pnl"] for m in top),full_pnl):+.1f}% of net PnL); '
              f'excluding top {count} ROI={pct(remaining_pnl,remaining_cost):+.2f}%')
    print_top('Largest WINNING markets',ranked,args.top)
    print_top('Largest LOSING markets',list(reversed(ranked)),args.top)
    print('\nPROPORTIONAL CONFIRMATION POSITION SIZES (actual copied orders)')
    first_by=defaultdict(int);adds_by=defaultdict(int)
    for e in full_ex:
        if e['kind']=='first':first_by[e['ticker']]+=e['copy_qty']
        if e['kind']=='repeat':adds_by[e['ticker']]+=e['copy_qty']
    add_values=list(adds_by.values())
    print(f'  Confirmed copied markets={len(add_values)}, '
          f'orders entering at confirmation={sum(e["kind"]=="repeat" for e in full_ex)}')
    if add_values:
        print(f'  Added contract quantity: median={percentile(add_values,.5):.1f}, '
              f'p90={percentile(add_values,.9):.1f}, '
              f'p95={percentile(add_values,.95):.1f}, max={max(add_values)}')
    total_positions=[first_by[t]+adds_by[t] for t in sorted(first_by)]
    print(f'  Final copied position size (all markets): '
          f'median={percentile(total_positions,.5):.1f}, '
          f'p95={percentile(total_positions,.95):.1f}, max={max(total_positions) if total_positions else 0}')
    print(f'  Planned confirmation cap skips={skipped["market_qty_cap_confirmation"]}; '
          f'actual completed order count first={sum(e["kind"]=="first" for e in full_ex)} '
          f'confirm={sum(e["kind"]=="repeat" for e in full_ex)}')
    print('  Confirmation increments by filled add size (market-level PnL difference):')
    bins=[(1,10),(11,24),(25,49),(50,99),(100,249)]
    for lo,hi in bins:
        ts=[t for t,q in adds_by.items() if lo<=q<=hi]
        if not ts: continue
        d_pnl=sum(f[t]['pnl']-b[t]['pnl'] for t in ts)
        d_cost=sum(f[t]['buy_cost']-b[t]['buy_cost'] for t in ts)
        print(f'    Added {lo:3}–{hi:3} contracts: markets={len(ts):2} '
              f'ΔPnL={money(d_pnl):>12} Δcapital=${d_cost:,.2f} '
              f'incremental ROI={pct(d_pnl,d_cost):+.2f}%')
    print('\nHISTORICAL EXPOSURE / SETTLEMENT P&L DRAWDOWN')
    reserved,at,max_market=exposure_curve(full_ex,full_ms)
    draw,low_at,streak,realized=settlement_curve(full_ms)
    utc=lambda t: datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ') if t else 'n/a'
    print(f'  Approx peak concurrent locked entry cost=${reserved:,.2f} at {utc(at)}')
    print(f'  Largest single-market outlay=${max_market:,.2f}')
    print(f'  Maximum realized PnL drawdown=${draw:,.2f} at {utc(low_at)}; '
          f'longest consecutive losing markets={streak}')
    print('  WARNING: PnL is booked at scheduled close, not actual payout credit; '
          'this is NOT a mark-to-market drawdown or proven capital requirement.')
    days=defaultdict(list)
    for m in full_ms:
        days[datetime.fromtimestamp(m['start']+900,timezone.utc).strftime('%Y-%m-%d')].append(m)
    worst=sorted(((summary(v)[0],day,len(v)) for day,v in days.items()))[:5]
    print('  Worst settlement days: '+', '.join(f'{day} N={n} {money(val)}' for val,day,n in worst))
    print('\nCAUTIONS: Results have model-selection bias (strategy chosen after exploring historical data).')
    print('Kalshi trade prints do not establish available historical ask or depth.')
    print('Wallet-publication latency and uncopied hedging/sells remain unknown. No orders or writes.')


def self_test():
    assert abs(percentile([1,2,3,4],.5)-2.5)<1e-10
    assert abs(pct(1,4)-25)<1e-10
    markets=[{'ticker':'A','start':0,'pnl':10,'buy_cost':5},
             {'ticker':'B','start':900,'pnl':-7,'buy_cost':10}]
    dd,at,streak,v=settlement_curve(markets)
    assert dd==7 and at==1800 and streak==1 and v==3
    entries=[{'ticker':'A','action':'BUY','copy_px':.4,'copy_qty':10,
              'fee':.2,'copy_at':100},
             {'ticker':'B','action':'BUY','copy_px':.2,'copy_qty':20,
              'fee':.2,'copy_at':200}]
    peak,at,max_market=exposure_curve(entries,markets)
    assert abs(peak-8.4)<1e-9 and at==200 and abs(max_market-4.2)<1e-9
    a=[dict(ticker='A',ts=100,line=1,size=10,side='UP',kind='first')]
    b=a+[dict(ticker='A',ts=110,line=2,size=5,side='UP',kind='repeat')]
    assert_paired_first_orders(a,b)
    ci=day_bootstrap([dict(ticker='A',start=0,pnl=1.,buy_cost=2.),
                      dict(ticker='B',start=86400,pnl=-.2,buy_cost=2.),
                      dict(ticker='C',start=172800,pnl=.4,buy_cost=2.)],
                     [dict(ticker='A',start=0,pnl=2.,buy_cost=3.),
                      dict(ticker='B',start=86400,pnl=-.3,buy_cost=3.),
                      dict(ticker='C',start=172800,pnl=.6,buy_cost=3.)],100)
    assert ci is not None and ci[1]==3
    print('PASS: paired-first attribution, concentration helpers, scheduled-close reserves, '
          'realized drawdown, day bootstrap; no network or file writes')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--wallet',type=Path)
    parser.add_argument('--markets-file',type=Path)
    parser.add_argument('--trades-file',type=Path)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--latency',type=int,default=5)
    parser.add_argument('--base-qty',type=int,default=10)
    parser.add_argument('--max-market-qty',type=int,default=250)
    parser.add_argument('--max-wait-s',type=float,default=5.)
    parser.add_argument('--penalties-cents',type=float,nargs='+',default=[0,1,2,5])
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--top',type=int,default=7)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if (args.latency<0 or args.latency>180 or args.base_qty<1 or
        args.max_market_qty<args.base_qty or args.bootstrap<0 or
        args.max_wait_s<0 or not args.penalties_cents or
        min(args.penalties_cents)<0 or args.top<1 or args.top>30):
        raise SystemExit('Invalid audit parameters')
    if args.offline and not (args.markets_file and args.trades_file):
        raise SystemExit('Offline mode requires --markets-file and --trades-file')
    audit(args)


if __name__=='__main__':
    main()
