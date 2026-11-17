#!/usr/bin/env python3
"""BreakTheBank V12 -- unbiased-by-book-price *accounting* bridge (read-only).

Purpose: compare exact same covered markets across source wallet-price
counterfactual and recorded-book IOC proxy, without removing any big winners.
This does NOT resolve missing-book selection or wallet-publication latency.
No API, trading, keys, network, downloads, source modifications or file writes.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import runpy


def summarize(label, rows):
    eligible = [m for m in rows if m['scored'] and m['buy_usdc'] > 1e-8]
    spent = sum(m['buy_usdc'] + m['fees'] for m in eligible)
    net = sum(m['pnl_usdc'] for m in eligible)
    unresolved = [m for m in rows if not m['scored'] and m['buy_usdc'] > 1e-8]
    print(f"  {label:37s}: scored={len(eligible):3} net=${net:+,.2f} "
          f"cost+fees=${spent:,.2f} ROI={100*net/spent:+.3f}%" if spent else
          f"  {label:37s}: scored=0 / no cost")
    return {'scored': len(eligible), 'net':net, 'spend': spent,
            'roi':100*net/spent if spent else None,
            'unresolved':len(unresolved),
            'unresolved_buy_cost':sum(m['buy_usdc'] for m in unresolved)}


def freshness(order, quotes, delay, age):
    series = quotes.get(order['token'])
    if series is None: return False
    times,_ = series
    ts = order['ts'] + delay
    i = bisect_right(times,ts)-1
    return i>=0 and ts-times[i] <= age+1e-8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'sports_copy_research')
    p.add_argument('--quotes',type=Path)
    p.add_argument('--scale',type=float,default=.001)
    p.add_argument('--delay',type=float,default=5.)
    p.add_argument('--max-cents',type=float,default=2.)
    p.add_argument('--max-book-age',type=float,default=10.)
    p.add_argument('--fee-coeff',type=float,default=.04)
    p.add_argument('--v8',type=Path,default=Path('src/sports-copy/breakthebank_v8_book_replay_readonly.py'))
    a=p.parse_args()
    root=a.root.expanduser()
    mod=runpy.run_path(str(a.v8))
    args=SimpleNamespace(v1_input_dir=root/'breakthebank_v1',v3_input_dir=root/'breakthebank_v3',
                         scale=a.scale,min_age=1,min_notional=1.,min_shares=0.,batch_window=60)
    orders,stat,v1,v3,payouts=mod['source_orders'](args)
    quotes=mod['read_quotes']((a.quotes or root/'breakthebank_v10/bbo_repaired.csv').expanduser())
    captured={}
    orig=v3.replay
    def capture(*arguments,**kwargs):
        captured['fills']=list(arguments[0]); captured['markets']=orig(*arguments,**kwargs)
        return captured['markets']
    v3.replay=capture
    try:
        exec_summary=mod['apply_books'](orders,quotes,scale=a.scale,delay=a.delay,
                  max_worse_cents=a.max_cents,max_quote_age=a.max_book_age,
                  fee_coeff=a.fee_coeff,v1=v1,v3=v3,payouts=payouts)
    finally:v3.replay=orig
    executions=captured['fills']
    markets=captured['markets']
    covered_scored={m['market'] for m in markets if m['scored'] and m['buy_usdc']>1e-8}
    assert len(covered_scored)==exec_summary['scored_only']['markets_scored']
    covered_orders=[o for o in orders if o['market'] in covered_scored]
    uncovered_orders=[o for o in orders if o['market'] not in covered_scored]
    covered_fresh=sum(freshness(o,quotes,a.delay,a.max_book_age) for o in covered_orders)
    other_fresh=sum(freshness(o,quotes,a.delay,a.max_book_age) for o in uncovered_orders)
    print('BREAKTHEBANK V12 -- MARKET-MATCHED BOOK-VS-WALLET AUDIT (OFFLINE)')
    print(f"Intent markets={len(set(o['market'] for o in orders))}, intents={len(orders)}, "
          f"book-selected scored markets={len(covered_scored)}")
    print(f"Within scored markets fresh-BBO intents: {covered_fresh}/{len(covered_orders)}; "
          f"other market fresh-BBO intents: {other_fresh}/{len(uncovered_orders)}")
    print('ALL intended order sides:',dict(Counter(o['side'] for o in orders)))
    print('Book-scored intended order sides:',dict(Counter(o['side'] for o in covered_orders)))
    print('Simulated book executions:',dict(Counter(e['side'] for e in executions)))
    if covered_fresh != len(covered_orders):
        print('WARNING: Scored set still has missing BBO intents; do not call it fully covered')
    if not covered_scored:raise RuntimeError('No book-scored markets')

    print('\nSOURCE MARKET SELECTION CHECK -- zero-fee comparison is NOT used')
    print('Same fee coefficient on wallet and book: ',a.fee_coeff)
    print('All 144-target intents, and identical 46 book-selected markets:')
    for cents in (0,2,5):
        full=orig(orders,payouts,v1,adverse_cents=cents,scale=a.scale,
                  fee_coeff=a.fee_coeff,tie_policy='buy-first')
        sel=orig(covered_orders,payouts,v1,adverse_cents=cents,scale=a.scale,
                 fee_coeff=a.fee_coeff,tie_policy='buy-first')
        oth=orig(uncovered_orders,payouts,v1,adverse_cents=cents,scale=a.scale,
                 fee_coeff=a.fee_coeff,tie_policy='buy-first') if uncovered_orders else []
        print(f'  Wallet-print worsening +{cents}c:')
        summarize('All sports intents',full)
        summarize('Same 46 book-scored markets',sel)
        summarize('Other markets (reported, not discarded)',oth)

    scored=[m for m in markets if m['scored'] and m['buy_usdc']>1e-8]
    book=summarize('BOOK execution: same 46 markets',scored)
    assert abs(book['net']-exec_summary['scored_only']['scored_net_pnl'])<.011
    exact_wallet=[]
    for e in executions:
        original=orders[e['seq']]
        assert (e['market'],e['token'],e['side']) == (original['market'],original['token'],original['side'])
        exact_wallet.append({**e,'wallet_price':original['wallet_price']})
    realized_wallet=orig(exact_wallet,payouts,v1,adverse_cents=0,scale=a.scale,
                         fee_coeff=a.fee_coeff,tie_policy='buy-first')
    twin=summarize('Wallet prices, EXACT same executed shares',realized_wallet)
    assert twin['scored']==book['scored']
    print(f"Book-price penalty for identical shares: ${book['net']-twin['net']:+,.2f}, "
          f"ROI change={book['roi']-twin['roi']:+.3f} percentage points")
    print('\nWEEKLY P&L OF COVERED BOOK EXECUTIONS (by earliest trade date UTC):')
    bymarket={m['market']:m for m in scored}
    first={}
    for o in covered_orders:
        if o['market'] not in first or o['ts'] < first[o['market']]:
            first[o['market']]=o['ts']
    weeks=Counter(); wkcost=Counter();wkcnt=Counter()
    for market,m in bymarket.items():
        y,w,_=datetime.fromtimestamp(first[market],timezone.utc).isocalendar()
        week=f'{y}-W{w:02d}'
        weeks[week]+=m['pnl_usdc'];wkcost[week]+=m['buy_usdc']+m['fees'];wkcnt[week]+=1
    for w in sorted(weeks):
        print(f'  {w}: markets={wkcnt[w]} PnL=${weeks[w]:+,.2f} '
              f'ROI={100*weeks[w]/wkcost[w]:+.2f}%')
    print('\nINTERPRETATION: Full market-set selection is not random, even when selected markets have 100% intent book coverage.')
    print('Price change contribution is observable; unsupported months/days, notification delay and order queue remain untested.')
    print('Zero trading/network requests and no file writes.')

if __name__=='__main__':main()
