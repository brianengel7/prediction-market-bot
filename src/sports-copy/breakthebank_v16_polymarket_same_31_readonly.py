#!/usr/bin/env python3
"""BreakTheBank V16 — Polymarket vs Kalshi on EXACT SAME 31 matched NBA markets.

READ ONLY: no network, no trading, no writes, no source changes. Existing V8
order formation, V3 settlement accounting, V14 matched-market plan, and V15
Kalshi replay JSON. Keeps ALL matched markets and ALL source copy intents.

Polymarket wallet-print scenarios are hypothetical, not historical executable
quotes. Optional historical L2 replay is limited to fully covered market intents
in archived PMData files. Neither venue has live signal-publication lag here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def assert_intent_matches_plan(record, original, scale):
    if not record.get('intent_id', '').isdigit():
        raise ValueError('Noninteger/empty V14 intent ID')
    if str(record['poly_market']).lower() != original['market']:
        raise ValueError('Mismatched Polymarket condition ID for intent '+record['intent_id'])
    if str(record['poly_token']) != original['token']:
        raise ValueError('Mismatched Polymarket token for intent '+record['intent_id'])
    if str(record['side']) != original['side']:
        raise ValueError('Mismatched copy side for intent '+record['intent_id'])
    if not math.isclose(float(record['shares']), original['wallet_qty']*scale, rel_tol=1e-8,abs_tol=1e-7):
        raise ValueError('Mismatched proportional share quantity for intent '+record['intent_id'])
    if not math.isclose(float(record['wallet_price']), original['wallet_price'], rel_tol=0.,abs_tol=1e-8):
        raise ValueError('Mismatched source price for intent '+record['intent_id'])
    if not math.isclose(float(record['ts']), original['ts'], rel_tol=0.,abs_tol=1e-6):
        raise ValueError('Mismatched source timestamp for intent '+record['intent_id'])


def read_plan(path, orders, scale):
    if not path.is_file():
        raise FileNotFoundError(path)
    by_id={}; ticker_to_cid={}; cid_to_ticker={}
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['league'] != 'nba':
                continue
            idx=int(r['intent_id'])
            if idx < 1 or idx > len(orders):
                raise ValueError('Out of range V14 intent ID')
            if idx in by_id:
                raise ValueError('Duplicate V14 NBA intent ID')
            order=orders[idx-1]
            assert_intent_matches_plan(r,order,scale)
            cid=order['market'];ticker=r['kalshi_ticker']
            if cid in cid_to_ticker and cid_to_ticker[cid]!=ticker:
                raise ValueError('Two Kalshi team contracts for same Polymarket market; unsupported aggregation')
            if ticker in ticker_to_cid and ticker_to_cid[ticker]!=cid:
                raise ValueError('Two Polymarket markets map to same Kalshi ticker; unsupported aggregation')
            by_id[idx]=r;cid_to_ticker[cid]=ticker;ticker_to_cid[ticker]=cid
    if not by_id:
        raise ValueError('No NBA intents in V14 plan')
    # Preserve every source copy order for each selected outcome market.
    selected={c for c in cid_to_ticker}
    source_ids={i for i,x in enumerate(orders,1) if x['market'] in selected}
    if source_ids != set(by_id):
        raise RuntimeError(f'Matched NBA market has missing intended order IDs: '
                           f'{len(source_ids - set(by_id))} missing, {len(set(by_id)-source_ids)} unexpected')
    return selected,cid_to_ticker,by_id


def summarize(label, rows):
    won=[r for r in rows if r['scored'] and r['buy_usdc']>1e-8]
    uns=[r for r in rows if not r['scored'] and r['buy_usdc']>1e-8]
    cost=sum(r['buy_usdc']+r['fees'] for r in won)
    pnl=sum(r['pnl_usdc'] for r in won)
    b=sum(r['buy_fills'] for r in won);s=sum(r['sell_fills'] for r in won)
    print(f'  {label:29s} markets={len(won):2d}  PnL=${pnl:+,.2f}  '
          f'cost+fees=${cost:,.2f}  ROI={(100*pnl/cost if cost else float("nan")):+.3f}% '
          f'buys={b} sell-intents={s} unresolved={len(uns)}')
    return {r['market']:r for r in won}


def fresh(o,quotes,delay,age):
    q=quotes.get(o['token'])
    if not q:return False
    stamps,_=q;t=o['ts']+delay;i=bisect_right(stamps,t)-1
    return i>=0 and t-stamps[i] <= age+1e-8


def selftest():
    o={'market':'0x'+'a'*64,'token':'101','side':'BUY','wallet_qty':123.,'wallet_price':.43,'ts':100.}
    r={'intent_id':'1','poly_market':o['market'],'poly_token':o['token'],
       'side':'BUY','shares':'0.123','wallet_price':'0.43','ts':'100'}
    assert_intent_matches_plan(r,o,.001)
    try:
        assert_intent_matches_plan(dict(r,poly_token='wrong'),o,.001)
        raise AssertionError('Incorrect token accepted')
    except ValueError: pass
    try:
        assert_intent_matches_plan(dict(r,shares='4'),o,.001)
        raise AssertionError('Incorrect scaled qty accepted')
    except ValueError: pass
    q={'101':([98.,99.],[(.2,1,.4,1),(.3,1,.41,1)])}
    assert fresh(o,q,0,2) and not fresh(o,q,0,.5)
    print('PASS V16: matching condition/token/side/shares/price/timestamp, quote freshness; OFFLINE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'sports_copy_research')
    p.add_argument('--plan',type=Path,default=None,help='V14 full matched market plan')
    p.add_argument('--pilot-plan',type=Path,default=None,help='V14 original 25-candidate plan')
    p.add_argument('--kalshi-report',type=Path,default=None,help='V15 full settlement report')
    p.add_argument('--quotes',type=Path,default=None,help='Optional V10 repaired Polymarket BBO CSV')
    p.add_argument('--v8',type=Path,default=Path('src/sports-copy/breakthebank_v8_book_replay_readonly.py'))
    p.add_argument('--scale',type=float,default=.001)
    p.add_argument('--fee-coeff',type=float,default=.05,
                   help='Hypothetical POLY quadratic fee coeff (not verified actual sports fees)')
    p.add_argument('--delay',type=float,default=5.,help='Historical Poly BBO delay, not wallet publication')
    p.add_argument('--max-cents',type=float,default=2.,help='Historical Poly BBO max adverse price')
    p.add_argument('--max-book-age',type=float,default=10.)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest();return
    if a.scale<=0 or a.fee_coeff<0 or a.delay<0 or a.max_cents<0 or a.max_book_age<0:
        p.error('Invalid numeric settings')
    root=a.root.expanduser()
    plan=a.plan or root/'breakthebank_v14/kalshi_matches_full.csv'
    pilot=a.pilot_plan or root/'breakthebank_v14/kalshi_matches_pilot.csv'
    report=a.kalshi_report or root/'breakthebank_v14/settlement_full.json'
    v8=runpy.run_path(str(a.v8))
    opts=SimpleNamespace(v1_input_dir=root/'breakthebank_v1',v3_input_dir=root/'breakthebank_v3',
                         scale=a.scale,min_age=1,min_notional=1.,min_shares=0.,batch_window=60)
    orders,stats,v1,v3,payouts=v8['source_orders'](opts)
    chosen,by_market,by_id=read_plan(plan,orders,a.scale)
    pilot_set,_,_=read_plan(pilot,orders,a.scale)
    if not pilot_set.issubset(chosen):
        raise RuntimeError('Original pilot contains markets missing from full plan')
    new_set=chosen-pilot_set
    with report.open() as f:
        kal=json.load(f)
    kal_markets=kal['delays']['5']['markets']
    if set(by_market.values())!=set(kal_markets):
        raise RuntimeError(f'Kalshi 5s report tickers differ: plan={len(by_market)}, report={len(kal_markets)}')
    chosen_orders=[r for r in orders if r['market'] in chosen]
    print('BREAKTHEBANK V16 — POLYMARKET vs KALSHI, SAME MATCHED NBA OUTCOMES')
    print('OFFLINE, READ ONLY; markets and source intents are NOT filtered by profitability')
    print(f'Full V8 source: {len(set(r["market"] for r in orders))} markets, {len(orders)} intents')
    print(f'Exact matched cohort: {len(chosen)} Polymarket markets = {len(by_market)} Kalshi tickers; '
          f'{len(chosen_orders)} same BUY/SELL intents (V14 IDs verified)')
    print(f'Original pilot={len(pilot_set)}, additional={len(new_set)}')
    print(f'POLY scenario: fixed proportional wallet quantities, $1/60s batches, post-taker +1s, '
          f'fee coefficient={a.fee_coeff} (sports taker fee STRESS); NO publication delay or executable quote assertion')
    rows_by_shock={}
    for shock in (0,2,5):
        direct=v3.replay(chosen_orders,payouts,v1,adverse_cents=shock,scale=a.scale,
                         fee_coeff=a.fee_coeff,tie_policy='buy-first')
        rows_by_shock[shock]={r['market']:r for r in direct}
        print(f'\nPOLYMARKET wallet-print price stress +{shock}¢ (fee stress included):')
        summarize('Original pilot', [r for r in direct if r['market'] in pilot_set])
        summarize('Additional games',[r for r in direct if r['market'] in new_set])
        summarize('All matched NBA games',direct)
    print('\nKALSHI 5s candle close proxy for SAME 31 teams (7% hypothetical fee):')
    kc=sum(float(m['cost'])+float(m['fees']) for m in kal_markets.values())
    kp=sum(float(m['pnl']) for m in kal_markets.values())
    print(f'  markets={len(kal_markets)} PnL=${kp:+,.2f}  cost+fees=${kc:,.2f} ROI={100*kp/kc:+.3f}%')
    print('  Different venue/fee assumptions; neither backtest establishes live executable returns.')
    print('\nSAME GAME BREAKDOWN (Polymarket +2¢ wallet price/fee vs Kalshi 5s candle/fee):')
    print('  POLY_PNL  KALSHI_PNL   PERIOD  KALSHI_TICKER')
    for cid,ticker in sorted(by_market.items(),key=lambda z:z[1]):
        pr=rows_by_shock[2][cid]
        pp=f'${pr["pnl_usdc"]:+,.2f}' if pr['scored'] else 'UNRESOLVED'
        print(f'  {pp:>11} ${float(kal_markets[ticker]["pnl"]):>+10,.2f} '
              f'  {"PILOT" if cid in pilot_set else "ADDED":5s}  {ticker}')
    if a.quotes:
        quotes=v8['read_quotes'](a.quotes.expanduser())
        bycid=defaultdict(list)
        for r in chosen_orders:bycid[r['market']].append(r)
        full_cov={cid for cid,rs in bycid.items() if all(fresh(r,quotes,a.delay,a.max_book_age) for r in rs)}
        partial={cid for cid,rs in bycid.items() if any(fresh(r,quotes,a.delay,a.max_book_age) for r in rs)}-full_cov
        covered=[r for r in chosen_orders if r['market'] in full_cov]
        total_fresh=sum(fresh(r,quotes,a.delay,a.max_book_age) for r in chosen_orders)
        print('\nPOLYMARKET HISTORICAL L2 — availability-only cohort diagnostic')
        print(f'  Quote-observable intents: {total_fresh}/{len(chosen_orders)}; '
              f'full-intent BBO markets: {len(full_cov)}/{len(chosen)}; '
              f'partial-intent markets: {len(partial)}')
        if covered:
            result=v8['apply_books'](covered,quotes,scale=a.scale,delay=a.delay,
                      max_worse_cents=a.max_cents,max_quote_age=a.max_book_age,
                      fee_coeff=a.fee_coeff,v1=v1,v3=v3,payouts=payouts)
            print(f'  Fully observable cohort only: filled/partial={result["orders_filled_or_partially_filled"]}/'
                  f'{len(covered)} scored={result["scored_only"]["markets_scored"]} '
                  f'PnL=${result["scored_only"]["scored_net_pnl"]:+,.2f} '
                  f'ROI={result["scored_only"]["scored_roi_pct"]}%')
        print('  WARNING: L2 covered outcomes are historical-selection biased; do not generalize this ROI.')
    print('\nAll figures are turnover ROI, NOT account returns. No unobserved real fills, no API or orders.')


if __name__=='__main__':main()
