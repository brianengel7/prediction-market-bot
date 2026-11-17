#!/usr/bin/env python3
"""Albert1953 V7: OFFLINE 1% economic-risk and category diagnostic.

Reads V6 JSON and CSV, V5 /v2/tokens cache, and V1 complete fills.
Never calls exchange APIs, places orders, or modifies prior research files.

Separates settled-market, source-print stress results from the unverified
settlement population; provides category net PnL/fees and profit concentration.
For unverified markets, models inventory and cashflows WITHOUT assuming that
remaining tokens have a payoff of zero, have settled, or have an executable bid.
Unverified inventory cost is NOT a realized loss or mark to market value.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from decimal import Decimal as D
import json
from pathlib import Path

import albert1953_copy_sell_audit_1pct_v3 as v3
import albert1953_settled_pnl_v4 as v4
import albert1953_settled_pnl_v5_tokens as v5

Z=D('0'); O=D('1')


def inventory_cash(rows, ratio, step, minnot):
    """Same source-inventory targeting as V6; separate unsettled cash / FIFO cost."""
    src=defaultdict(lambda:Z)
    held=defaultdict(lambda:Z)
    lots=defaultdict(deque)
    buys=sells=Z
    n_buy=n_sell=0
    for r in rows:
        tok=r['token']; q=r['q']; p=r['price']; side=r['side']
        src[tok]+=q if side=='BUY' else -q
        if src[tok]<-D('.00000001'):
            raise AssertionError('Invalid negative source inventory in unsettled market')
        if src[tok]<Z: src[tok]=Z
        desired=v3.exact_floor(src[tok]*ratio,step)
        diff=desired-held[tok]
        if side=='BUY' and diff>=step and diff*p>=minnot:
            held[tok]+=diff; buys+=diff*p; n_buy+=1;lots[tok].append([diff,p])
        elif side=='SELL' and diff<=-step:
            qty=min(-diff,held[tok])
            if qty>=step and qty*p>=minnot:
                held[tok]-=qty; sells+=qty*p; n_sell+=1
                rem=qty
                while rem>Z and lots[tok]:
                    front=lots[tok][0]
                    part=min(front[0],rem)
                    front[0]-=part;rem-=part
                    if front[0]==Z:lots[tok].popleft()
                if rem>D('.00000001'):
                    raise AssertionError('Missing copied lots')
    remaincost=sum((q*p for ll in lots.values() for q,p in ll),Z)
    remainq=sum(held.values(),Z)
    return {'buy_spend':buys,'sell_proceeds':sells,'remaining_cost':remaincost,
            'remaining_shares':remainq,'buy_events':n_buy,'sell_events':n_sell,
            'cash_outflow_net':buys-sells}


def assess_unverified(cache,ratio,step,minnot):
    rows,meta,order=v3.read_source(cache)
    v3.market_categories(rows)
    bad,amb=v3.validate_token_inventory(rows)
    ix=v4.outcomes_by_token(cache/'albert1953_trades.jsonl')
    path=cache/'albert1953_v5_token_resolution_cache.json'
    if not path.is_file(): raise ValueError('Missing V5 token cache')
    c=json.loads(path.read_text())
    if c.get('version')!=v5.VERSION or c.get('wallet')!=v3.WALLET: raise ValueError('Wrong V5 token cache')
    states,fail=v5.make_states(rows,c['rows_by_condition'],bad,amb,ix)
    grouped=defaultdict(list)
    for r in rows: grouped[r['condition'].lower()].append(r)
    unknown=[]
    reasons=Counter()
    for cid,rs in grouped.items():
        if cid in states: continue
        if any((r['condition'],r['token']) in bad for r in rs):
            reasons['negative_inventory']+=1;continue
        if any((r['condition'],r['token']) in amb for r in rs):
            reasons['ambiguous_same_second']+=1;continue
        _,reason=v5.verify_market(cid,rs,c['rows_by_condition'].get(cid,[]),ix)
        reasons[reason or 'unknown']+=1
        result=inventory_cash(rs,ratio,step,minnot)
        unknown.append({'condition_id':cid,'title':rs[-1]['title'],'category':rs[-1]['category'],
                        'last_trade_ts':rs[-1]['ts'],'reason':reason,**result})
    return unknown,dict(reasons),len(states),len(grouped)


def analyze_stresses(report,scen='whole_no_min'):
    data=report['scenarios'][scen]
    labels=('source_prints_no_fees','taker_fee_only','taker_plus_0.005_per_side',
            'taker_plus_0.01_per_side','taker_plus_0.02_per_side','taker_plus_0.03_per_side')
    print('\nFEE/PRICE STRESS, CUMULATIVE ON GROSS BUY SPEND')
    print('Scenario:',scen)
    print(f"{'Stress':32s} {'All net':>12s} {'All ROI':>10s} {'Recent net':>12s} {'Recent ex5':>12s} {'All ex5':>12s}")
    for lab in labels:
        if lab not in data: continue
        f=data[lab]['all'];r=data[lab]['markets_traded_last_90d']
        print(f"{lab:32s} {float(f['net_pnl_usd']):+12,.2f} {float(f['roi_on_gross_buys_pct']):+9.2f}% {float(r['net_pnl_usd']):+12,.2f} {float(r['ex_top5_positive_market_pnl_usd']):+12,.2f} {float(f['ex_top5_positive_market_pnl_usd']):+12,.2f}")
    print('\nSETTLED-MARKET CATEGORY P&L AT 1 CENT WORSE PER SIDE + STANDARD TAKER FEES')
    target=data['taker_plus_0.01_per_side']['all']
    cats=target['category']
    print(f"{'Category':22s} {'Markets':>8s} {'Buy spend':>12s} {'Net P&L':>11s} {'ROI':>9s} {'Fees':>9s}")
    for cat,d in sorted(cats.items(),key=lambda x:-float(x[1]['pnl'])):
        buy=float(d['spend']);pnl=float(d['pnl']);fee=float(d['fees'])
        print(f'{cat:22.22s} {int(d["markets"]):8,d} {buy:12,.2f} {pnl:+11,.2f} {(pnl/buy*100 if buy else 0):+8.2f}% {fee:9,.2f}')
    return target


def show_top_markets(path):
    if not path.is_file():raise FileNotFoundError(path)
    with path.open(newline='') as f:
        arr=list(csv.DictReader(f))
    for stress,rec in [('taker_plus_0.01_per_side',False),('taker_plus_0.01_per_side',True)]:
        x=[r for r in arr if r['stress']==stress and (not rec or r['last_90d'].lower()=='true')]
        x.sort(key=lambda r:float(r['copy_pnl']),reverse=True)
        n=sum(D(r['copy_pnl']) for r in x)
        top5=sum((D(r['copy_pnl']) for r in x[:5]),Z)
        print('\nTOP 5', 'MARKETS WITH LAST-90D ACTIVITY' if rec else 'SETTLED MARKETS','(1c worse + taker)')
        print('Total:',f'${n:,.2f}', 'Top 5:',f'${top5:,.2f}','Other net:',f'${n-top5:,.2f}')
        for r in x[:5]:print(f"  {float(r['copy_pnl']):+9.2f}  {r['category'][:14]:14s}  {r['title'][:90]}")
        print('BOTTOM 5')
        for r in x[-5:][::-1]:print(f"  {float(r['copy_pnl']):+9.2f}  {r['category'][:14]:14s}  {r['title'][:90]}")


def self_test():
    def r(ts,side,q,p,t='A'):
        return {'ts':ts,'side':side,'q':D(q),'price':D(p),'token':t}
    seq=[r(1,'BUY','60','.5'),r(2,'BUY','70','.4'),r(3,'SELL','50','.8')]
    got=inventory_cash(seq,D('.01'),O,Z)
    # target holdings: 0 -> 1 after 130 source shares, then 0 after selling 50
    assert got['buy_spend']==D('.4') and got['sell_proceeds']==D('.8') and got['remaining_cost']==Z and got['remaining_shares']==Z,got
    seq=[r(1,'BUY','300','.4'),r(2,'SELL','200','.6')]
    got=inventory_cash(seq,D('.01'),O,Z)
    assert got['buy_spend']==D('1.2') and got['sell_proceeds']==D('1.2') and got['remaining_cost']==D('.4') and got['remaining_shares']==O,got
    print('PASS: 1pct cumulative rounding, unsettled BUY/SELL inventory, FIFO lot cost; offline')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--ratio',type=D,default=D('.01'))
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test: return self_test()
    if not Z<a.ratio<=O:p.error('ratio must be > 0 and <= 1')
    suf='ratio'+format(a.ratio,'f').replace('.','p')
    report_path=a.cache/f'albert1953_execution_stress_v6_{suf}.json'
    trace_path=a.cache/f'albert1953_execution_stress_v6_{suf}_markets.csv'
    report=json.loads(report_path.read_text())
    if D(report['ratio'])!=a.ratio:raise ValueError('Wrong ratio on V6 report')
    print('ALBERT1953 V7 — OFFLINE CATEGORY, CONCENTRATION & UNVERIFIED INVENTORY RISK')
    analyze_stresses(report)
    show_top_markets(trace_path)
    unknown,reasons,verified,total=assess_unverified(a.cache,a.ratio,O,Z)
    print('\nUNVERIFIED FINAL PAYOUTS (WHOLE-CONTRACT, NO MINIMUM)')
    print('Cached settled:',verified,'Total markets:',total,'Excluded / unresolved reasons:',reasons)
    print('Excluded for incomplete source inventory histories: 7 should remain excluded')
    spent=sum((r['buy_spend'] for r in unknown),Z)
    proceeds=sum((r['sell_proceeds'] for r in unknown),Z)
    residual=sum((r['remaining_cost'] for r in unknown),Z)
    active=[r for r in unknown if r['remaining_shares']>Z]
    print(f'Payout-unverified markets with valid inventory: {len(unknown)}; unverified with copied residual shares: {len(active)}')
    print(f'Copy buys: ${spent:,.2f}; copy sells: ${proceeds:,.2f}; net historical cash spent: ${spent-proceeds:,.2f}')
    print(f'Unverified remaining copied shares cost basis: ${residual:,.2f} (NOT P&L, NOT current open-market value)')
    print('Top unverified by remaining copied inventory cost:')
    for r in sorted(active,key=lambda x:x['remaining_cost'],reverse=True)[:12]:
        print(f"  ${r['remaining_cost']:9,.2f} cost  ${r['cash_outflow_net']:9,.2f} net cash  {r['category'][:13]:13s} {r['title'][:85]}")
    print('\nNO HISTORICAL KALSHI QUOTES OR VERIFIED MARKET MATCHES. NO ORDERS. Unverified inventory has NO assumed settlement payout.')

if __name__=='__main__':main()
