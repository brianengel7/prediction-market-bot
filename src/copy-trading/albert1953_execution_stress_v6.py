#!/usr/bin/env python3
"""Albert1953: OFFLINE 1% follower stress test; no API calls, credentials, orders.

Reads previously cached COMPLETE Polymarket wallet fills and /v2/tokens records
created by albert1953_settled_pnl_v5_tokens.py. Does NOT interpolate historical
quotes, claim Kalshi availability, or validate market resolution equivalence.

Copies each SOURCE BUY and SELL using V4/V5's proportional cumulative-inventory
rules, then separately models:
- hypothetical adverse fill prices: buys +N cents, sells -N cents;
- 2026-07-07 Kalshi STANDARD taker fee at hypothetical copied fill prices:
    ceil_to_0.0001(0.07 * filled contracts * price * (1-price)).
  Series-specific fee rates, true order aggregation and fill behavior are UNKNOWN.
- payout at explicitly verified /v2/tokens final prices, for resolved markets only.

The fee stress is a WHAT-IF and not a genuine Kalshi replay; Polymarket source
prints are not executable Kalshi BBO quotes. An end-window recent flag means a
market TRADED during the final 90 days; it does not time-slice its historical PnL.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from decimal import Decimal as D, ROUND_CEILING
import json
from pathlib import Path

import albert1953_copy_sell_audit_1pct_v3 as v3
import albert1953_settled_pnl_v4 as v4
import albert1953_settled_pnl_v5_tokens as v5

Z, O = D('0'), D('1')
FEE_RATE = D('0.07')
CEN = D('0.0001')
STRESSES = (D('0'), D('.005'), D('.01'), D('.02'), D('.03'))
SCENARIOS = (
    ('whole_no_min', O, Z),
    ('whole_one_usd_min', O, O),
    ('fractional_no_min', D('.01'), Z),
    ('fractional_one_usd_min', D('.01'), O),
)


def fee(contracts, price):
    if contracts <= Z or price <= Z or price >= O:
        return Z
    raw = FEE_RATE * contracts * price * (O-price)
    return (raw/CEN).to_integral_value(rounding=ROUND_CEILING)*CEN


def simulate(m, ratio, step, minimum, adverse, with_taker_fee):
    """Same position deltas and min-order rules as v4; optional execution stresses."""
    source = defaultdict(lambda: Z)
    held = defaultdict(lambda: Z)
    buys = sells = fees = payout = buyq = sellq = Z
    n_buy = n_sell = caps = 0
    for r in m['rows']:
        tok, srcq, p, side = r['token'], r['q'], r['price'], r['side']
        source[tok] += srcq if side == 'BUY' else -srcq
        if source[tok] < -D('0.00000001'):
            raise AssertionError('Missing carry-in inventory: '+tok)
        if source[tok] < Z: source[tok] = Z
        target = v3.exact_floor(source[tok] * ratio, step)
        delta = target-held[tok]
        if side == 'BUY' and delta >= step and delta*p >= minimum:
            price = min(O, p+adverse)
            caps += int(price==O and p+adverse>O)
            held[tok] += delta
            buys += delta*price
            buyq += delta
            n_buy += 1
            if with_taker_fee: fees += fee(delta, price)
        elif side == 'SELL' and delta <= -step:
            qsell = min(held[tok], -delta)
            if qsell >= step and qsell*p >= minimum:
                price = max(Z, p-adverse)
                caps += int(price==Z and p-adverse<Z)
                held[tok] -= qsell
                sells += qsell*price
                sellq += qsell
                n_sell += 1
                if with_taker_fee: fees += fee(qsell, price)
    payout=sum((held[t]*m['payout'][m['outcome_map'][t]] for t in held), Z)
    return {'copy_pnl':sells+payout-buys-fees, 'copy_buy_spend':buys,
            'sell_proceeds':sells, 'settlement_payout':payout,
            'fees':fees,'buy_shares':buyq,'sell_shares':sellq,
            'buy_events':n_buy,'sell_events':n_sell,'price_caps':caps}


def aggregate(market_results):
    spend=sum((x['copy_buy_spend'] for x in market_results),Z)
    pnl=sum((x['copy_pnl'] for x in market_results),Z)
    fees=sum((x['fees'] for x in market_results),Z)
    pos=sorted((x['copy_pnl'] for x in market_results if x['copy_pnl']>Z),reverse=True)
    cats=defaultdict(lambda: {'markets':0,'pnl':Z,'spend':Z,'fees':Z})
    for x in market_results:
        z=cats[x['category']]
        z['markets']+=1;z['pnl']+=x['copy_pnl'];z['spend']+=x['copy_buy_spend'];z['fees']+=x['fees']
    return {'markets':len(market_results),'buy_spend_usd':str(spend),'net_pnl_usd':str(pnl),
            'roi_on_gross_buys_pct':str(100*pnl/spend) if spend else None,
            'fees_usd':str(fees),'fees_pct_gross_buys':str(100*fees/spend) if spend else None,
            'ex_top5_positive_market_pnl_usd':str(pnl-sum(pos[:5],Z)),
            'ex_top10_positive_market_pnl_usd':str(pnl-sum(pos[:10],Z)),
            'copy_buy_events':sum(x['buy_events'] for x in market_results),
            'copy_sell_events':sum(x['sell_events'] for x in market_results),
            'copy_buy_shares':str(sum((x['buy_shares'] for x in market_results),Z)),
            'copy_sell_shares':str(sum((x['sell_shares'] for x in market_results),Z)),
            'price_capped_events':sum(x['price_caps'] for x in market_results),
            'positive_pnl_markets':sum(x['copy_pnl']>Z for x in market_results),
            'category':{cat:{key:str(val) for key,val in d.items()} for cat,d in cats.items()}}


def run(states, end_exclusive, ratio, reference=None):
    out={}; detail=[]
    for scenario,step,minimum in SCENARIOS:
        out[scenario]={}
        for adverse in STRESSES:
            for taker in ((False,) if adverse==Z else ()) + (True,):
                label=('source_prints_no_fees' if not taker else ('taker_fee_only' if adverse==Z else 'taker_plus_'+str(adverse)+'_per_side'))
                rows=[]
                for cid,m in states.items():
                    r=simulate(m,ratio,step,minimum,adverse,taker)
                    last_90d=any(x['ts'] >= end_exclusive-90*86400 for x in m['rows'])
                    last=m['rows'][-1]
                    rows.append({'condition_id':cid,'title':last['title'],
                                 'category':last['category'],'last_90d':last_90d,
                                 'last_trade_ts':last['ts'],**r})
                full=aggregate(rows)
                recent=aggregate([r for r in rows if r['last_90d']])
                out[scenario][label]={'all':full,'markets_traded_last_90d':recent}
                if adverse in (Z,D('.01'),D('.02')) and (taker or adverse==Z) and scenario=='whole_no_min':
                    detail.extend([{'stress':label,**{k:str(v) for k,v in r.items()}} for r in rows])
                if reference and adverse==Z and not taker:
                    v=reference['scenarios'][scenario]['all']
                    targetp=D(v['copy_gross_pnl'])
                    targetb=D(v['copy_buy_spend'])
                    if abs(D(full['net_pnl_usd'])-targetp)>D('.000001') or abs(D(full['buy_spend_usd'])-targetb)>D('.000001'):
                        raise AssertionError(f'{scenario} does not reproduce V5 exactly: '+str((full['net_pnl_usd'],v['copy_gross_pnl'],full['buy_spend_usd'],v['copy_buy_spend'])))
    return out,detail


def self_test():
    m={'rows':[{'token':'a','side':'BUY','q':D('100'),'price':D('.40'),'ts':1,'title':'test','category':'crypto'},
               {'token':'a','side':'SELL','q':D('50'),'price':D('.60'),'ts':2,'title':'test','category':'crypto'}],
       'payout':[O,Z], 'outcome_map':{'a':0}}
    x=simulate(m,O,O,Z,Z,False)
    assert x['copy_pnl']==D('40') and x['buy_shares']==D('100') and x['sell_shares']==D('50'),x
    y=simulate(m,O,O,Z,D('.01'),False)
    assert y['copy_pnl']==D('38.5'),y
    z=simulate(m,O,O,Z,Z,True)
    expected=fee(D('100'),D('.4'))+fee(D('50'),D('.6'))
    assert z['fees']==expected and z['copy_pnl']==D('40')-expected,z
    assert fee(D('100'),D('.5'))==D('1.7500')
    print('PASS: copy BUY/SELL partial exit, settlement, adverse side prices and fee rounding; NO API')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--ratio',default='0.01',help='copy share ratio (0.01 is 1 percent)')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:return self_test()
    ratio=D(a.ratio)
    if not Z<ratio<=O: p.error('ratio must be in (0,1]')
    rows,meta,order=v3.read_source(a.cache)
    v3.market_categories(rows)
    bad,amb=v3.validate_token_inventory(rows)
    ix=v4.outcomes_by_token(a.cache/'albert1953_trades.jsonl')
    path=a.cache/'albert1953_v5_token_resolution_cache.json'
    if not path.exists():p.error('Missing V5 /v2/tokens cache; run V5 first')
    c=json.loads(path.read_text())
    if c.get('version') != v5.VERSION or c.get('wallet') != v3.WALLET:
        p.error('Token cache wrong version or wallet')
    states,fail=v5.make_states(rows,c['rows_by_condition'],bad,amb,ix)
    if len(states)<3000: p.error(f'Only {len(states)} verified markets, expected ~3625; refusing low-coverage audit')
    refpath=a.cache/('albert1953_settled_v5_ratio'+format(ratio,'f').replace('.','p')+'_report.json')
    ref=json.loads(refpath.read_text()) if refpath.exists() else None
    out,detail=run(states,meta['end_exclusive'],ratio,ref)
    suf='ratio'+format(ratio,'f').replace('.','p')
    outpath=a.cache/('albert1953_execution_stress_v6_'+suf+'.json')
    csvpath=a.cache/('albert1953_execution_stress_v6_'+suf+'_markets.csv')
    report={'wallet':v3.WALLET,'ratio':str(ratio),'source_fills':len(rows),'source_markets':len({r['condition'] for r in rows}),
            'verified_settled_markets':len(states),'verification_diagnostics':dict(fail),
            'model':'source-print price proxy with adverse price + optional general Kalshi taker fee; not Kalshi execution',
            'fee_formula':'ceil_0.0001(0.07 * contracts * P * (1-P)) on each modeled BUY and SELL',
            'scenarios':out}
    outpath.write_text(json.dumps(report,indent=2))
    if detail:
        with csvpath.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(detail[0]))
            w.writeheader();w.writerows(detail)
    print('ALBERT1953 V6 — SOURCE-PRICE STRESS, NO ORDERS, NO HISTORICAL QUOTES')
    print('Source fills:',len(rows),'settled markets:',len(states),'ratio:',ratio,'feed:',order)
    for name in ('whole_no_min','whole_one_usd_min','fractional_no_min','fractional_one_usd_min'):
        print('\n'+name)
        for lab, res in out[name].items():
            total=res['all'];recent=res['markets_traded_last_90d']
            print('  %-30s ALL %+10.2f (%+7.2f%%) fee %8.2f   90D %+8.2f   exTOP5 90D %+8.2f' %
                  (lab,float(total['net_pnl_usd']),float(total['roi_on_gross_buys_pct'] or 0),
                   float(total['fees_usd']),float(recent['net_pnl_usd']),float(recent['ex_top5_positive_market_pnl_usd'])))
    print('\nREPORT:',outpath)
    print('MARKET TRACE:',csvpath)
    print('NOTE: 90D measures markets with recent trades, including earlier fills in those same markets.')
    print('NO KALSHI QUOTES OR MARKET MATCHING; ALL RESULTS ARE HYPOTHETICAL SENSITIVITIES.')


if __name__ == '__main__':main()
