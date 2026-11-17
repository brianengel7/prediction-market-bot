#!/usr/bin/env python3
"""BreakTheBank V7 — proportional post-taker follower feasibility (OFFLINE).

Reuses completed V1 wallet feed and V3 resolution cache, via V5. No HTTP,
credentials, orders or source changes. Does NOT claim executable fills:
wallet print prices and adverse cents are stress proxies, not BBO quotations.

Tests a causally eligible sports cohort: all wallet fills at least L seconds after
first recorded taker fill in each market. Every selected BUY and SELL is copied
in wallet-share proportions, with sells capped by copied inventory.

Optional causal batching only emits an order after cumulative *observed* scaled
shares/notional clears a minimum within a max hold period; residual tiny fills
are dropped, not rounded up or shifted from other markets/outcomes/sides. Batch
execution price is the LAST observed wallet price, not a past average: quote
availability remains unverified. No future events are used to decide orders.

Examples:
 python3 -B src/sports-copy/breakthebank_v7_proportional_execution_readonly.py --self-test
 python3 -u -B src/sports-copy/breakthebank_v7_proportional_execution_readonly.py \
   --v1-input-dir ~/sports_copy_research/breakthebank_v1 \
   --v3-input-dir ~/sports_copy_research/breakthebank_v3 \
   --scales 0.0005,0.001,0.002,0.005 \
   --latencies 1,5,15,30,60,120 \
   --min-notional 1 --batch-window 60 \
   --out-json ~/sports_copy_research/breakthebank_v7/results.json

All quote-less results are diagnostics, NOT verified profit or an executable backtest.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def eligible(records, signal_age_s):
    """Select after a taker market signal with no future information in timing.

    A minimum 1 second protects against ambiguous same-second ordering.
    Historical public publication latency is NOT known and is not included.
    """
    if signal_age_s < 1:
        raise ValueError('signal_age_s must be >= 1; same-second ordering unknown')
    trigger = {}
    for r in records:
        if r['category']=='sports' and r['role']=='taker':
            cid=r['market']
            trigger[cid]=min(trigger.get(cid,r['ts']),r['ts'])
    selected=[r for r in records if r['category']=='sports' and
              r['market'] in trigger and r['ts'] >= trigger[r['market']]+signal_age_s]
    return sorted(selected,key=lambda r:(r['ts'],r['seq']))


def group_orders(rows, *, scale, min_notional, min_shares, max_window_s):
    """Causal same-outcome/side aggregation; keep fixed source scale.

    Max duration measured from FIRST pending trade, not last trade. Incomplete
    groups expire and cannot combine with future fills outside the window.
    Single-trade orders clear immediately if eligible. Never round quantities UP.
    """
    if not all(map(math.isfinite,(scale,min_notional,min_shares,max_window_s))):
        raise ValueError('Non-finite order parameters')
    if scale<=0 or min_notional<0 or min_shares<0 or max_window_s<0:
        raise ValueError('Invalid order parameters')
    buffers={}; emitted=[]; discarded=Counter(); disc_qty=0.0; disc_cost=0.0
    def drop(b):
        nonlocal disc_qty,disc_cost
        if b is not None:
            discarded['fills']+=b['fills']
            discarded['buffers']+=1
            disc_qty+=b['wallet_qty']*scale
            disc_cost+=b['source_notional']*scale
    for r in rows:
        k=(r['market'],r['token'],r['side'])
        b=buffers.get(k)
        if b is not None and r['ts']-b['first_ts']>max_window_s:
            drop(b);buffers.pop(k);b=None
        if b is None:
            b={'first_ts':r['ts'],'wallet_qty':0.,'source_notional':0.,'fills':0,'roles':Counter()}
            buffers[k]=b
        b['wallet_qty']+=r['wallet_qty']
        b['source_notional']+=r['wallet_qty']*r['wallet_price']
        b['fills']+=1;b['roles'][r['role']]+=1
        q=b['wallet_qty']*scale
        notional=b['source_notional']*scale
        if q+1e-10 >= min_shares and notional+1e-10>=min_notional:
            emitted.append({**r,
                'wallet_qty':b['wallet_qty'],
                # An order is placed ONLY now, not retroactively at historical VWAP.
                'wallet_price':r['wallet_price'],
                'source_vwap':b['source_notional']/b['wallet_qty'],
                'batch_source_fills':b['fills'],
                'batch_first_ts':b['first_ts'],
                'batch_roles':dict(b['roles']),
                'batch_wait_s':r['ts']-b['first_ts'],
            })
            buffers.pop(k)
    for b in buffers.values():drop(b)
    return emitted,{
        'selected_wallet_fills':len(rows),'copy_orders':len(emitted),
        'copied_source_fills_in_orders':sum(r['batch_source_fills'] for r in emitted),
        'discarded_wallet_fills':discarded['fills'],
        'discarded_buffers':discarded['buffers'],
        'discarded_scaled_shares':round(disc_qty,6),
        'discarded_scaled_wallet_notional':round(disc_cost,2),
        'copied_scaled_wallet_notional':round(sum(r['source_vwap']*r['wallet_qty']*scale for r in emitted),2),
        'copy_order_median_notional':round(_median([r['wallet_qty']*scale*r['wallet_price'] for r in emitted]),2) if emitted else None,
        'copy_order_max_batch_wait_s':max((r['batch_wait_s'] for r in emitted),default=None),
    }


def _median(xs):
    if not xs:return 0.
    xs.sort();n=len(xs)
    return xs[n//2] if n%2 else (xs[n//2-1]+xs[n//2])/2


def study(records,payouts,v1,v3,scale,lag,minimum,shares,batch_window,fee):
    cohort=eligible(records,lag)
    orders,batch=group_orders(cohort,scale=scale,min_notional=minimum,
                               min_shares=shares,max_window_s=batch_window)
    # All-zero filters is exactly V5 plus-one-second selection, at scaled source price.
    scenarios={}
    for shock in (0,2,5):
        coeff=0. if shock==0 else fee
        market=v3.replay(orders,payouts,v1,adverse_cents=shock,scale=scale,
                         fee_coeff=coeff,tie_policy='buy-first')
        a=v3.aggregate(market)
        exposed=[r for r in market if not r['scored']]
        scored=[r for r in market if r['scored'] and r['buy_usdc']>1e-8]
        # Residual $0 payout applies only to unresolved inventory, not all open
        # scored markets. It is an extreme downside bound, not a forecast.
        extreme_unresolved=sum(m['sell_usdc']-m['buy_usdc']-m['fees'] for m in exposed)
        scenarios[f'{shock}c']={
            **a,
            'hypothetical_fee_coefficient':coeff,
            'gross_buys_all_markets':round(sum(m['buy_usdc'] for m in market),2),
            'gross_sells_all_markets':round(sum(m['sell_usdc'] for m in market),2),
            'unresolved_markets':len(exposed),
            'unresolved_buy_exposure':round(sum(m['buy_usdc'] for m in exposed),2),
            'extreme_zero_payout_total_net':round(sum(m['pnl_usdc'] for m in scored)+extreme_unresolved,2),
            'scored_positive_markets':sum(m['pnl_usdc']>0 for m in scored),
            'scored_negative_markets':sum(m['pnl_usdc']<0 for m in scored),
            'source_hypothetical_order_count':len(orders),
        }
    return {'copy_scale':scale,'signal_age_s':lag,'min_notional':minimum,
            'min_shares':shares,'batch_window_s':batch_window,
            'batching':batch,'scenarios':scenarios}


def self_test():
    from types import SimpleNamespace
    from importlib.util import module_from_spec,spec_from_file_location
    base=Path(__file__).resolve().parent
    def load(name,path):
        sp=spec_from_file_location(name,base/path)
        if not sp or not sp.loader:raise RuntimeError('missing '+path)
        m=module_from_spec(sp);sp.loader.exec_module(m);return m
    v3=load('breakthebank_v3_v7_test','breakthebank_all_fills_v3_readonly.py')
    p='0x'+'f'*64
    def row(t,side,qty,price,role='maker',token='0'):
        return dict(market=p,token=p+'-'+token,idx=token,ts=t,seq=t,
                    side=side,wallet_qty=qty,wallet_price=price,category='sports',
                    role=role,title='Synthetic match')
    rec=[row(100,'BUY',100,.4,'taker'), row(101,'BUY',10,.40),
         row(102,'BUY',20,.50),row(103,'BUY',20,.60),
         row(104,'SELL',20,.70),row(105,'SELL',20,.80)]
    a=eligible(rec,1)
    assert len(a)==5 and len(eligible(rec,3))==3
    # 0.1 scale, 1.0 notional: BUY 10@.4+20@.5+20@.6 => at t=103
    # with 0.1 scale, cumulative notional $2.6; must have first order at t=103
    o,b=group_orders(a,scale=.1,min_notional=2.5,min_shares=1,max_window_s=60)
    assert len(o)==2 and o[0]['ts']==103 and o[0]['wallet_qty']==50 and o[0]['wallet_price']==.6, (o,b)
    assert o[1]['ts']==105 and o[1]['wallet_qty']==40 and o[1]['wallet_price']==.8
    assert b['copied_source_fills_in_orders']==5
    # Closing order cannot sell more than our copied inventory.
    v1=SimpleNamespace(payout_value=lambda cid,idx,r:r.get(cid,{}).get(idx))
    r=study(rec,{p:{'0':1.}},v1,v3,.1,1,2.5,1,60,.04)
    assert r['batching']['copy_orders']==2
    m=v3.replay(o,{p:{'0':1.}},v1,adverse_cents=0,scale=.1,fee_coeff=0.,tie_policy='buy-first')
    assert m[0]['unmatched_sells']==0 and abs(m[0]['pnl_usdc']-1.2)<1e-8, m
    # Separate tokens/sides cannot be mixed, no future batching across time gap.
    d,stats=group_orders([row(1,'BUY',1,.5),row(101,'BUY',1,.5)],scale=1,
                          min_notional=.8,min_shares=1,max_window_s=60)
    assert not d and stats['discarded_wallet_fills']==2
    # No future market taker trigger. Markets without takers produce no orders.
    foreign=[dict(row(101,'BUY',10,.5),market='0x'+'e'*64)]
    assert len(eligible(rec+foreign,1))==5
    # Zero minimum reproduces per-fill selected execution count (no future grouping).
    k,s=group_orders(a,scale=.001,min_notional=0,min_shares=0,max_window_s=60)
    assert len(k)==len(a) and all(x['batch_source_fills']==1 for x in k)
    print('PASS V7: proportional share quantities, causal batching, expiry, taker trigger,')
    print('         inventory-capped SELL, no size rounding, zero-min exact-fill baseline')
    print('OFFLINE: no network, trade keys or exchange orders')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--v1-input-dir',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v1')
    p.add_argument('--v3-input-dir',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v3')
    p.add_argument('--v1',type=Path,default=Path(__file__).with_name('breakthebank_copyability_audit.py'))
    p.add_argument('--v3',type=Path,default=Path(__file__).with_name('breakthebank_all_fills_v3_readonly.py'))
    p.add_argument('--v5',type=Path,default=Path(__file__).with_name('breakthebank_v5_taker_timing_readonly.py'))
    p.add_argument('--scales',default='.0005,.001,.002,.005',help='Wallet share ratios, separated by comma')
    p.add_argument('--latencies',default='1,5,15,30,60,120',help='Seconds after FIRST taker fill; excludes same-second')
    p.add_argument('--min-notional',type=float,default=1.,help='Scaled original wallet notional accumulated to emit a copy order; not an exchange minimum')
    p.add_argument('--min-shares',type=float,default=0.,help='Minimum copied shares to emit a batched order')
    p.add_argument('--batch-window',type=int,default=60,help='Max seconds to aggregate same market/token/side fills')
    p.add_argument('--fee-coeff',type=float,default=None,help='Hypothetical quadratic fee coefficient on each BUY/SELL')
    p.add_argument('--out-json',type=Path,help='Optional NEW JSON report outside repo; refuses overwrite')
    a=p.parse_args()
    if a.self_test:self_test();return
    # No credentials, no network: import previously validated offline readers.
    from importlib.util import spec_from_file_location,module_from_spec
    def load(path,name):
        if not path.is_file():raise FileNotFoundError(path)
        spec=spec_from_file_location(name,path)
        if spec is None or spec.loader is None:raise RuntimeError(path)
        m=module_from_spec(spec);spec.loader.exec_module(m);return m
    v1=load(a.v1,'v1_breakthebank_v7');v3=load(a.v3,'v3_breakthebank_v7');v5=load(a.v5,'v5_breakthebank_v7')
    prior,records,payouts=v5.require_sources(a.v1_input_dir.expanduser(),a.v3_input_dir.expanduser(),v1,v3)
    scales=[float(q) for q in a.scales.split(',')]
    ages=[int(q) for q in a.latencies.split(',')]
    if (not scales or not ages or len(scales)!=len(set(scales)) or len(ages)!=len(set(ages)) or
        any(not math.isfinite(x) or x<=0 for x in scales) or any(x<1 for x in ages)):
        p.error('scales must be unique >0 and latencies unique >=1')
    fee=float(prior['fee_coefficient_hypothetical'] if a.fee_coeff is None else a.fee_coeff)
    if not all(math.isfinite(x) for x in (fee,a.min_notional,a.min_shares)) or fee<0:
        p.error('Invalid fee/minimum parameters')
    # Source consistency check: V5 original precisely replays the source selection.
    original=v5.select(records,'sports_after_first_taker_plus_1s')
    selected=eligible(records,1)
    if sorted((r['ts'],r['seq']) for r in original)!=sorted((r['ts'],r['seq']) for r in selected):
        raise RuntimeError('V5 one-second sports cohort does not reconcile')
    baseline=v3.aggregate(v3.replay(selected,payouts,v1,adverse_cents=2,
                                     scale=float(prior['copy_scale']),fee_coeff=fee,tie_policy='buy-first'))
    print('\nBREAKTHEBANK V7 — PROPORTIONAL POST-TAKER COPY FEASIBILITY (OFFLINE)')
    print('Complete source:',len(records),'| baseline cohort:',len(selected),
          '| 1s 2c + modeled fee:',baseline['scored_roi_pct'],'%')
    print('Causality: after first TAKeR fill; earliest copy record >= taker timestamp + latency seconds.')
    print('IMPORTANT: elapsed seconds since WALLET TRADE are not measured public-notification delays.')
    print('Min order threshold:',a.min_notional,'USDC and',a.min_shares,'shares | batch horizon',a.batch_window,'seconds')
    print('Batch fills priced at MOST RECENT WALLET PRINT, then stress-shocked; NO book liquidity validated.')
    print('Return denominator is buy turnover plus modeled fees, NOT peak capital or account return.')
    print('\n Scale    Age(s) Orders  OrigFills  CopyFills Scored  0c gross   2c+fee   5c+fee  UnscoredBuy  ZeroPayout2c')
    result=[]
    for scale in scales:
        for age in ages:
            entry=study(records,payouts,v1,v3,scale,age,a.min_notional,
                        a.min_shares,a.batch_window,fee)
            batch=entry['batching'];s=entry['scenarios']
            def fmt(x):return f'{x:+7.2f}%' if x is not None else '    N/A'
            print(f'{scale:8.5f} {age:6d} {batch["copy_orders"]:6,d}  '
                  f'{batch["selected_wallet_fills"]:8,d}  {batch["copied_source_fills_in_orders"]:9,d} '
                  f'{s["2c"]["markets_scored"]:6,d} '
                  f'{fmt(s["0c"]["scored_roi_pct"]):>9s} {fmt(s["2c"]["scored_roi_pct"]):>9s} '
                  f'{fmt(s["5c"]["scored_roi_pct"]):>9s} '
                  f'${s["2c"]["unresolved_buy_exposure"]:11,.0f}  '
                  f'${s["2c"]["extreme_zero_payout_total_net"]:+12,.2f}')
            result.append(entry)
    print('\nINFO: all highest-profit markets retained; no winner exclusions or reweighting.')
    print('Sells are capped by acquired inventory; missing fills/expired tiny orders are not replaced.')
    print('Unknown settlement markets are excluded from scored ROI, disclosed via UnscoredBuy and zero-payout stress.')
    print('Observed maker prices are not guaranteed copyable. Fixed price shocks DO NOT validate execution.')
    print('NO API, NETWORK, PRIVATE KEYS, ORDER PLACEMENT, OR REPOSITORY MODIFICATIONS')
    if a.out_json:
        dest=a.out_json.expanduser().resolve()
        if str(dest).startswith('/workspaces/') or dest.exists():
            raise RuntimeError('Output must be NEW and outside repo')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps({'wallet':v1.WALLET,'source_fills':len(records),
            'method':'Wallet-print proportional batched counterfactual; NOT historical executable quotes',
            'cohort_reconciliation_2c':baseline,'results':result},indent=2),encoding='utf-8')
        print('NEW REPORT:',dest)

if __name__=='__main__':main()
