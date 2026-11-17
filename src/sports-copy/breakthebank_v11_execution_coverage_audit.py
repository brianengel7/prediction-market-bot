#!/usr/bin/env python3
"""Read-only, no-network audit of a completed V8/V10 BreakTheBank execution replay.
Compare identical simulated executed share quantities at actual recorded BBO and
source wallet prints; quantify market-level missing buy/sell book coverage.
Does not attempt to supply a complete population ROI or reconstruct unavailable books.
"""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import runpy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-script', type=Path,
                    default=Path('src/sports-copy/breakthebank_v8_book_replay_readonly.py'))
    ap.add_argument('--root', type=Path, default=Path.home()/'sports_copy_research')
    ap.add_argument('--quotes', type=Path, default=None)
    ap.add_argument('--scale', type=float, default=.001)
    ap.add_argument('--delay', type=float, default=5.)
    ap.add_argument('--max-cents', type=float, default=2.)
    ap.add_argument('--max-book-age', type=float, default=10.)
    ap.add_argument('--fee-coeff', type=float, default=.04)
    a=ap.parse_args()
    root=a.root.expanduser()
    args = SimpleNamespace(v1_input_dir=root/'breakthebank_v1',
                           v3_input_dir=root/'breakthebank_v3',
                           scale=a.scale,min_age=1,min_notional=1.,
                           min_shares=0.,batch_window=60)
    mod = runpy.run_path(str(a.repo_script))
    orders, batching, v1, v3, payouts = mod['source_orders'](args)
    qfile=a.quotes or root/'breakthebank_v10/bbo_repaired.csv'
    quotes=mod['read_quotes'](qfile.expanduser())
    recorded={}
    original_replay=v3.replay

    def capture(executions,resolutions,model,**kwargs):
        recorded['fills']=list(executions)
        result=original_replay(executions,resolutions,model,**kwargs)
        recorded['markets']=result
        return result
    v3.replay=capture
    try:
        report=mod['apply_books'](orders,quotes,scale=a.scale,delay=a.delay,
                    max_worse_cents=a.max_cents,max_quote_age=a.max_book_age,
                    fee_coeff=a.fee_coeff,v1=v1,v3=v3,payouts=payouts)
    finally:
        v3.replay=original_replay
    fills=recorded.get('fills',[])
    markets=recorded.get('markets',[])
    scored=[m for m in markets if m['scored'] and m['buy_usdc']>1e-8]
    scored_ids={m['market'] for m in scored}
    assert len(fills)==report['orders_filled_or_partially_filled']
    assert len(scored)==report['scored_only']['markets_scored']
    assert abs(sum(m['pnl_usdc'] for m in scored)-report['scored_only']['scored_net_pnl'])<.011

    intentions=Counter(r['side'] for r in orders)
    executed=Counter(r['side'] for r in fills)
    intent_by_market=Counter(r['market'] for r in orders)
    fresh_by_market=Counter()
    side_fresh=Counter()
    side_quote=Counter()
    for r in orders:
        token=r['token']; market=r['market'];side=r['side']
        series=quotes.get(token)
        if series is None:
            continue
        times, books=series
        t=r['ts']+a.delay
        ix=bisect_right(times,t)-1
        if ix<0 or t-times[ix]>a.max_book_age+1e-8:
            continue
        fresh_by_market[market]+=1
        side_fresh[side]+=1
        bid,bs,ask,ask_sz=books[ix]
        price,sz=(ask,ask_sz) if side=='BUY' else (bid,bs)
        if price is not None and sz is not None and sz>0:
            side_quote[side]+=1

    complete=[m for m in scored if fresh_by_market[m['market']]==intent_by_market[m['market']]]
    intent_sells_by_market=Counter(r['market'] for r in orders if r['side']=='SELL')
    fill_sells_by_market=Counter(r['market'] for r in fills if r['side']=='SELL')
    scored_missing_sells=[m for m in scored if intent_sells_by_market[m['market']]>0 and fill_sells_by_market[m['market']]==0]
    print('BREAKTHEBANK V11 — READ-ONLY EXECUTION AND MISSINGNESS AUDIT')
    print(f'Delay {a.delay:g}s | max adverse {a.max_cents:g}c | scale {a.scale:g}')
    print('Intended orders:',len(orders),' | filled/partial:',len(fills),
          '| scored markets:',len(scored),'| scored ROI:',report['scored_only']['scored_roi_pct'],'%')
    print('Intent orders by side:',dict(intentions))
    print('Orders with fresh BBO by side (before inventory/price gates):',dict(side_fresh))
    print('Fresh BBO with nonempty relevant side:',dict(side_quote))
    print('Actually filled orders by side:',dict(executed))
    print('Executed price worsening, share-weighted cents:',report['mean_executed_price_worsening_cents_share_weighted'])
    print('Rejection counters:',report['rejects'])
    print('Scored cohort: market buy cost=',round(sum(m['buy_usdc'] for m in scored),2),
          'sell proceeds=',round(sum(m['sell_usdc'] for m in scored),2),
          'settlement payouts=',round(sum(m['payout_usdc'] or 0 for m in scored),2),
          'fees=',round(sum(m['fees'] for m in scored),2),
          'net PnL=',round(sum(m['pnl_usdc'] for m in scored),2))
    print('Scored markets whose EVERY intent has a fresh historical BBO:',len(complete),'/',len(scored))
    if complete:
        cost=sum(m['buy_usdc']+m['fees'] for m in complete)
        pnl=sum(m['pnl_usdc'] for m in complete)
        print('  Complete-BBO scored subset PnL:',round(pnl,2),
              'ROI:',round(100*pnl/cost,3) if cost else None,'%')
    print('Scored markets with an intended SELL but ZERO filled sells:',len(scored_missing_sells))
    print('  Their combined net PnL:',round(sum(m['pnl_usdc'] for m in scored_missing_sells),2))
    print('Scored market orders, raw fresh BBO count:',sum(intent_by_market[k] for k in scored_ids),
          sum(fresh_by_market[k] for k in scored_ids))

    # Reprice IDENTICAL filled quantities at their original wallet batch print.
    # This is not a tradable counterfactual; it diagnoses whether returns came
    # from sample selection vs observed book prices.
    back_to_wallet=[]
    for e in fills:
        original=orders[e['seq']]
        assert e['market']==original['market'] and e['token']==original['token'] and e['side']==original['side']
        back_to_wallet.append({**e,'wallet_price':original['wallet_price']})
    if back_to_wallet:
        m=original_replay(back_to_wallet,payouts,v1,adverse_cents=0,
                scale=a.scale,fee_coeff=a.fee_coeff,tie_policy='buy-first')
        cmp=v3.aggregate(m)
        print('EXACT SAME executed quantities at wallet batch prints: ROI=',cmp['scored_roi_pct'],
              '% PnL=',cmp['scored_net_pnl'],
              'markets=',cmp['markets_scored'])
        assert cmp['markets_scored']==len(scored)
    print('Top 5 scored market contributions (no markets excluded):')
    for m in sorted(scored,key=lambda x:x['pnl_usdc'],reverse=True)[:5]:
        print('  +/− $',round(m['pnl_usdc'],2),'|',m['title'][:73],
              '| fresh intents',f"{fresh_by_market[m['market']]}/{intent_by_market[m['market']]}")
    print('BOTTOM 5 scored market contributions:')
    for m in sorted(scored,key=lambda x:x['pnl_usdc'])[:5]:
        print('  +/− $',round(m['pnl_usdc'],2),'|',m['title'][:73],
              '| fresh intents',f"{fresh_by_market[m['market']]}/{intent_by_market[m['market']]}")
    print('LIMITATIONS: Book presence selection is not random; fees hypothetical, no publication lag or queue proof.')
    print('READ ONLY — NO NETWORK, WRITES, PRIVATE KEYS OR ORDERS')

if __name__=='__main__':main()
