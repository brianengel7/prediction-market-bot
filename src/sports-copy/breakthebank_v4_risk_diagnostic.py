#!/usr/bin/env python3
"""BreakTheBank V4: offline diagnostic of V3 sample selection and P&L concentration.

Pure read-only: only reads V3 report.json and market_diagnostic.json.
No network, keys, orders, order book fabrication, or file writes.

Use: python3 -B src/sports-copy/breakthebank_v4_risk_diagnostic.py \
         --input-dir ~/sports_copy_research/breakthebank_v3
"""
from __future__ import annotations

import argparse
import json
import math
from statistics import median
from collections import Counter, defaultdict
from pathlib import Path


def cash(x):
    return f'${x:+,.2f}'


def percent(n, d):
    return 100 * n / d if d > 0 else None


def pct_format(v):
    return 'N/A' if v is None else f'{v:+.3f}%'


def get_scenario(report, cents, fee_label, tie='buy-first'):
    matches=[r for r in report['scenarios'][tie]
             if r['price_worsening_cents_each_leg']==cents and r['fee_model']==fee_label]
    if len(matches)!=1:
        raise ValueError(f'Expected exactly one {tie}/{cents}/{fee_label} scenario')
    return matches[0]


def analyze_markets(markets):
    if not markets or len({m['market'] for m in markets})!=len(markets):
        raise ValueError('Empty or duplicate markets')
    scored=[m for m in markets if m['scored'] and m['buy_usdc']>1e-8]
    unscored=[m for m in markets if not m['scored']]
    pnl=sum(float(m['pnl_usdc']) for m in scored)
    gross_spend=sum(float(m['buy_usdc']) for m in scored)
    unscored_buys=sum(float(m['buy_usdc']) for m in unscored)
    unscored_sells=sum(float(m['sell_usdc']) for m in unscored)
    unscored_remaining=sum(float(m['remaining_shares']) for m in unscored)
    ranked=sorted(scored,key=lambda m:float(m['pnl_usdc']),reverse=True)
    positive=[m for m in ranked if float(m['pnl_usdc'])>0]
    stripped=[]
    for k in (1,5,10,20):
        excluded={m['market'] for m in positive[:k]}
        rest=[m for m in scored if m['market'] not in excluded]
        remaining_cost=sum(float(m['buy_usdc']) for m in rest)
        remaining_pnl=sum(float(m['pnl_usdc']) for m in rest)
        stripped.append({'exclude_top_positive':k, 'scored_markets':len(rest),
                         'pnl_usdc':remaining_pnl, 'roi_pct':percent(remaining_pnl,remaining_cost)})
    maker_only=[m for m in scored if int(m.get('taker_fills',0))==0 and int(m.get('ambiguous_role_fills',0))==0]
    any_taker=[m for m in scored if int(m.get('taker_fills',0))>0]
    any_ambiguous=[m for m in scored if int(m.get('ambiguous_role_fills',0))>0]
    sectors=defaultdict(lambda:{'scored':0,'unscored':0,'buy_usdc_scored':0.,'pnl_usdc_scored':0.,'buy_usdc_unscored':0.,'sell_usdc_unscored':0.,'unmatched_sell_fills':0})
    for m in markets:
        s=sectors[m['category']]
        s['unmatched_sell_fills']+=int(m['unmatched_sells'])
        if m['scored'] and m['buy_usdc']>1e-8:
            s['scored']+=1
            s['buy_usdc_scored']+=float(m['buy_usdc'])
            s['pnl_usdc_scored']+=float(m['pnl_usdc'])
        elif not m['scored']:
            s['unscored']+=1
            s['buy_usdc_unscored']+=float(m['buy_usdc'])
            s['sell_usdc_unscored']+=float(m['sell_usdc'])
    return {
        'total_markets':len(markets), 'scored':len(scored),'unscored':len(unscored),
        'scored_pnl':pnl,'scored_buy_cost':gross_spend,'scored_roi_pct':percent(pnl,gross_spend),
        'unscored_buy_cost':unscored_buys,'unscored_sell_proceeds':unscored_sells,
        'unscored_remaining_shares':unscored_remaining,
        'zero_payout_all_unscored_gross_pnl':pnl+unscored_sells-unscored_buys,
        'zero_payout_all_unscored_gross_roi_pct':percent(pnl+unscored_sells-unscored_buys,gross_spend+unscored_buys),
        'top_10_profit_contribution':sum(float(m['pnl_usdc']) for m in ranked[:10]),
        'top_10_profit_share_of_net_pct':percent(sum(float(m['pnl_usdc']) for m in ranked[:10]),pnl),
        'top_10_positive':[{k:m[k] for k in ('title','category','pnl_usdc','buy_usdc')} for m in ranked[:10]],
        'bottom_10':[{k:m[k] for k in ('title','category','pnl_usdc','buy_usdc')} for m in sorted((m for m in scored if float(m['pnl_usdc'])<0),key=lambda z:float(z['pnl_usdc']))[:10]],
        'stripped':stripped,
        'maker_only':{'markets':len(maker_only),'pnl':sum(float(m['pnl_usdc']) for m in maker_only), 'spend':sum(float(m['buy_usdc']) for m in maker_only)},
        'any_taker':{'markets':len(any_taker),'pnl':sum(float(m['pnl_usdc']) for m in any_taker),'spend':sum(float(m['buy_usdc']) for m in any_taker)},
        'any_ambiguous':{'markets':len(any_ambiguous)},
        'sectors':dict(sectors),
    }


def test():
    def row(name,pnl,buys,scored=True,role=0,sector='sports',sells=0.):
        return dict(market=name,title=name,category=sector,scored=scored,
                    buy_usdc=buys,sell_usdc=sells,pnl_usdc=pnl if scored else None,
                    remaining_shares=5 if not scored else 0,
                    taker_fills=role,ambiguous_role_fills=0,unmatched_sells=0)
    x=analyze_markets([row('a',10,100),row('b',-5,50,role=1),row('c',None,20,False,sells=5)])
    assert x['scored']==2 and x['unscored']==1
    assert abs(x['scored_pnl']-5)<1e-9
    assert abs(x['zero_payout_all_unscored_gross_pnl']+10)<1e-9
    assert x['maker_only']['markets']==1 and x['any_taker']['markets']==1
    assert abs(x['stripped'][0]['pnl_usdc']+5)<1e-9
    print('PASS V4 synthetic: unresolved zero-pay stress, P&L concentration, cohort attribution')
    print('READ ONLY — no network, source edits, keys, or orders')


def fill_size_diagnostic(v1_path, scale):
    """Only record notional from V1 public cache, not proof of exchange minima."""
    p = v1_path / 'all_trades.jsonl'
    if not p.is_file():
        return None
    arrs = defaultdict(list)
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            side = str(r.get('side', '')).upper()
            if side not in ('BUY', 'SELL'):
                raise ValueError('Unexpected trade side in public source')
            value = float(r['price']) * float(r['size']) * scale
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid source notional')
            arrs[side].append(value)
    summary = {}
    for side, amounts in arrs.items():
        n = len(amounts)
        summary[side] = {
            'fills': n, 'total_usdc': sum(amounts), 'median_usdc': median(amounts),
            'under_1_usdc_pct': percent(sum(v < 1 for v in amounts), n),
            'under_5_usdc_pct': percent(sum(v < 5 for v in amounts), n),
            'under_10_usdc_pct': percent(sum(v < 10 for v in amounts), n),
        }
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path, default=Path.home()/'sports_copy_research'/'breakthebank_v3')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--v1-input-dir',type=Path,default=Path.home()/'sports_copy_research'/'breakthebank_v1')
    args=p.parse_args()
    if args.self_test:
        test();return
    path=args.input_dir.expanduser()
    report=json.loads((path/'report.json').read_text())
    markets=json.loads((path/'market_diagnostic.json').read_text())
    if not isinstance(markets,list) or len(markets)!=int(report['markets']):
        raise ValueError('V3 diagnostic rows do not agree with report market count')
    if report['copy_scale']<=0:
        raise ValueError('Invalid V3 scale')
    if len({m['market'] for m in markets}) != int(report['markets']):
        raise ValueError('Invalid market uniqueness')
    s0=get_scenario(report,0,'zero_fee')['flat_start_follower']
    s2=get_scenario(report,2,'hypothetical_fee')['flat_start_follower']
    s5=get_scenario(report,5,'hypothetical_fee')['flat_start_follower']
    x=analyze_markets(markets)
    if x['scored']!=s0['markets_scored'] or abs(x['scored_pnl']-s0['scored_net_pnl'])>0.011:
        raise RuntimeError('V4 reconstruction does not reconcile with V3')

    print('\nBREAKTHEBANK V4 — SELECTION & CONCENTRATION DIAGNOSTIC')
    print('READ ONLY; NO NEW TRADES OR BOOK DATA; 0c scenario market diagnostics')
    print(f"All markets: {x['total_markets']} | scored {x['scored']} | unscored {x['unscored']}")
    print('Scored 0c gross:',cash(x['scored_pnl']),pct_format(x['scored_roi_pct']))
    print('Scored 2c worse + modeled fees:',cash(s2['scored_net_pnl']),pct_format(s2['scored_roi_pct']))
    print('Scored 5c worse + modeled fees:',cash(s5['scored_net_pnl']),pct_format(s5['scored_roi_pct']))
    print(f"Unscored positions' buy spend: {cash(x['unscored_buy_cost'])}; sell proceeds: {cash(x['unscored_sell_proceeds'])}")
    print('EXTREME downside if every unscored residual paid $0, zero fees (not a forecast):')
    print('  all-market gross:',cash(x['zero_payout_all_unscored_gross_pnl']),pct_format(x['zero_payout_all_unscored_gross_roi_pct']))
    print('  For hedged both-token positions, simultaneous zero payouts are impossible, so this is deliberately conservative.')
    print('\nProfit concentration within scored markets at source execution prices:')
    print('  10 best markets contributed:',cash(x['top_10_profit_contribution']),
          f"({x['top_10_profit_share_of_net_pct']:.1f}% of scored net P&L)" if x['top_10_profit_share_of_net_pct'] is not None else 'N/A')
    for t in x['stripped']:
        print(f"  Excluding {t['exclude_top_positive']:>2} best positive markets: {t['scored_markets']:>3} scored, "+
              f"PnL={cash(t['pnl_usdc'])}, ROI={pct_format(t['roi_pct'])}")
    print('\nRole split by MARKET, not actual individual fill profitability:')
    for name, label in [('maker_only','Markets without any taker/ambiguous fills'),('any_taker','Markets with at least one taker fill')]:
        c=x[name]
        print(f"  {label}: {c['markets']} markets, {cash(c['pnl'])} PnL, ROI {pct_format(percent(c['pnl'],c['spend']))}")
    print('  Markets with ambiguous maker/taker identity:',x['any_ambiguous']['markets'])
    size = fill_size_diagnostic(args.v1_input_dir.expanduser(),float(report['copy_scale']))
    if size:
        print('\nCopied individual fill sizes at the chosen scale (NO platform order minimum asserted):')
        for side,c in sorted(size.items()):
            print(f"  {side:4s} fills={c['fills']:,} total={cash(c['total_usdc'])} median={cash(c['median_usdc'])} "
                  f"under_$1={c['under_1_usdc_pct']:.1f}% under_$5={c['under_5_usdc_pct']:.1f}% "
                  f"under_$10={c['under_10_usdc_pct']:.1f}%")
    print('\nCategory coverage; exclude unscored from all reported cohort ROI:')
    for name,c in sorted(x['sectors'].items()):
        print(f"  {name:9s} scored={c['scored']:3d} unscored={c['unscored']:3d} "+
              f"scored_PnL={cash(c['pnl_usdc_scored']):>14} "+
              f"unscored_buy_spend={cash(c['buy_usdc_unscored']):>14} "+
              f"unmatched_sell_fills={c['unmatched_sell_fills']:,}")
    print('\nTOP 5 WINNERS')
    for m in (m for m in x['top_10_positive'][:5] if float(m['pnl_usdc'])>0):
        print(' ',cash(m['pnl_usdc']),m['category'],m['title'][:88])
    print('BOTTOM 5 LOSERS')
    for m in x['bottom_10'][:5]:
        print(' ',cash(m['pnl_usdc']),m['category'],m['title'][:88])
    print('\nNO EXECUTABLE FILL ASSERTION. V4 does not model public signal delay or historical order book liquidity.')
    print('No network, no writes, no orders.')


if __name__=='__main__':
    main()
