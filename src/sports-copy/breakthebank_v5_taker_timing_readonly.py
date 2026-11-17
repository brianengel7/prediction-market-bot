#!/usr/bin/env python3
"""BreakTheBank V5: offline, no-lookahead taker-trigger and size feasibility audit.

Uses complete V1 wallet trade cache and V3 resolution cache. Reuses V3's exact
inventory and settlement accounting, with a flat starting inventory and sells
capped by inventory. No historical executable quotes, real latency, or orders.

Key test: distinguish a hindsight market label ('eventually had a taker fill')
from a causal public signal ('a taker fill has already been seen').

Only reads files, does not write or connect to the network unless --out-json
is supplied, in which case writes one NEW JSON file outside the Git repo.

  python3 -B src/sports-copy/breakthebank_v5_taker_timing_readonly.py --self-test
  python3 -u -B src/sports-copy/breakthebank_v5_taker_timing_readonly.py \
      --v1-input-dir ~/sports_copy_research/breakthebank_v1 \
      --v3-input-dir ~/sports_copy_research/breakthebank_v3
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path


def load_module(path, name):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required Python script: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require_sources(v1_dir, v3_dir, v1, v3):
    meta = json.loads((v1_dir/'source_meta.json').read_text(encoding='utf-8'))
    old = json.loads((v3_dir/'report.json').read_text(encoding='utf-8'))
    if meta.get('wallet','').lower() != v1.WALLET.lower() or old.get('wallet','').lower() != v1.WALLET.lower():
        raise ValueError('Wallet mismatch between V1/V3 sources')
    if not all(meta.get('feeds',{}).get(n,{}).get('complete') for n in ('all','taker')):
        raise ValueError('Incomplete V1 feed; not scoring')
    if int(old['all_fills']) != int(meta['feeds']['all']['rows']) or int(old['taker_feed_rows']) != int(meta['feeds']['taker']['rows']):
        raise ValueError('V1 feed counts differ from V3; not scoring')
    if int(old.get('source_window_start',-1)) != int(meta['start']) or int(old.get('source_window_end_exclusive',-1)) != int(meta['end']):
        raise ValueError('V1/V3 history windows differ; not scoring')
    if not math.isfinite(float(old['copy_scale'])) or float(old['copy_scale']) <= 0:
        raise ValueError('Invalid copy scale')
    a=v1.read_rows(v1_dir/'all_trades.jsonl')
    t=v1.read_rows(v1_dir/'taker_trades.jsonl')
    if len(a)!=int(old['all_fills']) or len(t)!=int(old['taker_feed_rows']):
        raise ValueError('V1 cache content mismatch')
    records=v3.prepare(a,v3.assign_roles(a,t),v1)
    resolutions=v1.resolutions_for([],v3_dir/'resolutions.jsonl',offline=True)
    if len({r['market'] for r in records})!=int(old['markets']):
        raise ValueError('V3 market counts not reconciled')
    return old,records,resolutions


def select(records, strategy):
    """Selections using only information known by the time of each wallet fill.

    For first-taker triggers, no later trades from a market are included until
    a taker fill has actually occurred. Same-second maker fills are dropped,
    because public ordering inside a second is unobservable here.
    """
    if strategy in ('all','sports_all','maker_only','taker_only','sports_taker_only'):
        return [r for r in records if (
            (strategy!='sports_all' and strategy!='sports_taker_only' or r['category']=='sports') and
            (strategy!='maker_only' or r['role']=='maker') and
            (strategy!='taker_only' and strategy!='sports_taker_only' or r['role']=='taker')
        )]
    if strategy in ('after_first_taker','sports_after_first_taker','sports_after_first_taker_plus_1s'):
        first = {}
        for r in records:
            if r['role']=='taker':
                first[r['market']] = min(r['ts'],first.get(r['market'],r['ts']))
        selected=[]
        for r in records:
            if strategy.startswith('sports_') and r['category']!='sports':
                continue
            trigger=first.get(r['market'])
            if trigger is None:
                continue
            if strategy.endswith('plus_1s'):
                if r['ts']>trigger:
                    selected.append(r)
            elif r['ts']>trigger or (r['ts']==trigger and r['role']=='taker'):
                selected.append(r)
        return selected
    if strategy == 'sports_taker_buys_all_later_sells':
        # A taker BUY is an entry signal. Keep BUYs that are themselves taker;
        # afterwards allow all sells on the same token, even if maker. These
        # seller quotes are NOT presumed executable, and that caveat matters.
        eligible={}
        for r in records:
            if r['category']=='sports' and r['role']=='taker' and r['side']=='BUY':
                key=(r['market'],r['token'])
                eligible[key]=min(r['ts'],eligible.get(key,r['ts']))
        return [r for r in records if r['category']=='sports' and (
            (r['role']=='taker' and r['side']=='BUY') or
            (r['side']=='SELL' and (r['market'],r['token']) in eligible and r['ts']>eligible[(r['market'],r['token'])])
        )]
    raise ValueError(f'Unknown selection: {strategy}')


def scenario(selected, resolutions, v1, v3, *, scale, shock, fee_coeff, min_fill_usdc=0):
    kept=[]; filtered=Counter(); total_notional=0.
    for r in selected:
        notional=r['wallet_price']*r['wallet_qty']*scale
        if notional+1e-12<min_fill_usdc:
            filtered[r['side']]+=1
            continue
        kept.append(r)
        total_notional+=notional
    ms=v3.replay(kept,resolutions,v1,adverse_cents=shock,scale=scale,
                 fee_coeff=fee_coeff,tie_policy='buy-first')
    a=v3.aggregate(ms)
    a.update({'selected_fills':len(selected),'kept_fills':len(kept),
              'filtered_small_fills':dict(filtered),'total_selected_notional_wallet_price':round(total_notional,2),
              'no_buy_markets':sum(m['buy_usdc']<=1e-8 for m in ms),
              'markets_in_replay':len(ms),
              'unscored_residual_markets':sum(not m['scored'] for m in ms),
              'remaining_unscored_buy_usdc':round(sum(m['buy_usdc'] for m in ms if not m['scored']),2),
              'remaining_unscored_sell_usdc':round(sum(m['sell_usdc'] for m in ms if not m['scored']),2),
              'maker_fills_selected':sum(r['role']=='maker' for r in kept),
              'taker_fills_selected':sum(r['role']=='taker' for r in kept),
              'ambiguous_fills_selected':sum(r['role']=='ambiguous' for r in kept)})
    return a,ms


def test():
    from types import SimpleNamespace
    v1=SimpleNamespace(payout_value=lambda cid,idx,p: p.get(cid,{}).get(idx),
                       CID=__import__('re').compile(r'^0x[a-f0-9]{64}$'),cat=lambda r:'sports')
    v3=load_module(Path(__file__).with_name('breakthebank_all_fills_v3_readonly.py'),'v3_for_test')
    a='0x'+'a'*64;b='0x'+'b'*64
    def mk(market,ts,side,price,qty,role):
        return dict(market=market,ts=ts,seq=ts,side=side,wallet_price=price,
                    wallet_qty=qty,category='sports',title='Test sports',role=role,
                    token=f'{market}-0',idx='0')
    rec=[mk(a,1,'BUY',.4,2,'maker'),mk(a,2,'BUY',.5,2,'taker'),
         mk(a,3,'SELL',.6,2,'maker'),mk(b,1,'BUY',.7,3,'maker')]
    payouts={a:{'0':1.},b:{'0':0.}}
    assert len(select(rec,'sports_after_first_taker'))==2
    assert len(select(rec,'sports_after_first_taker_plus_1s'))==1
    assert len(select(rec,'sports_taker_only'))==1
    assert len(select(rec,'sports_taker_buys_all_later_sells'))==2
    ans,_=scenario(select(rec,'sports_after_first_taker'),payouts,v1,v3,scale=1,shock=0,fee_coeff=0)
    assert ans['markets_scored']==1 and abs(ans['scored_net_pnl']-.2)<1e-10, ans
    # Simulated trading only after TAKER can't take the earlier maker BUY.
    ans,_=scenario(select(rec,'sports_taker_only'),payouts,v1,v3,scale=1,shock=0,fee_coeff=0)
    assert ans['markets_scored']==1 and abs(ans['scored_net_pnl']-1.)<1e-10,ans
    ans,_=scenario(rec,payouts,v1,v3,scale=1,shock=0,fee_coeff=0,min_fill_usdc=1.)
    assert ans['filtered_small_fills']=={'BUY':1},ans
    print('PASS V5: no-future taker activation, same-second maker exclusion, taker-only,')
    print('         settlement, sell inventory cap and minimum scaled-fill filtering')
    print('READ ONLY: no network, keys, orders, repository changes')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--v1-input-dir',type=Path,default=Path.home()/'sports_copy_research/breakthebank_v1')
    p.add_argument('--v3-input-dir',type=Path,default=Path.home()/'sports_copy_research/breakthebank_v3')
    p.add_argument('--v1',type=Path,default=Path(__file__).with_name('breakthebank_copyability_audit.py'))
    p.add_argument('--v3',type=Path,default=Path(__file__).with_name('breakthebank_all_fills_v3_readonly.py'))
    p.add_argument('--out-json',type=Path,help='Optional NEW output outside repo; refuses overwrite')
    args=p.parse_args()
    if args.self_test:
        test();return
    v1=load_module(args.v1,'breakthebank_v1_v5')
    v3=load_module(args.v3,'breakthebank_v3_v5')
    prior,records,payouts=require_sources(args.v1_input_dir.expanduser(),args.v3_input_dir.expanduser(),v1,v3)
    scale=float(prior['copy_scale']);fee=float(prior['fee_coefficient_hypothetical'])
    if not scale>0 or not math.isfinite(fee) or fee<0:
        raise ValueError('Invalid prior scale or fee')
    basic,_=scenario(records,payouts,v1,v3,scale=scale,shock=0,fee_coeff=0)
    v3baseline=next(s['flat_start_follower'] for s in prior['scenarios']['buy-first']
                    if s['price_worsening_cents_each_leg']==0 and s['fee_model']=='zero_fee')
    if basic['markets_scored']!=v3baseline['markets_scored'] or abs(basic['scored_net_pnl']-v3baseline['scored_net_pnl'])>.02:
        raise RuntimeError('V3 baseline does not reconcile; abort')
    print('\nBREAKTHEBANK V5 — TRADE-ROLE / NO-LOOKAHEAD / TINY-FILL DIAGNOSTIC')
    print('Entire source:',len(records),'wallet fills;','taker:',sum(r['role']=='taker' for r in records),
          '| scaled ratio:',scale)
    print('Baseline reconciliation: PASS — V3 scored',basic['markets_scored'],'P&L',f"${basic['scored_net_pnl']:+,.2f}")
    strategies=[('all','All original fills'),('sports_all','Sports all fills'),
                ('maker_only','Maker fills only'),('taker_only','Taker fills only'),
                ('sports_taker_only','Sports taker fills only'),
                ('after_first_taker','All fills after first taker (all sectors)'),
                ('sports_after_first_taker','Sports fills AFTER first taker'),
                ('sports_after_first_taker_plus_1s','Sports fills >=1s after first taker'),
                ('sports_taker_buys_all_later_sells','Sports taker BUYs + subsequent SELLs')]
    data={}
    print('\nAll returns: recorded wallet execution time/prices, not actually fillable copy quotes.')
    print('All scenarios start flat. Unknown settlements remain EXCLUDED, NOT counted as wins or losses.')
    print('\n'+f"{'Strategy':42s} {'Fills':>7s} {'Markets':>7s} {'Scored':>6s} {'0c/no fee':>11s} {'2c/+fee':>11s} {'5c/+fee':>11s} {'Unscored $':>11s}")
    for key,title in strategies:
        sub=select(records,key)
        a,_=scenario(sub,payouts,v1,v3,scale=scale,shock=0,fee_coeff=0)
        b,_=scenario(sub,payouts,v1,v3,scale=scale,shock=2,fee_coeff=fee)
        c,_=scenario(sub,payouts,v1,v3,scale=scale,shock=5,fee_coeff=fee)
        def fmt(v):return f"{v:+.2f}%" if v is not None else 'N/A'
        print(f'{title[:42]:42s} {len(sub):7,d} {a["markets_in_replay"]:7,d} {a["markets_scored"]:6d} '
              f'{fmt(a["scored_roi_pct"]):>11s} {fmt(b["scored_roi_pct"]):>11s} {fmt(c["scored_roi_pct"]):>11s} '
              f'${a["remaining_unscored_buy_usdc"]:>9,.0f}')
        data[key]={'label':title,'base':a,'adverse_2c_fees':b,'adverse_5c_fees':c}

    print('\nMINIMUM SCALED EXECUTION NOTIONAL: ALL FILLS (mechanical skip; not batching or true order minimum)')
    for threshold in (0,1,5,10):
        d,_=scenario(records,payouts,v1,v3,scale=scale,shock=2,fee_coeff=fee,min_fill_usdc=threshold)
        print(f'  >=${threshold:2d}/fill: kept={d["kept_fills"]:6,d}/{len(records):,} '
              f'scored={d["markets_scored"]:3,d} ROI={d["scored_roi_pct"] if d["scored_roi_pct"] is not None else "N/A":>7} '
              f'net=${d["scored_net_pnl"]:+,.2f} unmatched_sells={d["uncovered_sell_fills"]:,} '
              f'unscored_buy=${d["remaining_unscored_buy_usdc"]:,.2f}')
        data[f'min_fill_{threshold}_at_2c'] = d

    print('\nINTERPRETATION')
    print('  (1) Market taker-presence is a HINDSIGHT classification. Only after-first-taker is causal.')
    print('  (2) Taker-only simulations still assume immediate fills at source prices, not delay-adjusted orders.')
    print('  (3) 1¢ and 2¢ price shocks are fixed hypothetical sensitivity, not real order-book prices.')
    print('  (4) No exchange-specific fees, liquidity, signal publish lag, or maker fill-probability model.')
    print('  (5) Results are selection-sensitive while politics/open markets lack completed settlement.')
    if args.out_json:
        dest=args.out_json.expanduser().resolve()
        if str(dest).startswith('/workspaces/') or dest.exists():
            raise RuntimeError('Output path inside repo or already exists: refuses writes')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps({'wallet':v1.WALLET,'copy_scale':scale,'source_fills':len(records),
                                    'note':'Non-executable wallet-price simulations; see console caveats',
                                    'results':data},indent=2),encoding='utf-8')
        print('New JSON:',dest)
    print('READ ONLY — NO NETWORK, API REQUESTS, EXCHANGE ORDERS OR PROJECT WRITES.')


if __name__=='__main__':
    main()
