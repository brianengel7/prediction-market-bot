#!/usr/bin/env python3
"""Read-only, OFFLINE, no-lookahead inventory-risk audit of Polymarket crypto markets.

Designed for the COMPLETE wallet trade caches created by
poly_wallet_copy_audit_readonly.py. Reads cached JSONL only; no network calls,
keys, trading modules, exchanges, or orders. Writes a NEW report directory.

Why: positive paired-only gross return does not establish profitability of
ALL inventory. For a fully settled and genuinely complementary binary market
with no sells:
  worst_case_gross = min(total outcome0 shares, total outcome1 shares) - all buys cost
  best_case_gross  = max(total outcome0 shares, total outcome1 shares) - all buys cost
The final realized payoff lies between these bounds, BEFORE fees, provided
there are no external transfers/positions outside the observed sample.

These are historical wallet-inventory bounds, not executable copy profits.
Timestamps are one-second resolution; same-second order cannot be established.
The script does NOT infer real settlement outcomes or fillable follower quotes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import median

WINDOW_RE = re.compile(
    r'(?:^|[-_])(btc|bitcoin|eth|ethereum|sol|solana|xrp)[-_]updown[-_](5m|15m)[-_](\d{10})(?:$|[-_])',
    re.IGNORECASE,
)
EPS = 1e-8
SENSITIVITY_CENTS_PER_SHARE = (0, 0.25, 0.5, 1, 1.5, 2)


@dataclass
class Fill:
    ts: int
    lineno: int
    idx: str
    token: str
    price: float
    qty: float


def read_markets(path: Path, wanted_window: str, coin: str | None):
    markets = defaultdict(lambda: {'fills': [], 'slug': '', 'asset': '', 'start': None, 'bad': False, 'has_sells': False})
    stats = defaultdict(int)
    with path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats['all_rows'] += 1
            row = json.loads(line)
            slug = str(row.get('slug') or '')
            match = WINDOW_RE.search(slug)
            if not match or match.group(2).lower() != wanted_window:
                continue
            asset = match.group(1).lower()
            asset = {'bitcoin': 'btc', 'ethereum': 'eth', 'solana': 'sol'}.get(asset, asset)
            if coin and asset != coin:
                continue
            stats['window_rows'] += 1
            cid = str(row.get('condition_id') or '')
            if not cid:
                stats['missing_condition'] += 1
                continue
            m = markets[cid]
            if not m['slug']:
                m['slug'], m['asset'], m['start'] = slug, asset, int(match.group(3))
            elif m['slug'] != slug or m['asset'] != asset:
                m['bad'] = True
            side = str(row.get('side') or '').upper()
            if side == 'SELL':
                m['has_sells'] = True
                continue
            if side != 'BUY':
                m['bad'] = True
                continue
            try:
                ts = int(row['timestamp'])
                idx = str(row.get('outcome_index'))
                token = str(row.get('token_id') or '')
                qty, price = float(row['size']), float(row['price'])
                valid = (idx in ('0', '1') and bool(token) and math.isfinite(qty)
                         and math.isfinite(price) and qty > 0 and 0 < price < 1)
            except (TypeError, ValueError, KeyError):
                valid = False
            if not valid:
                m['bad'] = True
                stats['invalid_fills'] += 1
                continue
            m['fills'].append(Fill(ts, lineno, idx, token, price, qty))
    return markets, dict(stats)


def process_market(cid: str, market: dict):
    if market['bad'] or market['has_sells'] or not market['fills']:
        return None, 'has_sells' if market['has_sells'] else 'invalid_or_empty'
    rows = sorted(market['fills'], key=lambda x: (x.ts, x.lineno))
    byidx = defaultdict(set)
    for fill in rows:
        byidx[fill.idx].add(fill.token)
    if any(len(v) > 1 for v in byidx.values()):
        return None, 'multiple_token_ids_per_outcome'
    if len(set().union(*byidx.values())) != len(byidx):
        return None, 'same_token_both_outcomes'

    qty = {'0': 0., '1': 0.}
    pending = {'0': deque(), '1': deque()}
    spent = paired_cost = paired_qty = paired_gross = 0.
    profitable_pair_qty = 0.
    within5 = within60 = within300 = same_second_qty = 0.
    positive_floor_first_ts = None
    first_pair_ts = None
    pairing_intervals = defaultdict(lambda: {'qty': 0., 'gross': 0.})
    buy_qty = 0.

    for fill in rows:
        idx = fill.idx
        other = '1' if idx == '0' else '0'
        qty[idx] += fill.qty
        spent += fill.qty * fill.price
        buy_qty += fill.qty
        rem = fill.qty
        while rem > EPS and pending[other]:
            earlier = pending[other][0]
            pair = min(rem, earlier[1])
            old_ts, _, old_px = earlier
            gap = fill.ts - old_ts
            pcost = pair * (fill.price + old_px)
            pgross = pair - pcost
            paired_qty += pair
            paired_cost += pcost
            paired_gross += pgross
            if first_pair_ts is None:
                first_pair_ts = fill.ts
            if pcost < pair:
                profitable_pair_qty += pair
            if gap == 0:
                same_second_qty += pair
            if gap <= 5:
                within5 += pair
            if gap <= 60:
                within60 += pair
            if gap <= 300:
                within300 += pair
            bucket = 'same_second' if gap == 0 else ('1-5s' if gap <= 5 else
                     '6-15s' if gap <= 15 else '16-60s' if gap <= 60 else
                     '61-300s' if gap <= 300 else '>300s')
            pairing_intervals[bucket]['qty'] += pair
            pairing_intervals[bucket]['gross'] += pgross
            earlier[1] -= pair
            rem -= pair
            if earlier[1] <= EPS:
                pending[other].popleft()
        if rem > EPS:
            pending[idx].append([fill.ts, rem, fill.price])
        if positive_floor_first_ts is None and min(qty.values()) - spent > EPS:
            positive_floor_first_ts = fill.ts

    floor = min(qty.values()) - spent
    ceiling = max(qty.values()) - spent
    single = len(byidx) == 1
    # Qty unmatched to an opposite-side purchase; cost includes all fills.
    unmatched_qty = abs(qty['0'] - qty['1'])
    # This equals the total paid for unmatched inventory at its actual FIFO cost.
    unmatched_cost = spent - paired_cost
    assert paired_qty <= min(qty.values()) + 1e-5
    assert abs(paired_qty - min(qty.values())) < 1e-4
    assert abs(paired_gross - (paired_qty - paired_cost)) < 1e-4
    assert abs(floor - (paired_gross - unmatched_cost)) < 1e-4
    start = market['start']
    first_ts, last_ts = rows[0].ts, rows[-1].ts
    date = datetime.fromtimestamp(start, timezone.utc).date().isoformat()
    result = {
        'condition_id': cid, 'slug': market['slug'], 'asset': market['asset'],
        'utc_market_date': date, 'market_start_epoch': start,
        'fill_count': len(rows), 'first_buy_s_from_market_start': first_ts-start,
        'last_buy_s_from_market_start': last_ts-start,
        'buys_after_market_end_count': sum(r.ts >= start+900 for r in rows) if '15m' in market['slug'] else 0,
        'single_outcome_observed': single,
        'bought_shares_0': qty['0'], 'bought_shares_1': qty['1'],
        'total_shares_bought': buy_qty, 'total_cost_usdc': spent,
        'paired_shares': paired_qty, 'paired_cost_usdc': paired_cost,
        'paired_gross_usdc': paired_gross, 'paired_gross_cents_per_pair': 100*paired_gross/paired_qty if paired_qty else None,
        'unpaired_shares': unmatched_qty, 'unpaired_cost_usdc': unmatched_cost,
        'full_inventory_worst_gross_usdc': floor,
        'full_inventory_best_gross_usdc': ceiling,
        'full_inventory_guaranteed_positive': floor > EPS,
        'first_pair_formed_s_from_market_start': None if first_pair_ts is None else first_pair_ts-start,
        'first_positive_floor_s_from_market_start': None if positive_floor_first_ts is None else positive_floor_first_ts-start,
        'pair_qty_completed_within_5s': within5,
        'pair_qty_completed_within_60s': within60,
        'pair_qty_completed_within_300s': within300,
        'same_second_pair_qty': same_second_qty,
        'paired_positive_margin_qty': profitable_pair_qty,
    }
    return (result, {k:dict(v) for k,v in pairing_intervals.items()}), None


def percent(n, d):
    return 100*n/d if d else 0.


def aggregate(markets, stats, window, coin):
    rows = []
    skip = defaultdict(int)
    buckets = defaultdict(lambda: {'qty':0.,'gross':0.})
    for cid,m in markets.items():
        result, error = process_market(cid,m)
        if error:
            skip[error]+=1
            continue
        row, intervals = result
        rows.append(row)
        for k,v in intervals.items():
            buckets[k]['qty']+=v['qty']
            buckets[k]['gross']+=v['gross']
    def summ(key): return sum(r[key] for r in rows)
    buy_cost = summ('total_cost_usdc')
    total_qty = summ('total_shares_bought')
    paired_qty = summ('paired_shares')
    paired_cost = summ('paired_cost_usdc')
    paired_gross = summ('paired_gross_usdc')
    full_floor = summ('full_inventory_worst_gross_usdc')
    full_ceiling = summ('full_inventory_best_gross_usdc')
    unmatched_cost = summ('unpaired_cost_usdc')
    # This is NOT a portfolio P&L estimate; no outcomes and no executable quotes.
    summary = {
        'market_window': window, 'asset_filter': coin or 'all',
        'source_rows': stats, 'skipped_market_counts': dict(skip),
        'analyzed_markets': len(rows),
        'markets_with_both_outcomes_observed': sum(not r['single_outcome_observed'] for r in rows),
        'markets_with_only_one_outcome_observed': sum(r['single_outcome_observed'] for r in rows),
        'markets_with_positive_paired_gross': sum(r['paired_gross_usdc'] > EPS for r in rows),
        'markets_with_positive_whole_inventory_floor': sum(r['full_inventory_guaranteed_positive'] for r in rows),
        'wallet_spend_usdc': buy_cost,
        'paired_cost_usdc': paired_cost,
        'paired_qty': paired_qty,
        'paired_gross_usdc_before_fees': paired_gross,
        'paired_margin_cents_per_share': 100*paired_gross/paired_qty if paired_qty else None,
        'unmatched_inventory_cost_usdc': unmatched_cost,
        'unmatched_shares': summ('unpaired_shares'),
        'whole_inventory_settlement_worst_gross_usdc': full_floor,
        'whole_inventory_settlement_best_gross_usdc': full_ceiling,
        'paired_qty_within_5s_pct': percent(summ('pair_qty_completed_within_5s'), paired_qty),
        'paired_qty_within_60s_pct': percent(summ('pair_qty_completed_within_60s'), paired_qty),
        'paired_qty_within_300s_pct': percent(summ('pair_qty_completed_within_300s'), paired_qty),
        'paired_qty_same_second_pct': percent(summ('same_second_pair_qty'), paired_qty),
        'paired_qty_positive_margin_pct': percent(summ('paired_positive_margin_qty'), paired_qty),
        'paired_latency_buckets': {k:{**v, 'gross_cents_per_pair':100*v['gross']/v['qty'] if v['qty'] else None}
                                   for k,v in sorted(buckets.items())},
        'whole_inventory_worst_case_by_added_price_per_purchased_share': {
            f'{c:.2f}c': full_floor - total_qty*(c/100) for c in SENSITIVITY_CENTS_PER_SHARE
        },
        'paired_only_margin_after_added_cost_per_pair': {
            f'{c:.2f}c': paired_gross - paired_qty*(c/100) for c in SENSITIVITY_CENTS_PER_SHARE
        },
        'markets_with_buys_at_or_after_window_end': sum(r['buys_after_market_end_count']>0 for r in rows),
        'interpretation': 'Full inventory payoff bounds assume either binary outcome settles at $1 per share, exclude fees and external positions. Single-side markets are included in worst-case as $0 payoff. Paired margin includes only quantity-matched legs. No copy execution, settlements, or fee impact verified.'
    }
    return rows, summary


def write_report(outdir:Path, rows:list, summary:dict):
    outdir.mkdir(parents=True,exist_ok=True)
    fp = outdir/'market_inventory.csv'
    if rows:
        with fp.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Per-day aggregates with identical full-portfolio logic.
    days=defaultdict(list)
    for r in rows: days[r['utc_market_date']].append(r)
    dayrows=[]
    for day, rs in sorted(days.items()):
        dayrows.append({'utc_market_date':day,'markets':len(rs),
          'spent_usdc':sum(r['total_cost_usdc'] for r in rs),
          'paired_gross_usdc':sum(r['paired_gross_usdc'] for r in rs),
          'full_inventory_worst_gross_usdc':sum(r['full_inventory_worst_gross_usdc'] for r in rs),
          'full_inventory_best_gross_usdc':sum(r['full_inventory_best_gross_usdc'] for r in rs),
          'positive_floor_markets':sum(r['full_inventory_guaranteed_positive'] for r in rs)})
    if dayrows:
        with (outdir/'daily.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dayrows[0]));writer.writeheader();writer.writerows(dayrows)


def print_report(s, rows):
    print('=== OFFLINE CRYPTO INVENTORY RISK AUDIT — NOT EXECUTABLE COPY P&L ===')
    print(f"Window={s['market_window']} asset={s['asset_filter']} markets={s['analyzed_markets']:,}, "
          f"both_outcomes={s['markets_with_both_outcomes_observed']:,}, "
          f"one_outcome={s['markets_with_only_one_outcome_observed']:,}")
    print(f"Spend total=${s['wallet_spend_usdc']:,.2f}; matched paired cost=${s['paired_cost_usdc']:,.2f}; "
          f"unmatched inventory cost=${s['unmatched_inventory_cost_usdc']:,.2f}")
    print(f"Matched qty={s['paired_qty']:,.1f} gross paired=${s['paired_gross_usdc_before_fees']:,.2f} "
          f"({s['paired_margin_cents_per_share']:.3f} cents/pair)")
    print(f"FULL-BOOK settlement payoff bounds (gross, no fees): "
          f"WORST=${s['whole_inventory_settlement_worst_gross_usdc']:,.2f} "
          f"BEST=${s['whole_inventory_settlement_best_gross_usdc']:,.2f}")
    print(f"Positive gross: paired markets={s['markets_with_positive_paired_gross']:,}, "
          f"WHOLE-book guaranteed positive markets={s['markets_with_positive_whole_inventory_floor']:,}")
    print('Matched pair completion gaps (by qty):')
    for k, v in s['paired_latency_buckets'].items():
        print(f" {k:12} qty={v['qty']:>11,.0f} gross=${v['gross']:>11,.2f} "
              f"margin={v['gross_cents_per_pair']:>7.3f}c/pair")
    print('Added price/slippage stress on ALL observed purchased shares (before fees):')
    for k, v in s['whole_inventory_worst_case_by_added_price_per_purchased_share'].items():
        print(f' +{k} per share => worst-case gross ${v:,.2f}')
    print(f"Markets with buys >= end of declared window: {s['markets_with_buys_at_or_after_window_end']:,}")
    print(f"Skipped for invalid evidence: {s['skipped_market_counts']}")
    print('NO settlement outcomes or historical quotes: no strategy profitability or follower execution verdict.')


def self_test():
    def mk(rows,slug='btc-updown-15m-1792000000'):
        return {'fills':[Fill(ts,i,idx,'tok'+idx,p,q) for i,(ts,idx,p,q) in enumerate(rows,1)],
                'slug':slug,'asset':'btc','start':1792000000,'bad':False,'has_sells':False}
    a,err=process_market('a',mk([(1792000001,'0',.48,10),(1792000010,'1',.50,10)]))
    assert err is None
    r,_=a
    assert abs(r['paired_gross_usdc']-.2)<EPS and abs(r['full_inventory_worst_gross_usdc']-.2)<EPS
    b,err=process_market('b',mk([(1792000001,'0',.55,20),(1792000010,'1',.35,10)]))
    assert err is None
    r,_=b
    assert abs(r['paired_gross_usdc']-1)<EPS
    assert abs(r['full_inventory_worst_gross_usdc']+4.5)<EPS
    assert abs(r['full_inventory_best_gross_usdc']-5.5)<EPS
    assert abs(r['unpaired_cost_usdc']-5.5)<EPS  # 10 unmatched YES @55c
    c,err=process_market('c',mk([(1792000001,'0',.20,10)]))
    assert err is None
    r,_=c
    assert r['single_outcome_observed'] and abs(r['full_inventory_worst_gross_usdc']+2)<EPS
    assert abs(r['full_inventory_best_gross_usdc']-8)<EPS
    print('PASS: balanced pair; profitable pair with losing worst-case residual; single-outcome risk')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir',type=Path,default=Path('data/research/poly_wallet_copy_audit_20261111'))
    p.add_argument('--wallet',default='asdcaxz')
    p.add_argument('--window',default='15m',choices=('5m','15m'))
    p.add_argument('--asset',choices=('btc','eth','sol','xrp'),default=None)
    p.add_argument('--output-dir',type=Path,default=Path('data/research/poly_wallet_15m_risk_audit_v4'))
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        return self_test()
    fp=args.input_dir/f'{args.wallet}_trades.jsonl'
    meta=args.input_dir/f'{args.wallet}_trades_meta.json'
    if not fp.is_file() or not meta.is_file():
        p.error('Missing complete cached trade+metadata pair')
    md=json.loads(meta.read_text(encoding='utf-8'))
    if md.get('complete') is not True:
        p.error('Wallet trade cache is incomplete. Results would be misleading.')
    markets,stats=read_markets(fp,args.window,args.asset)
    rows,s=aggregate(markets,stats,args.window,args.asset)
    if not rows:
        p.error('No valid markets in selected class')
    print_report(s,rows)
    write_report(args.output_dir,rows,s)
    print(f'Wrote NEW reports: {args.output_dir}/summary.json, market_inventory.csv, daily.csv')
    print('READ ONLY source caches; no API calls, no orders, no changes to copier.')

if __name__=='__main__':
    main()
