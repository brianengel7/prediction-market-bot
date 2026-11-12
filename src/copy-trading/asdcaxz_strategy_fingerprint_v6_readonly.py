#!/usr/bin/env python3
"""Offline forensic fingerprint of Polymarket short-window BTC wallet fills.

Purpose: distinguish hypotheses about FIRST-LEG selection and inventory hedging.
Never submits orders, uses no API, and does not estimate executable follower P&L.

Input is the COMPLETE cached <wallet>_trades.jsonl and corresponding _meta.json
produced by poly_wallet_copy_audit_readonly.py; optionally cross-checks the v4
market_inventory.csv for accounting consistency.

Outputs: summary.json, market_fingerprints.csv, hedge_delay_cohorts.csv,
         timeline_examples.json.

Cautions: Polymarket trade timestamps are integer seconds, so within-second
ordering is unknown; all price and inventory calculations use observed wallet
fills, never executable public order-book quotes. A positive pair spread is not
whole-wallet profitability. Unobserved transfers, settlement, fees, maker/taker
status, other accounts, and fills outside the requested date window are unknown.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter, deque
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import median

PATTERN = re.compile(r'^(?:btc|bitcoin)[-_]updown[-_](5m|15m)[-_](\d{10})(?:$|[-_])', re.I)
EPS = 1e-7


@dataclass
class Fill:
    ts: int
    line: int
    side: int
    token: str
    px: float
    qty: float


def load(wallet_dir: Path, wallet: str, window: str):
    path = wallet_dir / f'{wallet}_trades.jsonl'
    meta = wallet_dir / f'{wallet}_trades_meta.json'
    if not path.is_file() or not meta.is_file():
        raise ValueError('Missing complete wallet cache and metadata at '+str(wallet_dir))
    metadata = json.loads(meta.read_text())
    if metadata.get('complete') is not True:
        raise ValueError('Trade cache incomplete; do not analyze truncated downloads')
    markets = defaultdict(lambda: {'slug':None,'start':None,'fills':[],'tokens':defaultdict(set),'bad':None})
    counters=Counter()
    with path.open(encoding='utf-8') as f:
        for line_no, ln in enumerate(f, 1):
            if not ln.strip():continue
            r=json.loads(ln)
            counters['all_trades_in_wallet_cache']+=1
            slug=str(r.get('slug') or '')
            m=PATTERN.search(slug)
            if not m or m.group(1).lower()!=window: continue
            counters['in_scope_trades']+=1
            cid=str(r.get('condition_id') or '')
            if not re.fullmatch(r'0x[0-9a-fA-F]{64}',cid):
                counters['excluded_invalid_condition_fills']+=1
                continue
            market=markets[cid.lower()]
            start=int(m.group(2))
            if market['slug'] is None:
                market['slug'], market['start'] = slug, start
            elif market['slug']!=slug or market['start']!=start:
                market['bad']='market metadata conflict'
            if str(r.get('side') or '').upper()!='BUY':
                market['bad']='contains non-BUY transaction'
                continue
            try:
                out=str(r['outcome_index'])
                ts=int(r['timestamp']);px=float(r['price']);qty=float(r['size'])
                token=str(r.get('token_id') or '')
            except (ValueError,KeyError,TypeError):
                market['bad']='unreadable trade';continue
            if out not in ('0','1') or not token or not(math.isfinite(px) and math.isfinite(qty)) or not(0<px<1 and qty>0):
                market['bad']='invalid outcome/token/price/quantity';continue
            market['tokens'][out].add(token)
            market['fills'].append(Fill(ts,line_no,int(out),token,px,qty))
    valid={}; excluded=Counter()
    for cid,m in markets.items():
        if m['bad']:
            excluded[m['bad']]+=1;continue
        tokens=m['tokens']
        if any(len(tok)!=1 for tok in tokens.values()) or len(set().union(*tokens.values()))!=len(tokens):
            excluded['invalid token mapping']+=1;continue
        if not m['fills']:
            excluded['empty market']+=1;continue
        valid[cid]=m
    return valid,dict(excluded),dict(counters)


def gap_bucket(seconds):
    if seconds is None:return 'no_opposite_side'
    if seconds==0:return 'ambiguous_same_second'
    if seconds<=5:return '1-5s'
    if seconds<=15:return '6-15s'
    if seconds<=60:return '16-60s'
    if seconds<=300:return '61-300s'
    return '>300s'


def first_bucket(ts_delta):
    if ts_delta<0:return 'before_start'
    if ts_delta<30:return '0-29s'
    if ts_delta<60:return '30-59s'
    if ts_delta<120:return '60-119s'
    if ts_delta<300:return '120-299s'
    if ts_delta<600:return '300-599s'
    if ts_delta<900:return '600-899s'
    return 'at_or_after_close'


def finger_market(cid, market, max_timeline=500):
    # Same-second ordering is unknown; original line-order is used only for
    # cost matching and total inventory, never treated as a causal signal.
    fs=sorted(market['fills'],key=lambda f:(f.ts,f.line))
    early_ts=fs[0].ts
    first_sec=[f for f in fs if f.ts==early_ts]
    early_sides={f.side for f in first_sec}
    first_side=next(iter(early_sides)) if len(early_sides)==1 else None
    first_opposite_ts=min((f.ts for f in fs if first_side is not None and f.side!=first_side),default=None)
    first_side_end=(first_opposite_ts if first_opposite_ts is not None else float('inf'))
    opening=[f for f in fs if first_side is not None and f.side==first_side and f.ts<first_side_end]
    opening_qty=sum(f.qty for f in opening)
    opening_cost=sum(f.qty*f.px for f in opening)
    first_vwap=sum(f.qty*f.px for f in first_sec)/sum(f.qty for f in first_sec)
    first_vwap_unambiguous=opening_cost/opening_qty if opening_qty>0 else None
    first_hedge=[f for f in fs if first_opposite_ts is not None and f.side!=first_side and f.ts==first_opposite_ts]
    first_hedge_vwap=sum(f.qty*f.px for f in first_hedge)/sum(f.qty for f in first_hedge) if first_hedge else None
    pending=[deque(),deque()]
    qty=[0.,0.]
    spent=paired_cost=paired_qty=paired_profit=0.
    late_unpaired=0.
    peak_imbalance_qty=0.
    peak_unmatched_cost=0.
    ever_positive_floor=False
    first_positive_floor_time=None
    positive_pair_qty=0.
    pair_lags=defaultdict(lambda:{'qty':0.,'gross':0.})
    paired_events=[]
    timeline=[]
    first_pair=None
    total_qty=0.
    for f in fs:
        qty[f.side]+=f.qty
        total_qty+=f.qty
        spent+=f.qty*f.px
        other=1-f.side
        remaining=f.qty
        while remaining>EPS and pending[other]:
            old_ts,old_qty,old_px=pending[other][0]
            matched=min(remaining,old_qty)
            unit_pair=old_px+f.px
            gross=matched*(1-unit_pair)
            gap=f.ts-old_ts
            paired_qty+=matched
            paired_cost+=matched*unit_pair
            paired_profit+=gross
            if unit_pair<1:positive_pair_qty+=matched
            if first_pair is None: first_pair={'ts':f.ts,'gap':gap,'combined_price':unit_pair,'margin_cents':(1-unit_pair)*100}
            bucket=gap_bucket(gap)
            pair_lags[bucket]['qty']+=matched
            pair_lags[bucket]['gross']+=gross
            if len(paired_events)<50:
                paired_events.append({'match_ts':f.ts,'gap_s':gap,'qty':round(matched,5),'unit_cost':round(unit_pair,6)})
            remaining-=matched
            old_qty-=matched
            if old_qty<=EPS:pending[other].popleft()
            else: pending[other][0]=(old_ts,old_qty,old_px)
        if remaining>EPS:pending[f.side].append((f.ts,remaining,f.px))
        excess_cost=sum(q*p for side in pending for _,q,p in side)
        peak_unmatched_cost=max(peak_unmatched_cost,excess_cost)
        peak_imbalance_qty=max(peak_imbalance_qty,abs(qty[0]-qty[1]))
        floor=min(qty)-spent
        if floor>EPS:
            ever_positive_floor=True
            if first_positive_floor_time is None:first_positive_floor_time=f.ts
        if len(timeline)<max_timeline:
            timeline.append({'t':f.ts-market['start'],'side':f.side,'price':round(f.px,6),'qty':round(f.qty,5),'sum_cost':round(spent,4),'qty_0':round(qty[0],5),'qty_1':round(qty[1],5),'floor':round(floor,4)})
    unmatched_cost=spent-paired_cost
    floor=min(qty)-spent
    ceiling=max(qty)-spent
    assert abs(paired_qty-min(qty))<1e-3
    assert abs(paired_profit-(paired_qty-paired_cost))<1e-3
    assert abs(floor-(paired_profit-unmatched_cost))<1e-3
    result={
        'condition_id':cid,'slug':market['slug'],'market_start_epoch':market['start'],
        'date_utc':datetime.fromtimestamp(market['start'],timezone.utc).date().isoformat(),
        'fills':len(fs),'buy_cost':round(spent,6),'buy_shares':round(total_qty,6),
        'initial_timestamp':early_ts,'initial_s_from_start':early_ts-market['start'],
        'first_entry_bucket':first_bucket(early_ts-market['start']),
        'first_side':first_side if first_side is not None else 'ambiguous',
        'first_timestamp_both_sides':len(early_sides)>1,
        'first_timestamp_vwap':round(first_vwap,6),
        'opening_side_qty_before_hedge':round(opening_qty,6),
        'opening_side_cost_before_hedge':round(opening_cost,6),
        'opening_side_vwap':round(first_vwap_unambiguous,6) if first_vwap_unambiguous is not None else None,
        'opening_side_above_50c':(first_vwap_unambiguous>=.5) if first_vwap_unambiguous is not None else None,
        'first_opposite_ts':first_opposite_ts,
        'first_opposite_s_from_start':first_opposite_ts-market['start'] if first_opposite_ts is not None else None,
        'first_opposite_lag_s':first_opposite_ts-early_ts if first_opposite_ts is not None else None,
        'first_opposite_price_vwap':round(first_hedge_vwap,6) if first_hedge_vwap is not None else None,
        'first_opposite_gap_bucket':('ambiguous_same_second' if first_side is None and len(early_sides)>1 else gap_bucket(first_opposite_ts-early_ts if first_opposite_ts is not None else None)),
        'first_pair_unit_cost':round(first_pair['combined_price'],6) if first_pair else None,
        'first_pair_margin_cents':round(first_pair['margin_cents'],4) if first_pair else None,
        'first_pair_gap_s':first_pair['gap'] if first_pair else None,
        'paired_shares':round(paired_qty,6),'paired_cost':round(paired_cost,6),
        'paired_gross':round(paired_profit,6),'paired_margin_cents':round(100*paired_profit/paired_qty,6) if paired_qty else None,
        'positive_margin_paired_qty':round(positive_pair_qty,6),
        'unmatched_cost':round(unmatched_cost,6),
        'unmatched_shares':round(abs(qty[0]-qty[1]),6),
        'ending_side_0_shares':round(qty[0],6),'ending_side_1_shares':round(qty[1],6),
        'ending_imbalance_shares':round(abs(qty[0]-qty[1]),6),
        'peak_imbalance_shares':round(peak_imbalance_qty,6),
        'peak_unmatched_cost':round(peak_unmatched_cost,6),
        'ending_worst_case_gross':round(floor,6),
        'ending_best_case_gross':round(ceiling,6),
        'end_guaranteed_positive_gross':floor>EPS,
        'ever_guaranteed_positive_gross':ever_positive_floor,
        'first_positive_floor_s_from_start':first_positive_floor_time-market['start'] if first_positive_floor_time is not None else None,
        'last_fill_s_from_start':fs[-1].ts-market['start'],
        'late_fills_at_or_after_end':sum(f.ts >= market['start']+(300 if '5m' in market['slug'] else 900) for f in fs),
    }
    return result,{'timeline':timeline,'paired_events_sample':paired_events,'pair_lags':dict(pair_lags)}


def summarize(rows, excluded, counters):
    S=lambda k:sum(float(r[k]) for r in rows)
    n=len(rows)
    first_known=[r for r in rows if isinstance(r['first_side'],int)]
    both=[r for r in rows if r['paired_shares']>EPS]
    positive_floor=[r for r in rows if r['end_guaranteed_positive_gross']]
    summary={
        'interpretation':'Observed trade-feed fingerprint; hypotheses only. Not a recovered bot policy, realized wallet net P&L, or executable follower P&L.',
        'markets':n,'total_fills':sum(r['fills'] for r in rows),
        'source':counters,'excluded_market_reasons':excluded,
        'spend_usdc':round(S('buy_cost'),2),'paired_shares':round(S('paired_shares'),4),
        'paired_cost_usdc':round(S('paired_cost'),2),
        'paired_gross_usdc_before_fees':round(S('paired_gross'),2),
        'unmatched_cost_usdc':round(S('unmatched_cost'),2),
        'whole_book_worst_case_gross_usdc':round(S('ending_worst_case_gross'),2),
        'whole_book_best_case_gross_usdc':round(S('ending_best_case_gross'),2),
        'two_sides_markets':len(both),'one_side_markets':n-len(both),
        'first_timestamp_both_sides_markets':sum(r['first_timestamp_both_sides'] for r in rows),
        'unambiguous_first_side_markets':len(first_known),
        'first_side_0_markets':sum(r['first_side']==0 for r in first_known),
        'first_side_1_markets':sum(r['first_side']==1 for r in first_known),
        'first_side_price_above_50c_pct':round(100*sum(r['opening_side_above_50c'] is True for r in first_known)/len(first_known),2) if first_known else None,
        'first_opposite_gap_median_s':median(r['first_opposite_lag_s'] for r in both if r['first_opposite_lag_s'] is not None) if both else None,
        'first_opposite_under_15s_markets':sum(0<=r['first_opposite_lag_s']<=15 for r in both if r['first_opposite_lag_s'] is not None),
        'first_opposite_under_60s_markets':sum(0<=r['first_opposite_lag_s']<=60 for r in both if r['first_opposite_lag_s'] is not None),
        'first_hedge_below_1usd_pct':round(100*sum(r['first_pair_unit_cost'] is not None and r['first_pair_unit_cost']<1-EPS for r in both)/len(both),2) if both else None,
        'paired_markets_positive_gross':sum(r['paired_gross']>EPS for r in rows),
        'positive_floor_markets':len(positive_floor),
        'ever_positive_floor_markets':sum(r['ever_guaranteed_positive_gross'] for r in rows),
        'peak_unmatched_cost_median_usdc':round(median(r['peak_unmatched_cost'] for r in rows),2) if rows else None,
        'first_entry_bucket_counts':dict(Counter(r['first_entry_bucket'] for r in rows)),
        'first_hedge_delay_bucket_counts':dict(Counter(r['first_opposite_gap_bucket'] for r in rows)),
        'warning':'Seconds-granularity ordering; no maker/taker classification, quotes, fills outside sample, spot signal or real fees.'
    }
    cohorts=[]
    groups=defaultdict(list)
    for r in rows: groups[r['first_opposite_gap_bucket']].append(r)
    order=['ambiguous_same_second','1-5s','6-15s','16-60s','61-300s','>300s','no_opposite_side']
    for k in order:
        group=groups.get(k,[])
        if not group:continue
        cohorts.append({'first_opposite_gap':k,'markets':len(group),
                        'spent':round(sum(r['buy_cost'] for r in group),2),
                        'paired_qty':round(sum(r['paired_shares'] for r in group),2),
                        'paired_gross':round(sum(r['paired_gross'] for r in group),2),
                        'gross_pair_margin_cents':round(100*sum(r['paired_gross'] for r in group)/sum(r['paired_shares'] for r in group),4) if sum(r['paired_shares'] for r in group)>0 else None,
                        'unmatched_cost':round(sum(r['unmatched_cost'] for r in group),2),
                        'positive_floor_markets':sum(r['end_guaranteed_positive_gross'] for r in group)})
    return summary,cohorts


def verify_v4(rows, directory):
    path=directory/'market_inventory.csv'
    if not path.is_file():raise ValueError('Missing V4 market_inventory.csv '+str(directory))
    with path.open(newline='',encoding='utf-8') as f:old={r['condition_id'].lower():r for r in csv.DictReader(f)}
    new={r['condition_id']:r for r in rows}
    if old.keys()!=new.keys():
        a=len(old.keys()-new.keys());b=len(new.keys()-old.keys())
        raise ValueError(f'V4 cohort differs: {a} missing in V6, {b} new in V6; DO NOT compare until fixed')
    for cid,r in new.items():
        o=old[cid]
        for n,k in [('buy_cost','total_cost_usdc'),('paired_shares','paired_shares'),('paired_cost','paired_cost_usdc'),('paired_gross','paired_gross_usdc'),('unmatched_cost','unpaired_cost_usdc')]:
            if abs(r[n]-float(o[k]))>max(.01,1e-6*abs(float(o[k]))):
                raise ValueError(f'V4/V6 reconciliation failed for {cid} field {n}: {r[n]} vs {o[k]}')
    print(f'PASS: V6 reconciled with V4 for all {len(rows):,} conditions and 5 monetary/inventory quantities')


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run(market_cache, wallet, window, output, verify_risk_dir=None):
    markets,excluded,counters=load(market_cache,wallet,window)
    if not markets:raise ValueError('No eligible markets in cache')
    records=[];detail={}
    for cid,m in markets.items():
        rec,ex=finger_market(cid,m)
        records.append(rec)
        detail[cid]=ex
    records.sort(key=lambda r:(r['market_start_epoch'],r['condition_id']))
    if verify_risk_dir: verify_v4(records,verify_risk_dir)
    report,cohorts=summarize(records,excluded,counters)
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    write_csv(output/'market_fingerprints.csv',records)
    write_csv(output/'hedge_delay_cohorts.csv',cohorts)
    # Examples include winning/losing pairs, large unmatched exposure, and quick hedges.
    samples={}
    sorts=[('largest_paired_gross',lambda r:r['paired_gross'],True),
           ('most_negative_paired_gross',lambda r:r['paired_gross'],False),
           ('largest_unmatched_cost',lambda r:r['unmatched_cost'],True),
           ('earliest_two_sided',lambda r:r['first_opposite_lag_s'] if r['first_opposite_lag_s'] is not None else 10**12,False)]
    for label,fun,reverse in sorts:
        examples=sorted(records,key=fun,reverse=reverse)[:5]
        samples[label]=[{'market':r,'evidence':detail[r['condition_id']]} for r in examples]
    (output/'timeline_examples.json').write_text(json.dumps(samples,indent=2)+'\n',encoding='utf-8')
    print('=== BTC '+window+' STRATEGY FINGERPRINT (OBSERVED WALLET FILLS; NOT A COPIER BACKTEST) ===')
    for k in ['markets','total_fills','spend_usdc','two_sides_markets','one_side_markets',
              'first_timestamp_both_sides_markets','first_side_0_markets','first_side_1_markets',
              'first_side_price_above_50c_pct','first_opposite_gap_median_s','first_opposite_under_15s_markets',
              'first_opposite_under_60s_markets','first_hedge_below_1usd_pct','paired_gross_usdc_before_fees',
              'unmatched_cost_usdc','whole_book_worst_case_gross_usdc','positive_floor_markets',
              'ever_positive_floor_markets','peak_unmatched_cost_median_usdc']:
        print(f' {k:40s} : {report[k]}')
    print('FIRST OPPOSITE-SIDE BUY GAP / COHORT:')
    for c in cohorts:
        print(f" {c['first_opposite_gap']:22s} markets={c['markets']:5d} paired_margin={c['gross_pair_margin_cents']}c final_floor_positive={c['positive_floor_markets']}")
    print('READ ONLY. No network, exchange orders, or modifications to the source cache.')
    print('Reports:',output)
    return report


def self_test():
    def m(fills):
        d={'slug':'btc-updown-15m-1792000000','start':1792000000,'fills':[], 'tokens':defaultdict(set)}
        for i,(dt,s,px,q) in enumerate(fills,1):
            d['fills'].append(Fill(1792000000+dt,i,s,'tok'+str(s),px,q))
            d['tokens'][str(s)].add('tok'+str(s))
        return d
    r,_=finger_market('x',m([(2,0,.45,10),(12,1,.50,10)]))
    assert r['first_side']==0 and r['first_opposite_lag_s']==10
    assert abs(r['paired_gross']-.50)<1e-5 and r['unmatched_cost']==0
    assert r['end_guaranteed_positive_gross'] and r['first_pair_margin_cents']>0
    r,_=finger_market('y',m([(3,0,.60,20),(20,1,.45,10)]))
    assert abs(r['paired_gross']+.50)<1e-5 and abs(r['unmatched_cost']-6)<1e-5
    assert not r['end_guaranteed_positive_gross']
    r,_=finger_market('z',m([(4,0,.5,3),(4,1,.48,3)]))
    assert r['first_timestamp_both_sides'] and r['first_side']=='ambiguous'
    r,_=finger_market('t',m([(30,1,.2,5)]))
    assert r['first_opposite_lag_s'] is None and r['paired_shares']==0
    print('PASS: positive and negative pair margins, unmatched exposure, first-leg chronology and ambiguous same-second order; OFFLINE')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir',type=Path,default=Path('data/research/poly_wallet_copy_audit_20261111'))
    ap.add_argument('--wallet',default='asdcaxz')
    ap.add_argument('--window',default='15m',choices=('5m','15m'))
    ap.add_argument('--risk-dir',type=Path,default=None,help='Optional v4 directory for strict per-condition reconciliation')
    ap.add_argument('--output-dir',type=Path,default=Path('data/research/asdcaxz_strategy_fingerprint_v6'))
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:return self_test()
    return run(args.input_dir,args.wallet,args.window,args.output_dir,args.risk_dir)

if __name__=='__main__':main()
