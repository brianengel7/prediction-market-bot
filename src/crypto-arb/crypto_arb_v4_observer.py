#!/usr/bin/env python3
"""V4.3 read-only Coinbase dated futures / spot price observer.

Public GET only. Appends timestamped non-executable snapshots to JSONL.
Requires V4.1 crypto_arb_v4_coinbase_probe.py in the same directory.
Never issues orders, signs requests, submits scanner candidates, or invents prices.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import crypto_arb_v4_coinbase_probe as cb

PAIRS = (
    ('BIT-27NOV26-CDE', 'BTC-USD'),
    ('BIT-24DEC26-CDE', 'BTC-USD'),
    ('ET-27NOV26-CDE', 'ETH-USD'),
    ('ET-24DEC26-CDE', 'ETH-USD'),
    ('SOL-27NOV26-CDE', 'SOL-USD'),
    ('SOL-24DEC26-CDE', 'SOL-USD'),
)
PROFILES = (10.0, 90.0)   # Hypothetical scenario assumptions, not account fees.


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def as_float(value, key):
    x = float(value)
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f'{key}: expected positive finite number')
    return x


def buy_spot_vwap(asks, base_qty):
    """One-contract ASK VWAP from returned levels. Return None if insufficient depth."""
    remaining = as_float(base_qty, 'base_qty')
    spent = 0.0
    prev_price = 0.0
    for i, level in enumerate(asks):
        price = as_float(level['price'], f'ask[{i}] price')
        size = as_float(level['size_as_returned'], f'ask[{i}] size')
        if price < prev_price:
            raise ValueError('Spot ASK levels not sorted ascending')
        prev_price = price
        take = min(remaining, size)
        spent += price*take
        remaining -= take
        if remaining <= base_qty*1e-10:
            return spent/base_qty
    return None


def evaluate(doc, *, skew_limit_s=2.0, fee_bps_scenarios=PROFILES,
             futures_roundtrip_usd=0.20, mismatch_reserve_bps=10.0,
             financing_apr_pct=0.0):
    if doc.get('status') != 'LIVE_PUBLIC_BOOK_DIAGNOSTIC_NOT_EXECUTABLE_ARB':
        raise ValueError('Bad probe status')
    m = doc['future_metadata']
    spot, fut = doc['spot_book'], doc['future_book']
    if m['product_id'] != fut['product_id']:
        raise ValueError('Product metadata/book mismatch')
    symbol = str(m['contract_root_unit']).upper()
    if spot['product_id'] != symbol + '-USD':
        raise ValueError('Spot/futures underlying mismatch')
    if m.get('product_type') != 'FUTURE' or 'perp' in str(m.get('display_name','')).lower():
        raise ValueError('Not a dated futures contract')
    end = cb.utc(m.get('contract_expiry_utc'))
    captured = cb.utc(doc['fetched_between_utc'][1])
    if not end or not captured or not (captured < end <= captured + cb.timedelta(days=730)):
        raise ValueError('Invalid expiry')
    skew = float(doc['book_timestamp_skew_s'])
    if not math.isfinite(skew) or skew > skew_limit_s or skew < 0:
        raise ValueError(f'Book timestamp skew {skew:.3f}s above {skew_limit_s:.3f}s')
    # Probe validates each exchange timestamp for book freshness <=30s.
    base_qty = as_float(m['contract_size'], 'contract_size')
    spot_asks = spot['asks']
    fut_bids = fut['bids']
    if not spot_asks or not fut_bids:
        raise ValueError('Empty bid or ask book')
    spot_top = as_float(spot_asks[0]['price'], 'spot top ask')
    fut_top = as_float(fut_bids[0]['price'], 'futures top bid')
    one_lot_spot_vwap = buy_spot_vwap(spot_asks, base_qty)
    days = (end - captured).total_seconds()/86400
    result = {
        'capture_utc': captured.isoformat(),
        'future_id': m['product_id'], 'asset': symbol,
        'spot_id': spot['product_id'], 'expiry_utc': end.isoformat(),
        'contract_size_base': base_qty, 'days_to_expiry': round(days, 5),
        'spot_top_ask': spot_top,
        'spot_top_ask_size_base': as_float(spot_asks[0]['size_as_returned'],'spot top size'),
        'futures_top_bid': fut_top,
        'futures_top_bid_size_RAW_UNVERIFIED': as_float(fut_bids[0]['size_as_returned'],'future top size'),
        'spot_vwap_one_contract': one_lot_spot_vwap,
        'spot_one_contract_10lvl_depth': one_lot_spot_vwap is not None,
        'spot_one_contract_TOP_depth': float(spot_asks[0]['size_as_returned']) >= base_qty,
        'book_skew_s': round(skew, 5),
        'gross_top_basis_bps': round((fut_top/spot_top - 1)*10000, 5),
        'futures_size_units_verified': False, 'account_fees_verified': False,
        'is_executable_trade_signal': False,
        'limitations': 'Sequential books; future size unit, total fees and settlement hedging unverified',
    }
    for fee in fee_bps_scenarios:
        if not (math.isfinite(fee) and fee >= 0):
            raise ValueError('Invalid fee scenario')
        suffix = f'{fee:g}bps_per_side'
        result[f'net_{suffix}_USD_PROVISIONAL'] = None
        if one_lot_spot_vwap is None:
            continue
        spot_notional = one_lot_spot_vwap*base_qty
        gross_dollars = (fut_top-one_lot_spot_vwap)*base_qty
        fees = spot_notional*fee*2/10000
        mismatch = spot_notional*mismatch_reserve_bps/10000
        finance = spot_notional*financing_apr_pct/100 * days/365
        net = gross_dollars-fees-futures_roundtrip_usd-mismatch-finance
        result[f'net_{suffix}_USD_PROVISIONAL'] = round(net, 6)
    if one_lot_spot_vwap is not None:
        notion = one_lot_spot_vwap*base_qty
        gross = (fut_top-one_lot_spot_vwap)*base_qty
        result['gross_one_lot_vwap_bps'] = round((fut_top/one_lot_spot_vwap-1)*10000, 5)
        result['break_even_spot_fee_bps_per_side_PROVISIONAL'] = round(
            (gross-futures_roundtrip_usd-notion*mismatch_reserve_bps/10000
             - notion*financing_apr_pct/100*days/365)*10000/(2*notion), 5)
    else:
        result['gross_one_lot_vwap_bps'] = None
        result['break_even_spot_fee_bps_per_side_PROVISIONAL'] = None
    return result


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n')


def collect(path, *, samples, interval_s, subset, max_skew_s, futures_roundtrip_usd,
            mismatch_reserve_bps, financing_apr_pct):
    import requests
    selected = [p for p in PAIRS if p[0] in subset] if subset else list(PAIRS)
    if not selected:
        raise ValueError('No matching products; choose exact ID from PAIRS')
    print(f'READ-ONLY PUBLIC GET collector: {len(selected)} contracts, {samples} cycles, {interval_s:g}s interval')
    print(f'Appending to {path}; no trading signals; scenario fee bps/side={PROFILES}')
    client = requests.Session()
    origin = time.monotonic()
    good=bad=0
    try:
        for i in range(samples):
            target = origin + i*interval_s
            remaining = target-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            for future_id, spot_id in selected:
                try:
                    doc = cb.probe(future_id, spot_id, out_dir=None, session=client)
                    derived = evaluate(doc, skew_limit_s=max_skew_s,
                        futures_roundtrip_usd=futures_roundtrip_usd,
                        mismatch_reserve_bps=mismatch_reserve_bps,
                        financing_apr_pct=financing_apr_pct)
                    append_jsonl(path, {'kind':'observation','cycle':i,'observed_at_utc':timestamp(),
                                        'metrics':derived, 'raw_probe':doc})
                    good += 1
                except (RuntimeError, ValueError, KeyError, TypeError, IndexError) as e:
                    append_jsonl(path, {'kind':'error','cycle':i,'observed_at_utc':timestamp(),
                                        'future_id':future_id,'error':f'{type(e).__name__}: {str(e)[:250]}'})
                    bad += 1
            print(f'cycle {i+1}/{samples}: collected={good}, rejected={bad}', flush=True)
    except KeyboardInterrupt:
        print('Interrupted; previously written observations retained')
    finally:
        client.close()
    return good, bad


def summarize(path):
    data = defaultdict(list)
    error_counts = defaultdict(int)
    count = 0
    with path.open(encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            count += 1
            if obj.get('kind')=='observation':
                m=obj['metrics']
                data[m['future_id']].append(m)
            elif obj.get('kind')=='error':
                error_counts[obj.get('future_id','UNKNOWN')] += 1
    print(f'Recorded lines={count}; markets={len(data)}; errors={sum(error_counts.values())}')
    print(f"{'Contract':22} {'Samples':>7} {'Median gross':>13} {'Peak gross':>12} {'10bp net>0':>12} {'90bp net>0':>12} {'Depth OK':>10} {'Errors':>8}")
    for name in sorted(set(data)|set(error_counts)):
        rows=data.get(name, [])
        valid=[r for r in rows if r['spot_one_contract_10lvl_depth']]
        gross=[r['gross_one_lot_vwap_bps'] for r in valid]
        med=f'{statistics.median(gross):.2f}' if gross else 'N/A'
        peak=f'{max(gross):.2f}' if gross else 'N/A'
        low=sum((r['net_10bps_per_side_USD_PROVISIONAL'] or 0)>0 for r in valid)
        hi=sum((r['net_90bps_per_side_USD_PROVISIONAL'] or 0)>0 for r in valid)
        def frac(n,den): return f'{n}/{den}' if den else '0/0'
        print(f'{name:22} {len(rows):7d} {med:>13} {peak:>12} {frac(low,len(valid)):>12} {frac(hi,len(valid)):>12} {frac(len(valid),len(rows)):>10} {error_counts.get(name,0):8d}')
    print('ALL net values are hypothetical execution proxies. Futures book size units, fees, fill probability, margin and settlement mismatch unverified.')
    print('Repeated snapshots are correlated observations, not independent arbitrage opportunities or trade returns.')
    return data, error_counts


def self_test():
    t='2026-11-13T14:00:00+00:00'
    d={'status':'LIVE_PUBLIC_BOOK_DIAGNOSTIC_NOT_EXECUTABLE_ARB',
       'fetched_between_utc':[t,t], 'book_timestamp_skew_s':0.5,
       'future_metadata':{'product_id':'BIT-27NOV26-CDE','contract_root_unit':'BTC',
                          'product_type':'FUTURE', 'display_name':'BTC 27-Nov',
                          'contract_size':'0.01','contract_expiry_utc':'2026-11-27T16:00:00+00:00'},
       'spot_book':{'product_id':'BTC-USD','book_time_utc':t,
                    'bids':[{'price':69200,'size_as_returned':.1}],
                    'asks':[{'price':69300,'size_as_returned':.005},
                            {'price':69310,'size_as_returned':.015}]},
       'future_book':{'product_id':'BIT-27NOV26-CDE','book_time_utc':t,
                      'bids':[{'price':69700,'size_as_returned':30}],
                      'asks':[{'price':69710,'size_as_returned':25}]}}
    x=evaluate(d)
    assert x['spot_one_contract_10lvl_depth'] and not x['spot_one_contract_TOP_depth']
    assert abs(x['spot_vwap_one_contract']-69305)<1e-6
    assert x['is_executable_trade_signal'] is False
    assert x['net_10bps_per_side_USD_PROVISIONAL'] is not None
    d['spot_book']['asks'] = d['spot_book']['asks'][:1]
    x=evaluate(d)
    assert not x['spot_one_contract_10lvl_depth']
    assert x['gross_one_lot_vwap_bps'] is None
    d['book_timestamp_skew_s']=4.0
    try:
        evaluate(d)
        raise AssertionError('skew incorrectly accepted')
    except ValueError:
        pass
    print('PASS V4.3: 10-level VWAP, limited depth, rejection of stale-skew pairs, no trade signals')
    print('OFFLINE self-test: no market requests and no orders')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['self-test','collect','summarize'])
    p.add_argument('--output', type=Path, default=Path('src/crypto-arb/data/v4/coinbase_observer/v4_snapshots.jsonl'))
    p.add_argument('--samples', type=int, default=60)
    p.add_argument('--interval-s', type=float, default=60.0)
    p.add_argument('--future-id', action='append', help='Optional exact contract ID; repeat for multiple')
    p.add_argument('--max-skew-s', type=float, default=2.0)
    p.add_argument('--futures-roundtrip-usd', type=float, default=.20)
    p.add_argument('--index-mismatch-bps', type=float, default=10.0)
    p.add_argument('--financing-apr-pct', type=float, default=0.0)
    a=p.parse_args()
    if a.action=='self-test':
        self_test()
    elif a.action=='summarize':
        summarize(a.output)
    else:
        if not (1<=a.samples<=100000): p.error('samples out of range')
        if not (10<=a.interval_s<=86400 and math.isfinite(a.interval_s)):
            p.error('interval-s must be 10..86400')
        if not (0<a.max_skew_s<=10 and math.isfinite(a.max_skew_s)):
            p.error('max-skew-s must be 0..10')
        for name in ('futures_roundtrip_usd','index_mismatch_bps','financing_apr_pct'):
            v=getattr(a,name)
            if not (math.isfinite(v) and v>=0): p.error(name + ' must be nonnegative')
        collect(a.output, samples=a.samples, interval_s=a.interval_s,
                subset=set(a.future_id or []), max_skew_s=a.max_skew_s,
                futures_roundtrip_usd=a.futures_roundtrip_usd,
                mismatch_reserve_bps=a.index_mismatch_bps,
                financing_apr_pct=a.financing_apr_pct)
        summarize(a.output)

if __name__=='__main__':
    main()
