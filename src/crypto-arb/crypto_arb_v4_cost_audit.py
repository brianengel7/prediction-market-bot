#!/usr/bin/env python3
"""V4.2: READ-ONLY fee, depth, and basis audit of saved Coinbase probe snapshots.

Reads *already captured* V4.1 JSON files. Never requests data, sends orders,
assumes futures book quantity units, or emits V4 candidate/quote CSV inputs.

This is a diagnostic using one contract's displayed FUTURES BID and SPOT ASK;
it is NOT an executable, guaranteed or risk-free arbitrage estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROFILES = {
    # Rates are scenarios, NOT verified account-specific fees.
    'US_ENTRY_TAKER_90BPS': 90.0,
    'HYPOTHETICAL_LOW_10BPS': 10.0,
}


def dt(x):
    if not isinstance(x, str):
        raise ValueError('Expected ISO8601 timestamp')
    v = datetime.fromisoformat(x.replace('Z', '+00:00'))
    if not v.tzinfo:
        raise ValueError('Timestamp has no timezone')
    return v.astimezone(timezone.utc)


def number(v, label, *, strictly_positive=True):
    out = float(v)
    if not math.isfinite(out) or (out <= 0 if strictly_positive else out < 0):
        raise ValueError(f'Invalid {label}: {v!r}')
    return out


def levels(book, key):
    obj = book.get(key)
    if not isinstance(obj, list) or not obj:
        raise ValueError(f'Missing {key}')
    return (number(obj[0]['price'], key+' price'),
            number(obj[0]['size_as_returned'], key+' size'))


def one_snapshot(doc, *, max_skew_s=2.0, max_age_s=30.0,
                 futures_entry_fee_usd=0.10, futures_exit_reserve_usd=0.10,
                 index_mismatch_bps=10.0, spot_financing_apr=0.0):
    if doc.get('status') != 'LIVE_PUBLIC_BOOK_DIAGNOSTIC_NOT_EXECUTABLE_ARB':
        raise ValueError('Unrecognized probe status')
    info = doc['future_metadata']
    spot, fut = doc['spot_book'], doc['future_book']
    if fut['product_id'] != info['product_id']:
        raise ValueError('Futures metadata/book mismatch')
    symbol = str(info['contract_root_unit']).upper()
    if spot['product_id'] != symbol + '-USD':
        raise ValueError('Spot root/currency mismatch')
    if str(info.get('product_type')).upper() != 'FUTURE':
        raise ValueError('Not futures')
    if 'perp' in str(info.get('display_name','')).lower():
        raise ValueError('Perpetual masquerading as expiring')
    capture = dt(doc['fetched_between_utc'][1])
    expiry = dt(info['contract_expiry_utc'])
    if expiry <= capture or (expiry-capture).total_seconds()>730*86400:
        raise ValueError('Expired or far-future synthetic expiration')
    spot_t, fut_t = dt(spot['book_time_utc']), dt(fut['book_time_utc'])
    skew = abs((spot_t-fut_t).total_seconds())
    if skew>max_skew_s:
        raise ValueError(f'Spot/future book skew {skew:.3f}s > {max_skew_s}')
    ages = ((capture-spot_t).total_seconds(), (capture-fut_t).total_seconds())
    if any(a < -3 or a > max_age_s for a in ages):
        raise ValueError(f'Stale/future-dated book ages {ages}')
    spot_bid, _ = levels(spot, 'bids')
    spot_ask, spot_ask_size = levels(spot, 'asks')
    future_bid, future_size = levels(fut, 'bids')
    future_ask, _ = levels(fut, 'asks')
    if spot_bid>spot_ask or future_bid>future_ask:
        raise ValueError('Crossed book')
    contract = number(info['contract_size'],'contract_size')
    # Only spot units are interpreted. Futures book size remains UNVERIFIED.
    top_spot_contracts = math.floor((spot_ask_size+1e-12)/contract)
    days = (expiry-capture).total_seconds()/86400.0
    gross_usd = (future_bid-spot_ask)*contract
    gross_bps = 10000*(future_bid/spot_ask-1)
    base = {
        'capture_utc': capture.isoformat(), 'asset': symbol,
        'future_id': info['product_id'], 'expiry_utc': expiry.isoformat(),
        'days_to_expiry':round(days,5), 'contract_size_base':contract,
        'spot_ask':spot_ask, 'futures_bid':future_bid,
        'gross_premium_bps':round(gross_bps,4),
        'gross_per_contract_usd':round(gross_usd,6),
        'spot_top_ask_size_base':spot_ask_size,
        'spot_top_contracts_max':top_spot_contracts,
        'future_bid_size_as_returned_UNVERIFIED':future_size,
        'skew_s':round(skew,6),
        'gross_annualized_basis_pct':round(gross_bps/100 * 365/days,4),
        'futures_size_units_status':'UNVERIFIED',
        'settlement_index_hedge_status':'UNVERIFIED',
        'trade_candidate':False,
        'notes':'One-contract thought experiment; no quote converted into an executable order',
    }
    out=[]
    spot_notional = contract*spot_ask
    for profile, spot_fee_bps in PROFILES.items():
        # Spot exit fee computed using entry spot as a PROXY, not known expiry price.
        fees = spot_notional * (2*spot_fee_bps)/10000
        fin = spot_notional*spot_financing_apr/100 * days/365
        mismatch_reserve = spot_notional * index_mismatch_bps/10000
        total_reserved = (fees+futures_entry_fee_usd+futures_exit_reserve_usd
                          +fin+mismatch_reserve)
        result = dict(base,
            profile=profile, spot_fee_per_side_bps=spot_fee_bps,
            spot_fee_round_trip_usd=round(fees,6),
            futures_fee_plus_exit_reserve_usd=round(futures_entry_fee_usd+futures_exit_reserve_usd,6),
            index_mismatch_reserve_usd=round(mismatch_reserve,6),
            spot_financing_usd=round(fin,6),
            provisional_net_per_contract_usd=round(gross_usd-total_reserved,6),
            break_even_spot_fee_bps_each_side=round(
                (gross_usd-futures_entry_fee_usd-futures_exit_reserve_usd-fin-mismatch_reserve)
                *10000/(2*spot_notional),4),
            )
        if top_spot_contracts < 1:
            result['screen_status'] = 'INSUFFICIENT_SPOT_TOP_DEPTH_FOR_ONE_CONTRACT'
        elif gross_usd <= 0:
            result['screen_status'] = 'NO_POSITIVE_GROSS_PREMIUM'
        elif result['provisional_net_per_contract_usd'] <= 0:
            result['screen_status'] = 'NEGATIVE_AFTER_SCENARIO_COSTS'
        else:
            result['screen_status'] = 'POSITIVE_HYPOTHETICAL_NOT_EXECUTABLE'
        out.append(result)
    return out


def run(directory, output, **kwargs):
    paths = sorted(directory.glob('probe_*.json'))
    if not paths:
        raise RuntimeError(f'No probe_*.json files in {directory}')
    results, errors = [], []
    for path in paths:
        try:
            obj=json.loads(path.read_text())
            results.extend(one_snapshot(obj, **kwargs))
        except (ValueError,KeyError,TypeError,IndexError) as e:
            errors.append((path.name,str(e)))
    if results:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"{'Contract':24s} {'Scenario':25s} {'Gross bps':>10s} {'Net $/contract':>14s} {'Spot 1-lot?':>12s} {'Verdict':>39s}")
        for r in results:
            print(f"{r['future_id']:24s} {r['profile']:25s} {r['gross_premium_bps']:10.2f} {r['provisional_net_per_contract_usd']:14.3f} {str(r['spot_top_contracts_max']>=1):>12s} {r['screen_status']:>39s}")
        print('Saved read-only hypothetical diagnostics to',output)
    for name,reason in errors:
        print('INVALID / SKIPPED:',name,reason)
    print(f'Parsed {len(results)//len(PROFILES)} snapshots; rejected {len(errors)}.')
    print('NO TRADING CANDIDATES, NO ORDERS. Futures size units, actual account fees, margin, spot exit liquidity, final index mismatch are UNVERIFIED.')
    return results,errors


def self_test():
    t=datetime(2026,11,13,14,0,tzinfo=timezone.utc)
    def sample(symbol='BTC',contract=.01,spot_size=.11):
        book_time=t.isoformat()
        return {'status':'LIVE_PUBLIC_BOOK_DIAGNOSTIC_NOT_EXECUTABLE_ARB',
                'fetched_between_utc':[book_time,book_time],
                'future_metadata':{'product_id':'BIT-27NOV26-CDE',
                     'contract_root_unit':symbol,'contract_size':str(contract),
                     'display_name':symbol+' 27-Nov', 'product_type':'FUTURE',
                     'contract_expiry_utc':'2026-11-27T16:00:00+00:00'},
                'spot_book':{'product_id':symbol+'-USD','book_time_utc':book_time,
                    'bids':[{'price':69388.50,'size_as_returned':.1}],
                    'asks':[{'price':69388.51,'size_as_returned':spot_size}]},
                'future_book':{'product_id':'BIT-27NOV26-CDE','book_time_utc':book_time,
                    'bids':[{'price':69535.0,'size_as_returned':27}],
                    'asks':[{'price':69545.0,'size_as_returned':25}]}}
    doc=sample()
    r=one_snapshot(doc)
    assert len(r)==2 and all(x['trade_candidate'] is False for x in r)
    assert abs(r[0]['gross_premium_bps']-21.11)<.02
    assert r[0]['spot_top_contracts_max']==11
    assert r[0]['provisional_net_per_contract_usd']<0
    assert r[1]['provisional_net_per_contract_usd']<0
    doc=sample()
    doc['future_metadata']['contract_expiry_utc']='2026-12-24T16:00:00Z'
    doc['future_book']['bids'][0]['price']=69755.0
    doc['future_book']['asks'][0]['price']=69765.0
    r=one_snapshot(doc)
    assert r[1]['provisional_net_per_contract_usd']>0
    assert r[1]['trade_candidate'] is False
    doc=sample(spot_size=.00003704)
    r=one_snapshot(doc)
    assert all(x['screen_status']=='INSUFFICIENT_SPOT_TOP_DEPTH_FOR_ONE_CONTRACT' for x in r)
    doc=sample()
    doc['future_book']['book_time_utc']='2026-11-13T13:59:56+00:00'
    try:
        one_snapshot(doc)
        raise AssertionError('Allowed skew')
    except ValueError as e:
        assert 'skew' in str(e)
    doc=sample()
    doc['future_metadata']['contract_expiry_utc']='2089-12-30T16:00:00Z'
    try:
        one_snapshot(doc)
        raise AssertionError('Allowed fake perpetual')
    except ValueError as e:
        assert 'far-future' in str(e)
    print('PASS V4.2: fee scenarios, one-contract basis, exact top depth, skew and fake-perp guard')
    print('OFFLINE READ ONLY; NO market requests, orders, executable-candidate output')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['self-test','audit'])
    p.add_argument('--input-dir',type=Path,default=Path('src/crypto-arb/data/v4/coinbase_probe'))
    p.add_argument('--output',type=Path,default=Path('src/crypto-arb/data/v4/coinbase_cost_audit.csv'))
    p.add_argument('--max-skew-s',type=float,default=2)
    p.add_argument('--max-age-s',type=float,default=30)
    p.add_argument('--futures-entry-fee-usd',type=float,default=.10)
    p.add_argument('--futures-exit-reserve-usd',type=float,default=.10)
    p.add_argument('--index-mismatch-bps',type=float,default=10)
    p.add_argument('--spot-financing-apr',type=float,default=0)
    a=p.parse_args()
    if a.action=='self-test':
        self_test()
        return
    keys=['max_skew_s','max_age_s','futures_entry_fee_usd',
          'futures_exit_reserve_usd','index_mismatch_bps','spot_financing_apr']
    vals={key:getattr(a,key) for key in keys}
    if any(not math.isfinite(v) or v<0 for v in vals.values()):
        p.error('All numeric parameters must be finite, nonnegative')
    if vals['max_skew_s']<=0 or vals['max_age_s']<=0:
        p.error('max-skew-s and max-age-s must be positive')
    run(a.input_dir,a.output,**vals)


if __name__=='__main__':
    main()
