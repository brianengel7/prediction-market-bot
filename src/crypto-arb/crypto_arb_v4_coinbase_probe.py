#!/usr/bin/env python3
"""Coinbase public-market-data connectivity probe for V4 dated-futures research.

READ ONLY: GET requests to Coinbase public Advanced Trade endpoints only.
Discovers public expiring futures metadata and inspects spot/futures book quotes.
Never emits V4 scanner candidate CSV: futures book quantity units and settlement
rules must first be independently verified. No trading, credentials, orders.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

BASE = 'https://api.coinbase.com/api/v3/brokerage/market'


def request_json(path: str, params=None, session=None):
    client = session or requests
    url = BASE + path
    try:
        response = client.get(url, params=params, timeout=20, headers={'Accept': 'application/json'})
        response.raise_for_status()
        out = response.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f'Public endpoint unavailable: {path} ({type(e).__name__}: {str(e)[:180]})') from e
    if not isinstance(out, dict) or 'error' in out:
        raise RuntimeError(f'Invalid/failed public response for {path}: {str(out)[:180]}')
    return out


def utc(s):
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            return None
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def metadata(p, now=None):
    now = now or datetime.now(timezone.utc)
    f = p.get('future_product_details') or {}
    if not isinstance(f, dict):
        return None
    typ = str(p.get('product_type') or '').upper()
    expiry_type = str(f.get('contract_expiry_type') or '').upper()
    exp = utc(f.get('contract_expiry'))
    # Fail closed: need explicit future classification and future expiration.
    # Coinbase may tag perpetual-style contracts as EXPIRING with synthetic
    # expiry dates (e.g. 2089-12-30). Exclude both explicit 'Perp' names and
    # unusually distant contracts, rather than assuming every expiry is real.
    name = str(p.get('display_name') or '').lower()
    if ('FUTURE' not in typ or not exp or exp <= now
            or 'PERPETUAL' in expiry_type or 'perp' in name
            or exp > now + timedelta(days=730)):
        return None
    return {
        'product_id': p.get('product_id'),
        'product_type': p.get('product_type'),
        'product_venue': p.get('product_venue'),
        'display_name': p.get('display_name'),
        'contract_code': f.get('contract_code'),
        'contract_size': f.get('contract_size'),
        'contract_root_unit': f.get('contract_root_unit'),
        'contract_expiry_utc': exp.isoformat(),
        'contract_expiry_type': f.get('contract_expiry_type'),
        'future_venue': f.get('venue'),
        'status': p.get('status'),
    }


def discover(session=None, limit_pages=5):
    found, cursor, seen = [], None, set()
    for _ in range(limit_pages):
        params = {'limit': 100, 'product_type': 'FUTURE', 'get_all_products': 'true'}
        if cursor:
            params['cursor'] = cursor
        data = request_json('/products', params, session)
        products = data.get('products')
        if not isinstance(products, list):
            raise RuntimeError('Products response has no products array')
        for p in products:
            if isinstance(p, dict):
                row = metadata(p)
                if row and row['product_id'] and row['product_id'] not in seen:
                    found.append(row)
                    seen.add(row['product_id'])
        page = data.get('pagination') or {}
        nxt = page.get('next_cursor') if isinstance(page, dict) else None
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    return found


def extract_book(payload, expected_id: str):
    pricebook = payload.get('pricebook')
    if not isinstance(pricebook, dict) or pricebook.get('product_id') != expected_id:
        raise ValueError('Missing book or product ID mismatch')
    t = utc(pricebook.get('time'))
    if not t:
        raise ValueError('Missing or unparsable EXCHANGE book timestamp')
    result = {'product_id': expected_id, 'book_time_utc': t.isoformat()}
    for k in ('bids', 'asks'):
        levels = pricebook.get(k)
        if not isinstance(levels, list) or not levels:
            raise ValueError(f'No {k} in product book')
        result[k] = []
        for item in levels[:10]:
            price, size = float(item.get('price', 'nan')), float(item.get('size', 'nan'))
            if not (math.isfinite(price) and math.isfinite(size) and price > 0 and size > 0):
                raise ValueError(f'Invalid {k} price or size')
            result[k].append({'price': price, 'size_as_returned': size})
    if result['bids'][0]['price'] > result['asks'][0]['price']:
        raise ValueError('Crossed book')
    return result


def one_book(product_id, session=None):
    payload = request_json('/product_book', {'product_id': product_id, 'limit': 10}, session)
    return extract_book(payload, product_id)


def probe(future_id: str, spot_id: str = 'BTC-USD', out_dir=None, session=None):
    product = request_json('/products/' + future_id, session=session)
    # Get Public Product normally returns the fields at top-level.
    if isinstance(product.get('product'), dict):
        product = product['product']
    m = metadata(product)
    if not m or m['product_id'] != future_id:
        raise RuntimeError('Not an explicitly identified EXPIRING FUTURES product; refusing snapshot')
    recv_before = datetime.now(timezone.utc)
    spot = one_book(spot_id, session)
    fut = one_book(future_id, session)
    recv_after = datetime.now(timezone.utc)
    max_age = 30
    for b in (spot, fut):
        dt = utc(b['book_time_utc'])
        age = (recv_after - dt).total_seconds()
        if age < -3 or age > max_age:
            raise RuntimeError(f'{b["product_id"]} book age {age:.1f}s outside [-3,{max_age}]; refusing snapshot')
    skew = abs((utc(spot['book_time_utc']) - utc(fut['book_time_utc'])).total_seconds())
    if skew > 10:
        raise RuntimeError(f'Spot/futures book timestamp skew {skew:.1f}s >10s; refusing snapshot')
    data = {
        'status': 'LIVE_PUBLIC_BOOK_DIAGNOSTIC_NOT_EXECUTABLE_ARB',
        'fetched_between_utc': [recv_before.isoformat(), recv_after.isoformat()],
        'future_metadata': m,
        'spot_book': spot,
        'future_book': fut,
        'book_timestamp_skew_s': skew,
        'limitations': [
            'Book levels not tested for reachable exchange depth or trades',
            'Futures book level size unit is UNVERIFIED (do not assume contracts)',
            'Contract root unit/size and final settlement methodology require confirmation',
            'Spot and futures bids/asks were fetched sequentially, not atomically',
            'Do not feed to V4 scan until validated',
        ],
    }
    if out_dir:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f'probe_{future_id.replace("/", "_")}.json'
        file.write_text(json.dumps(data, indent=2) + '\n')
        print('Saved read-only live-book diagnostic:', file)
    return data


def self_test():
    # No network calls during tests.
    t = datetime(2026,11,13,tzinfo=timezone.utc)
    p = {'product_id':'TEST-FUTURE','product_type':'FUTURE',
         'future_product_details': {'contract_expiry':'2026-12-01T16:00:00Z',
                                    'contract_expiry_type':'EXPIRING','contract_size':'0.01'}}
    assert metadata(p, t) is not None
    q = json.loads(json.dumps(p))
    q['future_product_details']['contract_expiry_type'] = 'PERPETUAL'
    assert metadata(q, t) is None
    q = json.loads(json.dumps(p))
    q['future_product_details'].pop('contract_expiry')
    assert metadata(q, t) is None
    q = json.loads(json.dumps(p))
    q['future_product_details']['contract_expiry'] = '2089-12-30T16:00:00Z'
    assert metadata(q, t) is None
    q = json.loads(json.dumps(p))
    q['display_name'] = 'BTC Perp'
    assert metadata(q, t) is None
    b = {'pricebook':{'product_id':'BTC-USD','time':'2026-11-13T14:00:00.005Z',
                      'bids':[{'price':'70000','size':'0.3'}],
                      'asks':[{'price':'70001','size':'0.4'}]}}
    assert extract_book(b,'BTC-USD')['asks'][0]['size_as_returned']==0.4
    try:
        extract_book(b, 'BAD-ID')
        raise AssertionError('wrong product accepted')
    except ValueError:
        pass
    print('PASS V4.1 Coinbase probe: far-future/perp exclusion, metadata, fractional times, book integrity')
    print('OFFLINE self-test: NO requests, NO orders, NO V4 candidate generation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['self-test','discover','probe','batch-probe'])
    parser.add_argument('--future-id',help='Exact expiring futures product_id discovered using discover')
    parser.add_argument('--spot-id',default='BTC-USD')
    parser.add_argument('--output-dir',default='src/crypto-arb/data/v4/coinbase_probe')
    a=parser.parse_args()
    if a.action=='self-test':
        self_test()
    elif a.action=='discover':
        try:
            products=discover()
            print('EXPIRING FUTURES CANDIDATES in public products:',len(products))
            for item in products:
                print(json.dumps(item,sort_keys=True))
            if not products:
                print('No verified expiring futures products exposed. This does NOT mean no futures exist; check venue market-data access.')
        except (RuntimeError, ValueError) as e:
            raise SystemExit('DISCOVERY UNAVAILABLE: '+str(e))
    elif a.action == 'batch-probe':
        # Exact product IDs grounded in the user's public discovery output.
        # Each metadata record and book is independently checked again.
        pairs = [
            ('BIT-27NOV26-CDE', 'BTC-USD'),
            ('BIT-24DEC26-CDE', 'BTC-USD'),
            ('ET-27NOV26-CDE', 'ETH-USD'),
            ('ET-24DEC26-CDE', 'ETH-USD'),
            ('SOL-27NOV26-CDE', 'SOL-USD'),
            ('SOL-24DEC26-CDE', 'SOL-USD'),
        ]
        ok = 0
        for fut_id, spot_id in pairs:
            try:
                result = probe(fut_id, spot_id, a.output_dir)
                print(json.dumps({
                    'future_id': fut_id, 'spot_id': spot_id,
                    'status': 'BOOK_OK_DIAGNOSTIC',
                    'spot_ask': result['spot_book']['asks'][0],
                    'future_bid': result['future_book']['bids'][0],
                    'skew_seconds': result['book_timestamp_skew_s'],
                    'warning': 'Futures size units unverified; NOT a trade signal.',
                }))
                ok += 1
            except (RuntimeError, ValueError) as e:
                print(json.dumps({'future_id': fut_id, 'spot_id': spot_id,
                                  'status': 'UNAVAILABLE',
                                  'reason': str(e)[:220]}))
        print(f'BATCH SUMMARY: {ok}/{len(pairs)} complete public-book diagnostics; '
              'no quotes synthesized, no orders, no V4 scanner rows')
    else:
        if not a.future_id:
            parser.error('probe requires --future-id from discovery; do not guess')
        try:
            result=probe(a.future_id,a.spot_id,a.output_dir)
            print(json.dumps({
                'status':result['status'],
                'future_metadata':result['future_metadata'],
                'spot_top_bid':result['spot_book']['bids'][0],
                'spot_top_ask':result['spot_book']['asks'][0],
                'future_top_bid':result['future_book']['bids'][0],
                'future_top_ask':result['future_book']['asks'][0],
                'book_timestamp_skew_s':result['book_timestamp_skew_s'],
            },indent=2))
        except (RuntimeError,ValueError) as e:
            raise SystemExit('PROBE UNAVAILABLE: '+str(e))

if __name__=='__main__':
    main()
