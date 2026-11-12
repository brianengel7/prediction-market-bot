#!/usr/bin/env python3
"""Audit whether wallet 954B's historical archive omitted sells and CTF inventory actions.

READ-ONLY: public Polymarket Data API v2 GETs and local JSONL reads only.
No orders, API keys, source rewrites, GitHub changes, or downloaded output files.

Run from the prediction-market-bot repository root:
  python3 -B src/copy-trading/w954_polymarket_sell_source_audit_readonly.py --self-test
  python3 -B src/copy-trading/w954_polymarket_sell_source_audit_readonly.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

API = 'https://data-api.polymarket.com'
DEFAULT_WALLET = '0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0'
DEFAULT_FILE = Path.home() / 'poly_wallet_comparison_20261105' / 'wallet_954b' / 'trades.jsonl'
PATTERN = re.compile(r'btc-updown-15m-(\d{10})', re.I)


def epoch(dt: str) -> int:
    return int(datetime.fromisoformat(dt.replace('Z', '+00:00')).replace(tzinfo=timezone.utc).timestamp())


def market_start(item: dict):
    for field in ('slug', 'event_slug', 'eventSlug', 'market_slug', 'title'):
        m = PATTERN.search(str(item.get(field) or ''))
        if m:
            return int(m.group(1))
    return None


def source_inventory(file: Path):
    counts = Counter()
    starts = set()
    addresses = set()
    if not file.is_file():
        return counts, starts, addresses
    with file.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                counts['non_dict'] += 1
                continue
            counts['total'] += 1
            action = str(row.get('side') or row.get('action') or '').upper()
            counts[action or 'unknown_side'] += 1
            start = market_start(row)
            if start is not None:
                starts.add(start)
            for field in ('proxy_wallet', 'proxyWallet', 'wallet', 'user'):
                v = row.get(field)
                if isinstance(v, str) and v.lower().startswith('0x') and len(v) == 42:
                    addresses.add(v.lower())
    return counts, starts, addresses


def parse_feed(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('data'), list):
        raise RuntimeError('API returned an unexpected format; expected {data:[...], pagination:{...}}')
    pg = raw.get('pagination') or {}
    if not isinstance(pg, dict):
        raise RuntimeError('Unexpected pagination format')
    cursor = pg.get('next_cursor') or pg.get('nextCursor') or raw.get('next_cursor')
    if cursor in ('', None, 'null'):
        cursor = None
    return raw['data'], cursor


def public_get(path, params):
    url = API + path + '?' + urllib.parse.urlencode(params)
    for attempt in range(4):
        req = urllib.request.Request(url, headers={'accept': 'application/json',
                                                   'user-agent': 'wallet954b-readonly-research/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f'Public GET failed: HTTP {exc.code}, path={path}; no local changes') from exc
            hdr = exc.headers.get('Retry-After', '2')
            try: delay = min(8., max(1., float(hdr)))
            except ValueError: delay = 2.
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == 3:
                raise RuntimeError(f'Public GET/network error: {exc}') from exc
            time.sleep(attempt + 1)
    raise AssertionError('unreachable')


def fetch_pages(path, params, max_pages):
    rows = []
    cursor = None
    encountered = set()
    truncated = False
    for page in range(1, max_pages + 1):
        q = dict(params)
        if cursor:
            q['cursor'] = cursor
        data, next_cursor = parse_feed(public_get(path, q))
        rows.extend(data)
        print(f'  page {page:2d}: {len(data):4d} records; cumulative={len(rows):5d}', flush=True)
        if not next_cursor:
            return rows, False
        if next_cursor in encountered:
            raise RuntimeError('Repeated pagination cursor; refusing to miscount')
        encountered.add(next_cursor)
        cursor = next_cursor
        if not data:
            raise RuntimeError('Empty page with a next cursor; refusing to miscount')
        time.sleep(.12)
    truncated = bool(cursor)
    return rows, truncated


def summarize(title, rows, capped, first_starts, known_starts):
    matched = [r for r in rows if isinstance(r, dict) and market_start(r) is not None]
    group = Counter(market_start(r) for r in matched)
    sides = Counter(str(r.get('side') or '').upper() for r in matched)
    types = Counter(str(r.get('type') or '').upper() for r in matched)
    sample_30 = sum(count for s, count in group.items() if s in first_starts)
    sample_all = sum(count for s, count in group.items() if s in known_starts)
    print(f'\n{title}')
    print('  API rows:', len(rows), '(AT LEAST; max-page cap reached)' if capped else '(all retrieved)')
    print('  BTC 15m rows:', len(matched), '| BTC markets:', len(group))
    print('  BTC 15m rows overlapping 30-market test:', sample_30)
    print('  BTC 15m rows overlapping local market archive:', sample_all)
    print('  BTC sides:', dict(sides), 'types:', dict(types))
    for row in matched[:4]:
        print('   ', str(row.get('timestamp')), str(row.get('side')), str(row.get('type')),
              str(row.get('outcome')), 'price=', row.get('price'), 'size=', row.get('size'),
              'market=', market_start(row),
              'tx=', str(row.get('transaction_hash') or '')[:18])
    return group


def self_test():
    assert market_start({'slug':'btc-updown-15m-1792351800'}) == 1792351800
    assert market_start({'slug':'eth-updown-15m-1792351800'}) is None
    rows, nxt = parse_feed({'data':[{'side':'SELL'}], 'pagination':{'next_cursor':'sample'}})
    assert len(rows) == 1 and nxt == 'sample'
    rows, nxt = parse_feed({'data':[], 'pagination':{'next_cursor':None}})
    assert not rows and nxt is None
    print('PASS: classification, pagination response, UTC parsing, offline self-test')


def main():
    ap = argparse.ArgumentParser(description='Wallet 954B source-sell completeness audit; public GET/read-only')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--wallet', default=DEFAULT_WALLET, help='Exact 0x proxy wallet for Polymarket Data API')
    ap.add_argument('--source', type=Path, default=DEFAULT_FILE)
    ap.add_argument('--start', default='2026-10-06', help='Inclusive UTC date')
    ap.add_argument('--end', default='2026-11-06', help='Exclusive UTC date')
    ap.add_argument('--max-pages', type=int, default=50, help='Per feed safety cap')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not re.fullmatch(r'0x[0-9a-fA-F]{40}', args.wallet):
        ap.error('wallet must be a 0x address with 40 hexadecimal characters')
    if args.max_pages <= 0:
        ap.error('--max-pages must be positive')
    try:
        start = epoch(args.start + 'T00:00:00+00:00')
        end = epoch(args.end + 'T00:00:00+00:00') - 1
    except ValueError as exc:
        ap.error(f'Invalid date: {exc}')
    if start >= end:
        ap.error('End date must follow start date')
    count, starts, addresses = source_inventory(args.source)
    print('WALLET 954B — POLYMARKET SELL / INVENTORY SOURCE AUDIT (PUBLIC GET, READ ONLY)')
    print('=' * 96)
    print('Wallet:', args.wallet, 'UTC date range:', args.start, 'through', args.end, '(exclusive)')
    print('Local archive:', args.source, '(exists)' if args.source.is_file() else '(NOT FOUND)')
    print('Local action counts:', dict(count), 'local BTC 15m starts:', len(starts))
    if addresses:
        print('Wallet addresses observed in archive:', sorted(addresses))
        if args.wallet.lower() not in addresses:
            raise RuntimeError('API wallet address does not match local raw archive. Supply the right --wallet!')
    else:
        print('Local archive has no usable wallet-address field: VERIFY wallet ID before interpreting API results.')
    first30 = set(sorted(starts)[:30])
    base = {'user': args.wallet, 'limit': 500, 'start': start, 'end': end}
    tests = [
        ('TRADE SELL, taker-side only', '/v2/trades', {**base, 'side': 'SELL', 'taker_only': 'true'}),
        ('TRADE SELL, maker and taker', '/v2/trades', {**base, 'side': 'SELL', 'taker_only': 'false'}),
        ('WALLET ACTIVITY SELL trades', '/v2/activity', {**base, 'side': 'SELL', 'type': 'TRADE'}),
        ('WALLET ACTIVITY splits/merges/redeems', '/v2/activity', {**base, 'type': 'SPLIT,MERGE,REDEEM'}),
    ]
    groups = {}
    for title, path, query in tests:
        print('\nFETCH:', title, flush=True)
        try:
            rows, capped = fetch_pages(path, query, args.max_pages)
        except Exception as exc:
            print('  FAILED:', repr(exc))
            continue
        groups[title] = summarize(title, rows, capped, first30, starts)
    all_sells = groups.get('TRADE SELL, maker and taker', Counter())
    first30_sells = sum(c for s, c in all_sells.items() if s in first30)
    print('\nASSESSMENT')
    if first30_sells:
        print(f'  FOUND {first30_sells} SELL-side trade rows in the original first-30 BTC 15m market windows.')
        print('  Do not replay those 30 markets as buy-and-hold. Recover full BUY/SELL chronologies first.')
    elif all_sells:
        print('  Found BTC 15m sells, but none in the first 30 local markets (or markets not aligned).')
        print('  Reconstruct the source for complete market sets before treating the earlier 30-market ROI as full copy.')
    else:
        print('  No matching BTC 15m sells established by this audit; check failed/capped feeds above.')
        print('  A buy-only position could be intentional, or trades might be unobserved in available APIs.')
    print('  MERGE/REDEEM/SPLIT is an INVENTORY action, not a trade SELL; model separately.')
    print('  Never equate absence from a feed with proof of no sales.')
    print('\nNo local writes, order submissions, or GitHub changes. Public GETs only.')


if __name__ == '__main__':
    main()
