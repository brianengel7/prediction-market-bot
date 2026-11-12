#!/usr/bin/env python3
"""Compare asdcaxz and wallet_954b using local Polymarket trade/activity JSONL.

READ ONLY: standard library; no network, file writes, credentials, or orders.

The observed trade records start October 6 and end November 5, 2026.
A complete simple market ledger requires every bought share to be accounted
for by a REDEEM of the same token, with no SELL/MERGE/SPLIT activity. Gross
cash P&L is then payout minus purchase spending (excluding fees/rebates).
All incomplete markets are left out of realized-P&L estimates.

This is NOT a maker order-fill or live copyability backtest. It cannot reveal
unfilled maker orders, cancellations, book queue position, or external signals.
"""

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path.home() / 'poly_wallet_comparison_20261105'
WALLETS = ('asdcaxz', 'wallet_954b')
TIME_RE = re.compile(r'Up or Down\s*-\s*([A-Za-z]+\s+\d{1,2}),\s*(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)\s*ET', re.I)
SLUG_RE = re.compile(r'(?:^|-)updown-(5m|15m)-(\d{10})(?:$|-)', re.I)
ET = ZoneInfo('America/New_York')


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def dollars(v):
    return '${:,.2f}'.format(v)


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f'Unexpected JSONL record: {path}')
                yield obj


@dataclass
class Market:
    cid: str
    title: str = ''
    slug: str = ''
    trades: list = field(default_factory=list)
    redeems: list = field(default_factory=list)
    other_ops: Counter = field(default_factory=Counter)
    rebate_usdc: float = 0.0


def load_wallet(base, wallet):
    markets = {}
    counters = Counter()
    def market_for(r):
        cid = str(r.get('condition_id') or '').lower()
        if not cid:
            counters['missing_condition'] += 1
            return None
        m = markets.setdefault(cid, Market(cid))
        if not m.title:
            m.title = str(r.get('title') or '')
        if not m.slug:
            m.slug = str(r.get('slug') or r.get('event_slug') or '')
        return m

    for r in jsonl(base / wallet / 'trades.jsonl'):
        m = market_for(r)
        if m is None:
            continue
        action = str(r.get('side') or '').upper()
        ts, size, price = (number(r.get(k)) for k in ('timestamp', 'size', 'price'))
        token = str(r.get('token_id') or '')
        if action not in ('BUY', 'SELL') or not token or ts is None or size is None or price is None or not (0 <= price <= 1 and size > 0):
            counters['bad_trade'] += 1
            continue
        m.trades.append((ts, token, action, price, size))
        counters['trade_rows'] += 1

    for r in jsonl(base / wallet / 'activity.jsonl'):
        m = market_for(r)
        if m is None:
            continue
        op = str(r.get('type') or '').upper()
        if op == 'TRADE':
            continue  # exchange fills already accounted for via trades.jsonl
        ts = number(r.get('timestamp'))
        token = str(r.get('token_id') or '')
        qty = number(r.get('size'))
        cash = number(r.get('usdc_size'))
        if op == 'REDEEM':
            if not token or ts is None or qty is None or cash is None or qty < 0 or cash < 0:
                m.other_ops['BAD_REDEEM'] += 1
                counters['bad_redeem'] += 1
                continue
            m.redeems.append((ts, token, qty, cash))
            counters['redeem_rows'] += 1
        elif op in ('MAKER_REBATE', 'TAKER_REBATE'):
            if cash is not None:
                m.rebate_usdc += cash
            counters[op] += 1
        elif op:
            m.other_ops[op] += 1
            counters[op] += 1
    return markets, counters


def market_window(m):
    match = SLUG_RE.search(m.slug)
    if match:
        return int(match.group(2)), 300 if match.group(1).lower() == '5m' else 900
    match = TIME_RE.search(m.title)
    if not match:
        return None, None
    day, a, b = match.groups()
    try:
        start = datetime.strptime(f'{day} 2026 {a.replace(" ", "")}', '%B %d %Y %I:%M%p')
        end = datetime.strptime(f'{day} 2026 {b.replace(" ", "")}', '%B %d %Y %I:%M%p')
    except ValueError:
        return None, None
    dur = int((end - start).total_seconds()) % 86400
    if dur not in (300, 900):
        return None, None
    # Fall-back DST 01:00 ET is ambiguous without an epoch-bearing market slug.
    if start.month == 11 and start.day == 1 and start.hour == 1:
        return None, dur
    return start.replace(tzinfo=ET).timestamp(), dur


def cohort(elapsed, duration):
    if elapsed is None or duration is None:
        return 'UNKNOWN'
    if elapsed < 0:
        return 'BEFORE_START'
    if elapsed >= duration:
        return 'AFTER_END'
    if elapsed < 60:
        return 'FIRST_60S'
    if elapsed < duration / 3:
        return 'EARLY_MIDDLE'
    if elapsed < duration - 60:
        return 'LATE_MIDDLE'
    return 'LAST_60S'


def analyze(m):
    buys = defaultdict(float)
    redeem_burns = defaultdict(float)
    actions = Counter()
    total_spend = 0.0
    proceeds = 0.0
    sales = 0.0
    first = None
    second_lag = None
    cost_path = 0.0
    share_path = defaultdict(float)
    floor_max = -float('inf')
    floor_at_end = None
    first_positive_at = None
    first_side_token = None
    for ts, token, action, price, qty in sorted(m.trades, key=lambda r: (r[0], r[1])):
        actions[action] += 1
        if action == 'BUY':
            if first is None:
                first = ts
                first_side_token = token
            if second_lag is None and token != first_side_token:
                second_lag = max(0.0, ts - first)
            buys[token] += qty
            share_path[token] += qty
            total_spend += qty * price
            cost_path += qty * price
            if len(share_path) <= 2:
                current_floor = (min(share_path.values()) if len(share_path) == 2 else 0.0) - cost_path
                floor_max = max(floor_max, current_floor)
                floor_at_end = current_floor
                if current_floor > 0.000001 and first_positive_at is None:
                    first_positive_at = ts
        else:
            sales += qty * price
    for ts, token, qty, cash in m.redeems:
        redeem_burns[token] += qty
        proceeds += cash
    start, duration = market_window(m)
    elapsed = first - start if first is not None and start is not None else None
    both = len(buys) == 2
    unexpected_tokens = len(buys) > 2 or any(tok not in buys for tok in redeem_burns)
    # Matching the actual token quantities is a strict, observable-coverage test.
    # This intentionally excludes missing burned losers or prior inventory.
    token_match = (
        bool(buys) and bool(m.redeems) and not unexpected_tokens
        and all(abs(buys[t] - redeem_burns.get(t, 0.0))
                <= max(0.03, buys[t] * 0.00001) for t in buys)
    )
    clean = token_match and actions['SELL'] == 0 and not m.other_ops
    if start is not None and first is not None and first + 1 < start:
        clean = False  # Cannot establish entire market history in the window.
    winning_tokens = {token for _, token, qty, cash in m.redeems if cash > 0.00001 and qty > 0}
    first_won = first_side_token in winning_tokens if clean and len(winning_tokens) == 1 else None
    return {
        'title': m.title, 'start': start, 'duration': duration, 'elapsed': elapsed,
        'cohort': cohort(elapsed, duration), 'first': first,
        'buys': actions['BUY'], 'sells': actions['SELL'],
        'both': both, 'second_lag': second_lag,
        'first_won': first_won, 'tokens': len(buys), 'clean': clean,
        'redemptions': len(m.redeems), 'spend': total_spend,
        'redeem': proceeds, 'sales': sales,
        'pnl': proceeds - total_spend if clean else None,
        'floor_max': floor_max if len(buys) == 2 else None,
        'floor_end': floor_at_end if len(buys) == 2 else None,
        'first_floor_positive_at': first_positive_at,
        'other_ops': dict(m.other_ops), 'token_match': token_match,
        'open_shares': sum(max(0, qty - redeem_burns.get(token, 0)) for token, qty in buys.items()),
    }


def pct(a, b):
    return f'{100 * a / b:.1f}%' if b else 'N/A'


def median(values):
    return statistics.median(values) if values else None


def report(wallet, mkt, stats, details):
    totals = Counter()
    by_dur = defaultdict(list)
    for d in details:
        totals['all'] += 1
        totals['buys'] += d['buys']
        if d['both']:
            totals['both'] += 1
        if d['token_match']:
            totals['token_matched'] += 1
        if d['clean']:
            totals['clean'] += 1
        if d['floor_end'] is not None and d['floor_end'] > 0.000001:
            totals['end_floor_positive'] += 1
        if d['floor_max'] is not None and d['floor_max'] > 0.000001:
            totals['ever_floor_positive'] += 1
        by_dur[d['duration']].append(d)

    clean = [d for d in details if d['clean']]
    print('\n' + '=' * 94)
    print(wallet.upper() + ' | REVERSE-ENGINEERING FEASIBILITY | 2026-10-06 THROUGH 2026-11-05')
    print('=' * 94)
    print('Markets observed:', totals['all'], '| Trades:', stats['trade_rows'],
          '| REDEEM records:', stats['redeem_rows'])
    print('BUY/REDEEM token-quantity matches:', totals['token_matched'],
          '| Simple fully accounted markets:', totals['clean'],
          f'({pct(totals["clean"], totals["all"])})')
    print('Markets buying both tokens:', totals['both'], f'({pct(totals["both"], totals["all"])})')
    print('Both-token markets ever reaching POSITIVE gross worst-case payout floor:',
          totals['ever_floor_positive'])
    print('Both-token markets ending with POSITIVE gross worst-case payout floor:',
          totals['end_floor_positive'])
    print('NOTE: positive floor depends on observed purchases only; excludes fees, fill delays and missing inventory.')
    print('Other activity operations:', {k:v for k,v in stats.items()
                                         if k not in ('trade_rows','redeem_rows') and v})

    print('\nBY CONTRACT DURATION (only market records with this duration)')
    print(f'{"Length":>8} {"Markets":>9} {"Both":>8} {"Clean":>8} {"Gross P&L ($)":>16} {"ROI on cost":>13} {"Clean wins":>12}')
    for length in (300, 900, None):
        group = by_dur.get(length, [])
        if not group:
            continue
        done = [d for d in group if d['clean']]
        cost = sum(d['spend'] for d in done)
        pnl = sum(d['pnl'] for d in done)
        wins = sum(d['pnl'] > 0 for d in done)
        print(f'{(str(length//60)+"m" if length else "UNKNOWN"):>8} {len(group):>9,} {sum(d["both"] for d in group):>8,} '
              f'{len(done):>8,} {dollars(pnl):>16} {pct(pnl,cost):>13} {str(wins)+"/"+str(len(done)):>12}')
    print('Gross simple-ledger cash P&L excludes fees/rebates and is NOT official wallet P&L.')

    print('\nBY MARKET INVENTORY TYPE (clean markets only)')
    for label, selected in [
        ('ONE-SIDED', [d for d in clean if not d['both']]),
        ('TWO-SIDED, FLOOR>0', [d for d in clean if d['both'] and (d['floor_end'] or 0) > 0]),
        ('TWO-SIDED, FLOOR<=0', [d for d in clean if d['both'] and (d['floor_end'] or 0) <= 0]),
    ]:
        cost = sum(d['spend'] for d in selected)
        pnl = sum(d['pnl'] for d in selected)
        print(f'{label:<26} markets={len(selected):>6,} P&L={dollars(pnl):>14} '
              f'ROI={pct(pnl, cost):>8} winners={sum(d["pnl"]>0 for d in selected)}/{len(selected)}')

    print('\nFIRST ENTRY RELATIVE TO MARKET START')
    for length in (300, 900):
        group = [d for d in details if d['duration'] == length]
        if not group:
            continue
        counts = Counter(d['cohort'] for d in group)
        print(f'{length//60}m: ' + ', '.join(f'{k}={counts[k]} ({pct(counts[k],len(group))})'
                                   for k in ('FIRST_60S','EARLY_MIDDLE','LATE_MIDDLE','LAST_60S','BEFORE_START','AFTER_END','UNKNOWN')))

    lags = [d['second_lag'] for d in details if d['second_lag'] is not None]
    if lags:
        print('\nFirst opposite token purchase: median lag =',
              f'{median(lags):.1f}s across {len(lags):,} markets')
        for limit in (5, 30, 60, 120, 300):
            print(f'  within {limit:3}s: {sum(x <= limit for x in lags):,}/{len(lags):,}')
    known_winner = [d for d in clean if d['first_won'] is not None]
    print('First purchased side eventually won among clean settled markets:',
          f'{sum(d["first_won"] for d in known_winner)}/{len(known_winner)}',
          '(not risk-adjusted edge; purchase prices and both-side positions matter)')

    print('\nCLEAN SETTLED MARKETS: TOP 5 GROSS WINNERS AND LOSERS')
    for label, rows in (
        ('WINNERS', sorted((d for d in clean if d['pnl'] > 0), key=lambda d: d['pnl'], reverse=True)[:5]),
        ('LOSERS', sorted((d for d in clean if d['pnl'] < 0), key=lambda d: d['pnl'])[:5]),
    ):
        print(label, '(none in clean sample)' if not rows else '')
        for d in rows:
            print(f'  {dollars(d["pnl"]):>13} | {d["duration"]//60 if d["duration"] else "?":>2}m | '
                  f'{"BOTH" if d["both"] else "ONE ":<4} | cost={dollars(d["spend"]):>12} '
                  f'| {d["title"][:78]}')

    print('\nDATA QUALITY:')
    print('  Incomplete markets and missing redeemed token amounts are excluded from cash P&L.')
    print('  A complete BUY-to-REDEEM share match is evidence of coverage, not proof of no token transfers.')
    print('  Full strategy reverse engineering still needs historical depth, maker order placement/cancels,')
    print('  contemporaneous underlying prices, exact fees and observation-latency timestamps.')


def self_test():
    # Both tokens bought for 45c and 50c => 5c guaranteed gross if both settled.
    a = Market('test', 'Bitcoin Up or Down - October 20, 12:00PM-12:05PM ET')
    start = datetime(2026, 10, 20, 16, 0, tzinfo=timezone.utc).timestamp()
    a.trades = [(start+5, 'UP', 'BUY', .45, 10), (start+15, 'DOWN', 'BUY', .50, 10)]
    a.redeems = [(start+400, 'UP', 10, 10), (start+400, 'DOWN', 10, 0)]
    d = analyze(a)
    assert d['clean'] and abs(d['pnl'] - 0.5) < 1e-9
    assert abs(d['floor_end'] - 0.5) < 1e-9 and d['second_lag'] == 10
    b = Market('test2')
    b.trades = [(100,'UP','BUY',.2,10)]
    b.redeems = [(401,'UP',5,5)]
    assert not analyze(b)['clean']
    c = Market('test3')
    c.trades = [(100,'DOWN','BUY',.8,10)]
    c.redeems = [(400,'DOWN',10,0)]
    assert analyze(c)['clean'] and abs(analyze(c)['pnl'] + 8) < 1e-9
    print('PASS: paired floor / gross profit / partial redemption exclusion / losing directional trade')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--wallets', nargs='+', choices=WALLETS, default=WALLETS)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    for wallet in args.wallets:
        trades = args.root / wallet / 'trades.jsonl'
        activity = args.root / wallet / 'activity.jsonl'
        if not trades.is_file() or not activity.is_file():
            p.error(f'Missing {trades} or {activity}; run the existing wallet collector first')
        markets, counts = load_wallet(args.root, wallet)
        details = [analyze(m) for m in markets.values() if m.trades]
        report(wallet, markets, counts, details)
    print('\nREAD ONLY — no network, file writes, GitHub operations, credentials, or trades.')


if __name__ == '__main__':
    main()
