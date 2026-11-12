#!/usr/bin/env python3
"""Wallet 954B post-first-fill sequence audit (read only).

Run from repository root:
  python3 -B src/copy-trading/w954_post_entry_sequence_audit_readonly.py

Uses the already-local first-fill loader and raw trade cache. No API, no orders,
no writes. The first-fill ROI measures hypothetical holding ONLY that first buy
until expiry, not actual realized portfolio P&L. Post-entry groupings are
DESCRIPTIVE and include information unavailable at first-entry time.
"""

import argparse
import json
import math
import re
import runpy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

DEFAULT_PATHS = (
    Path.home() / 'poly_wallet_comparison_20261105' / 'wallet_954b' / 'trades.jsonl',
    Path.home() / 'poly_wallet_comparison' / '20261105' / 'wallet_954b' / 'trades.jsonl',
    Path.home() / 'poly_wallet_20261104_v1' / 'wallet_954b' / 'trades.jsonl',
)
SLUG_RE = re.compile(r'btc-updown-15m-(\d{10})', re.I)


def fnum(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def epoch(x):
    if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().replace('.', '', 1).isdigit()):
        v = fnum(x)
        if v is None:
            return None
        if v > 1e12:  # milliseconds
            v /= 1000
        return int(v)
    if isinstance(x, str):
        try:
            return int(datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp())
        except ValueError:
            pass
    return None


def outcome(s):
    s = str(s or '').strip().upper()
    return s if s in ('UP', 'DOWN') else None


def first_present(obj, keys):
    for k in keys:
        v = obj.get(k)
        if v is not None and v != '':
            return v
    return None


def key_from_trade(row, ts):
    for k in ('slug', 'eventSlug', 'event_slug', 'market_slug', 'event', 'title'):
        v = row.get(k)
        if isinstance(v, str):
            m = SLUG_RE.search(v)
            if m:
                return int(m.group(1)), True
    start = epoch(first_present(row, ('start', 'start_time', 'market_start', 'startDate')))
    if start is not None and start % 900 == 0:
        return start, True
    if ts is None:
        return None, False
    return (ts // 900) * 900, False


def load_trades(path):
    grouped = defaultdict(list)
    raw_count = 0
    invalid = Counter()
    condition_by_start = defaultdict(set)
    slug_confirmed = 0
    with path.open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw_count += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                invalid['bad_json'] += 1
                continue
            if not isinstance(r, dict):
                invalid['not_object'] += 1
                continue
            ts = epoch(first_present(r, ('timestamp', 'ts', 'time', 'created_at')))
            price = fnum(first_present(r, ('price', 'fill_price')))
            qty = fnum(first_present(r, ('size', 'shares', 'quantity', 'qty')))
            action = str(first_present(r, ('side', 'action', 'type')) or '').upper()
            coin_side = outcome(first_present(r, ('outcome', 'outcome_name', 'token_outcome')))
            if ts is None or price is None or qty is None or qty <= 0 or not (0 < price <= 1) or action not in ('BUY', 'SELL') or coin_side is None:
                invalid['missing_core_fields'] += 1
                continue
            start, confirmed = key_from_trade(r, ts)
            if start is None or not (start <= ts < start + 900):
                invalid['outside_market'] += 1
                continue
            if confirmed:
                slug_confirmed += 1
            cond = first_present(r, ('conditionId', 'condition_id', 'conditionID', 'market'))
            if cond:
                condition_by_start[start].add(str(cond))
            grouped[start].append(dict(ts=ts, side=coin_side, action=action, price=price,
                                       size=qty, notional=qty*price, line=line_no))
    for market in grouped.values():
        market.sort(key=lambda x: (x['ts'], x['line']))
    collisions = {k:len(v) for k,v in condition_by_start.items() if len(v)>1}
    return grouped, raw_count, invalid, slug_confirmed, collisions


def price_band(p):
    if .20 <= p < .40: return '20-40c'
    if .40 <= p <= .60: return '40-60c'
    if .60 < p < .80: return '60-80c'
    if .80 <= p <= 1: return '80-100c'
    if 0 < p < .20: return '<20c'
    return 'unknown'


def classify(r, trades):
    t0 = int(r['ts'])
    selected = [t for t in trades if t['ts'] >= t0 - 2]
    if not selected:
        return None
    # Skip activity within 3 sec of first fill; often a single fragmented order.
    after = [t for t in selected if t['ts'] > t0 + 3]
    buys = [t for t in after if t['action'] == 'BUY']
    sells = [t for t in after if t['action'] == 'SELL']
    side = outcome(r['side'])
    same = [t for t in buys if t['side'] == side]
    opposite = [t for t in buys if t['side'] != side]
    if same and opposite: pattern = 'both-side additions'
    elif same: pattern = 'same-side additions'
    elif opposite: pattern = 'opposite-side additions'
    else: pattern = 'no late BUY'
    first_buy = [t for t in trades if t['action'] == 'BUY' and t['side'] == side and abs(t['ts']-t0)<=3]
    near = any(abs(t['price']-float(r['price'])) <= .015 for t in first_buy)
    return {
        'pattern': pattern, 'sell': bool(sells), 'same': len(same), 'opp': len(opposite),
        'later_buy': len(buys), 'later_sell': len(sells), 'buy_notional': sum(t['notional'] for t in trades if t['action']=='BUY'),
        'buy_count': sum(t['action']=='BUY' for t in trades), 'sell_notional':sum(t['notional'] for t in trades if t['action']=='SELL'),
        'first_opp_s': min((t['ts']-t0 for t in opposite), default=None),
        'first_same_s': min((t['ts']-t0 for t in same), default=None),
        'close_match': near,
    }


def summary(label, sample):
    if not sample:
        print(f'{label:<27} N=   0')
        return
    cash = sum(r['cost'] for r in sample)
    pnl = sum(r['pnl'] for r in sample)
    win = sum(bool(r['won']) for r in sample)
    price = median(float(r['price']) for r in sample)
    roi = 100 * pnl / cash if cash > 0 else float('nan')
    print(f'{label:<27} N={len(sample):4} first_win={100*win/len(sample):5.1f}% '
          f'first_hold_ROI={roi:+7.2f}% first_PnL=${pnl:+9.2f} median_paid={price:.3f}')


def report(rows):
    print('\nALL first-entry markets by LATER sequence type (no future action used in initial ROI):')
    for group in ('no late BUY','same-side additions','opposite-side additions','both-side additions'):
        summary(group, [r for r in rows if r['_sequence']['pattern']==group])
    print('\n20-60c first entries by LATER sequence type:')
    subset = [r for r in rows if .20<=r['price']<=.60]
    summary('20-60c overall', subset)
    for group in ('no late BUY','same-side additions','opposite-side additions','both-side additions'):
        summary(group,[r for r in subset if r['_sequence']['pattern']==group])
    summary('20-60c any later SELL',[r for r in subset if r['_sequence']['sell']])
    summary('20-60c no later SELL',[r for r in subset if not r['_sequence']['sell']])
    print('\n20-60c by first entry price AND follow-up activity:')
    for band in ('20-40c','40-60c'):
        b = [r for r in subset if price_band(r['price']) == band]
        summary(band,b)
        summary('  one-shot / no late BUY',[r for r in b if r['_sequence']['pattern']=='no late BUY'])
        summary('  any opposite-side BUY',[r for r in b if r['_sequence']['opp']>0])
        summary('  same-only late BUY',[r for r in b if r['_sequence']['same']>0 and r['_sequence']['opp']==0])
    print('\nCHRONOLOGICAL 20-60c first-entry comparisons (separate daily blocks):')
    dates = sorted(set(datetime.fromtimestamp(r['ts'], timezone.utc).date().isoformat() for r in subset))
    if dates:
        # Calendar-week comparisons avoid arbitrarily breaking correlated markets in a day.
        weeks = sorted(set(datetime.fromtimestamp(r['ts'],timezone.utc).strftime('%G-W%V') for r in subset))
        for week in weeks:
            sample=[r for r in subset if datetime.fromtimestamp(r['ts'],timezone.utc).strftime('%G-W%V')==week]
            summary(week,sample)
    print('\nSEQUENCE MECHANICS')
    for group in ('same-side additions','opposite-side additions','both-side additions'):
        pool=[r['_sequence'] for r in subset if r['_sequence']['pattern']==group]
        if not pool: continue
        same_lags=[x['first_same_s'] for x in pool if x['first_same_s'] is not None]
        opp_lags=[x['first_opp_s'] for x in pool if x['first_opp_s'] is not None]
        med_same=median(same_lags) if same_lags else 'n/a'
        med_opp=median(opp_lags) if opp_lags else 'n/a'
        med_buys=median(x['buy_count'] for x in pool)
        med_total=median(x['buy_notional'] for x in pool)
        print(f'{group:<27} markets={len(pool):4}, median same ADD lag={med_same}s, '
              f'opp ADD lag={med_opp}s, buy fills={med_buys:g}, gross buy notional=${med_total:.2f}')


def main():
    parser=argparse.ArgumentParser(description='Read-only wallet 954B post-first-entry trade sequence audit')
    parser.add_argument('--trades', type=Path, help='Existing wallet_954b/trades.jsonl file; overrides autodetection')
    args=parser.parse_args()
    path=args.trades if args.trades else next((p for p in DEFAULT_PATHS if p.is_file()),None)
    if path is None or not path.is_file():
        raise SystemExit('Missing local wallet_954b/trades.jsonl. Run with --trades /EXISTING/PATH/trades.jsonl; no download attempted.')
    base=Path('src/copy-trading/w954_binance_signal_audit_readonly.py')
    if not base.is_file():
        raise SystemExit(f'Missing first-fill loader: {base}')
    B=runpy.run_path(str(base))
    first,_=B['load_wallet']()
    grouped,raw_count,invalid,confirmed,collisions=load_trades(path)
    if collisions:
        raise SystemExit(f'ABORT: Multiple conditionIds in same 15m start, unsafe market grouping: {list(collisions.items())[:8]}')
    if confirmed < max(20,raw_count*.3):
        print(f'WARNING: only {confirmed}/{raw_count} records have confirmed 15m start/slug; timestamp floor used for some.')
    print('\n954B — POST-ENTRY TRADE-SEQUENCE AUDIT (READ ONLY)')
    print('='*105)
    print(f'Raw file: {path}')
    print(f'Raw lines: {raw_count}; grouped market starts: {len(grouped)}; confirmed-start rows={confirmed}; invalid={dict(invalid)}')
    print(f'First-entry loader rows: {len(first)}')
    rows=[]; missed=Counter()
    for r in first:
        try:
            start=int(r['start'])
            ts=int(r['ts'])
            price=float(r['price'])
            cost=float(r['cost'])
            pnl=float(r['pnl'])
            won=int(bool(r['won']))
            if not (0<price<1 and cost>0) or outcome(r['side']) is None:
                missed['bad_first_fill']+=1;continue
        except (KeyError,ValueError,TypeError):
            missed['bad_first_fill']+=1;continue
        trade_rows=grouped.get(start)
        if not trade_rows:
            missed['market_start_miss']+=1;continue
        classified=classify(r,trade_rows)
        if classified is None:
            missed['no_near_fills']+=1;continue
        if not classified['close_match']:
            missed['first_fill_price_side_mismatch']+=1;continue
        rows.append({**r,'price':price,'cost':cost,'pnl':pnl,'won':won,'_sequence':classified})
    print(f'Joined first-entry / all-fills markets: {len(rows)}; excluded={dict(missed)}')
    if len(rows) < 100 or len(rows)/max(1,len(first)) < .75:
        raise SystemExit('ABORT: Join is incomplete/unsafe; inspect source key naming and price-side matching before drawing conclusions.')
    report(rows)
    print('\nGUARDRAILS: Later actions happen AFTER the first decision; grouping is not a causal signal.')
    print('Hypothetical FIRST BUY hold-to-expiry ROI, not actual market P&L. No fees/spread/delays applied.')
    print('No orders, network requests, file writes, or GitHub changes.')

if __name__=='__main__':
    main()
