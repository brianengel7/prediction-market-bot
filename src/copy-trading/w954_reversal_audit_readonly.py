#!/usr/bin/env python3
"""Wallet 954B reversal-signature audit. Local files only; stdout only; no orders/writes."""
import runpy
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from statistics import median

B = runpy.run_path('src/copy-trading/w954_binance_signal_audit_readonly.py')
C = runpy.run_path('src/copy-trading/w954_chainlink_twap_comparison_readonly.py')
wallet, _ = B['load_wallet']()
spot = B['bars']('2026-10-17', False)
spot.update(B['bars']('2026-10-18', False))
root = Path.home() / 'wallet_954b_spot_research' / 'chainlink_samples'
ticks, _ = C['read_real_twap'](root, ('2026-10-17', '2026-10-18'))
assert ticks, 'No genuine historical Chainlink observations found'
WINDOWS = (5, 15, 30, 60, 120, 300)


def side_sign(x):
    s = str(x).strip().lower()
    return 1 if s == 'up' else (-1 if s == 'down' else 0)


def bp(x, y):
    if x is None or y is None or x <= 0 or y <= 0:
        return None
    return 10000 * (x / y - 1)


def classify(bps, side, deadband=0.0):
    if bps is None or abs(bps) <= deadband or not side:
        return None
    return 'with' if (bps > 0) == (side > 0) else 'against'


def report(label, subset):
    cost = sum(r['cost'] for r in subset)
    pnl = sum(r['pnl'] for r in subset)
    n = len(subset)
    win = sum(bool(r['won']) for r in subset)
    roi = (100 * pnl / cost) if cost > 0 else float('nan')
    medprice = median([r['price'] for r in subset]) if subset else float('nan')
    medleft = median([r['remaining'] / 60 for r in subset]) if subset else float('nan')
    print(f'{label:<24} N={n:3}  win={100*win/n if n else float("nan"):6.1f}%  '
          f'grossROI={roi:+8.2f}%  pnl=${pnl:+9.2f}  '
          f'median_entry={medprice:.3f}  median_left={medleft:.1f}m')


rows = []
day_count = 0
for r in wallet:
    try:
        ts = int(r['ts'])
        if datetime.fromtimestamp(ts, timezone.utc).date().isoformat() != '2026-10-18':
            continue
        day_count += 1
        start = int(r['start'])
        entry = ts - 3  # Known historical reporting safety lag; no post-fill price data
        side = side_sign(r['side'])
        if not side or not 0 <= entry - start < 900:
            continue
        cutoff_ms = entry * 1000
        now = C['chainlink_asof'](ticks, entry, cutoff_ms, 5)
        if now is None:
            continue
        record = {
            'side': side,
            'ts': entry,
            'price': float(r['price']),
            'cost': float(r['cost']),
            'pnl': float(r['pnl']),
            'won': bool(r['won']),
            'remaining': 900 - (entry - start),
            'cl': {}, 'spot': {},
        }
        for lookback in WINDOWS:
            prior = C['chainlink_asof'](ticks, entry - lookback, cutoff_ms, 5)
            record['cl'][lookback] = bp(now, prior)
            record['spot'][lookback] = bp(spot.get(entry), spot.get(entry - lookback))
        open_cl = C['chainlink_asof'](ticks, start, cutoff_ms, 0)
        record['cl']['open'] = bp(now, open_cl)
        record['spot']['open'] = bp(spot.get(entry), spot.get(start))
        rows.append(record)
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise RuntimeError(f'Invalid wallet row at timestamp={r.get("ts")}: {exc}') from exc

print('\n954B — SAME-ENTRY CHAINLINK REVERSAL SIGNATURE (READ ONLY)')
print('=' * 103)
print('Source: genuine historical Chainlink TWAP60, locally cached; Binance 1s reference')
print(f'Oct 18 wallet rows={day_count}, Chainlink usable={len(rows)}')
print('Entry timestamp = first fill minus 3s; lookbacks evaluated before entry.')
print('Gross historical first-fill P&L; no fees/slippage; no causal inference.\n')

for subset_name, subset in [('ALL', rows), ('20-60c', [r for r in rows if .20 <= r['price'] <= .60])]:
    print(f'\n{subset_name}: {len(subset)} Chainlink-aligned entries')
    report('Overall', subset)
    print('PER-LOOKBACK eligible samples (the denominators differ):')
    for w in ('open', 300, 120, 60, 30, 15, 5):
        valid = [r for r in subset if classify(r['cl'][w], r['side']) is not None]
        with_n = sum(classify(r['cl'][w], r['side']) == 'with' for r in valid)
        print(f'  {str(w):>4}: {with_n:>3}/{len(valid):<3} align = {100*with_n/len(valid) if valid else float("nan"):5.1f}%')
    for long, short in [('open', 30), (300, 30), (120, 15)]:
        for deadband in (0.0, 1.0):
            eligible = [r for r in subset
                        if (long != 'open' or 900 - r['remaining'] >= short)
                        and classify(r['cl'][long], r['side'], deadband)
                        and classify(r['cl'][short], r['side'], deadband)]
            print(f'\n  JOINT SAME-ENTRY {long} versus {short}s, deadband={deadband:g}bp (N={len(eligible)})')
            groups = defaultdict(list)
            for r in eligible:
                a = classify(r['cl'][long], r['side'], deadband)
                z = classify(r['cl'][short], r['side'], deadband)
                groups[(a, z)].append(r)
            for pair, label in [
                (('against', 'with'), 'REVERSAL signature'),
                (('against', 'against'), 'Countertrend still'),
                (('with', 'with'), 'Momentum both'),
                (('with', 'against'), 'Recent pullback')]:
                report(label, groups[pair])
            common = [r for r in eligible if r['spot'][long] is not None and r['spot'][short] is not None]
            cl_rev = sum(classify(r['cl'][long],r['side'],deadband) == 'against' and classify(r['cl'][short],r['side'],deadband) == 'with' for r in common)
            spot_rev = sum(classify(r['spot'][long],r['side'],deadband) == 'against' and classify(r['spot'][short],r['side'],deadband) == 'with' for r in common)
            print(f'   Same-entry feed check: N={len(common)}, CL reversals={cl_rev}, Binance reversals={spot_rev}')

print('\nNo fully matched Polymarket opportunity control is available from wallet fills alone:')
print('would need timestamped executable quotes for both UP and DOWN, including untraded opportunities.')
print('Overlapping lookback pairs are not independent; all outcomes are in-sample and gross of fees.')
print('No network, file writes, trades, or GitHub changes were made by this diagnostic.')
