#!/usr/bin/env python3
"""Wallet 954B historical Chainlink relative-value audit.

Run from the repo root:
    python3 -B src/copy-trading/w954_chainlink_fair_value_audit_readonly.py

Read-only. Uses pre-existing local wallet/Chainlink/Binance caches, writes nothing.
First-fill hold-to-resolution economics are NOT whole-wallet realized P&L.
"""

import argparse
import math
import runpy
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

BASE = Path('src/copy-trading')
B = runpy.run_path(str(BASE / 'w954_binance_signal_audit_readonly.py'))
C = runpy.run_path(str(BASE / 'w954_chainlink_twap_comparison_readonly.py'))


def spot_asof(d, ts, max_age=2):
    for delay in range(max_age + 1):
        value = d.get(ts - delay)
        if value is not None:
            try:
                val = float(value)
                if math.isfinite(val) and val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return None


def realized_sigma(d, ts, window_s, step=30):
    """Per-sqrt-second sigma from non-overlapping trailing 30s Binance log returns.

    Anchors end at entry time (already moved 3s before wallet fill), never after.
    Sample std, ignoring incomplete intervals. Report coverage separately.
    """
    returns = []
    expected = window_s // step
    for i in range(expected):
        newest = spot_asof(d, ts - i * step)
        oldest = spot_asof(d, ts - (i + 1) * step)
        if newest is None or oldest is None:
            continue
        returns.append(math.log(newest / oldest))
    if len(returns) < max(12, int(expected * .75)):
        return None, len(returns), expected
    sigma = stdev(returns) / math.sqrt(step)
    if not math.isfinite(sigma) or sigma <= 0:
        return None, len(returns), expected
    return sigma, len(returns), expected


def phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def fmt(v, nd=2, suffix=''):
    return f'{v:.{nd}f}{suffix}' if v is not None and math.isfinite(v) else 'n/a'


def stats(label, sample):
    n = len(sample)
    if not n:
        print(f'{label:<23} N=   0')
        return
    out = mean(x['won'] for x in sample)
    paid = mean(x['paid'] for x in sample)
    fair = mean(x['fair'] for x in sample)
    edge = mean(x['edge'] for x in sample)
    cash = sum(x['cost'] for x in sample)
    pnl = sum(x['pnl'] for x in sample)
    roi = 100 * pnl / cash if cash > 0 else None
    market_brier = mean((x['paid'] - x['won'])**2 for x in sample)
    model_brier = mean((x['fair'] - x['won'])**2 for x in sample)
    print(f'{label:<23} N={n:4} paid={paid:5.3f} fair={fair:5.3f} win={out:5.3f} '
          f'modelEdge={edge*100:+7.2f}pp grossROI={fmt(roi):>7}% '
          f'Brier(paid/model)={market_brier:.3f}/{model_brier:.3f}')


def calibrate(rows):
    print('\n  PRICE BUCKETS (per-entry averages; not executable opportunity controls)')
    for low, high in ((0,.20),(.20,.40),(.40,.60),(.60,.80),(.80,1.00001)):
        sample = [x for x in rows if low <= x['paid'] < high]
        stats(f'{low:.2f}–{min(high,1):.2f} price', sample)
    print('\n  MODEL EDGE BUCKETS (Chainlink fair value minus wallet paid price)')
    for lo, hi, label in [(-2,-.10,'edge < -10pp'),(-.10,-.03,'-10 to -3pp'),
                          (-.03,.03,'-3 to +3pp'),(.03,.10,'+3 to +10pp'),(.10,2,'edge >= +10pp')]:
        stats(label, [x for x in rows if lo <= x['edge'] < hi])
    print('\n  SIDE / TIME REMAINING')
    for side in ('UP', 'DOWN'):
        stats(side, [x for x in rows if x['side']==side])
    for lo, hi in ((0,300),(300,600),(600,900)):
        stats(f'{lo//60}–{hi//60} min left', [x for x in rows if lo <= x['left'] < hi])


def main():
    parser = argparse.ArgumentParser(description='Local historical Chainlink price vs paid probability audit')
    parser.add_argument('--date', default='2026-10-18', help='UTC wallet-entry date, local cache required')
    parser.add_argument('--sigma-window', type=int, default=3600, choices=(900,1800,3600),
                        help='Trailing Binance volatility window in seconds')
    args = parser.parse_args()
    day = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    import datetime as dt
    prev = (day - dt.timedelta(days=1)).strftime('%Y-%m-%d')
    curr = day.strftime('%Y-%m-%d')

    print('\n954B — CHAINLINK OPENING GAP / FAIR-VALUE AUDIT (READ ONLY)')
    print('='*110)
    print('First-fill wallet economics only; does NOT model subsequent fills, hedges, exits, taker fees or spread.')
    print('Chainlink TWAP60 reference at start; as-of Chainlink current; Binance trailing 30s vol, no future values.')
    print('Model is an illustrative zero-drift normal approximation, NOT a verified executable pricing edge.')

    wallet, _ = B['load_wallet']()
    prices = B['bars'](prev, False)
    prices.update(B['bars'](curr, False))
    root = Path.home() / 'wallet_954b_spot_research' / 'chainlink_samples'
    ticks, _ = C['read_real_twap'](root, (prev,curr))
    if not ticks:
        raise RuntimeError('No genuine historical Chainlink TWAP60 cache for requested dates')

    source_count = 0
    missing = defaultdict(int)
    prepared = []
    for r in wallet:
        ts = int(r['ts'])
        if datetime.fromtimestamp(ts,timezone.utc).date() != day.date():
            continue
        source_count += 1
        side = str(r['side']).lower()
        if side not in ('up','down'):
            missing['side']+=1;continue
        start = int(r['start'])
        entry = ts - 3  # reporting safety lag; no signal newer than entry
        age = entry - start
        if not 0 <= age < 900:
            missing['outside_market']+=1;continue
        # cut off the known historical Chainlink reports at the pre-fill entry instant
        cutoff = entry*1000
        current = C['chainlink_asof'](ticks,entry,cutoff,5)
        opening = C['chainlink_asof'](ticks,start,cutoff,0)
        if current is None or opening is None or current <= 0 or opening <= 0:
            missing['chainlink_ref']+=1;continue
        sigma, observed, expected = realized_sigma(prices,entry,args.sigma_window)
        if sigma is None:
            missing['spot_vol_coverage']+=1;continue
        paid = float(r['price']); cost=float(r['cost']); pnl=float(r['pnl']); won=int(bool(r['won']))
        if not (0 < paid < 1 and cost>0):
            missing['bad_entry']+=1;continue
        left=900-age
        gap=math.log(current/opening)
        implied_z = gap/(sigma*math.sqrt(left))
        record = dict(ts=entry, side=side.upper(),paid=paid,cost=cost,pnl=pnl,won=won,
                      current=current,opening=opening,left=left,gap_bp=10000*gap,
                      vol_bp_per_sqrt_s=10000*sigma,observed=observed,expected=expected,
                      z=implied_z)
        prepared.append(record)

    print(f'Date={args.date}; wallet first entries={source_count}; ref+vol eligible={len(prepared)}; excluded={dict(missing)}')
    if not prepared:
        print('No usable entries. Check Chainlink start reference coverage and Binance bar cache. No writes were made.')
        return

    # Check consistency between source P&L and elementary binary hold-to-resolution arithmetic
    discrepant=0
    for r in prepared:
        qty=r['cost']/r['paid']
        theoretical=qty*(r['won']-r['paid'])
        if abs(theoretical-r['pnl']) > max(.10, .01*qty):
            discrepant+=1
    print(f'Binary first-fill P&L reconciliation warnings: {discrepant}/{len(prepared)}')
    print('Do not interpret first-fill settlement ROI as actual wallet strategy realized return.')

    for mult in (.75,1.,1.5):
        print(f'\n{args.sigma_window//60}m trailing-vol scenario x{mult:g}')
        rows=[]
        for r in prepared:
            q=phi(r['z']/mult)
            q=min(max(q,.001),.999)
            fair=q if r['side']=='UP' else 1-q
            rows.append({**r, 'fair':fair,'edge':fair-r['paid']})
        print('Aggregate:')
        stats('ALL first fills',rows)
        stats('20–60c first fills',[r for r in rows if .20<=r['paid']<=.60])
        calibrate(rows)
        print('\n  TOP 8 ABSOLUTE MODEL DISCREPANCIES (diagnostic, not hindsight opportunities)')
        for r in sorted(rows,key=lambda x:abs(x['edge']),reverse=True)[:8]:
            when=datetime.fromtimestamp(r['ts'],timezone.utc).strftime('%H:%M:%S')
            print(f"  {when}Z {r['side']:4} paid={r['paid']:.3f} fair={r['fair']:.3f} "
                  f"edge={100*r['edge']:+6.1f}pp gap={r['gap_bp']:+7.2f}bp "
                  f"vol={r['vol_bp_per_sqrt_s']:.3f}bp/sqrt(s) "
                  f"left={r['left']/60:4.1f}m won={r['won']} grossPnl={r['pnl']:+7.2f}")

    print('\nINTERPRETATION GUARDRAILS')
    print('- This tests whether a simple opening-gap/volatility estimate explains price vs first fill on these entries.')
    print('- Overlapping volatility scenarios are sensitivity analyses, not independent test sets.')
    print('- To identify actual wallet P&L, reconstruct ALL fills per market (including opposite side/sells).')
    print('- To establish exploitable edge, get contemporaneous executable quotes including markets not traded.')
    print('- No network requests, file writes, trades or GitHub changes made by this diagnostic.')


if __name__=='__main__':
    main()
