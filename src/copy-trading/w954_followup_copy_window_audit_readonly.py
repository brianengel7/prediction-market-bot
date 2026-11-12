#!/usr/bin/env python3
"""Read-only wallet 954B early-repeat-buy audit.

Use from project root:
  python3 -B src/copy-trading/w954_followup_copy_window_audit_readonly.py
  python3 -B src/copy-trading/w954_followup_copy_window_audit_readonly.py --self-test

Requires the previously created local files:
  src/copy-trading/w954_post_entry_sequence_audit_readonly.py
  src/copy-trading/w954_binance_signal_audit_readonly.py
  ~/poly_wallet_comparison_20261105/wallet_954b/trades.jsonl

No network, no writes, no orders. This is a historical timing diagnostic,
not an executable backtest or a proxy for actual wallet realized P&L.
"""
import argparse
import math
import runpy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

BASE = Path('src/copy-trading')
AUDIT = BASE / 'w954_post_entry_sequence_audit_readonly.py'
LOADER = BASE / 'w954_binance_signal_audit_readonly.py'


def week(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%G-W%V')


def band(price):
    if 0.20 <= price < 0.40:
        return '20-40c'
    if 0.40 <= price <= 0.60:
        return '40-60c'
    return None


def fmtpct(x):
    return 'n/a' if x is None else f'{x:+.2f}%'


def summarize(label, rows, key='price', premium=0.0):
    """One share purchased per market, at recorded wallet BUY or a fixed price premium.

    This uses settlement result and one candidate execution price per market,
    but ignores the wallet's other purchases and the true live copy quote.
    """
    if not rows:
        print(f'{label:<35} N=   0')
        return
    prices = [min(.999, float(r[key]) + premium) for r in rows]
    wins = sum(r['won'] for r in rows)
    cost = sum(prices)
    pnl = wins - cost
    roi = 100 * pnl / cost if cost else None
    print(f'{label:<35} N={len(rows):4d} win={wins/len(rows)*100:5.1f}% '
          f'median_price={median(prices):.3f} one_share_ROI={fmtpct(roi):>9} '
          f'unitPnL={pnl:+8.2f}')


def build(first, grouped, helpers):
    results = []
    missed = Counter()
    for r in first:
        try:
            start = int(r['start'])
            ts = int(r['ts'])
            p = float(r['price'])
            won = int(bool(r['won']))
            side = helpers['outcome'](r['side'])
        except (KeyError, TypeError, ValueError):
            missed['invalid_first_fill'] += 1
            continue
        if side is None or not (0.0 < p < 1.0):
            missed['invalid_first_fill'] += 1
            continue
        if start not in grouped:
            missed['market_not_found'] += 1
            continue
        trades = grouped[start]
        if not any(t['action']=='BUY' and t['side']==side and abs(t['ts']-ts)<=3
                   and abs(t['price']-p)<=0.015 for t in trades):
            missed['entry_price_side_mismatch'] += 1
            continue
        later = [t for t in trades if t['action']=='BUY' and t['ts'] > ts+3]
        # The first follow-up decision is identifiable in real time. A market
        # with both eventual sides is NOT classified by future behavior here.
        same = [t for t in later if t['side']==side]
        opposite = [t for t in later if t['side']!=side]
        first_follow = later[0] if later else None
        first_same = same[0] if same else None
        first_opp = opposite[0] if opposite else None
        previous_opposite = bool(first_same and first_opp and
                                 (first_opp['ts'],first_opp['line']) <
                                 (first_same['ts'],first_same['line']))
        results.append(dict(start=start, ts=ts, price=p, won=won, side=side,
                            band=band(p), week=week(ts), first_follow=first_follow,
                            first_same=first_same, first_opp=first_opp,
                            prior_opposite=previous_opposite, buys=same))
    return results, missed


def triggered(rows, threshold):
    """Only acts when an additional same-side BUY has actually been observed."""
    selected = []
    for r in rows:
        t = r['first_same']
        if not t or t['ts']-r['ts'] > threshold:
            continue
        # If the first follow-up BUY was opposing the initial direction, we
        # would already have conflicting inventory information before copying.
        if r['prior_opposite']:
            continue
        selected.append({**r, 'signal_price':t['price'], 'signal_lag':t['ts']-r['ts'],
                         'price_change':t['price']-r['price'], 'signal_ts':t['ts']})
    return selected


def report_population(rows, heading):
    print(f'\n{heading} | first-buy opportunities={len(rows)}')
    summarize('all original first-buy prices', rows)
    print('Observed first follow-up BUY (not first-ever wallet fill):')
    for side in ('same', 'opposite', 'none'):
        subset = [r for r in rows if (('none' if r['first_follow'] is None else
                           'same' if r['first_follow']['side']==r['side'] else 'opposite')==side)]
        summarize(f'  first follow-up: {side}', subset)
    for sec in (10, 30, 60, 120, 300):
        subs = triggered(rows, sec)
        print(f'\n  Earliest same-side repeat observed within {sec}s (excluding prior opposing BUY): '
              f'{len(subs)}/{len(rows)} first-buy markets')
        summarize('  original first-buy price',subs)
        summarize('  wallet repeat-BUY price, 0 delay', subs, key='signal_price')
        if subs:
            changes = [r['price_change'] for r in subs]
            print(f'  Repeat lags median={median(r["signal_lag"] for r in subs):g}s; '
                  f'median price change={median(changes)*100:+.2f}c; '
                  f'repeat cost more in {sum(x>0.005 for x in changes)}/{len(subs)}, '
                  f'less in {sum(x < -0.005 for x in changes)}/{len(subs)}')
            summarize('  repeat price +2c sensitivity', subs, key='signal_price', premium=.02)
            summarize('  repeat price +5c sensitivity', subs, key='signal_price', premium=.05)
    return


def report_details(rows, threshold=30):
    sub = triggered(rows, threshold)
    print(f'\nFIRST EARLY SAME-SIDE REPEAT WITHIN {threshold}s: price movement and week stability')
    for name, check in (
        ('repeat more expensive >0.5c', lambda x:x['price_change'] > .005),
        ('repeat cheaper <-0.5c', lambda x:x['price_change'] < -.005),
        ('repeat price roughly flat', lambda x:abs(x['price_change']) <= .005),
    ):
        group = [r for r in sub if check(r)]
        summarize(name,group,key='signal_price')
    for w in sorted(set(r['week'] for r in rows)):
        one = [r for r in rows if r['week']==w]
        hit = [r for r in sub if r['week']==w]
        print(f'  {w}: {len(hit)}/{len(one)} markets early-repeat signal')
        summarize('    wallet repeat-BUY price',hit,key='signal_price')
    print('\nOBSERVED FOLLOW-ON WALLET FILLS AFTER THE FIRST REPEAT SIGNAL')
    print('Prices here are from future wallet buys, not from the order book. '
          'The subset with later trades is highly selected and is NOT a latency-adjusted backtest.')
    for delay in (3, 10, 30):
        eligible=[]
        for r in sub:
            follow = next((t for t in r['buys'] if t['ts'] >= r['signal_ts']+delay),None)
            if follow:
                eligible.append({**r,'later_price':follow['price']})
        print(f'  next wallet same-side BUY >= {delay}s after trigger: coverage {len(eligible)}/{len(sub)}')
        summarize('    conditional later wallet price', eligible, key='later_price')


def self_test():
    def t(ts, side, price, line):
        return dict(ts=ts, side=side, action='BUY', price=price, line=line)
    first=[dict(start=900,ts=920,price=.50,won=1,side='UP'),
           dict(start=1800,ts=1820,price=.48,won=0,side='DOWN'),
           dict(start=2700,ts=2720,price=.38,won=1,side='UP')]
    grouped={900:[t(920,'UP',.50,1), t(925,'UP',.55,2)],
             1800:[t(1820,'DOWN',.48,3),t(1827,'UP',.51,4),t(1833,'DOWN',.44,5)],
             2700:[t(2720,'UP',.38,6)]}
    # Only need the field normalizer from the shared audit for these fixtures.
    rows,missed=build(first,grouped,{'outcome':lambda x:x})
    assert len(rows)==3 and not missed, (len(rows),missed)
    assert len(triggered(rows,30))==1
    assert triggered(rows,30)[0]['signal_price']==.55
    assert len(triggered([rows[1]],30))==0, 'prior opposite buy must disqualify'
    assert band(.40)=='40-60c' and band(.20)=='20-40c'
    assert all(not triggered([rows[2]],n) for n in (10,30,300))
    print('PASS: synthetic join, early repeat, prior-opposite rejection, non-repeat, price bands.')


def main():
    p=argparse.ArgumentParser(description='Read-only post-entry repeat-BUY copy feasibility diagnostic')
    p.add_argument('--self-test',action='store_true',help='Pure synthetic test, no data files')
    p.add_argument('--trades',type=Path,default=None,help='Existing wallet trades.jsonl path')
    args=p.parse_args()
    if args.self_test:
        self_test(); return
    if not AUDIT.exists() or not LOADER.exists():
        raise SystemExit('Missing existing read-only wallet loaders under src/copy-trading/')
    A=runpy.run_path(str(AUDIT))
    B=runpy.run_path(str(LOADER))
    path=args.trades or next((path for path in A['DEFAULT_PATHS'] if path.is_file()),None)
    if not path or not path.is_file():
        raise SystemExit('No local wallet trades.jsonl; use --trades EXISTING_PATH (no download attempted)')
    first,_=B['load_wallet']()
    grouped,raw,invalid,confirmed,collisions=A['load_trades'](path)
    print('\n954B — EARLY FOLLOW-UP BUY / COPY WINDOW AUDIT (READ ONLY)')
    print('='*108)
    print(f'Source: {path}; raw lines={raw}; confirmed start={confirmed}; invalid={dict(invalid)}')
    if collisions:
        raise SystemExit(f'ABORT: timestamp-grouping collisions: {list(collisions.items())[:5]}')
    if confirmed < .75 * raw:
        raise SystemExit('ABORT: market starts cannot be confirmed in >25% of raw trades')
    rows,missed=build(first,grouped,A)
    print(f'First-fill markets={len(first)}; joined={len(rows)}; excluded={dict(missed)}')
    if len(rows)<100 or len(rows)<.95*len(first):
        raise SystemExit('ABORT: unsafe/incomplete join. Check data before interpreting.')
    for bn in ('20-40c','40-60c'):
        part=[r for r in rows if r['band']==bn]
        report_population(part,bn)
        report_details(part,30)
    print('\nINTERPRETATION')
    print('- Signal eligibility depends ONLY on an observed second wallet BUY; trade outcomes are labels used for evaluation.')
    print('- Exact price at second wallet BUY is an optimistic, ZERO-LATENCY proxy; do NOT treat it as executable.')
    print('- One share per market, no fees, spread, slippage, API delay or copied fill availability.')
    print('- 2c/5c price sensitivities are illustrative worst-price perturbations, not verified Kalshi/Polymarket quotes.')
    print('- Later wallet BUY proxy is conditional on wallet continuing to trade and is NOT an execution backtest.')
    print('- The first three seconds of repeats are excluded as possible order fragmentation.')
    print('- The trade file may be buy-biased; absence of SELL rows does NOT prove no exits.')
    print('- No network, file writes, orders or GitHub changes.')


if __name__=='__main__':
    main()
