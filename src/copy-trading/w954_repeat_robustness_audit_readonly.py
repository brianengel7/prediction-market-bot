#!/usr/bin/env python3
"""954B early-repeat signal robustness and possible fill-fragmentation audit.

Place this file in src/copy-trading/ and run from the repository root:
    python3 -B src/copy-trading/w954_repeat_robustness_audit_readonly.py
    python3 -B src/copy-trading/w954_repeat_robustness_audit_readonly.py --self-test

Reads ONLY pre-existing local wallet and first-entry files. No web/API calls,
order submission, GitHub operations, or file writes. Uses first-buy->settlement
one-share hypothetical P&L (not actual wallet realized P&L). Entry prices are
historic *wallet fill* proxies, NOT quotes necessarily executable by a copier.
"""
import argparse
import random
import runpy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

BASE = Path('src/copy-trading')
AUDIT = BASE / 'w954_post_entry_sequence_audit_readonly.py'
LOADER = BASE / 'w954_binance_signal_audit_readonly.py'


def band(p):
    return '20-40c' if .20 <= p < .40 else ('40-60c' if .40 <= p <= .60 else None)


def week(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%G-W%V')


def day(ts):
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


def rows_join(first, grouped, norm):
    joined, missed = [], Counter()
    for r in first:
        try:
            start, ts, p = int(r['start']), int(r['ts']), float(r['price'])
            side = norm(r['side'])
            if not side or not (0 < p < 1):
                raise ValueError('bad side/price')
        except (KeyError, TypeError, ValueError):
            missed['invalid_first'] += 1
            continue
        pool = grouped.get(start)
        if not pool:
            missed['missing_market'] += 1
            continue
        match = any(t['action'] == 'BUY' and t['side'] == side and abs(t['ts'] - ts) <= 3
                    and abs(t['price'] - p) <= .015 for t in pool)
        if not match:
            missed['first_price_side_mismatch'] += 1
            continue
        joined.append({'start': start, 'ts': ts, 'price': p, 'side': side,
                       'won': int(bool(r['won'])), 'band': band(p),
                       'trades': pool, 'date': day(ts), 'week': week(ts)})
    return joined, missed


def first_confirm(r, min_age=4, max_age=30):
    """First qualifying same-side BUY at age [min_age,max_age].

    Any earlier opposing BUY invalidates this signal, including opposing buys
    within the initial 3 seconds. This is stricter than the previous report.
    The signal is defined using only the wallet's activity up to its timestamp.
    """
    for t in r['trades']:
        if t['ts'] < r['ts'] or t['action'] != 'BUY':
            continue
        age = t['ts'] - r['ts']
        if age > max_age:
            break
        if t['side'] != r['side']:
            # Earliest opposite-side trade vetoes a later same-side signal.
            return None
        if t['side'] == r['side'] and min_age <= age <= max_age:
            return {'record': r, 'signal': t, 'age': age,
                    'price_delta': t['price'] - r['price'], 'mode': f'age>={min_age}s'}
    return None


def separate_waves(r, initial_gap=4, wave_gap=5, max_age=60):
    """Enter on the 2nd temporally-separated post-initial same-side buy wave.

    A wave is only a distinct timestamp by at least `wave_gap` seconds. This is
    NOT proof of a separate parent order; on-chain fills may still fragment.
    """
    first_wave = None
    for t in r['trades']:
        if t['ts'] < r['ts'] or t['action'] != 'BUY':
            continue
        age = t['ts'] - r['ts']
        if age > max_age:
            break
        if t['side'] != r['side']:
            return None
        if t['side'] != r['side'] or age < initial_gap:
            continue
        if first_wave is None:
            first_wave = t
        elif t['ts'] - first_wave['ts'] >= wave_gap:
            return {'record': r, 'signal': t, 'age': age,
                    'price_delta': t['price'] - r['price'],
                    'mode': f'2 waves +{wave_gap}s apart'}
    return None


def estimate(signals, penalty=0.0, fixed_fee=0.0):
    if not signals:
        return None
    p = [min(.999, s['signal']['price'] + penalty) for s in signals]
    qty = len(signals)
    pnl = sum(s['record']['won'] - price - fixed_fee for s, price in zip(signals,p))
    cost = sum(price + fixed_fee for price in p)
    return {'n': qty, 'wins': sum(s['record']['won'] for s in signals),
            'median': median(p), 'mean':sum(p)/qty,
            'pnl_per_share': pnl/qty, 'roi':100*pnl/cost if cost else float('nan'),
            'days':len({s['record']['date'] for s in signals})}


def bootstrap_days(signals, penalty, fee, iterations=2000):
    """Resample UTC trading days, not individual trades; descriptive CI only."""
    by_day = defaultdict(list)
    for s in signals:
        by_day[s['record']['date']].append(s)
    days = list(by_day)
    if len(days) < 4:
        return None
    rng = random.Random(954)
    estimates=[]
    for _ in range(iterations):
        sample = [s for d in rng.choices(days, k=len(days)) for s in by_day[d]]
        result=estimate(sample,penalty,fee)
        estimates.append(result['pnl_per_share'])
    estimates.sort()
    return estimates[int(.025*iterations)], estimates[int(.975*iterations)-1]


def report(label, signals, plus=.05, fee=.01, ci=False):
    if not signals:
        print(f'{label:<30} N=  0')
        return
    scenarios = [(0,0),(0,fee),(.02,fee),(plus,fee)]
    stats = [estimate(signals,x,y) for x,y in scenarios]
    a,b,c,d=stats
    text = (f'{label:<30} N={a["n"]:3d} days={a["days"]:2d}'
            f' win={a["wins"]/a["n"]*100:5.1f}% med={a["median"]:.3f}'
            f' lag={median(s["age"] for s in signals):4g}s'
            f' pnl/share={a["pnl_per_share"]:+.3f}'
            f' ROI={a["roi"]:+6.1f}%'
            f' | +2c+fee={c["roi"]:+6.1f}% +{plus*100:.0f}c+fee={d["roi"]:+6.1f}%')
    if ci:
        interval=bootstrap_days(signals, plus, fee)
        text += (' | day-bootstrap +%.0fc+fee pnl/share 95%% [%+.3f,%+.3f]' % (plus*100, *interval)) if interval else ' | few days for day-bootstrap'
    print(text)


def main_report(rows,missed,raw,confirmed,invalid,src):
    print('\n954B — SAME-SIDE REPEAT CONFIRMATION / FRAGMENTATION AUDIT (READ ONLY)')
    print('='*148)
    print(f'Source: {src}; raw={raw}; confirmed_start={confirmed}; invalid={dict(invalid)}; joined={len(rows)}; excluded={dict(missed)}')
    print('Cohort: one historical first BUY per market. All results: one-share hold to resolution, NOT actual wallet P&L.')
    print('Fee=1c per share means an ILLUSTRATIVE flat expense; neither actual platform fees nor a true fill model.')
    for bn in ('20-40c','40-60c'):
        cohort=[r for r in rows if r['band']==bn]
        print(f'\n{bn} first-fill cohort N={len(cohort)}')
        print('Entry requires an observable same-side buy. To test order fragmentation, vary minimum post-first-fill age:')
        for age in (4,7,10,15,20,30):
            signal=[t for r in cohort if (t:=first_confirm(r,age,30))]
            report(f'first repeat age>={age}s, <=30s',signal,ci=age in (4,10,20))
        print('Require an additional independent-timestamp wave after the first qualifying repeat:')
        for gap in (5,10,15):
            signal=[t for r in cohort if (t:=separate_waves(r,4,gap,60))]
            report(f'2 later waves apart >={gap}s',signal,ci=gap==5)
        main=[t for r in cohort if (t:=first_confirm(r,4,30))]
        print('First qualifying repeat (>=4s <=30s), split by FIRST-entry contract price:')
        for lo,hi in ((.20,.30),(.30,.40),(.40,.50),(.50,.601)):
            segment=[s for s in main if lo <= s['record']['price'] < hi]
            if segment:
                report(f'  first price {lo:.2f}-{hi:.2f}',segment)
        print('First qualifying repeat, by UTC calendar week:')
        for w in sorted({r['week'] for r in cohort}):
            segment=[s for s in main if s['record']['week']==w]
            report('  '+w,segment)
        print('First qualifying repeat, by higher/lower price relative to first fill:')
        for name, pred in [('repeat >=0.5c HIGHER',lambda x: x>.005),
                           ('repeat >0.5c LOWER',lambda x:x<-.005),
                           ('repeat +/-0.5c',lambda x:abs(x)<=.005)]:
            segment=[s for s in main if pred(s['price_delta'])]
            report('  '+name,segment)
    print('\nGUARDRAILS')
    print('- No contract quotes at the copier time; wallet repeat price is not a guaranteed executable quote.')
    print('- A 4/7/10/20s repeat can still belong to one parent order, and exchange time may differ from observer receipt time.')
    print('- The date-cluster bootstrap assesses uncertainty given this date sample, not out-of-sample viability.')
    print('- Requiring more BUY waves is a future signal at that later wave; never backfill an entry to the original price.')
    print('- Prices and outcomes are historical; no transaction costs besides explicitly shown illustrative fee cases.')
    print('- No network requests, orders, file writes or GitHub changes were made.')


def self_test():
    def tx(sec,side,price,line,action='BUY'):
        return {'ts':sec,'side':side,'price':price,'line':line,'action':action}
    a={'ts':100,'price':.30,'side':'UP','won':1,'date':'2026-10-01','week':'2026-W40','band':'20-40c',
       'trades':[tx(100,'UP',.30,1),tx(102,'UP',.31,2),tx(105,'UP',.32,3),tx(116,'UP',.34,4)]}
    assert first_confirm(a,4,30)['age']==5
    assert first_confirm(a,10,30)['age']==16
    assert separate_waves(a,4,5)['age']==16
    assert separate_waves(a,4,15) is None
    b={**a,'trades':[tx(100,'UP',.30,1),tx(103,'DOWN',.5,2),tx(116,'UP',.34,4)]}
    assert first_confirm(b,4,30) is None
    assert separate_waves(b,4,5) is None
    assert abs(estimate([first_confirm(a,4,30)],.02,.01)['pnl_per_share']-(1-.32-.02-.01))<1e-10
    print('PASS: distinct-second confirmation, delay, opposite veto, independent-wave and one-share P&L fixtures')


def main():
    ap=argparse.ArgumentParser(description='Read-only wallet 954B repeated-buy robustness')
    ap.add_argument('--trades',type=Path,help='Existing local wallet_954b/trades.jsonl file')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    if not AUDIT.exists() or not LOADER.exists():
        raise SystemExit('Missing existing read-only wallet loaders in src/copy-trading')
    mod=runpy.run_path(str(AUDIT))
    firstmod=runpy.run_path(str(LOADER))
    path=args.trades or next((p for p in mod['DEFAULT_PATHS'] if p.is_file()),None)
    if path is None or not path.is_file():
        raise SystemExit('Missing existing raw wallet trades.jsonl; no network download is attempted')
    first,_=firstmod['load_wallet']()
    grouped,raw,invalid,confirmed,collisions=mod['load_trades'](path)
    if collisions or confirmed<.75*raw:
        raise SystemExit(f'ABORT: ambiguous market grouping: collisions={list(collisions.items())[:5]} confirmation={confirmed}/{raw}')
    rows,missed=rows_join(first,grouped,mod['outcome'])
    if len(rows)<100 or len(rows)<.95*len(first):
        raise SystemExit(f'ABORT: first-fill join incomplete: {len(rows)}/{len(first)}; excluded={dict(missed)}')
    main_report(rows,missed,raw,confirmed,invalid,path)


if __name__=='__main__':
    main()
