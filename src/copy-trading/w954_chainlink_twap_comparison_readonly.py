#!/usr/bin/env python3
"""Wallet 954b signal-source comparison: Binance spot, a Binance 60s-TWAP
proxy, and optional *genuine* historical Chainlink TWAP60 recordings.

Read-only. Requires the earlier w954_binance_signal_audit_readonly.py and cached
Binance ZIPs. Reads optional genuine TWAP60 .csv.gz (outcometick sample) or
.jsonl.gz (Poly Tick Tape). No writes, downloads, database or trading actions.

Historical collector receive timestamps are required for causal genuine-TWAP
features. Binance candle data is not the Chainlink settlement oracle.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import runpy
from statistics import median

BINANCE_SCRIPT = 'src/copy-trading/w954_binance_signal_audit_readonly.py'


def band(price):
    if .20 <= price < .60: return '20-60c'
    if .60 <= price < .80: return '60-80c'
    if price >= .80: return '80c+'
    return '<20c'


def rolling_mean(px, t, window=60):
    # Require every 1s bar in the lookback: no interpolation or extrapolation.
    seq = [px.get(s) for s in range(t-window+1, t+1)]
    return sum(seq)/window if all(v is not None and v > 0 for v in seq) else None


def read_real_twap(root, dates):
    """Load genuine Chainlink TWAP only, requiring observed receive timestamps.

    Supported:
    outcometick: BTCUSD-twap60s-prices-YYYY-MM-DD.csv.gz
    Poly Tick Tape: btc-twap60-YYYY-MM-DD.jsonl.gz
    """
    data = {}
    counts = Counter()
    paths = sorted(root.rglob('*.csv.gz')) + sorted(root.rglob('*.jsonl.gz'))
    for path in paths:
        p = path.name.lower()
        if not ('twap60' in p or 'twap-60s' in p or 'twap60s' in p):
            continue
        matched_day = next((d for d in dates if d in p), None)
        if not matched_day:
            continue
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                stream = csv.DictReader(f) if p.endswith('.csv.gz') else (json.loads(line) for line in f if line.strip())
                for r in stream:
                    if p.endswith('.csv.gz'):
                        feed_ms = int(float(r['feed_ts_ms']))
                        recv_ms = int(float(r['recv_ms']))
                        price = float(r['value'])
                        ts = feed_ms//1000
                    else:
                        ts = int(r['ts'])
                        recv_ms = int(r['recv_ms'])
                        price = float(r.get('v') or r['p'])
                    if price <= 0 or recv_ms < ts*1000:
                        continue
                    # Later received correction of the same feed-second supersedes.
                    old = data.get(ts)
                    if old is None or recv_ms > old[1]:
                        data[ts] = (price, recv_ms)
                    counts[matched_day] += 1
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
            print(f'WARNING: could not parse {path}: {exc}')
    return data, counts


def chainlink_asof(ticks, target_sec, available_by_ms, max_age=5):
    # Later reports aren't permitted, even if their observation time is old.
    for s in range(int(target_sec), int(target_sec)-max_age-1, -1):
        tick = ticks.get(s)
        if tick is not None and tick[1] <= available_by_ms:
            return tick[0]
    return None


def roi(rows):
    cost = sum(r['cost'] for r in rows)
    return 100*sum(r['pnl'] for r in rows)/cost if cost else float('nan')


def report(rows, tag, field):
    print('\n'+tag)
    print('-'*98)
    print(f"{'First price':<12} {'N':>5} {'Sign agrees':>12} {'First ROI':>11} {'Follow ROI':>11} {'Against ROI':>12} {'No-signal':>11}")
    for name in ('<20c', '20-60c', '60-80c', '80c+'):
        group = [r for r in rows if band(r['price']) == name and field in r]
        strong = [r for r in group if abs(r[field]) >= .5]
        follows = [r for r in strong if (r[field]>0)==(r['side']=='up')]
        against = [r for r in strong if (r[field]>0)!=(r['side']=='up')]
        if not group: continue
        print(f'{name:<12} {len(group):>5} {len(follows)/len(strong):>11.1%} '
              f'{roi(group):>+10.2f}% {roi(follows):>+10.2f}% {roi(against):>+11.2f}% {len(group)-len(strong):>11}' if strong else
              f'{name:<12} {len(group):>5} N/A | no moves >= 0.5bp')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start-date', default='2026-10-06')
    ap.add_argument('--end-date', default='2026-11-05')
    ap.add_argument('--binance-script', default=BINANCE_SCRIPT)
    ap.add_argument('--real-chainlink-dir', type=Path,
                    help='Extracted genuine TWAP60 CSV.GZ or JSONL.GZ archive directory')
    ap.add_argument('--lag-seconds', type=int, default=3,
                    help='Conservative receipt/candle safety lag, min 2s')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        px={i:100+i*.1 for i in range(1000)}
        assert abs(rolling_mean(px,99)-(100+69.5*.1))<1e-8
        assert rolling_mean({1:100},61) is None
        assert chainlink_asof({10:(100.,11500)},10,11000) is None
        assert chainlink_asof({10:(100.,11500)},10,12000)==100
        assert band(.5)=='20-60c'
        print('PASS: strict rolling window, receipt-time anti-lookahead, grouping')
        return
    if args.lag_seconds < 2 or args.end_date < args.start_date:
        ap.error('Invalid lag or date range')
    src = runpy.run_path(str(args.binance_script))
    wallet, excluded = src['load_wallet']()
    selected = defaultdict(list)
    for row in wallet:
        day = datetime.fromtimestamp(row['ts'], timezone.utc).strftime('%Y-%m-%d')
        if args.start_date <= day <= args.end_date:
            selected[day].append(row)
    rows=[]
    stats=Counter()
    for day in sorted(selected):
        prev = (datetime.strptime(day,'%Y-%m-%d').date()-timedelta(days=1)).isoformat()
        px=src['bars'](prev, False)
        px.update(src['bars'](day, False))
        real={}
        if args.real_chainlink_dir:
            real, counts=read_real_twap(args.real_chainlink_dir, (prev,day))
            print(f'{day}: real TWAP records parsed={sum(counts.values()):,}')
        for r in selected[day]:
            t, s = r['ts'],r['start']
            e=t-args.lag_seconds
            spot=px.get(e)
            spot_start=px.get(s-args.lag_seconds)
            twap=rolling_mean(px,e)
            twap_start=rolling_mean(px,s-args.lag_seconds)
            if None in (spot,spot_start,twap,twap_start):
                stats['missing_binance']+=1;continue
            out=dict(r)
            out['binance_spot_open_bps']=(spot/spot_start-1)*10000
            out['binance_twap60_proxy_open_bps']=(twap/twap_start-1)*10000
            out['proxy_gap_bps']=(spot/twap-1)*10000
            if real:
                # Preserve the true oracle observation at market start, if available
                # by the wallet's entry. Never read a future report.
                cutoff=t*1000-args.lag_seconds*1000
                ref=chainlink_asof(real,s,cutoff,0)
                current=chainlink_asof(real,e,cutoff,5)
                if ref is not None and current is not None:
                    out['chainlink_twap60_open_bps']=(current/ref-1)*10000
                    out['binance_vs_chainlink_bps']=(spot/current-1)*10000
                    stats['aligned_real_chainlink']+=1
                else: stats['unavailable_real_chainlink_at_entry']+=1
            rows.append(out)
        print(f'{day}: wallet entries={len(selected[day])}, Binance aligned={sum(1 for x in rows if datetime.fromtimestamp(x["ts"],timezone.utc).strftime("%Y-%m-%d")==day)}',flush=True)
    print('\nWALLET 954B — SIGNAL SOURCE ATTRIBUTION (EXPLORATORY, READ ONLY)')
    print('N=',len(rows), 'Excluded=',dict(excluded), 'data gaps=',dict(stats))
    report(rows,'BINANCE INSTANTANEOUS SPOT: DIRECTION VS OPEN','binance_spot_open_bps')
    report(rows,'BINANCE 60s ROLLING MEAN: ***PROXY, NOT CHAINLINK***','binance_twap60_proxy_open_bps')
    if args.real_chainlink_dir:
        report(rows,'GENUINE CHAINLINK TWAP60 (RECEIPT-TIME FILTERED)','chainlink_twap60_open_bps')
        real_rows=[r for r in rows if 'chainlink_twap60_open_bps' in r]
        switches=[r for r in real_rows if abs(r['chainlink_twap60_open_bps'])>=.5 and abs(r['binance_spot_open_bps'])>=.5 and (r['chainlink_twap60_open_bps']>0)!=(r['binance_spot_open_bps']>0)]
        print(f'Genuine Chainlink aligned={len(real_rows)}, spot/Chainlink opening-direction disagreements={len(switches)}')
        if real_rows:
            print('Median Binance spot minus Chainlink TWAP (basis points)=',round(median(r['binance_vs_chainlink_bps'] for r in real_rows),3))
    print('\nLimitations: conditioning on executed fills, single-second Binance closes, no actual unfilled orders.')
    print('A sign correlation cannot prove the trader observes a particular feed.')
    print('Genuine Chainlink requires collected receive timestamps; this script never fabricates them.')
    print('READ ONLY: no network, writes, orders, DB changes or GitHub modifications.')

if __name__=='__main__':
    main()
