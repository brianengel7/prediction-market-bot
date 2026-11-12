#!/usr/bin/env python3
"""Offline, read-only position audit for cached Polymarket Data API wallet trades.

No network, trading credentials, order placement, database or existing-file writes.
Use the raw *_trades.jsonl cache created by poly_wallet_copy_audit_readonly.py.

Important boundaries:
* Polymarket fills have second-level timestamps: intra-second order may be uncertain.
* Buys paired with later buys are NOT simultaneous executable arbitrage.
* FIFO matched buy/sell gross proceeds exclude fees, outside-window inventory,
  settlement payouts, and opening/closing position marks.
* A paired buy-only position only locks a gross payoff after BOTH legs exist;
  all profit figures in this report are conditional on shares held to settlement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
WALLETS = ('wallet_954b', 'asdcaxz', 'ChubbyGooseLearningCenter', 'HighTempTation')


def pct(a: float, b: float) -> float | None:
    return round(100*a/b, 2) if b else None


def quantile(xs, q):
    if not xs: return None
    xs = sorted(xs)
    pos = (len(xs)-1)*q
    lo = int(pos)
    hi = min(lo+1, len(xs)-1)
    return round(xs[lo]*(hi-pos+lo)+xs[hi]*(pos-lo), 3)


def fmt(x, decimals=2):
    return 'N/A' if x is None else f'{x:,.{decimals}f}'


def read_rows(path):
    markets = defaultdict(list)
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                r = json.loads(line)
                ts = int(float(r['timestamp']))
                side = str(r['side']).upper()
                price, shares = float(r['price']), float(r['size'])
                if side not in ('BUY', 'SELL') or not (0 < price < 1) or shares <= 0: continue
                market = str(r.get('condition_id') or '')
                token = str(r.get('token_id') or '')
                if not market or not token: continue
                idx = str(r.get('outcome_index'))
                title = str(r.get('title') or r.get('slug') or '')
                slug = str(r.get('slug') or '')
                markets[market].append((ts, line_no, side, idx, token, price, shares, title, slug))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{path}:{line_no}: invalid row ({exc})') from exc
    return markets


def weather_kind(s):
    s = s.lower()
    if 'highest temp' in s or 'highest-temperature' in s or 'high temperature' in s:
        return 'daily high'
    if 'lowest temp' in s or 'lowest-temperature' in s or 'low temperature' in s:
        return 'daily low'
    if 'rain' in s or 'precipitation' in s: return 'precipitation'
    if 'snow' in s: return 'snow'
    if 'hurricane' in s or 'storm' in s or 'tropical' in s: return 'storm'
    if 'temperature' in s or 'temp' in s or 'degrees' in s: return 'other temperature'
    return 'other weather'


def analyze(markets):
    buy_n, sell_n, buys_usdc, sells_usdc = 0, 0, 0.0, 0.0
    buy_hours, sell_hours, market_kinds = Counter(), Counter(), Counter()
    buy_days, sell_days = Counter(), Counter()
    gross_roundtrip = matched_sell_shares = uncovered_sell_shares = 0.0
    total_sell_shares = 0.0
    hold_times, matching_rows = [], []
    pairs_qty = pairs_cost = pairs_gross = 0.0
    profitable_qty = 0.0
    pairs_within_5s = pairs_within_60s = pairs_within_300s = 0.0
    paired_market_count = paired_positive_market_count = 0
    pair_lags, market_pair_results = [], []
    market_results, market_roundtrip = [], []
    for market_id, unsorted in markets.items():
        # An API record's timestamp is only seconds. Stable file order breaks ties but
        # must not be mistaken for verified millisecond sequencing.
        rows = sorted(unsorted, key=lambda r: (r[0], r[1]))
        title = next((r[7] for r in rows if r[7]), '')
        slug = next((r[8] for r in rows if r[8]), '')
        weather_category = weather_kind(title + ' ' + slug)
        market_kinds[weather_category] += 1
        by_token = defaultdict(deque)  # (buy_time, available_qty, buy_price)
        paired_pending = {'0': deque(), '1': deque()}  # buy-only pair accounting
        has_sell = any(r[2] == 'SELL' for r in rows)
        gross_market, matched_market_shares, unmatched_market_shares = 0.0, 0.0, 0.0
        market_buy_usdc, market_sell_usdc = 0.0, 0.0
        market_pair_qty = market_pair_gross = market_pair_cost = 0.0
        for ts, _ln, side, idx, token, price, qty, _title, _slug in rows:
            hour_et = datetime.fromtimestamp(ts, ET).hour
            day_et = datetime.fromtimestamp(ts, ET).date().isoformat()
            value = qty * price
            if side == 'BUY':
                buy_n += 1; buys_usdc += value; market_buy_usdc += value
                buy_hours[hour_et] += 1; buy_days[day_et] += 1
                by_token[token].append([ts, qty, price])
                if not has_sell and idx in paired_pending:
                    # Greedily pair chronological opposite-outcome purchases.
                    # This is a realized inventory-cost decomposition, not a quote test.
                    remain = qty
                    other = paired_pending['1' if idx == '0' else '0']
                    while remain > 1e-9 and other:
                        other_ts, other_qty, other_price = other[0]
                        take = min(remain, other_qty)
                        cost = take * (price + other_price)
                        profit = take - cost
                        lag = abs(ts - other_ts)
                        pairs_qty += take; pairs_cost += cost; pairs_gross += profit
                        market_pair_qty += take; market_pair_cost += cost; market_pair_gross += profit
                        profitable_qty += take if profit > 0 else 0
                        if lag <= 5: pairs_within_5s += take
                        if lag <= 60: pairs_within_60s += take
                        if lag <= 300: pairs_within_300s += take
                        # Share-weighted percentiles; avoid storing thousands of duplicate shares.
                        pair_lags.append((lag, take))
                        remain -= take
                        other_qty -= take
                        if other_qty <= 1e-9: other.popleft()
                        else: other[0][1] = other_qty
                    if remain > 1e-9: paired_pending[idx].append([ts, remain, price])
            elif side == 'SELL':
                sell_n += 1; sells_usdc += value; market_sell_usdc += value
                sell_hours[hour_et] += 1; sell_days[day_et] += 1
                total_sell_shares += qty
                remain = qty
                queue = by_token[token]
                while remain > 1e-9 and queue:
                    buy_ts, buy_qty, buy_price = queue[0]
                    take = min(remain, buy_qty)
                    realized = take*(price-buy_price)
                    gross_roundtrip += realized
                    gross_market += realized
                    matched_sell_shares += take
                    matched_market_shares += take
                    hold_times.append((max(ts-buy_ts,0), take))
                    remain -= take
                    buy_qty -= take
                    if buy_qty <= 1e-9: queue.popleft()
                    else: queue[0][1] = buy_qty
                if remain > 1e-9:
                    uncovered_sell_shares += remain
                    unmatched_market_shares += remain
        if market_pair_qty > 0:
            paired_market_count += 1
            if market_pair_gross > 0: paired_positive_market_count += 1
            market_pair_results.append({'market': market_id, 'title': title, 'slug': slug,
                'matched_pair_shares': round(market_pair_qty,3),
                'paired_gross_pnl_before_fees': round(market_pair_gross,3),
                'avg_paired_cost_per_share': round(market_pair_cost/market_pair_qty,5)})
        if matched_market_shares > 0:
            market_roundtrip.append({'market': market_id, 'title': title,
                'matched_sell_shares': round(matched_market_shares,3),
                'gross_fifo_sale_pnl_before_fees': round(gross_market,3),
                'unattributed_sell_shares': round(unmatched_market_shares,3)})
        market_results.append({'market': market_id, 'title': title, 'slug': slug,
            'weather_type':weather_category,'fills':len(rows),
            'buy_notional':round(market_buy_usdc,2),'sell_notional':round(market_sell_usdc,2)})

    def weighted_quantile(pairs, q):
        if not pairs: return None
        ss = sorted(pairs)
        total = sum(weight for _,weight in ss)
        threshold = total*q
        cumulative = 0.0
        for val,weight in ss:
            cumulative += weight
            if cumulative >= threshold: return round(val,2)
        return round(ss[-1][0],2)

    return {
        'markets':len(markets), 'buy_fills':buy_n, 'sell_fills':sell_n,
        'buy_notional_usdc':round(buys_usdc,2),'sell_notional_usdc':round(sells_usdc,2),
        'sell_shares_matched_to_window_buys_pct':pct(matched_sell_shares,total_sell_shares),
        'uncovered_sell_shares':round(uncovered_sell_shares,2),
        'matched_sell_shares':round(matched_sell_shares,2),
        'fifo_sale_gross_pnl_before_fees_usdc':round(gross_roundtrip,2),
        'median_fifo_hold_seconds_weighted':weighted_quantile(hold_times,.5),
        'p90_fifo_hold_seconds_weighted':weighted_quantile(hold_times,.9),
        'paired_buy_only_markets':paired_market_count,
        'paired_buy_only_positive_gross_markets':paired_positive_market_count,
        'paired_shares':round(pairs_qty,2),
        'paired_cost_usdc':round(pairs_cost,2),
        'paired_gross_pnl_before_fees_usdc':round(pairs_gross,2),
        'paired_roi_before_fees_pct':pct(pairs_gross,pairs_cost),
        'paired_profitable_qty_pct':pct(profitable_qty,pairs_qty),
        'paired_qty_legs_within_5s_pct':pct(pairs_within_5s,pairs_qty),
        'paired_qty_legs_within_60s_pct':pct(pairs_within_60s,pairs_qty),
        'paired_qty_legs_within_300s_pct':pct(pairs_within_300s,pairs_qty),
        'median_pair_legs_gap_s_weighted':weighted_quantile(pair_lags,.5),
        'p90_pair_legs_gap_s_weighted':weighted_quantile(pair_lags,.9),
        'buy_hours_et':dict(sorted(buy_hours.items())),
        'sell_hours_et':dict(sorted(sell_hours.items())),
        'buy_days_et':dict(sorted(buy_days.items())),
        'market_weather_type_counts':dict(market_kinds),
        'top_markets_by_buy_usdc':sorted(market_results,key=lambda r:-r['buy_notional'])[:12],
        'top_fifo_sale_markets':sorted(market_roundtrip,key=lambda r:-r['gross_fifo_sale_pnl_before_fees'])[:10],
        'bottom_fifo_sale_markets':sorted(market_roundtrip,key=lambda r:r['gross_fifo_sale_pnl_before_fees'])[:10],
        'top_paired_gross_markets':sorted(market_pair_results,key=lambda r:-r['paired_gross_pnl_before_fees'])[:10],
        'bottom_paired_gross_markets':sorted(market_pair_results,key=lambda r:r['paired_gross_pnl_before_fees'])[:10],
        'limitations':[
          'Second-granularity fill timestamps may distort exact order inside one second',
          'Matched FIFO sale results exclude initial inventory, settlements, unobserved transfers, rebates, and fees',
          'Buy-only pair gross profit assumes both quantities were held through resolution; does not imply instantaneous arbitrage',
          'All prices are historical fills, not executable delayed book quotes',
        ],
    }


def show(wallet,r):
    print('\n'+'='*85+'\n'+wallet+' — OFFLINE POSITION DEEP DIVE\n'+'='*85)
    for k in ('markets','buy_fills','sell_fills','buy_notional_usdc','sell_notional_usdc',
              'sell_shares_matched_to_window_buys_pct','uncovered_sell_shares',
              'fifo_sale_gross_pnl_before_fees_usdc','median_fifo_hold_seconds_weighted',
              'p90_fifo_hold_seconds_weighted','paired_buy_only_markets',
              'paired_buy_only_positive_gross_markets','paired_shares',
              'paired_cost_usdc','paired_gross_pnl_before_fees_usdc',
              'paired_roi_before_fees_pct','paired_profitable_qty_pct',
              'paired_qty_legs_within_5s_pct','paired_qty_legs_within_60s_pct',
              'paired_qty_legs_within_300s_pct','median_pair_legs_gap_s_weighted'):
        print(f'{k:44}: {r[k]}')
    print('\nBuy fills by ET hour:',r['buy_hours_et'])
    print('Sell fills by ET hour:',r['sell_hours_et'])
    print('Markets by weather category (heuristic):',r['market_weather_type_counts'])
    print('\nTop markets by buy USDC:')
    for m in r['top_markets_by_buy_usdc'][:8]:
        print(f"  ${m['buy_notional']:>11,.2f} bought | ${m['sell_notional']:>11,.2f} sold | {m['title'][:82]} | {m['slug'][:80]}")
    print('\nTop matched FIFO sale markets (gross, excludes fees/opening inventory):')
    for m in r['top_fifo_sale_markets'][:5]:
        print(f"  ${m['gross_fifo_sale_pnl_before_fees']:>11,.2f} | {m['title'][:95]}")
    print('\nTop buy-only matched-pair markets (gross, assumes hold to settlement):')
    for m in r['top_paired_gross_markets'][:5]:
        print(f"  ${m['paired_gross_pnl_before_fees']:>11,.2f} | shares={m['matched_pair_shares']:,.1f} | {m['title'][:70]}")


def self_test():
    # Positive paired inventory with 10 shares on each outcome; two sales later match their own buys.
    def row(ts, seq, side, outcome, price, qty, token=None):
        return (ts,seq,side,str(outcome),token or str(outcome),price,qty,'Highest temperature in NYC','highest-temperature-in-nyc')
    a=analyze({'a':[row(100,1,'BUY',0,.4,10),row(130,2,'BUY',1,.5,10)]})
    assert a['paired_shares']==10 and a['paired_gross_pnl_before_fees_usdc']==1.0
    assert a['paired_qty_legs_within_60s_pct']==100.0 and a['paired_qty_legs_within_5s_pct']==0
    b=analyze({'b':[row(100,1,'SELL',0,.7,3),row(110,2,'BUY',0,.4,10),row(120,3,'SELL',0,.6,5)]})
    assert b['uncovered_sell_shares']==3 and b['matched_sell_shares']==5
    assert b['fifo_sale_gross_pnl_before_fees_usdc']==1 and b['sell_shares_matched_to_window_buys_pct']==62.5
    print('PASS: offline synthetic test of buy-only pairing, timestamps and uncovered opening inventory')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir',type=Path,default=Path('data/research/poly_wallet_copy_audit_20261111'))
    p.add_argument('--wallets',nargs='+',choices=WALLETS,default=['HighTempTation','asdcaxz','wallet_954b'])
    p.add_argument('--output-json',type=Path,help='Optional NEW output report file; no other writes')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test(); return
    results={}
    for wallet in args.wallets:
        path=args.input_dir/(wallet+'_trades.jsonl')
        meta_path=args.input_dir/(wallet+'_trades_meta.json')
        if not path.exists() or not meta_path.exists():
            p.error(f'Missing cached trade file and meta for {wallet}: {path}')
        meta=json.loads(meta_path.read_text())
        if meta.get('complete') is not True:
            p.error(f'{wallet} source cache was not completed; refusing to score')
        markets=read_rows(path)
        stats=analyze(markets)
        show(wallet,stats)
        results[wallet]=stats
    if args.output_json:
        if args.output_json.exists():
            p.error('Refusing to overwrite an existing output report file')
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps({'method':'offline-only, inventory/fee caveats in each report',
                                               'wallets':results},indent=2))
        print('\nNEW report written:',args.output_json)
    print('\nREAD ONLY — existing cached trades never modified; no external requests or orders.')

if __name__=='__main__':main()
