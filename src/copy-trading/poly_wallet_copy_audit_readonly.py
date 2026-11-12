#!/usr/bin/env python3
"""Read-only Polymarket wallet copy-feasibility audit (Data API v2).

Public GETs only; no keys, orders, imports from trading modules or mutations.
Run with --offline after the initial download. Historical replay is deliberately
UNAVAILABLE unless real timestamped executable quote snapshots are supplied.

Snapshot CSV schema:
  timestamp,token_id,ask,ask_size
where timestamp is epoch seconds (fractional accepted), ask is price in USDC
per share, and ask_size is available shares at that *one ask level*.
At latency L, the first snapshot timestamp in [signal_time+L,
 signal_time+L+max_quote_lag] is eligible. These are observational proxies,
not proof of order execution; snapshot capture/notification latency is separate.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = 'https://data-api.polymarket.com'
WALLETS = {
    'wallet_954b': '0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0',
    'asdcaxz': '0x5a06acbaee263540a16a0109d584f6672101ffa2',
    'ChubbyGooseLearningCenter': '0x04b0915f2edb018644b28dc0d3e21ca2089a000d',
    'HighTempTation': '0x6011655c4afb76f36dd1b08a137a1ba73466b31e',
}
BTC_WINDOW = re.compile(r'(?:^|[-_])(btc|bitcoin|eth|ethereum|sol|solana|xrp)[-_](?:updown|up-or-down)[-_](5m|15m)[-_](\d{10})(?:$|[-_])', re.I)


def epoch_date(s: str) -> int:
    return int(datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())


def get_json(path: str, params: dict, tries: int = 6) -> dict:
    url = API + path + '?' + urlencode(params)
    for attempt in range(tries):
        try:
            req = Request(url, headers={'Accept': 'application/json', 'User-Agent': 'readonly-wallet-audit/1.0'})
            with urlopen(req, timeout=40) as response:
                data = json.load(response)
            if not isinstance(data, dict) or 'data' not in data:
                raise RuntimeError(f'Unexpected API envelope for {path}')
            return data
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            delay = exc.headers.get('Retry-After')
            try:
                sleep = min(60, max(0.5, float(delay))) if delay else min(45, 2 ** attempt)
            except ValueError:
                sleep = min(45, 2 ** attempt)
            time.sleep(sleep)
        except (URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(min(45, 2 ** attempt))
    raise AssertionError('unreachable')


def fetch_trades(wallet: str, start: int, end_exclusive: int, outfile: Path, max_pages: int) -> dict:
    """Cursor must preserve identical filters, and we keep even same-second fills."""
    count, pages, cursor, seen = 0, 0, None, set()
    params = {'user': wallet, 'limit': 1000, 'taker_only': 'false',
              'start': start, 'end': end_exclusive - 1}
    tmp = outfile.with_suffix(outfile.suffix + '.partial')
    with tmp.open('w', encoding='utf-8') as f:
        while pages < max_pages:
            request = dict(params)
            if cursor:
                request['cursor'] = cursor
            payload = get_json('/v2/trades', request)
            rows = payload.get('data')
            page = payload.get('pagination')
            if not isinstance(rows, list) or not isinstance(page, dict):
                raise RuntimeError('API returned no valid trades/pagination; not interpreting as zero fills')
            for row in rows:
                if not isinstance(row, dict):
                    raise RuntimeError('Non-object trade row')
                ts = int(row['timestamp'])
                if ts < start or ts >= end_exclusive:
                    raise RuntimeError(f'API violated requested date range: {ts}')
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
                count += 1
            pages += 1
            nxt = page.get('next_cursor')
            has_more = page.get('has_more')
            if nxt is None:
                if has_more is True:
                    raise RuntimeError('has_more true but next_cursor null')
                tmp.replace(outfile)
                return {'rows': count, 'pages': pages, 'complete': True}
            if not isinstance(nxt, str) or not nxt or nxt == cursor or nxt in seen:
                raise RuntimeError('Invalid or cycling pagination cursor')
            seen.add(nxt)
            cursor = nxt
            if pages % 20 == 0:
                print(f'  {wallet[:12]}: {count:,} fills through {pages} pages', flush=True)
            time.sleep(0.05)
    # A partial window MUST NOT masquerade as a complete sample.
    partial_path = outfile.with_suffix(outfile.suffix + '.truncated')
    tmp.replace(partial_path)
    return {'rows': count, 'pages': pages, 'complete': False, 'partial_path': str(partial_path)}


def read_trades(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def classify(r: dict) -> str:
    s = ' '.join(str(r.get(k) or '') for k in ('slug', 'event_slug', 'title')).lower()
    if re.search(r'temperature|highest temp|degrees|weather|rainfall|snowfall|precipitation', s):
        return 'weather'
    if re.search(r'bitcoin|\bbtc\b|ethereum|\beth\b|\bsol\b|solana|\bxrp\b|\bdoge\b|crypto', s):
        return 'crypto'
    if re.search(r'\bnba\b|\bnfl\b|\bnhl\b|\bmlb\b|\batp\b|\bwta\b|tennis|soccer|\bvs\.?\b|\bfc\b|basketball|football', s):
        return 'sports'
    return 'other'


def median(vals):
    return round(statistics.median(vals), 4) if vals else None


def pct(n, d):
    return round(100 * n / d, 2) if d else None


def groups_and_signals(rows: list[dict], window: int = 2) -> list[dict]:
    """Approximate copy signals by clustering adjacent BUY fills same token within seconds."""
    buys = sorted((r for r in rows if str(r.get('side', '')).upper() == 'BUY'),
                  key=lambda r: (str(r.get('condition_id', '')), str(r.get('token_id', '')), int(r['timestamp'])))
    output = []
    last_by_token = {}
    for r in buys:
        key = (r.get('condition_id'), r.get('token_id'))
        ts = int(r['timestamp'])
        price = float(r.get('price') or 0)
        shares = float(r.get('size') or 0)
        if not (0 < price < 1 and shares > 0):
            continue
        last = last_by_token.get(key)
        if last is not None and ts - last['last_ts'] <= window:
            last['last_ts'] = ts
            last['shares'] += shares
            last['usdc'] += shares * price
            last['fills'] += 1
        else:
            item = {'timestamp': ts, 'last_ts': ts, 'condition_id': key[0],
                    'token_id': key[1], 'side': 'BUY', 'slug': r.get('slug') or '',
                    'category': classify(r), 'shares': shares, 'usdc': shares * price,
                    'fills': 1, 'outcome_index': r.get('outcome_index')}
            output.append(item)
            last_by_token[key] = item
    for x in output:
        x['avg_price'] = round(x['usdc'] / x['shares'], 6)
        x['usdc'] = round(x['usdc'], 4)
        x['shares'] = round(x['shares'], 5)
        x['burst_seconds'] = x['last_ts'] - x['timestamp']
    return sorted(output, key=lambda x: x['timestamp'])


def pnl_delta(points: list[dict], start: int, end: int) -> dict:
    if not points:
        return {'status': 'no_points'}
    ps = sorted((p for p in points if isinstance(p, dict) and isinstance(p.get('timestamp'), (float, int))), key=lambda p: p['timestamp'])
    before = [p for p in ps if p['timestamp'] <= start]
    after = [p for p in ps if start < p['timestamp'] < end]
    if not before or not after:
        return {'status': 'insufficient_bracketing', 'first': ps[0]['timestamp'] if ps else None,
                'last': ps[-1]['timestamp'] if ps else None}
    p0, p1 = before[-1], after[-1]
    if end - p1['timestamp'] > 2 * 86400:
        return {'status': 'stale_endpoint', 'last': p1['timestamp']}
    fields = ['economic_pnl', 'position_pnl', 'realized_pnl', 'settled_pnl',
              'maker_rebate', 'taker_rebate', 'fees_paid', 'reward_income',
              'yield_income', 'referral_income', 'realized_market_pnl']
    result = {'status': 'ok', 'from_timestamp': p0['timestamp'], 'to_timestamp': p1['timestamp']}
    for k in fields:
        a, b = p0.get(k), p1.get(k)
        result[k] = round(float(b) - float(a), 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    return result


def audit(rows: list[dict], start: int, end: int) -> tuple[dict, list[dict]]:
    if not rows:
        return {'fills': 0, 'markets': 0, 'copy_signals': 0}, []
    categories = Counter(classify(r) for r in rows)
    sides = Counter(str(r.get('side', '')).upper() for r in rows)
    markets = {str(r.get('condition_id', '')) for r in rows if r.get('condition_id')}
    buys = [r for r in rows if str(r.get('side', '')).upper() == 'BUY']
    sell_market = {str(r.get('condition_id', '')) for r in rows if str(r.get('side', '')).upper() == 'SELL'}
    buy_market_outcomes = defaultdict(lambda: defaultdict(list))
    for r in buys:
        try:
            size, price = float(r['size']), float(r['price'])
            if size <= 0 or not (0 < price < 1):
                continue
        except (ValueError, TypeError, KeyError):
            continue
        k = (str(r.get('condition_id')), str(r.get('outcome_index')))
        buy_market_outcomes[k[0]][k[1]].append((size, price))
    both_market = set()
    pair_costs = []
    for market, outcomes in buy_market_outcomes.items():
        if '0' in outcomes and '1' in outcomes:
            both_market.add(market)
            if market not in sell_market:
                prices = []
                for index in ('0', '1'):
                    lot = outcomes[index]
                    prices.append(sum(q * p for q, p in lot) / sum(q for q, _ in lot))
                pair_costs.append(sum(prices))
    signals = groups_and_signals(rows)
    sig_sizes = [r['usdc'] for r in signals]
    # Recurring crypto slugs with an explicit UTC window start (not market close).
    windowed = []
    open_60 = 0
    for s in signals:
        m = BTC_WINDOW.search(s.get('slug') or '')
        if not m:
            continue
        t0 = int(m.group(3))
        if t0 <= s['timestamp'] < t0 + (300 if m.group(2).lower() == '5m' else 900):
            windowed.append(s)
            if s['timestamp'] - t0 < 60:
                open_60 += 1
    busy_second = Counter(s['timestamp'] for s in signals)
    result = {'fills': len(rows), 'markets': len(markets), 'buy_fills': sides.get('BUY', 0),
              'sell_fills': sides.get('SELL', 0), 'category_fills': dict(categories),
              'copy_signals': len(signals), 'median_buy_signal_usdc': median(sig_sizes),
              'copy_signals_under_5_usdc_pct': pct(sum(s < 5 for s in sig_sizes), len(sig_sizes)),
              'copy_signals_under_20_usdc_pct': pct(sum(s < 20 for s in sig_sizes), len(sig_sizes)),
              'peak_copy_signals_same_second': max(busy_second.values(), default=0),
              'both_sides_bought_markets': len(both_market),
              'both_sides_bought_pct': pct(len(both_market), len(markets)),
              'two_sided_no_sell_wavg_cost_lt_1_markets': sum(c < 1 for c in pair_costs),
              'two_sided_no_sell_cost_samples': len(pair_costs),
              'windowed_crypto_signals': len(windowed),
              'windowed_crypto_first_60s_pct': pct(open_60, len(windowed)),
              'historical_executable_latency_verdict': 'NOT TESTABLE WITHOUT QUOTE SNAPSHOTS'}
    return result, signals


def load_quotes(path: Path) -> dict:
    quotes = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required = {'timestamp', 'token_id', 'ask', 'ask_size'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('quotes CSV needs timestamp,token_id,ask,ask_size')
        for r in reader:
            t, ask, size = float(r['timestamp']), float(r['ask']), float(r['ask_size'])
            if math.isfinite(t) and 0 < ask < 1 and size >= 0:
                quotes[r['token_id']].append((t, ask, size))
    return {k: sorted(v) for k, v in quotes.items()}


def latency_replay(signals: list[dict], quotes: dict, delays: list[int],
                   copy_usdc: float, max_lag: float) -> list[dict]:
    by_token_times = {token: [t for t, _, _ in records] for token, records in quotes.items()}
    results = []
    for delay in delays:
        eligible, quote_seen, enough_size, slip = 0, 0, 0, []
        for s in signals:
            if s['avg_price'] <= 0 or s['usdc'] <= 0:
                continue
            eligible += 1
            token = str(s['token_id'])
            arr = quotes.get(token, [])
            if not arr:
                continue
            target = s['timestamp'] + delay
            i = bisect.bisect_left(by_token_times[token], target)
            if i >= len(arr) or arr[i][0] > target + max_lag:
                continue
            _, ask, ask_size = arr[i]
            quote_seen += 1
            shares_wanted = copy_usdc / ask
            if ask_size + 1e-9 < shares_wanted:
                continue
            enough_size += 1
            slip.append(100 * (ask - s['avg_price']))  # cents/share vs wallet's avg fill
        results.append({'latency_s': delay, 'signal_count': eligible,
                        'fresh_ask_coverage_pct': pct(quote_seen, eligible),
                        'top_ask_depth_coverage_pct': pct(enough_size, eligible),
                        'filled_proxy_count': enough_size,
                        'median_ask_worsening_cents': median(slip),
                        'status': 'SNAPSHOT_PROXY_NOT_EXECUTION'})
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)



def self_test() -> None:
    base = 1790000000
    market = '0x' + 'a' * 64
    def row(ts, side, idx, price, size):
        return {'timestamp': ts, 'side': side, 'outcome_index': idx,
                'price': price, 'size': size, 'condition_id': market,
                'token_id': str(idx), 'slug': 'btc-updown-15m-1790000000',
                'event_slug': 'btc-updown-15m', 'title': 'Bitcoin Up or Down'}
    trades = [row(base+5, 'BUY', 0, .45, 10), row(base+5, 'BUY', 0, .45, 20),
              row(base+20, 'BUY', 1, .50, 10), row(base+65, 'SELL', 0, .55, 5)]
    result, signals = audit(trades, base, base+900)
    assert result['fills'] == 4 and result['markets'] == 1
    assert result['both_sides_bought_markets'] == 1
    assert result['two_sided_no_sell_wavg_cost_lt_1_markets'] == 0
    assert result['copy_signals'] == 2
    assert result['windowed_crypto_first_60s_pct'] == 100.0
    assert classify(trades[0]) == 'crypto'
    quotes = {'0': [(base+6, .48, 100)], '1': [(base+21, .55, 100)]}
    replay = latency_replay(signals, quotes, [1, 5, 15], 10, 2)
    assert replay[0]['filled_proxy_count'] == 2
    assert replay[1]['filled_proxy_count'] == 0
    points = [{'timestamp': base-10, 'economic_pnl': 100, 'maker_rebate': 10},
              {'timestamp': base+100, 'economic_pnl': 110, 'maker_rebate': 12}]
    assert pnl_delta(points, base, base+200)['economic_pnl'] == 10
    print('PASS: synthetic stats, no-sell filter, time windows, quote freshness, PnL deltas')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true', help='Run offline synthetic checks; no network')
    p.add_argument('--start', default='2026-10-12', help='Inclusive UTC date')
    p.add_argument('--end', default='2026-11-11', help='Exclusive UTC date')
    p.add_argument('--wallets', nargs='+', choices=sorted(WALLETS), default=list(WALLETS))
    p.add_argument('--output-dir', default='data/research/poly_wallet_copy_audit_20261111')
    p.add_argument('--max-pages', type=int, default=1000)
    p.add_argument('--offline', action='store_true', help='Read cached trades and PnL only; never call API')
    p.add_argument('--quotes-csv', type=Path, help='Historical ask snapshots; NOT current quotes or trades')
    p.add_argument('--max-quote-lag', type=float, default=2)
    p.add_argument('--copy-usdc', type=float, default=10)
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_pages < 1 or args.copy_usdc <= 0 or args.max_quote_lag < 0:
        p.error('max-pages must be >=1, copy-usdc >0, max-quote-lag >=0')
    start, end = epoch_date(args.start), epoch_date(args.end)
    if start >= end:
        p.error('end must be after start')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_summary, all_signals, all_replay = [], [], []
    quotes = load_quotes(args.quotes_csv) if args.quotes_csv else None
    for name in args.wallets:
        wallet = WALLETS[name]
        path = outdir / f'{name}_trades.jsonl'
        meta_path = outdir / f'{name}_trades_meta.json'
        pnl_path = outdir / f'{name}_pnl.json'
        if not args.offline:
            print(f'Downloading {name}: {wallet} (public GET only)', flush=True)
            meta = fetch_trades(wallet, start, end, path, args.max_pages)
            meta.update({'wallet': wallet, 'start': start, 'end_exclusive': end, 'taker_only': False})
            meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
            pnl = get_json('/v2/user-pnl', {'user': wallet, 'interval': 'max', 'fidelity': '1d'})
            pnl_path.write_text(json.dumps(pnl, indent=2), encoding='utf-8')
        else:
            if not path.exists() or not meta_path.exists():
                raise FileNotFoundError(f'Missing complete cache for {name}: {path}')
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if not meta.get('complete') or meta.get('start') != start or meta.get('end_exclusive') != end:
            print(f'WARNING: {name} sample incomplete or outside selected window; NOT scoring it', file=sys.stderr)
            continue
        rows = read_trades(path)
        result, signals = audit(rows, start, end)
        if len(rows) != meta['rows']:
            raise RuntimeError(f'{name}: cached count differs from metadata')
        result.update({'wallet': name, 'address': wallet, 'window_start': args.start,
                       'window_end_exclusive': args.end, 'sample_complete': True})
        if pnl_path.exists():
            d = json.loads(pnl_path.read_text(encoding='utf-8')).get('data', {})
            result['pnl_window'] = pnl_delta(d.get('points', []) if isinstance(d, dict) else [], start, end)
        else:
            result['pnl_window'] = {'status': 'unavailable'}
        all_summary.append(result)
        all_signals.extend({'wallet': name, **s} for s in signals)
        if quotes is not None:
            all_replay.extend({'wallet': name, **x} for x in latency_replay(signals, quotes,
                              [1, 5, 15], args.copy_usdc, args.max_quote_lag))
        print(f"{name}: {len(rows):,} fills, {result['markets']:,} markets, "
              f"{result['copy_signals']:,} buy clusters; both-sides-bought "
              f"{result['both_sides_bought_markets']:,}; PnL {result['pnl_window']['status']}", flush=True)
    report = {'methodology': {
        'window_utc': f'[{args.start}, {args.end})', 'api': '/v2/trades, /v2/user-pnl',
        'trade_rows': 'wallet perspective, taker_only=false, cursor completed',
        'signal': 'BUY fills same condition+token clustered by <=2s gap',
        'paired': 'both outcome indices BUY at any point during window, NOT guaranteed arbitrage',
        'pnl': 'delta of published cumulative PnL points; may lag window endpoints',
        'execution': 'no executable historical evidence without ask snapshots; snapshots are upper-bound proxies',
        'kalshi': 'no presumed market equivalence or settlement compatibility'},
        'wallets': all_summary, 'quote_replay': all_replay}
    (outdir / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_csv(outdir / 'signals.csv', all_signals)
    write_csv(outdir / 'replay.csv', all_replay)
    print('READ ONLY. Report: ' + str(outdir / 'summary.json'))
    if quotes is None:
        print('No historical quotes supplied: 1/5/15s executable copy P&L NOT ESTIMATED.')


if __name__ == '__main__':
    main()
