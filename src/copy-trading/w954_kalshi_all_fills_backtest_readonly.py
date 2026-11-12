#!/usr/bin/env python3
"""Wallet 954B -> Kalshi: replay EVERY recorded fill (UP/DOWN BUY and SELL).

Place in src/copy-trading/ alongside:
  w954_post_entry_sequence_audit_readonly.py
  w954_kalshi_crossvenue_backtest_readonly.py
  w954_kalshi_historical_bbo_replay_readonly.py (only for --price-source book)

READ ONLY: no POST, no orders, no file writes. Public Kalshi historical GETs in
--price-source prints unless --markets-file and --trades-file are provided;
--price-source book reads downloaded historical BBO / snapshots (still GETs
market results unless --markets-file is supplied).

This is an observational *signal replay*, NOT proof a Kalshi order would fill:
- Polymarket wallet fill time != time it became public to a copier.
- Buyer-aggressor Kalshi trade prints are not buyable asks or sellable bids.
- BBO/snapshot quote is past as-of visible depth, not guaranteed delivery/fill.
- Source wallet cache could omit SELLs; an absent exit is not evidence of no exit.
- Fees use an illustrative general Kalshi taker formula, rounded per copied fill.
- Market settlement is sourced from Kalshi, never from the Polymarket outcome.

Default: replay wallet-sized orders, rounded to whole contracts. For a 10-share
copy of EACH wallet fill use --quantity-mode fixed --qty 10. This is not the
same as copying only the first fill: all recorded buys/additions/hedges/sells
are processed chronologically, one attempted copied order per record.
"""
from __future__ import annotations

import argparse
import bisect
import math
import runpy
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / 'w954_kalshi_crossvenue_backtest_readonly.py'
WALLET_PATH = HERE / 'w954_post_entry_sequence_audit_readonly.py'
BOOK_PATH = HERE / 'w954_kalshi_historical_bbo_replay_readonly.py'


def script(path):
    if not path.is_file():
        raise RuntimeError(f'Missing companion research script: {path}')
    return runpy.run_path(str(path))


def prepare_wallet(path=None, start_day=None, end_day=None, max_markets=0):
    loader = script(WALLET_PATH)
    p = Path(path) if path else next((p for p in loader['DEFAULT_PATHS'] if p.is_file()), None)
    if p is None or not p.is_file():
        raise RuntimeError('Wallet trades.jsonl absent. Use --wallet /path/to/wallet_954b/trades.jsonl')
    grouped, raw, invalid, confirmed, collisions = loader['load_trades'](p)
    if invalid or collisions or confirmed != raw:
        raise RuntimeError(f'Unsafe wallet source: raw={raw} invalid={dict(invalid)} '
                           f'confirmed={confirmed} collisions={len(collisions)}')
    markets = sorted(grouped.items())
    if start_day: markets = [(s, rows) for s, rows in markets if datetime.fromtimestamp(s, timezone.utc).strftime('%Y-%m-%d') >= start_day]
    if end_day: markets = [(s, rows) for s, rows in markets if datetime.fromtimestamp(s, timezone.utc).strftime('%Y-%m-%d') <= end_day]
    if max_markets: markets = markets[:max_markets]
    events = []
    for start, rows in markets:
        for r in rows:
            if not (start <= r['ts'] < start + 900): raise RuntimeError('Wallet record outside market')
            events.append({**r, 'start': start, 'close': start + 900})
    events.sort(key=lambda r: (r['ts'], r['start'], r['line']))
    return events, {'raw': raw, 'grouped_markets': len(grouped), 'selected_markets': len(markets),
                    'selected_fills': len(events), 'invalid': dict(invalid), 'path': str(p)}


def match_markets(events, markets, core):
    by_close = defaultdict(list)
    for m in markets:
        if not str(m.get('ticker', '')).startswith('KXBTC15M-'): continue
        t = core['market_close'](m)
        if t is not None: by_close[round(t)].append(m)
    mapped = {}
    errors = Counter()
    for start in sorted({e['start'] for e in events}):
        choices = by_close.get(start + 900, [])
        if len(choices) != 1:
            errors['no_market' if not choices else 'ambiguous_close'] += 1
            continue
        result = str(choices[0].get('result', '')).lower().strip()
        if result not in ('yes', 'no'):
            errors['unknown_settlement'] += 1
            continue
        mapped[start] = (choices[0]['ticker'], result)
    matched = []
    for e in events:
        if e['start'] not in mapped: continue
        ticker, result = mapped[e['start']]
        matched.append({**e, 'ticker': ticker, 'result': result,
                        'kalshi_side': 'yes' if e['side'] == 'UP' else 'no'})
    return matched, errors


def qty_for(event, args):
    qty = args.qty if args.quantity_mode == 'fixed' else int(math.floor(event['size'] * args.scale + .5))
    if qty < 1: return None, 'rounds_to_zero'
    if args.max_qty is not None and qty > args.max_qty: return None, 'order_above_qty_cap'
    return qty, None


def is_buy_aggressor(event):
    # Kalshi binary trade 'taker_outcome_side': a YES taker buys YES and sells NO.
    # Therefore to sell YES, we need a NO-taker print (and vice versa).
    if event['action'] == 'BUY': return event['kalshi_side']
    return 'no' if event['kalshi_side'] == 'yes' else 'yes'


def decode_print(row, core):
    if row.get('is_block_trade') in (True, 'true', 'True'): return None
    taker = str(row.get('taker_outcome_side') or row.get('taker_side') or '').lower()
    if taker not in ('yes', 'no'): return None
    t = core['timestamp'](row.get('created_time') or row.get('ts'))
    if t is None: return None
    price_yes = core['trade_price'](row)
    q = float(row.get('count_fp', row.get('count', 0)))
    if not (0 < price_yes < 1 and math.isfinite(q) and q > 0): return None
    return {'t': t, 'yes': price_yes, 'taker': taker, 'qty': q,
            'id': str(row.get('trade_id') or row.get('id') or '')}


def load_trade_prints(events, args, core):
    by_ticker = defaultdict(list)
    tickers = {e['ticker'] for e in events}
    if args.trades_file:
        print(f'Loading local Kalshi trade prints: {args.trades_file}', flush=True)
        for row in core['objrows'](args.trades_file):
            ticker = str(row.get('ticker', ''))
            if ticker not in tickers: continue
            try: p = decode_print(row, core)
            except (ValueError, TypeError): p = None
            if p is not None: by_ticker[ticker].append(p)
    else:
        if args.offline: raise RuntimeError('Offline mode requires --trades-file in prints mode')
        cutoff_data = core['api_get']('/historical/cutoff')
        cutoff = core['timestamp'](cutoff_data.get('trades_created_ts'))
        if cutoff is None: raise RuntimeError('No Kalshi trade cutoff')
        grouped = defaultdict(list)
        for e in events: grouped[e['ticker']].append(e)
        for idx, (ticker, market_events) in enumerate(sorted(grouped.items()), 1):
            # Narrow per-event windows, merging nearby requests to limit volume.
            bounds = []
            for e in market_events:
                lo = int(e['ts'] + min(args.latencies))
                hi = int(min(e['close'] - 1, e['ts'] + max(args.latencies) + args.max_wait_s + 2))
                if lo <= hi: bounds.append((lo, hi))
            bounds.sort()
            merged = []
            for lo, hi in bounds:
                if merged and lo <= merged[-1][1] + args.merge_gap_s:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else: merged.append((lo, hi))
            seen = set()
            for lo, hi in merged:
                for row in core['public_trades_api'](ticker, lo, hi, cutoff):
                    try: p = decode_print(row, core)
                    except (ValueError, TypeError): continue
                    if p is None or not lo <= p['t'] <= hi: continue
                    key = p['id'] if p['id'] else (p['t'], p['yes'], p['taker'], p['qty'])
                    if key in seen: continue
                    seen.add(key)
                    by_ticker[ticker].append(p)
            if idx % 20 == 0 or idx == len(grouped):
                print(f'  Kalshi trade prints checked: {idx}/{len(grouped)} markets', flush=True)
    for ticker in by_ticker: by_ticker[ticker].sort(key=lambda r: (r['t'], r['id']))
    return by_ticker


def prep_prints(prints):
    prepared = {}
    for ticker, rows in prints.items():
        buckets = {'yes': [], 'no': []}
        for n, r in enumerate(rows):
            buckets[r['taker']].append({**r, 'idx': n, 'used': 0.0})
        prepared[ticker] = {s: (rows, [r['t'] for r in rows]) for s, rows in buckets.items()}
    return prepared


def execute_print(event, qty, time_at, prepared, max_wait):
    d = prepared.get(event['ticker'], {})
    bucket = d.get(is_buy_aggressor(event))
    if not bucket: return None, 'no_print_for_aggressor_side'
    rows, times = bucket
    i = bisect.bisect_left(times, time_at)
    enough_any = False
    while i < len(rows) and rows[i]['t'] <= min(time_at + max_wait, event['close'] - 1e-6):
        row = rows[i]
        enough_any = True
        if row['qty'] - row['used'] + 1e-8 >= qty:
            row['used'] += qty  # no duplicate borrowing from same print
            price = row['yes'] if event['kalshi_side'] == 'yes' else 1 - row['yes']
            return {'price': price, 'quote_time': row['t'], 'volume': row['qty'], 'kind': 'trade-print'}, None
        i += 1
    return None, 'insufficient_print_size' if enough_any else 'no_fresh_aggressor_print'


def prep_books(books):
    return {ticker: ([r[0] for r in rows], rows) for ticker, rows in books.items()}


def consume_levels(levels, qty):
    available = sum(sz for _, sz in levels)
    if available + 1e-8 < qty: return None
    left = float(qty); paid = 0.
    for level in levels:
        px, sz = level
        take = min(sz, left)
        paid += take * px
        level[1] -= take
        left -= take
        if left <= 1e-8: break
    return paid / qty


def execute_book(event, qty, time_at, prepared, max_age, taken):
    entry = prepared.get(event['ticker'])
    if not entry: return None, 'no_book_for_market'
    times, rows = entry
    i = bisect.bisect_right(times, time_at) - 1
    if i < 0: return None, 'no_prior_quote'
    at, yes_levels, no_levels = rows[i]
    if time_at - at < -1e-8: raise AssertionError('Future quote used')
    if time_at - at > max_age: return None, 'stale_quote'
    # For SELL YES we lift NO ask (equivalent to hitting YES bid).
    # Consume book-side depth once per snapshot, not once for every wallet fill.
    use_yes_ask = (event['action'] == 'BUY') == (event['kalshi_side'] == 'yes')
    k = (event['ticker'], i, 'yes_ask' if use_yes_ask else 'no_ask')
    if k not in taken:
        raw = yes_levels if use_yes_ask else no_levels
        taken[k] = [[float(p), float(q)] for p, q in raw]
    vwap = consume_levels(taken[k], qty)
    if vwap is None: return None, 'insufficient_historical_depth'
    side_price = vwap if event['action'] == 'BUY' else 1 - vwap
    return {'price': side_price, 'quote_time': at, 'kind': 'as-of-book'}, None


def fee(price, qty, multiplier):
    return math.ceil(100 * multiplier * .07 * qty * price * (1 - price) - 1e-10) / 100


def simulate(events, args, latency, prepared_prints=None, prepared_books=None):
    states = defaultdict(lambda: {'yes': 0, 'no': 0, 'cash': 0., 'fees': 0., 'buy_cost': 0.,
                                  'sell_proceeds': 0., 'max_outlay': 0., 'accepted': 0,
                                  'rejected': 0, 'actions': Counter()})
    rejected = Counter(); executed = []; used_book = {}
    # Print consumption must reset for each independent latency scenario.
    if prepared_prints is not None:
        pp = {t: {s: ([{**r, 'used': 0.0} for r in group[0]], group[1])
                  for s, group in z.items()} for t, z in prepared_prints.items()}
    else: pp = None
    for e in events:
        state = states[e['ticker']]
        state['actions'][e['action'] + '_' + e['kalshi_side']] += 1
        qty, err = qty_for(e, args)
        if err:
            rejected[err] += 1; state['rejected'] += 1; continue
        if e['action'] == 'SELL' and state[e['kalshi_side']] < qty:
            rejected['sell_exceeds_copied_inventory'] += 1; state['rejected'] += 1; continue
        at = e['ts'] + latency
        if at >= e['close']:
            rejected['too_late_to_copy'] += 1; state['rejected'] += 1; continue
        if pp is not None:
            quote, err = execute_print(e, qty, at, pp, args.max_wait_s)
        else:
            quote, err = execute_book(e, qty, at, prepared_books, args.max_quote_age_s, used_book)
        if err:
            rejected[err] += 1; state['rejected'] += 1; continue
        px = quote['price']
        if not (0 < px < 1):
            rejected['invalid_quote_price'] += 1; state['rejected'] += 1; continue
        # The observation can differ from the printed price of the wallet's fill.
        # No filter on wallet price: we attempt *every* recorded fill.
        trans_fee = fee(px, qty, args.fee_multiplier)
        if e['action'] == 'BUY':
            state[e['kalshi_side']] += qty
            state['buy_cost'] += px * qty + trans_fee
            state['cash'] -= px * qty + trans_fee
        else:
            state[e['kalshi_side']] -= qty
            state['sell_proceeds'] += px * qty - trans_fee
            state['cash'] += px * qty - trans_fee
        state['fees'] += trans_fee
        state['max_outlay'] = max(state['max_outlay'], -state['cash'])
        state['accepted'] += 1
        executed.append({**e, 'copy_qty': qty, 'copy_px': px, 'fee': trans_fee,
                         'copy_at': at, 'source_at': quote['quote_time']})
    pnl_by_market = []
    for ticker, s in states.items():
        # Market result is independently supplied by Kalshi.
        row = next(e for e in events if e['ticker'] == ticker)
        payout = s[row['result']]
        pnl = s['cash'] + payout
        pnl_by_market.append({'ticker': ticker, 'start': row['start'], 'pnl': pnl,
                              'buy_cost': s['buy_cost'], 'max_outlay': s['max_outlay'],
                              'filled': s['accepted'], 'missed': s['rejected'],
                              'payout': payout, 'fees': s['fees'],
                              'yes_remain': s['yes'], 'no_remain': s['no']})
    return executed, pnl_by_market, rejected


def present(label, events, executed, bymarket, rejected):
    n = len(events)
    filled = len(executed)
    pnl = sum(r['pnl'] for r in bymarket)
    cost = sum(r['buy_cost'] for r in bymarket)
    complete = [r for r in bymarket if r['missed'] == 0]
    partial = [r for r in bymarket if r['missed'] != 0]
    print(f'\n{label} | attempted={n:,} accepted={filled:,} ({100*filled/n if n else 0:.1f}%) '
          f'markets={len(bymarket):,} COMPLETE={len(complete):,} partial={len(partial):,}')
    print(f'  copied fills BUY={sum(e["action"]=="BUY" for e in executed):,} '
          f'SELL={sum(e["action"]=="SELL" for e in executed):,} '
          f'UP={sum(e["side"]=="UP" for e in executed):,} DOWN={sum(e["side"]=="DOWN" for e in executed):,}')
    print(f'  ALL-market partial-copy PnL=${pnl:+,.2f} gross-buy-outlay=${cost:,.2f} '
          f'ROI-on-gross-buys={100*pnl/cost:+.2f}%' if cost else '  No buy outlay — no ROI')
    complete_cost = sum(r['buy_cost'] for r in complete)
    complete_pnl = sum(r['pnl'] for r in complete)
    print(f'  COMPLETE-market-only PnL=${complete_pnl:+,.2f} '
          f'gross-buy-outlay=${complete_cost:,.2f} '
          f'ROI={100*complete_pnl/complete_cost:+.2f}%' if complete_cost else '  No complete-market ROI')
    if bymarket:
        print(f'  max single-market cash outlay=${max(r["max_outlay"] for r in bymarket):,.2f} '
              f'fees=${sum(r["fees"] for r in bymarket):,.2f}')
    print(f'  rejected fills by reason: {dict(rejected.most_common())}')
    for week in sorted({datetime.fromtimestamp(r['start'], timezone.utc).strftime('%G-W%V') for r in bymarket}):
        grp = [r for r in bymarket if datetime.fromtimestamp(r['start'], timezone.utc).strftime('%G-W%V') == week]
        v = sum(x['pnl'] for x in grp); c = sum(x['buy_cost'] for x in grp)
        print(f'  {week} markets={len(grp):4} filled={sum(x["filled"] for x in grp):5} '
              f'missed={sum(x["missed"] for x in grp):5} '
              f'pnl=${v:+10.2f} ROI={100*v/c:+7.2f}%' if c else f'  {week}: no buy cost')


def self_test():
    from types import SimpleNamespace
    class NS: pass
    a = NS(); a.qty = 10; a.scale = 1.; a.max_qty = None; a.quantity_mode = 'fixed'
    a.max_wait_s = 1.; a.max_quote_age_s = 2.; a.fee_multiplier = 1.
    start = 1_792_332_000
    events = [dict(ticker='KXBTC15M-X', start=start, close=start+900, result='yes',
                   side='UP', kalshi_side='yes', action=act, ts=start+offset,
                   size=20., price=.3, line=i)
              for i, (act, offset) in enumerate([('BUY',10),('BUY',14),('SELL',19)], 1)]
    events.append(dict(ticker='KXBTC15M-X', start=start, close=start+900, result='yes',
                       side='DOWN', kalshi_side='no', action='BUY', ts=start+20,
                       size=12., price=.6, line=4))
    books = {'KXBTC15M-X': ([start+9,start+13,start+18,start+20], [
        (start+9,[(.30,10)],[(.74,10)]),
        (start+13,[(.33,10)],[(.71,10)]),
        (start+18,[(.36,10)],[(.68,10)]),
        (start+20,[(.36,10)],[(.65,10)]),
    ])}
    v, m, r = simulate(events, a, 0, prepared_books=books)
    assert len(v)==4 and not r, (v,r)
    # yes 10 remain, no 10 remain; YES settles -> $10 payout
    assert m[0]['yes_remain']==10 and m[0]['no_remain']==10 and m[0]['payout']==10
    assert abs(v[2]['copy_px'] - .32)<1e-9, v[2]  # SELL yes at complement of NO ask
    # SELL without prior BUY cannot short.
    one = [dict(events[0], action='SELL')]
    v, m, r = simulate(one, a, 0, prepared_books=books)
    assert not v and r['sell_exceeds_copied_inventory']==1
    # Future-only book is not a valid as-of quote.
    v, m, r = simulate([events[0]], a, -2, prepared_books=books)
    assert not v and r['no_prior_quote']==1
    # Trade prints require matching aggressor side, preserve print quantity.
    prints = {'KXBTC15M-X': {'yes': ([{'t': start+10, 'yes': .30, 'taker':'yes','qty':10.,'idx':0,'used':0.}], [start+10]),
                             'no': ([], [])}}
    v, m, r = simulate([events[0], dict(events[0], line=22)], a, 0, prepared_prints=prints)
    assert len(v)==1 and r['insufficient_print_size']==1, (v,r)
    assert fee(.4,10,1)==.17
    print('PASS: all-fills chronology, BUY/SELL, opposite-side hedge, quote timing, depth, '
          'no repeated liquidity, trade-side proxy, Kalshi payoff, fee')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--wallet', type=Path)
    p.add_argument('--markets-file', type=Path, help='Previously downloaded Kalshi market metadata JSON/JSONL')
    p.add_argument('--trades-file', type=Path, help='Local Kalshi historical trade prints JSONL/CSV (optional)')
    p.add_argument('--price-source', choices=['prints','book'], default='prints')
    p.add_argument('--book-root', type=Path, help='Downloaded historical BBO or book_snap_1s directory')
    p.add_argument('--book-kind', choices=['bbo','book_snap_1s'], default='bbo')
    p.add_argument('--offline', action='store_true', help='Do not call any public historical APIs')
    p.add_argument('--date-start', help='UTC YYYY-MM-DD; subset of wallet market starts')
    p.add_argument('--date-end', help='UTC YYYY-MM-DD; subset of wallet market starts')
    p.add_argument('--max-markets', type=int, default=0, help='Optional chronological smoke-test cap; 0=ALL markets')
    p.add_argument('--quantity-mode', choices=['wallet','fixed'], default='wallet')
    p.add_argument('--qty', type=int, default=10, help='Contracts per copied fill with --quantity-mode fixed')
    p.add_argument('--scale', type=float, default=1., help='Multiplier on wallet shares when --quantity-mode wallet')
    p.add_argument('--max-qty', type=int, default=None, help='Safety: REJECT copied orders above this size; do not truncate')
    p.add_argument('--latencies', nargs='+', type=int, default=[0,5,10,20])
    p.add_argument('--max-wait-s', type=float, default=3., help='Trade print must appear within this many seconds')
    p.add_argument('--max-quote-age-s', type=float, default=2., help='As-of book quote max age')
    p.add_argument('--fee-multiplier', type=float, default=1.)
    p.add_argument('--merge-gap-s', type=int, default=15, help='Merge nearby historical-trades API request windows')
    args = p.parse_args()
    if args.self_test: self_test(); return
    if (args.qty <= 0 or args.scale <= 0 or args.fee_multiplier < 0 or
        not args.latencies or min(args.latencies) < 0 or max(args.latencies) > 300 or
        args.max_wait_s < 0 or args.max_quote_age_s < 0 or
        args.merge_gap_s < 0 or args.max_markets < 0 or
        args.max_qty is not None and args.max_qty < 1):
        raise SystemExit('Invalid qty / fee / latency / max-age / test parameters')
    if args.price_source == 'book' and not args.book_root:
        raise SystemExit('--price-source book requires --book-root and downloaded historical BBO/snapshots')
    if args.offline and not args.markets_file:
        raise SystemExit('--offline requires --markets-file')
    core = script(CORE_PATH)
    events, data = prepare_wallet(args.wallet, args.date_start, args.date_end, args.max_markets)
    if not events: raise SystemExit('No wallet fills selected by date/market limits')
    print('WALLET 954B -> KALSHI, EVERY RECORDED FILL (READ ONLY)')
    print('='*90)
    print(f'Source={data["path"]} raw={data["raw"]:,} markets={data["grouped_markets"]:,} '
          f'selected markets={data["selected_markets"]:,} fills={data["selected_fills"]:,}')
    print(f'Wallet action mix: {dict(Counter(e["action"] + "_" + e["side"] for e in events))}')
    if not any(e['action']=='SELL' for e in events):
        print('WARNING: source contains ZERO sells; this is all visible BUYs, not proven full wallet activity')
    print('No first-price filter. Each recorded fill is a separate attempted Kalshi order.')
    print(f'Quantity={args.quantity_mode} scale={args.scale} fixedQty={args.qty} '
          f'priceSource={args.price_source}; latencies={args.latencies}')
    print('Market metadata: local file' if args.markets_file else 'Loading Kalshi metadata via public GET...', flush=True)
    mock = [{'start': start} for start in sorted({e['start'] for e in events})]
    markets = list(core['objrows'](args.markets_file)) if args.markets_file else core['markets_api'](mock)
    paired, errors = match_markets(events, markets, core)
    matched_m = len({e['start'] for e in paired})
    print(f'Kalshi matched markets={matched_m}/{data["selected_markets"]} '
          f'paired fill records={len(paired)}/{len(events)} excludes={dict(errors)}')
    if not paired: raise SystemExit('No matching settled Kalshi BTC 15-minute windows')
    pp = None; pb = None
    if args.price_source == 'prints':
        print('TRADE-PRINT PROXY ONLY: buyer-aggressor Kalshi prints are NOT verified executable asks.', flush=True)
        pp = prep_prints(load_trade_prints(paired, args, core))
    else:
        book = script(BOOK_PATH)
        print('Loading as-of historical quotes WITH DEPTH (no orders/network for quotes)...', flush=True)
        db, files, count, bad = book['ingest'](args.book_root, args.book_kind,
                                               {e['ticker'] for e in paired})
        print(f'Historical book files={files}, relevant rows={count:,}, malformed={bad}')
        pb = prep_books(db)
    for delay in args.latencies:
        execs, markets_out, fails = simulate(paired, args, delay, pp, pb)
        present(f'LATENCY {delay}s, {args.price_source.upper()}', paired, execs, markets_out, fails)
    print('\nInterpretation: ALL MARKET positions include same-side accumulation and opposite-outcome BUYs.')
    print('The hypothetical P&L is marked to official Kalshi settlement; unfilled orders are reported, not imputed.')
    print('Partial-copy P&L may not reflect the intended full strategy, especially when SELLs are missing.')
    print('No historical BBO = no proof of executable entry. Wallet-publication latency remains unknown.')
    print('No orders, writes, or GitHub changes. API calls, if any, are public GETs only.')


if __name__ == '__main__':
    try: main()
    except (RuntimeError, ValueError, OSError) as exc:
        sys.exit('ABORT: ' + str(exc))
