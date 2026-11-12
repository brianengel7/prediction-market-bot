#!/usr/bin/env python3
"""Frozen 954B strategy test on an earlier, untouched 30-60 day period.

DATA WINDOW (defaults): 2026-08-06 <= BTC 15-minute market start < 2026-10-06 UTC.
The old tuning data starts October 6, and the newer November 6-10 data was
already consulted; neither is used for strategy signals or pricing here.

Frozen trading rules (do not tune them on this pre-period):
 * First wallet BUY in a Polymarket BTC Up/Down 15-minute market at [20,40)c:
   buy 10 matched Kalshi YES/NO contracts, with simulated 5s latency.
 * If the first same-side repeat occurs 4-30 seconds later with no earlier
   opposing BUY, proportionally scale the Kalshi position to wallet's
   cumulative same-side shares / first shares.
 * Copy EVERY further same-side purchase, proportionally, with no window,
   additional wait, or practical position cap; stop adding at the first
   opposite-side wallet BUY. Do not copy sells or opposite-side positions.
 * Hold to the independently verified Kalshi settlement; charge taker fees.

This is read-only (public GET and local reads only). Its Kalshi fill model uses
historical aggressive-buy TRADE PRINTS, NOT historical executable BBO asks.
Wallet-publication delay is not included. Never treat this as proof of real fills.

Requires existing w954_all_subsequent_caps_oos_readonly.py and its companion
research scripts in the same src/copy-trading directory.

Examples:
 python3 -B src/copy-trading/w954_frozen_preperiod_60d_backtest_readonly.py --self-test
 python3 -B src/copy-trading/w954_frozen_preperiod_60d_backtest_readonly.py --probe-only
 python3 -B src/copy-trading/w954_frozen_preperiod_60d_backtest_readonly.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'w954_all_subsequent_caps_oos_readonly.py'
TRAIN_START = '2026-10-06'
CAP = 1_000_000  # same effective-uncapped configuration used in November


def iso_day(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d')


def pct(pnl, cost):
    return 100 * pnl / cost if cost else float('nan')


def line_of_range(name, rows):
    cash = sum(float(r['pnl']) for r in rows)
    cost = sum(float(r['buy_cost']) for r in rows)
    print(f'  {name:<22} N={len(rows):>4} cost=${cost:>11,.2f} PnL=${cash:>+11,.2f} ROI={pct(cash,cost):>+8.2f}%')


def by_calendar(rows, kind):
    groups = defaultdict(list)
    for r in rows:
        date = datetime.fromtimestamp(r['start'], timezone.utc)
        if kind == 'month':
            key = date.strftime('%Y-%m')
        elif kind == 'week':
            key = date.strftime('%G-W%V')
        else:
            key = date.strftime('%Y-%m-%d')
        groups[key].append(r)
    for k in sorted(groups):
        line_of_range(k, groups[k])


def fetch_bounded(old, a, begin, end, max_pages):
    """7-day user-scoped GET windows: no offset limit and no calendar leaks."""
    output = []
    left = begin
    windows = 0
    while left < end:
        right = min(end, left + 7 * 86400)
        windows += 1
        rows = old['polymarket_trades'](left, right, max_pages)
        for r in rows:
            ts = a['epoch'](a['first_present'](r, ('timestamp','ts','time','created_at')))
            if ts is None or not left <= ts < right:
                raise RuntimeError('API returned a trade outside its requested bounded period; aborting instead of mixing eras')
        output.extend(rows)
        print(f'  FETCH WINDOW {iso_day(left)} to {iso_day(right)} (exclusive): {len(rows):,} rows', flush=True)
        left = right
    print(f'  Completed {windows} bounded windows; raw wallet rows={len(output):,}')
    return output


def guard_old_window(begin, end, old):
    original_start = old['date_epoch'](TRAIN_START)
    if end > original_start:
        raise ValueError(f'Historical pre-period end must be <= {TRAIN_START} (exclusive); refusing in-sample/future leakage')
    if begin >= end:
        raise ValueError('Start must be before end')
    days = (end - begin) / 86400
    if not 28 <= days <= 75:
        raise ValueError(f'Request a contiguous 28-75 day untouched period, not {days:g} days')
    return days


def load_preperiod(a, follow, old, begin, end, max_pages):
    raw = fetch_bounded(old, a, begin, end, max_pages)
    grouped, missed, dup = old['decode_wallet'](raw, a, begin, end,
                  int(datetime.now(timezone.utc).timestamp()))
    # Outside-market and user/boundary mismatches already abort in decode_wallet.
    buys = [r for market in grouped.values() for r in market]
    if any(r['action'] == 'SELL' for r in buys):
        raise RuntimeError('Unexpected SELL history; buy-only strategy cannot faithfully replay inventory. Stop.')
    signals, initial, bad = old['derive_first'](grouped, follow)
    active_days = {iso_day(start) for start in grouped}
    qualifying_days = {iso_day(s['start']) for s in signals}
    print('\nPRE-PERIOD SOURCE / COHORT')
    print(f'  Raw user trade records: {len(raw):,} | BTC 15m markets: {len(grouped):,} | BTC first buys: {len(initial):,}')
    print(f'  Qualifying 20-40c first buys: {len(signals)} | qualifying dates: {len(qualifying_days)} | active BTC dates: {len(active_days)}')
    print(f'  Exact-id duplicates={dup} | non-BTC skips={dict(missed)} | join excludes={dict(bad)}')
    if grouped:
        print(f'  Earliest and latest BTC market starts: {iso_day(min(grouped))} ... {iso_day(max(grouped))}')
    if grouped:
        for mo in sorted({iso_day(k)[:7] for k in grouped}):
            mdays = {iso_day(k) for k in grouped if iso_day(k).startswith(mo)}
            m_signals = sum(iso_day(s['start']).startswith(mo) for s in signals)
            print(f'  Month {mo}: active BTC market days={len(mdays):>2}, qualifying first buys={m_signals:>4}')
    if signals:
        print(f'  Earliest and latest QUALIFYING market starts: {iso_day(min(s["start"] for s in signals))} ... '
              f'{iso_day(max(s["start"] for s in signals))}')
    return signals, grouped, active_days, qualifying_days


def run(args):
    if not SOURCE.is_file():
        raise RuntimeError(f'Missing existing companion backtester: {SOURCE}')
    old = runpy.run_path(str(SOURCE))
    m, p, core, af, a, follow = old['modules']()
    begin, end = old['date_epoch'](args.start), old['date_epoch'](args.end)
    days = guard_old_window(begin, end, old)
    print('954B FROZEN CONTINUOUS PROPORTIONAL COPY — EARLIER UNTOUCHED PERIOD')
    print('='*103)
    print(f'  Window [{args.start}, {args.end}) UTC = {days:.0f} calendar days; no Nov-2026 or Oct-6 onward trades included')
    print('  First buy [20c,40c), confirmation within 4-30s, copy ALL later same-side BUYs, stop on opposing BUY')
    print(f'  Base=10 Kalshi contracts, effective cap={CAP:,}, order-price proxy=Kalshi buyer-aggressor prints')
    print('  Freeze: 5s primary latency; 10s/20s sensitivity; +2c/+5c adverse price stress; official Kalshi settlement')

    signals, grouped, active_days, signal_days = load_preperiod(a, follow, old, begin, end, args.max_pages)
    if args.probe_only:
        print('\nPROBE ONLY: no Kalshi requests, P&L, or backtest performed.')
        return
    if len(active_days) < args.min_active_days or len(signals) < args.min_signals:
        print(f'\nINSUFFICIENT HISTORICAL COVERAGE for a convincing 1-2-month replay: '
              f'active BTC days={len(active_days)} (need >= {args.min_active_days}), '
              f'qualifying signals={len(signals)} (need >= {args.min_signals}).')
        print('NO STRATEGY VERDICT. An older wallet history may not exist; a 30-60 day FUTURE forward test would then be needed.')
        return
    index = p['index_wallet']([x for rows in grouped.values() for x in rows])
    cfg = old['params'](args, CAP)
    continuous, omissions, actions, observations = m['build_all'](
        signals, index, cfg, p, stop_on_opposite=True)
    if omissions.get('market_qty_cap_confirmation') or omissions.get('market_qty_cap_all_followups'):
        raise RuntimeError(f'Even {CAP:,} cap was hit; test is not actually uncapped: {dict(omissions)}')
    reference,_,_ = p['build'](signals,index,'first',0,cfg)
    firstonly=[x for x in reference if x['kind']=='first']
    variants={'FIRST_ONLY':firstonly, 'FIRST+CONFIRM+0':reference,
              'ALL_STOP_UNCAPPED':continuous}
    print('\nFROZEN SIGNAL PLAN')
    for name, ev in variants.items():
        print(f'  {name:<22} markets={len({x["start"] for x in ev})} orders={len(ev)} '
              f'kinds={dict(Counter(x["kind"] for x in ev))}')
    print(f'  Stops/adds={dict(observations)}; constraints={dict(omissions)}')

    # Preserve the exact existing OOS market/print loader argument contract.
    args.mode = 'oos'
    args.cap = CAP
    args.caps = [CAP]
    # Only this pre-period's market identifiers and time-stamped Kalshi prints
    # are sent to the existing settlement/fee/volume-proxy replay.
    print('\nFETCHING MATCHED HISTORICAL KALSHI MARKETS AND TRADE PRINTS')
    paired, prints, excludes = old['markets_and_prints'](variants,args,p,core,af)
    if prints is None:
        print('NO KALSHI PAIRING. Inconclusive, not a loss/win.')
        return
    first_n = len({x['start'] for x in paired['FIRST_ONLY']})
    coverage=first_n/len(signals)
    print(f'  Distinct matched first-entry markets={first_n}/{len(signals)} ({coverage:.1%}); pooled excludes={dict(excludes)}')
    if coverage < args.min_match_fraction:
        raise RuntimeError(f'Kalshi matching {coverage:.1%} below minimum {args.min_match_fraction:.0%}; '
                           'cannot report a representative historical conclusion.')

    sim=SimpleNamespace(quantity_mode='wallet',scale=1.,qty=1,max_qty=None,
                        max_wait_s=args.max_wait_s,max_quote_age_s=2.,fee_multiplier=1.)
    for latency in args.latencies:
        print(f'\n=== HISTORICAL SIMULATED LATENCY {latency}s ===')
        print('Market P&L includes every winning and losing market. No outlier trimming or new parameter selection.')
        for name, events in paired.items():
            fills, markets, rejected, dependent = p['simulate_dependent'](
                events, sim, latency, af, prints, None)
            cash=sum(r['pnl'] for r in markets)
            cost=sum(r['buy_cost'] for r in markets)
            stress=p['run_price_stress'](fills,markets,[0.,2.,5.],af['fee'],1.)
            print(f'  {name:<23} markets={len(markets):>3} fills={len(fills)}/{len(events)} '
                  f'PnL=${cash:+,.2f} outlay=${cost:,.2f} ROI={pct(cash,cost):+.2f}% '
                  f'+2c={stress[1][3]:+.2f}% +5c={stress[2][3]:+.2f}%')
            if rejected or dependent:
                print(f'    Execution-proxy rejects={dict(rejected)}, dependent skipped={dependent}')
            if name=='ALL_STOP_UNCAPPED':
                timeline = sorted(markets, key=lambda r:r['start'])
                equity=0.0
                peak=0.0
                drawdown=0.0
                for row in timeline:
                    equity += row['pnl']
                    peak = max(peak, equity)
                    drawdown = max(drawdown, peak-equity)
                print(f'  Realized settlement-PnL max drawdown=${drawdown:,.2f} (NOT mark-to-market)')
                print('  CONTINUOUS strategy TWO CONSECUTIVE 30/31-DAY BLOCKS:')
                midpoint = begin + 30*86400
                line_of_range(f'{iso_day(begin)} to {iso_day(midpoint)}',
                              [r for r in markets if r['start'] < midpoint])
                line_of_range(f'{iso_day(midpoint)} to {iso_day(end)}',
                              [r for r in markets if r['start'] >= midpoint])
                print('  CONTINUOUS strategy FULL MONTHLY PERFORMANCE:')
                by_calendar(markets,'month')
                print('  CONTINUOUS strategy FULL WEEKLY PERFORMANCE:')
                by_calendar(markets,'week')
                winners=sum(r['pnl']>0 for r in markets)
                print(f'  Winning markets={winners} / {len(markets)} (wins do NOT need to be a majority)')
    print('\nINTERPRETATION: old-sample look-ahead is prevented, but retrospective pre-period selection is still a test choice.')
    print('No exact historical best asks/depth; trade-print acceptance is NOT real fillability. Wallet publication latency is unknown.')
    print('This measures independent historical directional/price-proxy repeatability, NOT definitive live copy-tradability.')
    print('READ ONLY: public GET + in-memory calculations. No orders, writes, or repository edits.')


def self_test():
    if not SOURCE.is_file():
        raise RuntimeError('Missing companion script')
    old=runpy.run_path(str(SOURCE))
    old['self_test']()
    epoch=old['date_epoch']
    assert guard_old_window(epoch('2026-08-06'),epoch('2026-10-06'),old)==61
    try:
        guard_old_window(epoch('2026-09-01'),epoch('2026-11-01'),old)
        raise AssertionError('Leaky window accepted')
    except ValueError: pass
    assert iso_day(epoch('2026-09-21'))=='2026-09-21'
    fixture=[{'pnl':100.,'buy_cost':50.,'start':epoch('2026-08-08')},
             {'pnl':-50.,'buy_cost':50.,'start':epoch('2026-09-09')}]
    assert sum(r['pnl'] for r in fixture)==50 and pct(50,100)==50
    print('PASS: pre-period no-overlap guard, frozen backtest imports, calendar accounting, existing end-to-end replay fixture')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--start',default='2026-08-06',help='UTC inclusive (old tuning begins October 6)')
    ap.add_argument('--end',default='2026-10-06',help='UTC exclusive')
    ap.add_argument('--probe-only',action='store_true',help='Get prior wallet records and count qualifying signals, no Kalshi calls')
    ap.add_argument('--min-active-days',type=int,default=25)
    ap.add_argument('--min-signals',type=int,default=30)
    ap.add_argument('--min-match-fraction',type=float,default=.90)
    ap.add_argument('--base-qty',type=int,default=10)
    ap.add_argument('--latencies',nargs='+',type=int,default=[5,10,20])
    ap.add_argument('--max-pages',type=int,default=100,help='Max cursor pages per seven-day wallet segment')
    ap.add_argument('--max-wait-s',type=float,default=5.)
    ap.add_argument('--trades-file',type=Path,help='Optional preexisting local Kalshi trade archive')
    ap.add_argument('--markets-file',type=Path,help='Optional preexisting local Kalshi metadata archive')
    ap.add_argument('--offline',action='store_true',help='Only Kalshi API offline; wallet GET remains required')
    ap.add_argument('--bootstrap',type=int,default=0,help='Reserved: unused (no CI asserted for this retrospective test)')
    args=ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.base_qty !=10 or args.latencies !=[5,10,20]:
        ap.error('Frozen primary test: base-qty=10, latencies 5 10 20 only; avoid post hoc parameter tuning')
    if not (0<args.min_match_fraction<=1) or args.min_active_days<0 or args.min_signals<0 or args.max_pages<1:
        ap.error('Invalid minimum coverage or pagination settings')
    if args.offline and not(args.trades_file and args.markets_file):
        ap.error('Offline Kalshi mode needs both --trades-file and --markets-file')
    run(args)


if __name__=='__main__':
    main()
