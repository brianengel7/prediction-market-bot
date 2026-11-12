#!/usr/bin/env python3
"""Read-only SELL relevance audit for 954B's frozen Aug 6–Oct 5 test.

Find out whether SELL events in the preperiod are unrelated wallet activity or
occur in qualifying BTC 15m markets where the buy-only copy would diverge.
Uses the original date-bounded, cursor-paginated Polymarket GET collector.
No local writes, order submissions, paid APIs, or Kalshi calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import runpy
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PRE = HERE / 'w954_frozen_preperiod_60d_backtest_readonly.py'
OOS = HERE / 'w954_all_subsequent_caps_oos_readonly.py'


def as_day(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime('%Y-%m-%d')


def first_buy(rows):
    return next((r for r in rows if r['action'] == 'BUY'), None)


def position_relevance(signals, grouped):
    """Audit trade activity relative to the first eligible BUY, with no future filtering."""
    summary = Counter()
    samples = []
    reports = []
    for s in signals:
        rows = grouped[s['start']]
        first = first_buy(rows)
        if first is None:
            raise RuntimeError('Qualifying first signal has no wallet BUY')
        start_clock = (first['ts'], first['line'])
        sells = [r for r in rows if r['action'] == 'SELL']
        before = [r for r in sells if (r['ts'], r['line']) < start_clock]
        after = [r for r in sells if (r['ts'], r['line']) >= start_clock]
        # Confirmation is considered eligible only if the original frozen
        # signal's 4–30s, no-prior-opposite-BUY rule is met.
        repeat = s.get('first_same')
        confirms = bool(repeat and 4 <= repeat['ts']-s['ts'] <= 30
                        and not s.get('prior_opposite', False))
        repeat_clock = (repeat['ts'], repeat['line']) if confirms else None
        after_conf = [r for r in after if repeat_clock is not None
                      and (r['ts'], r['line']) > repeat_clock]
        sells_same = [r for r in after if r['side'] == s['side']]
        sells_other = [r for r in after if r['side'] != s['side']]
        summary['qualifying_markets'] += 1
        summary['qualifying_confirmed_markets'] += int(confirms)
        summary['qualifying_markets_with_any_sell'] += bool(sells)
        summary['qualifying_markets_with_sell_before_first'] += bool(before)
        summary['qualifying_markets_with_sell_after_first'] += bool(after)
        summary['qualifying_markets_with_sell_after_confirm'] += bool(after_conf)
        summary['qualifying_markets_with_sell_same_side_after_first'] += bool(sells_same)
        summary['qualifying_markets_with_sell_other_side_after_first'] += bool(sells_other)
        summary['qualifying_sells_before_first'] += len(before)
        summary['qualifying_sells_after_first'] += len(after)
        summary['qualifying_sells_after_confirm'] += len(after_conf)
        summary['qualifying_sells_same_side_after_first'] += len(sells_same)
        summary['qualifying_sells_other_side_after_first'] += len(sells_other)
        if sells:
            reports.append((s, first, sells, bool(confirms)))
            for r in sells:
                if len(samples) >= 16:
                    break
                samples.append((s, first, r, confirms))
    return summary, samples, reports


def self_test():
    f = lambda ts, action, side, line: dict(ts=ts, action=action,
                                          side=side, line=line, size=50.)
    # First market: a SELL follows the actual entry and confirmation.
    market_a = [f(1010,'BUY','UP',1), f(1018,'BUY','UP',2), f(1022,'SELL','UP',3)]
    # Second market: SELL before first BUY, which may mean pre-existing inventory.
    market_b = [f(1910,'SELL','DOWN',4), f(1911,'BUY','UP',5)]
    sigs = [dict(start=900,ts=1010,side='UP',first_same=market_a[1],prior_opposite=False),
            dict(start=1800,ts=1911,side='UP',first_same=None,prior_opposite=False)]
    stats, samples, reports = position_relevance(sigs,{900:market_a,1800:market_b})
    assert stats['qualifying_markets']==2
    assert stats['qualifying_confirmed_markets']==1
    assert stats['qualifying_markets_with_sell_after_first']==1
    assert stats['qualifying_markets_with_sell_after_confirm']==1
    assert stats['qualifying_markets_with_sell_before_first']==1
    assert len(samples)==2 and len(reports)==2
    print('PASS: first-buy/confirmation SELL timing, pre-entry SELL, strict signal-market separation (offline)')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--start',default='2026-08-06')
    parser.add_argument('--end',default='2026-10-06',help='exclusive')
    parser.add_argument('--max-pages',type=int,default=100)
    parser.add_argument('--run-if-safe',action='store_true',
                        help='If NO qualifying BTC market has any SELL, immediately run frozen Kalshi replay from same in-memory wallet history')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not PRE.exists() or not OOS.exists():
        raise SystemExit('Place beside w954_frozen_preperiod_60d_backtest_readonly.py and w954_all_subsequent_caps_oos_readonly.py')
    pre=runpy.run_path(str(PRE))
    old=runpy.run_path(str(OOS))
    _, p, _, _, a, follow=old['modules']()
    begin,end=old['date_epoch'](args.start),old['date_epoch'](args.end)
    pre['guard_old_window'](begin,end,old)
    print('954B PRIOR 61-DAY SELL RELEVANCE AUDIT — PUBLIC GET, READ ONLY')
    print('='*94)
    print(f'UTC wallet window [{args.start}, {args.end}), no later test periods used')
    raw=pre['fetch_bounded'](old,a,begin,end,args.max_pages)
    grouped,missed,duplicates=old['decode_wallet'](raw,a,begin,end,
                             int(datetime.now(timezone.utc).timestamp()))
    signals,initial,bad=old['derive_first'](grouped,follow)
    allrows=[r for rows in grouped.values() for r in rows]
    all_sells=[r for r in allrows if r['action']=='SELL']
    first_cohort={s['start'] for s in signals}
    sell_markets={s for s,rows in grouped.items() if any(x['action']=='SELL' for x in rows)}
    summary,samples,reports=position_relevance(signals,grouped)
    monthly=defaultdict(lambda:Counter())
    for start,rows in grouped.items():
        month=as_day(start)[:7]
        monthly[month]['markets']+=1
        monthly[month]['qualifying']+=start in first_cohort
        monthly[month]['markets_with_sells']+=start in sell_markets
        monthly[month]['sells']+=sum(x['action']=='SELL' for x in rows)
    print('\nFULL WALLET SOURCE COVERAGE')
    print(f'  Raw API rows: {len(raw):,}, valid BTC 15m rows: {len(allrows):,}')
    print(f'  BTC 15m markets: {len(grouped):,}, valid first BUY markets: {len(initial):,}')
    print(f'  BTC BUY records: {sum(x["action"]=="BUY" for x in allrows):,}')
    print(f'  BTC SELL records: {len(all_sells):,} across {len(sell_markets):,} BTC markets')
    print(f'  Qualifying 20–40c first BUY markets: {len(signals):,}')
    print(f'  Duplicates: {duplicates}, exclusions: {dict(missed)}, signal join exclusions: {dict(bad)}')
    print('\nPER MONTH (BTC 15m ONLY)')
    for month,counts in sorted(monthly.items()):
        print(f'  {month}: markets={counts["markets"]}, qualifying={counts["qualifying"]}, '
              f'sell_markets={counts["markets_with_sells"]}, sells={counts["sells"]}')
    print('\nSELLS AFFECTING OUR QUALIFYING STRATEGY MARKETS')
    for name in (
        'qualifying_markets', 'qualifying_confirmed_markets',
        'qualifying_markets_with_any_sell', 'qualifying_markets_with_sell_before_first',
        'qualifying_markets_with_sell_after_first',
        'qualifying_markets_with_sell_after_confirm',
        'qualifying_markets_with_sell_same_side_after_first',
        'qualifying_markets_with_sell_other_side_after_first',
        'qualifying_sells_before_first','qualifying_sells_after_first',
        'qualifying_sells_after_confirm',
    ):
        print(f'  {name}: {summary[name]}')
    print('\nUP TO 16 SELL EXAMPLES WITHIN QUALIFYING MARKETS')
    for s,first,r,confirms in samples:
        state=('BEFORE_FIRST' if (r['ts'],r['line']) < (first['ts'],first['line']) else
               'AFTER_CONFIRM' if confirms and
                  (r['ts'],r['line']) > (s['first_same']['ts'],s['first_same']['line']) else
               'AFTER_FIRST')
        print(f'  market={as_day(s["start"])} start={s["start"]}, '
              f'first={as_day(first["ts"])} {first["ts"]}, '
              f'sell_ts={r["ts"]}, side={r["side"]}, qty={r["size"]:g}, {state}')
    print('\nASSESSMENT')
    if not reports:
        print('  No SELL in any eligible 20–40c first-buy market. The broad guard was overconservative.')
        print('  Our frozen buy-only signal set was not contaminated by BTC-market sells in other markets.')
        print('  A targeted guard can replace the old any-SELL abort; do not change entry/position rules.')
    else:
        print(f'  {len(reports)} eligible markets have SELL activity; original buy-only replay can diverge.')
        print('  We should reconcile SELL timing, copy SELL quantity at executable bid proxies, and rerun.')
        print('  Ignoring sells and reporting the result as exact wallet copying would be misleading.')
    if args.run_if_safe:
        if reports:
            print('\nRUN-IF-SAFE: BLOCKED. Eligible markets include SELLs. Do not claim a faithful buy-only wallet copy.')
        else:
            print('\nRUN-IF-SAFE: No eligible-market SELLs. Reusing same verified wallet records in frozen Kalshi replay.')
            active_days={as_day(start) for start in grouped}
            signal_days={as_day(s['start']) for s in signals}
            # Replace ONLY the in-memory wallet fetch for this invocation.
            # Exact original historical pairing, trade-print matching and
            # settlement model are otherwise unchanged.
            pre['run'].__globals__['load_preperiod'] = (lambda *_: (
                signals,grouped,active_days,signal_days))
            frozen_args=SimpleNamespace(
                start=args.start,end=args.end,max_pages=args.max_pages,
                probe_only=False,min_active_days=25,min_signals=30,
                min_match_fraction=.90,base_qty=10,latencies=[5,10,20],
                max_wait_s=5.,trades_file=None,markets_file=None,
                offline=False,caps=[1_000_000],cap=1_000_000,
                bootstrap=0)
            pre['run'](frozen_args)
    print('\nREAD ONLY: no file writes, orders or POST requests. Kalshi order books not fetched.')

if __name__=='__main__':
    main()
