#!/usr/bin/env python3
"""Read-only structural crypto arbitrage V3 training audit.

Place beside crypto_arb_v3.py. No network or order APIs, only cached V3 data.
Does NOT analyze/test/optimize on the held-out OOS period.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_arb_v3 as v3


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return dict(trades=0, wins=0, avg_gross_bps=None, avg_net_bps=None,
                    median_net_bps=None, avg_hedge_bps=None, avg_funding_bps=None,
                    avg_fees_bps=None, avg_financing_bps=None, margin_breaches=0,
                    sum_net_usd=0.)
    n = len(trades)
    notional = np.array([r['qty_asset'] * r['spot_entry'] for r in trades], float)
    # V3 sets qty_asset * spot_entry == spot_notional for every trade.
    if not np.all(notional > 0):
        raise ValueError('Nonpositive notional')
    def avg_bps(fn):
        return float(np.mean([10000 * fn(r) / notion for r, notion in zip(trades, notional)]))
    return dict(
        trades=n,
        wins=sum(r['net_usd'] > 0 for r in trades),
        avg_gross_bps=avg_bps(lambda r:r['gross_usd']),
        avg_net_bps=avg_bps(lambda r:r['net_usd']),
        median_net_bps=float(np.median([r['net_bps_spot_notional'] for r in trades])),
        avg_hedge_bps=avg_bps(lambda r:r['spot_pnl_usd']+r['perp_pnl_usd']),
        avg_funding_bps=avg_bps(lambda r:r['funding_pnl_usd']),
        avg_fees_bps=avg_bps(lambda r:r['fees_usd']),
        avg_financing_bps=avg_bps(lambda r:r['financing_usd']),
        margin_breaches=sum(bool(r['margin_buffer_breach_proxy']) for r in trades),
        sum_net_usd=float(sum(r['net_usd'] for r in trades)),
    )


def pct_positive(arr):
    return float((np.asarray(arr) > 0).mean()*100) if len(arr) else float('nan')


def run(args) -> None:
    data = Path(args.data_dir)
    spot, perp, fund = v3.load_market(data, args.symbol, args.max_funding_gap_hours)
    split = int(len(spot) * args.train_fraction)
    if split < 180:
        raise ValueError('Insufficient training hours')
    s = spot.iloc[:split].copy()
    p = perp.iloc[:split].copy()
    start = pd.to_datetime(s['timestamp'].iloc[0], utc=True)
    end = pd.to_datetime(s['timestamp'].iloc[-1], utc=True) + pd.Timedelta(hours=1)
    fund_times = v3.effective_funding_times(fund['funding_time'])
    fund_train = fund.loc[(fund_times >= start) & (fund_times < end)]
    basis = 10000 * (p['close'].to_numpy(float)/s['close'].to_numpy(float)-1)
    rates = 10000 * fund_train['funding_rate'].to_numpy(float)

    print(f'V3 AUDIT: {args.symbol} | TRAIN ONLY {start} to {end} | {split} hours | OOS untouched')
    print(f'funding events={len(rates)}; positive={pct_positive(rates):.1f}%  '
          f'avg={np.mean(rates):+.3f} bps/event, median={np.median(rates):+.3f} bps/event')
    print(f'closing perp-spot basis: positive={pct_positive(basis):.1f}%  '
          f'avg={np.mean(basis):+.2f} bps, median={np.median(basis):+.2f} bps, '
          f'p10/p90={np.percentile(basis, 10):+.2f}/{np.percentile(basis,90):+.2f} bps')

    costs = v3.Costs()  # default V3 assumptions; audit preserves analyzer defaults
    rows = []
    for pa in v3.grid():
        trades = v3.trade_ledger(spot, perp, fund, pa, costs, 1, split,
                                 args.symbol, args.notional)
        st = summarize(trades)
        lower = v3.stats(trades)['score_lower95_bps_heuristic']
        qualified = (st['trades'] >= args.min_train_trades and
                     st['margin_breaches'] == 0 and lower is not None and lower > 0)
        rows.append(dict(hold_h=pa.hold_hours, min_basis_bps=pa.min_basis_bps,
                         trailing_funding_min_bps=pa.min_trailing_funding_bps_per_event,
                         lower95_heuristic_bps=lower, training_qualified=qualified, **st))
    table = pd.DataFrame(rows)
    table = table.sort_values(['trades', 'hold_h'], ascending=[False,True])
    print('\nALL 27 PARAMETER COMBINATIONS (TRAIN ONLY, V3 DEFAULT COSTS)')
    cols = ['hold_h','min_basis_bps','trailing_funding_min_bps','trades',
            'avg_hedge_bps','avg_funding_bps','avg_fees_bps','avg_net_bps',
            'lower95_heuristic_bps','margin_breaches','training_qualified']
    print(table[cols].round(2).to_string(index=False, na_rep='---'))

    # A priori baseline: no thresholds and re-enter after scheduled exits.
    # Note: still flat for one bar between positions by V3 design; not a continuously held carry position.
    print('\nNO-THRESHOLD BENCHMARK (TRAIN ONLY; one-bar gap between positions)')
    base_rows = []
    for hold in (24,48,72):
        pa = v3.Params(hold,-1e9,-1e9)
        for name,cost in [('V3_DEFAULT', costs),
                          ('ZERO_EXEC_COST_DIAGNOSTIC', replace(costs,
                           spot_fee_bps=0., perp_fee_bps=0.,
                           spot_slippage_bps=0., perp_slippage_bps=0.))]:
            ledger = v3.trade_ledger(spot,perp,fund,pa,cost,1,split,
                                     args.symbol,args.notional)
            base_rows.append(dict(hold_h=hold, scenario=name, **summarize(ledger)))
    bdf = pd.DataFrame(base_rows)
    print(bdf[['hold_h','scenario','trades','wins','avg_hedge_bps',
               'avg_funding_bps','avg_fees_bps','avg_net_bps','margin_breaches']]
          .round(2).to_string(index=False, na_rep='---'))
    print('\nNote: hedge P&L includes slippage; funding uses approximated completed-hour marks; '
          'this benchmark is not an executable US venue.')
    print('No out-of-sample data evaluated. No new parameter selection or trading actions.')
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out/'v3_training_all_27_audit.csv', index=False)
        bdf.to_csv(out/'v3_training_unconditional_audit.csv', index=False)
        print(f'Wrote {out} (TRAIN ONLY)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default=str(v3.BASE))
    ap.add_argument('--symbol',choices=v3.SYMBOLS,default='BTCUSDT')
    ap.add_argument('--train-fraction',type=float,default=.7)
    ap.add_argument('--max-funding-gap-hours',type=float,default=12.)
    ap.add_argument('--notional',type=float,default=1000.)
    ap.add_argument('--min-train-trades',type=int,default=10)
    ap.add_argument('--output-dir',default='')
    args=ap.parse_args()
    if not .5 <= args.train_fraction <= .9:
        ap.error('train fraction must be between 0.5 and 0.9')
    run(args)

if __name__ == '__main__':
    main()
