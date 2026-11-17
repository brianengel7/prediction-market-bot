#!/usr/bin/env python3
"""V4 dated-futures cash-and-carry quote scanner, READ ONLY, no trading APIs.

Input requires SAME-ASSET timestamped executable bid/ask quotes for spot and a
LINEAR CASH-SETTLED dated future. Never assumes V3 perp bars are dated futures.
An entry estimate is NOT a guaranteed arbitrage: spot and settlement index can diverge.
Realized outcomes require an explicit settlement-index + spot-exit-bid observation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = [
    'timestamp_utc', 'contract_id', 'venue', 'asset', 'expiry_utc',
    'contract_size_base', 'spot_bid', 'spot_ask', 'future_bid', 'future_ask',
    'spot_ask_depth_base', 'future_bid_depth_contracts',
    'spot_timestamp_utc', 'future_timestamp_utc',
]
SETTLEMENT_FIELDS = ['contract_id', 'expiry_utc', 'index_settlement_price', 'spot_exit_bid',
                     'spot_exit_timestamp_utc']


def ts(v):
    return pd.to_datetime(v, format='ISO8601', utc=True, errors='raise')


def check(df: pd.DataFrame, cfg) -> pd.DataFrame:
    absent = sorted(set(REQUIRED) - set(df.columns))
    if absent:
        raise ValueError('Missing quote columns: ' + ', '.join(absent))
    q = df.copy()
    for c in ('timestamp_utc', 'expiry_utc', 'spot_timestamp_utc', 'future_timestamp_utc'):
        q[c] = ts(q[c])
    for c in ('contract_size_base', 'spot_bid', 'spot_ask', 'future_bid', 'future_ask',
              'spot_ask_depth_base', 'future_bid_depth_contracts'):
        q[c] = pd.to_numeric(q[c], errors='raise').astype(float)
        if not np.isfinite(q[c]).all():
            raise ValueError(f'{c}: nonfinite')
    for c in ('contract_id', 'venue', 'asset'):
        if q[c].isna().any() or (~q[c].astype(str).str.strip().astype(bool)).any():
            raise ValueError(f'{c}: missing or blank')
    if q.duplicated(['timestamp_utc','venue','contract_id']).any():
        raise ValueError('Duplicate venue/contract/timestamp quote snapshots; resolve duplicates first')
    q['seconds_to_expiry'] = (q.expiry_utc - q.timestamp_utc).dt.total_seconds()
    q['spot_quote_age_s'] = (q.timestamp_utc - q.spot_timestamp_utc).dt.total_seconds()
    q['future_quote_age_s'] = (q.timestamp_utc - q.future_timestamp_utc).dt.total_seconds()
    q['reject_reason'] = ''
    def reject(mask, why):
        # First rejection reason retained to simplify audits.
        q.loc[mask & (q.reject_reason == ''), 'reject_reason'] = why
    reject(q.seconds_to_expiry <= 0, 'EXPIRED')
    reject(q.seconds_to_expiry > cfg.max_days * 86400, 'TOO_FAR_TO_EXPIRY')
    reject(q.seconds_to_expiry < cfg.min_hours * 3600, 'TOO_CLOSE_TO_EXPIRY')
    reject((q.spot_quote_age_s < 0) | (q.future_quote_age_s < 0), 'FUTURE_DATED_QUOTE')
    reject((q.spot_quote_age_s > cfg.max_age_s) | (q.future_quote_age_s > cfg.max_age_s), 'STALE_QUOTE')
    reject((q[['spot_bid','spot_ask','future_bid','future_ask','contract_size_base']] <= 0).any(axis=1), 'NONPOSITIVE_PRICE_SIZE')
    reject((q.spot_ask < q.spot_bid) | (q.future_ask < q.future_bid), 'CROSSED_BOOK')
    reject((q[['spot_ask_depth_base','future_bid_depth_contracts']] < 0).any(axis=1), 'NEGATIVE_DEPTH')
    return q


def scan(quotes, cfg, settlements=None):
    q = check(quotes,cfg)
    # Restrict to whole linear futures contracts; one contract represents contract_size_base units.
    q['spot_notional_usd_per_contract'] = q.contract_size_base * q.spot_ask
    q['n_contracts'] = np.floor(np.minimum(
        q.future_bid_depth_contracts,
        q.spot_ask_depth_base / q.contract_size_base)).fillna(0).astype(int)
    # Capital includes fully paid spot and independently reserved futures margin.
    denom = q.spot_notional_usd_per_contract + cfg.margin_usd_per_contract
    q['n_contracts'] = np.minimum(q.n_contracts, np.floor(cfg.capital_usd / denom).astype(int))
    q.loc[q.reject_reason != '', 'n_contracts'] = 0
    q['btc_or_asset_units'] = q.n_contracts * q.contract_size_base
    q['days_to_expiry'] = q.seconds_to_expiry / 86400
    q['gross_premium_bps'] = 1e4 * (q.future_bid - q.spot_ask) / q.spot_ask
    # Conservative projection of the terminal spot EXIT relative to settlement index:
    # reuse present half-spread, and separately subtract configurable index-vs-spot haircut.
    q['exit_spread_reserve_usd'] = q.btc_or_asset_units * (q.spot_ask - q.spot_bid) / 2
    q['index_mismatch_reserve_usd'] = (q.btc_or_asset_units * q.spot_ask
                                      * cfg.index_mismatch_bps / 1e4)
    q['basis_capture_usd'] = q.btc_or_asset_units * (q.future_bid - q.spot_ask)
    q['spot_entry_fee_usd'] = q.btc_or_asset_units * q.spot_ask * cfg.spot_fee_bps / 1e4
    # Proxy for terminal spot-sale fee. Must verify against actual futures index/spot BBO.
    q['spot_exit_fee_usd'] = q.btc_or_asset_units * q.spot_ask * cfg.spot_fee_bps / 1e4
    q['futures_fees_usd'] = q.n_contracts * (cfg.future_entry_fee_usd + cfg.future_settlement_fee_usd)
    q['spot_financing_usd'] = (q.btc_or_asset_units * q.spot_ask
                               * cfg.spot_financing_apr / 100 * q.days_to_expiry / 365)
    q['total_reserved_capital_usd'] = q.n_contracts * denom
    q['expected_net_usd'] = (q.basis_capture_usd - q.exit_spread_reserve_usd
                             - q.index_mismatch_reserve_usd - q.spot_entry_fee_usd
                             - q.spot_exit_fee_usd - q.futures_fees_usd - q.spot_financing_usd)
    q['estimated_annualized_roi_pct'] = np.where(
        (q.total_reserved_capital_usd > 0) & (q.days_to_expiry > 0),
        100 * q.expected_net_usd / q.total_reserved_capital_usd * 365 / q.days_to_expiry, np.nan)
    q.loc[(q.reject_reason == '') & (q.n_contracts <= 0), 'reject_reason'] = 'DEPTH_OR_CAPITAL'
    q.loc[(q.reject_reason == '') & (q.expected_net_usd <= cfg.min_net_usd), 'reject_reason'] = 'NET_BELOW_MINIMUM'
    q.loc[(q.reject_reason == '') & (q.estimated_annualized_roi_pct < cfg.min_apr_pct), 'reject_reason'] = 'APR_BELOW_MINIMUM'
    q['candidate'] = q.reject_reason == ''
    q['pnl_type'] = 'ENTRY_ESTIMATE_NOT_GUARANTEED'
    if settlements is not None:
        missing = sorted(set(SETTLEMENT_FIELDS)-set(settlements.columns))
        if missing:
            raise ValueError('Missing settlement columns: '+', '.join(missing))
        s = settlements.copy()
        s['expiry_utc'] = ts(s['expiry_utc'])
        s['spot_exit_timestamp_utc'] = ts(s['spot_exit_timestamp_utc'])
        for c in ('index_settlement_price','spot_exit_bid'):
            s[c] = pd.to_numeric(s[c], errors='raise')
            if (~np.isfinite(s[c]) | (s[c] <= 0)).any():
                raise ValueError('Invalid settlement price '+c)
        if s.duplicated(['contract_id','expiry_utc']).any():
            raise ValueError('Duplicate contract settlements')
        if (s.spot_exit_timestamp_utc < s.expiry_utc).any():
            raise ValueError('Spot exit priced BEFORE expiry; invalid settlement audit')
        if ((s.spot_exit_timestamp_utc - s.expiry_utc).dt.total_seconds() > cfg.max_exit_delay_s).any():
            raise ValueError('Spot liquidation quote too far after expiry; require closer data')
        q = q.merge(s, on=['contract_id','expiry_utc'], how='left', validate='many_to_one')
        ok = q.index_settlement_price.notna() & (q.n_contracts > 0)
        q['realized_net_usd'] = np.nan
        # The futures leg is cash settled at the actual expiry index.
        # The spot leg is liquidated at actual executable BID from the settlement dataset.
        # A futures settlement processing fee and liquidating spot fee are both charged.
        actual = (q.btc_or_asset_units * (q.future_bid - q.index_settlement_price
                  + q.spot_exit_bid - q.spot_ask)
                  - q.spot_entry_fee_usd
                  - q.btc_or_asset_units * q.spot_exit_bid * cfg.spot_fee_bps/1e4
                  - q.futures_fees_usd - q.spot_financing_usd)
        q.loc[ok, 'realized_net_usd'] = actual[ok]
        q['pnl_type'] = np.where(ok, 'REALIZED_AT_SUPPLIED_EXPIRY_QUOTES',
                                  'SETTLEMENT_MISSING_NO_REALIZED_PNL')
    return q.sort_values(['candidate','estimated_annualized_roi_pct'], ascending=[False,False], na_position='last')


def example():
    """Illustrative mock quotes only. Never publish as market observations."""
    return pd.DataFrame([dict(timestamp_utc='2026-11-13T14:00:00Z',
        contract_id='EXAMPLE-2026-11-27', venue='SYNTHETIC', asset='BTC',
        expiry_utc='2026-11-27T16:00:00Z', contract_size_base=.01,
        spot_bid=70000, spot_ask=70005, future_bid=71320, future_ask=71330,
        spot_ask_depth_base=.25, future_bid_depth_contracts=10,
        spot_timestamp_utc='2026-11-13T13:59:59Z',
        future_timestamp_utc='2026-11-13T13:59:59Z')])


def self_test(args):
    q = scan(example(), args)
    assert len(q) == 1 and q.iloc[0].n_contracts > 0
    assert q.iloc[0].candidate and q.iloc[0].expected_net_usd > 0
    stale = example()
    stale['spot_timestamp_utc'] = '2026-11-13T13:00:00Z'
    assert scan(stale,args).iloc[0].reject_reason == 'STALE_QUOTE'
    crossed = example()
    crossed['future_bid'] = 71400
    crossed['future_ask'] = 71300
    assert scan(crossed,args).iloc[0].reject_reason == 'CROSSED_BOOK'
    future = example()
    future['future_timestamp_utc'] = '2026-11-13T14:00:01Z'
    assert scan(future,args).iloc[0].reject_reason == 'FUTURE_DATED_QUOTE'
    settlement = pd.DataFrame([dict(contract_id='EXAMPLE-2026-11-27',
        expiry_utc='2026-11-27T16:00:00Z', index_settlement_price=71000,
        spot_exit_bid=70990, spot_exit_timestamp_utc='2026-11-27T16:00:01Z')])
    x = scan(example(),args,settlement).iloc[0]
    qty=x.btc_or_asset_units
    expected = (qty*(71320-71000+70990-70005) - x.spot_entry_fee_usd
                -qty*70990*args.spot_fee_bps/1e4-x.futures_fees_usd-x.spot_financing_usd)
    assert abs(x.realized_net_usd - expected) < 1e-8
    assert x.pnl_type == 'REALIZED_AT_SUPPLIED_EXPIRY_QUOTES'
    s2=settlement.copy()
    s2['spot_exit_timestamp_utc']='2026-11-27T15:59:59Z'
    try:
        scan(example(),args,s2)
        raise AssertionError('Pre-expiry exit incorrectly accepted')
    except ValueError as e:
        assert 'BEFORE expiry' in str(e)
    bad=example()
    bad.loc[0,'future_bid_depth_contracts']=0
    assert scan(bad,args).iloc[0].reject_reason=='DEPTH_OR_CAPITAL'
    print('PASS V4: quotes, age, depth, crossed books, cash settlement, financing, fees, no lookahead')
    print('READ ONLY; SYNTHETIC TEST DATA; NO ORDERS OR AUTOMATED MARKET DATA')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['self-test','example','scan'])
    p.add_argument('--quotes',type=Path)
    p.add_argument('--settlements',type=Path)
    p.add_argument('--output',type=Path,default=Path('src/crypto-arb/data/v4/dated_futures_scan.csv'))
    p.add_argument('--max-age-s',type=float,default=10)
    p.add_argument('--max-exit-delay-s',type=float,default=60)
    p.add_argument('--min-hours',type=float,default=4)
    p.add_argument('--max-days',type=float,default=90)
    p.add_argument('--capital-usd',type=float,default=5000)
    p.add_argument('--margin-usd-per-contract',type=float,default=250)
    p.add_argument('--spot-fee-bps',type=float,default=10)
    p.add_argument('--future-entry-fee-usd',type=float,default=1.5)
    p.add_argument('--future-settlement-fee-usd',type=float,default=1.5)
    p.add_argument('--spot-financing-apr',type=float,default=5)
    p.add_argument('--index-mismatch-bps',type=float,default=10)
    p.add_argument('--min-net-usd',type=float,default=0)
    p.add_argument('--min-apr-pct',type=float,default=0)
    a=p.parse_args()
    for c in ('max_age_s','max_exit_delay_s','min_hours','max_days','capital_usd',
              'margin_usd_per_contract','spot_fee_bps','future_entry_fee_usd',
              'future_settlement_fee_usd','spot_financing_apr','index_mismatch_bps'):
        if not math.isfinite(getattr(a,c)) or getattr(a,c)<0:
            p.error(f'--{c.replace("_","-")} must be finite and nonnegative')
    if a.capital_usd <= 0 or a.max_days <= 0 or a.max_age_s <= 0:
        p.error('capital, max-days, max-age must be positive')
    if a.action=='self-test':
        self_test(a)
        return
    if a.action=='example':
        data=example()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        data.to_csv(a.output,index=False)
        print('Wrote SYNTHETIC EXAMPLE input to',a.output)
        return
    if not a.quotes or not a.quotes.is_file():
        p.error('scan requires --quotes <existing live/historical quote CSV>; no data are invented')
    q=pd.read_csv(a.quotes)
    s=pd.read_csv(a.settlements) if a.settlements else None
    result=scan(q,a,s)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    cols=['timestamp_utc','venue','asset','contract_id','days_to_expiry','n_contracts',
          'gross_premium_bps','expected_net_usd','estimated_annualized_roi_pct',
          'candidate','reject_reason','pnl_type']
    print(result[cols].to_string(index=False))
    print(f'Rows={len(result)} candidates={result.candidate.sum()} written={a.output}')
    print('ESTIMATES NOT EXECUTABLE P&L. Confirm account fees, depth, margin, settlement index and licensed data.')

if __name__=='__main__':
    main()
