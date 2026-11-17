#!/usr/bin/env python3
"""Albert1953 V8: OFFLINE finance/crypto robustness and entry-cohort audit.

Uses V6's validated, settled-only, SOURCE-PRINT hypothetical execution outputs
plus V1's complete source fills to group each market by its FIRST observed trade.
Does not estimate Kalshi market matches, orders, quote availability, account
capital, portfolio return, or time-of-realization of settled P&L.

The finance+crypto filter is SELECTED AFTER OBSERVING past category performance.
This is an in-sample diagnostic, NOT a pre-registered or OOS strategy test.

Does not make network requests. Does not modify prior research files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal as D
import json
from pathlib import Path
import tempfile

ZERO = D('0')
WALLET = '0x777fae71d2ff9ec48a1213d48ba1d9d91024a1bb'
LABELS = (
    'source_prints_no_fees',
    'taker_fee_only',
    'taker_plus_0.01_per_side',
    'taker_plus_0.02_per_side',
)
FIN_CRYPTO = ('finance/economics', 'crypto')


def parse_d(x):
    return D(str(x))


def money(x):
    return f'${x:+,.2f}'


def moment(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc)


def time_group(ts):
    t = moment(ts)
    if t.year <= 2024:
        return '2022-2024'
    if t.year == 2025:
        return '2025'
    if t.year == 2026:
        return '2026-H1' if t.month <= 6 else '2026-H2'
    return str(t.year)


def read_source_earliest(cache):
    meta = json.loads((cache / 'albert1953_trades_meta.json').read_text())
    if (not meta.get('complete') or meta.get('taker_only') is not False or
            str(meta.get('wallet','')).lower() != WALLET):
        raise ValueError('V1 source history must be complete, maker+taker, and Albert1953')
    first, last = {}, {}
    count = 0
    with (cache / 'albert1953_trades.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            cid = str(d.get('condition_id','')).lower()
            ts = int(d['timestamp'])
            if not cid or ts < int(meta['start']) or ts >= int(meta['end_exclusive']):
                raise ValueError('Malformed/out-of-range V1 trade')
            first[cid] = min(ts, first.get(cid, ts))
            last[cid] = max(ts, last.get(cid, ts))
            count += 1
    if count != int(meta['rows']):
        raise ValueError('V1 source count differs from complete cache metadata')
    return first, last, meta


def read_v6(cache, ratio, scen):
    suffix = 'ratio' + format(ratio, 'f').replace('.', 'p')
    root = 'albert1953_execution_stress_v6_' + suffix
    rep = json.loads((cache / (root + '.json')).read_text())
    if D(str(rep['ratio'])) != ratio:
        raise ValueError('V6 report uses different proportional copy ratio')
    scenario = rep['scenarios'][scen]
    allrows = defaultdict(list)
    seen = set()
    with (cache / (root + '_markets.csv')).open(newline='',encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            lab = r['stress']
            if lab not in LABELS:
                continue
            cid = r['condition_id'].lower()
            key = (lab, cid)
            if key in seen:
                raise ValueError('Duplicate market/stress in V6 trace: '+str(key))
            seen.add(key)
            allrows[lab].append({
                'cid':cid,'category':r['category'], 'title':r['title'],
                'pnl':parse_d(r['copy_pnl']), 'spend':parse_d(r['copy_buy_spend']),
                'fees':parse_d(r['fees']), 'buy_events':int(r['buy_events']),
                'sell_events':int(r['sell_events']), 'price_caps':int(r['price_caps']),
                'v6_last90':r['last_90d'].strip().lower() == 'true',
                'v6_last_trade_ts':int(r['last_trade_ts']),
            })
    expect = set()
    for lab in LABELS:
        if lab not in scenario:
            raise ValueError('Missing V6 scenario '+lab)
        rows=allrows[lab]
        got = {r['cid'] for r in rows}
        if not expect:
            expect=got
        elif got != expect:
            raise ValueError(f'Inconsistent market population at {lab}: {len(got)} vs {len(expect)}')
        full=scenario[lab]['all']
        if len(rows) != int(full['markets']):
            raise ValueError(f'V6 market count mismatch at {lab}')
        if abs(sum((r['pnl'] for r in rows), ZERO) - D(full['net_pnl_usd'])) > D('.000001'):
            raise ValueError(f'V6 P&L mismatch at {lab}')
        if abs(sum((r['spend'] for r in rows), ZERO) - D(full['buy_spend_usd'])) > D('.000001'):
            raise ValueError(f'V6 buy spend mismatch at {lab}')
    return allrows, rep


def metrics(rows):
    pnl = sum((r['pnl'] for r in rows), ZERO)
    spend = sum((r['spend'] for r in rows), ZERO)
    fees = sum((r['fees'] for r in rows), ZERO)
    winners = sorted((r for r in rows if r['pnl']>0), key=lambda r:r['pnl'], reverse=True)
    top5 = sum((r['pnl'] for r in winners[:5]), ZERO)
    top10 = sum((r['pnl'] for r in winners[:10]), ZERO)
    return {
        'markets':len(rows),'profitable_markets':sum(r['pnl']>ZERO for r in rows),
        'buy_events':sum(r['buy_events'] for r in rows),
        'sell_events':sum(r['sell_events'] for r in rows),
        'gross_buy_spend_usd':str(spend),
        'net_pnl_usd':str(pnl),
        'roi_on_gross_buys_pct':str(100*pnl/spend) if spend else None,
        'fees_usd':str(fees),
        'top5_winners_pnl_usd':str(top5),
        'top10_winners_pnl_usd':str(top10),
        'ex_top5_winners_pnl_usd':str(pnl-top5),
        'ex_top10_winners_pnl_usd':str(pnl-top10),
        'price_capped_events':sum(r['price_caps'] for r in rows),
    }


def build_results(allrows, first, last, meta, ratio, scenario):
    date_cut = int(meta['end_exclusive']) - 90*86400
    for lab,rows in allrows.items():
        for r in rows:
            cid=r['cid']
            if cid not in first:
                raise ValueError('V6 market absent from source history: '+cid)
            r['first_ts'] = first[cid]
            r['last_ts'] = last[cid]
            if r['v6_last_trade_ts'] != last[cid]:
                raise ValueError('Last source trade mismatch for '+cid)
            if r['v6_last90'] != (last[cid]>=date_cut):
                raise ValueError('V6 90D flag inconsistent with V1 meta '+cid)
    result = {'wallet':WALLET, 'ratio':str(ratio), 'scenario':scenario,
              'observation_end_exclusive':int(meta['end_exclusive']),
              'last90_cutoff_epoch':date_cut,
              'method':('MARKET-level historical settlement P&L grouped by FIRST recorded trade date. '
                        'Not realized P&L in that quarter. Categories selected post hoc. '
                        'Prices are hypothetical source-print +/- adverse cents plus modeled fees; '
                        'no historical executable Kalshi quotes.'),
              'stress':{}}
    for lab, rows in allrows.items():
        fc = [r for r in rows if r['category'] in FIN_CRYPTO]
        result['stress'][lab] = {
            'all':metrics(rows),
            'finance':metrics([r for r in rows if r['category']=='finance/economics']),
            'crypto':metrics([r for r in rows if r['category']=='crypto']),
            'finance_crypto':metrics(fc),
            'finance_crypto_new_first_entry_90d':metrics([r for r in fc if r['first_ts']>=date_cut]),
            'finance_crypto_active_last_90d':metrics([r for r in fc if r['last_ts']>=date_cut]),
            'finance_crypto_first_entry_cohorts':{
                cohort:metrics([r for r in fc if time_group(r['first_ts'])==cohort])
                for cohort in sorted({time_group(r['first_ts']) for r in fc})},
        }
    return result


def test():
    def r(pnl,cat='finance/economics'):
        return {'pnl':D(str(pnl)),'spend':D(10),'fees':D('.1'),'buy_events':1,'sell_events':0,'price_caps':0,'category':cat}
    rows=[r(100),r(10),r(-15),r(-3),r(1),r(2),r(3)]
    m=metrics(rows)
    assert D(m['net_pnl_usd'])==D(98)
    assert D(m['ex_top5_winners_pnl_usd'])==D(-18)
    assert D(m['ex_top10_winners_pnl_usd'])==D(-18)
    assert m['markets']==7 and m['profitable_markets']==5
    assert time_group(1782000000)=='2026-H1' and time_group(1788000000)=='2026-H2'
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)
        # Smoke-test the file schema against a minimal synthetic market and report.
        root='albert1953_execution_stress_v6_ratio0p01'
        cols=['stress','condition_id','title','category','last_90d','last_trade_ts','copy_pnl','copy_buy_spend','sell_proceeds','settlement_payout','fees','buy_shares','sell_shares','buy_events','sell_events','price_caps']
        report={'ratio':'0.01','scenarios':{'whole_no_min':{}}}
        with (d/(root+'_markets.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
            for lab in LABELS:
                w.writerow(dict(stress=lab,condition_id='0xa',title='A',category='crypto',last_90d='True',last_trade_ts=103,copy_pnl='2',copy_buy_spend='10',sell_proceeds='0',settlement_payout='12',fees='0',buy_shares='20',sell_shares='0',buy_events=1,sell_events=0,price_caps=0))
                report['scenarios']['whole_no_min'][lab]={'all':{'markets':1,'net_pnl_usd':'2','buy_spend_usd':'10'}}
        (d/(root+'.json')).write_text(json.dumps(report))
        parsed,_=read_v6(d,D('.01'),'whole_no_min')
        assert all(len(v)==1 for v in parsed.values())
    print('PASS: winners exclusion, entry-date cohorts, V6 CSV/report reconciliation; no API/orders')


def print_one(name, m):
    pnl=D(m['net_pnl_usd']); spend=D(m['gross_buy_spend_usd']);
    r=(100*pnl/spend) if spend else ZERO
    print(f'{name:30s}  N={m["markets"]:5,d}  BUY=${spend:>11,.2f}  net={money(pnl):>12s}  ROI={r:+8.2f}%  ex5={money(D(m["ex_top5_winners_pnl_usd"])):>12s}  ex10={money(D(m["ex_top10_winners_pnl_usd"])):>12s}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--ratio',default='0.01',help='Proportional share multiplier, 0.01 means 1 percent')
    p.add_argument('--scenario',default='whole_no_min', choices=('whole_no_min',))
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        test();return
    ratio=D(a.ratio)
    if not ZERO<ratio<=D('1'):
        p.error('ratio must be within (0, 1]')
    first,last,meta=read_source_earliest(a.cache)
    allrows,ref=read_v6(a.cache,ratio,a.scenario)
    results=build_results(allrows,first,last,meta,ratio,a.scenario)
    print('ALBERT1953 V8 — FINANCE + CRYPTO ROBUSTNESS (IN-SAMPLE; READ ONLY)')
    print('Ratio:',ratio,'| markets with validated V6 records:',len(allrows['taker_plus_0.01_per_side']))
    print('Profit grouped by first MARKET entry, not the year of settlement. No actual Kalshi quotes.')
    for lab in LABELS:
        print('\n'+lab)
        d=results['stress'][lab]
        print_one('all categories',d['all'])
        print_one('finance only',d['finance'])
        print_one('crypto only',d['crypto'])
        print_one('FINANCE + CRYPTO',d['finance_crypto'])
    d=results['stress']['taker_plus_0.01_per_side']
    print('\nAT 1 CENT WORSE + TAKER FEES — FINANCE/CRYPTO BY FIRST TRADE DATE')
    for cohort,m in d['finance_crypto_first_entry_cohorts'].items():
        print_one(cohort,m)
    print('\nLAST-90-DAY CONTRAST (1c worse + taker)')
    print_one('new market entries in 90d',d['finance_crypto_new_first_entry_90d'])
    print_one('any market activity in 90d',d['finance_crypto_active_last_90d'])
    print('IMPORTANT: last-90d activity includes markets entered earlier; neither measure is a realized 90-day portfolio return.')
    print('Markets missing verified payouts are excluded. Post-hoc sector selection carries data-mining risk.')
    out=a.cache/('albert1953_finance_crypto_v8_ratio'+format(ratio,'f').replace('.','p')+'.json')
    out.write_text(json.dumps(results,indent=2))
    # Comprehensive candidate universe; intentionally sorted by FIRST entry time,
    # not past P&L, to avoid cherry-picking profitable Kalshi match attempts.
    core = {r['cid']:r for r in allrows['taker_plus_0.01_per_side']
            if r['category'] in FIN_CRYPTO}
    by_stress = {lab:{r['cid']:r for r in rs} for lab,rs in allrows.items()}
    universe=a.cache/('albert1953_finance_crypto_v8_ratio'+format(ratio,'f').replace('.','p')+'_universe.csv')
    cols=('condition_id','first_trade_utc','last_trade_utc','category','title','buy_spend_1c_usd',
          'pnl_taker_fee_only_usd','pnl_1c_usd','pnl_2c_usd','fees_1c_usd','buy_events_1c','sell_events_1c')
    with universe.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for cid,r in sorted(core.items(), key=lambda item:(item[1]['first_ts'],item[0])):
            w.writerow({
                'condition_id':cid,
                'first_trade_utc':moment(r['first_ts']).isoformat(),
                'last_trade_utc':moment(r['last_ts']).isoformat(),
                'category':r['category'], 'title':r['title'],
                'buy_spend_1c_usd':r['spend'],
                'pnl_taker_fee_only_usd':by_stress['taker_fee_only'][cid]['pnl'],
                'pnl_1c_usd':r['pnl'],
                'pnl_2c_usd':by_stress['taker_plus_0.02_per_side'][cid]['pnl'],
                'fees_1c_usd':r['fees'], 'buy_events_1c':r['buy_events'],
                'sell_events_1c':r['sell_events']})
    print('REPORT:',out)
    print('FULL FINANCE/CRYPTO UNIVERSE:',universe)
    print('Do not filter to past winners before matching Kalshi markets.')
    print('OFFLINE. NO EXCHANGE CALLS OR ORDERS. NOT AN EXECUTABLE COPY BACKTEST.')


if __name__=='__main__':
    main()
