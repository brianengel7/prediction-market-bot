#!/usr/bin/env python3
"""Albert1953 V11 — OFFLINE Kalshi discovery coverage / economic selection audit.

Uses the user's V8 complete finance+crypto discovery universe (1,739 verified-settled
plus additional unverified markets), V10 candidate list and V6 market-level stress
trace.  Uses 0.01 copy ratio and V6 whole-contract 1c-adverse/taker-cost scenario.

No API, no orders.  DOES NOT determine Kalshi equivalence, fill feasibility,
market-specific fees, available historical quotes or executable follower P&L.
Even the strongest ranked candidate is UNVERIFIED until resolution criteria,
underlying reference prices, strike operators, timing, side mapping and contract
availability are verified independently.  Source-printed P&L is only a way to
measure the selection bias of a DISCOVERY LIST, not Kalshi return.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import csv
import json
from pathlib import Path
import re
import tempfile

STRESS = 'taker_plus_0.01_per_side'
SOURCE_SCENARIO = 'whole_no_min'
EXPECTED_RATIO = Decimal('0.01')
ZERO = Decimal('0')
SECTORS = {'finance/economics','crypto'}
STRONG_FLAGS = {'TEXT_STRIKE_MATCH','YEAR_MATCH','LISTING_TIME_COMPATIBLE','FAMILY_MATCH'}
BLOCK_FLAGS = {'STRIKE_UNVERIFIED','YEAR_UNVERIFIED','OPEN_AT_SIGNAL_NOT_VERIFIED','FAMILY_REVIEW_REQUIRED',
               'EXPIRY_TERMS_UNVERIFIED','SETTLEMENT_SOURCE_UNVERIFIED','SIDE_MAPPING_UNVERIFIED'}


def D(x):
    try: return Decimal(str(x))
    except (InvalidOperation, TypeError) as e: raise ValueError(f'Invalid decimal {x!r}') from e


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as fh: return list(csv.DictReader(fh))


def bool_field(v):return str(v).strip().lower() in ('true','1','yes')


def stamp(s):
    try:
        t=datetime.fromisoformat(str(s).replace('Z','+00:00'))
        if t.tzinfo is None:raise ValueError('timezone missing')
        return t.astimezone(timezone.utc)
    except Exception as e:raise ValueError(f'Bad UTC timestamp {s!r}') from e


def source_family(title):
    s=str(title).lower()
    if re.search(r'\b(dip|fall to|fall below|how low|lowest|minimum|touch a low)\b',s):return 'touch_low'
    if re.search(r'\b(hit|reach|touch|how high|highest|maximum|all.time high)\b',s):return 'touch_high'
    if re.search(r'\b(close|closing|settle|settles|final trading day|end.of.day)\b',s):return 'closing'
    if 'market cap' in s or 'largest company' in s:return 'market_cap/ranking'
    if re.search(r'\b(ipo|ceo|board|acquisition|merger)\b',s):return 'corporate'
    if re.search(r'\b(between|range|by at least|by less than)\b',s):return 'range'
    return 'other/unspecified'


def underlying(title):
    s=str(title).lower()
    tokens=[
        ('BTC',r'\bbitcoin\b|\bbtc\b'),('ETH',r'\bethereum\b|\beth\b'),
        ('SOL',r'\bsolana\b|\bsol\b'),('Silver',r'\bsilver\b|\bsi\)'),
        ('Gold',r'\bgold\b'),('Oil',r'\boil\b|\bcrude\b|\bwti\b'),
        ('MSFT',r'\bmicrosoft\b|\bmsft\b'),('AAPL',r'\bapple\b|\baapl\b'),
        ('NVDA',r'\bnvidia\b|\bnvda\b'),('META',r'\bmeta\b|\bfacebook\b'),
        ('AMZN',r'\bamazon\b|\bamzn\b'),('GOOG',r'\balphabet\b|\bgoogle\b|\bgoogl\b'),
        ('S&P500',r'\bs&p\b|\bs&p 500\b|\bspx\b'),('DOGE',r'\bdoge\b|\bdogecoin\b'),
        ('XRP',r'\bxrp\b|\bripple\b')]
    found=[name for name,pat in tokens if re.search(pat,s)]
    return '+'.join(found[:3]) if found else 'Unclassified'


def metrics(rows):
    rows=list(rows)
    pnl=sum((r['pnl'] for r in rows),ZERO)
    spend=sum((r['spend'] for r in rows),ZERO)
    fees=sum((r['fees'] for r in rows),ZERO)
    positive=sorted((r['pnl'] for r in rows if r['pnl']>0),reverse=True)
    return {'markets':len(rows),'buy_spend_usd':str(spend),'modeled_net_pnl_usd':str(pnl),
            'roi_on_buy_spend_pct':str(100*pnl/spend) if spend else None,
            'fee_usd':str(fees),
            'ex_top5_winners_usd':str(pnl-sum(positive[:5],ZERO)),
            'ex_top10_winners_usd':str(pnl-sum(positive[:10],ZERO)),
            'buy_events':sum(r['buy_events'] for r in rows),
            'sell_events':sum(r['sell_events'] for r in rows)}


def validate_v6(cache):
    prefix='albert1953_execution_stress_v6_ratio0p01'
    report=json.loads((cache/(prefix+'.json')).read_text(encoding='utf-8'))
    if D(report['ratio'])!=EXPECTED_RATIO:raise ValueError('V6 ratio mismatch; require 0.01')
    lab=report['scenarios'][SOURCE_SCENARIO][STRESS]['all']
    rows={}
    for r in read_csv(cache/(prefix+'_markets.csv')):
        if r['stress']!=STRESS:continue
        cid=r['condition_id'].lower()
        if cid in rows:raise ValueError('Duplicated V6 condition id '+cid)
        rows[cid]={'cid':cid,'category':r['category'],'pnl':D(r['copy_pnl']),
                   'spend':D(r['copy_buy_spend']),'fees':D(r['fees']),
                   'buy_events':int(r['buy_events']),'sell_events':int(r['sell_events'])}
    if len(rows)!=int(lab['markets']):raise ValueError('V6 market total mismatch')
    if abs(sum((r['pnl'] for r in rows.values()),ZERO)-D(lab['net_pnl_usd']))>Decimal('.000001'):
        raise ValueError('V6 P&L reconciliation failed')
    if abs(sum((r['spend'] for r in rows.values()),ZERO)-D(lab['buy_spend_usd']))>Decimal('.000001'):
        raise ValueError('V6 spend reconciliation failed')
    return rows


def load_data(cache):
    universe=read_csv(cache/'albert1953_finance_crypto_v8_ratio0p01_universe.csv')
    v8={}
    for r in universe:
        cid=r['condition_id'].lower()
        if cid in v8:raise ValueError('duplicate V8 market')
        if r['category'] not in SECTORS:raise ValueError('Non finance/crypto V8 source')
        stamp(r['first_trade_utc'])
        v8[cid]=r
    cand=read_csv(cache/'albert1953_v10_candidate_matches.csv')
    by_source=defaultdict(list)
    for r in cand:
        cid=r['condition_id'].lower()
        if r['category'] not in SECTORS:raise ValueError('unexpected V10 category')
        by_source[cid].append(r)
    if set(v8)-set(by_source):raise ValueError('V10 missing some V8 condition IDs')
    summary=json.loads((cache/'albert1953_v10_candidate_coverage.json').read_text())
    if int(summary['source_universe'])!=len(by_source):raise ValueError('V10 summary universe count differs from rows')
    if int(summary['v8_settled'])!=len(v8):raise ValueError('V10/V8 settled count mismatch')
    if int(summary['semantically_plausible_markets'])!=sum(any(x['kalshi_ticker'] for x in xs) for xs in by_source.values()):
        raise ValueError('V10 candidate count mismatch')
    v6=validate_v6(cache)
    for cid in v8:
        if cid not in v6:raise ValueError('Settled V8 market missing V6 trace: '+cid)
    return v8,by_source,v6,summary


def audit(cache):
    v8,candidates,v6,v10=load_data(cache)
    full=[]; extra=[]; review=[]
    ticker_sources=defaultdict(set)
    for cid,rows in candidates.items():
        titles={r['poly_title'] for r in rows}
        if len(titles)>1:raise ValueError('V10 inconsistent source title '+cid)
        src=rows[0]
        viable=[r for r in rows if r['kalshi_ticker'] and r['decision']=='UNVERIFIED_REVIEW_CANDIDATE']
        viable.sort(key=lambda x:(-float(x['score'] or 0),x['kalshi_ticker']))
        for row in viable: ticker_sources[row['kalshi_ticker']].add(cid)
        strict=[]
        for x in viable:
            f=set(x['review_flags'].split('|'))
            if STRONG_FLAGS.issubset(f):strict.append(x)
        if viable: state='STRONG_REVIEW_UNVERIFIED' if strict else 'WEAK_REVIEW_UNVERIFIED'
        else:state='NO_CANDIDATE_IN_SCANNED_SERIES'
        base={'condition_id':cid,'category':src['category'],'poly_title':src['poly_title'],
              'first_trade_utc':src['first_trade_utc'],'underlying':underlying(src['poly_title']),
              'source_payoff_family':source_family(src['poly_title']),'discovery_state':state,
              'review_candidates':len(viable),'strong_review_candidates':len(strict),
              'top_kalshi_ticker':viable[0]['kalshi_ticker'] if viable else '',
              'top_kalshi_title':viable[0]['kalshi_title'] if viable else '',
              'top_review_flags':viable[0]['review_flags'] if viable else '',
              'rule_verified':'false','kalshi_quote_verified':'false'}
        if cid in v6:
            base.update({'source_settled':'true',
                        'source_copy_buy_spend_usd':str(v6[cid]['spend']),
                        'source_price_stressed_pnl_usd':str(v6[cid]['pnl']),
                        'source_price_stressed_fee_usd':str(v6[cid]['fees']),
                        'source_copy_buy_events':v6[cid]['buy_events'],
                        'source_copy_sell_events':v6[cid]['sell_events']})
            full.append({**base,**v6[cid]})
        else:
            base.update({'source_settled':'false','source_copy_buy_spend_usd':'',
                         'source_price_stressed_pnl_usd':'','source_price_stressed_fee_usd':'',
                         'source_copy_buy_events':'','source_copy_sell_events':''})
            extra.append(base)
        review.append(base)
    if len(full)!=len(v8):raise ValueError('Unexpected settled join count')
    settled_metrics={g:metrics(r for r in full if r['discovery_state']==g)
                     for g in ('STRONG_REVIEW_UNVERIFIED','WEAK_REVIEW_UNVERIFIED','NO_CANDIDATE_IN_SCANNED_SERIES')}
    candidate=[r for r in full if r['review_candidates']]
    totals=metrics(full)
    spend=D(totals['buy_spend_usd'])
    candidate_spend=sum((r['spend'] for r in candidate),ZERO)
    source_year=defaultdict(list)
    for r in full:
        source_year[r['first_trade_utc'][:4]].append(r)
    source_underlying=defaultdict(list)
    for r in full:source_underlying[(r['underlying'],r['source_payoff_family'])].append(r)
    families=[]
    for (asset,fam),rs in source_underlying.items():
        m=metrics(rs)
        cov=sum(r['review_candidates']>0 for r in rs)
        cov_spend=sum((r['spend'] for r in rs if r['review_candidates']),ZERO)
        families.append({'underlying':asset,'source_family':fam,'total_markets':len(rs),
                         'candidate_markets':cov,'total_buy_spend_usd':m['buy_spend_usd'],
                         'candidate_buy_spend_usd':str(cov_spend),
                         'source_price_stressed_pnl_usd':m['modeled_net_pnl_usd']})
    families.sort(key=lambda x:(-D(x['total_buy_spend_usd']),-x['total_markets']))
    def remove_internal(x):return {k:v for k,v in x.items() if k not in ('pnl','spend','fees','buy_events','sell_events','cid')}
    review.sort(key=lambda r:(-D(r['source_copy_buy_spend_usd']) if r['source_copy_buy_spend_usd'] else ZERO,r['condition_id']))
    out={'wallet':'0x777fae71d2ff9ec48a1213d48ba1d9d91024a1bb','copy_ratio':'0.01',
         'stress':STRESS,'source_scenario':SOURCE_SCENARIO,
         'source_universe_total':len(candidates),'settled_v8':len(full),'additional_unverified_or_excluded':len(extra),
         'kalshi_catalog_rows':v10['catalog_markets'],'kalshi_selected_series':v10['selected_series'],
         'selected_series_page_sets_complete':v10['scans_complete'],
         'kalshi_equivalents_verified':0,
         'distinct_candidate_kalshi_tickers':len(ticker_sources),
         'kalshi_tickers_reused_by_multiple_poly_markets':sum(len(ids)>1 for ids in ticker_sources.values()),
         'top_reused_kalshi_tickers':[{'ticker':ticker,'source_markets':len(cids)} for ticker,cids in sorted(ticker_sources.items(),key=lambda x:(-len(x[1]),x[0]))[:15]],
         'discovery_state_counts':dict(Counter(r['discovery_state'] for r in review)),
         'settled_all':totals,'settled_with_any_candidate':metrics(candidate),
         'settled_without_candidate':metrics(r for r in full if not r['review_candidates']),
         'settled_by_review_strength':settled_metrics,
         'candidate_coverage_of_settled_buy_spend_pct':str(100*candidate_spend/spend) if spend else None,
         'settled_by_first_entry_year':{k:{'all':metrics(rs),'candidates_only':metrics(r for r in rs if r['review_candidates'])} for k,rs in sorted(source_year.items())},
         'extra_unverified_states':dict(Counter(r['discovery_state'] for r in extra)),
         'source_family_priority':families,
         'warnings':['Candidate-subset historical P&L uses Polymarket source prints at hypothetical Kalshi costs; NOT Kalshi P&L',
                     'Every candidate unverified: strikes, exact deadlines, oracle/source, settlement rule and side mapping missing',
                     'Series pages complete is NOT full-exchange catalog coverage',
                     'Unverified/uncounted source markets never assigned zero P&L',
                     'Ranking by source buy-spend avoids choosing only past winners, but inspecting results still introduces selection risk']}
    outcsv=cache/'albert1953_v11_economic_candidate_review.csv'
    cols=['condition_id','category','poly_title','first_trade_utc','underlying','source_payoff_family',
          'discovery_state','review_candidates','strong_review_candidates','top_kalshi_ticker','top_kalshi_title',
          'top_review_flags','source_settled','source_copy_buy_spend_usd','source_price_stressed_pnl_usd',
          'source_price_stressed_fee_usd','source_copy_buy_events','source_copy_sell_events','rule_verified','kalshi_quote_verified']
    with outcsv.open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(review)
    outjson=cache/'albert1953_v11_coverage_selection_risk.json'
    outjson.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out,review,outcsv,outjson


def money(val):return f'${D(val):+,.2f}'


def printable(out,review,outcsv,outjson):
    print('ALBERT1953 V11 — CANDIDATE ECONOMICS AND DISCOVERY COVERAGE; NOT KALSHI P&L')
    print(f"Universe={out['source_universe_total']:,} source markets: {out['settled_v8']:,} settled, {out['additional_unverified_or_excluded']:,} additional")
    print(f"Kalshi catalog={out['kalshi_catalog_rows']:,} rows in {out['kalshi_selected_series']} selected series; selected page sets complete={out['selected_series_page_sets_complete']}")
    print('Candidate review counts:',out['discovery_state_counts'])
    print('Distinct Kalshi tickers among candidates:',out['distinct_candidate_kalshi_tickers'],
          '| reused by multiple source markets:',out['kalshi_tickers_reused_by_multiple_poly_markets'])
    print('Verified equivalent markets: ZERO')
    print('\nSource-printed 1c adverse/side + hypothetical taker fees, only SETTLED source markets:')
    for name,m in [('ALL finance+crypto',out['settled_all']),
                   ('ANY semantic candidate',out['settled_with_any_candidate']),
                   ('NO candidate',out['settled_without_candidate']),
                   ('STRONG review only (not verified)',out['settled_by_review_strength']['STRONG_REVIEW_UNVERIFIED'])]:
        roi=m['roi_on_buy_spend_pct']
        print(f'  {name:34} N={m["markets"]:4,d} BUY={money(m["buy_spend_usd"]):>11} '
              f'P&L={money(m["modeled_net_pnl_usd"]):>11} '
              f'ROI={float(roi):+7.2f}%' if roi is not None else f'  {name}: no BUY spend')
        if roi is not None:print(f'     ex-top5={money(m["ex_top5_winners_usd"])}  ex-top10={money(m["ex_top10_winners_usd"])}')
    print('Candidate market BUY spend coverage (settled source only):',
          round(float(out['candidate_coverage_of_settled_buy_spend_pct'] or 0),3),'%')
    print('\nHighest-spend source families (NOT prioritized by profit):')
    for x in out['source_family_priority'][:15]:
        print(f"  {x['underlying'][:13]:13} {x['source_family'][:19]:19} markets {x['total_markets']:4d} "
              f"candidates {x['candidate_markets']:4d} BUY={money(x['total_buy_spend_usd'])}")
    print('\nTop BUY-spend candidate review queue (all UNVERIFIED):')
    found=0
    for r in review:
        if not r['top_kalshi_ticker'] or r['source_settled']!='true':continue
        print(f"  BUY={money(r['source_copy_buy_spend_usd']):>10} {r['discovery_state']:23} {r['top_kalshi_ticker'][:32]:32} {r['poly_title'][:76]}")
        found+=1
        if found>=12:break
    print('\nOUTPUT CSV:',outcsv,'\nOUTPUT JSON:',outjson)
    print('NO ORDERS. NO EXECUTABLE KALSHI HISTORICAL QUOTES. NO VERIFIED SETTLEMENT-EQUIVALENT MATCHES.')


def selftest():
    assert source_family('Will Ethereum dip to $1,500 by December 31?')=='touch_low'
    assert source_family('Will Microsoft close above $410?')=='closing'
    assert underlying('Will silver hit $90?')=='Silver'
    assert D(metrics([{'pnl':D(5),'spend':D(20),'fees':D(1),'buy_events':1,'sell_events':0},
                      {'pnl':D(-3),'spend':D(10),'fees':D(2),'buy_events':1,'sell_events':1}])['modeled_net_pnl_usd'])==D(2)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        src=[{'condition_id':'0xa','category':'crypto','title':'Will Ethereum dip to $1,500 by December 31, 2026?','first_trade_utc':'2026-10-10T00:00:00+00:00'},
             {'condition_id':'0xb','category':'finance/economics','title':'Will Microsoft close above $410 on February 9, 2026?','first_trade_utc':'2026-01-10T00:00:00+00:00'}]
        def csvwrite(name,rows):
            with (p/name).open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        csvwrite('albert1953_finance_crypto_v8_ratio0p01_universe.csv',src)
        flag='TEXT_STRIKE_MATCH|YEAR_MATCH|LISTING_TIME_COMPATIBLE|FAMILY_MATCH|EXPIRY_TERMS_UNVERIFIED|SETTLEMENT_SOURCE_UNVERIFIED|SIDE_MAPPING_UNVERIFIED'
        match=[dict(condition_id='0xa',category='crypto',poly_title=src[0]['title'],first_trade_utc=src[0]['first_trade_utc'],kalshi_ticker='KXETHMINY-X',kalshi_title='Ethereum dip',score=100,review_flags=flag,source_settled_in_v8='true',decision='UNVERIFIED_REVIEW_CANDIDATE'),
               dict(condition_id='0xb',category='finance/economics',poly_title=src[1]['title'],first_trade_utc=src[1]['first_trade_utc'],kalshi_ticker='',kalshi_title='',score='',review_flags='NO_SEMANTIC_CANDIDATE_IN_SCANNED_CATALOG',source_settled_in_v8='true',decision='NOT_FOUND_IN_THIS_SCAN')]
        csvwrite('albert1953_v10_candidate_matches.csv',match)
        (p/'albert1953_v10_candidate_coverage.json').write_text(json.dumps({'source_universe':2,'v8_settled':2,'semantically_plausible_markets':1,'catalog_markets':10,'selected_series':3,'scans_complete':True}))
        traces=[dict(stress=STRESS,condition_id='0xa',title='ETH dip',category='crypto',last_90d='True',last_trade_ts=12,copy_pnl='3',copy_buy_spend='20',sell_proceeds='0',settlement_payout='23',fees='1',buy_shares=10,sell_shares=0,buy_events=1,sell_events=0,price_caps=0),
                dict(stress=STRESS,condition_id='0xb',title='MSFT close',category='finance/economics',last_90d='False',last_trade_ts=8,copy_pnl='-1',copy_buy_spend='10',sell_proceeds='0',settlement_payout='9',fees='1',buy_shares=5,sell_shares=0,buy_events=1,sell_events=0,price_caps=0)]
        csvwrite('albert1953_execution_stress_v6_ratio0p01_markets.csv',traces)
        (p/'albert1953_execution_stress_v6_ratio0p01.json').write_text(json.dumps({'ratio':'0.01','scenarios':{'whole_no_min':{STRESS:{'all':{'markets':2,'net_pnl_usd':'2','buy_spend_usd':'30'}}}}}))
        o,rr,_,_=audit(p)
        assert o['settled_with_any_candidate']['markets']==1 and D(o['settled_with_any_candidate']['modeled_net_pnl_usd'])==D(3)
        assert o['settled_without_candidate']['markets']==1
        assert o['candidate_coverage_of_settled_buy_spend_pct']==str(D('200')/D(3))
        assert o['discovery_state_counts']['STRONG_REVIEW_UNVERIFIED']==1
        assert o['distinct_candidate_kalshi_tickers']==1
    print('PASS: joins, exact ratio, V6 reconciliation, candidate bookkeeping, top-winner metrics, no APIs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:return selftest()
    out,rows,csvpath,jsonp=audit(a.cache)
    printable(out,rows,csvpath,jsonp)

if __name__=='__main__':main()
