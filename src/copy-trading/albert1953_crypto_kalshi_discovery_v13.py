#!/usr/bin/env python3
"""Albert1953 V13 — crypto-ONLY, coverage-first Kalshi market discovery.

Requires alongside: albert1953_kalshi_match_discovery_v9_1.py (uses its
read-only Kalshi GET client, complete Polymarket cache checks, and pagination).
No account keys, orders, or PnL extrapolation from unverified market matches.

Input: complete V1 raw trades + V8 settled finance/crypto universe.
Universe: ALL crypto V8 markets plus additional crypto markets from the raw
trade history; settled/unsettled separated.  The V8 1c stress economics are
source-print simulations, NOT Kalshi executable economics.

Match restrictions: same single underlying + compatible payout family + matching
observed threshold when present + historically listed by first copy signal.
Never calls any match 'verified': Kalshi/Polymarket resolution source, exact
observation times, settlement price averaging, and side mapping still require
manual rule comparison.  Excluded/unmatched outcomes remain in denominator.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import re
import sys

DEP=Path(__file__).with_name('albert1953_kalshi_match_discovery_v9_1.py')
if not DEP.is_file():
    raise SystemExit(f'Missing read-only V9.1 dependency beside V13: {DEP}')
spec=importlib.util.spec_from_file_location('albert1953_v9_1_helper',DEP)
v9=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v9;spec.loader.exec_module(v9)
v9.CATEGORIES=('Crypto','Financials','Economics')

ASSET_RE={
 'BTC':re.compile(r'(?<!\w)(?:bitcoin|btc)(?!\w)|(?<!\w)(?:KX)?BTC[A-Z0-9]*',re.I),
 'ETH':re.compile(r'(?<!\w)(?:ethereum|ether|eth)(?!\w)|(?<!\w)(?:KX)?ETH[A-Z0-9]*',re.I),
 'SOL':re.compile(r'(?<!\w)(?:solana|sol)(?!\w)|(?<!\w)(?:KX)?SOL[A-Z0-9]*',re.I),
 'XRP':re.compile(r'(?<!\w)(?:xrp|ripple)(?!\w)|(?<!\w)(?:KX)?XRP[A-Z0-9]*',re.I),
 'DOGE':re.compile(r'(?<!\w)(?:dogecoin|doge)(?!\w)|(?<!\w)(?:KX)?DOGE[A-Z0-9]*',re.I),
}
SHORT=re.compile(r'\b(?:5|15|30)[ -]?(?:m|min|minutes)\b|\b(?:hourly|minute|every hour|five minutes|fifteen minutes)\b',re.I)
OTHER_EVENT=re.compile(r'\b(etf|etps?|approval|halving|fund|ipo|launch|partnership|wallet|satoshi|staked|staking|market cap|capitalization|fdv|bill|congress|regulation|exchange listing|lawsuit|reserve|dominance)\b',re.I)
COMPARE=re.compile(r'\b(vs\.?|versus|outperform|first to|first one|which first|which will|before (?:bitcoin|ethereum|btc|eth)|either)\b',re.I)
TOUCH_LOW=re.compile(r'\b(dip(?:s|ped)?|fall(?:s)? (?:to|below)|drop(?:s)? (?:to|below)|touch(?:es)? (?:a )?low|how low|lowest|minimum|new low)\b',re.I)
TOUCH_HIGH=re.compile(r'\b(hit(?:s)?|reach(?:es)?|touch(?:es)?|exceed(?:s)?|surpass(?:es)?|how high|highest|maximum|new high|all.time high)\b',re.I)
CLOSE=re.compile(r'\b(close(?:s|d)?|closing|settle(?:s|d)?|finish(?:es)?|end(?:s)? (?:above|below|over|under)|ending (?:above|below)|end.of.(?:day|week|month|year))\b',re.I)
RANGE=re.compile(r'\b(between|range|from \$[\d,.]+ to \$[\d,.]+)\b',re.I)
LOW_SER=re.compile(r'\b(how low|lowest|minimum|min price|one.touch low|monthly low|yearly low|below)\b',re.I)
HIGH_SER=re.compile(r'\b(how high|highest|maximum|max price|one.touch high|monthly high|yearly high|above)\b',re.I)
YEAR=re.compile(r'\b20(?:2\d|3\d)\b')
MONTHS={'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
        'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}


def assets(s):
    return {a for a,rx in ASSET_RE.items() if rx.search(str(s))}


def family(title,ticker='',series=False):
    s=str(title);t=str(ticker).upper()
    if OTHER_EVENT.search(s) or COMPARE.search(s):return 'OTHER'
    # Never mix $2000 OR $4000 FIRST into an ordinary one-touch contract.
    if re.search(r'\b(or|first)\b',s,re.I) and len(v9.strikes(s))>=2:return 'OTHER'
    if series:
        # Numeric and ticker-specific classification where title is abbreviated.
        if re.search(r'(?:^|[A-Z])(?:BTC|ETH|SOL|DOGE|XRP)MIN',t) or re.search(r'\bhow low\b',s,re.I):return 'TOUCH_LOW'
        if re.search(r'(?:^|[A-Z])(?:BTC|ETH|SOL|DOGE|XRP)MAX',t) or re.search(r'\bhow high\b',s,re.I):return 'TOUCH_HIGH'
    if CLOSE.search(s):return 'CLOSE'
    if re.search(r'\b(?:above|below|over|under)\b',s,re.I) and (re.search(r'\bon\s+(?:'+ '|'.join(MONTHS) + r')\s+\d{1,2}\b',s,re.I) or re.search(r'\bat\s+(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|midnight|noon)\b',s,re.I)):
        return 'CLOSE'
    if TOUCH_LOW.search(s):return 'TOUCH_LOW'
    if TOUCH_HIGH.search(s):return 'TOUCH_HIGH'
    if series:
        if LOW_SER.search(s):return 'REVIEW_LOW'
        if HIGH_SER.search(s):return 'REVIEW_HIGH'
    if RANGE.search(s):return 'RANGE'
    return 'OTHER'


def horizon(s):
    s=str(s).lower()
    if re.search(r'\b(?:december\s+31|end\s+of\s+(?:the\s+)?year)\b',s):return 'YEAR'
    if SHORT.search(s):return 'INTRADAY'
    if re.search(r'\b(hour|today|tomorrow|daily|tonight)\b',s):return 'INTRADAY'
    if re.search(r'\b(week|weekly)\b',s):return 'WEEK'
    if any(re.search(r'\b'+m+r'\b',s) for m in MONTHS) or re.search(r'\b(month|monthly|mon)\b',s):return 'MONTH'
    if re.search(r'\b(year|annual|year.end|end of (?:20\d\d)|december 31)\b',s):return 'YEAR'
    return 'UNKNOWN'


def series_family(title,ticker):
    a=assets(title+' '+ticker)
    if len(a)!=1:return 'OTHER'
    if OTHER_EVENT.search(title) or COMPARE.search(title):return 'OTHER'
    f=family(title,ticker,series=True)
    if f=='OTHER':
        tu=ticker.upper()
        if re.search(r'\b(?:MINMON|MINY|MINM)\b',tu):return 'TOUCH_LOW'
        if re.search(r'\b(?:MAXMON|MAXY|MAXM)\b',tu):return 'TOUCH_HIGH'
    return f


def source_rows(cache):
    f=cache/'albert1953_finance_crypto_v8_ratio0p01_universe.csv'
    universe=v9.load_universe(f)
    seen={r['condition_id'].lower() for r in universe}
    addition=v9.extra_raw_unsettled(cache,seen)
    all_rows=universe+addition
    if len({r['condition_id'].lower() for r in all_rows})!=len(all_rows):
        raise RuntimeError('Duplicate source conditions')
    rows=[]
    for r in all_rows:
        if r['category']!='crypto':continue
        x=dict(r)
        x['_asset']=next(iter(assets(r['title']))) if len(assets(r['title']))==1 else ''
        x['_family']=family(r['title'])
        x['_horizon']=horizon(r['title'])
        x['_settled']=r in universe
        x['_buy1c']=Decimal(str(r.get('buy_spend_1c_usd') or '0')) if x['_settled'] else None
        x['_pnl1c']=Decimal(str(r.get('pnl_1c_usd') or '0')) if x['_settled'] else None
        rows.append(x)
    return rows,Counter(x['_settled'] for x in rows)


def select_series(series,rows):
    src_by_asset=defaultdict(list)
    for r in rows:
        if r['_asset'] and r['_family']!='OTHER':src_by_asset[r['_asset']].append(r)
    out=[]
    for ticker,s in series.items():
        title=str(s.get('title') or '')
        aa=assets(ticker+' '+title)
        if len(aa)!=1:continue
        a=next(iter(aa));f=series_family(title,ticker)
        if f=='OTHER' or a not in src_by_asset:continue
        h=horizon(title)
        # Exclude micro-duration series when source trades are NOT micro-duration.
        # This prevents 15m listings drowning out event-duration markets.
        relevant=[r for r in src_by_asset[a] if (r['_family']==f or (r['_family']=='TOUCH_LOW' and f=='REVIEW_LOW') or (r['_family']=='TOUCH_HIGH' and f=='REVIEW_HIGH'))
                  and (h!='INTRADAY' or r['_horizon']=='INTRADAY')]
        if not relevant:continue
        score=(sum(float(r['_buy1c'] or 0) for r in relevant) + len(relevant)*0.05)
        if f.startswith('REVIEW_'):score*=0.45
        if h=='UNKNOWN':score*=.75
        out.append({'series_ticker':ticker,'series_title':title,'asset':a,'family':f,'horizon':h,
                    'source_market_mentions':len(relevant),'source_settled_buy_spend_usd':round(sum(float(r['_buy1c'] or 0) for r in relevant),2),
                    'score':round(score,3),'category':s.get('category','')})
    out.sort(key=lambda r:(-r['score'],r['asset'],r['series_ticker']))
    return out


def kalshi_market_strikes(m):
    """Extract only explicit money price strikes. Do not decode ticker internals."""
    text=' '.join(str(m.get(k) or '') for k in ('title','subtitle','yes_sub_title','no_sub_title'))
    ks=v9.strikes(text)
    # floor/cap are structured strike prices; inspect only if a single bound.
    structured=[]
    for key in ('floor_strike','cap_strike'):
        val=m.get(key)
        try:
            if val is not None and str(val).strip()!='' and float(val)>0:structured.append(float(val))
        except (ValueError,TypeError):pass
    return sorted(set(ks+structured))


def year_set(s):return {int(x) for x in YEAR.findall(str(s))}


def candidate(src,m,series_title=''):
    source=src['title']
    a=src['_asset'];sf=src['_family'];s_h=src['_horizon']
    if not a or sf not in ('TOUCH_LOW','TOUCH_HIGH','CLOSE','RANGE'):return None
    ticker=str(m.get('ticker') or '')
    kt=' '.join(str(m.get(x) or '') for x in ('title','subtitle','yes_sub_title','no_sub_title'))
    # Market-level asset is often omitted: use series title and ticker as well.
    aa=assets(kt+' '+series_title+' '+ticker)
    if aa!={a}:return None
    f=series_family(series_title,ticker)
    if f not in (sf,'REVIEW_LOW' if sf=='TOUCH_LOW' else 'REVIEW_HIGH' if sf=='TOUCH_HIGH' else 'NOT_ALLOWED'):return None
    mf=family(kt)
    if mf!='OTHER' and mf!=sf:return None
    if COMPARE.search(kt) or OTHER_EVENT.search(kt):return None
    kh=horizon(series_title+' '+str(m.get('title') or ''))
    if s_h=='YEAR' and kh in ('INTRADAY','WEEK','MONTH'):return None
    if s_h=='MONTH' and kh in ('INTRADAY','WEEK'):return None
    if s_h in ('MONTH','YEAR') and SHORT.search(kt+' '+series_title):return None
    sk=v9.strikes(source)
    mk=kalshi_market_strikes(m)
    if len(sk)!=1:return None  # multi-strike, no-strike source requires manual review separately
    if len(mk)>1:
        # market titles may contain $-figures other than the actual contract strike;
        # with multiple different strikes, require explicit rules review, not auto-pair.
        return None
    if not mk:return None
    ss=sk[0];kk=mk[0]
    # Exact nominal thresholds only: no fuzzy percent tolerances or ticker decoding.
    if abs(ss-kk)>max(0.015,abs(ss)*0.00000001):return None
    src_t=v9.parse_iso(src['first_trade_utc'])
    op=v9.parse_iso(m.get('open_time') or m.get('created_time'))
    close=v9.parse_iso(m.get('close_time'))
    if src_t is not None and close is not None and src_t>=close:return None
    if src_t is not None and op is not None and op>src_t:return None
    source_years=year_set(source)
    if source_years and close is not None:
        endyear=datetime.fromtimestamp(close,timezone.utc).year
        if all(abs(yy-endyear)>1 for yy in source_years):return None
    flags=['SAME_CRYPTO_ASSET','SAME_PRICE_THRESHOLD','COMPATIBLE_PAYOFF_FAMILY',
           'MARKET_OPEN_AT_FIRST_TRADE' if op is not None and src_t is not None else 'LISTING_TIME_UNVERIFIED',
           'EXACT_DEADLINE_UNVERIFIED','PRICE_INDEX_AND_AVERAGING_UNVERIFIED',
           'SIDE_MAPPING_UNVERIFIED','EXECUTION_AND_DEPTH_UNVERIFIED']
    return {'condition_id':src['condition_id'],'poly_title':source,'poly_asset':a,
            'poly_family':sf,'poly_first_trade_utc':src['first_trade_utc'],
            'poly_settled':bool(src['_settled']),'source_copy_buy_spend_1c_usd':float(src['_buy1c']) if src['_buy1c'] is not None else None,
            'source_pnl_1c_usd':float(src['_pnl1c']) if src['_pnl1c'] is not None else None,
            'kalshi_ticker':ticker,'kalshi_series':str(m.get('event_ticker') or '').split('-')[0] or ticker.split('-')[0],
            'kalshi_title':kt,'poly_strike_usd':ss,'kalshi_strike_usd':kk,
            'kalshi_open_time':m.get('open_time'),'kalshi_close_time':m.get('close_time'),
            'status':'PRELIMINARY_STRIKE_FAMILY_MATCH_NOT_VERIFIED','flags':'|'.join(flags)}


def write_csv(path,items,cols):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(items)


def self_test():
    def row(title,asset='ETH',fam='TOUCH_LOW',h='YEAR'):
        return {'condition_id':'p','title':title,'_asset':asset,'_family':fam,'_horizon':h,
                'first_trade_utc':'2026-01-15T00:00:00+00:00','_settled':True,'_buy1c':Decimal('10'),'_pnl1c':Decimal('3')}
    m={'ticker':'KXETHMINY-26DEC31-1750','title':'How low will Ethereum fall this year?',
       'yes_sub_title':'Below $1,750','floor_strike':1750,'open_time':'2025-12-15T00:00:00Z','close_time':'2026-12-31T23:59:00Z'}
    s=row('Will Ethereum dip to $1,750 by December 31, 2026?')
    assert assets('KXETHMINY')=={'ETH'}
    assert family(s['title'])=='TOUCH_LOW'
    assert family('Will ETH hit $2000 or $4000 first?')=='OTHER'
    assert series_family('Nvidia CEO leaving','KXCEONVDA')=='OTHER'
    assert series_family('How low will Ethereum fall this year?','KXETHMINY')=='TOUCH_LOW'
    assert candidate(s,m,'How low will Ethereum fall this year?') is not None
    wrong=dict(m, floor_strike=1800,yes_sub_title='Below $1,800')
    assert candidate(s,wrong,'How low will Ethereum fall this year?') is None
    late=dict(m,open_time='2026-02-15T00:00:00Z')
    assert candidate(s,late,'How low will Ethereum fall this year?') is None
    high=dict(m,ticker='KXETHMAXY-26DEC31-1750',title='How high will Ethereum get this year?')
    assert candidate(s,high,'How high will Ethereum get this year?') is None
    assert candidate(row('Will ETH hit $2000 or $4000 first?'),m,'How low will Ethereum fall this year?') is None
    ser={'KXETHMINY':{'title':'How low will Ethereum fall this year?','category':'Crypto'},
         'KXETH15M':{'title':'Ethereum 15 minute range','category':'Crypto'},
         'KXETHETF':{'title':'ETH ETF listed','category':'Crypto'}}
    selected=select_series(ser,[s])
    assert len(selected)==1 and selected[0]['series_ticker']=='KXETHMINY',selected
    print('PASS: crypto-only source selection, threshold, family, open-time, and unrelated-series rejection; NO API')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--series-only',action='store_true')
    p.add_argument('--offline-catalog',type=Path,help='Use existing catalog JSONL; no external GET')
    p.add_argument('--max-series',type=int,default=120)
    p.add_argument('--max-pages-per-series',type=int,default=15)
    p.add_argument('--page-size',type=int,default=1000)
    a=p.parse_args()
    if a.self_test:return self_test()
    if not(1<=a.max_series<=500 and 1<=a.max_pages_per_series<=200 and 1<=a.page_size<=1000):p.error('Invalid page/series limits')
    rows,by_settled=source_rows(a.cache)
    print(f'ALBERT V13 CRYPTO ONLY: source markets={len(rows)} settled={by_settled[True]} unverified={by_settled[False]}',flush=True)
    ct=Counter((r['_asset'] or 'MULTI_OR_UNCLASSIFIED',r['_family']) for r in rows)
    for (asset,fam),n in ct.most_common(20):
        subset=[r for r in rows if r['_asset']==(asset if asset!='MULTI_OR_UNCLASSIFIED' else '') and r['_family']==fam]
        spend=sum((r['_buy1c'] or 0) for r in subset if r['_settled'])
        pnl=sum((r['_pnl1c'] or 0) for r in subset if r['_settled'])
        print(f'   {asset:20} {fam:12} markets={n:4} BUY=${spend:10,.2f} PNL_1C=${pnl:+9,.2f}',flush=True)
    if a.offline_catalog:
        selected=[]; catalog=[]; diagnostics=[]
        for line in a.offline_catalog.open(encoding='utf8'):
            if line.strip():catalog.append(json.loads(line))
        print(f'OFFLINE catalog={len(catalog)} rows; this does not establish universe coverage',flush=True)
    else:
        series=v9.get_series()
        ranked=select_series(series,rows)
        selected=ranked[:a.max_series]
        write_csv(a.cache/'albert1953_v13_crypto_series.csv',ranked,
                  ['series_ticker','series_title','asset','family','horizon','source_market_mentions',
                   'source_settled_buy_spend_usd','score','category'])
        print(f'Identified {len(ranked)} relevant crypto-price series, will scan top {len(selected)}',flush=True)
        for r in selected[:35]:
            print(f"    {r['series_ticker']:22} {r['asset']:4} {r['family']:13} ${r['source_settled_buy_spend_usd']:9,.2f}  {r['series_title'][:55]}",flush=True)
        if a.series_only:
            print('SERIES ONLY. Candidate market analysis has not run.');return
        catalog=[];diagnostics=[];seen=set()
        for i,s in enumerate(selected,1):
            for tier in ('archive','live'):
                markets,d=v9.page_series(tier,s['series_ticker'],a.max_pages_per_series,a.page_size)
                diagnostics.append(d)
                print(f"  [{i}/{len(selected)}] {s['series_ticker']} {tier}: {len(markets)} markets, pages={d['pages']}, complete={d['complete']}",flush=True)
                for m in markets:
                    key=m.get('ticker')
                    if not key or key in seen:continue
                    seen.add(key)
                    m['_v13_series']=s['series_ticker'];m['_v13_series_title']=s['series_title']
                    catalog.append(m)
        with (a.cache/'albert1953_v13_crypto_kalshi_catalog.jsonl').open('w',encoding='utf8') as h:
            for m in catalog:h.write(json.dumps(m,separators=(',',':'))+'\n')
    selected_by_asset=defaultdict(list)
    for m in catalog:
        seri=m.get('_v13_series') or str(m.get('ticker','')).split('-')[0]
        title=m.get('_v13_series_title') or str(m.get('title') or '')
        for asset in assets(title+' '+seri):selected_by_asset[asset].append((m,title))
    results=[]
    for r in rows:
        if not r['_asset'] or r['_family']=='OTHER':continue
        for m,title in selected_by_asset[r['_asset']]:
            x=candidate(r,m,title)
            if x is not None:results.append(x)
    results.sort(key=lambda x:(x['condition_id'],x['kalshi_ticker']))
    matched={r['condition_id'] for r in results}
    settled=[r for r in rows if r['_settled']]
    matched_settled=[r for r in settled if r['condition_id'] in matched]
    def sums(rr):return (sum((r['_buy1c'] or 0) for r in rr),sum((r['_pnl1c'] or 0) for r in rr))
    total_buy,total_pnl=sums(settled);matched_buy,matched_pnl=sums(matched_settled)
    largest=sorted(matched_settled,key=lambda r:float(r['_pnl1c'] or 0),reverse=True)
    matched_ex5=sum((r['_pnl1c'] or 0) for r in largest[5:])
    matched_ex10=sum((r['_pnl1c'] or 0) for r in largest[10:])
    path=a.cache/'albert1953_v13_crypto_market_candidates.csv'
    cols=['condition_id','poly_title','poly_asset','poly_family','poly_first_trade_utc','poly_settled',
          'source_copy_buy_spend_1c_usd','source_pnl_1c_usd','kalshi_ticker','kalshi_series','kalshi_title',
          'poly_strike_usd','kalshi_strike_usd','kalshi_open_time','kalshi_close_time','status','flags']
    write_csv(path,results,cols)
    diag={
      'crypto_source_markets_all':len(rows),'crypto_source_markets_verified_settled':len(settled),
      'crypto_source_markets_payout_unverified_or_excluded':len(rows)-len(settled),
      'catalog_markets_checked':len(catalog),'candidate_source_markets':len(matched),
      'candidate_settled_source_markets':len(matched_settled),'candidate_rows':len(results),
      'settled_source_buy_spend_1c_usd':float(total_buy),'candidate_settled_source_buy_spend_1c_usd':float(matched_buy),
      'candidate_settled_buy_spend_coverage_pct':round(100*float(matched_buy/total_buy),3) if total_buy else None,
      'source_print_stress_pnl_1c_all_crypto_usd':float(total_pnl),
      'source_print_stress_pnl_1c_candidate_crypto_usd':float(matched_pnl),
      'source_print_stress_pnl_1c_candidate_ex_top5_usd':float(matched_ex5),
      'source_print_stress_pnl_1c_candidate_ex_top10_usd':float(matched_ex10),
      'all_market_pages_complete':all(d.get('complete',False) for d in diagnostics) if diagnostics else None,
      'scan_series':len(selected),'scan_pages':sum(d.get('pages',0) for d in diagnostics),
      'verified_settlement_equivalent_kalshi_markets':0,
      'warning':'Potential market titles/strikes only. Different index, averaging, expiry and execution remain unverified.'}
    (a.cache/'albert1953_v13_crypto_coverage.json').write_text(json.dumps(diag,indent=2))
    print('\nCRYPTO-ONLY CANDIDATE DISCOVERY (NOT VERIFIED EQUIVALENCE OR COPY P&L)')
    print(f"  source={len(rows)} settled={len(settled)} catalog={len(catalog)} proposed_source_matches={len(matched)} candidate_rows={len(results)}")
    print(f"  settled buy spend covered=${matched_buy:,.2f}/${total_buy:,.2f} ({diag['candidate_settled_buy_spend_coverage_pct']}%)")
    print(f'  settled 1c-stress SOURCE PnL of candidates=${matched_pnl:+,.2f}; ex5=${matched_ex5:+,.2f}; ex10=${matched_ex10:+,.2f}; all crypto=${total_pnl:+,.2f}')
    print(f"  selected page sets fully scanned={diag['all_market_pages_complete']} (not the same as searching all relevant series)")
    print('  VERIFIED EQUIVALENTS = ZERO; payout, index, expiry, YES/NO, market-listing and real quotes require review')
    print('  CANDIDATE CSV:',path)
    print('  COVERAGE JSON:',a.cache/'albert1953_v13_crypto_coverage.json')
    print('READ ONLY; NO ORDERS')

if __name__=='__main__':main()
