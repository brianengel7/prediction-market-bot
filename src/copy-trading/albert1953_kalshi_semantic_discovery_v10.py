#!/usr/bin/env python3
"""Albert1953 V10: conservative Kalshi market counterpart discovery.

Research ONLY. Uses the existing V9.1 downloader and public unauthenticated Kalshi
GET endpoints; never places orders or authenticates. Does not assert equivalence,
copyability, historical liquidity, executable prices or profitability.

Differences from V9.1:
- Searches Commodities as well as Financials/Companies/Economics/Crypto.
- Chooses series using source market payoff families and relevant underlying, NOT
  simply based on the frequency of a company name (CEO/capex false positives).
- Rejects unrelated payout families (touch-low vs daily closing price, etc.).
- Maintains complete universe and records not-found as unverified coverage.
- Caps giant high-frequency series individually, explicitly marks incomplete.
- Writes V10 files only, leaving all previous caches and audits untouched.

Requires albert1953_kalshi_match_discovery_v9_1.py alongside this file.
"""
from __future__ import annotations
import argparse, collections, csv, importlib.util, json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path(__file__).with_name('albert1953_kalshi_match_discovery_v9_1.py')
if not SOURCE.is_file():
    raise SystemExit(f'Missing V9.1 in same folder: {SOURCE}')
spec = importlib.util.spec_from_file_location('albert_v9_1_readonly', SOURCE)
v9 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v9
spec.loader.exec_module(v9)
v9.CATEGORIES = ('Financials','Companies','Economics','Crypto','Commodities')

# Weak words that do not establish identical outcomes. Never infer settlement
# equivalence from these aliases or the scoring alone.
FAMILY_TERMS = {
    'touch_low': ('dip', 'fall below', 'fall to', 'lowest', 'low will', 'how low', 'minimum', 'min price', 'min/max low'),
    'touch_high': ('hit', 'reach', 'touch', 'highest', 'how high', 'maximum', 'all time high', 'all-time high'),
    'closing': ('close above', 'close below', 'close at', 'close over', 'closing', 'settle', 'settles', 'settlement price', 'final trading day'),
    'ranking': ('largest company', 'biggest company', 'market cap rank', 'market cap on'),
    'corporate': ('ceo', 'board', 'acquisition', 'merger', 'antitrust', 'ipo', 'launch an ipo', 'capex', 'shareholder'),
    'range': ('range', 'between', 'interval'),
}
SHORT_SPANS = re.compile(r'\b(5[ -]?min|15[ -]?min|30[ -]?min|five[- ]minute|fifteen[- ]minute|daily|hourly|every hour|weekly|weekly range)\b',re.I)
LONG_SPANS = re.compile(r'\b(by |before |end of |end-of-|december|january|february|march|april|may |june|july|august|september|october|november|year|202[2-9])',re.I)
QUOTES = re.compile(r'\b(price|dollars|stock|shares|market cap|market capitalization|index|bitcoin|ethereum|silver|gold|oil|nasdaq|s&p|btc|eth)\b',re.I)


def family(text: str) -> str:
    s=' '.join(str(text).lower().replace('-', ' ').split())
    # Order matters: close over a price is NOT the same as ever touching it.
    if any(t in s for t in FAMILY_TERMS['ranking']):return 'ranking'
    if any(t in s for t in FAMILY_TERMS['corporate']):return 'corporate'
    if any(t in s for t in FAMILY_TERMS['closing']):return 'closing'
    if 'dip' in s or 'how low' in s or 'lowest' in s or 'minimum' in s or 'fall below' in s or 'fall to' in s or 'low will' in s:return 'touch_low'
    if any(t in s for t in FAMILY_TERMS['touch_high']):return 'touch_high'
    if any(t in s for t in FAMILY_TERMS['range']):return 'range'
    # A generic price above/below at a stated date is normally a snapshot,
    # though it still requires rule-level verification.
    if re.search(r'\b(above|below|over|under|at least|at most)\b',s) and QUOTES.search(s):return 'closing'
    return 'unknown'


def specific_years(txt):return v9.years(txt)


def compat(source_title, series_title, series_ticker=''):
    """Conservative compatibility for PAYOFF FAMILY, not a verified match."""
    sf=family(source_title); tf=family(series_title)
    s=str(source_title).lower(); t=str(series_title).lower(); ticker=str(series_ticker).upper()
    if not (v9.asset(source_title) & v9.asset(series_title+' '+ticker)):
        return False,'ASSET_NOT_SHARED'
    # Explicit families block unrelated series. An unknown series remains a
    # review-only prospect, never strong evidence of a match.
    if sf=='touch_low':
        if tf in ('closing','corporate','ranking','range','touch_high'):return False,'INCOMPATIBLE_LOW'
        if ('MAX' in ticker or 'HIGH' in ticker) and ('MIN' not in ticker):return False,'MAX_FOR_LOW'
    elif sf=='touch_high':
        if tf in ('closing','corporate','ranking','range','touch_low'):return False,'INCOMPATIBLE_HIGH'
        if ('MIN' in ticker or 'LOW' in ticker) and ('MAX' not in ticker):return False,'MIN_FOR_HIGH'
    elif sf=='closing':
        if tf in ('corporate','ranking','touch_low','touch_high'):return False,'NOT_SNAPSHOT'
        if re.search(r'\b(hourly|5[ -]?min|15[ -]?min)\b',t) and LONG_SPANS.search(s):return False,'HORIZON_MISMATCH'
    elif sf=='ranking':
        if tf not in ('ranking','unknown'):return False,'NOT_RANKING'
    elif sf=='corporate':
        if tf in ('touch_low','touch_high','closing','range','ranking'):return False,'NOT_CORPORATE_EVENT'
        # Must match a specific event concept, not just the company.
        concepts={'ipo','ceo','merger','acquisition','antitrust','capex','board'}
        if not any(z in s and z in t for z in concepts):return False,'CORPORATE_EVENT_MISMATCH'
    elif sf=='range':
        if tf not in ('range','unknown'):return False,'NOT_RANGE'
    else:
        if tf=='corporate' and not any(z in s for z in ('ceo','ipo','board','merger','acquisition','antitrust','capex')):
            return False,'UNRELATED_CORPORATE'
    if sf in ('touch_low','touch_high') and SHORT_SPANS.search(t) and LONG_SPANS.search(s):
        return False,'HORIZON_MISMATCH'
    return True,'FAMILY_COMPATIBLE' if tf==sf else 'SERIES_FAMILY_UNVERIFIED'


def semantic_series_score(key, info, universe, source_groups):
    title=str(info.get('title') or '')
    a=v9.asset(title+' '+key)
    if not a:return 0,0,[]
    supporting=0;strong=0; examples=[]
    for key_asset in a:
        for row in source_groups.get(key_asset,[]):
            ok,why=compat(row['title'],title,key)
            if not ok:continue
            supporting+=1
            if why=='FAMILY_COMPATIBLE':strong+=1
            if len(examples)<2:examples.append(row['title'][:100])
    if not supporting:return 0,0,[]
    # Penalize giant micro-duration books for long-dated event markets.
    sfset={family(r['title']) for x in a for r in source_groups.get(x,[])}
    penalty=0
    if SHORT_SPANS.search(title) and not any(f=='closing' for f in sfset):penalty=100
    if re.search(r'\b(range|daily range)\b',title,re.I) and 'range' not in sfset:penalty+=100
    return max(0,10*strong+min(50,supporting)-penalty),supporting,examples


def select_series(all_series,universe,limit,only=None):
    if only:
        selected=[x.strip().upper() for x in only.split(',') if x.strip()]
        unknown=[x for x in selected if x not in all_series]
        if unknown:raise ValueError('Unknown requested series '+', '.join(unknown))
        return [(x,0,0,[]) for x in selected[:limit]]
    groups=collections.defaultdict(list)
    for row in universe:
        for a in v9.asset(row['title']):groups[a].append(row)
    rows=[]
    for key,info in all_series.items():
        score,n,examples=semantic_series_score(key,info,universe,groups)
        if score>0:rows.append((key,score,n,examples))
    # Guarantee diversity of underlyings so a large BTC cohort doesn't consume
    # all series slots. Ranking remains diagnostic, never a verified matcher.
    rows.sort(key=lambda x:(-x[1],-x[2],x[0]))
    chosen=[];count=collections.Counter()
    for key,score,n,examples in rows:
        aa=v9.asset(str(all_series[key].get('title',''))+' '+key)
        if any(count[a]>=6 for a in aa):continue
        chosen.append((key,score,n,examples))
        for a in aa:count[a]+=1
        if len(chosen)>=limit:break
    return chosen


def proposed_assessment(source, market, series_title):
    src=source['title']; kal=v9.market_text(market); ticker=str(market.get('ticker') or '')
    extended=kal+' '+series_title
    same=v9.asset(src) & v9.asset(extended+' '+ticker)
    if not same:return None
    sf=family(src);tf=family(series_title)
    ok,reason=compat(src,series_title,ticker)
    if not ok:return None
    # Do not accept source 'hit' against Kalshi explicit 'close'.
    mk=family(kal)
    if sf in ('touch_low','touch_high') and mk in ('closing','corporate','ranking','range'):return None
    if sf=='closing' and mk in ('touch_low','touch_high','corporate','ranking'):return None
    if sf=='ranking' and mk in ('touch_low','touch_high','closing','corporate'):return None
    # Prefer genuine shared thresholds. A strike is only confirmed when both
    # sides show an explicit dollar amount; ticks like T44.99 are NOT decoded.
    ss=v9.strikes(src);ks=v9.strikes(kal)
    flags=[];score=25
    if ss and ks:
        if not any(abs(x-y)<=max(.015,min(abs(x),abs(y))*.00001) for x in ss for y in ks):return None
        flags.append('TEXT_STRIKE_MATCH');score+=35
    else:flags.append('STRIKE_UNVERIFIED')
    ys=v9.years(src);yk=v9.years(kal+' '+str(market.get('close_time') or ''))
    if ys and yk:
        if not (ys & yk):return None
        flags.append('YEAR_MATCH');score+=15
    else:flags.append('YEAR_UNVERIFIED')
    st=v9.parse_iso(source.get('first_trade_utc'))
    ot=v9.parse_iso(market.get('open_time') or market.get('created_time'))
    ct=v9.parse_iso(market.get('close_time'))
    if ot is not None and st is not None and ot>st:return None
    if ct is not None and st is not None and ct<st:return None
    flags.append('OPEN_AT_SIGNAL_NOT_VERIFIED' if ot is None or ct is None else 'LISTING_TIME_COMPATIBLE')
    if sf==tf:score+=25;flags.append('FAMILY_MATCH')
    else:flags.append('FAMILY_REVIEW_REQUIRED')
    sw=v9.terms(src);kw=v9.terms(kal+' '+series_title)
    shared=len(sw & kw);score+=min(20,shared*4)
    # Missing exact date/threshold/settlement source is never an automatic match.
    flags.extend(['EXPIRY_TERMS_UNVERIFIED','SETTLEMENT_SOURCE_UNVERIFIED','SIDE_MAPPING_UNVERIFIED'])
    return score,flags


def write_csv(path,rows,columns):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--max-series',type=int,default=50)
    p.add_argument('--max-pages-per-series',type=int,default=5)
    p.add_argument('--page-size',type=int,default=1000)
    p.add_argument('--max-candidates',type=int,default=3)
    p.add_argument('--only-series',type=str)
    p.add_argument('--series-only',action='store_true')
    p.add_argument('--offline-v9',action='store_true',help='Use previous V9.1 market catalog only; incomplete coverage')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:return self_test()
    if not (1<=a.max_series<=250 and 1<=a.max_pages_per_series<=100 and 1<=a.page_size<=1000 and 1<=a.max_candidates<=10):p.error('Invalid limits')
    input_csv=a.cache/'albert1953_finance_crypto_v8_ratio0p01_universe.csv'
    universe=v9.load_universe(input_csv)
    for r in universe:r['in_v8_verified_settled_universe']='true'
    supplement=v9.extra_raw_unsettled(a.cache,{r['condition_id'].lower() for r in universe})
    universe.extend(supplement)
    print(f'Universe {len(universe):,} (settled V8={len(universe)-len(supplement):,}; extra unverified={len(supplement):,})')
    selected=[];scans=[];catalog={};metadata={}
    if a.offline_v9:
        c=a.cache/'albert1953_v9_kalshi_catalog.jsonl'
        if not c.exists():p.error('Missing V9 catalog. Run V9.1 first.')
        series_titles={}
        old_series=a.cache/'albert1953_v9_selected_series.csv'
        if old_series.exists():
            for row in csv.DictReader(old_series.open(newline='',encoding='utf8')):
                if row.get('series_ticker'):
                    series_titles[row['series_ticker']]=row.get('series_title','')
        for line in c.open(encoding='utf8'):
            if line.strip():
                m=json.loads(line)
                if m.get('ticker'):
                    m['_v10_series']=str(m['ticker']).split('-')[0]
                    m['_v10_series_title']=series_titles.get(m['_v10_series'],'')
                    catalog[m['ticker']]=m
        oldmeta=a.cache/'albert1953_v9_catalog_meta.json'
        if oldmeta.exists():metadata=json.loads(oldmeta.read_text())
        print(f'Using old, potentially incomplete V9 catalog: {len(catalog):,} market rows')
    else:
        series=v9.get_series()
        selected=select_series(series,universe,a.max_series,a.only_series)
        if not selected:raise RuntimeError('No semantically relevant series found')
        sd=[{'rank':i+1,'ticker':s,'title':series[s].get('title',''),'category':series[s].get('category',''),
             'score':sc,'compatible_source_market_mentions':cnt,'examples':' | '.join(ex)}
            for i,(s,sc,cnt,ex) in enumerate(selected)]
        sp=a.cache/'albert1953_v10_selected_series.csv'
        write_csv(sp,sd,list(sd[0]))
        print('Candidate series ranking (NOT verified):')
        for i,(s,score,cnt,ex) in enumerate(selected[:min(len(selected),30)],1):
            print(f'  {i:2d}. {s:20} score={score:5d} compatible_mentions={cnt:5d} {series[s].get("title", "")[:70]}')
        if a.series_only:
            print('SERIES ONLY:',sp);return
        # Standard metadata for predicates omitted in market titles.
        for i,(s,sc,cnt,ex) in enumerate(selected,1):
            for tier in ('archive','live'):
                batch,diag=v9.page_series(tier,s,a.max_pages_per_series,a.page_size)
                scans.append(diag)
                for m in batch:
                    if m.get('ticker'):
                        m=dict(m);m['_v10_series_title']=series[s].get('title','');m['_v10_series']=s
                        catalog[m['ticker']]=m
                print(f'[{i:02d}/{len(selected)}] {s:22} {tier:7} n={len(batch):5,d} complete={diag["complete"]}')
            time.sleep(.05)
        cp=a.cache/'albert1953_v10_kalshi_catalog.jsonl'
        with cp.open('w',encoding='utf8') as f:
            for m in catalog.values():f.write(json.dumps(m,separators=(',',':'))+'\n')
        print('Catalog saved:',cp)
    series_name={m.get('_v10_series') or str(m['ticker']).split('-')[0]:m.get('_v10_series_title') or '' for m in catalog.values()}
    by_asset=collections.defaultdict(list)
    for m in catalog.values():
        txt=v9.market_text(m)+' '+m.get('_v10_series_title','')+' '+str(m.get('ticker') or '')
        for x in v9.asset(txt):by_asset[x].append(m)
    report=[];coverage=collections.Counter(); strict_count=0;possible=0
    for src in universe:
        possibles=[];probed=set()
        for x in v9.asset(src['title']):
            for m in by_asset.get(x,[]):
                if m['ticker'] in probed:continue
                probed.add(m['ticker'])
                title=m.get('_v10_series_title') or series_name.get(str(m['ticker']).split('-')[0],'')
                r=proposed_assessment(src,m,title)
                if r:possibles.append((r[0],m,r[1]))
        possibles.sort(key=lambda z:(-z[0],str(z[1].get('ticker',''))))
        coverage[src['category']+'_all']+=1
        if possibles:
            possible+=1;coverage[src['category']+'_candidate']+=1
        else:
            report.append({'condition_id':src['condition_id'],'category':src['category'],'poly_title':src['title'],
                'first_trade_utc':src['first_trade_utc'],'kalshi_ticker':'','kalshi_title':'','score':'',
                'review_flags':'NO_SEMANTIC_CANDIDATE_IN_SCANNED_CATALOG',
                'source_settled_in_v8':src['in_v8_verified_settled_universe'],'decision':'NOT_FOUND_IN_THIS_SCAN'})
        for score,m,flags in possibles[:a.max_candidates]:
            report.append({'condition_id':src['condition_id'],'category':src['category'],'poly_title':src['title'],
                'first_trade_utc':src['first_trade_utc'],'kalshi_ticker':m['ticker'],
                'kalshi_title':v9.market_text(m),'score':score,'review_flags':'|'.join(flags),
                'source_settled_in_v8':src['in_v8_verified_settled_universe'],'decision':'UNVERIFIED_REVIEW_CANDIDATE'})
    out=a.cache/'albert1953_v10_candidate_matches.csv'
    write_csv(out,report,['condition_id','category','poly_title','first_trade_utc','kalshi_ticker','kalshi_title','score','review_flags','source_settled_in_v8','decision'])
    complete=bool(scans) and all(z.get('complete') for z in scans)
    summary={'source_universe':len(universe),'v8_settled':len(universe)-len(supplement),'extra_unverified':len(supplement),
        'catalog_markets':len(catalog),'selected_series':len(selected) if selected else None,
        'semantically_plausible_markets':possible,'plausible_pct':round(100*possible/len(universe),2),
        'verified_equivalent':0,'scans_complete':complete,'offline_v9_catalog':a.offline_v9,
        'coverage_by_category':dict(coverage),'scans':scans,
        'warning':'No matches have been verified against exact settlement rules, sides, prices, historical availability or books. Incomplete catalog makes not-found inconclusive.'}
    jp=a.cache/'albert1953_v10_candidate_coverage.json';jp.write_text(json.dumps(summary,indent=2))
    print(f'V10 RESULTS: source={len(universe):,}; scanned={len(catalog):,}; SEMANTIC candidates={possible} ({100*possible/len(universe):.2f}%), VERIFIED=0')
    print('Catalog all selected page sets complete:',complete)
    print('CANDIDATES:',out,'\nCOVERAGE:',jp,'\nNo orders or executable backtest.')


def self_test():
    good_src='Will Ethereum dip to $1,500 by December 31, 2026?'
    assert family(good_src)=='touch_low'
    assert family('Will Microsoft close above $410 on February 9?')=='closing'
    assert compat(good_src,'How low will Ethereum fall this year?','KXETHMINY')[0]
    assert not compat(good_src,'How high will Ethereum get this year?','KXETHMAXY')[0]
    assert not compat('Will Nvidia close above $180?','Nvidia CEO leaving','KXCEONVDA')[0]
    assert not compat('Will silver hit $90 by January?','Silver monthly closing price','KXSILVERMON')[0]
    src={'title':good_src,'first_trade_utc':'2026-10-10T12:00:00+00:00'}
    good={'ticker':'KXETHMINY-26DEC31-T1500','title':'Will Ethereum ever dip below $1,500 by December 31, 2026?',
          'close_time':'2026-12-31T23:00:00Z','open_time':'2026-01-01T00:00:00Z'}
    assert proposed_assessment(src,good,'How low will Ethereum fall this year?') is not None
    bad={**good,'title':'Will Ethereum ever dip below $1,700 by December 31, 2026?'}
    assert proposed_assessment(src,bad,'How low will Ethereum fall this year?') is None
    after={**good,'open_time':'2026-11-11T00:00:00Z'}
    assert proposed_assessment(src,after,'How low will Ethereum fall this year?') is None
    assert proposed_assessment(src,good,'How high will Ethereum get this year?') is None
    print('PASS: financial family screening, strike/year matching, open-at-signal, no auto verification')

if __name__=='__main__':main()
