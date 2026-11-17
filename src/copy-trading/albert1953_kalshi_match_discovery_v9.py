#!/usr/bin/env python3
"""Albert1953 V9: COVERAGE-FIRST Kalshi candidate discovery (public GET only).

Purpose: find POSSIBLE Kalshi counterparts for the COMPLETE finance+crypto V8
universe, including losing markets. Do not approve matches, trade, or estimate P&L.
Settlement rules, asset price source, observation window, expiry, strike and YES/NO
mapping REQUIRE separate human/algorithmic verification before any replay.

Kalshi API: /series, /markets, /historical/markets (cursor-paginated).
Discovery is deliberately bounded and usually INCOMPLETE. "No candidate found"
means not found in the scanned data, NOT that no Kalshi equivalent exists.
No credentials, authenticated endpoints, orders, changes to earlier reports.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, json, re, sys, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE = 'https://api.elections.kalshi.com/trade-api/v2'
CATEGORIES = ('Financials', 'Companies', 'Economics', 'Crypto')
STOP = set('a an and are at as be by can could do does did end event every for from has have in is it its market of on or over per price prices the their there this to under vs was were what when where whether who will win with year how much than above below before after greater less high low yes no may new'.split())
ALIASES = {
    'bitcoin':('bitcoin','btc'), 'ethereum':('ethereum','eth'), 'solana':('solana','sol'),
    'silver':('silver','si'), 'gold':('gold','gc'), 'crude oil':('crude oil','wti','brent','oil','cl'),
    'apple':('apple','aapl'), 'microsoft':('microsoft','msft'), 'meta':('meta','meta platforms','fb'),
    'nvidia':('nvidia','nvda'), 'tesla':('tesla','tsla'), 'amazon':('amazon','amzn'),
    'alphabet':('alphabet','google','googl','goog'), 'sp500':('s&p 500','s&p500','spx','sp500'),
    'nasdaq':('nasdaq','ndx'), 'dogecoin':('doge','dogecoin'), 'xrp':('xrp','ripple'),
    'coinbase':('coinbase','coin'), 'openai':('openai',), 'palantir':('palantir','pltr'),
    'netflix':('netflix','nflx'), 'dollar':('dollar index','dxy'),
}
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9]+")
# compile a simpler fallback to avoid '$' condition with ambiguous suffix parsing
_USD = re.compile(r'\$\s*(\d[\d,]*(?:\.\d+)?)\s*(?:(trillion|billion|million|thousand|[kmbt])\b)?', re.I)
_AMOUNT = {'k':1e3,'thousand':1e3,'m':1e6,'million':1e6,'b':1e9,'billion':1e9,'t':1e12,'trillion':1e12}


def terms(txt):
    return set(w for w in _WORD.findall(str(txt).lower()) if len(w)>2 and w not in STOP)


def asset(txt):
    s=str(txt).lower()
    # Avoid ambiguous short ticker matches in running prose except standalone parentheses.
    hits=[]
    for key, words in ALIASES.items():
        for w in words:
            if len(w)<=3 and not re.search(r'\('+re.escape(w)+r'\)',s,re.I) and not re.search(r'\b'+re.escape(w)+r'\b',s,re.I):
                continue
            if re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)',s):
                hits.append(key);break
    return set(hits)


def strikes(txt):
    out=[]
    for m in _USD.finditer(str(txt)):
        val=float(m.group(1).replace(',',''))*_AMOUNT.get((m.group(2) or '').lower(),1)
        out.append(val)
    return sorted(set(out))


def predicate(txt):
    s=str(txt).lower()
    if re.search(r'\b(hit|touch|reach|dip|dipped|reached|traded|ever)\b',s):return 'touch'
    if re.search(r'\b(close|closes|closing|settle|settles|settlement|final trading day)\b',s):return 'close'
    if re.search(r'\b(above|below|greater|less|at least|at most|over|under)\b',s):return 'threshold'
    return 'unknown'


def years(txt):
    return set(re.findall(r'\b20(?:2[2-9]|3[0-9])\b',str(txt)))


def parse_iso(s):
    try:
        dt=datetime.fromisoformat(str(s).replace('Z','+00:00'))
        if dt.tzinfo is None:dt=dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError,TypeError,OverflowError):return None


def get_json(path,params=None,attempts=5):
    url=BASE+path+('?' + urlencode(params) if params else '')
    for i in range(attempts):
        try:
            with urlopen(Request(url,headers={'Accept':'application/json','User-Agent':'albert1953-research-readonly-v9/1'}),timeout=35) as r:
                p=json.load(r)
            if not isinstance(p,dict):raise RuntimeError('Non-object response '+url)
            return p
        except HTTPError as e:
            if e.code not in (429,500,502,503,504) or i==attempts-1:raise
            time.sleep(min(25,2**i+1))
        except (URLError,TimeoutError):
            if i==attempts-1:raise
            time.sleep(min(25,2**i+1))
    raise AssertionError('unreachable')


def load_universe(path):
    with path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows or not {'condition_id','first_trade_utc','title','category'}.issubset(rows[0]):
        raise ValueError('Missing complete V8 universe CSV fields')
    seen=set()
    for r in rows:
        cid=r['condition_id'].lower()
        if cid in seen:raise ValueError('Duplicate Polymarket condition '+cid)
        if r['category'] not in ('finance/economics','crypto'):raise ValueError('Non-finance/crypto row in universe')
        if parse_iso(r['first_trade_utc']) is None:raise ValueError('Invalid first trade timestamp')
        seen.add(cid)
    return rows


def extra_raw_unsettled(cache, seen):
    """Supplement the settled-only V8 universe with every other finance/crypto
    market in the full, maker+taker source cache. This avoids excluding live
    positions from the discovery universe. Classification matches V3 heuristics;
    these added rows have NO settled-P&L attribution.
    """
    meta=json.loads((cache/'albert1953_trades_meta.json').read_text())
    if not meta.get('complete') or meta.get('taker_only') is not False or str(meta.get('wallet','')).lower()!='0x777fae71d2ff9ec48a1213d48ba1d9d91024a1bb':
        raise ValueError('Source history incomplete or wrong wallet')
    by_condition={};n=0
    with (cache/'albert1953_trades.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():continue
            r=json.loads(line);n+=1
            cid=str(r.get('condition_id') or '').lower()
            if not cid:raise ValueError('Missing source condition id')
            if cid in seen:continue
            ts=int(r['timestamp'])
            title=str(r.get('title') or r.get('slug') or '')
            info=by_condition.setdefault(cid,{'ts':ts,'title':title,'slug':str(r.get('slug') or '')})
            if ts<info['ts']:info.update({'ts':ts,'title':title,'slug':str(r.get('slug') or '')})
    if n!=int(meta['rows']):raise ValueError('Source count differs from completeness metadata')
    out=[]
    for cid,v in by_condition.items():
        s=(v['title']+' '+v['slug']).lower()
        if re.search(r'temperature|weather|rainfall|snowfall',s):continue
        if re.search(r'bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|\bxrp\b|dogecoin|hyperliquid|crypto',s):cat='crypto'
        elif re.search(r'\bnfl\b|\bnba\b|\bmlb\b|\bnhl\b|\btennis\b|\bsoccer\b|\bvs\.?\b|\bsports\b',s):continue
        elif re.search(r'election|president|senate|governor|congress|democrat|republican|vote|trump|harris',s):continue
        elif re.search(r'\bstock\b|\bshares\b|\bcrude\b|\boil\b|\bgold\b|\bsilver\b|\bwti\b|\bmeta\b|\baapl\b|\bmsft\b|\bnvda\b|\bnasdaq\b|\bs&p\b|\bipo\b|\bmarket cap\b|\bfed\b|\binflation\b|\btreasury\b',s):cat='finance/economics'
        else:continue
        out.append({'condition_id':cid,'title':v['title'],'category':cat,
                    'first_trade_utc':datetime.fromtimestamp(v['ts'],timezone.utc).isoformat(),
                    'in_v8_verified_settled_universe':'false',
                    'source_type':'raw_unverified_non_v8'})
    return out


def get_series():
    ret={}
    for category in CATEGORIES:
        resp=get_json('/series',{'category':category})
        data=resp.get('series')
        if not isinstance(data,list):raise RuntimeError(f'/series did not return a list for {category}')
        for s in data:
            if not isinstance(s,dict) or not s.get('ticker'):continue
            ret[s['ticker']]=s
        print(f'Series discovery {category}: {len(data):,}',flush=True)
    return ret


def choose_series(series, universe, limit, user_list=None):
    if user_list:
        wanted=[s.strip().upper() for s in user_list.split(',') if s.strip()]
        absent=[x for x in wanted if x not in series]
        if absent:raise ValueError('Unknown Kalshi series ticker(s): '+', '.join(absent))
        return [(s,999) for s in wanted[:limit]]
    source_assets=Counter()
    for r in universe:
        for x in asset(r['title']):source_assets[x]+=1
    # Avoid letting dozens of high-frequency BTC series crowd out silver,
    # equity and other financial-market series in a small discovery budget.
    per_asset=defaultdict(list)
    for key,item in series.items():
        title=str(item.get('title') or '')
        aa=asset(title+' '+key)
        low=title.lower()
        penalty=25 if re.search(r'\b(5|15|30)[ -]?min|minute|hourly|hour by hour',low) else 0
        for x in aa:
            if source_assets.get(x):
                per_asset[x].append((key,max(1,source_assets[x]-penalty)))
    for group in per_asset.values():group.sort(key=lambda x:(-x[1],x[0]))
    selected=[];included=set();round_number=0
    keys=sorted(per_asset,key=lambda a:(-source_assets[a],a))
    while len(selected)<limit and any(round_number<len(per_asset[k]) for k in keys):
        for a in keys:
            group=per_asset[a]
            if round_number>=len(group):continue
            key,sc=group[round_number]
            if key not in included:
                included.add(key);selected.append((key,sc))
                if len(selected)==limit:break
        round_number+=1
    return selected


def page_series(tier,ticker,page_cap,limit):
    result=[];cursor=None;seen=set();complete=False;pg=0
    while pg<page_cap:
        args={'series_ticker':ticker,'limit':limit,'mve_filter':'exclude'}
        if cursor:args['cursor']=cursor
        payload=get_json('/historical/markets' if tier=='archive' else '/markets',args)
        rows=payload.get('markets')
        if not isinstance(rows,list):raise RuntimeError('Invalid '+tier+' markets response for '+ticker)
        result.extend(rows);pg+=1
        nxt=payload.get('cursor')
        if not nxt:
            complete=True;break
        if not isinstance(nxt,str) or nxt==cursor or nxt in seen:
            raise RuntimeError('Invalid pagination cursor for '+tier+':'+ticker)
        seen.add(nxt);cursor=nxt
        if pg%10==0:print(f'  {tier} {ticker}: {pg:,} pages, {len(result):,} markets',flush=True)
        time.sleep(.07)
    return result,{'tier':tier,'series':ticker,'pages':pg,'markets':len(result),'complete':complete}


def market_text(m):
    return ' '.join(str(m.get(k) or '') for k in ('title','subtitle','sub_title','yes_sub_title','no_sub_title'))


def index_catalog(markets):
    by_asset=defaultdict(set);inverted=defaultdict(set)
    for i,m in enumerate(markets):
        text=market_text(m)
        for a in asset(text):by_asset[a].add(i)
        for t in terms(text):inverted[t].add(i)
    return by_asset,inverted


def assess(source, market):
    sm=source['title'];km=market_text(market)
    ast=asset(sm);akt=asset(km)
    s_terms=terms(sm);k_terms=terms(km)
    shared=len(s_terms & k_terms)
    score=5*shared
    reasons=[]
    if ast and akt:
        if not ast & akt:return None
        score+=40
    elif shared<2:return None
    sa=strikes(sm);ka=strikes(km)
    if sa and ka:
        common=any(abs(s-k)<=max(.011,min(abs(s),abs(k))*.00001) for s in sa for k in ka)
        if common:score+=15
        else:reasons.append('STRIKE_MISMATCH')
    elif sa or ka:reasons.append('STRIKE_NOT_CONFIRMED')
    sy=years(sm);ky=years(km)
    if sy and ky:
        if not sy & ky:reasons.append('YEAR_MISMATCH')
        else:score+=10
    elif sy or ky:reasons.append('YEAR_NOT_CONFIRMED')
    sp=predicate(sm);kp=predicate(km)
    if sp==kp and sp!='unknown':score+=10
    elif sp!='unknown' and kp!='unknown':reasons.append('PREDICATE_MISMATCH')
    else:reasons.append('PREDICATE_UNVERIFIED')
    opened=parse_iso(market.get('open_time') or market.get('created_time'))
    close=parse_iso(market.get('close_time'))
    first=parse_iso(source['first_trade_utc'])
    if close and first and first>close:reasons.append('KALSHI_CLOSED_BEFORE_SIGNAL')
    if opened and first and opened>first:reasons.append('KALSHI_NOT_OPEN_AT_SIGNAL')
    if shared==0 and not (ast & akt):return None
    if 'STRIKE_MISMATCH' in reasons or 'YEAR_MISMATCH' in reasons:score-=30
    if 'KALSHI_CLOSED_BEFORE_SIGNAL' in reasons or 'KALSHI_NOT_OPEN_AT_SIGNAL' in reasons:score-=35
    return score,reasons


def shortlist(universe,markets,max_candidates):
    by_asset,inverted=index_catalog(markets)
    out=[]; found=0
    for src in universe:
        ass=asset(src['title']);tokens=terms(src['title'])
        ids=set()
        for a in ass:ids.update(by_asset.get(a,set()))
        if not ids:
            # Candidate recall can be incomplete for unrecognized assets. Only join on rare terms.
            rare=sorted((t for t in tokens if len(inverted.get(t,()))<=1500),key=lambda t:len(inverted.get(t,())))[:3]
            for t in rare:ids.update(inverted.get(t,()))
        guesses=[]
        for i in ids:
            m=markets[i]
            result=assess(src,m)
            if result and result[0]>=15 and not set(result[1]).intersection({'STRIKE_MISMATCH','YEAR_MISMATCH','PREDICATE_MISMATCH','KALSHI_CLOSED_BEFORE_SIGNAL','KALSHI_NOT_OPEN_AT_SIGNAL'}):
                score,reasons=result
                guesses.append((score,m,reasons))
        guesses.sort(key=lambda x:(-x[0],str(x[1].get('ticker'))))
        if guesses:found+=1
        # exactly one or more rows PER source market, including those with zero proposals
        for rank,(score,m,flags) in enumerate(guesses[:max_candidates],1):
            out.append({'condition_id':src['condition_id'], 'category':src['category'],
                'poly_title':src['title'],'first_trade_utc':src['first_trade_utc'],
                'candidate_rank':rank,'kalshi_ticker':m.get('ticker',''),
                'kalshi_title':market_text(m),'kalshi_status':m.get('status',''),
                'kalshi_result':m.get('result',''),'kalshi_close_time':m.get('close_time',''),
                'candidate_score':score,'disqualifiers_or_unknowns':'|'.join(flags),
                'in_v8_verified_settled_universe':src['in_v8_verified_settled_universe'], 'decision':'UNVERIFIED_CANDIDATE'})
        if not guesses:
            out.append({'condition_id':src['condition_id'],'category':src['category'],
                'poly_title':src['title'],'first_trade_utc':src['first_trade_utc'],
                'candidate_rank':'','kalshi_ticker':'','kalshi_title':'',
                'kalshi_status':'','kalshi_result':'','kalshi_close_time':'',
                'candidate_score':'','disqualifiers_or_unknowns':'',
                'in_v8_verified_settled_universe':src['in_v8_verified_settled_universe'], 'decision':'NO_CANDIDATE_IN_SCANNED_CATALOG'})
    return out,found


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test():
    s={'condition_id':'0x1','category':'crypto','title':'Will Bitcoin dip to $72,500 by December 31, 2026?',
       'first_trade_utc':'2026-10-15T13:00:00+00:00', 'in_v8_verified_settled_universe':'true'}
    good={'ticker':'KXBTC-26DEC31-B72500','title':'Will Bitcoin hit $72,500 by December 31, 2026?',
          'close_time':'2026-12-31T23:59:00Z'}
    bad={'ticker':'KXBTC-26DEC31-B75000','title':'Will Bitcoin close above $75,000 on December 31, 2026?',
         'close_time':'2026-12-31T23:59:00Z'}
    r=assess(s,good); assert r and r[0]>55 and 'STRIKE_MISMATCH' not in r[1]
    r=assess(s,bad); assert r and 'STRIKE_MISMATCH' in r[1]
    o,n=shortlist([s],[good,bad],3);assert n==1 and o[0]['decision']=='UNVERIFIED_CANDIDATE'
    earlier={**good,'close_time':'2025-12-31T00:00:00Z'}
    assert 'KALSHI_CLOSED_BEFORE_SIGNAL' in assess(s,earlier)[1]
    assert abs(strikes('Silver $90 by Jan 2026')[0]-90)<1e-6
    assert strikes('Will Bitcoin hit $72,500 by December?') == [72500.0]
    assert strikes('Will silver hit $90.50?') == [90.5]
    assert strikes('Will the IPO be worth $1.5 trillion?') == [1500000000000.0]
    print('PASS: candidate discovery, strict flags, no auto-verified matches, no API/orders')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('data/research/albert1953_ratio001'))
    p.add_argument('--max-series',type=int,default=30)
    p.add_argument('--max-pages-per-series',type=int,default=10)
    p.add_argument('--page-size',type=int,default=1000)
    p.add_argument('--max-candidates',type=int,default=3)
    p.add_argument('--only-series',type=str,help='Comma-separated Kalshi series tickers to scan')
    p.add_argument('--series-only',action='store_true',help='Discover relevant series, no market scans')
    p.add_argument('--offline',action='store_true',help='Use previously collected V9 catalog, no API requests')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:test();return
    if a.max_series<1 or a.max_pages_per_series<1 or not 1<=a.page_size<=1000 or not 1<=a.max_candidates<=20:
        p.error('Invalid caps; use positive --max-series and --max-pages-per-series, page <=1000')
    inp=a.cache/'albert1953_finance_crypto_v8_ratio0p01_universe.csv'
    universe=load_universe(inp)
    for row in universe:
        row['in_v8_verified_settled_universe']='true'
        row['source_type']='v8_verified_settled'
    supplement=extra_raw_unsettled(a.cache,{r['condition_id'].lower() for r in universe})
    universe.extend(supplement)
    print(f'Universe: {len(universe):,} finance/crypto markets = {len(universe)-len(supplement):,} V8 settled + {len(supplement):,} additional source markets (payout unverified/excluded)',flush=True)
    series_path=a.cache/'albert1953_v9_selected_series.csv'
    catalog_path=a.cache/'albert1953_v9_kalshi_catalog.jsonl'
    meta_path=a.cache/'albert1953_v9_catalog_meta.json'
    proposals_path=a.cache/'albert1953_v9_candidate_matches.csv'
    if a.offline:
        if not catalog_path.exists() or not meta_path.exists():p.error('Need V9 catalog and metadata to run offline')
        meta=json.loads(meta_path.read_text()); markets=[json.loads(s) for s in catalog_path.read_text().splitlines() if s.strip()]
        if meta.get('universe_count')!=len(universe):p.error('V9 cached universe count has changed; recollect')
    else:
        series=get_series()
        selection=choose_series(series,universe,a.max_series,a.only_series)
        write_csv(series_path,[{'rank':i+1,'series_ticker':s,'series_title':series[s].get('title',''),
                                'series_category':series[s].get('category',''),
                                'source_asset_weight':score,'selected_for_scan':True}
                               for i,(s,score) in enumerate(selection)])
        print(f'Finance/crypto universe={len(universe):,} markets. Selected candidate series={len(selection)}',flush=True)
        for i,(s,score) in enumerate(selection[:15],1):print(f'  {i:2}. {s:20} weight={score:4} {str(series[s].get("title",""))[:75]}')
        if a.series_only:
            print('SERIES-ONLY. Review:',series_path,'(these are discovery candidates, NOT verified equivalent markets).')
            return
        if not selection:raise RuntimeError('No relevant series found; use --only-series or review /series')
        markets_by_ticker={};scans=[]
        for i,(s,weight) in enumerate(selection,1):
            for tier in ('archive','live'):
                rows,diag=page_series(tier,s,a.max_pages_per_series,a.page_size)
                for m in rows:
                    if isinstance(m,dict) and m.get('ticker'):
                        # Prefer live-tier record if same ticker appears in both.
                        markets_by_ticker[m['ticker']]=m
                scans.append(diag)
            print(f'[{i}/{len(selection)}] {s}: catalog now {len(markets_by_ticker):,} distinct markets',flush=True)
        markets=list(markets_by_ticker.values())
        meta={'source':'Kalshi public /series, /markets, /historical/markets',
              'universe_count':len(universe),'v8_verified_settled_count':len(universe)-len(supplement), 'additional_raw_source_count':len(supplement),'selected_series':len(selection),
              'markets_retrieved':len(markets),'scans':scans,
              'all_pages_complete':all(d['complete'] for d in scans),
              'limitations':'Series discovery and page caps can exclude matches; NEVER assume missing candidate means no equivalent exists. No settlement equivalence approval.'}
        a.cache.mkdir(parents=True,exist_ok=True)
        with catalog_path.open('w') as f:
            for m in markets:f.write(json.dumps(m,separators=(',',':'))+'\n')
        meta_path.write_text(json.dumps(meta,indent=2))
    out,count=shortlist(universe,markets,a.max_candidates)
    write_csv(proposals_path,out)
    coverage={'universe':len(universe),'candidate_found_for':count,'candidate_found_pct':round(100*count/len(universe),2),
              'scanned_catalog_markets':len(markets),'catalog_page_complete':meta['all_pages_complete'], 'v8_verified_settled_count':len(universe)-len(supplement),'unverified_other_markets_count':len(supplement),
              'note':'RETRIEVAL proposals only; verified equivalent matches = ZERO until rules manually validated.'}
    (a.cache/'albert1953_v9_candidate_coverage.json').write_text(json.dumps(coverage,indent=2))
    print('\nALBERT1953 V9 — KALSHI CANDIDATE DISCOVERY; NEVER VERIFIED OR TRADABLE')
    print(f'Universe {len(universe):,}; scanned Kalshi markets {len(markets):,}; possible candidates for {count:,} Polymarket markets')
    print('All scanned page sets complete:',meta['all_pages_complete'],'(FALSE means incomplete search; no-match is not definitive)')
    print('Verified equivalent markets: ZERO; settlement rules and side mapping MUST be checked separately')
    print('CANDIDATES:',proposals_path)
    print('COVERAGE:',a.cache/'albert1953_v9_candidate_coverage.json')
    print('No Kalshi quotes, no orders, no historical copying P&L. Research discovery only.')

if __name__=='__main__':main()
