#!/usr/bin/env python3
"""Read-only Polymarket wallet trade and copy-feasibility audit.

Uses public Polymarket Data API v2 trades, with explicit completeness checks.
No authenticated endpoints, orders, or trading-module imports. API-compatible
with the project's historical read-only audit; results are NOT execution P&L.

Optional executable quote observations: CSV columns timestamp,token_id,ask,ask_size.
At signal_time + latency, the first quote up to max-quote-lag seconds later
is an observation, NOT proof that a copier could fill at that price.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = 'https://data-api.polymarket.com'
WALLETS = {
    'goodfallen': '0x0ca4e9d27ff826a109ecbf74754fb66b93f277a6',
    'wallet_954b': '0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0',
    'asdcaxz': '0x5a06acbaee263540a16a0109d584f6672101ffa2',
    'ChubbyGooseLearningCenter': '0x04b0915f2edb018644b28dc0d3e21ca2089a000d',
    'HighTempTation': '0x6011655c4afb76f36dd1b08a137a1ba73466b31e',
}
WINDOW = re.compile(r'(?:^|[-_])(btc|bitcoin|eth|ethereum|sol|solana|xrp)[-_](?:updown|up-or-down)[-_](5m|15m)[-_](\d{10})(?:$|[-_])', re.I)

def epoch(s):
    return int(datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

def get_json(path, params, tries=6):
    url = API + path + '?' + urlencode(params)
    for attempt in range(tries):
        try:
            req = Request(url, headers={'Accept':'application/json','User-Agent':'poly-wallet-readonly-audit/1.0'})
            with urlopen(req, timeout=45) as resp:
                obj = json.load(resp)
            if not isinstance(obj, dict) or 'data' not in obj:
                raise RuntimeError(f'Unexpected response envelope for {path}')
            return obj
        except HTTPError as e:
            if e.code not in (429,500,502,503,504) or attempt == tries-1:
                raise
            time.sleep(min(30, 2**attempt))
        except (URLError, TimeoutError):
            if attempt == tries-1:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError('unreachable')

def download_trades(wallet, start, end, path, max_pages):
    params={'user':wallet,'limit':1000,'taker_only':'false','start':start,'end':end-1}
    cursor=None; seen=set(); rows=0
    tmp=path.with_suffix('.partial')
    with tmp.open('w',encoding='utf8') as f:
        for page in range(1,max_pages+1):
            q=dict(params)
            if cursor: q['cursor']=cursor
            obj=get_json('/v2/trades',q)
            data=obj['data']; pagination=obj.get('pagination')
            if not isinstance(data,list) or not isinstance(pagination,dict):
                raise RuntimeError('Unexpected trade/pagination response; do not treat as empty')
            for item in data:
                ts=int(item['timestamp'])
                if not start <= ts < end: raise RuntimeError('Trade outside requested window')
                f.write(json.dumps(item,separators=(',',':'))+'\n'); rows+=1
            if page%20==0: print(f'  downloaded {rows:,} fills ({page} pages)',flush=True)
            nxt=pagination.get('next_cursor')
            if nxt is None:
                if pagination.get('has_more') is True: raise RuntimeError('Pagination missing cursor')
                tmp.replace(path)
                return {'complete':True,'rows':rows,'pages':page,'start':start,'end_exclusive':end,'wallet':wallet}
            if not isinstance(nxt,str) or not nxt or nxt in seen:
                raise RuntimeError('Cursor repeated or invalid')
            cursor=nxt;seen.add(cursor);time.sleep(.05)
    trunc=path.with_suffix('.truncated');tmp.replace(trunc)
    return {'complete':False,'rows':rows,'pages':max_pages,'partial_file':str(trunc),'start':start,'end_exclusive':end,'wallet':wallet}

def classify(r):
    s=' '.join(str(r.get(k) or '') for k in ('slug','event_slug','title')).lower()
    if re.search(r'temperature|weather|rainfall|snow|precipitation|highest temp',s):return 'weather'
    if re.search(r'bitcoin|btc|ethereum|eth|solana|sol|xrp|crypto',s):return 'crypto'
    if re.search(r'\bnba\b|\bnfl\b|\bnhl\b|\bmlb\b|tennis|soccer|basketball|football|\bvs\.?\b',s):return 'sports'
    return 'other'

def num(x):
    try:
        f=float(x)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError):return None

def signals_from(rows, cluster_seconds=2):
    buys=sorted((r for r in rows if str(r.get('side','')).upper()=='BUY'),key=lambda r:(str(r.get('condition_id')),str(r.get('token_id')),int(r['timestamp'])))
    previous={};signals=[]
    for r in buys:
        price=num(r.get('price'));size=num(r.get('size'))
        if price is None or size is None or not (0<price<1 and size>0):continue
        key=(str(r.get('condition_id')),str(r.get('token_id')))
        t=int(r['timestamp']); old=previous.get(key)
        if old is not None and t-old['last_ts']<=cluster_seconds:
            old['last_ts']=t;old['size']+=size;old['spend']+=size*price;old['fills']+=1
        else:
            x={'timestamp':t,'last_ts':t,'condition_id':key[0],'token_id':key[1],
               'outcome_index':r.get('outcome_index'),'slug':r.get('slug',''),
               'category':classify(r),'size':size,'spend':size*price,'fills':1}
            signals.append(x);previous[key]=x
    for x in signals:x['avg_price']=round(x['spend']/x['size'],6)
    return sorted(signals,key=lambda x:x['timestamp'])

def analyze(rows,signals):
    markets={str(r.get('condition_id')) for r in rows if r.get('condition_id')}
    sides=Counter(str(r.get('side','')).upper() for r in rows)
    categories=Counter(classify(r) for r in rows)
    bought=defaultdict(lambda:defaultdict(list));sell_markets=set()
    for r in rows:
        market=str(r.get('condition_id'));side=str(r.get('side','')).upper()
        if side=='SELL':sell_markets.add(market)
        if side!='BUY':continue
        p=num(r.get('price'));q=num(r.get('size'))
        if p is not None and q is not None and 0<p<1 and q>0:
            bought[market][str(r.get('outcome_index'))].append((p,q))
    both=[];pair_prices=[]
    for m,o in bought.items():
        if '0' not in o or '1' not in o:continue
        both.append(m)
        if m not in sell_markets:
            pair_prices.append(sum(sum(p*q for p,q in o[i])/sum(q for _,q in o[i]) for i in ('0','1')))
    seconds=Counter(x['timestamp'] for x in signals)
    windows=[];open60=0
    for x in signals:
        match=WINDOW.search(str(x.get('slug') or ''))
        if not match:continue
        start=int(match.group(3));duration=300 if match.group(2).lower()=='5m' else 900
        if start <= x['timestamp'] < start+duration:
            windows.append(x)
            if x['timestamp']-start<60:open60+=1
    def pct(n,d):return round(100*n/d,2) if d else None
    return {'fills':len(rows),'markets':len(markets),'buy_fills':sides.get('BUY',0),'sell_fills':sides.get('SELL',0),
       'categories':dict(categories),'buy_clusters':len(signals),'median_buy_cluster_usdc':round(statistics.median(x['spend'] for x in signals),4) if signals else None,
       'peak_buy_clusters_same_second':max(seconds.values(),default=0),'both_sides_bought_markets':len(both),
       'both_sides_bought_pct':pct(len(both),len(markets)),
       'no_sell_both_sides_avg_price_lt_1_markets':sum(p<1 for p in pair_prices),
       'no_sell_both_sides_price_samples':len(pair_prices),
       'windowed_crypto_buy_clusters':len(windows),'windowed_crypto_first_60s_pct':pct(open60,len(windows)),
       'execution_verdict':'NOT TESTABLE WITHOUT HISTORICAL EXECUTABLE QUOTES'}

def quote_replay(signals,path,latencies,max_quote_lag,copy_usdc):
    d=defaultdict(list)
    with path.open(newline='',encoding='utf8') as f:
        reader=csv.DictReader(f)
        if not {'timestamp','token_id','ask','ask_size'}.issubset(reader.fieldnames or []):
            raise ValueError('Quote CSV requires timestamp,token_id,ask,ask_size')
        for r in reader:
            t=num(r['timestamp']);price=num(r['ask']);size=num(r['ask_size'])
            if t is not None and price is not None and size is not None and 0<price<1 and size>=0:
                d[str(r['token_id'])].append((t,price,size))
    for v in d.values():v.sort()
    times={k:[row[0] for row in v] for k,v in d.items()}
    result=[]
    for lag in latencies:
        matched=0;depth=0;slips=[]
        for x in signals:
            token=str(x['token_id']);v=d.get(token,[])
            if not v:continue
            t=x['timestamp']+lag;j=bisect.bisect_left(times[token],t)
            if j>=len(v) or v[j][0]>t+max_quote_lag:continue
            matched+=1;ask=v[j][1];available=v[j][2]
            if available>=copy_usdc/ask:
                depth+=1;slips.append(100*(ask-x['avg_price']))
        result.append({'latency_seconds':lag,'buy_clusters':len(signals),'fresh_quote_count':matched,
            'sufficient_best_ask_depth_count':depth,'median_price_slippage_cents':round(statistics.median(slips),3) if slips else None,
            'note':'Quote observation, not executable fill or P&L'})
    return result

def write_csv(path,rows):
    if not rows:
        path.write_text('',encoding='utf8');return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def self_test():
    rows=[{'timestamp':100,'side':'BUY','price':'.45','size':'10','condition_id':'a','token_id':'x','outcome_index':0,'slug':'btc-updown-15m-0'},
          {'timestamp':101,'side':'BUY','price':'.46','size':'5','condition_id':'a','token_id':'x','outcome_index':0},
          {'timestamp':103,'side':'BUY','price':'.50','size':'10','condition_id':'a','token_id':'y','outcome_index':1}]
    signals=signals_from(rows);assert len(signals)==2
    result=analyze(rows,signals);assert result['both_sides_bought_markets']==1 and result['no_sell_both_sides_avg_price_lt_1_markets']==1
    print('PASS: clustering, BUY classification, two-sided inventory; NO network or trades')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wallets',nargs='+',choices=sorted(WALLETS),default=['goodfallen'])
    p.add_argument('--start',default='2026-11-06',help='UTC inclusive')
    p.add_argument('--end',default='2026-11-14',help='UTC exclusive')
    p.add_argument('--output-dir',type=Path,default=Path('data/research/goodfallen_20261113'))
    p.add_argument('--max-pages',type=int,default=1000)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--quotes-csv',type=Path)
    p.add_argument('--max-quote-lag',type=float,default=2)
    p.add_argument('--copy-usdc',type=float,default=10)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:self_test();return
    start=epoch(args.start);end=epoch(args.end)
    if start>=end or args.max_pages<1 or args.copy_usdc<=0 or args.max_quote_lag<0:p.error('Invalid window/pages/copy-USDC/quote lag')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summaries=[];all_signals=[];all_replay=[]
    for name in args.wallets:
        addr=WALLETS[name];path=args.output_dir/f'{name}_trades.jsonl';meta_path=args.output_dir/f'{name}_trades_meta.json'
        if args.offline:
            if not path.exists() or not meta_path.exists():raise FileNotFoundError(f'Missing complete cache for {name}')
            meta=json.loads(meta_path.read_text(encoding='utf8'))
        else:
            print(f'Downloading {name}: {addr}',flush=True)
            meta=download_trades(addr,start,end,path,args.max_pages)
            meta_path.write_text(json.dumps(meta,indent=2),encoding='utf8')
        if not meta.get('complete') or meta.get('start')!=start or meta.get('end_exclusive')!=end or meta.get('wallet')!=addr:
            print(f'WARNING {name}: incomplete/mismatched sample - skipped',flush=True);continue
        with path.open(encoding='utf8') as f:rows=[json.loads(line) for line in f if line.strip()]
        if len(rows)!=meta['rows']:raise RuntimeError('Cache row count mismatch')
        signals=signals_from(rows);stats=analyze(rows,signals)
        stats.update(wallet=name,address=addr,start_utc=args.start,end_utc_exclusive=args.end,complete=True,
                     pnl_status='NOT VERIFIED; use reconciled settlement and reward histories')
        summaries.append(stats);all_signals.extend(dict(wallet=name,**x) for x in signals)
        if args.quotes_csv:all_replay.extend(dict(wallet=name,**x) for x in quote_replay(signals,args.quotes_csv,[0,1,5,10],args.max_quote_lag,args.copy_usdc))
        print(f'{name}: {stats["fills"]:,} fills; {stats["markets"]:,} markets; {stats["buy_clusters"]:,} buy clusters; {stats["both_sides_bought_markets"]:,} two-sided markets',flush=True)
    report={'methodology':{'window_end':'exclusive UTC','source':'public Data API /v2/trades','pnl':'not established','pair_cost':'average prices, not matched-size arbitrage','latency':'historical quotes required; no execution assumption'},'wallets':summaries,'quote_replay':all_replay}
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    write_csv(args.output_dir/'signals.csv',all_signals);write_csv(args.output_dir/'replay.csv',all_replay)
    print('READ ONLY:',args.output_dir/'summary.json')

if __name__=='__main__':main()
