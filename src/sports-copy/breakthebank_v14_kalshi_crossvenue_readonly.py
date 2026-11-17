#!/usr/bin/env python3
"""BreakTheBank V14: Polymarket -> Kalshi moneyline market discovery and historical
1-minute price PROXY. Read-only public GETs; NEVER submits orders.

Plan deliberately restricts to plain NBA/NFL game-winner slugs and uses the
public Polymarket CLOB's OUTCOME LABEL for the exact token. Kalshi candidate
series/event/team/date tickers are looked up directly, never assumed to exist.
NFL matches are REVIEW ONLY until settlement/tie terms are checked manually.
Historical Kalshi candle closes have NO executable depth. The candle audit is
NOT a backtest, realized PnL, or validation of latency-adjusted execution.
"""
from __future__ import annotations
import argparse, bisect, csv, json, math, re, time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

POLY='https://clob.polymarket.com'
KAL='https://external-api.kalshi.com/trade-api/v2'
SLUG=re.compile(r'^(nba|nfl)-([a-z0-9]+)-([a-z0-9]+)-(20\d{2}-\d{2}-\d{2})$',re.I)
CENTS=0.02
NBA={
'atl':'Atlanta Hawks|Hawks','bos':'Boston Celtics|Celtics','bkn':'Brooklyn Nets|Nets',
'cha':'Charlotte Hornets|Hornets','chi':'Chicago Bulls|Bulls','cle':'Cleveland Cavaliers|Cavaliers',
'dal':'Dallas Mavericks|Mavericks','den':'Denver Nuggets|Nuggets','det':'Detroit Pistons|Pistons',
'gsw':'Golden State Warriors|Warriors','hou':'Houston Rockets|Rockets','ind':'Indiana Pacers|Pacers',
'lac':'Los Angeles Clippers|Clippers','lal':'Los Angeles Lakers|Lakers','mem':'Memphis Grizzlies|Grizzlies',
'mia':'Miami Heat|Heat','mil':'Milwaukee Bucks|Bucks','min':'Minnesota Timberwolves|Timberwolves',
'nop':'New Orleans Pelicans|Pelicans','nyk':'New York Knicks|Knicks','okc':'Oklahoma City Thunder|Thunder',
'orl':'Orlando Magic|Magic','phi':'Philadelphia 76ers|76ers|Sixers','phx':'Phoenix Suns|Suns',
'por':'Portland Trail Blazers|Trail Blazers|Blazers','sac':'Sacramento Kings|Kings',
'sas':'San Antonio Spurs|Spurs','tor':'Toronto Raptors|Raptors','uta':'Utah Jazz|Jazz',
'was':'Washington Wizards|Wizards'}
NFL={
'ari':'Arizona Cardinals|Cardinals','atl':'Atlanta Falcons|Falcons','bal':'Baltimore Ravens|Ravens',
'buf':'Buffalo Bills|Bills','car':'Carolina Panthers|Panthers','chi':'Chicago Bears|Bears',
'cin':'Cincinnati Bengals|Bengals','cle':'Cleveland Browns|Browns','dal':'Dallas Cowboys|Cowboys',
'den':'Denver Broncos|Broncos','det':'Detroit Lions|Lions','gb':'Green Bay Packers|Packers',
'hou':'Houston Texans|Texans','ind':'Indianapolis Colts|Colts','jax':'Jacksonville Jaguars|Jaguars',
'kc':'Kansas City Chiefs|Chiefs','lv':'Las Vegas Raiders|Raiders','lac':'Los Angeles Chargers|Chargers',
'lar':'Los Angeles Rams|Rams','mia':'Miami Dolphins|Dolphins','min':'Minnesota Vikings|Vikings',
'ne':'New England Patriots|Patriots','no':'New Orleans Saints|Saints','nyg':'New York Giants|Giants',
'nyj':'New York Jets|Jets','phi':'Philadelphia Eagles|Eagles','pit':'Pittsburgh Steelers|Steelers',
'sf':'San Francisco 49ers|49ers','sea':'Seattle Seahawks|Seahawks','tb':'Tampa Bay Buccaneers|Buccaneers',
 'ten':'Tennessee Titans|Titans','was':'Washington Commanders|Commanders'}
TEAMS={'nba':NBA,'nfl':NFL}
KAL_CODES={'nfl':{'jax':'JAC'},'nba':{}}

def output_path(s):
    p=Path(s).expanduser().resolve()
    if str(p).startswith('/workspaces/') or p.exists():
        raise ValueError('Output must be a NEW file outside the repository')
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def get(base,path,params=None,*,not_found=False):
    url=base+path+('?' + urlencode(params) if params else '')
    req=Request(url,headers={'User-Agent':'btb-kalshi-crossvenue-research/1.0','Accept':'application/json'})
    for i in range(5):
        try:
            with urlopen(req,timeout=30) as f: return json.load(f)
        except HTTPError as e:
            if e.code==404 and not_found:return None
            if e.code not in (429,500,502,503,504) or i==4:raise
            time.sleep(min(12,2**i))
        except (URLError,TimeoutError):
            if i==4:raise
            time.sleep(min(12,2**i))
    raise RuntimeError('Unreachable')

def norm(s):return re.sub('[^a-z0-9]','',str(s).lower())

def team_from_outcome(label,league,a,b):
    catalog=TEAMS[league]
    candidate=[]
    for code in (a,b):
        aliases=[code]+catalog[code].split('|')
        if any(norm(label)==norm(x) for x in aliases):candidate.append(code)
    return candidate[0] if len(candidate)==1 else None

def extract_slug(rows):
    choices={str(x.get('slug','')).lower() for x in rows if x.get('slug')}
    return next(iter(choices)) if len(choices)==1 else None

def read_inputs(manifest,trades):
    intents=defaultdict(list)
    with Path(manifest).expanduser().open(newline='') as f:
        for r in csv.DictReader(f):
            intents[str(r['market']).lower()].append(r)
    if not intents:raise ValueError('No manifest intents')
    source=defaultdict(list)
    with Path(trades).expanduser().open() as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);cid=str(r.get('condition_id','')).lower()
            if cid in intents:source[cid].append(r)
    return intents,source

def calc_tickers(league,a,b,d,chosen):
    mapping=KAL_CODES[league]
    x,y,z=(mapping.get(k,k.upper()) for k in (a,b,chosen))
    dt=date.fromisoformat(d)
    series='KXNBAGAME' if league=='nba' else 'KXNFLGAME'
    # Game date may differ by one day between Polymarket UTC slug and Kalshi local event date.
    for delta in (0,-1,1):
        dd=dt+timedelta(days=delta)
        key=dd.strftime('%y%b%d').upper()
        for pair in (x+y,y+x):
            event=f'{series}-{key}{pair}'
            yield event,f'{event}-{z}',delta

def find_kalshi(league,a,b,d,chosen,cache):
    found={}
    for event,ticker,delta in calc_tickers(league,a,b,d,chosen):
        if ticker not in cache:
            m=get(KAL,'/markets/'+quote(ticker,safe=''),not_found=True)
            if m is None:m=get(KAL,'/historical/markets/'+quote(ticker,safe=''),not_found=True)
            cache[ticker]=m.get('market') if isinstance(m,dict) else None
            time.sleep(.08)
        m=cache[ticker]
        if not isinstance(m,dict):continue
        if m.get('ticker')!=ticker or m.get('event_ticker')!=event:
            raise RuntimeError('Kalshi response ticker/event mismatch: '+ticker)
        found[ticker]=(m,delta)
    return found

def plan(a):
    dest=output_path(a.out)
    intents,source=read_inputs(a.manifest,a.trades)
    rows=[]; reasons=Counter(); examined=0; kal_cache={}
    for cid,items in sorted(intents.items(),key=lambda kv:min(float(x['wallet_trade_ts_utc']) for x in kv[1])):
        slug=extract_slug(source[cid]); match=SLUG.fullmatch(slug or '')
        if not match:
            reasons['not_plain_nba_nfl_moneyline']+=1;continue
        league,left,right,day=match.groups();league=league.lower();left=left.lower();right=right.lower()
        if left==right or left not in TEAMS[league] or right not in TEAMS[league]:
            reasons['unknown_team_code']+=1;continue
        if examined >= a.max_candidates:break
        examined+=1
        try:
            poly=get(POLY,'/markets/'+quote(cid,safe=''),not_found=True)
            if not isinstance(poly,dict) or str(poly.get('condition_id','')).lower()!=cid:
                reasons['poly_cid_unverified']+=1;continue
            tmap={str(t.get('token_id')):str(t.get('outcome','')) for t in poly.get('tokens',[]) if isinstance(t,dict)}
            if len(tmap)!=2:
                reasons['poly_binary_token_unverified']+=1;continue
            tokens={str(x['token_id']) for x in items}
            if any(t not in tmap for t in tokens):
                reasons['poly_token_id_mismatch']+=1;continue
            mapped={t:team_from_outcome(tmap[t],league,left,right) for t in tokens}
            if any(not v for v in mapped.values()):
                reasons['poly_outcome_label_ambiguous']+=1;continue
            for tok,team in sorted(mapped.items()):
                matched=find_kalshi(league,left,right,day,team,kal_cache)
                if len(matched)!=1:
                    reasons['no_unique_kalshi_market']+=1;continue
                ticker,(market,delta)=next(iter(matched.items()))
                # NFL game-winner settlement and tied-game payoff require manual rule comparison.
                status='REVIEW_NFL_TIE_RULES' if league=='nfl' else 'REVIEW_NBA_RULES'
                for x in items:
                    if str(x['token_id'])!=tok:continue
                    rows.append(dict(poly_market=cid,poly_slug=slug,league=league,
                        poly_token=tok,poly_outcome=tmap[tok],team=team,
                        kalshi_ticker=ticker,kalshi_result=market.get('result',''),
                        kalshi_rules=market.get('rules_primary',''),
                        rules_status=status,date_shift=delta,intent_id=x['intent_id'],
                        ts=x['wallet_trade_ts_utc'],side=x['side'],shares=x['target_copy_shares'],
                        wallet_price=x['wallet_price']))
            print(f'checked {examined}/{a.max_candidates}: {slug} matched intents={sum(z["poly_market"]==cid for z in rows)}',flush=True)
        except (HTTPError,URLError,TimeoutError,ValueError) as exc:
            reasons['lookup_error']+=1
            print(f'LOOKUP ERROR {cid[:14]}: {type(exc).__name__}: {exc}',flush=True)
            continue
    cols=['poly_market','poly_slug','league','poly_token','poly_outcome','team','kalshi_ticker',
          'kalshi_result','kalshi_rules','rules_status','date_shift','intent_id','ts','side','shares','wallet_price']
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    print(f'KALSHI MATCH PLAN: {len(intents)} target markets, {examined} plain game-winner candidates checked,')
    print(f'{len(rows)} mapped intents, {len(set(x["poly_market"] for x in rows))} matched markets; exclusions={dict(reasons)}')
    print('Rules are NOT automatically certified; NFL tie rules require special review.')
    print('Output:',dest,'\nPUBLIC GET ONLY — NO ORDERS OR CREDENTIALS')

def candle_history(ticker,start,end):
    args=dict(start_ts=int(start)-120,end_ts=int(end)+60,period_interval=1)
    series=ticker.split('-')[0]
    live=get(KAL,f'/series/{series}/markets/{quote(ticker,safe="")}/candlesticks',args,not_found=True)
    historical=get(KAL,f'/historical/markets/{quote(ticker,safe="")}/candlesticks',args,not_found=True)
    result={}
    for data in (live,historical):
        if not isinstance(data,dict):continue
        for c in data.get('candlesticks',[]):
            t=c.get('end_period_ts')
            if not isinstance(t,(int,float)):continue
            bid=(c.get('yes_bid') or {}).get('close_dollars')
            ask=(c.get('yes_ask') or {}).get('close_dollars')
            def positive(v):
                try: v=float(v);return v if 0<v<1 else None
                except (TypeError,ValueError):return None
            b,z=positive(bid),positive(ask)
            if b is not None and z is not None and b>z+1e-10:continue
            if t in result and result[t]!=(b,z):
                raise RuntimeError('Conflicting Kalshi historical candle at '+str(ticker)+' '+str(t))
            result[int(t)]=(b,z)
    ts=sorted(result)
    return ts,result

def candles(a):
    dest=output_path(a.out)
    with Path(a.plan).expanduser().open(newline='') as f:records=list(csv.DictReader(f))
    if not records:raise ValueError('No Kalshi market matches in plan. Do not simulate unmatched markets.')
    by_ticker=defaultdict(list)
    for r in records:
        if r['rules_status']!='REVIEW_NBA_RULES' and not a.include_nfl:
            continue
        by_ticker[r['kalshi_ticker']].append(r)
    stats=Counter();out=[]
    for ticker,orders in by_ticker.items():
        try:
            st=min(float(x['ts']) for x in orders)+min(a.delays)
            end=max(float(x['ts']) for x in orders)+max(a.delays)
            times,quotes=candle_history(ticker,st,end)
        except (HTTPError,URLError,TimeoutError,RuntimeError) as exc:
            stats['candles_error']+=len(orders)
            print('candle fetch error',ticker,type(exc).__name__,str(exc)[:100],flush=True);continue
        for x in orders:
            for delay in a.delays:
                stats['intents']+=1
                target=float(x['ts'])+delay
                ix=bisect.bisect_right(times,target-a.candle_publish_lag)-1
                d=dict(x,delay_s=delay,market_price='',candle_end_ts='',candle_age_s='',
                       status='no_asof_candle',price_within_limit='')
                if ix>=0:
                    qt=times[ix];age=target-qt
                    d['candle_end_ts']=qt;d['candle_age_s']=round(age,2)
                    if age> a.max_candle_age:
                        d['status']='stale_candle'
                    else:
                        bid,ask=quotes[qt]
                        price=ask if x['side']=='BUY' else bid
                        if price is None:d['status']='missing_bid_or_ask'
                        else:
                            d['market_price']=price
                            ok=price<=float(x['wallet_price'])+a.limit_cents/100+1e-9 if x['side']=='BUY' else price>=float(x['wallet_price'])-a.limit_cents/100-1e-9
                            d['price_within_limit']=int(ok);d['status']='within_price_limit' if ok else 'outside_price_limit'
                stats[d['status']]+=1;out.append(d)
        print('Candles checked:',ticker,'order intents:',len(orders),flush=True)
    if out:
        with dest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    else:dest.write_text('status\nno_eligible_matches\n')
    print('HISTORICAL KALSHI 1-MIN CANDLE PRICE SCREEN:',dict(stats))
    print('No depth, no 5-second executable quotes, no realized P&L or trades.')
    print('Output:',dest,'\nPUBLIC GET ONLY — NO ORDERS OR CREDENTIALS')

def selftest():
    assert team_from_outcome('Green Bay Packers','nfl','dal','gb')=='gb'
    assert team_from_outcome('Cowboys','nfl','dal','gb')=='dal'
    assert team_from_outcome('Yes','nfl','dal','gb') is None
    assert team_from_outcome('Knicks','nba','nyk','bos')=='nyk'
    assert SLUG.fullmatch('nba-bos-det-2026-10-20')
    assert not SLUG.fullmatch('nba-bos-det-2026-10-20-spread-home-2pt5')
    tickers=list(calc_tickers('nba','bos','det','2026-10-20','bos'))
    assert tickers[0][:2]==('KXNBAGAME-26OCT20BOSDET','KXNBAGAME-26OCT20BOSDET-BOS')
    assert len(tickers)==6
    print('PASS: exact outcome mapping, no spread/totals matching, date candidate generation; OFFLINE')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('self-test')
    d=sub.add_parser('plan')
    d.add_argument('--manifest',default=str(Path.home()/'sports_copy_research/breakthebank_v8/intents.csv'))
    d.add_argument('--trades',default=str(Path.home()/'sports_copy_research/breakthebank_v1/all_trades.jsonl'))
    d.add_argument('--max-candidates',type=int,default=15)
    d.add_argument('--out',required=True)
    c=sub.add_parser('candles');c.add_argument('--plan',required=True);c.add_argument('--out',required=True)
    c.add_argument('--delays',default='5,15,30,60');c.add_argument('--limit-cents',type=float,default=2.)
    c.add_argument('--max-candle-age',type=float,default=75.)
    c.add_argument('--candle-publish-lag',type=float,default=10.,help='Conservative delay after minute end before treating candle close as observable')
    c.add_argument('--include-nfl',action='store_true',help='Exploratory only: NFL tie rule equivalence NOT verified')
    a=p.parse_args()
    if a.cmd=='self-test':selftest()
    elif a.cmd=='plan':
        if a.max_candidates<1 or a.max_candidates>150:p.error('--max-candidates must be 1..150')
        plan(a)
    else:
        a.delays=[int(x) for x in a.delays.split(',')]
        if any(x<0 or x>86400 for x in a.delays) or not a.delays or a.limit_cents<0 or a.max_candle_age<0 or a.candle_publish_lag<0:p.error('Invalid timing / price arguments')
        candles(a)
if __name__=='__main__':main()
