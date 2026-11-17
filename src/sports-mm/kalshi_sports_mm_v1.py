#!/usr/bin/env python3
"""Kalshi pregame sports maker research v1: discover, authenticated WSS record, offline replay.
READ-ONLY. No order endpoints. Requires requests, cryptography, websockets>=13 for recording.

Conservative replay: join CURRENT best bid on YES and NO. Queue ahead is public displayed
quantity at that price. Only subsequent *aggressor* trades at exactly our quoted price
reduce queue and potentially fill. No credit for cancellations. Requotes lose queue priority.
Neither fill modeling nor mark-to-market is a historically verified executable P&L.
"""
import argparse
import asyncio
import base64
import collections
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
import os
from pathlib import Path
import sys
import time

BASE = 'https://external-api.kalshi.com/trade-api/v2'
WS = 'wss://external-api-ws.kalshi.com/trade-api/ws/v2'
D = Decimal
ZERO = D('0')
ONE = D('1')


def num(x):
    return D(str(x))


def iso_to_ts(x):
    z = datetime.fromisoformat(x.replace('Z', '+00:00'))
    if z.tzinfo is None:
        raise ValueError('Timestamp MUST include timezone: YYYY-MM-DDTHH:MM:SSZ')
    return z.timestamp()


def public_get(path, params=None):
    import requests
    r = requests.get(BASE+path, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def discover(args):
    markets = []
    cursor = None
    for i in range(6):
        params = {'series_ticker':args.series, 'status':'open', 'limit':1000}
        if cursor:
            params['cursor'] = cursor
        obj = public_get('/markets', params)
        markets += obj.get('markets', [])
        cursor = obj.get('cursor')
        if not cursor: break
    # Do not choose the game start from market expiration/closing time.
    markets.sort(key=lambda x: (-float(x.get('volume_fp') or x.get('volume') or 0), x.get('ticker','')))
    print('Open', args.series, 'markets:',len(markets),'\n')
    for m in markets[:args.limit]:
        print(m.get('ticker'), '|', (m.get('title') or '')[:75], '| event=',m.get('event_ticker'),
              '| volume=',m.get('volume_fp',m.get('volume')),
              '| bid=',m.get('yes_bid_dollars'), '| ask=',m.get('yes_ask_dollars'))
    print('\nIMPORTANT: independently verify actual GAME TIPOFF, not market close time.')


def auth_headers():
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding, ed25519, rsa
    key_id = os.environ.get('KALSHI_API_KEY_ID')
    key_file = os.environ.get('KALSHI_PRIVATE_KEY_PATH')
    if not key_id or not key_file:
        raise RuntimeError('Recording requires existing KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH env variables. Use a read-only key if possible.')
    private = serialization.load_pem_private_key(Path(key_file).expanduser().read_bytes(), password=None)
    stamp = str(int(time.time()*1000))
    data = (stamp + 'GET' + '/trade-api/ws/v2').encode()
    if isinstance(private, ed25519.Ed25519PrivateKey):
        sig = private.sign(data)
    elif isinstance(private, rsa.RSAPrivateKey):
        sig = private.sign(data,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())
    else:
        raise TypeError('Unsupported private key algorithm')
    return {'KALSHI-ACCESS-KEY':key_id, 'KALSHI-ACCESS-TIMESTAMP':stamp,
            'KALSHI-ACCESS-SIGNATURE':base64.b64encode(sig).decode()}


async def record(args):
    from websockets.asyncio.client import connect
    cutoff = iso_to_ts(args.game_start_utc) - args.pregame_buffer
    start = time.time()
    end = min(start + args.minutes*60,cutoff)
    if end <= start + 20:
        raise ValueError('Start recording at least 20 seconds before pregame cutoff. Confirm GAME tipoff UTC.')
    ticker = args.ticker.strip().upper()
    if not ticker.startswith('KXNBAGAME-') and not args.allow_other_series:
        raise ValueError('V1 is restricted to KXNBAGAME. Use --allow-other-series only for other game-winner binary markets.')
    market = public_get('/markets/'+ticker)['market']
    if market.get('status') not in ('active','open'):
        raise ValueError(f'Market not open: status={market.get("status")}')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    print('READ-ONLY recorder |',ticker,'| pregame cutoff:',datetime.fromtimestamp(end,timezone.utc).isoformat())
    print('Output:',path,'| no trading or order endpoints exist in this script')
    count=0
    snapshot_count=0
    failed_connects=0
    def write(f,kind,data):
        nonlocal count
        f.write(json.dumps({'recv_ts':time.time(),'kind':kind,'data':data},separators=(',',':'))+'\n')
        count += 1
    with gzip.open(path, 'xt', encoding='utf8', compresslevel=5) as f:
        write(f,'meta',{'ticker':ticker,'market':market,'game_start_utc':args.game_start_utc,
                        'pregame_buffer':args.pregame_buffer,'record_started_ts':start,
                        'note':'READ ONLY. Game start supplied by user; no exchange orders.'})
        while time.time()<end:
            try:
                async with connect(WS, additional_headers=auth_headers(), ping_interval=15,ping_timeout=15,max_size=8*1024*1024) as ws:
                    write(f,'session_start',{'reason':'connect'})
                    await ws.send(json.dumps({'id':1,'cmd':'subscribe','params':{'channels':['orderbook_delta','trade'], 'market_tickers':[ticker]}}))
                    book_sid=None
                    last_seq={}
                    last_report=time.monotonic()
                    while time.time()<end:
                        try:
                            raw = await asyncio.wait_for(ws.recv(),timeout=min(1,max(.01,end-time.time())))
                        except asyncio.TimeoutError:
                            continue
                        data=json.loads(raw)
                        kind=data.get('type','unknown')
                        write(f,'ws',data)
                        if kind=='orderbook_snapshot':
                            snapshot_count+=1
                            failed_connects=0
                        if kind=='error':
                            raise ValueError('Kalshi WS subscription error: '+json.dumps(data.get('msg',data)))
                        if kind=='subscribed' and data.get('msg',{}).get('channel')=='orderbook_delta':
                            book_sid = data['msg'].get('sid')
                        if kind in ('orderbook_snapshot','orderbook_delta'):
                            sid=data.get('sid'); seq=data.get('seq')
                            if sid is not None and isinstance(seq,int):
                                if sid in last_seq and seq!=last_seq[sid]+1:
                                    write(f,'gap',{'sid':sid,'prev':last_seq[sid],'now':seq})
                                    if book_sid is not None:
                                        await ws.send(json.dumps({'id':1000+count,'cmd':'update_subscription',
                                          'params':{'sids':[book_sid],'action':'get_snapshot','market_tickers':[ticker]}}))
                                last_seq[sid]=seq
                        if time.monotonic()-last_report >= 15:
                            print('Recorded websocket frames:',count,flush=True)
                            f.flush();last_report=time.monotonic()
            except (KeyboardInterrupt,asyncio.CancelledError):
                raise
            except ValueError:
                raise  # Fatal API/subscription error: don't loop for the entire recording period.
            except Exception as ex:
                failed_connects+=1
                if failed_connects>=4 and snapshot_count==0:
                    raise RuntimeError('Could not obtain a WS orderbook snapshot after 4 attempts') from ex
                write(f,'disconnect',{'error':str(ex)[:300]})
                f.flush()
                print('WS reconnecting after:',str(ex)[:200],file=sys.stderr,flush=True)
                await asyncio.sleep(min(4,max(0,end-time.time())))
        write(f,'end',{'messages':count,'stopped_before_start':True})
    if snapshot_count==0:
        raise RuntimeError(f'NO orderbook snapshots recorded; inspect {path} and credentials')
    print('CAPTURE COMPLETE',path,'records:',count,'snapshots:',snapshot_count)


@dataclass
class Quote:
    side: str
    price: Decimal
    remaining: Decimal
    queue_ahead: Decimal
    placed_ts: float


class Engine:
    def __init__(self,args):
        self.a=args
        self.book={'yes':{},'no':{}}
        self.book_valid=False
        self.last_ts=0.
        self.last_refresh=-1.e20
        self.quotes={}
        self.pending_markouts=[]
        self.markouts=collections.defaultdict(list)
        self.fills=[]
        self.lots={'yes':collections.deque(),'no':collections.deque()}
        self.inventory={'yes':ZERO,'no':ZERO}
        self.total_cost=ZERO
        self.paired_qty=ZERO
        self.paired_profit=ZERO
        self.fills_count=0
        self.quote_count=0
        self.trade_count=0
        self.trade_invalid=0
        self.seq_gaps=0
        self.snapshots=0
        self.deltas=0
        self.last_seq={}
        self.trade_ids=set()
        self.skip_trade=0
        self.events=0
        self.stale_books=0

    def reset(self):
        self.book_valid=False
        self.quotes={}
        self.last_seq={}
        self.last_refresh=-1.e20

    def midpoint(self):
        if not self.book_valid:return None
        y=self.book['yes']; n=self.book['no']
        if not y or not n:return None
        by=max(y); bn=max(n)
        if by<=ZERO or bn<=ZERO or by+bn>=ONE:return None
        return (by+ONE-bn)/2

    def best(self,side):
        return max(self.book[side]) if self.book[side] else None

    def queue_quote(self,side,now):
        price=self.best(side)
        if price is None or price<=D('.10') or price>=D('.90'):return
        if self.inventory[side]-self.inventory['no' if side=='yes' else 'yes']>=num(self.a.max_unpaired):return
        self.quotes[side]=Quote(side,price,num(self.a.size),
                                self.book[side][price]*num(self.a.queue_fraction),now)
        self.quote_count+=1

    def refresh(self,now):
        if not self.book_valid or now-self.last_refresh<self.a.refresh_sec:return
        self.quotes={}
        self.last_refresh=now
        y=self.best('yes');n=self.best('no')
        if y is None or n is None or self.midpoint() is None:return
        if (ONE-y-n)*100 < num(self.a.min_pair_edge_cents):return
        if self.book['yes'][y]<num(self.a.min_depth) or self.book['no'][n]<num(self.a.min_depth):return
        self.queue_quote('yes',now)
        self.queue_quote('no',now)

    def market(self,rec):
        now=rec['recv_ts']; x=rec['data'];kind=rec['kind']
        self.last_ts=now
        if kind in ('session_start','gap','disconnect'):
            if kind=='gap':self.seq_gaps+=1
            self.reset();return
        if kind!='ws':return
        t=x.get('type'); msg=x.get('msg',{})
        if t not in ('orderbook_snapshot','orderbook_delta','trade'):
            return
        if msg.get('market_ticker')!=self.a.ticker:return
        if t in ('orderbook_snapshot','orderbook_delta'):
            sid=x.get('sid');seq=x.get('seq')
            if sid is not None and isinstance(seq,int):
                prev=self.last_seq.get(sid)
                if prev is not None and seq!=prev+1:
                    self.seq_gaps+=1;self.reset()
                    if t!='orderbook_snapshot':return
                self.last_seq[sid]=seq
        if t=='orderbook_snapshot':
            self.book={'yes':{},'no':{}}
            for side in ('yes','no'):
                pairs=msg.get(side+'_dollars_fp',msg.get(side+'_dollars',[]))
                for price,qty in pairs:
                    price=num(price);qty=num(qty)
                    if qty>ZERO and ZERO<price<ONE:self.book[side][price]=qty
            self.book_valid=True;self.quotes={};self.last_refresh=-1.e20
            self.snapshots+=1
        elif t=='orderbook_delta':
            if not self.book_valid:return
            side=msg.get('side')
            if side not in ('yes','no'):return
            price=num(msg.get('price_dollars',0)); delta=num(msg.get('delta_fp',0))
            if price<=ZERO or price>=ONE:return
            new=self.book[side].get(price,ZERO)+delta
            if new<ZERO:
                self.seq_gaps+=1;self.reset();return
            if new==ZERO:self.book[side].pop(price,None)
            else:self.book[side][price]=new
            self.deltas+=1
        elif t=='trade':
            self.trade_count+=1
            tid=msg.get('trade_id')
            if tid:
                if tid in self.trade_ids:return
                self.trade_ids.add(tid)
            if not self.book_valid:
                self.skip_trade+=1;return
            side=msg.get('taker_outcome_side') or msg.get('taker_side')
            action=msg.get('taker_book_side')
            # Explicit taker aggressor semantics. Unknown sides never infer fills.
            if side not in ('yes','no') or action not in ('ask','bid'):
                self.trade_invalid+=1;return
            # Kalshi: taker_book_side is just the outcome side in book vocabulary:
            # yes == bid, no == ask. It does NOT mean a buyer/seller aggressor action.
            if action != ('bid' if side=='yes' else 'ask'):
                self.trade_invalid+=1;return
            if msg.get('is_block_trade'):
                self.skip_trade+=1;return
            maker=('no' if side=='yes' else 'yes')
            p=num(msg.get(maker+'_price_dollars',0));qty=num(msg.get('count_fp',0))
            q=self.quotes.get(maker)
            if q and p==q.price and qty>ZERO and now>q.placed_ts:
                ahead=min(q.queue_ahead,qty)
                q.queue_ahead-=ahead
                available=qty-ahead
                if available>ZERO:
                    filled=min(available,q.remaining)
                    if filled>ZERO:self.fill(maker,q.price,filled,now)
                    q.remaining-=filled
                    if q.remaining<=ZERO:self.quotes.pop(maker,None)
        if t in ('orderbook_snapshot','orderbook_delta','trade'):
            self.refresh(now)
            self.update_markouts(now)

    def fill(self,side,price,qty,now):
        self.fills_count+=1
        self.inventory[side]+=qty
        self.total_cost+=price*qty
        self.fills.append({'ts':now,'side':side,'price':str(price),'qty':str(qty)})
        mid=self.midpoint()
        if mid is not None:
            self.pending_markouts.append({'ts':now,'side':side,'price':price,'mid':mid,'targets':set([5,30,60])})
        other='no' if side=='yes' else 'yes'
        own_qty=qty
        while own_qty>ZERO and self.lots[other]:
            lot_qty,lot_cost=self.lots[other][0]
            used=min(own_qty,lot_qty)
            self.paired_qty+=used
            self.paired_profit+=(ONE-price-lot_cost)*used
            own_qty-=used
            if used==lot_qty:self.lots[other].popleft()
            else:self.lots[other][0]=(lot_qty-used,lot_cost)
        if own_qty>ZERO:self.lots[side].append((own_qty,price))

    def update_markouts(self,now):
        mid=self.midpoint()
        if mid is None:return
        remain=[]
        for d in self.pending_markouts:
            for sec in sorted(tuple(d['targets'])):
                if now>=d['ts']+sec:
                    mark=(mid if d['side']=='yes' else ONE-mid)-d['price']
                    self.markouts[sec].append(float(mark*100))
                    d['targets'].remove(sec)
            if d['targets']:remain.append(d)
        self.pending_markouts=remain

    def report(self):
        yes_left=sum((qty for qty,_ in self.lots['yes']),ZERO)
        no_left=sum((qty for qty,_ in self.lots['no']),ZERO)
        assert yes_left==self.inventory['yes']-self.paired_qty
        assert no_left==self.inventory['no']-self.paired_qty
        y=self.best('yes');n=self.best('no')
        unpaired_mark=None
        # Liquidation mark is unavailable if the final book is not synchronized.
        if self.book_valid and y is not None and n is not None and self.midpoint() is not None:
            unpaired_mark=yes_left*y+no_left*n
        unpaired_cost=sum((q*p for side in ('yes','no') for q,p in self.lots[side]),ZERO)
        total_pnl=(self.paired_profit + unpaired_mark-unpaired_cost) if unpaired_mark is not None else None
        out={'mode':'READ_ONLY_OFFLINE_REPLAY','ticker':self.a.ticker,
             'assumptions':{'queue_fraction':self.a.queue_fraction,'quote_size':self.a.size,
                            'refresh_seconds':self.a.refresh_sec,'min_pair_edge_cents':self.a.min_pair_edge_cents,
                            'max_unpaired_contracts':self.a.max_unpaired,'min_depth':self.a.min_depth},
             'trade_prints':self.trade_count,'invalid_trade_direction':self.trade_invalid,
             'trades_while_book_unsynced':self.skip_trade,'book_seq_gaps':self.seq_gaps,
             'snapshots':self.snapshots,'book_deltas':self.deltas,'quotes_created':self.quote_count,
             'simulated_fill_events':self.fills_count,
             'filled_yes_contracts':str(self.inventory['yes']),'filled_no_contracts':str(self.inventory['no']),
             'paired_contracts':str(self.paired_qty),
             'paired_gross_pnl_usd':str(self.paired_profit),
             'unpaired_yes_contracts':str(yes_left),'unpaired_no_contracts':str(no_left),
             'unpaired_cost_usd':str(unpaired_cost),'unpaired_liquidation_mark_usd':str(unpaired_mark) if unpaired_mark is not None else None,
             'gross_purchased_notional_usd':str(self.total_cost),
             'total_mark_to_market_pnl_usd':str(total_pnl) if total_pnl is not None else None,
             'markouts_mean_cents_by_seconds':{str(k):round(sum(v)/len(v),4) if v else None for k,v in sorted(self.markouts.items())},
             'markouts_n_by_seconds':{str(k):len(v) for k,v in sorted(self.markouts.items())},
             'friction_stress_total_mark_pnl_usd':{},
             'warnings':['Never actual orders, queue position unobservable, and simulated prices do not include our market impact.',
                         'Conservative equality-only trade-through model; queue reductions from cancellations ignored.',
                         'Market midpoint used as pricing reference; no sportsbook fair-value anchor yet.',
                         'Pregame trades only; unmatched inventory marked at displayed best bid, not settled.',
                         'The replay uses local receive ordering, not guaranteed exchange causal ordering.',
                         'Markouts use first observed subsequent book quote at or beyond each horizon.']}
        for cents in ['0','0.25','0.5','1','2']:
            out['friction_stress_total_mark_pnl_usd'][cents] = (str(total_pnl - (self.inventory['yes']+self.inventory['no'])*num(cents)/100)
                 if total_pnl is not None else None)
        return out


def replay(args):
    eng=Engine(args)
    p=Path(args.input)
    with gzip.open(p,'rt',encoding='utf8') if str(p).endswith('.gz') else p.open('rt',encoding='utf8') as f:
        for i,line in enumerate(f,1):
            if not line.strip():continue
            rec=json.loads(line)
            if rec.get('kind')=='meta':
                if rec['data'].get('ticker')!=args.ticker:
                    raise ValueError('Tape ticker does not match --ticker')
                continue
            eng.market(rec)
    out=eng.report()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (output/'simulated_fills.csv').open('w') as f:
        f.write('ts,side,price,qty\n')
        for x in eng.fills:f.write(f"{x['ts']},{x['side']},{x['price']},{x['qty']}\n")
    print(json.dumps(out,indent=2))
    print('OUTPUT',output,'(summary.json, simulated_fills.csv)')


def self_test():
    class A:
        ticker='KXNBAGAME-TEST';refresh_sec=60;size=5;max_unpaired=10;min_pair_edge_cents=2;min_depth=10;queue_fraction=1
    a=A();eng=Engine(a)
    def event(ts,kind,msg=None,seq=None):
        return {'recv_ts':ts,'kind':kind,'data':{'type':kind,'sid':1,'seq':seq,'msg':dict(msg or {},market_ticker=a.ticker)}}
    eng.market({'recv_ts':1,'kind':'session_start','data':{}})
    eng.market({'recv_ts':2,'kind':'ws','data':event(2,'orderbook_snapshot',{'yes_dollars_fp':[['0.47','10']],'no_dollars_fp':[['0.50','10']]},1)['data']})
    assert set(eng.quotes)=={'yes','no'}
    assert eng.quotes['yes'].queue_ahead==10
    eng.market({'recv_ts':3,'kind':'ws','data':event(3,'trade',{'trade_id':'a','yes_price_dollars':'0.47','no_price_dollars':'0.53','count_fp':'8','taker_outcome_side':'no','taker_book_side':'ask'},None)['data']})
    assert eng.fills_count==0 and eng.quotes['yes'].queue_ahead==2
    eng.market({'recv_ts':4,'kind':'ws','data':event(4,'trade',{'trade_id':'b','yes_price_dollars':'0.47','no_price_dollars':'0.53','count_fp':'7','taker_outcome_side':'no','taker_book_side':'ask'},None)['data']})
    assert eng.inventory['yes']==5 and 'yes' not in eng.quotes
    eng.market({'recv_ts':5,'kind':'ws','data':event(5,'trade',{'trade_id':'c','yes_price_dollars':'0.50','no_price_dollars':'0.50','count_fp':'15','taker_outcome_side':'yes','taker_book_side':'bid'},None)['data']})
    assert eng.inventory['no']==5 and eng.paired_qty==5
    assert eng.paired_profit==D('.15')
    assert eng.report()['total_mark_to_market_pnl_usd']=='0.15'
    # Inconsistent taker flags and block trades can never count as resting fills.
    eng.market({'recv_ts':5.2,'kind':'ws','data':event(5.2,'trade',{'trade_id':'bad','yes_price_dollars':'0.47','no_price_dollars':'0.53','count_fp':'100','taker_outcome_side':'yes','taker_book_side':'ask'},None)['data']})
    assert eng.trade_invalid==1
    # Gap must cancel quotes without fabricating fills.
    eng.market({'recv_ts':6,'kind':'ws','data':event(6,'orderbook_delta',{'side':'yes','price_dollars':'.47','delta_fp':'-1'},4)['data']})
    assert not eng.book_valid
    # A fresh snapshot resynchronizes.
    eng.market({'recv_ts':7,'kind':'ws','data':event(7,'orderbook_snapshot',{'yes_dollars_fp':[['.46','15']],'no_dollars_fp':[['.51','12']]},5)['data']})
    assert eng.book_valid
    print('PASS: queue-ahead fills, taker direction, paired economics, seq-gap invalidation, resnapshot (NO ORDERS)')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('discover');a.add_argument('--series',default='KXNBAGAME');a.add_argument('--limit',type=int,default=20)
    a=sub.add_parser('record');a.add_argument('--ticker',required=True);a.add_argument('--minutes',type=float,default=90)
    a.add_argument('--game-start-utc',required=True,help='Actual GAME start time, not Kalshi market close time')
    a.add_argument('--pregame-buffer',type=int,default=300)
    a.add_argument('--output',required=True);a.add_argument('--allow-other-series',action='store_true')
    a=sub.add_parser('replay');a.add_argument('--input',required=True);a.add_argument('--ticker',required=True)
    a.add_argument('--output-dir',default='data/research/sports_mm_v1')
    a.add_argument('--refresh-sec',type=int,default=60);a.add_argument('--size',type=int,default=5)
    a.add_argument('--min-pair-edge-cents',type=float,default=2)
    a.add_argument('--max-unpaired',type=int,default=10);a.add_argument('--min-depth',type=int,default=10)
    a.add_argument('--queue-fraction',type=float,choices=[0,0.5,1],default=1,help='1: full visible queue (baseline); 0.5 or 0: optimistic sensitivity')
    sub.add_parser('self-test')
    args=p.parse_args()
    if args.cmd=='self-test':self_test()
    elif args.cmd=='discover':discover(args)
    elif args.cmd=='record':asyncio.run(record(args))
    elif args.cmd=='replay':replay(args)

if __name__=='__main__':main()
