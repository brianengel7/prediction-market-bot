#!/usr/bin/env python3
"""Read-only forensic diagnostic for kalshi_sports_mm_v1.py capture.

Uses original replay Engine and the unchanged local tape. Counts quote eligibility,
actual trade touches, trade-through candidates, and displayed queue sizes.
Does NOT simulate executable P&L, modify the tape, connect to Kalshi or place orders.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

D = Decimal


def analyze(script, tape, ticker, *, queue_fraction=1.0, refresh_sec=60,
            size=5, min_pair_edge_cents=2.0, min_depth=10, max_unpaired=10):
    engine_class = runpy.run_path(str(script))['Engine']
    a = SimpleNamespace(ticker=ticker, queue_fraction=queue_fraction,
                        refresh_sec=refresh_sec, size=size, min_pair_edge_cents=min_pair_edge_cents,
                        min_depth=min_depth, max_unpaired=max_unpaired)
    e = engine_class(a)
    stat = Counter()
    volumes = defaultdict(lambda: D('0'))
    queue_init = defaultdict(list)
    first = last = None
    seen_trade_ids=set()
    hist = Counter()
    quote_times=[]
    z = D('0'); one=D('1')
    opener = gzip.open if str(tape).endswith('.gz') else open
    with opener(tape, 'rt', encoding='utf8') as f:
        for line in f:
            if not line.strip(): continue
            rec=json.loads(line)
            if rec.get('kind')=='meta':
                if rec.get('data',{}).get('ticker')!=ticker:
                    raise ValueError('Capture ticker differs from --ticker')
                continue
            now=float(rec.get('recv_ts',0))
            if first is None: first=now
            last=now
            kind=rec.get('kind')
            data=rec.get('data') or {}
            typ=data.get('type') if kind=='ws' else None
            msg=data.get('msg') or {}
            before_quotes={side:(q.price,q.queue_ahead,q.remaining,q.placed_ts)
                           for side,q in e.quotes.items()}
            prev_refresh=e.last_refresh
            # Inspect orders as they existed BEFORE the incoming trade.
            if typ=='trade' and msg.get('market_ticker')==ticker:
                trade_id=msg.get('trade_id')
                if trade_id and trade_id in seen_trade_ids:
                    stat['duplicate_trade_messages']+=1
                else:
                    if trade_id:seen_trade_ids.add(trade_id)
                    stat['unique_trades']+=1
                    side=msg.get('taker_outcome_side') or msg.get('taker_side')
                    maker={'yes':'no','no':'yes'}.get(side)
                    if maker and maker in before_quotes and not msg.get('is_block_trade'):
                        stat['trades_while_relevant_quote_active']+=1
                        quote_price, ahead, remaining, placed=before_quotes[maker]
                        price=D(str(msg.get(maker+'_price_dollars','0')))
                        qty=D(str(msg.get('count_fp','0')))
                        if now > placed and qty>0:
                            if price==quote_price:
                                stat['trades_exactly_at_quote']+=1
                                volumes['exact_trade_contracts']+=qty
                                if qty > ahead:stat['single_print_exceeds_remaining_queue']+=1
                            elif 0<price<quote_price:
                                stat['trade_through_better_bid_price']+=1
                                volumes['through_trade_contracts']+=qty
                            else:
                                stat['trade_price_above_quote']+=1
                    elif maker:
                        stat['trades_without_relevant_quote']+=1
            e.market(rec)
            if e.last_refresh!=prev_refresh and e.last_refresh > -1.e19:
                stat['refresh_attempts']+=1
                if e.book_valid:
                    y=e.best('yes');n=e.best('no')
                    if y is None or n is None: reason='missing_side'
                    elif e.midpoint() is None: reason='invalid_mid'
                    elif (one-y-n)*100<D(str(min_pair_edge_cents)): reason='pair_edge_too_small'
                    elif e.book['yes'][y]<D(str(min_depth)) or e.book['no'][n]<D(str(min_depth)):reason='low_displayed_depth'
                    else:reason='eligible'
                else:reason='unsynced'
                hist[reason]+=1
            for side,q in e.quotes.items():
                if side not in before_quotes or q.placed_ts!=before_quotes[side][3]:
                    queue_init[side].append(float(q.queue_ahead))
                    quote_times.append(now)
    result=e.report()
    from datetime import datetime,timezone
    def date(x):return datetime.fromtimestamp(x,timezone.utc).isoformat() if x is not None else None
    return {'mode':'OFFLINE_DIAGNOSTIC_NO_ORDERS','ticker':ticker,
            'capture_start_utc':date(first),'capture_end_utc':date(last),
            'capture_hours':round((last-first)/3600,3) if first is not None else None,
            'settings':{'queue_fraction':queue_fraction,'refresh_sec':refresh_sec,'min_pair_edge_cents':min_pair_edge_cents,'min_depth':min_depth},
            'refresh_checks_by_eligibility':dict(hist),
            'trade_touch_diagnostics':dict(stat),
            'trade_volume_contracts':{k:str(v) for k,v in volumes.items()},
            'initial_queue_ahead_contracts':{side:{'samples':len(vals),
                'median':sorted(vals)[len(vals)//2] if vals else None,
                'max':max(vals) if vals else None} for side,vals in queue_init.items()},
            'original_replay_fills':result['simulated_fill_events'],
            'original_replay_quotes':result['quotes_created'],
            'note':'Trade-through is a diagnostic (not a credited fill). Public tape cannot reconstruct cancellations by queue position, latency, or market impact.'}


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'test.jsonl.gz'
        ticker='KXNBAGAME-TEST'
        sample=[
            {'recv_ts':1,'kind':'meta','data':{'ticker':ticker}},
            {'recv_ts':2,'kind':'session_start','data':{}},
            {'recv_ts':3,'kind':'ws','data':{'type':'orderbook_snapshot','sid':1,'seq':1,'msg':{'market_ticker':ticker,'yes_dollars_fp':[['0.47','10']],'no_dollars_fp':[['0.50','10']]}}},
            {'recv_ts':4,'kind':'ws','data':{'type':'trade','msg':{'market_ticker':ticker,'trade_id':'a','taker_outcome_side':'no','taker_book_side':'ask','yes_price_dollars':'0.47','no_price_dollars':'0.53','count_fp':'3'}}},
            {'recv_ts':5,'kind':'ws','data':{'type':'trade','msg':{'market_ticker':ticker,'trade_id':'b','taker_outcome_side':'no','taker_book_side':'ask','yes_price_dollars':'0.46','no_price_dollars':'0.54','count_fp':'3'}}}
        ]
        with gzip.open(p,'wt') as f:
            for rec in sample:f.write(json.dumps(rec)+'\n')
        r=analyze(Path(__file__).parent/'kalshi_sports_mm_v1.py',p,ticker)
        assert r['trade_touch_diagnostics']['trades_exactly_at_quote']==1, r
        assert r['trade_touch_diagnostics']['trade_through_better_bid_price']==1, r
        assert r['refresh_checks_by_eligibility']['eligible']==1, r
        assert r['original_replay_fills']==0, r
        print('PASS: read-only diagnostic recognizes exact touches and trade-through; no orders')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--script',default='src/sports-mm/kalshi_sports_mm_v1.py')
    p.add_argument('--input')
    p.add_argument('--ticker')
    p.add_argument('--queue-fraction',type=float,choices=[0.,0.5,1.],default=1.)
    p.add_argument('--refresh-sec',type=int,default=60)
    p.add_argument('--min-pair-edge-cents',type=float,default=2.)
    p.add_argument('--min-depth',type=int,default=10)
    a=p.parse_args()
    if a.self_test:
        self_test();return
    if not a.input or not a.ticker:p.error('--input and --ticker required')
    r=analyze(Path(a.script),Path(a.input),a.ticker,queue_fraction=a.queue_fraction,
              refresh_sec=a.refresh_sec,min_pair_edge_cents=a.min_pair_edge_cents,min_depth=a.min_depth)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
