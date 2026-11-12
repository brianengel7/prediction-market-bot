#!/usr/bin/env python3
"""Wallet 954B -> Kalshi BTC15M copier, v1 + first-seen feed telemetry. Paper DEFAULT; gated LIVE option.

Strategy frozen from historical research:
- A wallet's FIRST BUY in a BTC 15m market must be 0.20 <= price < 0.40.
- Buy 10 Kalshi UP=YES / DOWN=NO at observed, executable book price.
- First same-outcome repeat 4..30 seconds after first triggers proportional increase.
- Continue all same-outcome BUYs; cumulative wallet shares / first shares sets
  cumulative desired Kalshi contracts (round half up).
- Stop adding at first opposite-outcome BUY or ANY SELL; hold bought inventory
  until exchange settlement. No share-count strategy cap, but mandatory live risk
  dollar limits, maximum quote drift, and liquidity checks.

No arbitrary market/ticker inference, no historical catch-up orders, no live
orders unless both CLI and environment expressly enable them. All processed
wallet records and submissions persist to a dedicated SQLite database.

Install: python -m pip install requests cryptography
Run:     python src/copy-trading/w954_live_copy_v1.py --self-test
         python -u src/copy-trading/w954_live_copy_v1.py --mode paper

WARNING: Public Polymarket wallet feed latency and missing/fragmented fills
are not established; paper results need reconciliation before any real trade.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal as D, ROUND_HALF_UP, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import time
import uuid
from urllib.parse import urlparse

WALLET = '0x954b91dd68eca3aa4ea603ecea83c01e8539c0e0'
POLY_URL = 'https://data-api.polymarket.com/v2/trades'
KALSHI_URL = 'https://external-api.kalshi.com/trade-api/v2'
DEFAULT_DB = '.collector_runs/w954_live_copy_v1.sqlite3'
SLUG = re.compile(r'btc-updown-15m-(\d{10})(?:$|\b)', re.IGNORECASE)
ONE = D('1')


def log(what, **kw):
    print(datetime.now(timezone.utc).isoformat(timespec='milliseconds'), what,
          json.dumps(kw, default=str, sort_keys=True), flush=True)


def money(v):
    d = D(str(v))
    if not d.is_finite():
        raise ValueError('Non-finite price/size')
    return d


def epoch(v):
    if isinstance(v, (int, float)):
        n=float(v)
    elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
        n=float(v)
    elif isinstance(v, str):
        return datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()
    else:
        raise ValueError('Missing event timestamp')
    if n > 1e17: n /= 1e9
    elif n > 1e14: n /= 1e6
    elif n > 1e11: n /= 1e3
    return n


def round_target(first_contracts, cumulative_shares, anchor_shares):
    if anchor_shares <= 0 or cumulative_shares < anchor_shares:
        raise ValueError('Bad wallet sizing anchor')
    return int((D(first_contracts)*cumulative_shares/anchor_shares)
               .quantize(D('1'), rounding=ROUND_HALF_UP))


def parse_trade(row):
    if not isinstance(row, dict):
        return None
    slug = next((row.get(k) for k in ('slug','event_slug','eventSlug','market_slug')
                 if isinstance(row.get(k), str) and SLUG.search(row[k])), None)
    if slug is None:
        return None
    start = int(SLUG.search(slug).group(1))
    user = next((row[k] for k in ('proxy_wallet','proxyWallet','user','wallet')
                 if row.get(k)), None)
    if user is not None and str(user).lower() != WALLET:
        raise RuntimeError('Polymarket source wallet mismatch')
    ts = epoch(next((row[k] for k in ('timestamp','ts','time','created_at')
                     if row.get(k) is not None), None))
    side = str(row.get('outcome', row.get('outcome_name',''))).upper()
    action = str(row.get('side', row.get('action', ''))).upper()
    if side not in ('UP','DOWN') or action not in ('BUY','SELL'):
        raise RuntimeError(f'Unknown BTC wallet action/outcome: {action} {side}')
    price, qty = money(row.get('price')), money(row.get('size', row.get('shares')))
    if not (D(0)<price<ONE and qty>0 and start<=ts<start+900):
        raise RuntimeError('Invalid BTC wallet trade price/quantity/time')
    uid = next((str(row[k]) for k in ('trade_id','tradeId','id') if row.get(k)), None)
    # Some feeds expose an on-chain transaction + log index instead of trade_id.
    if not uid:
        tx = row.get('transaction_hash', row.get('transactionHash'))
        seq = row.get('log_index', row.get('logIndex'))
        if tx is not None and seq is not None:
            uid = f'{tx}:{seq}'
    if not uid:
        # Identical payloads cannot be distinguished; caller must reject
        # collisions (not use row ordering for a live trade identity).
        uid = 'payload:' + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    uid = str(start) + ':' + uid
    return {'id': uid, 'start': start, 'ts': ts, 'side': side, 'action': action,
            'price': price, 'size': qty}


def top_levels(orderbook, buy_side):
    """Convert opposite-side BID levels to executable asks for YES or NO."""
    book = orderbook.get('orderbook_fp')
    if not isinstance(book, dict):
        raise RuntimeError('Orderbook response missing orderbook_fp')
    opposite = 'no_dollars' if buy_side=='UP' else 'yes_dollars'
    raw = book.get(opposite)
    if not isinstance(raw, list):
        raise RuntimeError('Orderbook missing price levels')
    levels = []
    for pair in raw:
        bid, qty = map(money, pair)
        if not (D(0)<bid<ONE and qty>=0):
            raise RuntimeError('Invalid Kalshi orderbook level')
        if qty:
            levels.append((ONE-bid,qty))
    return sorted(levels)


def executable_quote(book, side, qty):
    levels=top_levels(book, side)
    remain=D(qty)
    total=D(0)
    last=None
    for ask, size in levels:
        take=min(remain,size)
        total+=take*ask
        remain-=take
        last=ask
        if remain<=0:
            return {'limit_side_price': last if side=='UP' else ONE-last,
                    'best_ask': levels[0][0], 'worst_ask': last,
                    'vwap': total/D(qty), 'gross': total,
                    'visible_top_qty': levels[0][1], 'qty': qty}
    return None


def init_db(path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(p)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=5000')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS runtime (name TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS seen (trade_id TEXT PRIMARY KEY, market_start INTEGER NOT NULL, first_seen REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS markets (
      market_start INTEGER PRIMARY KEY, side TEXT, first_ts REAL,
      anchor TEXT, wallet_total TEXT, planned INTEGER DEFAULT 0,
      filled INTEGER DEFAULT 0, spent TEXT DEFAULT '0', confirmed INTEGER DEFAULT 0,
      stopped INTEGER DEFAULT 0, reason TEXT DEFAULT '', ticker TEXT, last_wallet_ts REAL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS orders (
      client_id TEXT PRIMARY KEY, market_start INTEGER NOT NULL, trade_id TEXT NOT NULL,
      created_at REAL NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,
      qty INTEGER NOT NULL, price TEXT NOT NULL, est_spent TEXT NOT NULL,
      status TEXT NOT NULL, filled INTEGER DEFAULT 0, actual_spent TEXT DEFAULT '0',
      order_id TEXT DEFAULT '');
    CREATE TABLE IF NOT EXISTS observations (
      ts REAL, trade_id TEXT, market_start INTEGER, age_s REAL,
      kind TEXT, side TEXT, qty INTEGER, wallet_price TEXT,
      kalshi_ask TEXT, depth TEXT, detail TEXT);
    -- Added by feed-latency instrumentation; does not modify existing seen,
    -- markets, orders, or positions. Only FIRST observation is recorded.
    CREATE TABLE IF NOT EXISTS feed_first_seen (
      trade_id TEXT PRIMARY KEY,
      market_start INTEGER NOT NULL,
      wallet_ts REAL NOT NULL,
      first_seen_at REAL NOT NULL,
      age_s REAL NOT NULL,
      side TEXT NOT NULL,
      action TEXT NOT NULL,
      wallet_price TEXT NOT NULL,
      wallet_qty TEXT NOT NULL,
      startup INTEGER NOT NULL CHECK(startup IN (0,1)),
      age_limit_s REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS feed_first_seen_time_idx
      ON feed_first_seen(first_seen_at, startup);
    ''')
    return db


def market_row(db,start):
    return db.execute('SELECT market_start, side, first_ts, anchor, wallet_total, planned, filled, spent, confirmed, stopped, reason, ticker, last_wallet_ts FROM markets WHERE market_start=?',(start,)).fetchone()


def mark_stop(db,start,reason):
    db.execute('UPDATE markets SET stopped=1,reason=? WHERE market_start=?',(reason,start))
    db.commit()
    log('STOP_MARKET', start=start, reason=reason)


def get_json(url, params=None, timeout=5):
    import requests
    r=requests.get(url,params=params,headers={'Accept':'application/json','User-Agent':'w954-live-copy/1.0'},timeout=timeout)
    r.raise_for_status()
    payload=r.json()
    if not isinstance(payload,dict):
        raise RuntimeError('Unexpected response JSON')
    return payload


def fetch_wallet(now, pages=8):
    # Recent rolling read; if more than eight pages appear, FAIL CLOSED.
    query={'user':WALLET,'start':int(now)-1100,'end':int(now)+2,
           'limit':500,'taker_only':'false'}
    records=[]; seen=set()
    for _ in range(pages):
        payload=get_json(POLY_URL,query,timeout=8)
        data=payload.get('data'); pg=payload.get('pagination')
        if not isinstance(data,list) or not isinstance(pg,dict):
            raise RuntimeError('Unrecognized Polymarket v2 trade feed')
        records.extend(data)
        next_cursor=pg.get('next_cursor') or pg.get('nextCursor')
        if not next_cursor:
            if pg.get('has_more') is True:
                raise RuntimeError('Missing Polymarket pagination cursor')
            return records
        if next_cursor in seen:
            raise RuntimeError('Stalled Polymarket cursor')
        seen.add(next_cursor)
        query['cursor']=next_cursor
    raise RuntimeError('Polymarket v2 trade feed exceeded page cap')


class Kalshi:
    def __init__(self, live):
        self.live=live
        self.base=KALSHI_URL
        self.key=None
        if live:
            from cryptography.hazmat.primitives import serialization
            path=Path(os.environ['KALSHI_PRIVATE_KEY_PATH']).expanduser()
            self.key=serialization.load_pem_private_key(path.read_bytes(),password=None)
            self.key_id=os.environ['KALSHI_API_KEY_ID']

    def headers(self, method,path):
        import cryptography.hazmat.primitives.asymmetric.ed25519 as ed
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes
        stamp=str(int(time.time()*1000))
        msg=(stamp+method+urlparse(self.base+path).path).encode()
        if isinstance(self.key,ed.Ed25519PrivateKey): sig=self.key.sign(msg)
        else:
            sig=self.key.sign(msg,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                         salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())
        return {'KALSHI-ACCESS-KEY':self.key_id,
                'KALSHI-ACCESS-TIMESTAMP':stamp,
                'KALSHI-ACCESS-SIGNATURE':base64.b64encode(sig).decode(),
                'Content-Type':'application/json'}

    def market_for_start(self,start):
        markets=[]; cursor=None
        for i in range(4):
            q={'series_ticker':'KXBTC15M','status':'open','limit':200}
            if cursor:q['cursor']=cursor
            data=get_json(self.base+'/markets',q)
            for m in data.get('markets',[]):
                close=m.get('close_time')
                if close:
                    try:t=epoch(close)
                    except (ValueError,TypeError):continue
                    if abs(t-(start+900))<1:
                        markets.append(m)
            cursor=data.get('cursor')
            if not cursor:break
        if len(markets)!=1:
            raise RuntimeError(f'Expected exactly one live Kalshi market at {start+900}; found {len(markets)}')
        m=markets[0]
        if not str(m.get('ticker','')).startswith('KXBTC15M-'):
            raise RuntimeError('Unexpected market series')
        return m['ticker']

    def book(self,ticker):
        return get_json(self.base+'/markets/'+ticker+'/orderbook',timeout=4)

    def balance(self):
        import requests
        path='/portfolio/balance'
        r=requests.get(self.base+path,headers=self.headers('GET',path),timeout=5)
        r.raise_for_status(); data=r.json()
        return money(data['balance_dollars'])

    def send_fok(self,ticker,side,quantity,book_limit,client_id):
        """V2 bid buys YES at YES price; ask buys NO at YES-scale price."""
        import requests
        path='/portfolio/events/orders'
        body={'ticker':ticker,'side':'bid' if side=='UP' else 'ask',
              'count':f'{quantity}.00','price':f'{book_limit:.4f}',
              'time_in_force':'fill_or_kill','client_order_id':client_id,
              'self_trade_prevention_type':'taker_at_cross',
              'cancel_order_on_pause':True}
        try:
            r=requests.post(self.base+path,json=body,headers=self.headers('POST',path),timeout=5)
        except requests.RequestException as e:
            raise RuntimeError('UNKNOWN_ORDER_SUBMISSION: '+str(e)) from e
        if r.status_code in (408,409,429) or r.status_code>=500:
            raise RuntimeError('UNKNOWN_ORDER_SUBMISSION: status '+str(r.status_code))
        if r.status_code!=201:
            raise RuntimeError(f'ORDER_REJECTED HTTP {r.status_code}: {r.text[:200]}')
        data=r.json(); actual=D(str(data.get('fill_count','-1')))
        if actual!=quantity:
            raise RuntimeError(f'UNKNOWN_OR_PARTIAL_FILL: {data}')
        return data

    def verified_fill(self,order_id,expected):
        import requests
        path='/portfolio/orders/'+str(order_id)
        # Read back the authoritative order state. An unsuccessful verification
        # is an UNKNOWN order, never a reason to submit a replacement.
        r=requests.get(self.base+path,headers=self.headers('GET',path),timeout=5)
        r.raise_for_status()
        order=r.json()['order']
        filled=money(order['fill_count_fp'])
        remaining=money(order['remaining_count_fp'])
        if filled!=expected or remaining!=0:
            raise RuntimeError(f'Unexpected exchange order fill: filled={filled} remaining={remaining}')
        return order


def record_obs(db, trade, age, kind, q=None, qty=0, detail=''):
    db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?)',
       (time.time(),trade['id'],trade['start'],age,kind,trade['side'],qty,
        str(trade['price']),str(q['vwap']) if q else None,
        str(q['visible_top_qty']) if q else None,detail))
    db.commit()


def place_copy(db,k,args,tr,st,qty,kind):
    """One attempt only. If uncertain order result: fail closed globally."""
    now=time.time(); age=now-tr['ts']
    if age< -2 or age>args.max_age_seconds:
        record_obs(db,tr,age,'STALE',qty=qty,detail=kind)
        mark_stop(db,tr['start'],'stale_wallet_signal')
        return False
    if now>=tr['start']+900-args.close_buffer_seconds:
        record_obs(db,tr,age,'TOO_LATE',qty=qty,detail=kind)
        mark_stop(db,tr['start'],'too_close_to_settlement')
        return False
    try:
        ticker=st['ticker'] or k.market_for_start(tr['start'])
        quote=executable_quote(k.book(ticker),tr['side'],qty)
        if not quote:
            record_obs(db,tr,age,'INSUFFICIENT_BOOK_DEPTH',qty=qty,detail=kind)
            mark_stop(db,tr['start'],'insufficient_depth')
            return False
        # Check the worst executable price for the ENTIRE FOK order, not just its VWAP.
        if quote['worst_ask'] > args.max_ask or quote['worst_ask'] > tr['price']+args.max_drift:
            record_obs(db,tr,age,'ASK_TOO_HIGH',q=quote,qty=qty,detail=kind)
            mark_stop(db,tr['start'],'unfavorable_live_ask')
            return False
        # Flat per-contract upper fee bound; pessimistic vs 0.07*p*(1-p).
        est=quote['gross']+(D('0.018')*D(qty)).quantize(D('.01'),rounding=ROUND_CEILING)
        market_spent=money(st['spent'])
        day_key=datetime.now(timezone.utc).strftime('%Y-%m-%d')
        day= db.execute("SELECT COALESCE(SUM(CAST(est_spent AS REAL)),0) FROM orders WHERE strftime('%Y-%m-%d', created_at, 'unixepoch')=? AND status IN ('PENDING','FILLED','PAPER')",(day_key,)).fetchone()[0]
        if est>args.max_order_spend or market_spent+est>args.max_market_spend or money(day)+est>args.max_day_spend:
            record_obs(db,tr,age,'RISK_BUDGET',q=quote,qty=qty,detail=kind)
            mark_stop(db,tr['start'],'risk_budget')
            return False
        if args.mode=='live' and k.balance()<est:
            record_obs(db,tr,age,'INSUFFICIENT_BALANCE',q=quote,qty=qty,detail=kind)
            mark_stop(db,tr['start'],'balance')
            return False
    except Exception as exc:
        log('QUOTE_OR_BALANCE_ERROR',start=tr['start'],error=str(exc))
        mark_stop(db,tr['start'],'quote_error')
        return False
    # At this point paper will assume the FOK book price for modelling, but
    # only actual returned live fill_count is allowed to create live holdings.
    client_id=str(uuid.uuid4())
    status='PAPER' if args.mode=='paper' else 'PENDING'
    db.execute('INSERT INTO orders (client_id,market_start,trade_id,created_at,ticker,side,qty,price,est_spent,status) VALUES(?,?,?,?,?,?,?,?,?,?)',
       (client_id,tr['start'],tr['id'],time.time(),ticker,tr['side'],qty,
        str(quote['limit_side_price']),str(est),status))
    db.commit()   # Durable BEFORE external POST; uncertain orders are not retried.
    filled=qty
    if args.mode=='live':
        try:
            response=k.send_fok(ticker,tr['side'],qty,quote['limit_side_price'],client_id)
            order_id=str(response.get('order_id',''))
            if not order_id: raise RuntimeError('UNKNOWN_ORDER_ID: '+str(response))
            verified=k.verified_fill(order_id,qty)
            actual_cost=sum((money(verified.get(field,'0')) for field in
                ('taker_fill_cost_dollars','maker_fill_cost_dollars',
                 'taker_fees_dollars','maker_fees_dollars')),D(0))
            if actual_cost>0:
                est=actual_cost
        except Exception as exc:
            db.execute('UPDATE orders SET status=? WHERE client_id=?',('UNKNOWN',client_id));db.commit()
            log('CRITICAL_ORDER_UNKNOWN',id=client_id,error=str(exc))
            raise RuntimeError('FAIL-CLOSED: reconcile Kalshi order manually before restart') from exc
    else:
        order_id='PAPER'
    db.execute('UPDATE orders SET status=?,filled=?,actual_spent=?,order_id=? WHERE client_id=?',
       ('FILLED' if args.mode=='live' else 'PAPER',filled,str(est),order_id,client_id))
    db.execute('UPDATE markets SET filled=filled+?,spent=?,ticker=? WHERE market_start=?',
       (filled,str(money(st['spent'])+est),ticker,tr['start']))
    db.commit()
    record_obs(db,tr,age,'SIMULATED_FILL' if args.mode=='paper' else 'LIVE_FILL',q=quote,qty=qty,detail=kind)
    log('COPY_ORDER',mode=args.mode,kind=kind,market=tr['start'],ticker=ticker,side=tr['side'],
        wallet_price=str(tr['price']),wallet_age_s=round(age,3),qty=qty,
        ask=str(quote['worst_ask']),vwap=str(quote['vwap']),cost_cap=str(est),id=client_id)
    return True


def handle_trade(db,k,args,tr):
    start=tr['start']; now=time.time(); age=now-tr['ts']
    st=market_row(db,start)
    if st is None:
        # Always identify FIRST action chronologically; no hindsight entry.
        qualifies=tr['action']=='BUY' and D('.20')<=tr['price']<D('.40')
        side=tr['side'] if qualifies else None
        db.execute('INSERT INTO markets (market_start,side,first_ts,anchor,wallet_total,reason,stopped) VALUES(?,?,?,?,?,?,?)',
                   (start,side,tr['ts'] if qualifies else None,str(tr['size']) if qualifies else '0',
                    str(tr['size']) if qualifies else '0', '' if qualifies else 'first_trade_ineligible',0 if qualifies else 1))
        db.commit();st=market_row(db,start)
        if not qualifies:
            return
        if place_copy(db,k,args,tr,dict(zip(('market_start','side','first_ts','anchor','wallet_total','planned','filled','spent','confirmed','stopped','reason','ticker','last_wallet_ts'),st)),args.base_qty,'first'):
            db.execute('UPDATE markets SET planned=? WHERE market_start=?',(args.base_qty,start));db.commit()
        return
    cols=('market_start','side','first_ts','anchor','wallet_total','planned','filled','spent','confirmed','stopped','reason','ticker','last_wallet_ts')
    st=dict(zip(cols,st))
    if st['stopped'] or st['side'] is None:
        return
    if tr['ts']<=st['first_ts']:
        mark_stop(db,start,'late_or_ambiguous_first_second_fill')
        return
    if tr['ts'] < st['last_wallet_ts'] - .000001:
        mark_stop(db,start,'out_of_order_wallet_feed')
        return
    db.execute('UPDATE markets SET last_wallet_ts=? WHERE market_start=?',(tr['ts'],start));db.commit()
    if tr['action']=='SELL':
        mark_stop(db,start,'wallet_sell_seen')
        return
    if tr['side']!=st['side']:
        mark_stop(db,start,'wallet_opposite_buy')
        return
    total=money(st['wallet_total'])+tr['size']
    db.execute('UPDATE markets SET wallet_total=? WHERE market_start=?',(str(total),start));db.commit()
    elapsed=tr['ts']-st['first_ts']
    confirmed=bool(st['confirmed'])
    if not confirmed and not (4<=elapsed<=30):
        if elapsed>30:
            mark_stop(db,start,'confirmation_window_expired')
        return
    if not confirmed:
        db.execute('UPDATE markets SET confirmed=1 WHERE market_start=?',(start,));db.commit()
    target=round_target(args.base_qty,total,money(st['anchor']))
    delta=target-int(st['planned'])
    if delta<=0:
        return
    # Bookkeeping is based on the WALLET target, never a retry of earlier unfilled orders.
    db.execute('UPDATE markets SET planned=? WHERE market_start=?',(target,start));db.commit()
    if int(st['filled'])==0:
        mark_stop(db,start,'first_order_not_filled')
        return
    place_copy(db,k,args,tr,st,delta,'confirmation' if not confirmed else 'add')


def pct(values, p):
    """Linear-interpolated percentile in seconds, including small N."""
    if not values:
        return None
    ordered=sorted(values)
    rank=(len(ordered)-1)*p/100.0
    low=int(rank)
    high=min(low+1,len(ordered)-1)
    return round(ordered[low]+(ordered[high]-ordered[low])*(rank-low),3)


def feed_age_summary(db, since=None):
    """Unique BTC fill ages at FIRST observation, excluding startup backlog.

    The observation includes polling interval, HTTP transfer, and clock offset.
    This is NOT a direct server-side data publication latency measurement.
    """
    query='SELECT age_s,age_limit_s FROM feed_first_seen WHERE startup=0'
    params=[]
    if since is not None:
        query+=' AND first_seen_at>=?'
        params.append(since)
    rows=db.execute(query,params).fetchall()
    good=[float(age) for age,_ in rows if 0 <= float(age) < 86400]
    skew=sum(float(age)<0 for age,_ in rows)
    bad=sum(float(age)>=86400 for age,_ in rows)
    total=len(good)
    return dict(
        unique_post_startup=len(rows),valid_ages=total,
        negative_clock_skew=skew,implausibly_old=bad,
        median_age_s=pct(good,50),p90_age_s=pct(good,90),
        p95_age_s=pct(good,95),max_age_s=round(max(good),3) if good else None,
        within_8s_pct=round(100*sum(x<=8 for x in good)/total,1) if total else None,
        within_15s_pct=round(100*sum(x<=15 for x in good)/total,1) if total else None,
        within_20s_pct=round(100*sum(x<=20 for x in good)/total,1) if total else None,
        within_configured_pct=(round(100*sum(0<=float(age)<=float(limit) for age,limit in rows)/total,1)
                               if total else None),
    )


def feed_latency_report(db_path):
    """Offline SQLite report: no Polymarket requests, no order submissions."""
    db=sqlite3.connect(f'file:{Path(db_path).resolve()}?mode=ro',uri=True)
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='feed_first_seen'").fetchone():
            print('No feed latency table yet. Start the updated bot in paper mode first.')
            return
        startup=db.execute('SELECT COUNT(*) FROM feed_first_seen WHERE startup=1').fetchone()[0]
        print('954B UNIQUE-FIRST-OBSERVED WALLET TRADE AGE — READ ONLY')
        print('Startup backlog excluded from latency statistics:',startup)
        print('ALL NEWLY FIRST-OBSERVED BTC TRADES:',json.dumps(feed_age_summary(db),sort_keys=True))
        print('MOST RECENT 60 MINUTES:',json.dumps(feed_age_summary(db,time.time()-3600),sort_keys=True))
        print('LATEST 12 POST-STARTUP WALLET TRADES:')
        for ts,wallet_ts,age,action,side,price,market in db.execute("""
              SELECT first_seen_at,wallet_ts,age_s,action,side,wallet_price,market_start
              FROM feed_first_seen WHERE startup=0
              ORDER BY first_seen_at DESC LIMIT 12"""):
            print(' observed=',datetime.fromtimestamp(ts,timezone.utc).isoformat(timespec='milliseconds'),
                  ' wallet=',datetime.fromtimestamp(wallet_ts,timezone.utc).isoformat(timespec='seconds'),
                  ' age_s=',round(age,3),'action=',action,'side=',side,
                  'price=',price,'market=',market)
        print('NOTE: First-observed ages are an upper bound on public visibility delay;')
        print('they also include polling/response time and any clock offset. Cannot recover')
        print('exact first-observation ages for trades previously recorded by older bot versions.')
    finally:
        db.close()


def poll_once(db,k,args,boot=False):
    raw=fetch_wallet(time.time())
    parsed=[t for r in raw if (t:=parse_trade(r)) is not None]
    ids=Counter(t['id'] for t in parsed)
    ambiguous={t['start'] for t in parsed if ids[t['id']]>1}
    if ambiguous:
        for start in ambiguous:
            log('AMBIGUOUS_SOURCE_IDENTITIES',market=start)
            if market_row(db,start): mark_stop(db,start,'ambiguous_trade_identity')
        if args.mode=='live':
            raise RuntimeError('Duplicate trade IDs in live wallet feed; halted')
    parsed.sort(key=lambda x:(x['ts'],x['id']))
    # When the source timestamp resolves only to whole seconds, the first
    # trade within a second may be ambiguous. Do not pick a direction/price
    # merely because the feed or a hash sorted it first.
    first_sec={}
    for tr in parsed:
        if tr['action']!='BUY':
            continue
        if tr['start'] not in first_sec or tr['ts']<first_sec[tr['start']]:
            first_sec[tr['start']]=tr['ts']
    conflicted=set()
    for start, t0 in first_sec.items():
        if market_row(db,start) is not None:
            continue
        group=[tr for tr in parsed if tr['start']==start and tr['ts']==t0 and tr['action']=='BUY']
        types={(tr['side'], D('.20')<=tr['price']<D('.40')) for tr in group}
        if len(group)>1 or len(types)>1:
            conflicted.add(start)
            log('AMBIGUOUS_FIRST_SECOND',market=start,candidates=len(group),
                reason='No reliable within-second ordering of the first wallet fill')
    fresh=0
    newly_seen=0
    newly_late=0
    startup_seen=0
    current_start=int(time.time()//900)*900
    for tr in parsed:
        # Both inserts commit atomically. Existing (pre-upgrade) IDs in seen
        # remain deduplicated and are NEVER misreported as newly observed.
        observed_at=time.time()
        first_age=observed_at-tr['ts']
        with db:
            inserted=db.execute('INSERT OR IGNORE INTO seen VALUES(?,?,?)',
                                (tr['id'],tr['start'],observed_at)).rowcount
            if inserted:
                db.execute('''INSERT INTO feed_first_seen (
                    trade_id,market_start,wallet_ts,first_seen_at,age_s,side,
                    action,wallet_price,wallet_qty,startup,age_limit_s
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                    (tr['id'],tr['start'],tr['ts'],observed_at,first_age,
                     tr['side'],tr['action'],str(tr['price']),str(tr['size']),
                     int(boot),args.max_age_seconds))
        if not inserted:
            continue
        if boot:
            startup_seen+=1
        else:
            newly_seen+=1
            newly_late+=int(first_age>args.max_age_seconds)
            log('FEED_FIRST_SEEN',market=tr['start'],trade_id=tr['id'],
                action=tr['action'],side=tr['side'],wallet_price=str(tr['price']),
                wallet_size=str(tr['size']),
                first_observed_age_s=round(first_age,3),
                over_max_age=first_age>args.max_age_seconds,
                within_15s=0<=first_age<=15)
        # Do not start from a pre-existing market at boot; preserve known
        # active positions created by this DB across restarts.
        if boot and tr['start']<=current_start and market_row(db,tr['start']) is None:
            db.execute('INSERT OR IGNORE INTO markets (market_start,stopped,reason) VALUES(?,1,?)',
                       (tr['start'],'startup_market_no_retro_entry'));db.commit()
            continue
        if tr['start'] in ambiguous:
            continue
        if tr['start'] in conflicted:
            db.execute('INSERT OR IGNORE INTO markets (market_start,stopped,reason) VALUES(?,1,?)',
                       (tr['start'],'ambiguous_first_second'));db.commit()
            continue
        if tr['ts']>time.time()+2:
            raise RuntimeError('Future-dated wallet trade; clock sync issue')
        if tr['start']+900<=time.time():
            continue
        handle_trade(db,k,args,tr)
        fresh+=1
    log('POLL',raw=len(raw),new=fresh,first_seen_new=newly_seen,
        lagged=newly_late,startup_backfill=startup_seen)


def self_test():
    first={'slug':'btc-updown-15m-1793923200','timestamp':1793923202,
           'outcome':'Up','side':'BUY','price':'0.3','size':'100',
           'proxy_wallet':WALLET,'trade_id':'a'}
    tr=parse_trade(first)
    assert tr['side']=='UP' and tr['start']==1793923200 and tr['size']==100
    assert [round_target(10,D(x),D(100)) for x in (100,150,200,225,275)]==[10,15,20,23,28]
    # NO ask 30c when YES bid 70c -> book-side ask limit 70c.
    q=executable_quote({'orderbook_fp':{'yes_dollars':[['0.70','20']],
                                    'no_dollars':[['0.60','40']]}},'DOWN',10)
    assert q['best_ask']==D('.30') and q['limit_side_price']==D('.70')
    p=executable_quote({'orderbook_fp':{'yes_dollars':[['0.70','20']],
                                    'no_dollars':[['0.60','4'],['0.55','8']]}},'UP',10)
    assert p['vwap']==D('.43') and p['worst_ask']==D('.45')
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'copy.sqlite')
        class Fake:
            def __init__(self):self.orders=[]
            def market_for_start(self,start):return 'KXBTC15M-FAKE'
            def book(self,ticker):return {'orderbook_fp':{'no_dollars':[['0.65','1000']],
                                        'yes_dollars':[['0.65','1000']]}}
        fake=Fake()
        a=argparse.Namespace(mode='paper',base_qty=10,max_age_seconds=10,
             close_buffer_seconds=3,max_ask=D('.95'),max_drift=D('.15'),
             max_order_spend=D('1000'),max_market_spend=D('1000'),max_day_spend=D('1000'))
        now=int(time.time()); start=now//900*900
        # Simulated clock must lie in the middle of a 15-min market.
        # Use a synthetic market ~5s ago, not aligned, for pure sizing tests.
        start=now-5
        def act(uid,sec,side,size,price='0.30',action='BUY'):
            tr={'id':uid,'start':start,'ts':start+sec,'side':side,
                'action':action,'size':D(str(size)),'price':D(price)}
            handle_trade(db,fake,a,tr)
        act('x',0,'UP',100)
        act('y',2,'UP',10)
        act('z',5,'UP',40)
        act('w',6,'UP',25)
        st=market_row(db,start)
        assert st[5]==18 and st[6]==18 and st[8]==1,st
        act('u',7,'DOWN',10)
        act('v',8,'UP',50)
        assert market_row(db,start)[6]==18 and market_row(db,start)[9]==1
        assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==3
        db.close()
    # Mock overlapping live feed polls to verify first-seen telemetry, dedupe,
    # persistent restart behavior, exclusion of startup backlog, and late age.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'telemetry.sqlite')
        t=int(time.time()); start=t-110
        def example(uid,ago):
            return dict(slug=f'btc-updown-15m-{start}', timestamp=t-ago,
                        outcome='Up',side='BUY',price='0.30',size='10',
                        proxy_wallet=WALLET,trade_id=uid)
        bootrow=example('old_backlog',90)
        early=example('new_early',4)
        late=example('new_late',24)
        original_fetch=globals()['fetch_wallet']
        original_handle=globals()['handle_trade']
        original_log=globals()['log']
        seen_logs=[]
        try:
            globals()['fetch_wallet']=lambda _now: current[0]
            globals()['handle_trade']=lambda *_a,**_kw: None
            globals()['log']=lambda what,**kw: seen_logs.append((what,kw))
            current=[[bootrow]]
            poll_once(db,None,a,boot=True)
            assert feed_age_summary(db)['unique_post_startup']==0
            current[0]=[bootrow,early,late]
            poll_once(db,None,a,boot=False)
            assert feed_age_summary(db)['unique_post_startup']==2
            assert feed_age_summary(db)['within_15s_pct']==50.0
            assert sum(what=='FEED_FIRST_SEEN' for what,_ in seen_logs)==2
            assert [kw['lagged'] for what,kw in seen_logs if what=='POLL']==[0,1]
            poll_once(db,None,a,boot=False)
            assert feed_age_summary(db)['unique_post_startup']==2
            assert [kw['lagged'] for what,kw in seen_logs if what=='POLL']==[0,1,0]
            # Restart with same DB must not duplicate first-seen events.
            db.close();db=init_db(Path(folder)/'telemetry.sqlite')
            poll_once(db,None,a,boot=True)
            assert feed_age_summary(db)['unique_post_startup']==2
        finally:
            globals()['fetch_wallet']=original_fetch
            globals()['handle_trade']=original_handle
            globals()['log']=original_log
            db.close()
    log('SELF_TEST_PASSED',checks='pricing, order directions, target rounding, confirmation, additions, stop, SQLite, first-seen age, startup exclusion, dedupe, restart')


def report(db_path):
    db=sqlite3.connect(f'file:{Path(db_path).resolve()}?mode=ro',uri=True)
    print('954B LIVE COPIER JOURNAL — PAPER COSTS/ESTIMATES, NOT VERIFIED FILLS')
    for row in db.execute('SELECT status,COUNT(*),SUM(filled),SUM(CAST(actual_spent AS REAL)) FROM orders GROUP BY status'):
        print('orders',row)
    for row in db.execute('SELECT reason,COUNT(*) FROM markets WHERE stopped=1 GROUP BY reason ORDER BY COUNT(*) DESC'):
        print('stops',row)
    for row in db.execute('SELECT kind,COUNT(*),ROUND(AVG(age_s),3) FROM observations GROUP BY kind ORDER BY COUNT(*) DESC'):
        print('observations',row)
    print('Unreconciled submissions:',db.execute("SELECT COUNT(*) FROM orders WHERE status IN ('PENDING','UNKNOWN')").fetchone()[0])
    db.close()


def settlement_report(db_path):
    """Public-market settlement check. Paper fills and costs remain hypothetical."""
    db=sqlite3.connect(f'file:{Path(db_path).resolve()}?mode=ro',uri=True)
    rows=db.execute("SELECT market_start,ticker,side,COALESCE(SUM(filled),0),"
          "COALESCE(SUM(CAST(actual_spent AS REAL)),0) FROM orders "
          "WHERE status IN ('FILLED','PAPER') GROUP BY market_start,ticker,side").fetchall()
    total_cost=D(0);total_pnl=D(0);settled=0;unsettled=0
    for start,ticker,side,qty,cost in rows:
        try:
            m=get_json(KALSHI_URL+'/markets/'+ticker).get('market',{})
            result=str(m.get('result','')).lower()
            if result not in ('yes','no'):
                unsettled+=1;continue
        except Exception as exc:
            log('UNSETTLED_OR_UNAVAILABLE',ticker=ticker,error=str(exc));unsettled+=1;continue
        payout=D(qty) if (side=='UP' and result=='yes') or (side=='DOWN' and result=='no') else D(0)
        spent=money(cost);pnl=payout-spent
        total_cost+=spent;total_pnl+=pnl;settled+=1
        print(datetime.fromtimestamp(start,timezone.utc).strftime('%Y-%m-%d %H:%MZ'),
              ticker,side,'contracts=',qty,'cost=',f'${spent:.2f}',
              'payout=',f'${payout:.2f}','P&L=',f'${pnl:+.2f}')
    print(f'SETTLED={settled}, UNKNOWN/UNSETTLED={unsettled}, '
          f'COST=${total_cost:.2f}, PNL=${total_pnl:+.2f}, '
          f'ROI={(100*total_pnl/total_cost) if total_cost else D(0):+.2f}%')
    print('PAPER: assumed fills and fee estimates; LIVE: journaled order costs, not exchange reconciliation.')
    db.close()


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd()/'.env')
    except ImportError:
        pass
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--report',action='store_true')
    ap.add_argument('--feed-latency-report',action='store_true',
                    help='Offline unique first-observed age percentiles, excludes startup backlog')
    ap.add_argument('--settlement-report',action='store_true',help='Use public Kalshi GET to estimate settled copy P&L')
    ap.add_argument('--probe-feed',action='store_true',help='One read-only Polymarket API schema/age check; no DB or Kalshi orders')
    ap.add_argument('--stop-file',default='.collector_runs/STOP_W954_COPY')
    ap.add_argument('--mode',choices=('paper','live'),default='paper')
    ap.add_argument('--db',default=DEFAULT_DB)
    ap.add_argument('--poll-seconds',type=float,default=1.5)
    ap.add_argument('--max-age-seconds',type=float,default=8)
    ap.add_argument('--base-qty',type=int,default=10)
    ap.add_argument('--max-drift-cents',type=float,default=5)
    ap.add_argument('--max-ask',type=float,default=.95)
    ap.add_argument('--max-order-spend',type=float,default=20)
    ap.add_argument('--max-market-spend',type=float,default=50)
    ap.add_argument('--max-day-spend',type=float,default=100)
    ap.add_argument('--close-buffer-seconds',type=int,default=5)
    ap.add_argument('--confirm-live',default='',help='Must be I_ACCEPT_REAL_TRADES for live mode')
    args=ap.parse_args()
    if args.self_test:return self_test()
    if args.report:return report(args.db)
    if args.feed_latency_report:return feed_latency_report(args.db)
    if args.settlement_report:return settlement_report(args.db)
    if args.probe_feed:
        now=time.time()
        rows=fetch_wallet(now)
        btc=[x for r in rows if (x:=parse_trade(r)) is not None]
        ages=[now-x['ts'] for x in btc]
        log('FEED_PROBE',wallet=WALLET,rows=len(rows),btc_trades=len(btc),
            newest_age=round(min(ages),3) if ages else None,
            oldest_age=round(max(ages),3) if ages else None,
            note='Time since exchange trade, NOT directly measured dissemination latency')
        return
    args.max_drift=D(str(args.max_drift_cents))/100
    args.max_ask=D(str(args.max_ask))
    for name in ('max_order_spend','max_market_spend','max_day_spend'):
        setattr(args,name,D(str(getattr(args,name))))
    if (args.poll_seconds<.5 or args.max_age_seconds<0 or args.max_age_seconds>30 or
        args.base_qty!=10 or args.close_buffer_seconds<1 or
        any(getattr(args,n)<=0 for n in ('max_order_spend','max_market_spend','max_day_spend'))):
        ap.error('Invalid execution/risk configuration; base qty frozen at 10')
    if args.mode=='live':
        if args.confirm_live!='I_ACCEPT_REAL_TRADES' or os.getenv('KALSHI_LIVE_TRADING_ENABLED','').lower()!='true':
            ap.error('LIVE disabled: require --confirm-live I_ACCEPT_REAL_TRADES AND KALSHI_LIVE_TRADING_ENABLED=true')
    db=init_db(args.db)
    mode=db.execute("SELECT value FROM runtime WHERE name='mode'").fetchone()
    if mode and mode[0]!=args.mode:
        raise RuntimeError(f'State DB belongs to {mode[0]} mode; use a NEW --db for {args.mode}')
    db.execute("INSERT OR IGNORE INTO runtime(name,value) VALUES('mode',?)",(args.mode,))
    db.commit()
    if db.execute("SELECT count(*) FROM orders WHERE status IN ('PENDING','UNKNOWN')").fetchone()[0]:
        raise RuntimeError('Unreconciled previous order exists; DO NOT RESTART until manually resolved')
    k=Kalshi(args.mode=='live')
    log('START',mode=args.mode,wallet=WALLET,poll_s=args.poll_seconds,db=args.db,
        risk_limits={'order':str(args.max_order_spend),'market':str(args.max_market_spend),
                     'day':str(args.max_day_spend)},max_age=args.max_age_seconds)
    # Refrain from copying any market already open on startup. Except positions
    # persisted from this same DB; they will continue only for fresh events.
    boot=True
    last_feed_age_summary=time.monotonic()
    while True:
        if Path(args.stop_file).exists():
            log('STOP_FILE_DETECTED',path=args.stop_file);break
        try:
            poll_once(db,k,args,boot=boot)
            boot=False
            if time.monotonic()-last_feed_age_summary>=60:
                log('FEED_AGE_SUMMARY',**feed_age_summary(db))
                last_feed_age_summary=time.monotonic()
        except KeyboardInterrupt:
            log('STOP',reason='SIGINT');break
        except Exception as exc:
            log('FATAL_FAIL_CLOSED',error=str(exc));raise
        time.sleep(args.poll_seconds)

if __name__=='__main__':main()
