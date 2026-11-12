#!/usr/bin/env python3
"""Wallet 954B -> Kalshi BTC15M copier, v2.4.1 small-account configuration — verified zero-fill FOK handling. Paper DEFAULT; gated LIVE option.

Strategy frozen from historical research:
- A wallet's FIRST BUY in a BTC 15m market must be 0.20 <= price < 0.40.
- Buy 5 Kalshi UP=YES / DOWN=NO at observed, executable book price.
- First same-outcome repeat 4..30 seconds after first triggers proportional increase.
- Continue all same-outcome BUYs; cumulative wallet shares / first shares sets
  cumulative desired Kalshi contracts (round half up).
- Stop adding at first opposite-outcome BUY or ANY SELL; hold bought inventory
  until exchange settlement. No share-count strategy cap, but mandatory live risk
  dollar limits, maximum quote drift, and liquidity checks.

No arbitrary market/ticker inference, no historical catch-up orders, no live
orders unless both CLI and environment expressly enable them. Same-second
wallet fills follow the V2 API keyset sequence, NOT an ID/hash tie-break.
All processed
wallet records and submissions persist to a dedicated SQLite database.
When the public feed lacks execution indices, distinct same-transaction same-economics
fills cannot be uniquely identified. Re-query suspicious duplicates; count at most
one economic identity in paper. In live, skip persistently ambiguous identities.

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
CRYPTO_EXCHANGE_INDEX = 2
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
    if not uid:
        tx = row.get('transaction_hash', row.get('transactionHash'))
        seq = row.get('log_index', row.get('logIndex'))
        if tx:
            tx = str(tx).lower()
            if not re.fullmatch(r'0x[0-9a-f]{64}', tx):
                raise RuntimeError('Malformed transaction hash in wallet trade')
            if seq is not None:
                # The actual transaction-log index is a unique event anchor.
                uid = f'tx_log:{tx}:{seq}'
            else:
                # The Polymarket V2 response has no stable trade ID/log index.
                # Hash only immutable-looking execution economics, NEVER full
                # JSON: mutable profile/icon/bio fields can change between polls.
                asset = row.get('token_id', row.get('tokenId', row.get('asset')))
                condition = row.get('condition_id', row.get('conditionId'))
                if asset is None and condition is None:
                    raise RuntimeError('Trade has transaction hash but no asset/condition identifier')
                core = [tx, str(asset or ''), str(condition or ''),
                        int(ts), action, side,
                        str(price.quantize(D('0.00000001'))),
                        str(qty.quantize(D('0.000000001')))]
                fingerprint = hashlib.sha256(
                    json.dumps(core, separators=(',', ':')).encode()).hexdigest()
                uid = 'tx_economic:' + fingerprint
    if not uid:
        # An entire-response payload hash is NOT a reliable execution ID.
        # Without a transaction identifier, do not risk buying the same fill twice.
        raise RuntimeError('Missing stable trade identity; fail closed')
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


def ensure_trade_identity_version(db):
    """Fail-safe migration for existing v2 journals; no order/position deletion.

    An older payload hash cannot be mapped reliably to the newer transaction
    fingerprint. Freeze only still-open market states from the earlier schema
    so they cannot receive duplicate additions after a restart.
    """
    version = db.execute("SELECT value FROM runtime WHERE name='trade_identity_schema'").fetchone()
    if version:
        if version[0] != 'tx_economic_v1':
            raise RuntimeError('Unsupported trade identity schema: ' + version[0])
        return
    previously_seen = db.execute('SELECT COUNT(*) FROM seen').fetchone()[0]
    paused = []
    if previously_seen:
        paused = [r[0] for r in db.execute(
            'SELECT market_start FROM markets WHERE stopped=0 AND market_start+900>?',
            (time.time(),)).fetchall()]
        with db:
            for market in paused:
                db.execute('UPDATE markets SET stopped=1,reason=? WHERE market_start=?',
                           ('identity_schema_migration_pause', market))
    with db:
        db.execute("INSERT INTO runtime(name,value) VALUES('trade_identity_schema','tx_economic_v1')")
    log('IDENTITY_SCHEMA',new_version='tx_economic_v1',prior_seen=previously_seen,
        paused_existing_markets=paused)


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


class DefinitiveOrderRejection(RuntimeError):
    """An explicit Kalshi HTTP error that did not accept a new order."""


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
        if m.get('exchange_index') is None or int(m['exchange_index']) != CRYPTO_EXCHANGE_INDEX:
            raise RuntimeError('BTC15m market not confirmed on crypto exchange index 2: ' + str(m.get('exchange_index')))
        return m['ticker']

    def book(self,ticker):
        return get_json(self.base+'/markets/'+ticker+'/orderbook',timeout=4)

    def balance_snapshot(self, exchange_index=None):
        """Authenticated read-only balance, with bounded 429 backoff.

        Kalshi Basic/Advanced permits one balance request per second. The
        exchange-index-specific balance is the only balance used for crypto.
        Exhaustion fails closed; never infer buying power from cached funds.
        """
        import requests
        path='/portfolio/balance'
        url=self.base+path
        if exchange_index is not None:
            url+='?exchange_index='+str(int(exchange_index))
        # Kalshi signs the URL path WITHOUT query parameters. Sign each retry
        # anew because KALSHI-ACCESS-TIMESTAMP is part of the signature.
        for attempt in range(3):
            r=requests.get(url,headers=self.headers('GET',path),timeout=5)
            if r.status_code!=429:
                r.raise_for_status()
                data=r.json()
                if not isinstance(data,dict) or 'balance_dollars' not in data:
                    raise RuntimeError('Missing Kalshi available balance in response')
                return data
            if attempt==2:
                raise RuntimeError('Kalshi balance rate limit (HTTP 429) persisted after 3 attempts; refusing to assume funds')
            delay=1.25 * (2 ** attempt)
            log('BALANCE_API_RATE_LIMIT',attempt=attempt+1,backoff_seconds=delay,
                note='Only retrying read-only balance GET; no order submitted')
            time.sleep(delay)

    def balance(self):
        # Aggregate balance includes other matching engines and cannot fund crypto orders.
        return money(self.balance_snapshot(CRYPTO_EXCHANGE_INDEX)['balance_dollars'])

    def send_fok(self,ticker,side,quantity,book_limit,client_id):
        """V2 bid buys YES at YES price; ask buys NO at YES-scale price."""
        import requests
        path='/portfolio/events/orders'
        body={'ticker':ticker,'side':'bid' if side=='UP' else 'ask',
              'count':f'{quantity}.00','price':f'{book_limit:.4f}',
              'time_in_force':'fill_or_kill','client_order_id':client_id,
              'self_trade_prevention_type':'taker_at_cross',
              'cancel_order_on_pause':True,
              'exchange_index':CRYPTO_EXCHANGE_INDEX}
        try:
            r=requests.post(self.base+path,json=body,headers=self.headers('POST',path),timeout=5)
        except requests.RequestException as e:
            raise RuntimeError('UNKNOWN_ORDER_SUBMISSION: '+str(e)) from e
        if r.status_code in (400,401,403,404,422):
            # No accepted order. Keep no-fill distinct from a network timeout.
            raise DefinitiveOrderRejection(f'ORDER_REJECTED HTTP {r.status_code}: {r.text[:200]}')
        if r.status_code!=201:
            # Especially 408, 409, 429, 5xx: the request may have been accepted.
            raise RuntimeError(f'UNKNOWN_ORDER_SUBMISSION HTTP {r.status_code}: {r.text[:200]}')
        try:
            data=r.json()
            if not isinstance(data,dict) or not data.get('order_id'):
                raise ValueError('missing order_id in accepted order response')
            got=money(data['fill_count'])
            remaining=money(data['remaining_count'])
            if got<0 or got>D(quantity) or remaining<0 or remaining>D(quantity):
                raise ValueError('invalid quantity in accepted order response')
        except (ValueError,KeyError,TypeError) as exc:
            raise RuntimeError('UNKNOWN_ACCEPTED_ORDER_RESPONSE: '+str(exc)) from exc
        # For a FOK order, zero fills can be a normal CANCELED order. Return
        # the accepted response to the journaler, which MUST verify exchange state.
        return data

    def verified_order(self, order_id, client_id, ticker):
        """Read-only exact order lookup via Exchange 2 order LIST.

        GET /portfolio/orders/{id} can temporarily return 404 even after a
        successful V2 create response. The paginated order-list endpoint also
        supports exchange_index and ticker (and was verified on this account).
        Do not treat absence as zero fills: fail closed after bounded retries.
        """
        import requests
        path='/portfolio/orders'
        for attempt in range(3):
            cursor=None
            for page in range(30):
                params={'ticker':ticker,'exchange_index':CRYPTO_EXCHANGE_INDEX,'limit':200}
                if cursor:params['cursor']=cursor
                try:
                    r=requests.get(self.base+path,params=params,
                        headers=self.headers('GET',path),timeout=8)
                    r.raise_for_status()
                    data=r.json()
                except Exception as exc:
                    if attempt==2:
                        raise RuntimeError(f'Order list verification failed; preserve UNKNOWN: {exc}') from exc
                    log('ORDER_VERIFY_RETRY',order_id=order_id,reason='GET_ERROR',attempt=attempt+1)
                    break
                orders=data.get('orders')
                if not isinstance(orders,list):
                    raise RuntimeError('Malformed Kalshi orders list; preserve UNKNOWN')
                for order in orders:
                    if str(order.get('order_id')) != str(order_id):
                        continue
                    if (str(order.get('client_order_id')) != str(client_id) or
                        str(order.get('ticker')) != str(ticker)):
                        raise RuntimeError('Order identity mismatch; preserve UNKNOWN')
                    if (order.get('exchange_index') is None or
                        int(order['exchange_index'])!=CRYPTO_EXCHANGE_INDEX):
                        raise RuntimeError('Order exchange-index mismatch; preserve UNKNOWN')
                    return order
                cursor=data.get('cursor')
                if not cursor:break
            else:
                raise RuntimeError('Order-list pagination incomplete; preserve UNKNOWN')
            if attempt<2:
                log('ORDER_VERIFY_RETRY',order_id=order_id,reason='NOT_YET_VISIBLE',attempt=attempt+1)
                time.sleep(0.6*(2**attempt))
        raise RuntimeError(f'Accepted order {order_id} not verified by order list; preserve UNKNOWN')

    def verified_no_fill(self,order_id,client_id,ticker):
        order=self.verified_order(order_id,client_id,ticker)
        if (order.get('status')!='canceled' or
            money(order['fill_count_fp'])!=0 or
            money(order['remaining_count_fp'])!=0):
            raise RuntimeError('Order not conclusively canceled with zero fills')
        return order

    def verified_fill(self,order_id,expected,client_id,ticker):
        order=self.verified_order(order_id,client_id,ticker)
        filled=money(order['fill_count_fp'])
        remaining=money(order['remaining_count_fp'])
        if (order.get('status')!='executed' or
            filled != D(expected) or remaining != 0):
            raise RuntimeError(f'Unexpected exchange order state: status={order.get("status")} filled={filled} remaining={remaining}')
        return order


def account_check():
    """No orders, transfers, database changes, or mode gate: authenticate and read."""
    k=Kalshi(True)
    # Only one authenticated balance read. Two back-to-back balance GETs
    # exceed Kalshi Basic/Advanced balance-specific quota (1 request/sec).
    crypto=k.balance_snapshot(CRYPTO_EXCHANGE_INDEX)
    crypto_balance=money(crypto['balance_dollars'])
    markets=get_json(KALSHI_URL+'/markets', {
        'series_ticker':'KXBTC15M','status':'open','limit':10})
    candidates=markets.get('markets',[])
    if not isinstance(candidates,list):
        raise RuntimeError('Invalid public Kalshi market metadata')
    reported=[]
    for m in candidates[:5]:
        reported.append({'ticker':m.get('ticker'),
                         'exchange_index':m.get('exchange_index')})
        if m.get('exchange_index') is None or int(m['exchange_index']) != CRYPTO_EXCHANGE_INDEX:
            raise RuntimeError('Unexpected BTC15m market exchange index; abort live activation')
    log('ACCOUNT_PREFLIGHT_READ_ONLY',
        crypto_index=CRYPTO_EXCHANGE_INDEX,crypto_available_dollars=str(crypto_balance),
        first_five_btc_markets=reported,
        status='FUNDS_PRESENT' if crypto_balance>=D('5') else 'TRANSFER_FUNDS_TO_CRYPTO_INDEX_2',
        note='No orders or transfers sent. Confirm allocation in Kalshi UI.')


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
    # Quote retrieval and an authenticated balance read consume wall-clock time.
    # Recheck freshness immediately before journaling/submitting any real order.
    fresh_age=time.time()-tr['ts']
    if fresh_age < -2 or fresh_age > args.max_age_seconds:
        record_obs(db,tr,fresh_age,'STALE_AFTER_CHECKS',q=quote,qty=qty,detail=kind)
        mark_stop(db,tr['start'],'signal_expired_before_submit')
        return False
    if time.time()>=tr['start']+900-args.close_buffer_seconds:
        record_obs(db,tr,fresh_age,'TOO_LATE_AFTER_CHECKS',q=quote,qty=qty,detail=kind)
        mark_stop(db,tr['start'],'market_expired_before_submit')
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
        order_id=''
        try:
            response=k.send_fok(ticker,tr['side'],qty,quote['limit_side_price'],client_id)
            order_id=str(response['order_id'])
            # An accepted exchange order ID is durable BEFORE verification;
            # subsequent GET failure must not discard this reconciliation key.
            db.execute('UPDATE orders SET order_id=? WHERE client_id=?',(order_id,client_id))
            db.commit()
            response_filled=money(response['fill_count'])
            response_remaining=money(response['remaining_count'])
            if response_filled==0:
                if response_remaining!=0:
                    raise RuntimeError('ZERO_FILL_BUT_ORDER_REMAINS: require manual reconciliation')
                k.verified_no_fill(order_id,client_id,ticker)
                db.execute('UPDATE orders SET status=?,filled=0,actual_spent=? WHERE client_id=?',
                           ('NO_FILL','0',client_id))
                db.commit()
                mark_stop(db,tr['start'],'kalshi_fok_no_fill')
                record_obs(db,tr,age,'LIVE_NO_FILL',q=quote,qty=qty,detail=kind)
                log('LIVE_NO_FILL',id=client_id,order_id=order_id,market=tr['start'],
                    ticker=ticker,qty=qty,note='Exchange confirms canceled, zero filled; not retrying')
                return False
            if response_filled!=D(qty) or response_remaining!=0:
                raise RuntimeError(f'UNEXPECTED_FOK_RESPONSE: filled={response_filled} remaining={response_remaining}')
            verified=k.verified_fill(order_id,qty,client_id,ticker)
            actual_cost=sum((money(verified.get(field,'0')) for field in
                ('taker_fill_cost_dollars','maker_fill_cost_dollars',
                 'taker_fees_dollars','maker_fees_dollars')),D(0))
            if actual_cost>0:
                est=actual_cost
        except DefinitiveOrderRejection as exc:
            db.execute('UPDATE orders SET status=?,filled=0,actual_spent=? WHERE client_id=?',
                       ('REJECTED','0',client_id))
            db.commit()
            mark_stop(db,tr['start'],'kalshi_order_rejected')
            record_obs(db,tr,age,'LIVE_REJECTED',q=quote,qty=qty,detail=kind)
            log('LIVE_REJECTED',id=client_id,market=tr['start'],error=str(exc))
            return False
        except Exception as exc:
            db.execute('UPDATE orders SET status=? WHERE client_id=?',('UNKNOWN',client_id))
            db.commit()
            mark_stop(db,tr['start'],'order_unknown_requires_manual_reconciliation')
            log('CRITICAL_ORDER_UNKNOWN',id=client_id,order_id=order_id,error=str(exc))
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
        # FIRST wallet fill in the exchange feed sequence (not first qualifying
        # fill). A later same-second fill cannot retroactively become FIRST.
        qualifies=tr['action']=='BUY' and D('.20')<=tr['price']<D('.40')
        log('FIRST_WALLET_FILL',market=start,trade_id=tr['id'],
            action=tr['action'],side=tr['side'],wallet_price=str(tr['price']),
            wallet_shares=str(tr['size']),eligible=qualifies,
            source_order='polymarket_v2_keyset_reverse')
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
    if tr['ts']<st['first_ts']:
        # An older fill arrived after we had already acted. Never reverse or
        # resubmit the first order; freeze additional exposure for this market.
        mark_stop(db,start,'older_wallet_fill_arrived_after_entry')
        return
    if tr['ts']==st['first_ts']:
        # The source timestamp is whole-second resolution. Later fills in the
        # API sequence may legitimately share it. Process them normally:
        # same-side BUY contributes shares; opposite BUY or SELL stops adds.
        # Same-second BUY alone does not count as 4..30s confirmation.
        log('SAME_SECOND_FOLLOWUP',market=start,trade_id=tr['id'],
            side=tr['side'],action=tr['action'],price=str(tr['price']),
            shares=str(tr['size']))
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
    counts=Counter(t['id'] for t in parsed)
    duplicate_ids={uid for uid,n in counts.items() if n>1}
    if duplicate_ids:
        # Do NOT reject a whole market just because the API repeated a row.
        # Make one immediate, bounded follow-up GET, then use that response
        # instead of counting an old record which may have disappeared.
        # This only runs when a duplicate STABLE execution ID is present.
        log('SOURCE_DUPLICATE_REQUERY',duplicate_identities=len(duplicate_ids),
            repeated_rows=sum(counts[uid]-1 for uid in duplicate_ids),
            candidate_markets=sorted({t['start'] for t in parsed if t['id'] in duplicate_ids}))
        raw=fetch_wallet(time.time())
        parsed=[t for r in raw if (t:=parse_trade(r)) is not None]
        counts=Counter(t['id'] for t in parsed)
        duplicate_ids={uid for uid,n in counts.items() if n>1}
        if duplicate_ids:
            log('SOURCE_DUPLICATE_UNRESOLVED',
                duplicate_identities=len(duplicate_ids),
                markets=sorted({t['start'] for t in parsed if t['id'] in duplicate_ids}),
                action='collapse_to_one_in_paper_skip_affected_markets_in_live')
    # Preserve source newest-first/keyset ordering. Same execution identity
    # must NEVER produce multiple copied orders or increase wallet shares twice.
    # Different transaction hashes or different real log indices remain distinct.
    seen_in_snapshot=set()
    unique=[]
    for t in parsed:
        if t['id'] in seen_in_snapshot:
            continue
        seen_in_snapshot.add(t['id'])
        unique.append(t)
    if len(unique)<len(parsed):
        log('SOURCE_DUPLICATE_COLLAPSED',rows_removed=len(parsed)-len(unique))
    parsed=unique
    # Re-query can still leave an identical row repeated with no log index.
    # Paper may count one economic identity; LIVE skips the affected market
    # until the feed resolves the ambiguity, without permanently stopping it.
    unsafe_live_markets=({t['start'] for t in parsed if t['id'] in duplicate_ids}
                         if args.mode=='live' else set())
    # The V2 wallet trade feed returns newest-first, with a keyset cursor
    # anchored on (block_timestamp, sequence_id). The caller concatenates
    # pages newest -> oldest, so reversing uses the API's within-second
    # sequence rather than an arbitrary hash/ID tie-break.
    # Fail closed if a provider change violates chronological page ordering.
    for previous, following in zip(parsed,parsed[1:]):
        if previous['ts'] < following['ts']:
            raise RuntimeError('Polymarket v2 feed not newest-first; '
                               'refusing to infer first-trade sequence')
    parsed.reverse()
    # Diagnostic only. Multiple first-second fills do NOT invalidate a market.
    # Their trade order follows the API keyset order, without claiming that
    # this is independently verified Polygon event/log ordering.
    first_by_market={}
    for tr in parsed:
        first_by_market.setdefault(tr['start'],tr['ts'])
    first_groups={}
    for tr in parsed:
        if tr['ts']==first_by_market[tr['start']]:
            first_groups.setdefault(tr['start'],[]).append(tr)
    for start,group in first_groups.items():
        if len(group)>1 and market_row(db,start) is None:
            log('MULTIFILL_FIRST_SECOND',market=start,number_of_fills=len(group),
                first_api_side=group[0]['side'],first_api_action=group[0]['action'],
                first_api_price=str(group[0]['price']),
                outcomes=[f"{x['action']}:{x['side']}@{x['price']}" for x in group[:12]],
                ordering='polymarket_v2_reverse_keyset_not_chain_verified')
    fresh=0
    newly_seen=0
    newly_late=0
    startup_seen=0
    current_start=int(time.time()//900)*900
    for tr in parsed:
        if tr['start'] in unsafe_live_markets:
            # Do not mark this identity as seen: a subsequent clean snapshot
            # should be able to re-evaluate it before the age cutoff.
            continue
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
    assert [round_target(5,D(x),D(100)) for x in (100,150,200,225,275)]==[5,8,10,11,14]
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
            current[0]=[early,late,bootrow]  # V2 returns newest-first.
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
    # End-to-end paper mock: same-second fills are not fatal. Source order
    # must win over hash sort. Ineligible FIRST still stays ineligible.
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'same_second.sqlite')
        class FakeBooks:
            def market_for_start(self,_start): return 'KXBTC15M-FAKE'
            def book(self,_ticker):
                return {'orderbook_fp':{
                    'yes_dollars':[['0.65','1000']],
                    'no_dollars':[['0.65','1000']]}}
        mock=FakeBooks()
        test_args=argparse.Namespace(mode='paper',base_qty=10,
             max_age_seconds=15,close_buffer_seconds=3,max_ask=D('.95'),
             max_drift=D('.15'),max_order_spend=D('1000'),
             max_market_spend=D('1000'),max_day_spend=D('1000'))
        now=int(time.time())
        start=(now//900)*900
        # If launched within the first 15 seconds of a candle, use the
        # previous interval for tests and move its synthetic event time back.
        # Mock a current market by using a synthetic start 9 seconds ago.
        start=now-9
        def tr(market,uid,second,side,price,size=100,action='BUY'):
            return {'slug':f'btc-updown-15m-{market}',
                    'timestamp':market+second,'outcome':side,'side':action,
                    'price':price,'size':str(size),'proxy_wallet':WALLET,
                    'trade_id':uid}
        a1=tr(start,'z-first',3,'Up','0.30',100)
        a2=tr(start,'a-second',3,'Up','0.31',50)
        a3=tr(start,'m-confirm',7,'Up','0.32',50)
        # Newest-first with same-second second trade before first trade.
        current=[[a3,a2,a1]]
        original_fetch=globals()['fetch_wallet']
        try:
            globals()['fetch_wallet']=lambda _now: current[0]
            poll_once(db,mock,test_args,boot=False)
            m=market_row(db,start)
            assert m[9]==0 and m[8]==1, m  # no stop, confirmed
            assert m[3]=='100' and m[4]=='200',m
            assert m[5]==20 and m[6]==20,m
            assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==2
            # Rereading the rolling API result must not duplicate an order.
            poll_once(db,mock,test_args,boot=False)
            assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==2
            # Opposite-side second fill: copy the initial trade, then stop.
            start2=start+1
            b1=tr(start2,'z-first2',2,'Down','0.30',100)
            b2=tr(start2,'a-opposite2',2,'Up','0.70',50)
            current[0]=[b2,b1]
            poll_once(db,mock,test_args,boot=False)
            m2=market_row(db,start2)
            assert m2[9]==1 and m2[10]=='wallet_opposite_buy' and m2[1]=='DOWN',m2
            assert m2[6]==10,m2
            # Two same-second fills: true first in API order is ineligible,
            # so the second qualifying 30c trade must not create an entry.
            start3=start+2
            c1=tr(start3,'z-ineligible',2,'Up','0.60',100)
            c2=tr(start3,'a-qualifies',2,'Up','0.30',100)
            current[0]=[c2,c1]
            poll_once(db,mock,test_args,boot=False)
            m3=market_row(db,start3)
            assert m3[9]==1 and m3[10]=='first_trade_ineligible',m3
            assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==3
            # Provider returning ascending timestamps must fail before
            # inserting any first-seen rows or copying another trade.
            start4=start+3
            d1=tr(start4,'d-earlier',1,'Up','0.30')
            d2=tr(start4,'d-later',2,'Up','0.30')
            current[0]=[d1,d2]
            try:
                poll_once(db,mock,test_args,boot=False)
                assert False, 'out-of-order V2 response was accepted'
            except RuntimeError as exc:
                assert 'not newest-first' in str(exc),exc
            assert market_row(db,start4) is None
        finally:
            globals()['fetch_wallet']=original_fetch
            db.close()
    # Identity regression: a provider can update display metadata without
    # creating a second economic fill. But distinct transactions and indexed
    # events MUST remain distinct.
    canon = {'slug':f'btc-updown-15m-{start}', 'timestamp':start+2,
             'outcome':'Up','side':'BUY','price':'0.8299999979',
             'size':'51.204817','proxy_wallet':WALLET,
             'transaction_hash':'0x'+'a'*64,'token_id':'asset123',
             'condition_id':'condition456', 'bio':'old',
             'profile_image':'old.png'}
    oldid=parse_trade(canon)['id']
    revised=dict(canon,bio='new biography',profile_image='new.png',
                 icon='new-icon.png')
    assert parse_trade(revised)['id']==oldid
    assert parse_trade(dict(canon,transaction_hash='0x'+'b'*64))['id']!=oldid
    assert parse_trade(dict(canon,size='51.204818'))['id']!=oldid
    assert parse_trade(dict(canon,log_index=0))['id'] != parse_trade(dict(canon,log_index=1))['id']
    try:
        parse_trade({k:v for k,v in canon.items() if k!='transaction_hash'})
        raise AssertionError('Missing execution identity was accepted')
    except RuntimeError as exc:
        assert 'Missing stable trade identity' in str(exc), exc
    # Re-query a duplicate stable execution ID: if only one row survives the
    # second API read, make just ONE initial paper copy; replay never doubles.
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'requery.sqlite')
        fresh_start=int(time.time())-8
        base={'slug':f'btc-updown-15m-{fresh_start}',
              'timestamp':fresh_start+3,'outcome':'Up','side':'BUY',
              'price':'0.30','size':'100','proxy_wallet':WALLET,
              'transaction_hash':'0x'+'c'*64,'token_id':'asset',
              'condition_id':'condition'}
        revised=dict(base,bio='new profile',icon='changed icon')
        original_fetch=globals()['fetch_wallet']
        original_log=globals()['log']
        calls=[]; captured=[]
        try:
            responses=[[base,revised],[revised]]
            def fetch_test(_):
                calls.append(1)
                return responses.pop(0) if responses else [revised]
            globals()['fetch_wallet']=fetch_test
            globals()['log']=lambda what,**kw: captured.append((what,kw))
            poll_once(db,FakeBooks(),test_args,boot=False)
            assert len(calls)==2,('no single requery',len(calls))
            assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==1
            assert db.execute('SELECT COUNT(*) FROM seen').fetchone()[0]==1
            poll_once(db,FakeBooks(),test_args,boot=False)
            assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==1
            assert any(what=='SOURCE_DUPLICATE_REQUERY' for what,_ in captured)
        finally:
            globals()['fetch_wallet']=original_fetch
            globals()['log']=original_log
            db.close()
    # Persistent indistinguishable rows are one paper identity, but cannot
    # authorize live orders. Different transactions must remain distinct.
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'persistent.sqlite')
        live_args=argparse.Namespace(**vars(test_args))
        live_args.mode='live'
        fresh_start=int(time.time())-8
        base={'slug':f'btc-updown-15m-{fresh_start}',
              'timestamp':fresh_start+3,'outcome':'Up','side':'BUY',
              'price':'0.30','size':'100','proxy_wallet':WALLET,
              'transaction_hash':'0x'+'d'*64,'token_id':'asset',
              'condition_id':'condition'}
        original_fetch=globals()['fetch_wallet']
        try:
            calls=[]
            def fetch_repeat(_):
                calls.append(1)
                return [base,dict(base,bio='different')]
            globals()['fetch_wallet']=fetch_repeat
            poll_once(db,FakeBooks(),live_args,boot=False)
            assert len(calls)==2
            assert db.execute('SELECT COUNT(*) FROM seen').fetchone()[0]==0
            assert db.execute('SELECT COUNT(*) FROM orders').fetchone()[0]==0
            assert market_row(db,fresh_start) is None
        finally:
            globals()['fetch_wallet']=original_fetch
            db.close()
    # Existing v2 records may use payload hashes. The upgrade cannot map
    # them back with certainty; freeze only previously open market states.
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'v2_upgrade.sqlite')
        active=int(time.time()//900)*900
        with db:
            db.execute('INSERT INTO seen VALUES(?,?,?)',
                       (str(active)+':payload:oldhash',active,time.time()))
            db.execute('INSERT INTO markets(market_start,side,stopped) VALUES(?,?,0)',
                       (active,'UP'))
        ensure_trade_identity_version(db)
        row=market_row(db,active)
        assert row[9]==1 and row[10]=='identity_schema_migration_pause', row
        ensure_trade_identity_version(db)
        assert db.execute("SELECT COUNT(*) FROM runtime WHERE name='trade_identity_schema'").fetchone()[0]==1
        db.close()
    # Regression: accepted FOK can be canceled without a fill. It MUST be
    # journaled as NO_FILL, not UNKNOWN, and must stop the market without retry.
    with tempfile.TemporaryDirectory() as folder:
        db=init_db(Path(folder)/'live_zero.sqlite')
        class FakeLive:
            def __init__(self, response='zero'): self.response=response; self.attempts=0
            def market_for_start(self, _start): return 'KXBTC15M-FAKE'
            def book(self, _ticker):
                return {'orderbook_fp':{'no_dollars':[['0.70','1000']],
                                        'yes_dollars':[['0.70','1000']]}}
            def balance(self): return D('100')
            def send_fok(self,*args):
                self.attempts+=1
                if self.response=='rejected':
                    raise DefinitiveOrderRejection('ORDER_REJECTED HTTP 400')
                return {'order_id':'ORDER-TEST','fill_count':'0.00','remaining_count':'0.00'}
            def verified_no_fill(self,order_id,client_id,ticker):
                assert order_id=='ORDER-TEST' and ticker=='KXBTC15M-FAKE' and client_id
                if self.response=='verification_failure': raise RuntimeError('GET timed out')
                return {'status':'canceled','fill_count_fp':'0.00'}
        test_args=argparse.Namespace(mode='live',base_qty=5,max_age_seconds=15,
            close_buffer_seconds=3,max_ask=D('.95'),max_drift=D('.05'),
            max_order_spend=D('10'),max_market_spend=D('15'),max_day_spend=D('25'))
        for mode, expected in (('zero','NO_FILL'),('rejected','REJECTED'),
                               ('verification_failure','UNKNOWN')):
            start=int(time.time())-5
            uid='TEST-'+mode
            tr={'id':uid,'start':start,'ts':start+2,'side':'UP','action':'BUY',
                'price':D('.32'),'size':D('100')}
            fake=FakeLive(mode)
            if mode=='verification_failure':
                try: handle_trade(db,fake,test_args,tr)
                except RuntimeError as exc: assert 'FAIL-CLOSED' in str(exc)
                else: raise AssertionError('Uncertain live order not blocked')
            else: handle_trade(db,fake,test_args,tr)
            state=db.execute('SELECT status,order_id,filled FROM orders WHERE trade_id=?',(uid,)).fetchone()
            assert state and state[0]==expected and state[2]==0, state
            if mode!='rejected': assert state[1]=='ORDER-TEST',state
            assert fake.attempts==1 and market_row(db,start)[9]==1
            # Distinct market starts required for independent tests
            if mode!='verification_failure': time.sleep(.004)
            # Remove synthetic market row only, never an actual record.
            db.execute('DELETE FROM markets WHERE market_start=?',(start,))
            db.commit()
        db.close()
    # Verification fallback regression: the V2 POST has returned an order ID,
    # but direct GET /portfolio/orders/<id> is 404; listing still has it.
    # This test only mocks READ requests and never creates real orders.
    from unittest.mock import patch
    class FakeGetResponse:
        status_code=200
        def raise_for_status(self): pass
        def json(self):
            return {'orders':[{'order_id':'ORDER-OK','client_order_id':'CLIENT-OK',
                'ticker':'KXBTC15M-FAKE','exchange_index':2,
                'status':'executed','fill_count_fp':'5.00',
                'remaining_count_fp':'0.00'}], 'cursor':''}
    k=Kalshi(False)
    k.headers=lambda method,path: {}
    import requests
    with patch.object(requests,'get',return_value=FakeGetResponse()) as mocked:
        found=k.verified_fill('ORDER-OK',5,'CLIENT-OK','KXBTC15M-FAKE')
        assert found['fill_count_fp']=='5.00'
        assert mocked.call_args.kwargs['params']['exchange_index']==2
        assert mocked.call_args.args[0].endswith('/portfolio/orders')
    print('PASS: Exchange 2 order-list verification with identity checks (no direct ID GET)')
    log('SELF_TEST_PASSED',checks='including live accepted canceled FOK, explicit rejection, uncertain verification, durable order IDs')


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
    ap.add_argument('--account-check',action='store_true',help='AUTHENTICATED READ ONLY: check crypto index 2 funds and active BTC15M market')
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
    ap.add_argument('--base-qty',type=int,default=5)
    ap.add_argument('--max-drift-cents',type=float,default=5)
    ap.add_argument('--max-ask',type=float,default=.95)
    ap.add_argument('--max-order-spend',type=float,default=10)
    ap.add_argument('--max-market-spend',type=float,default=15)
    ap.add_argument('--max-day-spend',type=float,default=25)
    ap.add_argument('--close-buffer-seconds',type=int,default=5)
    ap.add_argument('--confirm-live',default='',help='Must be I_ACCEPT_REAL_TRADES for live mode')
    args=ap.parse_args()
    if args.self_test:return self_test()
    if args.account_check:return account_check()
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
        args.base_qty!=5 or args.close_buffer_seconds<1 or
        any(getattr(args,n)<=0 for n in ('max_order_spend','max_market_spend','max_day_spend'))):
        ap.error('Invalid execution/risk configuration; base qty frozen at 5')
    if args.mode=='live':
        if args.max_order_spend>D('10') or args.max_market_spend>D('15') or args.max_day_spend>D('25'):
            ap.error('Small-account live risk ceilings are $10/order, $15/market, $25/UTC day')
        if args.confirm_live!='I_ACCEPT_REAL_TRADES' or os.getenv('KALSHI_LIVE_TRADING_ENABLED','').lower()!='true':
            ap.error('LIVE disabled: require --confirm-live I_ACCEPT_REAL_TRADES AND KALSHI_LIVE_TRADING_ENABLED=true')
    db=init_db(args.db)
    mode=db.execute("SELECT value FROM runtime WHERE name='mode'").fetchone()
    if mode and mode[0]!=args.mode:
        raise RuntimeError(f'State DB belongs to {mode[0]} mode; use a NEW --db for {args.mode}')
    db.execute("INSERT OR IGNORE INTO runtime(name,value) VALUES('mode',?)",(args.mode,))
    existing_qty=db.execute("SELECT value FROM runtime WHERE name='base_qty'").fetchone()
    if existing_qty and existing_qty[0]!=str(args.base_qty):
        raise RuntimeError('Different base quantity already recorded in journal; use a NEW database')
    db.execute("INSERT OR IGNORE INTO runtime(name,value) VALUES('base_qty',?)",(str(args.base_qty),))
    db.commit()
    if db.execute("SELECT count(*) FROM orders WHERE status IN ('PENDING','UNKNOWN')").fetchone()[0]:
        raise RuntimeError('Unreconciled previous order exists; DO NOT RESTART until manually resolved')
    ensure_trade_identity_version(db)
    if args.mode=='live':
        # Prevent two live instances from opening parallel independent journals.
        import fcntl
        lockfile=Path('.collector_runs/w954_live_instance.lock')
        lockfile.parent.mkdir(parents=True,exist_ok=True)
        singleton_lock=lockfile.open('a+')
        try:
            fcntl.flock(singleton_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Another live 954B bot holds the single-instance lock') from e
    k=Kalshi(args.mode=='live')
    if args.mode=='live':
        shard_cash=k.balance()
        if shard_cash<D('5'):
            raise RuntimeError('Crypto index 2 available cash is below $5; no live orders permitted')
        log('LIVE_COLLATERAL_PREFLIGHT',crypto_index=CRYPTO_EXCHANGE_INDEX,
            available_cash=str(shard_cash),base_qty=args.base_qty,
            max_day_spend=str(args.max_day_spend))
    log('START',mode=args.mode,wallet=WALLET,poll_s=args.poll_seconds,db=args.db,
        base_qty=args.base_qty,exchange_index=CRYPTO_EXCHANGE_INDEX,
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
