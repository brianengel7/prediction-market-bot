#!/usr/bin/env python3
"""Audit one explicitly rejected Kalshi FOK 409; no orders sent, read-only by default.

The HTTP 409 / fill_or_kill_insufficient_resting_volume error was independently
observed in the operator console. An API search showing no order is supporting
cross-check evidence, not by itself a proof that no order was submitted.

Optional --apply requires the exact error-code acknowledgment, completes read-only
Kalshi order/fill checks, backs up SQLite, updates only that rejected UNKNOWN row,
and preserves any already-filled earlier orders in the same market.
"""
from __future__ import annotations
import argparse, fcntl, runpy, sqlite3, time
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

TARGET_CLIENT='78180f17-9855-4bad-b829-d7b1ffb0db69'
EXPECTED_MARKET=1794516300
EXPECTED_TICKER='KXBTC15M-26NOV121600-00'
EXPECTED_SIDE='DOWN'
EXCHANGE_INDEX=2
CODE='fill_or_kill_insufficient_resting_volume'
DEFAULT_DB='.collector_runs/w954_copy_live_5.sqlite3'


def get_pages(k,path,key,params):
    import requests
    cursor=None
    out=[]
    for page in range(30):
        query={**params,'limit':200}
        if cursor: query['cursor']=cursor
        for attempt in range(3):
            response=requests.get(k.base+path,params=query,
                headers=k.headers('GET',path),timeout=10)
            if response.status_code in (429,502,503,504) and attempt<2:
                time.sleep(1.25*(attempt+1))
                continue
            response.raise_for_status()
            data=response.json()
            break
        chunk=data.get(key)
        if not isinstance(chunk,list):
            raise RuntimeError('Malformed exchange '+key+' response; no reconciliation')
        out.extend(chunk)
        cursor=data.get('cursor')
        if not cursor: return out
    raise RuntimeError('Incomplete exchange '+key+' pagination; no reconciliation')


def event_time(obj):
    val=obj.get('created_time')
    if isinstance(val,str) and val:
        try: return datetime.fromisoformat(val.replace('Z','+00:00')).timestamp()
        except ValueError: pass
    val=obj.get('ts')
    if isinstance(val,(int,float)):
        n=float(val)
        return n/1000 if n>1e11 else n
    return None


def assess(j,orders,fills):
    """Fail closed if the rejected client may have an accepted order or fill."""
    cid=j['client_id']; event_ts=float(j['created_at'])
    matching_orders=[x for x in orders if x.get('client_order_id')==cid]
    if matching_orders:
        raise RuntimeError(f'{len(matching_orders)} EXCHANGE ORDER(S) match this client ID; use normal exchange-order reconciliation')
    # Every nearby fill must have an order ID which belongs to a known distinct
    # client order. A fill with unknown attribution could belong to this attempt.
    by_oid={str(o['order_id']):o for o in orders if o.get('order_id')}
    nearby=[]
    for fill in fills:
        ft=event_time(fill)
        if ft is not None and abs(ft-event_ts)<=120:
            nearby.append(fill)
            oid=fill.get('order_id')
            owner=by_oid.get(str(oid)) if oid else None
            if owner is None or not owner.get('client_order_id') or owner['client_order_id']==cid:
                raise RuntimeError('Nearby fill lacks a known DIFFERENT client order identity; no reconciliation')
        elif fill.get('client_order_id')==cid:
            raise RuntimeError('A fill matches rejected client ID; no reconciliation')
    return len(nearby)


def self_test():
    j={'client_id':TARGET_CLIENT,'created_at':1700000000}
    separate=[{'order_id':'INITIAL','client_order_id':'ANOTHER','ticker':EXPECTED_TICKER}]
    f=[{'order_id':'INITIAL','created_time':'2023-11-14T22:13:20Z'}]
    assert assess(j,separate,f)==1
    assert assess(j,[],[])==0
    try:assess(j,[{'order_id':'EXISTS','client_order_id':TARGET_CLIENT}],[])
    except RuntimeError:pass
    else:raise AssertionError('Matching accepted order was not rejected')
    try:assess(j,[],f)
    except RuntimeError:pass
    else:raise AssertionError('Orphan nearby fill was not rejected')
    print('SELF_TEST_PASSED: matching order blocked; orphan fill blocked; other known initial fill preserved')


def read_local(path,cid):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as ro:
        ro.row_factory=sqlite3.Row
        rows=ro.execute('SELECT * FROM orders WHERE client_id=?',(cid,)).fetchall()
        if len(rows)!=1: raise RuntimeError('Expected exactly one journal order for specified client_id')
        j=dict(rows[0])
        market=ro.execute('SELECT * FROM markets WHERE market_start=?',(j['market_start'],)).fetchone()
        if market is None:raise RuntimeError('Journal market row missing')
        m=dict(market)
        if (j['market_start']!=EXPECTED_MARKET or j['ticker']!=EXPECTED_TICKER or
            j['side']!=EXPECTED_SIDE or j['status']!='UNKNOWN' or
            j['order_id'] not in ('',None) or int(j['filled'])!=0 or
            Decimal(str(j['actual_spent']))!=0):
            raise RuntimeError('Unexpected journal identity/status/quantity; no reconciliation')
        if m['stopped']!=1 or m['reason']!='order_unknown_requires_manual_reconciliation':
            raise RuntimeError('Market stop reason changed; no reconciliation')
        return j,m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',default=DEFAULT_DB)
    parser.add_argument('--client-id',default=TARGET_CLIENT)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--confirm-error-code',default='',help='Required with --apply, exact observed 409 error code')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:return self_test()
    if args.client_id!=TARGET_CLIENT:
        raise RuntimeError('Tool is specific to one rejected order; refusing other client IDs')
    if args.apply and args.confirm_error_code!=CODE:
        parser.error('--apply requires --confirm-error-code '+CODE)
    try:
        from dotenv import load_dotenv
        if Path('.env').is_file():load_dotenv(dotenv_path=Path('.env'))
    except ImportError:pass
    j,m=read_local(args.db,args.client_id)
    print('LOCAL ORDER:',{x:j[x] for x in ('client_id','order_id','ticker','side','qty','status','filled','actual_spent','created_at')})
    print('EXISTING MARKET:',{x:m[x] for x in ('filled','spent','planned','confirmed','stopped','reason')})
    # Authentication via existing bot, but never start main() or send an order.
    code_path=Path('src/copy-trading/w954_live_copy_v2_5_1_fok409_fix.py')
    if not code_path.is_file(): code_path=Path('src/copy-trading/w954_live_copy_v2_5_order_verify.py')
    if not code_path.is_file():raise RuntimeError('Cannot locate the existing Kalshi client')
    k=runpy.run_path(str(code_path))['Kalshi'](True)
    orders=get_pages(k,'/portfolio/orders','orders',{'ticker':j['ticker'],'exchange_index':EXCHANGE_INDEX})
    fills=get_pages(k,'/portfolio/fills','fills',{'ticker':j['ticker'],'exchange_index':EXCHANGE_INDEX})
    # Check consistency of ticker and shard for all records returned by the filtered APIs.
    for obj in orders+fills:
        if obj.get('ticker',obj.get('market_ticker'))!=j['ticker']:
            raise RuntimeError('Unexpected ticker returned by filtered endpoint; no reconciliation')
        if obj.get('exchange_index') is not None and int(obj['exchange_index'])!=EXCHANGE_INDEX:
            raise RuntimeError('Unexpected exchange index; no reconciliation')
    nearby=assess(j,orders,fills)
    print('EXCHANGE CHECK:',{'total_orders_for_ticker':len(orders),'total_fills_for_ticker':len(fills),
        'orders_with_rejected_client_id':0,'nearby_fills_attributed_to_other_orders':nearby})
    print('OBSERVED CONSOLE REJECTION:',CODE,'(HTTP 409; must independently match your original terminal log)')
    print('RECONCILIATION CANDIDATE: REJECTED with zero fills; preserve existing market holdings and spent')
    if not args.apply:
        print('READ-ONLY COMPLETE. Apply only if the explicit 409 log and exchange cross-check both match.')
        return
    # Refuse while any other 954B live instance holds the singleton lock.
    lock_path=Path('.collector_runs/w954_live_instance.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('A live 954B bot is still running; no journal edits') from e
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=Path(args.db+'.backup_fok409_'+stamp)
        with sqlite3.connect(args.db) as old,sqlite3.connect(backup) as saved:
            old.backup(saved)
        print('SQLITE BACKUP:',backup)
        rw=sqlite3.connect(args.db,isolation_level=None,timeout=10)
        rw.row_factory=sqlite3.Row
        try:
            rw.execute('BEGIN IMMEDIATE')
            live=rw.execute('SELECT * FROM orders WHERE client_id=?',(j['client_id'],)).fetchone()
            mk=rw.execute('SELECT * FROM markets WHERE market_start=?',(j['market_start'],)).fetchone()
            if live is None or mk is None or dict(live)!=j or dict(mk)!=m:
                raise RuntimeError('Journal changed since verification; refusing to modify')
            rw.execute('UPDATE orders SET status=?,filled=0,actual_spent=? WHERE client_id=? AND status=?',
                       ('REJECTED','0',j['client_id'],'UNKNOWN'))
            rw.execute('UPDATE markets SET reason=? WHERE market_start=? AND stopped=1',
                       ('manual_reconciled_fok_insufficient_resting_volume',j['market_start']))
            rw.execute('COMMIT')
        except Exception:
            rw.execute('ROLLBACK')
            raise
        finally:rw.close()
        print('APPLIED: only rejected follow-up updated to REJECTED; existing 5-contract FILLED order untouched')
        print('NEXT: --report must show Unreconciled submissions: 0 before considering live restart')

if __name__=='__main__':main()
