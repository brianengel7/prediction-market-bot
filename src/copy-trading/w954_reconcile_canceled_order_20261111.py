#!/usr/bin/env python3
"""One-time journal reconciliation of 954B Nov 11 Kalshi canceled zero-fill FOK.

Default is read-only with respect to SQLite. --apply writes ONLY the specified
UNKNOWN order, once, after authenticated Kalshi GET verifies the canceled
zero-fill state and a complete no-fills response. NEVER posts orders.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import runpy
import sqlite3
import sys

CLIENT = 'dfe14e7b-c101-49e8-ba4f-ebbaa969f85c'
ORDER = '01a1cd31-a428-76e8-854c-cb66d019e117'
TICKER = 'KXBTC15M-26NOV111945-45'
START = 1794443400
DB_PATH = Path('.collector_runs/w954_copy_live_5.sqlite3')
BOT = Path('src/copy-trading/w954_live_copy_v2_4_zero_fill.py')


def reconcile(apply: bool) -> None:
    import requests
    try:
        from dotenv import load_dotenv
        p=Path('.env')
        if p.exists(): load_dotenv(dotenv_path=str(p.resolve()))
    except ImportError:
        pass
    if not DB_PATH.is_file():
        raise RuntimeError('Live journal does not exist: ' + str(DB_PATH))
    if not BOT.is_file():
        raise RuntimeError('Install v2.4 first: ' + str(BOT))

    # The live copier takes the same lock. Fail if it is running.
    lockpath=Path('.collector_runs/w954_live_instance.lock')
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Live bot lock is held; stop bot before reconciliation') from e
        mod=runpy.run_path(str(BOT))
        k=mod['Kalshi'](True)
        # Verify against the actual exchange, NOT the local journal alone.
        order=k.verified_no_fill(ORDER,CLIENT,TICKER)
        print('Kalshi GET /portfolio/orders/{order_id}:')
        print({key:order.get(key) for key in (
            'order_id','client_order_id','ticker','status','exchange_index',
            'fill_count_fp','remaining_count_fp','created_time')})
        path='/portfolio/fills'
        resp=requests.get(k.base+path,params={'order_id':ORDER,'exchange_index':2,'limit':1000},
                          headers=k.headers('GET',path),timeout=10)
        resp.raise_for_status()
        body=resp.json()
        if 'fills' not in body or not isinstance(body['fills'],list):
            raise RuntimeError('Kalshi fills response malformed')
        if body['fills'] or body.get('cursor'):
            raise RuntimeError('Unexpected fills or incomplete pagination; DO NOT reconcile')
        print('Kalshi GET /portfolio/fills: verified 0 fills for order')

        uri='file:'+str(DB_PATH.resolve())+('?mode=rw' if apply else '?mode=ro')
        db=sqlite3.connect(uri,uri=True)
        db.row_factory=sqlite3.Row
        try:
            rows=db.execute('''SELECT * FROM orders WHERE client_id=?''',(CLIENT,)).fetchall()
            if len(rows)!=1: raise RuntimeError('Expected exactly one matching local order')
            row=dict(rows[0]); print('Local journal:',{key:row.get(key) for key in (
                'client_id','order_id','market_start','ticker','side','qty','price',
                'status','filled','actual_spent')})
            if (row['ticker']!=TICKER or row['market_start']!=START or row['side']!='UP'
                or int(row['qty'])!=5 or row['price']!='0.3400'
                or row['status']!='UNKNOWN' or int(row['filled'])!=0
                or float(row['actual_spent'])!=0.0
                or (row['order_id'] and row['order_id']!=ORDER)):
                raise RuntimeError('Journal does not match the one authorized reconciliation target')
            mode=db.execute("SELECT value FROM runtime WHERE name='mode'").fetchone()
            if not mode or mode[0]!='live':
                raise RuntimeError('This is not the expected LIVE journal')
            if not apply:
                print('DRY RUN PASSED. To update this one local journal row, run with --apply.')
                return
            backup=DB_PATH.with_name(DB_PATH.name+'.backup-before-zero-fill-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
            with sqlite3.connect(backup) as dest:
                db.backup(dest)
            print('SQLite backup:',backup)
            # Single transaction; WHERE enforces original UNKNOWN state.
            db.execute('BEGIN IMMEDIATE')
            updated=db.execute('''UPDATE orders
                SET status='NO_FILL', filled=0, actual_spent='0', order_id=?
                WHERE client_id=? AND status='UNKNOWN' AND filled=0''',(ORDER,CLIENT))
            if updated.rowcount!=1:
                db.rollback()
                raise RuntimeError('Journal changed during reconciliation')
            db.execute('''UPDATE markets SET stopped=1,reason='kalshi_fok_no_fill_reconciled'
                          WHERE market_start=?''',(START,))
            db.execute('''INSERT INTO observations
                (ts,trade_id,market_start,age_s,kind,side,qty,wallet_price,
                 kalshi_ask,depth,detail) VALUES
                (strftime('%s','now'),?,?,?,?,?,?,?,?,?,?)''',
                (row['trade_id'],START,0,'EXCHANGE_ZERO_FILL_RECONCILED','UP',0,
                 row['price'],None,None,'Kalshi canceled; 0 fills; order_id='+ORDER))
            db.commit()
            left=db.execute("SELECT COUNT(*) FROM orders WHERE status IN ('UNKNOWN','PENDING')").fetchone()[0]
            print('SUCCESS: recorded NO_FILL; order ID saved; previous order preserved.')
            print('Other unresolved submissions:',left)
            if left: print('Keep live trading disabled until ALL unresolved submissions are reconciled.')
        finally:
            db.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply',action='store_true',help='After exchange verification, back up and update exactly one journal order')
    args=ap.parse_args()
    reconcile(args.apply)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('SAFE ABORT:',str(e),file=sys.stderr)
        sys.exit(1)
