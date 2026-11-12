#!/usr/bin/env python3
"""954B LIVE unknown-order reconciliation. READ ONLY by default.

--apply requires an independently verified, final, fully-filled or zero-fill
order on Kalshi, matching the journal order_id/client_id, and complete fills.
No POST requests, no orders, no transfers. If uncertain: STOP, DO NOT PATCH.
"""
from __future__ import annotations
import argparse, runpy, sqlite3, time
from pathlib import Path
from decimal import Decimal as D
from datetime import datetime, timezone

HERE=Path('/workspaces/prediction-market-bot')
DEFAULT_DB='.collector_runs/w954_copy_live_5.sqlite3'
INDEX=2

def d(x):
    v=D(str(x))
    if not v.is_finite(): raise ValueError('Bad monetary value')
    return v

def pages(k,path,root,params,limit_pages=30):
    import requests
    cursor=None
    for i in range(limit_pages):
        query=dict(params,limit=200)
        if cursor:query['cursor']=cursor
        # Bounded retries apply only to read-only GET calls.
        for a in range(3):
            r=requests.get(k.base+path,params=query,
                headers=k.headers('GET',path),timeout=10)
            if r.status_code in (429,502,503,504) and a<2:
                time.sleep(1.2*(a+1))
                continue
            r.raise_for_status()
            data=r.json()
            break
        rows=data.get(root)
        if not isinstance(rows,list):raise RuntimeError(f'Malformed {root} response')
        yield from rows
        cursor=data.get('cursor')
        if not cursor:return
    raise RuntimeError(f'{root} pagination incomplete; cannot safely reconcile')

def confirm_direction(order, fills, journal):
    """Verify exposure using canonical Kalshi V2 direction fields.

    V2 bid=>UP/YES and ask=>DOWN/NO. The legacy action for an
    ask/sell-YES fill may legitimately be "sell" even when opening NO.
    Never classify an ask as a buy-YES based on action alone.
    """
    expected_outcome = 'yes' if journal['side'] == 'UP' else 'no'
    expected_book = 'bid' if journal['side'] == 'UP' else 'ask'
    allowed_legacy = ({('buy','yes'), ('sell','no')} if expected_outcome == 'yes'
                      else {('buy','no'), ('sell','yes')})
    for label, obj in [('order', order)] + [(f'fill_{i+1}', f) for i, f in enumerate(fills)]:
        outcome = obj.get('outcome_side')
        book = obj.get('book_side')
        if outcome is not None and str(outcome).lower() != expected_outcome:
            raise RuntimeError(f'{label}: outcome_side={outcome!r} disagrees with journal {journal["side"]}')
        if book is not None and str(book).lower() != expected_book:
            raise RuntimeError(f'{label}: book_side={book!r} disagrees with expected {expected_book}')
        # Canonical direction, when present, is definitive. The old action/side
        # representation is deprecated and may indicate selling YES to open NO.
        if outcome is None and book is None:
            action, side = obj.get('action'), obj.get('side')
            if (action, side) not in allowed_legacy:
                raise RuntimeError(f'{label}: no valid canonical direction or legacy action/side; no journal edits')
    return expected_outcome, expected_book


def verify(k,journal):
    oid=journal['order_id']; ticker=journal['ticker']; cid=journal['client_id']
    if not oid: raise RuntimeError('No exchange order_id saved; manual investigation required')
    orders=[x for x in pages(k,'/portfolio/orders','orders',
                              {'ticker':ticker,'exchange_index':INDEX})
            if str(x.get('order_id'))==oid]
    if len(orders)!=1: raise RuntimeError(f'Found {len(orders)} matching exchange orders; cannot reconcile')
    order=orders[0]
    if (str(order.get('client_order_id'))!=cid or
        str(order.get('ticker'))!=ticker or
        int(order.get('exchange_index',-1))!=INDEX):
        raise RuntimeError('Order identity mismatch; no journal edits')
    fills=list(pages(k,'/portfolio/fills','fills',
                     {'ticker':ticker,'order_id':oid}))
    if any(str(f.get('order_id'))!=oid or str(f.get('ticker',f.get('market_ticker')))!=ticker for f in fills):
        raise RuntimeError('Unexpected fill order ID/ticker; no journal edits')
    count=sum((d(f['count_fp']) for f in fills),D('0'))
    status=str(order.get('status'))
    reported=d(order['fill_count_fp'])
    remain=d(order['remaining_count_fp'])
    qty=d(journal['qty'])
    if reported!=count:
        raise RuntimeError(f'Exchange order filled={reported} but fills total={count}; no journal edits')
    if status=='executed' and reported==qty and remain==0:
        side, _ = confirm_direction(order, fills, journal)
        cost=D('0')
        seen_ids=set()
        for fill in fills:
            fid=fill.get('fill_id') or fill.get('trade_id')
            if fid is not None:
                if fid in seen_ids: raise RuntimeError('Duplicate fill ID in response; no journal edits')
                seen_ids.add(fid)
            if fill.get('exchange_index') is not None and int(fill['exchange_index'])!=INDEX:
                raise RuntimeError('Fill exchange index mismatch; no journal edits')
            shares=d(fill['count_fp'])
            unit=d(fill[side+'_price_dollars'])
            fee=d(fill['fee_cost'])
            if not (shares>0 and D('0')<=unit<=D('1') and fee>=0):
                raise RuntimeError('Invalid fill quantity, price, or fee; no journal edits')
            cost+=shares*unit+fee
        cost=cost.quantize(D('0.000001'))
        if cost<=0:
            raise RuntimeError('No positive verified purchase cost; no journal edits')
        order_cost=sum((d(order.get(k,'0')) for k in ('taker_fill_cost_dollars',
            'maker_fill_cost_dollars','taker_fees_dollars','maker_fees_dollars')),D('0'))
        if order_cost>0 and abs(order_cost-cost)>D('0.05'):
            raise RuntimeError(f'Order vs fill cost disagree ({order_cost} vs {cost}); no journal edits')
        return 'FILLED',qty,cost,order,count,len(fills)
    if status=='canceled' and reported==0 and remain==0 and count==0:
        return 'NO_FILL',D('0'),D('0'),order,count,len(fills)
    raise RuntimeError(f'Order not terminal/full or zero canceled: status={status}, fill={reported}, remain={remain}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db',default=DEFAULT_DB)
    ap.add_argument('--order-id',default='01a1d05a-1900-780e-bb72-8edfdaa13b0a')
    ap.add_argument('--apply',action='store_true',help='Explicitly commit verified state to existing journal after SQLite backup')
    args=ap.parse_args()
    try:
        from dotenv import load_dotenv
        if Path('.env').exists():load_dotenv(dotenv_path=Path('.env'))
    except ImportError:pass
    db=sqlite3.connect('file:'+args.db+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    rows=db.execute('SELECT * FROM orders WHERE order_id=? AND status IN ("UNKNOWN","PENDING")',(args.order_id,)).fetchall()
    if len(rows)!=1:
        raise RuntimeError(f'Expected exactly one UNKNOWN/PENDING record for {args.order_id}, found {len(rows)}')
    j=dict(rows[0]);db.close()
    print('LOCAL JOURNAL:',{k:j[k] for k in ('client_id','order_id','ticker','side','qty','status','filled','actual_spent')})
    bot_path=Path('src/copy-trading/w954_live_copy_v2_4_zero_fill.py')
    if not bot_path.exists():bot_path=Path('src/copy-trading/w954_live_copy_v2_4_1_balance_fix.py')
    b=runpy.run_path(str(bot_path))
    k=b['Kalshi'](True)
    state, filled, cost, exchange, total, n=verify(k,j)
    print('VERIFIED EXCHANGE:',{'order_id':exchange['order_id'],'status':exchange['status'],
        'fill_count':str(total),'remaining_count':exchange['remaining_count_fp'],
        'fill_records':n,'exchange_index':exchange['exchange_index'],
        'journal_decision':state,'verified_cost_dollars':str(cost)})
    if not args.apply:
        print('READ-ONLY COMPLETE. If identifiers and fills are correct, re-run with --apply to reconcile this exact order.')
        return
    # Backup first, using SQLite online backup to include any WAL data safely.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=Path(args.db+'.backup_'+stamp)
    if backup.exists():raise RuntimeError('Backup path already exists; refusing to overwrite')
    with sqlite3.connect(args.db) as rw, sqlite3.connect(backup) as bak:
        rw.backup(bak)
    print('BACKUP:',backup)
    rw=sqlite3.connect(args.db,timeout=5,isolation_level=None)
    try:
        rw.execute('BEGIN IMMEDIATE')
        cur=rw.execute('SELECT status, filled, actual_spent, market_start, qty FROM orders WHERE client_id=? AND order_id=?',
            (j['client_id'],j['order_id'])).fetchone()
        if not cur or cur[0] not in ('UNKNOWN','PENDING') or cur[1]!=0 or d(cur[2])!=0 or cur[4]!=j['qty']:
            raise RuntimeError('Journal order changed since verification; refusing apply')
        mk=rw.execute('SELECT filled,spent FROM markets WHERE market_start=?',(cur[3],)).fetchone()
        if not mk or mk[0]!=0 or d(mk[1])!=0:
            raise RuntimeError('Market journal not empty; refusing apply to prevent double-counting')
        rw.execute('UPDATE orders SET status=?,filled=?,actual_spent=? WHERE client_id=? AND order_id=?',
            (state,int(filled),str(cost),j['client_id'],j['order_id']))
        rw.execute('UPDATE markets SET planned=CASE WHEN planned<? THEN ? ELSE planned END, filled=?, spent=?, stopped=1, reason=? WHERE market_start=?',
            (int(filled),int(filled),int(filled),str(cost),
             'manual_reconciled_verified_fill' if state=='FILLED' else 'manual_reconciled_no_fill',cur[3]))
        rw.execute('COMMIT')
    except Exception:
        rw.execute('ROLLBACK')
        raise
    finally:rw.close()
    print('APPLIED:',state,'for',j['client_id'],'; market stopped; no new orders created.')
    print('IMPORTANT: Re-run the bot --report and confirm Unreconciled submissions: 0 before any live restart.')

if __name__=='__main__':main()
