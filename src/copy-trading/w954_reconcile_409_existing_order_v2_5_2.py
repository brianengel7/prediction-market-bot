#!/usr/bin/env python3
"""READ-ONLY by default. Reconcile the exact 954B FOK-409 attempt using
Kalshi's *existing* exchange order and its fills. No trading API POSTs.

Only terminal full execution (10/10) or terminal canceled zero-fill (0/10)
is eligible for --apply. Any ambiguous result fails closed. Preserves the
pre-existing 5 filled DOWN contracts and their recorded cost.
"""
from __future__ import annotations
import argparse
import fcntl
import runpy
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

CID = '78180f17-9855-4bad-b829-d7b1ffb0db69'
TICKER = 'KXBTC15M-26NOV121600-00'
MARKET_START = 1794516300
SIDE = 'DOWN'
QTY = Decimal('10')
INDEX = 2
DEFAULT_DB = '.collector_runs/w954_copy_live_5.sqlite3'
BOT_PATHS = (
    'src/copy-trading/w954_live_copy_v2_5_1_fok409_fix.py',
    'src/copy-trading/w954_live_copy_v2_5_order_verify.py',
)

def dec(value):
    x = Decimal(str(value))
    if not x.is_finite():
        raise ValueError('Invalid nonfinite quantity or cost')
    return x


def all_pages(k, path, key, params):
    import requests
    rows = []
    cursor = None
    seen = set()
    for page in range(30):
        p = {**params, 'limit': 200}
        if cursor: p['cursor'] = cursor
        for attempt in range(3):
            response = requests.get(k.base + path, params=p,
                                    headers=k.headers('GET', path), timeout=12)
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(1.25 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            break
        chunk = payload.get(key)
        if not isinstance(chunk, list):
            raise RuntimeError('Invalid Kalshi ' + key + ' response; refusing')
        rows.extend(chunk)
        cursor = payload.get('cursor')
        if not cursor:
            return rows
        if cursor in seen:
            raise RuntimeError('Repeated cursor; refusing incomplete audit')
        seen.add(cursor)
    raise RuntimeError('Incomplete Kalshi pagination; refusing')


def direction(obj, label):
    outcome = obj.get('outcome_side')
    book = obj.get('book_side')
    if outcome is not None and str(outcome).lower() != 'no':
        raise RuntimeError(f'{label}: outcome_side not NO; refusing')
    if book is not None and str(book).lower() != 'ask':
        raise RuntimeError(f'{label}: book_side not ask; refusing')
    if outcome is None and book is None:
        if (str(obj.get('action')).lower(), str(obj.get('side')).lower()) not in (
                ('sell', 'yes'), ('buy', 'no')):
            raise RuntimeError(f'{label}: ambiguous DOWN direction; refusing')


def assess(journal, orders, fills):
    matching = [o for o in orders if o.get('client_order_id') == CID]
    if len(matching) != 1:
        raise RuntimeError(f'Found {len(matching)} exchange orders for client ID; refusing')
    order = matching[0]
    oid = str(order.get('order_id') or '')
    if not oid or str(order.get('ticker')) != TICKER:
        raise RuntimeError('Exchange order identity mismatch; refusing')
    if order.get('exchange_index') is not None and int(order['exchange_index']) != INDEX:
        raise RuntimeError('Wrong exchange index on order; refusing')
    initial = order.get('initial_count_fp')
    if initial is not None and dec(initial) != QTY:
        raise RuntimeError(f'Initial count {initial} is not {QTY}; refusing')
    direction(order, 'order')
    these_fills = [f for f in fills if str(f.get('order_id')) == oid]
    if any(f.get('client_order_id') == CID and str(f.get('order_id')) != oid for f in fills):
        raise RuntimeError('Client ID is associated with another fill order ID; refusing')
    cost = Decimal('0')
    quantity = Decimal('0')
    seen_fills = set()
    for fill in these_fills:
        if str(fill.get('ticker', fill.get('market_ticker'))) != TICKER:
            raise RuntimeError('Fill ticker mismatch; refusing')
        if fill.get('exchange_index') is not None and int(fill['exchange_index']) != INDEX:
            raise RuntimeError('Fill exchange index mismatch; refusing')
        direction(fill, 'fill')
        fid = fill.get('fill_id') or fill.get('trade_id')
        if not fid or fid in seen_fills:
            raise RuntimeError('Missing or duplicate fill ID; refusing')
        seen_fills.add(fid)
        n = dec(fill['count_fp'])
        unit = dec(fill['no_price_dollars'])
        fee = dec(fill['fee_cost'])
        if n <= 0 or unit < 0 or unit > 1 or fee < 0:
            raise RuntimeError('Invalid fill quantity, price or fee; refusing')
        quantity += n
        cost += n * unit + fee
    reported = dec(order['fill_count_fp'])
    remaining = dec(order['remaining_count_fp'])
    status = str(order['status']).lower()
    if reported != quantity:
        raise RuntimeError(f'Order fill_count={reported} vs distinct fill records={quantity}; refusing')
    if remaining != 0:
        raise RuntimeError(f'Order has {remaining} remaining contracts; refusing')
    if status == 'canceled' and reported == 0 and not these_fills:
        final = 'NO_FILL'
        cost = Decimal('0')
    elif status == 'executed' and reported == QTY and quantity == QTY and cost > 0:
        final = 'FILLED'
        order_cost = sum((dec(order.get(k, '0')) for k in (
            'taker_fill_cost_dollars', 'maker_fill_cost_dollars',
            'taker_fees_dollars', 'maker_fees_dollars')), Decimal('0'))
        if order_cost > 0 and abs(order_cost-cost) > Decimal('0.05'):
            raise RuntimeError(f'Order cost {order_cost} differs from fill cost {cost}; refusing')
        cost = cost.quantize(Decimal('.000001'))
    else:
        raise RuntimeError(f'Nonreconcilable exchange status={status}, filled={reported}, remain={remaining}; refusing')
    return {'journal_decision': final, 'order_id': oid, 'status': status,
            'fill_count': reported, 'remaining': remaining, 'fill_records': len(these_fills),
            'verified_cost_dollars': cost, 'order': order}


def local_state(path):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        found = db.execute('SELECT * FROM orders WHERE client_id=?', (CID,)).fetchall()
        if len(found) != 1:
            raise RuntimeError('Expected exactly one journal row for target client')
        j = dict(found[0])
        mk = db.execute('SELECT * FROM markets WHERE market_start=?', (MARKET_START,)).fetchone()
        if mk is None:
            raise RuntimeError('Target market is missing')
        m = dict(mk)
        original = [dict(x) for x in db.execute(
            "SELECT client_id,qty,filled,actual_spent,status FROM orders "
            "WHERE market_start=? AND client_id<>?", (MARKET_START, CID))]
    if (j['market_start'] != MARKET_START or j['ticker'] != TICKER or
        j['side'] != SIDE or dec(j['qty']) != QTY or j['status'] != 'UNKNOWN' or
        j['order_id'] not in ('', None) or int(j['filled']) != 0 or
        dec(j['actual_spent']) != 0):
        raise RuntimeError('Local target order not exactly expected UNKNOWN; refusing')
    if (m['stopped'] != 1 or m['reason'] != 'order_unknown_requires_manual_reconciliation'
        or int(m['filled']) != 5 or dec(m['spent']) != Decimal('0.847100')
        or int(m['planned']) != 15 or int(m['confirmed']) != 1):
        raise RuntimeError('Existing market inventory/state changed; refusing automatic reconciliation')
    prior_filled = [x for x in original if x['status'] == 'FILLED']
    if (sum(int(x['filled']) for x in prior_filled) != 5 or
        sum((dec(x['actual_spent']) for x in prior_filled), Decimal(0)) != dec(m['spent'])):
        raise RuntimeError('Existing 5 filled contracts do not reconcile to market totals')
    return j, m, original


def self_test():
    j = {}
    base = {'order_id': 'ORDER-X', 'client_order_id': CID, 'ticker': TICKER,
            'exchange_index': 2, 'outcome_side': 'no', 'book_side': 'ask',
            'initial_count_fp': '10.00', 'remaining_count_fp': '0.00'}
    canceled = dict(base, status='canceled', fill_count_fp='0.00')
    res = assess(j, [canceled], [])
    assert res['journal_decision'] == 'NO_FILL'
    filled = dict(base, status='executed', fill_count_fp='10.00',
                  taker_fill_cost_dollars='2.8000', taker_fees_dollars='.1000')
    fill = {'order_id': 'ORDER-X', 'ticker': TICKER, 'fill_id': 'FILL-X',
            'exchange_index': 2, 'outcome_side': 'no', 'book_side': 'ask',
            'count_fp': '10.00', 'no_price_dollars': '.2800', 'fee_cost': '.1000'}
    res = assess(j, [filled], [fill])
    assert res['journal_decision'] == 'FILLED' and res['verified_cost_dollars'] == Decimal('2.900000')
    for order, fs in [(dict(base,status='canceled',fill_count_fp='1.00'),[]),
                      (canceled,[fill]),
                      (dict(filled, outcome_side='yes'),[fill]),
                      (filled,[fill,fill]),
                      (dict(base,status='resting',fill_count_fp='0.00',remaining_count_fp='10.00'),[])]:
        try: assess(j,[order],fs)
        except RuntimeError: pass
        else: raise AssertionError('Bad or uncertain order passed safety test')
    print('SELF_TEST_PASSED: canceled zero, verified full fill, inconsistent/partial/duplicate refused')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=DEFAULT_DB)
    parser.add_argument('--apply', action='store_true',
                        help='Make exactly one SQLite journal correction after independently checking Kalshi')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--expect-decision', choices=['FILLED','NO_FILL'],
                        help='Required with --apply: state independently reviewed in read-only output')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.apply and not args.expect_decision:
        parser.error('--apply requires --expect-decision FILLED or NO_FILL after read-only review')
    try:
        from dotenv import load_dotenv
        if Path('.env').is_file(): load_dotenv(dotenv_path='.env')
    except ImportError:
        pass
    j, m, others = local_state(args.db)
    print('LOCAL UNKNOWN:', {k:j[k] for k in (
        'client_id','order_id','ticker','side','qty','status','filled','actual_spent')})
    print('EXISTING POSITION:', {k:m[k] for k in ('filled','spent','planned','confirmed','stopped','reason')})
    botpath = next((Path(p) for p in BOT_PATHS if Path(p).is_file()), None)
    if botpath is None:
        raise RuntimeError('Missing existing v2.5 Kalshi client script')
    k = runpy.run_path(str(botpath))['Kalshi'](True)
    orders = all_pages(k, '/portfolio/orders', 'orders',
                       {'ticker': TICKER, 'exchange_index': INDEX})
    fills = all_pages(k, '/portfolio/fills', 'fills',
                      {'ticker': TICKER, 'exchange_index': INDEX})
    matching = [o for o in orders if o.get('client_order_id') == CID]
    for order in matching:
        keys = ('order_id','client_order_id','ticker','status','outcome_side',
                'book_side','side','action','initial_count_fp','fill_count_fp',
                'remaining_count_fp','taker_fill_cost_dollars','maker_fill_cost_dollars',
                'taker_fees_dollars','maker_fees_dollars')
        print('EXCHANGE MATCH:', {key:order.get(key) for key in keys})
    result = assess(j, orders, fills)
    print('EXCHANGE VERIFIED:', {key: str(value) for key,value in result.items() if key!='order'})
    print('OTHER ORDERS IN MARKET:', [(x['status'],x['filled'],x['actual_spent']) for x in others])
    if args.apply and args.expect_decision != result['journal_decision']:
        raise RuntimeError('Explicit expected decision disagrees with independently verified Kalshi outcome; refusing edits')
    if not args.apply:
        print('READ-ONLY. No orders placed and no journal edits.')
        print('Do NOT --apply until the verified decision and cost are reviewed.')
        return
    lockfile = Path('.collector_runs/w954_live_instance.lock')
    lockfile.parent.mkdir(parents=True,exist_ok=True)
    with lockfile.open('a+') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Live copier lock is held; refusing journal edits') from exc
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = Path(args.db+'.backup_existing409_'+stamp)
        if backup.exists(): raise RuntimeError('Backup exists; refusing overwrite')
        with sqlite3.connect(args.db) as original, sqlite3.connect(backup) as saved:
            original.backup(saved)
        print('SQLITE BACKUP:',backup)
        rw = sqlite3.connect(args.db, isolation_level=None, timeout=10)
        rw.row_factory = sqlite3.Row
        try:
            rw.execute('BEGIN IMMEDIATE')
            jc = rw.execute('SELECT * FROM orders WHERE client_id=?',(CID,)).fetchone()
            mc = rw.execute('SELECT * FROM markets WHERE market_start=?',(MARKET_START,)).fetchone()
            rest = [dict(x) for x in rw.execute(
                'SELECT client_id,qty,filled,actual_spent,status FROM orders WHERE market_start=? AND client_id<>?',
                (MARKET_START,CID))]
            if jc is None or mc is None or dict(jc)!=j or dict(mc)!=m or rest!=others:
                raise RuntimeError('Journal has changed since verification; refusing edits')
            new_filled = int(m['filled']) + int(result['fill_count'])
            new_cost = dec(m['spent']) + result['verified_cost_dollars']
            rw.execute('UPDATE orders SET order_id=?,status=?,filled=?,actual_spent=? WHERE client_id=? AND status=?',
                       (result['order_id'], result['journal_decision'],int(result['fill_count']),
                        str(result['verified_cost_dollars']), CID,'UNKNOWN'))
            rw.execute('UPDATE markets SET filled=?,spent=?,stopped=1,reason=? WHERE market_start=?',
                       (new_filled,str(new_cost),
                        'manual_reconciled_409_'+result['journal_decision'].lower(),MARKET_START))
            rw.execute('COMMIT')
        except Exception:
            rw.execute('ROLLBACK')
            raise
        finally:
            rw.close()
        print('APPLIED:', result['journal_decision'], 'order', result['order_id'],
              'filled_added',int(result['fill_count']), 'cost_added',result['verified_cost_dollars'])
        print('Existing five-contract position preserved; no new orders submitted.')
        print('Run --report and verify Unreconciled submissions: 0. Keep v2.5.1 live disabled.')

if __name__ == '__main__':
    main()
