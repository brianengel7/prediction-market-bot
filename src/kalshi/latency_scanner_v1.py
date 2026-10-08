"""Kalshi read-only orderbook/threshold scanner. NEVER places orders."""
import argparse
import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal as D

WS = 'wss://external-api-ws.kalshi.com/trade-api/ws/v2'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def number(value):
    x = D(str(value))
    if not x.is_finite():
        raise ValueError('Nonfinite price/quantity')
    return x


class Book:
    def __init__(self):
        self.levels = {'yes': {}, 'no': {}}
        self.ready = False
        self.updated = 0.0
        self.sent_ms = None

    def apply(self, typ, msg, sent_ms):
        if typ == 'orderbook_snapshot':
            for side in ('yes', 'no'):
                levels = {}
                for p, q in msg.get(f'{side}_dollars_fp', []):
                    p, q = number(p), number(q)
                    if not 0 < p < 1 or q < 0:
                        raise ValueError('Bad snapshot level')
                    if q:
                        levels[p] = q
                self.levels[side] = levels
            self.ready = True
        else:
            if not self.ready:
                raise RuntimeError('Delta before snapshot')
            side = msg['side']
            if side not in self.levels:
                raise ValueError('Unknown side')
            p, delta = number(msg['price_dollars']), number(msg['delta_fp'])
            if not 0 < p < 1:
                raise ValueError('Invalid delta price')
            side_levels = self.levels[side]
            quantity = side_levels.get(p, D(0)) + delta
            if quantity < 0:
                raise RuntimeError('Negative book quantity: reconnect')
            if quantity:
                side_levels[p] = quantity
            else:
                side_levels.pop(p, None)
        self.updated, self.sent_ms = time.monotonic(), sent_ms

    def bid(self, side):
        levels = self.levels[side]
        p = max(levels) if self.ready and levels else None
        return None if p is None else (p, levels[p])

    def ask(self, side):
        opposite = self.bid('no' if side == 'yes' else 'yes')
        return None if opposite is None else (D(1) - opposite[0], opposite[1])


def candidates(meta, books, freshness, fee):
    events = defaultdict(list)
    for ticker, m in meta.items():
        if (m.get('strike_type') or '').lower() == 'greater' and m.get('floor_strike') is not None:
            events[m['event_ticker']].append((number(m['floor_strike']), ticker))
    t = time.monotonic()
    for event, strikes in events.items():
        strikes.sort()
        for i, (low, lt) in enumerate(strikes):
            a = books.get(lt)
            if not a or not a.ready or t - a.updated > freshness or not a.ask('yes'):
                continue
            for high, ht in strikes[i+1:]:
                b = books.get(ht)
                if high <= low or not b or not b.ready or t - b.updated > freshness or not b.ask('no'):
                    continue
                yp, yq = a.ask('yes')
                np, nq = b.ask('no')
                gross = D(1) - yp - np
                if min(yq, nq) >= 1 and gross > 0 and gross - 2 * fee >= D('0.02'):
                    yield {'event': event, 'yes': lt, 'no': ht,
                           'yes_ask': str(yp), 'no_ask': str(np),
                           'gross_margin': str(gross), 'fee_buffer_per_leg': str(fee),
                           'margin_after_assumed_fee': str(gross - 2 * fee),
                           'visible_pairs': str(min(yq, nq)),
                           'RULES_AND_EXECUTION_NOT_VERIFIED': True}


def database_write(rows):
    from src.database.db import get_connection
    with get_connection() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS latency_scanner_events (
            id TEXT PRIMARY KEY, received_at_utc TEXT NOT NULL,
            kind TEXT NOT NULL, ticker TEXT, event_ticker TEXT,
            payload_json TEXT NOT NULL)''')
        for row in rows:
            con.execute('''INSERT INTO latency_scanner_events
                (id,received_at_utc,kind,ticker,event_ticker,payload_json)
                VALUES (?,?,?,?,?,?)''', row)


def event_row(kind, ticker, event, payload):
    return (str(uuid.uuid4()), now(), kind, ticker, event, json.dumps(payload))


def discover(series, maximum):
    from src.kalshi.client import get_markets
    lists = {}
    for s in series:
        items = get_markets(series_ticker=s, status='open')['markets']
        lists[s] = sorted(items, key=lambda m: (m.get('close_time') or '', m.get('ticker') or ''))
        print(f'{s}: {len(items)} open markets', flush=True)
    selected = {}
    for i in range(max(map(len, lists.values()), default=0)):
        for s in series:
            if i >= len(lists[s]):
                continue
            m = lists[s][i]
            if m.get('ticker') and len(selected) < maximum:
                selected[m['ticker']] = m
    return selected


async def collect(args, meta, deadline):
    import websockets
    from src.kalshi.trading_client import build_auth_headers
    books, sequences, sampled, announced = {}, {}, {}, {}
    rows = []
    restart_at = time.monotonic() + args.refresh_minutes * 60
    last_flush = time.monotonic()
    async with websockets.connect(WS, additional_headers=build_auth_headers('GET', WS),
                                  ping_interval=20, ping_timeout=20, max_queue=2048) as ws:
        await ws.send(json.dumps({'id': 1, 'cmd': 'subscribe',
                                  'params': {'channels': ['orderbook_delta'],
                                             'market_tickers': list(meta)}}))
        print(f'Subscribed to {len(meta)} books; READ ONLY.', flush=True)
        try:
            while time.monotonic() < min(deadline, restart_at):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1)
                except asyncio.TimeoutError:
                    raw = None
                if raw:
                    msg = json.loads(raw)
                    typ = msg.get('type')
                    if typ == 'error':
                        raise RuntimeError(f'WebSocket rejected subscription: {msg.get("msg")}')
                    if typ in ('orderbook_snapshot', 'orderbook_delta'):
                        sid, seq = msg['sid'], msg['seq']
                        if sid in sequences and seq != sequences[sid] + 1:
                            raise RuntimeError(f'Sequence gap sid={sid}: {sequences[sid]}->{seq}')
                        sequences[sid] = seq
                        data = msg['msg']
                        ticker = data['market_ticker']
                        if ticker not in meta:
                            continue
                        book = books.setdefault(ticker, Book())
                        book.apply(typ, data, msg.get('sending_ts_ms'))
                        t = time.monotonic()
                        if t - sampled.get(ticker, -1e12) >= args.sample_seconds:
                            payload = {'yes_bid': book.bid('yes'), 'no_bid': book.bid('no'),
                                       'yes_ask': book.ask('yes'), 'no_ask': book.ask('no'),
                                       'ws_sent_ms': book.sent_ms,
                                       'series': next((s for s in args.series if ticker.startswith(s + '-')), None)}
                            rows.append(event_row('BOOK_SAMPLE', ticker,
                                                  meta[ticker].get('event_ticker'),
                                                  {k: [str(v[0]), str(v[1])] if isinstance(v, tuple) else v
                                                   for k, v in payload.items()}))
                            sampled[ticker] = t
                        for candidate in candidates(meta, books, args.max_age_seconds, D(args.fee_per_leg)):
                            key = candidate['yes'], candidate['no']
                            if t - announced.get(key, -1e12) >= 30:
                                announced[key] = t
                                print('POTENTIAL ARB (NOT VERIFIED):', json.dumps(candidate), flush=True)
                                rows.append(event_row('ARB_CANDIDATE', candidate['yes'],
                                                      candidate['event'], candidate))
                if rows and time.monotonic() - last_flush >= 5:
                    await asyncio.to_thread(database_write, rows)
                    print(f'Saved {len(rows)} rows to database', flush=True)
                    rows = []
                    last_flush = time.monotonic()
        finally:
            if rows:
                await asyncio.to_thread(database_write, rows)


async def run(args):
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv('PREDICTION_DATABASE_URL'):
        raise RuntimeError('PREDICTION_DATABASE_URL required; refusing local SQLite fallback')
    await asyncio.to_thread(database_write, [])
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else float('inf')
    while time.monotonic() < deadline:
        try:
            meta = await asyncio.to_thread(discover, args.series, args.max_markets)
            if not meta:
                print('No open markets; retry discovery in 60 seconds', flush=True)
                await asyncio.sleep(60)
                continue
            await collect(args, meta, deadline)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}; invalidating books, retrying in 5 seconds', flush=True)
            await asyncio.sleep(5)
    print('Stopped. No orders were placed.', flush=True)


def test():
    a, b = Book(), Book()
    a.apply('orderbook_snapshot', {'yes_dollars_fp': [['0.45', '5']],
                                   'no_dollars_fp': [['0.60', '3']]}, None)
    b.apply('orderbook_snapshot', {'yes_dollars_fp': [['0.55', '7']],
                                   'no_dollars_fp': [['0.42', '4']]}, None)
    assert a.ask('yes') == (D('.4'), D('3'))
    meta = {'low': {'event_ticker': 'E', 'strike_type': 'greater', 'floor_strike': 70},
            'high': {'event_ticker': 'E', 'strike_type': 'greater', 'floor_strike': 71}}
    result = list(candidates(meta, {'low': a, 'high': b}, 10, D('.03')))
    assert len(result) == 1 and result[0]['gross_margin'] == '0.15'
    a.apply('orderbook_delta', {'side': 'no', 'price_dollars': '0.60', 'delta_fp': '-3'}, None)
    assert a.ask('yes') is None
    assert not list(candidates(meta, {'low': a, 'high': b}, 10, D('.03')))
    print('PASS: order-book arithmetic, delta updates, fees, and pair detection')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--series', action='append', default=[])
    p.add_argument('--minutes', type=float, default=0, help='0 = continuous')
    p.add_argument('--max-markets', type=int, default=60)
    p.add_argument('--refresh-minutes', type=float, default=10)
    p.add_argument('--sample-seconds', type=float, default=30)
    p.add_argument('--max-age-seconds', type=float, default=30)
    p.add_argument('--fee-per-leg', default='0.03')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return test()
    if not args.series:
        p.error('Specify --series (repeatable)')
    if min(args.max_markets, args.refresh_minutes, args.sample_seconds, args.max_age_seconds) <= 0:
        p.error('Limits and intervals must be positive')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
