"""Collect the public Kalshi hourly-weather index with receipt-time audit.

Research only. Never places orders. The endpoint is *Kalshi's published*
index: this collector measures dissemination and is NOT an earlier external feed.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

API = 'https://external-api.kalshi.com/trade-api/v2/live_data/weather'


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def require_payload(data, city):
    if not isinstance(data, dict) or not isinstance(data.get('timeseries'), list):
        raise ValueError(f'{city}: missing weather timeseries')
    if data.get('city') and data['city'] != city:
        raise ValueError(f'{city}: unexpected city {data["city"]}')
    return data


def station_receipt_range(point):
    stamps = [int(s['received_at_ms']) for s in point.get('stations', [])
              if s.get('received_at_ms') is not None]
    return (min(stamps), max(stamps)) if stamps else (None, None)


def point_record(city, point, config_version, first_seen):
    # Kalshi's t is the event-minute timestamp, not when this API published it.
    ms = point.get('t')
    if not isinstance(ms, (int, float)) or not 1_000_000_000_000 <= ms <= 9_999_999_999_999:
        raise ValueError(f'{city}: unexpected event-minute timestamp {ms!r}')
    if point.get('v') is None and point.get('status') != 'incomplete':
        raise ValueError(f'{city}: missing v outside incomplete point')
    raw = encode(point)
    earliest, latest = station_receipt_range(point)
    return (city, int(ms), hashlib.sha256(raw.encode()).hexdigest(), first_seen,
            str(point.get('status') or ''),
            float(point['v']) if point.get('v') is not None else None,
            config_version, earliest, latest, raw)


def setup_database():
    from src.database.db import get_connection
    with get_connection() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS hourly_weather_index_versions (
            city TEXT NOT NULL, index_minute_ms BIGINT NOT NULL,
            payload_sha256 TEXT NOT NULL, first_seen_utc TEXT NOT NULL,
            status TEXT NOT NULL, value_f DOUBLE PRECISION,
            config_version TEXT, station_receipt_first_ms BIGINT,
            station_receipt_last_ms BIGINT, raw_json TEXT NOT NULL,
            PRIMARY KEY (city, index_minute_ms, payload_sha256))''')
        con.execute('''CREATE TABLE IF NOT EXISTS hourly_weather_index_polls (
            city TEXT NOT NULL, request_started_utc TEXT NOT NULL,
            response_received_utc TEXT NOT NULL, elapsed_ms DOUBLE PRECISION,
            latest_minute_ms BIGINT, new_versions INTEGER NOT NULL,
            status TEXT NOT NULL, error TEXT,
            PRIMARY KEY (city, request_started_utc))''')
        con.execute('''CREATE TABLE IF NOT EXISTS hourly_weather_index_calibrations (
            city TEXT NOT NULL, config_version TEXT NOT NULL,
            effective_at_ms BIGINT NOT NULL, first_seen_utc TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (city, config_version, effective_at_ms))''')


def save_index(city, data, request_started, received, elapsed_ms):
    from src.database.db import get_connection
    version = data.get('config_version') or None
    points = [point_record(city, p, version, received) for p in data['timeseries']]
    latest = max((p[1] for p in points), default=None)
    # INSERT ... ON CONFLICT DO NOTHING is portable across SQLite and Postgres.
    with get_connection() as con:
        newly_inserted = 0
        for item in points:
            cursor = con.execute('''INSERT INTO hourly_weather_index_versions
                (city,index_minute_ms,payload_sha256,first_seen_utc,status,value_f,
                 config_version,station_receipt_first_ms,station_receipt_last_ms,raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING''', item)
            newly_inserted += max(cursor.rowcount, 0)
        con.execute('''INSERT INTO hourly_weather_index_polls
            (city,request_started_utc,response_received_utc,elapsed_ms,
             latest_minute_ms,new_versions,status,error)
            VALUES (?,?,?,?,?,?,?,?)''',
            (city, request_started, received, elapsed_ms, latest, newly_inserted, 'OK', None))
    
    return len(points), newly_inserted, latest


def save_error(city, started, received, elapsed_ms, error):
    from src.database.db import get_connection
    with get_connection() as con:
        con.execute('''INSERT INTO hourly_weather_index_polls
            (city,request_started_utc,response_received_utc,elapsed_ms,
             latest_minute_ms,new_versions,status,error)
            VALUES (?,?,?,?,?,?,?,?)''',
            (city, started, received, elapsed_ms, None, 0, 'ERROR', str(error)[:1000]))


def save_calibration(city, data):
    from src.database.db import get_connection
    if not isinstance(data, dict) or not isinstance(data.get('calibrations'), list):
        raise ValueError(f'{city}: bad calibration response')
    with get_connection() as con:
        for c in data['calibrations']:
            con.execute('''INSERT INTO hourly_weather_index_calibrations
                (city, config_version, effective_at_ms, first_seen_utc, raw_json)
                VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING''',
                (city, str(c['config_version']), int(c['effective_at_ms']),
                 utc_now(), encode(c)))
    return len(data['calibrations'])


def collect(args):
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv('PREDICTION_DATABASE_URL'):
        raise RuntimeError('PREDICTION_DATABASE_URL required (no SQLite fallback)')
    setup_database()
    session = requests.Session()
    cities = list(dict.fromkeys(args.city))
    print('Weather index collector; read-only; no orders.', flush=True)
    for city in cities:
        try:
            res = session.get(f'{API}/{city}/calibrations', timeout=15)
            res.raise_for_status()
            count = save_calibration(city, res.json())
            print(f'{city}: {count} calibration records', flush=True)
        except Exception as exc:
            print(f'{city}: calibration unavailable: {type(exc).__name__}: {exc}', flush=True)
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else float('inf')
    while time.monotonic() < deadline:
        cycle_start = time.monotonic()
        for city in cities:
            started, tic = utc_now(), time.monotonic()
            try:
                response = session.get(f'{API}/{city}',
                                       params={'last_sec': args.last_sec, 'detailed': 'true'},
                                       timeout=15)
                received = utc_now()
                elapsed_ms = (time.monotonic() - tic) * 1000
                response.raise_for_status()
                data = require_payload(response.json(), city)
                count, new, latest = save_index(city, data, started, received, elapsed_ms)
                print(f'{city}: {count} points, {new} new versions, '
                      f'latest minute={latest}, REST={elapsed_ms:.0f}ms', flush=True)
            except Exception as exc:
                received = utc_now()
                elapsed_ms = (time.monotonic() - tic) * 1000
                print(f'{city}: ERROR {type(exc).__name__}: {exc}', flush=True)
                save_error(city, started, received, elapsed_ms, exc)
        delay = args.interval - (time.monotonic() - cycle_start)
        if delay > 0 and time.monotonic() < deadline:
            time.sleep(min(delay, deadline - time.monotonic()))
    print('Weather collector stopped; no orders placed.', flush=True)


def self_test():
    import sqlite3
    from unittest.mock import patch
    point = {'t': 1791480000000, 'status': 'normal', 'v': 80.12,
             'stations': [{'station_id': 's1', 'received_at_ms': 1791480002000}]}
    incomplete = {'t': 1791480060000, 'status': 'incomplete',
                  'stations': [{'station_id': 's1', 'received_at_ms': 1791480061000}]}
    assert point_record('miami', point, 'v1', utc_now())[5] == 80.12
    assert point_record('miami', incomplete, 'v1', utc_now())[5] is None
    from src.database import db
    # Isolate self-test from Supabase: use transient SQLite on disk and restore.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = folder + '/weather_test.db'
        with patch.object(db, 'get_connection', side_effect=lambda: sqlite3.connect(path)):
            setup_database()
            payload = {'city': 'miami', 'config_version': 'v1', 'timeseries': [point, incomplete]}
            assert save_index('miami', payload, '2026-10-08T19:00:00+00:00', utc_now(), 9)[1] == 2
            assert save_index('miami', payload, '2026-10-08T19:00:01+00:00', utc_now(), 9)[1] == 0
            con = sqlite3.connect(path)
            assert con.execute('SELECT COUNT(*) FROM hourly_weather_index_versions').fetchone()[0] == 2
            assert con.execute('SELECT COUNT(*) FROM hourly_weather_index_polls').fetchone()[0] == 2
            assert con.execute('SELECT COUNT(*) FROM hourly_weather_index_versions WHERE value_f IS NULL').fetchone()[0] == 1
            con.close()
    print('PASS: schema, version deduplication, incomplete points, timestamps')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', action='append', default=[], help='repeat for supported index city IDs')
    parser.add_argument('--interval', type=float, default=20, help='seconds per polling cycle')
    parser.add_argument('--last-sec', type=int, default=600, help='overlap for late publications')
    parser.add_argument('--minutes', type=float, default=0, help='0 = continuous')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.interval < 5 or args.last_sec < 60 or args.minutes < 0:
        parser.error('--interval >=5, --last-sec >=60, --minutes >=0 required')
    if not args.city:
        args.city = ['miami']  # only default to documented, initially supported city
    collect(args)


if __name__ == '__main__':
    main()
