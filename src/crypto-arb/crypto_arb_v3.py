#!/usr/bin/env python3
"""Crypto structural arbitrage research V3; PUBLIC MARKET DATA / READ ONLY.

Binance GLOBAL USD-M / spot benchmark ONLY. Not U.S.-executable, not a brokerage
trading model, not a liquidation engine. No authentication, no order APIs.

Inputs: same-venue spot and linear perpetual USD(T) candles, and realized
funding events. Signals use only preceding fully closed bar and prior funding.
Trades fill at following candle OPEN proxy and exit H candle opens later.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

INTERVAL_MS = 3_600_000
BASE = Path(__file__).resolve().parent / 'data' / 'v3' / 'binance_global_benchmark'
SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
SPOT_URL = 'https://data-api.binance.vision/api/v3/klines'
PERP_URL = 'https://fapi.binance.com/fapi/v1/klines'
FUND_URL = 'https://fapi.binance.com/fapi/v1/fundingRate'
SOURCE = 'binance_global_usdt_public_benchmark_RESEARCH_ONLY'


def parse_utc_iso(values: Any) -> pd.Series:
    """Parse ISO8601 strings with mixed fractional-second precision, UTC-only."""
    return pd.to_datetime(values, utc=True, format='ISO8601', errors='raise')


def effective_funding_times(funding_time: Any) -> pd.Series:
    """Use scheduled boundary for tiny timestamp recording delays, not cash timing.

    Binance archives sometimes record an 08:00 funding settlement as
    08:00:00.005Z. Treat timestamps 0..10 seconds AFTER a UTC hourly boundary
    as that boundary for settlement ownership; preserve funding_time unmodified.
    Larger offsets remain as reported. This prevents earning funding by
    opening a position a few milliseconds after the event's scheduled time.
    """
    times = parse_utc_iso(funding_time)
    hourly = times.dt.floor('h')
    delayed = (times - hourly) <= pd.Timedelta(seconds=10)
    return times.where(~delayed, hourly)


def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def iso_from_ms(t: int) -> str:
    return datetime.fromtimestamp(t / 1000, timezone.utc).isoformat()


def num(v: Any, name: str) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f'Nonfinite {name}: {v}')
    return x


def http_get(url: str, params: dict[str, Any], session: requests.Session | None = None) -> Any:
    """Only anonymous public GET endpoints. Respect rate limits/geoblocks."""
    s = session or requests
    for attempt in range(4):
        try:
            r = s.get(url, params=params, timeout=25)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 3:
                raise RuntimeError(f'Public market-data endpoint unavailable: {url}: {exc}') from exc
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (403, 418, 451):
            raise RuntimeError(f'HTTP {r.status_code} for {url}. Geo/access restricted; do not bypass. '
                               'Obtain an authorized historical CSV source instead.')
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == 3:
                raise RuntimeError(f'HTTP {r.status_code} for {url}: {r.text[:180]}')
            retry = r.headers.get('Retry-After', '')
            try:
                delay = min(60, max(1, int(retry))) if retry else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            time.sleep(delay)
            continue
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and 'code' in payload:
            raise RuntimeError(f'Exchange error for {url}: {payload}')
        return payload
    raise AssertionError('unreachable')


def fetch_klines(url: str, symbol: str, start_ms: int, end_ms: int,
                 session: requests.Session | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        data = http_get(url, {'symbol': symbol, 'interval': '1h', 'startTime': cursor,
                              'endTime': end_ms, 'limit': 1000}, session)
        if not isinstance(data, list) or not data:
            raise ValueError(f'{symbol}: incomplete {url} candles at {iso_from_ms(cursor)}')
        last = cursor - 1
        for r in data:
            stamp = int(r[0])
            if stamp < cursor or stamp > end_ms:
                raise ValueError('Out-of-window candle returned')
            rows.append({'timestamp': iso_from_ms(stamp), 'open': num(r[1], 'open'),
                         'high': num(r[2], 'high'), 'low': num(r[3], 'low'),
                         'close': num(r[4], 'close'), 'volume': num(r[5], 'volume')})
            last = max(last, stamp)
        if last < cursor:
            raise ValueError('Non-advancing kline pagination')
        cursor = last + INTERVAL_MS
    df = pd.DataFrame(rows)
    validate_candles(df, symbol, require_complete=True)
    return df


def fetch_funding(symbol: str, start_ms: int, end_ms: int,
                  session: requests.Session | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        data = http_get(FUND_URL, {'symbol': symbol, 'startTime': cursor,
                                   'endTime': end_ms, 'limit': 1000}, session)
        if not isinstance(data, list):
            raise ValueError('Invalid funding payload')
        if not data:
            break
        last = cursor - 1
        for r in data:
            stamp = int(r['fundingTime'])
            if stamp < cursor or stamp > end_ms:
                raise ValueError('Out-of-window funding event')
            if r.get('rateType', 'Regular') != 'Regular':
                raise ValueError(f'Unsupported funding rateType {r.get("rateType")}')
            px = r.get('markPrice')
            if px is None or num(px, 'markPrice') <= 0:
                raise ValueError('Missing markPrice in funding event; cannot price realized funding')
            rows.append({'funding_time': iso_from_ms(stamp),
                         'funding_rate': num(r['fundingRate'], 'fundingRate'),
                         'mark_price': num(px, 'markPrice')})
            last = max(last, stamp)
        if last < cursor:
            raise ValueError('Non-advancing funding pagination')
        cursor = last + 1
    df = pd.DataFrame(rows, columns=['funding_time', 'funding_rate', 'mark_price'])
    validate_funding(df, symbol, require_nonempty=True)
    return df



def vision_archive(url: str, session: requests.Session | None = None) -> bytes:
    """Official static archives only. Requires published SHA256 checksum."""
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(url, timeout=55)
            ch = s.get(url + '.CHECKSUM', timeout=20)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == 2:
                raise RuntimeError(f'Archive inaccessible: {url}: {exc}') from exc
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200 or ch.status_code != 200:
            raise RuntimeError(f'Archive or checksum missing: HTTP {r.status_code}/{ch.status_code} {url}. '
                               'Monthly files may not yet be published.')
        actual = hashlib.sha256(r.content).hexdigest()
        expected = ch.text.strip().split()[0].lower()
        if actual != expected:
            raise ValueError(f'SHA256 mismatch for {url}; archive may be corrupted/replaced')
        return r.content
    raise AssertionError('unreachable')


def vision_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [x for x in z.namelist() if x.lower().endswith('.csv') and not x.startswith('__MACOSX')]
        if len(names) != 1:
            raise ValueError(f'Expected exactly one archive CSV, got {names}')
        if z.getinfo(names[0]).file_size > 30_000_000:
            raise ValueError('Unexpectedly large 1h monthly CSV')
        raw = z.read(names[0])
    return pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)


def archive_klines(content: bytes, name: str, *, require_complete: bool = True) -> pd.DataFrame:
    raw = vision_csv(content)
    if raw.empty or raw.shape[1] < 6:
        raise ValueError(f'{name}: invalid archived candle rows')
    # Some futures CSVs have a header; spot klines often do not.
    if not str(raw.iat[0, 0]).strip().isdigit():
        raw = raw.iloc[1:].reset_index(drop=True)
    ts = pd.to_numeric(raw.iloc[:,0], errors='raise').astype('int64')
    # Binance Vision SPOT uses microseconds from 2025; futures uses milliseconds.
    if (ts >= 10**15).all():
        stamps = ts // 1000
    elif ((ts >= 10**12) & (ts < 10**15)).all():
        stamps = ts
    else:
        raise ValueError(f'{name}: ambiguous timestamp units')
    df = pd.DataFrame({'timestamp': [iso_from_ms(int(t)) for t in stamps],
                       'open': pd.to_numeric(raw.iloc[:,1],errors='raise'),
                       'high': pd.to_numeric(raw.iloc[:,2],errors='raise'),
                       'low':pd.to_numeric(raw.iloc[:,3],errors='raise'),
                       'close': pd.to_numeric(raw.iloc[:,4],errors='raise'),
                       'volume': pd.to_numeric(raw.iloc[:,5],errors='raise')})
    validate_candles(df, name, require_complete=require_complete)
    return df


def archive_funding(content: bytes, name: str) -> pd.DataFrame:
    raw = vision_csv(content)
    # Archive funding files have headings: calc_time, funding_interval_hours,
    # last_funding_rate. Fail rather than guess if schema changes.
    headings = [str(x).strip().lower() for x in raw.iloc[0].tolist()]
    if 'calc_time' not in headings or 'last_funding_rate' not in headings:
        raise ValueError(f'{name}: unrecognized funding archive columns: {headings}')
    rows = raw.iloc[1:].reset_index(drop=True)
    tm = pd.to_numeric(rows.iloc[:,headings.index('calc_time')],errors='raise').astype('int64')
    if not ((tm >= 10**12)&(tm < 10**15)).all():
        raise ValueError(f'{name}: funding calc_time expected in milliseconds')
    rate = pd.to_numeric(rows.iloc[:,headings.index('last_funding_rate')], errors='raise')
    events = pd.DataFrame({'funding_time':[iso_from_ms(int(x)) for x in tm],
                           'funding_rate':rate})
    if 'funding_interval_hours' in headings:
        intervals = pd.to_numeric(rows.iloc[:,headings.index('funding_interval_hours')],errors='raise')
        if not intervals.between(1,24).all():
            raise ValueError(f'{name}: implausible published funding_interval_hours')
    if events.empty:
        raise ValueError(f'{name}: no funding events in archive')
    return events



def attach_archived_funding_marks(events: pd.DataFrame, mark: pd.DataFrame,
                                  symbol: str) -> tuple[pd.DataFrame, int]:
    """Attach only completed 1h mark closes, never stale or future candles.

    Monthly Binance mark archives can be sparse; this method tolerates missing
    *irrelevant* hours but refuses to infer mark prices for settlement events.
    Funding itself is still subject to validate_funding_coverage().

    We have only 1h candle data, not exchange's exact settlement mark. The
    sampled mark close is explicitly an approximation, NOT the exact settlement mark.
    """
    validate_candles(mark, f'{symbol} mark (possibly sparse)', require_complete=False)
    if events.empty:
        raise ValueError(f'{symbol}: no archived funding events')
    fund_times = parse_utc_iso(events['funding_time'])
    # Candles indexed by their OPEN; last complete close is available at
    # OPEN + one hour, not at their OPEN time.
    mark_closes = pd.to_datetime(mark['timestamp'], utc=True) + pd.Timedelta(hours=1)
    close_ns = mark_closes.dt.as_unit('ns').astype('int64').to_numpy()
    fund_ns = fund_times.dt.as_unit('ns').astype('int64').to_numpy()
    pos = np.searchsorted(close_ns, fund_ns, side='right') - 1
    # Events on/before the first available candle close occurred before a mark
    # could be sampled. They cannot affect any backtest entry in this dataset.
    before = pos < 0
    if before.any():
        print(f'{symbol}: exclude {int(before.sum())} initial funding events with '
              'no preceding COMPLETED mark candle', flush=True)
    valid_events = events.loc[~before].copy().reset_index(drop=True)
    valid_pos = pos[~before]
    valid_ns = fund_ns[~before]
    if len(valid_pos) == 0:
        raise ValueError(f'{symbol}: no funding events have completed mark candles')
    age_ns = valid_ns - close_ns[valid_pos]
    # The most recent completed hourly mark candle must end within the previous
    # hour. A missing immediately preceding hour would make this >= 1 hour.
    bad = (age_ns < 0) | (age_ns >= INTERVAL_MS * 1_000_000)
    if bad.any():
        ix = np.flatnonzero(bad)[0]
        event_time = iso_from_ms(int(valid_ns[ix] // 1_000_000))
        candle_time = iso_from_ms(int(close_ns[valid_pos[ix]] // 1_000_000))
        raise ValueError(
            f'{symbol}: missing/stale completed mark candle for funding event '
            f'{event_time}; latest available candle close={candle_time}; '
            f'age={age_ns[ix] / 3.6e12:.3f}h. Do not impute funding marks.'
        )
    valid_events['mark_price'] = mark['close'].to_numpy(float)[valid_pos]
    validate_funding(valid_events, symbol)
    return valid_events, int(before.sum())

def restore_daily_mark_hours(spot: pd.DataFrame, mark: pd.DataFrame, symbol: str,
                             base_url: str,
                             get_archive: Any = None) -> tuple[pd.DataFrame, int, list[str]]:
    """Supplement *missing* monthly mark rows from exact daily 1h ZIPs.

    Does not manufacture candles or fetch from a different venue. The regular
    vision_archive() function checks the SHA-256 sidecar for every downloaded
    daily file. Existing monthly values win; conflicting overlaps hard-fail.
    """
    getter = get_archive or vision_archive
    spot_times = parse_utc_iso(spot['timestamp'])
    mark_times = parse_utc_iso(mark['timestamp'])
    missing = spot_times[~spot_times.isin(mark_times)]
    if missing.empty:
        return mark, 0, []
    missing_days = sorted(set(missing.dt.strftime('%Y-%m-%d').tolist()))
    supplements = []
    errors = []
    for day in missing_days:
        url = (f'{base_url}/futures/um/daily/markPriceKlines/{symbol}/1h/'
               f'{symbol}-1h-{day}.zip')
        print(f'{symbol} {day} mark daily recovery', flush=True)
        try:
            daily = archive_klines(getter(url), f'{symbol}-{day}-daily-mark', require_complete=False)
        except RuntimeError as exc:
            # No fallback price allowed; continue to inspect other published days.
            errors.append(f'{day}: {exc}')
            continue
        daily_times = parse_utc_iso(daily['timestamp'])
        if not (daily_times.dt.strftime('%Y-%m-%d') == day).all():
            raise ValueError(f'{symbol}: daily mark archive timestamps outside {day}')
        original = mark.set_index('timestamp')
        existing = daily[daily.timestamp.isin(mark.timestamp)]
        for _, row in existing.iterrows():
            old = original.loc[row['timestamp']]
            if not np.isclose(float(old['close']), float(row['close']), rtol=1e-10, atol=1e-8):
                raise ValueError(f'{symbol}: daily/monthly mark close conflict at {row["timestamp"]}')
        additions = daily.loc[~daily.timestamp.isin(mark.timestamp)]
        if not additions.empty:
            supplements.append(additions)
    if supplements:
        mark = pd.concat([mark, *supplements], ignore_index=True)
        mark = mark.sort_values('timestamp', kind='stable').reset_index(drop=True)
        validate_candles(mark, f'{symbol} recovered mark', require_complete=False)
    remaining = spot_times[~spot_times.isin(parse_utc_iso(mark['timestamp']))]
    recovered = len(missing) - len(remaining)
    if errors:
        print(f'{symbol}: unavailable daily mark recovery archives: {errors[:3]}', flush=True)
    if len(remaining):
        first = ', '.join(t.isoformat() for t in remaining.iloc[:5])
        print(f'{symbol}: {len(remaining)} mark hours remain missing after daily recovery '
              f'(first: {first}); funding-event audit will decide if usable', flush=True)
    return mark, recovered, errors


def month_keys(first: str, last: str) -> list[str]:
    a = pd.Period(first, freq='M')
    b = pd.Period(last, freq='M')
    if a > b or (b-a).n > 23:
        raise ValueError('Invalid archive month range (max 24 months)')
    return [str(m) for m in pd.period_range(a,b,freq='M')]


def download_vision(args: argparse.Namespace) -> None:
    """Public monthly ZIPs, not trading endpoints; no geo circumvention."""
    today = datetime.now(timezone.utc)
    last_complete = str(pd.Period(today.strftime('%Y-%m'),freq='M')-1)
    months = month_keys(args.first_month or str(pd.Period(last_complete,freq='M')-(args.months-1)),
                        args.last_month or last_complete)
    if months[-1] >= today.strftime('%Y-%m'):
        raise ValueError('Monthly archive may be incomplete; choose through last complete month')
    out = Path(args.data_dir)
    base = 'https://data.binance.vision/data'
    meta = {'source':SOURCE, 'timeframe':'1h','interval_ms':INTERVAL_MS,
            'archive_source':'official Binance Vision monthly ZIP files; SHA256 verified',
            'funding_mark_method':'latest COMPLETED 1h MARK candle close within 1h of each event; NOT exact settlement mark',
            'mark_gap_policy':'report missing mark candle hours; reject any stale funding mark, never impute',
            'mark_gap_counts':{}, 'mark_daily_recovered_hours':{}, 'mark_daily_recovery_errors':{},
            'funding_time_policy':'ISO8601 mixed precision; <=10-second post-hour stamps accounted at UTC hour (original preserved)',
            'initial_funding_events_excluded':{},
            'months':months,'symbols':args.symbols,
            'notice':'RESEARCH ONLY. U.S. non-executable benchmark; no authenticated API calls.'}
    for symbol in args.symbols:
        if symbol not in SYMBOLS:
            raise ValueError(f'Unsupported symbol {symbol}')
        groups: dict[str,list[pd.DataFrame]] = {'spot':[],'perp':[],'mark':[],'fund':[]}
        for month in months:
            paths={
                'spot':f'spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip',
                'perp':f'futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip',
                'mark':f'futures/um/monthly/markPriceKlines/{symbol}/1h/{symbol}-1h-{month}.zip',
                'fund':f'futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip'}
            for kind,path in paths.items():
                print(f'{symbol} {month} {kind}',flush=True)
                blob=vision_archive(f'{base}/{path}')
                value=(archive_funding(blob, f'{symbol}-{month}-{kind}') if kind == 'fund'
                       else archive_klines(blob, f'{symbol}-{month}-{kind}',
                                           require_complete=(kind != 'mark')))
                groups[kind].append(value)
        spot=pd.concat(groups['spot'],ignore_index=True)
        perp=pd.concat(groups['perp'],ignore_index=True)
        mark=pd.concat(groups['mark'],ignore_index=True)
        events=pd.concat(groups['fund'],ignore_index=True)
        validate_candles(spot,symbol+' spot')
        validate_candles(perp,symbol+' perp')
        validate_candles(mark, symbol+' mark', require_complete=False)
        if spot.timestamp.tolist() != perp.timestamp.tolist():
            raise ValueError(f'{symbol}: spot and perpetual archive candles do not align')
        # Mark bars may be genuinely absent from the official archive; prices
        # used to simulate entries and exits must NOT have any missing hours.
        spot_times = pd.to_datetime(spot.timestamp, utc=True)
        mark_times = pd.to_datetime(mark.timestamp, utc=True)
        missing = spot_times[~spot_times.isin(mark_times)]
        extra = mark_times[~mark_times.isin(spot_times)]
        if len(extra):
            raise ValueError(f'{symbol}: {len(extra)} mark timestamps outside spot window')
        meta['mark_gap_counts'][symbol] = int(len(missing))
        if len(missing):
            examples = ', '.join(t.isoformat() for t in missing.iloc[:5])
            print(f'{symbol}: {len(missing)} missing hourly MARK candles '
                  f'(first timestamps: {examples}); auditing funding-event coverage',
                  flush=True)
        mark, recovered, recovery_errors = restore_daily_mark_hours(spot, mark, symbol, base)
        meta['mark_daily_recovered_hours'][symbol] = recovered
        meta['mark_daily_recovery_errors'][symbol] = recovery_errors
        if recovered:
            print(f'{symbol}: recovered {recovered} MARK candles from verified daily archives',
                  flush=True)
        events, skipped_initial = attach_archived_funding_marks(events, mark, symbol)
        meta['initial_funding_events_excluded'][symbol] = skipped_initial
        validate_funding(events, symbol)
        validate_funding_coverage(events,spot,args.max_funding_gap_hours)
        save_csv(spot,out/f'spot_{symbol}_1h.csv')
        save_csv(perp,out/f'perp_{symbol}_1h.csv')
        save_csv(events,out/f'funding_{symbol}.csv')
        print(f'{symbol}: {len(spot)} spot/perp hours; {len(mark)} mark hours; ' + f'{len(events)} funding events',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Archived data saved to {out}; source={SOURCE}')

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    df.to_csv(temp, index=False)
    temp.replace(path)


def download(args: argparse.Namespace) -> None:
    # Only completed hourly bars are included. Never download partial future bar.
    now_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_ms = utc_ms(now_hour) - INTERVAL_MS
    start_ms = end_ms - (args.days * 24 - 1) * INTERVAL_MS
    out = Path(args.data_dir)
    metadata = {'source': SOURCE, 'timeframe': '1h', 'interval_ms': INTERVAL_MS,
                'utc_start_inclusive': iso_from_ms(start_ms),
                'utc_last_candle_open': iso_from_ms(end_ms),
                'symbols': args.symbols,
                'notice': 'NON-U.S.-EXECUTABLE BENCHMARK. Public REST candles are not executable quotes. '
                          'The historical funding events do not predict future funding.'}
    for symbol in args.symbols:
        if symbol not in SYMBOLS:
            raise ValueError(f'Unsupported benchmark symbol {symbol}')
        print(f'{symbol} spot candles', flush=True)
        spot = fetch_klines(SPOT_URL, symbol, start_ms, end_ms)
        print(f'{symbol} perpetual candles', flush=True)
        perp = fetch_klines(PERP_URL, symbol, start_ms, end_ms)
        print(f'{symbol} realized funding events', flush=True)
        fund = fetch_funding(symbol, start_ms, end_ms + INTERVAL_MS - 1)
        # Refuse silently incomplete datasets and missing rate intervals.
        validate_funding_coverage(fund, spot, max_gap_h=args.max_funding_gap_hours)
        save_csv(spot, out / f'spot_{symbol}_1h.csv')
        save_csv(perp, out / f'perp_{symbol}_1h.csv')
        save_csv(fund, out / f'funding_{symbol}.csv')
        print(f'  saved {len(spot)} spot + {len(perp)} perp hours, {len(fund)} funding events', flush=True)
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'All data saved to {out}; source={SOURCE}')


def validate_candles(frame: pd.DataFrame, name: str, *, require_complete: bool = True) -> pd.DataFrame:
    required = ('timestamp', 'open', 'high', 'low', 'close')
    if any(c not in frame.columns for c in required):
        raise ValueError(f'{name}: missing candle columns {required}')
    if len(frame) < 3:
        raise ValueError(f'{name}: fewer than 3 candle rows')
    t = pd.to_datetime(frame['timestamp'], utc=True, errors='raise')
    if t.isna().any() or not t.is_monotonic_increasing or t.duplicated().any():
        raise ValueError(f'{name}: invalid / unordered / duplicate UTC timestamps')
    stamps = t.dt.as_unit('ms').astype('int64')
    if (stamps % INTERVAL_MS != 0).any():
        raise ValueError(f'{name}: non-hourly-aligned candle')
    if require_complete and not (np.diff(stamps.to_numpy()) == INTERVAL_MS).all():
        raise ValueError(f'{name}: missing hourly candles; no forward fill allowed')
    for c in ('open', 'high', 'low', 'close'):
        vals = pd.to_numeric(frame[c], errors='raise')
        if not np.isfinite(vals.to_numpy(float)).all() or (vals <= 0).any():
            raise ValueError(f'{name}: invalid {c}')
    if ((frame['high'].astype(float) < frame[['open', 'close']].max(axis=1)) | 
        (frame['low'].astype(float) > frame[['open', 'close']].min(axis=1)) | 
        (frame['low'].astype(float) > frame['high'].astype(float))).any():
        raise ValueError(f'{name}: invalid high/low geometry')
    return frame


def validate_funding(frame: pd.DataFrame, name: str, *, require_nonempty: bool = True) -> pd.DataFrame:
    for c in ('funding_time', 'funding_rate', 'mark_price'):
        if c not in frame.columns:
            raise ValueError(f'{name}: missing funding column {c}')
    if frame.empty and require_nonempty:
        raise ValueError(f'{name}: no funding events; missing is not zero')
    if frame.empty:
        return frame
    times = parse_utc_iso(frame.funding_time)
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError(f'{name}: duplicate, unsorted or malformed funding times')
    for c in ('funding_rate', 'mark_price'):
        values = pd.to_numeric(frame[c], errors='raise').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: nonfinite {c}')
    if (frame.mark_price.astype(float) <= 0).any():
        raise ValueError(f'{name}: nonpositive funding mark price')
    return frame


def validate_funding_coverage(fund: pd.DataFrame, candles: pd.DataFrame, max_gap_h: float = 12.0) -> None:
    times = effective_funding_times(fund.funding_time)
    a = pd.Timestamp(candles.timestamp.iloc[0])
    b = pd.Timestamp(candles.timestamp.iloc[-1]) + pd.Timedelta(hours=1)
    eligible = times[(times >= a) & (times < b)]
    if eligible.empty:
        raise ValueError('No funding data inside candle window')
    all_times = pd.DatetimeIndex([a, *eligible.tolist(), b])
    gaps = np.diff(all_times.as_unit('ns').asi8) / 3.6e12
    if len(gaps) and max(gaps) > max_gap_h + 1e-8:
        raise ValueError(f'Funding gap {max(gaps):.2f}h exceeds {max_gap_h}h; cannot assume 0')


@dataclass(frozen=True)
class Costs:
    spot_fee_bps: float = 10.0
    perp_fee_bps: float = 5.0
    spot_slippage_bps: float = 2.0
    perp_slippage_bps: float = 2.0
    spot_finance_annual_pct: float = 0.0
    perp_initial_margin_fraction: float = 0.20
    perp_extra_margin_fraction: float = 0.10
    maintenance_margin_fraction: float = 0.05


@dataclass(frozen=True)
class Params:
    hold_hours: int
    min_basis_bps: float
    min_trailing_funding_bps_per_event: float


def load_market(data_dir: Path, symbol: str, max_funding_gap_hours: float):
    if symbol not in SYMBOLS:
        raise ValueError(f'Unexpected symbol {symbol}')
    mpath = data_dir / 'manifest.json'
    if not mpath.exists():
        raise ValueError('Missing manifest.json; do not mix unknown venues, funding conventions, or price units')
    meta = json.loads(mpath.read_text())
    if meta.get('source') != SOURCE or meta.get('interval_ms') != INTERVAL_MS:
        raise ValueError('Data venue or interval does not match V3 benchmark requirements')
    spot = pd.read_csv(data_dir / f'spot_{symbol}_1h.csv')
    perp = pd.read_csv(data_dir / f'perp_{symbol}_1h.csv')
    fund = pd.read_csv(data_dir / f'funding_{symbol}.csv')
    validate_candles(spot, f'{symbol} spot')
    validate_candles(perp, f'{symbol} perpetual')
    validate_funding(fund, symbol)
    validate_funding_coverage(fund, spot, max_gap_h=max_funding_gap_hours)
    if spot.timestamp.tolist() != perp.timestamp.tolist():
        raise ValueError(f'{symbol}: missing or misaligned spot/perp candles; no cross-venue proxy')
    return spot, perp, fund


def trade_ledger(spot: pd.DataFrame, perp: pd.DataFrame, fund: pd.DataFrame,
                 params: Params, costs: Costs, start_i: int, end_i: int,
                 symbol: str, notional: float, funding_lookback: int = 3,
                 max_last_funding_age_h: float = 12.0) -> list[dict]:
    """end_i exclusive; exit bar index must lie strictly inside slice. Flat-to-flat.

    The previous candle closes at the current candle's open timestamp. We
    deliberately use close[i-1] and funding only with time < entry timestamp.
    No future realized funding rate is allowed into entry decisions.
    """
    if start_i < 1 or end_i > len(spot) or len(spot) != len(perp):
        raise ValueError('Invalid backtest period')
    candles_ts = pd.to_datetime(spot.timestamp, utc=True)
    fund_t = effective_funding_times(fund.funding_time)
    rate = fund.funding_rate.to_numpy(float)
    marks = fund.mark_price.to_numpy(float)
    ft = fund_t.dt.as_unit('ns').astype('int64').to_numpy()
    tns = candles_ts.dt.as_unit('ns').astype('int64').to_numpy()
    s_open = spot.open.to_numpy(float)
    s_close = spot.close.to_numpy(float)
    f_open = perp.open.to_numpy(float)
    f_close = perp.close.to_numpy(float)
    f_high = perp.high.to_numpy(float)
    fee_s = costs.spot_fee_bps / 10000
    fee_f = costs.perp_fee_bps / 10000
    slip_s = costs.spot_slippage_bps / 10000
    slip_f = costs.perp_slippage_bps / 10000
    out: list[dict] = []
    i = max(start_i, 1)
    while i + params.hold_hours < end_i:
        # Price signal from the last completed 1h candle.
        basis_bps = 10000 * (f_close[i-1] / s_close[i-1] - 1)
        # History observed strictly before the next open (never use the current event).
        pos = int(np.searchsorted(ft, tns[i], side='left'))
        known = pos >= funding_lookback
        if known:
            trailing = rate[pos-funding_lookback:pos]
            last_age_h = (tns[i] - ft[pos-1]) / 3.6e12
            known = last_age_h <= max_last_funding_age_h
        if (not known or basis_bps < params.min_basis_bps or
            np.mean(trailing) * 10000 < params.min_trailing_funding_bps_per_event):
            i += 1
            continue
        j = i + params.hold_hours
        spot_buy = s_open[i] * (1 + slip_s)
        perp_sell = f_open[i] * (1 - slip_f)
        spot_sell = s_open[j] * (1 - slip_s)
        perp_buy = f_open[j] * (1 + slip_f)
        q = notional / spot_buy
        spot_pnl = q * (spot_sell - spot_buy)
        perp_pnl = q * (perp_sell - perp_buy)
        # Exact entry/exit funding boundaries excluded (conservative at fill timestamp).
        lo = int(np.searchsorted(ft, tns[i], side='right'))
        hi = int(np.searchsorted(ft, tns[j], side='left'))
        funding_pnl = float(q * np.dot(marks[lo:hi], rate[lo:hi]))
        fees = fee_s*q*(spot_buy+spot_sell) + fee_f*q*(perp_sell+perp_buy)
        financing = (costs.spot_finance_annual_pct / 100) * notional * (params.hold_hours / 8760)
        gross_usd = spot_pnl + perp_pnl + funding_pnl
        net_usd = gross_usd - fees - financing
        perp_notional = q * perp_sell
        margin = perp_notional * (costs.perp_initial_margin_fraction + costs.perp_extra_margin_fraction)
        capital = q*spot_buy + margin
        # Coarse adverse intrabar stress: NO guaranteed absence of actual liquidation.
        # Fund payments counted as earned only if occurred by previous hour (not forward).
        breach = False
        worst_excess_usd = math.inf
        for k in range(i, j):
            funded_by_k = int(np.searchsorted(ft, tns[k], side='left'))
            accrued = float(q * np.dot(marks[lo:min(funded_by_k, hi)], rate[lo:min(funded_by_k, hi)]))
            high = f_high[k]
            short_equity = margin + q*(perp_sell-high) + accrued - fee_f*q*perp_sell
            min_needed = costs.maintenance_margin_fraction * q * high
            worst_excess_usd = min(worst_excess_usd, short_equity-min_needed)
            if short_equity <= min_needed:
                breach = True
        out.append({'asset': symbol, 'entry': str(candles_ts.iloc[i]), 'exit': str(candles_ts.iloc[j]),
                    'hold_h': params.hold_hours, 'signal_basis_bps': basis_bps,
                    'trailing_funding_avg_bps': 10000 * float(np.mean(trailing)),
                    'qty_asset': q, 'spot_entry': spot_buy, 'perp_entry': perp_sell,
                    'spot_exit': spot_sell, 'perp_exit': perp_buy,
                    'basis_entry_bps_executable_proxy': 10000*(perp_sell/spot_buy-1),
                    'basis_exit_bps_executable_proxy': 10000*(perp_buy/spot_sell-1),
                    'spot_pnl_usd': spot_pnl, 'perp_pnl_usd': perp_pnl,
                    'funding_pnl_usd': funding_pnl, 'funding_events': hi-lo,
                    'fees_usd': fees, 'financing_usd': financing,
                    'gross_usd': gross_usd, 'net_usd': net_usd,
                    'net_bps_spot_notional': 10000*net_usd/notional,
                    'capital_reserved_usd': capital, 'roi_capital_pct': 100*net_usd/capital,
                    'worst_margin_surplus_usd_proxy': worst_excess_usd,
                    'margin_buffer_breach_proxy': breach})
        i = j + 1  # no overlapping positions in a single asset
    return out


def stats(trades: list[dict]) -> dict[str, float | int | None]:
    # Include every trade in the diagnostic mark-to-exit summary: removing
    # margin-stressed losers would introduce survivorship bias. Any breach
    # invalidates executability, and training selection rejects such models.
    bad_n = sum(bool(r['margin_buffer_breach_proxy']) for r in trades)
    if not trades:
        return {'trades': len(trades), 'margin_breaches': bad_n, 'avg_net_bps': None,
                'median_net_bps': None, 'score_lower95_bps_heuristic': None,
                'net_usd_total': None, 'win_rate': None}
    x = np.asarray([r['net_bps_spot_notional'] for r in trades], float)
    lower = float(x.mean() - 1.96*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else None
    return {'trades': len(trades), 'margin_breaches': bad_n, 'avg_net_bps': float(x.mean()),
            'median_net_bps': float(np.median(x)), 'score_lower95_bps_heuristic': lower,
            'net_usd_total': float(sum(r['net_usd'] for r in trades)),
            'win_rate': float((x>0).mean())}


def grid():
    for h in (24, 48, 72):
        for b in (0., 10., 25.):
            for f in (0., 1., 3.):
                yield Params(h,b,f)


def analyze(args: argparse.Namespace) -> None:
    directory = Path(args.data_dir)
    dest = Path(args.output_dir)
    costs = Costs(args.spot_fee_bps,args.perp_fee_bps,
                  args.spot_slippage_bps,args.perp_slippage_bps,
                  args.spot_finance_annual_pct,args.perp_initial_margin_fraction,
                  args.perp_extra_margin_fraction,args.maintenance_margin_fraction)
    if min(costs.spot_fee_bps,costs.perp_fee_bps,costs.spot_slippage_bps,costs.perp_slippage_bps,
           costs.spot_finance_annual_pct,costs.perp_initial_margin_fraction,
           costs.perp_extra_margin_fraction,costs.maintenance_margin_fraction) < 0:
        raise ValueError('Costs and margin fractions must be non-negative')
    if costs.perp_initial_margin_fraction+costs.perp_extra_margin_fraction <= costs.maintenance_margin_fraction:
        raise ValueError('Margin reserve must exceed maintenance')
    summaries = []
    selected_trades = []
    for symbol in args.symbols:
        spot, perp, fund = load_market(directory,symbol,args.max_funding_gap_hours)
        split = int(len(spot)*args.train_fraction)
        if split < 180 or len(spot)-split < 72:
            raise ValueError(f'{symbol}: not enough train/test hours')
        candidates = []
        for p in grid():
            training = trade_ledger(spot,perp,fund,p,costs,1,split,symbol,args.spot_notional)
            s = stats(training)
            # No model can be 'selected' on its test results; minimum sample and
            # pessimistic training-only lower-bound heuristic are required.
            qualified = (s['trades']>=args.min_train_trades and s['margin_breaches']==0 and
                         s['score_lower95_bps_heuristic'] is not None and
                         s['score_lower95_bps_heuristic']>0)
            candidates.append((qualified, s['score_lower95_bps_heuristic'] or -math.inf,p,s))
        candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
        passers = [x for x in candidates if x[0]]
        chosen = passers[0] if passers else None
        display = chosen if chosen else candidates[0]
        _, score, p, tr = display
        if chosen:
            # OOS test independently enters at >= split and exits before end,
            # uses earlier data only for indicators and funding history.
            oos = trade_ledger(spot,perp,fund,p,costs,split,len(spot),symbol,args.spot_notional)
            selected_trades.extend(oos)
            test = stats(oos)
        else:
            oos = []
            test = stats([])
        summaries.append({'symbol':symbol,'status':'TRAIN_SELECTED' if chosen else 'NO_TRAIN_EDGE',
                          'tested_params':len(candidates), 'train_hours':split,'test_hours':len(spot)-split,
                          'hold_h':p.hold_hours,'min_basis_bps':p.min_basis_bps,
                          'min_trailing_funding_bps_per_event':p.min_trailing_funding_bps_per_event,
                          'train_trades':tr['trades'],'train_breaches':tr['margin_breaches'],
                          'train_avg_net_bps':tr['avg_net_bps'],
                          'train_lower95_bps_heuristic':tr['score_lower95_bps_heuristic'],
                          'oos_trades':test['trades'],'oos_breaches':test['margin_breaches'],
                          'oos_avg_net_bps':test['avg_net_bps'],
                          'oos_net_usd_total':test['net_usd_total'],
                          'oos_win_rate':test['win_rate']})
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(dest/'v3_selection.csv',index=False)
    pd.DataFrame(selected_trades).to_csv(dest/'v3_oos_selected_trades.csv',index=False)
    configuration = {'source':SOURCE,'costs':asdict(costs), 'symbol_set':args.symbols,
                     'train_fraction':args.train_fraction,'train_min_trades':args.min_train_trades,
                     'spot_notional':args.spot_notional,
                     'method':'Close t-1 signal -> next open entry -> hold hours -> open exit. '
                              'Funding actual historical cash events strictly inside hold period; '
                              'training signal uses only funding strictly before entry. '
                              'Single asset position at once. No funding forecasting, depth or executable quotes.',
                     'WARNINGS':['U.S. venue not represented; Binance Global research benchmark only',
                                 'Margin stress is NOT a liquidation simulator',
                                 'Funding at entry/exit instant excluded',
                                 'No cross-asset portfolio-level ROI / correlated collateral modeling',
                                 'Model search lower95 is heuristic and ignores multiple-testing bias',
                                 'Historical funding is known for P&L only; trailing funding is signal, not forward rates']}
    (dest/'v3_config.json').write_text(json.dumps(configuration,indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f'\nTRAIN SELECTED {sum(x["status"]=="TRAIN_SELECTED" for x in summaries)}/{len(summaries)}; '
          f'OOS selected trades={len(selected_trades)}; source={SOURCE}')
    print(f'Saved {dest}; no orders. Margin breaches invalidate trade executability; paper P&L marks to scheduled exit.')
    print('Execution proxy only, NOT executable or U.S.-venue P&L.')


def synthetic_data(n: int = 600):
    t0 = pd.Timestamp('2026-01-01T00:00:00Z')
    spot = []
    perp = []
    for i in range(n):
        t = (t0+pd.Timedelta(hours=i)).isoformat()
        # Spike in the previous close, followed by price-compatible open and reversion.
        last_spike = (i-1)%48 == 10
        spike = i%48 == 10
        op = 102.0 if last_spike else 100.0
        cl = 102.0 if spike else 100.0
        spot.append({'timestamp':t,'open':100.,'high':100.,'low':100.,'close':100.})
        perp.append({'timestamp':t,'open':op,'high':max(op,cl),'low':min(op,cl),'close':cl})
    fund = []
    for i in range(0,n,8):
        fund.append({'funding_time':(t0+pd.Timedelta(hours=i)).isoformat(),
                     'funding_rate':0.0001,'mark_price':100.0})
    return pd.DataFrame(spot),pd.DataFrame(perp),pd.DataFrame(fund)


def self_test() -> None:
    s,p,f = synthetic_data()
    validate_candles(s,'spot');validate_candles(p,'perp');validate_funding(f,'fund')
    validate_funding_coverage(f,s)
    costs = Costs(0,0,0,0,0,.2,.1,.05)
    pp = Params(24,10,0)
    train = trade_ledger(s,p,f,pp,costs,1,420,'BTCUSDT',100.)
    test = trade_ledger(s,p,f,pp,costs,420,600,'BTCUSDT',100.)
    assert len(train)>=6 and len(test)>=2 and min(x['net_usd'] for x in train)>0
    assert train[0]['signal_basis_bps']>100 and train[0]['spot_pnl_usd']==0
    assert abs(train[0]['perp_pnl_usd']-2.0)<1e-8,train[0]
    assert all(not x['margin_buffer_breach_proxy'] for x in train)
    # Non-anticipated future funding changes cannot alter training-side decisions.
    altered=f.copy();altered.loc[altered.index>=13,'funding_rate']=-.1
    before=trade_ledger(s,p,f,pp,costs,1,98,'BTCUSDT',100.)
    after=trade_ledger(s,p,altered,pp,costs,1,98,'BTCUSDT',100.)
    assert len(before)==1 and before==after
    # Events exactly at entry/exit boundaries do not accrue funding.
    ss=s.iloc[:7].copy().reset_index(drop=True)
    ff=p.iloc[:7].copy().reset_index(drop=True)
    ff.loc[0,'close']=102.;ff.loc[0,'high']=102.
    ff.loc[1,'open']=102.;ff.loc[1,'high']=102.;ff.loc[1,'close']=100.
    boundary=pd.DataFrame([
        {'funding_time':ss.timestamp.iloc[1],'funding_rate':.5,'mark_price':100.},
        {'funding_time':ss.timestamp.iloc[2],'funding_rate':.01,'mark_price':100.},
        {'funding_time':ss.timestamp.iloc[3],'funding_rate':.5,'mark_price':100.},
    ])
    # Verify settlement boundary conventions independent of entry-signal logic.
    tf=pd.to_datetime(boundary.funding_time,utc=True).dt.as_unit('ns').astype('int64').to_numpy()
    ti=pd.to_datetime(ss.timestamp,utc=True).dt.as_unit('ns').astype('int64').to_numpy()
    lo=int(np.searchsorted(tf,ti[1],side='right'))
    hi=int(np.searchsorted(tf,ti[3],side='left'))
    assert hi-lo==1 and abs(boundary.funding_rate.iloc[lo]-.01)<1e-12
    # A sudden wick above reserved margin is flagged as stress, not claimed liquidation.
    danger=p.copy()
    danger.loc[59,'high']=200.
    danger.loc[59,'close']=100.
    risk=trade_ledger(s,danger,f,pp,costs,1,100,'BTCUSDT',100.)
    assert any(x['margin_buffer_breach_proxy'] for x in risk)
    # Mixed ISO timestamps (with and without fractional seconds) must parse.
    mixed = pd.DataFrame([{'funding_time': f.funding_time.iloc[0], 'funding_rate': .001,
                           'mark_price': 100.},
                          {'funding_time': '2026-01-01T08:00:00.005000+00:00',
                           'funding_rate': .001, 'mark_price': 100.}])
    validate_funding(mixed, 'mixed ISO8601 funding')
    assert effective_funding_times(mixed.funding_time).iloc[1] == pd.Timestamp('2026-01-01T08:00:00Z')
    # Millisecond-late exchange event reporting must not change holding-period
    # ownership of payments at hourly entry / exit boundaries.
    delayed = f.copy()
    delayed['funding_time'] = (parse_utc_iso(f.funding_time)
                               + pd.Timedelta(milliseconds=5)).dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    changed = trade_ledger(s,p,delayed,pp,costs,1,420,'BTCUSDT',100.)
    assert len(changed) == len(train)
    for a,b in zip(train,changed):
        assert a['funding_events'] == b['funding_events']
        assert abs(a['funding_pnl_usd'] - b['funding_pnl_usd']) < 1e-9
    # A daily 1h ZIP can restore a missing monthly mark row without borrowing
    # prices from other candles, using the same archive parser as production.
    sample = s.iloc[:72].copy().reset_index(drop=True)
    day_missing = sample.drop(index=list(range(24,48))).reset_index(drop=True)
    daily_day = sample.iloc[24:48].copy()
    csv_rows = []
    for _, candle in daily_day.iterrows():
        stamp = int(pd.Timestamp(candle['timestamp']).timestamp()*1000)
        csv_rows.append([stamp,candle['open'],candle['high'],candle['low'],candle['close'],1.])
    import csv
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['open_time','open','high','low','close','volume'])
    writer.writerows(csv_rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('daily.csv', stream.getvalue())
    def fake_daily(url):
        assert url.endswith('BTCUSDT-1h-2026-01-02.zip')
        return buf.getvalue()
    recovered_mark, recovered_n, failures = restore_daily_mark_hours(
        sample, day_missing, 'BTCUSDT', 'https://unused.test', fake_daily)
    assert recovered_n == 24 and not failures and len(recovered_mark)==len(sample)
    assert recovered_mark.timestamp.tolist() == sample.timestamp.tolist()
    # Sparse mark archives are permitted only when no settlement relies on
    # missing data. An 08:00 funding event needs the 07:00 hourly mark bar.
    marks=s.copy()
    events,_=attach_archived_funding_marks(f, marks.drop(index=2).reset_index(drop=True), 'SYNTH')
    assert len(events)==len(f)-1 and events['mark_price'].notna().all()
    try:
        attach_archived_funding_marks(f, marks.drop(index=7).reset_index(drop=True), 'SYNTH')
        raise AssertionError('stale missing-event mark incorrectly accepted')
    except ValueError as exc:
        assert 'missing/stale completed mark candle' in str(exc), str(exc)
    # A settlement event at the precise completion of a mark candle uses that
    # candle; not the not-yet-completed candle starting at settlement time.
    ev=pd.DataFrame([{'funding_time':s.timestamp.iloc[8], 'funding_rate':.001}])
    ev,skip=attach_archived_funding_marks(ev,marks,'SYNTH')
    assert skip==0 and ev.mark_price.iloc[0]==marks.close.iloc[7]
    # Reject any missing candle or funding interval; no imputing zeros.
    try:
        validate_candles(s.drop(index=5).reset_index(drop=True),'bad')
        raise AssertionError('missing hour accepted')
    except ValueError:
        pass
    try:
        validate_funding_coverage(f.drop(index=3).reset_index(drop=True),s)
        raise AssertionError('missing funding accepted')
    except ValueError:
        pass
    # V3 uses only public anonymous GET, no trading endpoint in downloader.
    assert all('/klines' in x or 'fundingRate' in x for x in (SPOT_URL,PERP_URL,FUND_URL))
    print(f'PASS V3.2: {len(train)} train and {len(test)} test synthetic trades; '
          'event-time funding, next-open entries, fee + margin accounting, gap guards.')
    print('OFFLINE READ-ONLY — Binance Global research benchmark; NO ORDERS / AUTH')


def parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    d=sub.add_parser('download',help='Public Binance GLOBAL market data only; may be geo-restricted')
    d.add_argument('--days',type=int,default=180)
    d.add_argument('--data-dir',default=str(BASE))
    d.add_argument('--symbols',nargs='+',default=list(SYMBOLS),choices=SYMBOLS)
    d.add_argument('--max-funding-gap-hours',type=float,default=12)
    v=sub.add_parser('download-vision',help='Official monthly static ZIP archives; no trading API')
    v.add_argument('--months',type=int,default=6)
    v.add_argument('--first-month',default=None,help='YYYY-MM, optional explicit range start')
    v.add_argument('--last-month',default=None,help='YYYY-MM, optional explicit range end')
    v.add_argument('--data-dir',default=str(BASE))
    v.add_argument('--symbols',nargs='+',default=list(SYMBOLS),choices=SYMBOLS)
    v.add_argument('--max-funding-gap-hours',type=float,default=12)
    a=sub.add_parser('analyze',help='Read cached data; no network and no orders')
    a.add_argument('--data-dir',default=str(BASE))
    a.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/'data'/'v3'/'results'))
    a.add_argument('--symbols',nargs='+',default=list(SYMBOLS),choices=SYMBOLS)
    a.add_argument('--train-fraction',type=float,default=.7)
    a.add_argument('--min-train-trades',type=int,default=10)
    a.add_argument('--spot-notional',type=float,default=1000)
    a.add_argument('--spot-fee-bps',type=float,default=10)
    a.add_argument('--perp-fee-bps',type=float,default=5)
    a.add_argument('--spot-slippage-bps',type=float,default=2)
    a.add_argument('--perp-slippage-bps',type=float,default=2)
    a.add_argument('--spot-finance-annual-pct',type=float,default=0)
    a.add_argument('--perp-initial-margin-fraction',type=float,default=.20)
    a.add_argument('--perp-extra-margin-fraction',type=float,default=.10)
    a.add_argument('--maintenance-margin-fraction',type=float,default=.05)
    a.add_argument('--max-funding-gap-hours',type=float,default=12)
    return ap


def main():
    a=parser().parse_args()
    if a.command=='self-test':self_test()
    elif a.command=='download':
        if a.days < 7 or a.days>720:raise ValueError('--days must be 7..720')
        download(a)
    elif a.command=='download-vision':
        if a.months<1 or a.months>24:raise ValueError('bad months')
        download_vision(a)
    elif a.command=='analyze':
        if not (0.5<=a.train_fraction<=0.9):raise ValueError('train fraction outside [0.5,0.9]')
        if a.spot_notional<=0 or a.min_train_trades<2:raise ValueError('bad notional/min trades')
        analyze(a)


if __name__=='__main__':main()
