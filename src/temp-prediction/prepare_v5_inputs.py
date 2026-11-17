#!/usr/bin/env python3
"""Prepare precise KNYC labels and historical hourly METAR cloud inputs.

Read-only against existing research. Writes exclusively under v5_inputs/.
"""
from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path('data/research/nyc_observation_20260701_20261026')
DEST = ROOT / 'v5_inputs'
CACHE = DEST / 'monthly_cache'
URL = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py'
STATIONS = {'NYC': 'NY_ASOS', 'LGA': 'NY_ASOS', 'EWR': 'NJ_ASOS',
            'TEB': 'NJ_ASOS', 'JFK': 'NY_ASOS'}
MONTHS = ['2026-07', '2026-08', '2026-09', '2026-10']
SKY = {'CLR', 'SKC', 'FEW', 'SCT', 'BKN', 'OVC', 'VV'}
T_GROUP = re.compile(r'(?<!\S)T([01])(\d{3})([01])(\d{3})(?=\s|$)')


def precise_f(metar):
    if not isinstance(metar, str):
        return float('nan')
    m = T_GROUP.search(metar)
    if m is None:
        return float('nan')
    temp_c = (-1 if m[1] == '1' else 1) * (int(m[2]) / 10)
    return temp_c * 1.8 + 32


def request_month(session, station, network, month):
    start = pd.Timestamp(month + '-01', tz='UTC')
    end = start + pd.offsets.MonthBegin(1)
    params = [
        ('station', station), ('network', network),
        *[('data', name) for name in
          ('tmpf', 'dwpf', 'sknt', 'drct', 'metar',
           'skyc1', 'skyc2', 'skyc3', 'skyc4',
           'skyl1', 'skyl2', 'skyl3', 'skyl4')],
        ('report_type', '3'), ('report_type', '4'),
        ('year1', start.year), ('month1', start.month), ('day1', start.day),
        ('year2', end.year), ('month2', end.month), ('day2', end.day),
        ('tz', 'Etc/UTC'), ('format', 'onlycomma'),
        ('missing', 'M'), ('latlon', 'no'), ('elev', 'no'),
    ]
    for attempt in range(3):
        try:
            response = session.get(URL, params=params, timeout=120)
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text), comment='#',
                                keep_default_na=False)
            if not {'station','valid','metar','skyc1'}.issubset(frame.columns):
                raise ValueError('Unexpected response columns: ' + str(frame.columns.tolist()))
            return frame
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f'{station} {month}: {exc}') from exc
            time.sleep(2 * (attempt + 1))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    knyc = pd.read_csv(ROOT / 'knyc_high_frequency_observations.csv',
                       keep_default_na=False)
    if not {'station','valid','tmpf','metar'}.issubset(knyc):
        raise ValueError('KNYC archive must contain station, valid, tmpf, metar')
    knyc['observed_utc'] = pd.to_datetime(knyc['valid'], utc=True, errors='coerce')
    knyc['precise_f'] = knyc['metar'].map(precise_f)
    knyc['rounded_archive_f'] = pd.to_numeric(knyc['tmpf'], errors='coerce')
    knyc = knyc[
        knyc['station'].astype(str).str.upper().isin(['NYC','KNYC'])
        & knyc['observed_utc'].notna()
        & (knyc['observed_utc'].dt.minute == 51)
    ].copy()
    # Preserve only unambiguous target labels, without falling back to rounded temperatures.
    conflicts = knyc.groupby('observed_utc')['precise_f'].agg(
        lambda x: x.dropna().nunique())
    knyc = knyc[~knyc['observed_utc'].isin(conflicts[conflicts > 1].index)]
    knyc = knyc.sort_values('observed_utc').drop_duplicates('observed_utc')
    knyc[['observed_utc','precise_f','rounded_archive_f','metar']].to_csv(
        DEST / 'knyc_precise_targets.csv', index=False)
    print('Precise KNYC :51 targets:', knyc['precise_f'].notna().sum(),
          '/', len(knyc), flush=True)

    session = requests.Session()
    session.headers.update({'User-Agent': 'PolyQuant-weather-research/1.0'})
    all_frames = []
    for station, network in STATIONS.items():
        for month in MONTHS:
            cached = CACHE / f'{station}_{month}.csv'
            if cached.exists():
                frame = pd.read_csv(cached, keep_default_na=False)
            else:
                frame = request_month(session, station, network, month)
                frame.to_csv(cached, index=False)
                time.sleep(0.2)
            frame['station_name'] = station
            all_frames.append(frame)
            print(f'{station} {month}: {len(frame)} records', flush=True)
    obs = pd.concat(all_frames, ignore_index=True)
    obs['observed_utc'] = pd.to_datetime(obs['valid'], utc=True, errors='coerce')
    obs = obs.dropna(subset=['observed_utc'])
    for name in ('tmpf','dwpf','sknt','drct','skyl1','skyl2','skyl3','skyl4'):
        if name in obs.columns:
            obs[name] = pd.to_numeric(obs[name], errors='coerce')
    for name in ('skyc1','skyc2','skyc3','skyc4'):
        obs[name] = obs[name].astype(str).str.strip().str.upper()
        obs.loc[~obs[name].isin(SKY), name] = ''
    sky_cols = ['skyc1','skyc2','skyc3','skyc4']
    obs['has_sky'] = obs[sky_cols].ne('').any(axis=1)
    obs['precise_f'] = obs['metar'].map(precise_f)
    # Keep deterministic duplicate resolution; report conflicts separately.
    conflict_counts = obs.groupby(['station_name','observed_utc'])['metar'].nunique()
    print('Conflicting METAR duplicate groups:', int((conflict_counts > 1).sum()))
    obs = obs.sort_values(['station_name','observed_utc']).drop_duplicates(
        ['station_name','observed_utc'], keep='first')
    obs.to_csv(DEST / 'hourly_metar_clouds.csv', index=False)
    cov = obs.groupby('station_name').agg(
        reports=('observed_utc','size'),
        sky_reports=('has_sky','sum'),
        precise_reports=('precise_f', lambda x: x.notna().sum()),
        days=('observed_utc',lambda x: x.dt.date.nunique()),
    )
    cov['sky_pct'] = (100 * cov['sky_reports'] / cov['reports']).round(1)
    cov.to_csv(DEST / 'coverage.csv')
    print('\n=== V5 INPUT COVERAGE ===')
    print(cov.to_string())
    print('Saved:', DEST.resolve())
    print('CAUTION: historical observation timestamps are not publication timestamps.')


if __name__ == '__main__':
    main()
