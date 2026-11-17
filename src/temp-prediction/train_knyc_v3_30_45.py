#!/usr/bin/env python3
"""V3 KNYC forecast experiment: 30 and 45 minute focus, strictly causal features.

Uses existing historical METAR CSVs, not delayed ASOS one-minute readings.
Trains on Jul-Aug; tunes on Sep; scores on Oct. No trading or repo access.
"""
import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path('data/research/nyc_observation_20260701_20261026')
CSV = BASE / 'knyc_high_frequency_observations.csv'
AIRPORT_DIR = BASE / 'live_feasible_research' / 'airport_metars'
OUT = BASE / 'v3_30_45'
HORIZONS = [5, 10, 15, 20, 30, 45, 60]
STATIONS = ['NYC', 'LGA', 'EWR', 'TEB']


def load_station(station, lag):
    path = CSV if station == 'NYC' else AIRPORT_DIR / f'{station}_20260701_20261026.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} missing. Run the V2 collection/trainer first.')
    r = pd.read_csv(path, low_memory=False)
    if not {'valid', 'tmpf'}.issubset(r.columns):
        raise ValueError(f'{path}: expected valid and tmpf columns. Got: {list(r.columns)}')
    if 'station' in r:
        r = r[r.station.astype(str).str.upper().isin([station, 'K'+station])].copy()
    r['observed'] = pd.to_datetime(r['valid'], utc=True, errors='coerce')
    for source, name in [('tmpf','temp'),('dwpf','dew'),('sknt','wind'),('drct','direction')]:
        r[name] = pd.to_numeric(r[source], errors='coerce') if source in r else np.nan
    r = r.dropna(subset=['observed', 'temp']).sort_values('observed')
    # Do not arbitrarily choose among conflicting simultaneous temperature reports.
    inconsistent = r.groupby('observed')['temp'].nunique()
    r = r[~r.observed.isin(inconsistent[inconsistent > 1].index)]
    r = r.drop_duplicates('observed').copy().reset_index(drop=True)
    r['prev_temp'] = r.temp.shift(1)
    r['prev2_temp'] = r.temp.shift(2)
    r['prev_time'] = r.observed.shift(1)
    r['prev2_time'] = r.observed.shift(2)
    r['prev_wind'] = r.wind.shift(1)
    r['prev_direction'] = r.direction.shift(1)
    r['available'] = r.observed + pd.Timedelta(minutes=lag)

    dt1 = (r.observed - r.prev_time).dt.total_seconds() / 3600
    dt2 = (r.observed - r.prev2_time).dt.total_seconds() / 3600
    r['trend1'] = ((r.temp-r.prev_temp)/dt1).where(dt1.between(0.1, 2.5))
    r['trend2'] = ((r.temp-r.prev2_temp)/dt2).where(dt2.between(0.3, 4.5))
    r['acceleration'] = r.trend1-r.trend2
    r['wind_change'] = r.wind - r.prev_wind
    theta = np.deg2rad(r.direction)
    r['wind_u'] = -r.wind * np.sin(theta)
    r['wind_v'] = -r.wind * np.cos(theta)
    old_theta = np.deg2rad(r.prev_direction)
    r['wind_u_change'] = r.wind_u + r.prev_wind * np.sin(old_theta)
    r['wind_v_change'] = r.wind_v + r.prev_wind * np.cos(old_theta)

    # Magnus formula; only observed temp/dew are needed.
    tc = (r.temp - 32) * 5 / 9
    dc = (r.dew - 32) * 5 / 9
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        rh = 100 * np.exp(17.625*dc/(243.04+dc) - 17.625*tc/(243.04+tc))
    r['relative_humidity'] = rh.clip(0, 100)
    r['dew_depression'] = r.temp - r.dew
    r['station'] = station
    return r


def make_grid(knyc):
    # Same target routine reports as V2, at :51 each hour.
    targets = knyc[knyc.observed.dt.minute.eq(51)][['observed','temp']].copy()
    targets = targets.rename(columns={'observed':'target_time','temp':'target'})
    grid = pd.concat([targets.assign(horizon=h) for h in HORIZONS], ignore_index=True)
    grid['decision'] = grid.target_time - pd.to_timedelta(grid.horizon, unit='minute')
    local_decision = grid.decision.dt.tz_convert('America/New_York')
    local_target = grid.target_time.dt.tz_convert('America/New_York')
    grid = grid[local_decision.dt.hour.between(8,16) & local_target.dt.hour.between(8,16)].copy()
    local_decision = grid.decision.dt.tz_convert('America/New_York')
    grid['target_day'] = grid.target_time.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    hour = local_decision.dt.hour + local_decision.dt.minute/60
    doy = local_decision.dt.dayofyear
    grid['hour_sin'] = np.sin(2*np.pi*hour/24)
    grid['hour_cos'] = np.cos(2*np.pi*hour/24)
    grid['day_sin'] = np.sin(2*np.pi*doy/365.25)
    grid['day_cos'] = np.cos(2*np.pi*doy/365.25)
    return grid.reset_index(drop=True)


def join_station(grid, df, station):
    # Selection based on *availability*, never measurement time alone.
    report_columns = ['temp','dew','wind','direction','prev_temp','trend1',
                      'trend2','acceleration','wind_change','wind_u','wind_v',
                      'wind_u_change','wind_v_change','relative_humidity',
                      'dew_depression']
    right = df[['available','observed']+report_columns].copy()
    right = right.rename(columns={k:f'{station}_{k}' for k in ['observed']+report_columns})
    ordered = grid[['decision']].sort_values('decision')
    merged = pd.merge_asof(ordered, right.sort_values('available'),
                           left_on='decision', right_on='available',
                           direction='backward',
                           tolerance=pd.Timedelta(hours=4))
    merged.index = ordered.index
    age = (merged.decision - merged[f'{station}_observed']).dt.total_seconds()/60
    merged[f'{station}_age_min'] = age
    stale = age > 120
    cols = [f'{station}_{c}' for c in report_columns]
    merged.loc[stale, cols] = np.nan
    merged.loc[stale, f'{station}_age_min'] = np.nan
    return merged[cols + [f'{station}_age_min']].reindex(grid.index)


def metric(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y-p)
    return dict(n=len(y), mae_f=round(float(err.mean()),3),
                within_half_pct=round(float((err<=0.5).mean()*100),1),
                within_one_pct=round(float((err<=1.0).mean()*100),1),
                same_rounded_degree_pct=round(float((np.rint(y)==np.rint(p)).mean()*100),1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--knyc-lag', type=int, default=5)
    ap.add_argument('--airport-lag', type=int, default=6)
    args = ap.parse_args()
    if min(args.knyc_lag, args.airport_lag) < 0:
        ap.error('Lags must be nonnegative')
    tag = f'knyc{args.knyc_lag}_airports{args.airport_lag}'
    run_dir = OUT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = {s:load_station(s, args.knyc_lag if s=='NYC' else args.airport_lag) for s in STATIONS}
    grid = make_grid(reports['NYC'])
    for station in STATIONS:
        grid = pd.concat([grid, join_station(grid, reports[station], station)], axis=1)
    grid = grid[grid.NYC_temp.notna() & grid[[f'{s}_temp' for s in STATIONS[1:]]].notna().any(axis=1)].copy()
    for s in STATIONS[1:]:
        grid[f'{s}_minus_nyc'] = grid[f'{s}_temp'] - grid.NYC_temp

    base = ['horizon','hour_sin','hour_cos','day_sin','day_cos']
    for s in STATIONS:
        base += [f'{s}_{x}' for x in ['temp','dew','wind','direction',
                                     'prev_temp','trend1','age_min']]
    base += [f'{s}_minus_nyc' for s in STATIONS[1:]]
    groups = {
        'humidity': [f'{s}_{x}' for s in STATIONS for x in ['relative_humidity','dew_depression']],
        'wind_vectors': [f'{s}_{x}' for s in STATIONS for x in ['wind_u','wind_v','wind_change','wind_u_change','wind_v_change']],
        'two_hour_trends': [f'{s}_{x}' for s in STATIONS for x in ['trend2','acceleration']],
    }
    variants = {
        'V2_like': [],
        'humidity': groups['humidity'],
        'wind_vectors': groups['wind_vectors'],
        'two_hour_trends': groups['two_hour_trends'],
        'V3_combined': sum(groups.values(), []),
    }
    train = grid[grid.target_day < '2026-09-01']
    valid = grid[grid.target_day.between('2026-09-01','2026-09-30')]
    test = grid[grid.target_day >= '2026-10-01']
    focus = [30,45]
    print('Train/val/test rows:',len(train),len(valid),len(test))
    print('October counts:',test.horizon.value_counts().sort_index().to_dict())
    results, preds, fitted = [], [], {}
    for horizon in focus:
        mask = valid.horizon.eq(horizon)
        for variant, addition in variants.items():
            features = base + addition
            winner = None
            for leaves in [7,15,31]:
                model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05,
                            max_leaf_nodes=leaves,min_samples_leaf=30,
                            l2_regularization=2.0,random_state=42)
                # Train across all horizons for enough data, tune for the intended horizon.
                model.fit(train[features],train.target-train.NYC_temp)
                pred_val = valid.loc[mask,'NYC_temp'] + model.predict(valid.loc[mask,features])
                val_mae = float(np.abs(valid.loc[mask,'target'].to_numpy()-pred_val).mean())
                if winner is None or val_mae < winner[0]:
                    winner = (val_mae, leaves, model)
            val_mae, leaves, model = winner
            fitted[(horizon,variant)] = dict(model=model,features=features,
                                   validation_mae=val_mae,leaves=leaves)
            rows = test[test.horizon.eq(horizon)].copy()
            pred = rows.NYC_temp.to_numpy()+model.predict(rows[features])
            results.append({'variant':variant,'horizon':horizon,'sep_mae':round(val_mae,3),
                            'leaves':leaves, **metric(rows.target,pred)})
            preds.append(pd.DataFrame({'target_time':rows.target_time,
                         'decision':rows.decision,'horizon':horizon,'variant':variant,
                         'actual':rows.target,'prediction':pred}))
    results_df = pd.DataFrame(results)
    predictions = pd.concat(preds,ignore_index=True)
    results_df.to_csv(run_dir/'comparison.csv',index=False)
    predictions.to_csv(run_dir/'predictions.csv',index=False)
    # Select one model per horizon based only on September validation.
    selected = {h:min(variants,key=lambda v:fitted[(h,v)]['validation_mae']) for h in focus}
    joblib.dump({'selected_by_horizon':selected,'trained':fitted,'lags_minutes':{'NYC':args.knyc_lag,
                 'airports':args.airport_lag},'horizons_focus':focus},run_dir/'research_models.joblib')
    print('\n=== OCTOBER (30 AND 45 MINUTES) ===')
    print(results_df[['variant','horizon','n','mae_f','within_half_pct',
                      'within_one_pct','same_rounded_degree_pct','sep_mae']].to_string(index=False))
    print('\nSelected by SEPTEMBER validation (not October):',selected)
    print('Outputs:',run_dir.resolve())
    print('Note: ±0.5°F uses raw numerical predictions; previous V2 ±1°F used rounded predictions.')
    print('Cloud cover not yet included: older cached station files did not request skyc1-4.')

if __name__=='__main__':
    main()
