
#!/usr/bin/env python3
"""Paired MADIS/V3 NYC temperature forecasting experiment.

Uses existing V3 October predictions, without retraining.
Adds experimental 5-minute airport temperature adjustments.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, chartostring

BASE = Path(
    "data/research/nyc_observation_20260701_20261026"
)
V3 = BASE / "v3_30_45"
OUT = BASE / "madis_v4_paired_pilot"

ARCHIVE = (
    "https://madis-data.ncep.noaa.gov/"
    "madisPublic1/data/LDAD/hfmetar/netCDF/"
)

STATIONS = {
    "KLGA": "LGA",
    "KTEB": "TEB",
    "KEWR": "EWR",
}

LAGS = [15, 25, 30]
WEIGHTS = [0.25, 0.50]


def get(url, session, attempts=3):
    for attempt in range(attempts):
        try:
            response = session.get(
                url, timeout=(15, 75)
            )
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def load_v3_run(knyc_lag, airport_lag):
    folder = V3 / (
        f"knyc{knyc_lag}_airports{airport_lag}"
    )

    prediction_file = folder / "predictions.csv"
    trained_file = folder / "research_models.joblib"

    if not prediction_file.exists():
        raise FileNotFoundError(prediction_file)

    if not trained_file.exists():
        raise FileNotFoundError(trained_file)

    # Load only your own locally trained models.
    saved = joblib.load(trained_file)
    selected = saved["selected_by_horizon"]

    df = pd.read_csv(prediction_file)

    df["target_time"] = pd.to_datetime(
        df["target_time"], utc=True
    )

    df["decision"] = pd.to_datetime(
        df["decision"], utc=True
    )

    df = df[
        df["horizon"].isin([30, 45])
    ].copy()

    df = df[
        df.apply(
            lambda r: (
                r["variant"]
                == selected[int(r["horizon"])]
            ),
            axis=1,
        )
    ].copy()

    df["target_day"] = (
        df["target_time"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    print(
        "September-selected V3 variants:",
        selected
    )

    return (
        df.sort_values("decision")
        .reset_index(drop=True)
    )


def load_hourly(station, lag_min):
    path = (
        BASE
        / "live_feasible_research"
        / "airport_metars"
        / f"{station}_20260701_20261026.csv"
    )

    df = pd.read_csv(path)

    df["observation"] = pd.to_datetime(
        df["valid"], utc=True, errors="coerce"
    )

    df["temp"] = pd.to_numeric(
        df["tmpf"], errors="coerce"
    )

    df = df.dropna(
        subset=["observation", "temp"]
    ).copy()

    df = df[
        df["station"].astype(str).str.upper()
        .isin([station, "K" + station])
    ].copy()

    # Exclude conflicting readings at the same time.
    counts = df.groupby(
        "observation"
    )["temp"].nunique()

    bad = counts[counts > 1].index

    df = (
        df[~df["observation"].isin(bad)]
        .sort_values("observation")
        .drop_duplicates("observation")
        .copy()
    )

    df["prev_temp"] = df["temp"].shift(1)
    df["prev_observation"] = (
        df["observation"].shift(1)
    )

    hours = (
        df["observation"]
        - df["prev_observation"]
    ).dt.total_seconds() / 3600

    slope = (
        df["temp"] - df["prev_temp"]
    ) / hours

    df["hourly_slope"] = (
        slope.where(hours.between(0.5, 2.5))
        .clip(-5, 5)
    )

    df["available"] = (
        df["observation"]
        + pd.Timedelta(minutes=lag_min)
    )

    return df[
        [
            "observation",
            "temp",
            "hourly_slope",
            "available",
        ]
    ].sort_values("available")


def list_needed_files(predictions, session):
    response = get(ARCHIVE, session)

    found = sorted(set(re.findall(
        r"20\d{6}_\d{4}\.gz",
        response.text,
    )))

    if not found:
        raise RuntimeError(
            "No MADIS files in public directory"
        )

    needed_hours = set()

    for decision in predictions["decision"]:
        for back in [0, 1, 2]:
            needed_hours.add(
                (
                    decision
                    - pd.Timedelta(hours=back)
                ).floor("h")
            )

    selected = []

    for filename in found:
        hour = pd.to_datetime(
            filename[:13],
            format="%Y%m%d_%H%M",
            utc=True,
        )

        if hour in needed_hours:
            selected.append(filename)

    print(
        "MADIS archive:",
        found[0], "through", found[-1]
    )
    print(
        "Overlapping hourly files:",
        len(selected)
    )

    return selected


def parse_file(blob, filename):
    raw = gzip.decompress(blob)

    records = []

    with Dataset(
        "inmemory", memory=raw
    ) as ds:

        ids = np.asarray(
            chartostring(
                ds.variables["stationId"][:]
            )
        ).astype(str)

        ids = np.char.strip(ids)

        indices = np.flatnonzero(
            np.isin(ids, list(STATIONS))
        )

        observations = (
            ds.variables["observationTime"][indices]
        )

        temperatures = (
            ds.variables["temperature"][indices]
        )

        for index, observed, temperature in zip(
            indices, observations, temperatures
        ):
            if (
                np.ma.is_masked(observed)
                or np.ma.is_masked(temperature)
            ):
                continue

            kelvin = float(temperature)
            epoch = float(observed)

            if not (
                200 < kelvin < 335
                and 1_500_000_000 < epoch < 2_000_000_000
            ):
                continue

            records.append({
                "station": ids[index],
                "observation": pd.Timestamp(
                    epoch, unit="s", tz="UTC"
                ),
                "temperature_f": (
                    (kelvin - 273.15) * 1.8 + 32
                ),
                "source_file": filename,
            })

    return pd.DataFrame(
        records,
        columns=[
            "station",
            "observation",
            "temperature_f",
            "source_file",
        ],
    )


def load_madis(files, session):
    cache = OUT / "netcdf_cache"
    cache.mkdir(parents=True, exist_ok=True)

    frames = []

    for i, filename in enumerate(files, 1):
        path = cache / filename

        try:
            if path.exists():
                blob = path.read_bytes()
            else:
                blob = get(
                    ARCHIVE + filename, session
                ).content

            frame = parse_file(blob, filename)

            if not path.exists():
                path.write_bytes(blob)

            frames.append(frame)

            print(
                f"[{i}/{len(files)}] "
                f"{filename}: {len(frame)} records",
                flush=True,
            )

        except Exception as error:
            print(
                f"WARNING {filename}: {error}",
                file=sys.stderr,
            )

    frames = [
        f for f in frames if not f.empty
    ]

    if not frames:
        raise RuntimeError(
            "No matching MADIS temperatures found"
        )

    madis = pd.concat(
        frames, ignore_index=True
    )

    # Remove inconsistent duplicate observations.
    conflict = (
        madis.groupby(
            ["station", "observation"]
        )["temperature_f"]
        .agg(lambda x: x.max() - x.min())
    )

    bad = conflict[
        conflict > 0.2
    ].index

    if len(bad):
        bad_df = pd.DataFrame(
            list(bad),
            columns=["station", "observation"],
        )

        madis = madis.merge(
            bad_df.assign(conflict=True),
            on=["station", "observation"],
            how="left",
        )

        madis = madis[
            madis["conflict"].isna()
        ].drop(columns="conflict")

    madis = (
        madis.sort_values(
            ["station", "observation"]
        )
        .drop_duplicates(
            ["station", "observation"]
        )
    )

    print(
        "MADIS observations:",
        madis.groupby("station").size().to_dict()
    )

    return madis


def enrich(predictions, madis, lag, airport_lag):
    grid = (
        predictions.sort_values("decision")
        .reset_index(drop=True)
        .copy()
    )

    grid["madis_lag"] = lag

    innovations = []

    for station, short_name in STATIONS.items():
        hourly = load_hourly(
            short_name, airport_lag
        )

        hourly = hourly.rename(columns={
            "observation": "h_obs",
            "temp": "h_temp",
            "hourly_slope": "h_slope",
            "available": "h_available",
        })

        merged = pd.merge_asof(
            grid[["decision"]],
            hourly,
            left_on="decision",
            right_on="h_available",
            direction="backward",
        )

        station_madis = madis[
            madis["station"] == station
        ][["observation", "temperature_f"]].copy()

        station_madis["m_available"] = (
            station_madis["observation"]
            + pd.Timedelta(minutes=lag)
        )

        station_madis = station_madis.rename(
            columns={
                "observation": "m_obs",
                "temperature_f": "m_temp",
            }
        ).sort_values("m_available")

        merged = pd.merge_asof(
            merged.sort_values("decision"),
            station_madis,
            left_on="decision",
            right_on="m_available",
            direction="backward",
        )

        # Only consider genuinely newer MADIS inputs.
        fresh = (
            (merged["m_obs"] > merged["h_obs"])
            & (
                (
                    merged["decision"]
                    - merged["m_obs"]
                ).dt.total_seconds() / 60
                <= 60
            )
        )

        elapsed_hours = (
            merged["m_obs"] - merged["h_obs"]
        ).dt.total_seconds() / 3600

        expected = (
            merged["h_temp"]
            + merged["h_slope"].fillna(0)
            * elapsed_hours.clip(
                lower=0, upper=1.5
            )
        )

        innovation = (
            merged["m_temp"] - expected
        ).where(fresh).clip(-4, 4)

        innovations.append(
            innovation.rename(station)
        )

    regional = pd.concat(
        innovations, axis=1
    )

    grid["station_count_with_fresh_madis"] = (
        regional.notna().sum(axis=1)
    )

    grid["regional_innovation_f"] = (
        regional.median(axis=1)
    )

    actionable = (
        grid["station_count_with_fresh_madis"] >= 2
    )

    grid["regional_innovation_f"] = (
        grid["regional_innovation_f"]
        .where(actionable, 0)
    )

    grid["has_fresh_madis"] = actionable

    for weight in WEIGHTS:
        grid[f"forecast_weight_{weight:.2f}"] = (
            grid["prediction"]
            + weight * grid["regional_innovation_f"]
        )

    return grid


def metrics(df, column):
    errors = np.abs(
        df["actual"].to_numpy(dtype=float)
        - df[column].to_numpy(dtype=float)
    )

    if not len(errors):
        return {
            "n": 0,
            "mae_f": np.nan,
            "within_half_pct": np.nan,
            "within_one_pct": np.nan,
        }

    return {
        "n": len(errors),
        "mae_f": round(
            float(errors.mean()), 3
        ),
        "within_half_pct": round(
            float(np.mean(errors <= 0.5) * 100),
            1,
        ),
        "within_one_pct": round(
            float(np.mean(errors <= 1.0) * 100),
            1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag", type=int, default=5
    )

    parser.add_argument(
        "--airport-lag", type=int, default=6
    )

    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=LAGS,
    )

    args = parser.parse_args()

    if min([
        args.knyc_lag,
        args.airport_lag,
        *args.lags,
    ]) < 0:
        parser.error(
            "Reporting delays must be nonnegative"
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    predictions = load_v3_run(
        args.knyc_lag,
        args.airport_lag,
    )

    session = requests.Session()
    session.headers["User-Agent"] = (
        "PolyQuant-KNYC-MADIS-research/1.0"
    )

    files = list_needed_files(
        predictions, session
    )

    if not files:
        raise SystemExit(
            "No MADIS files overlap V3 predictions."
        )

    madis = load_madis(files, session)

    # Require an overlapping MADIS hourly file
    # for each prediction being compared.
    available = set(files)

    predictions["hour_file"] = (
        predictions["decision"]
        .dt.strftime("%Y%m%d_%H00.gz")
    )

    predictions = predictions[
        predictions["hour_file"].isin(available)
    ].drop(columns="hour_file")

    if predictions.empty:
        raise SystemExit(
            "No paired forecast events found"
        )

    summaries = []
    details = []

    for lag in args.lags:
        paired = enrich(
            predictions,
            madis,
            lag,
            args.airport_lag,
        )

        details.append(paired)

        for horizon in [30, 45]:
            subset = paired[
                paired["horizon"] == horizon
            ]

            if subset.empty:
                continue

            covered = subset[
                subset["has_fresh_madis"]
            ]

            variants = [
                ("V3_original", "prediction"),
                (
                    "MADIS_weight_0.25",
                    "forecast_weight_0.25",
                ),
                (
                    "MADIS_weight_0.50",
                    "forecast_weight_0.50",
                ),
            ]

            for name, column in variants:
                summaries.append({
                    "horizon": horizon,
                    "assumed_madis_lag": lag,
                    "variant": name,
                    "days": (
                        subset["target_day"].nunique()
                    ),
                    "covered_n": len(covered),
                    "coverage_pct": round(
                        len(covered) / len(subset) * 100,
                        1,
                    ),
                    **metrics(subset, column),
                    "mae_on_covered_only": (
                        metrics(covered, column)["mae_f"]
                    ),
                })

    summary = pd.DataFrame(summaries)

    detail = pd.concat(
        details, ignore_index=True
    )

    tag = (
        f"nyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    summary.to_csv(
        OUT / f"comparison_{tag}.csv",
        index=False,
    )

    detail.to_csv(
        OUT / f"paired_predictions_{tag}.csv",
        index=False,
    )

    print(
        "\n=== PRELIMINARY PAIRED MADIS TEST ==="
    )
    print(
        summary.to_string(index=False)
    )

    print(
        "\nNOTE: MADIS adjustments are fixed "
        "heuristics, not retrained ML models."
    )

    print(
        "Historical publication delays are "
        "assumed, not measured."
    )

    print(
        "The available six-day archive provides "
        "only a preliminary comparison."
    )

    print("\nOutputs:", OUT.resolve())


if __name__ == "__main__":
    main()
