#!/usr/bin/env python3
"""V9 long-history climatology research. No trading activity."""

import argparse
import importlib.util
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/research/nyc_observation_20260701_20261026")
SRC = Path("src/temp-prediction")
OUT = ROOT / "v9_historical_climatology/long_history_2010_2025"
RAW = OUT / "raw"
HISTORY = OUT / "precise_hourly_2010_2023.csv"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
KEYS = ["horizon", "target_time"]
LENGTHS = (2, 6, 11, 16)
METHODS = ("window14", "harmonic2")


def load(filename, name):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def utc(values):
    return pd.to_datetime(
        values, utc=True, errors="coerce", format="mixed"
    )


def download_year(year):
    path = RAW / f"NYC_{year}.csv"
    if path.exists() and path.stat().st_size > 1000:
        return path

    params = {
        "network": "NY_ASOS",
        "station": "NYC",
        "data": "all",
        "year1": year,
        "month1": 1,
        "day1": 1,
        "year2": year + 1,
        "month2": 1,
        "day2": 1,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": [3, 4],
    }

    for attempt in range(1, 9):
        try:
            response = requests.get(
                IEM, params=params, timeout=(60, 180)
            )
            response.raise_for_status()

            if response.text.lstrip().startswith("<"):
                raise ValueError("IEM returned HTML, not CSV")

            frame = pd.read_csv(
                io.StringIO(response.text),
                comment="#", low_memory=False
            )

            required = {"station", "valid", "metar"}
            if not required.issubset(frame.columns):
                raise ValueError(
                    f"Unexpected IEM columns: {list(frame.columns)}"
                )

            if len(frame) < 1000:
                raise ValueError(
                    f"{year} returned only {len(frame)} rows"
                )

            dates = utc(frame["valid"])
            in_year = int((dates.dt.year == year).sum())
            if in_year < 1000:
                raise ValueError(
                    f"{year}: insufficient valid timestamps"
                )

            RAW.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            temporary.write_text(response.text)
            temporary.replace(path)

            print(
                f"{year}: downloaded {len(frame)} reports",
                flush=True
            )
            time.sleep(2)
            return path

        except (requests.RequestException, ValueError) as exc:
            if attempt == 8:
                raise
            delay = min(10 * 2 ** (attempt - 1), 120)
            print(
                f"{year}: {type(exc).__name__}, "
                f"retrying after {delay}s",
                flush=True
            )
            time.sleep(delay)


def collect():
    v7 = load("train_knyc_v7_history.py", "v7_long_collect")

    for year in range(2010, 2024):
        download_year(year)

    all_years = []
    print("\n=== PRECISE KNYC OBSERVATION COVERAGE ===")

    for year in range(2010, 2024):
        path = RAW / f"NYC_{year}.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame["observed_utc"] = utc(frame["valid"])

        frame = frame[
            frame["observed_utc"].dt.year.eq(year)
            & frame["observed_utc"].dt.minute.eq(51)
            & frame["station"].astype(str).str.upper()
                .isin(["NYC", "KNYC"])
        ].copy()

        frame["temperature_f"] = frame["metar"].map(
            v7.precise_f
        )
        frame = frame.dropna(
            subset=["observed_utc", "temperature_f"]
        )

        grouped = frame.groupby("observed_utc")[
            "temperature_f"
        ].agg(["min", "max", "count"])

        conflicts = grouped[
            (grouped["max"] - grouped["min"]) > 0.001
        ]

        valid = grouped[
            (grouped["max"] - grouped["min"]) <= 0.001
        ].reset_index()

        valid = valid.rename(
            columns={"min": "temperature_f"}
        )[["observed_utc", "temperature_f"]]

        valid["year"] = year
        all_years.append(valid)

        print(
            f"{year}: {len(valid):5d} precise :51 readings; "
            f"{len(conflicts)} conflicting timestamps",
            flush=True
        )

        if len(valid) < 3000:
            raise RuntimeError(
                f"{year} has sparse precise observations; "
                "investigate coverage before modeling"
            )

    result = pd.concat(all_years, ignore_index=True)
    if result.duplicated("observed_utc").any():
        raise RuntimeError("Duplicate archived observation time")

    OUT.mkdir(parents=True, exist_ok=True)
    if HISTORY.exists():
        raise RuntimeError(
            f"Normalized archive already exists: {HISTORY}"
        )

    result.sort_values("observed_utc").to_csv(
        HISTORY, index=False
    )

    print("\nCollection complete:", len(result), "readings")
    print("Saved:", HISTORY)


def seasonal_day(value):
    return pd.Timestamp(
        "2000-" + value.strftime("%m-%d")
    ).dayofyear


def harmonic_matrix(days):
    radians = 2 * np.pi * np.asarray(days) / 366.0
    return np.column_stack([
        np.ones_like(radians),
        np.sin(radians),
        np.cos(radians),
        np.sin(2 * radians),
        np.cos(2 * radians),
    ])


def climatology_history(v7):
    older = pd.read_csv(HISTORY)
    older["observed_utc"] = utc(older["observed_utc"])

    recent = pd.read_csv(
        v7.EXT / "NYC_20240101_20261026.csv",
        usecols=["valid", "metar"],
        low_memory=False,
    )
    recent["observed_utc"] = utc(recent["valid"])
    recent = recent[
        recent["observed_utc"].dt.year.isin([2024, 2025])
        & recent["observed_utc"].dt.minute.eq(51)
    ].copy()

    recent["temperature_f"] = recent["metar"].map(
        v7.precise_f
    )
    recent = recent.dropna(
        subset=["observed_utc", "temperature_f"]
    )
    recent["year"] = recent["observed_utc"].dt.year

    history = pd.concat([
        older[["observed_utc", "temperature_f", "year"]],
        recent[["observed_utc", "temperature_f", "year"]],
    ], ignore_index=True)

    local = history["observed_utc"].dt.tz_convert(
        "America/New_York"
    )
    history["hour"] = local.dt.hour
    history["season_day"] = local.apply(seasonal_day)

    print("\nHistory by year:")
    print(history.groupby("year").size().to_string())

    return history


def add_long_climatology(grid, history, original_v9):
    result = grid.copy()

    # Reuse the exact two-year feature from the original V9.
    result["clim_2y_window14"] = (
        original_v9["clim_anomaly_f"].to_numpy()
    )

    target_local = result["target_time"].dt.tz_convert(
        "America/New_York"
    )
    observation_time = (
        result["decision"]
        - pd.to_timedelta(result["NYC_age_min"], unit="m")
    )
    observed_local = observation_time.dt.tz_convert(
        "America/New_York"
    )

    current_days = observed_local.apply(seasonal_day).to_numpy()
    current_hours = observed_local.dt.hour.to_numpy()

    # Only 2024-2025 observations contribute to the
    # original two-year reference. Older years are added
    # without using any 2026 temperature outcomes.
    for years in LENGTHS:
        subset = history[
            history["year"] >= 2026 - years
        ].copy()

        hourly = {}
        harmonic = {}

        for hour, part in subset.groupby("hour"):
            days = part["season_day"].to_numpy(int)
            temperatures = part[
                "temperature_f"
            ].to_numpy(float)

            hourly[hour] = (days, temperatures)

            coefficients = np.linalg.lstsq(
                harmonic_matrix(days),
                temperatures,
                rcond=None,
            )[0]

            harmonic[hour] = coefficients

        cache = {}

        def normal(hour, day, method):
            key = (int(hour), int(day), method)
            if key in cache:
                return cache[key]

            if hour not in hourly:
                return np.nan

            if method == "window14":
                days, temp = hourly[hour]
                distance = np.abs(days - day)
                distance = np.minimum(
                    distance, 366 - distance
                )
                selected = temp[distance <= 14]

                value = (
                    float(selected.mean())
                    if len(selected) >= 20
                    else np.nan
                )
            else:
                value = float(
                    harmonic_matrix([day])[0]
                    @ harmonic[hour]
                )

            cache[key] = value
            return value

        for method in METHODS:
            name = f"clim_{years}y_{method}"

            if years == 2 and method == "window14":
                continue

            expected = np.array([
                normal(int(hour), int(day), method)
                for hour, day in zip(
                    current_hours, current_days
                )
            ])

            result[name] = (
                result["NYC_temp"].to_numpy(float)
                - expected
            )

        print(
            f"Built {years}-year seasonal climatology",
            flush=True
        )

    return result


def metric(actual, predicted):
    absolute = np.abs(
        np.asarray(actual) - np.asarray(predicted)
    )
    return (
        float(absolute.mean()),
        float(100 * (absolute <= 1).mean()),
        int((absolute > 2).sum()),
    )


def ci_by_day(days, actual, reference, challenger):
    gain = (
        np.abs(actual - reference)
        - np.abs(actual - challenger)
    )
    daily = pd.DataFrame({
        "day": days,
        "gain": gain,
    }).groupby("day")["gain"].agg(["sum", "count"])

    rng = np.random.default_rng(20261109)
    n = len(daily)
    draws = rng.integers(0, n, size=(5000, n))

    sums = daily["sum"].to_numpy()
    counts = daily["count"].to_numpy()

    samples = (
        sums[draws].sum(axis=1)
        / counts[draws].sum(axis=1)
    )

    return np.quantile(samples, [0.025, 0.975])


def evaluate():
    if not HISTORY.exists():
        raise FileNotFoundError(
            "Collect historical data first: " + str(HISTORY)
        )

    v7 = load("train_knyc_v7_history.py", "v7_long")
    v8 = load("train_knyc_v8_gfs.py", "v8_long")
    v9 = load("v9_historical_walkforward.py", "v9_long")
    scorer = load("v8_unseen_holdout_score.py", "scorer_long")

    print("Rebuilding original V7 and V8 features...")
    grid = scorer.build_extended_grid(
        v7, v7.import_v3()
    )
    grid = grid[
        grid["horizon"].isin([30, 45])
        & grid["target_day"].between(
            "2026-07-01", "2026-11-08"
        )
    ].copy()

    grid["target_time"] = utc(grid["target_time"])
    grid["decision"] = utc(grid["decision"])

    grid = v9.add_baselines(grid)
    grid = v9.add_gfs(grid, v8)

    original_v9 = v9.add_climatology(
        grid.copy(), v7
    )
    history = climatology_history(v7)

    grid = add_long_climatology(
        grid, history, original_v9
    )

    variants = {
        "V8_reference": [
            "gfs_delta_f", "gfs_bias_f"
        ],
    }

    for years in LENGTHS:
        for method in METHODS:
            variants[f"V9_{years}y_{method}"] = [
                "gfs_delta_f",
                "gfs_bias_f",
                f"clim_{years}y_{method}",
            ]

    needed = sorted(
        set(sum(variants.values(), []))
    ) + ["target", "v7_pred"]

    if grid[needed].isna().any().any():
        print(grid[needed].isna().sum())
        raise RuntimeError("Incomplete model inputs")

    folds = [
        ("August", "2026-08-01", "2026-08-31",
         "2026-08-01", 279),
        ("September", "2026-09-01", "2026-09-30",
         "2026-09-01", 269),
        ("October", "2026-10-01", "2026-10-26",
         "2026-10-01", 234),
        ("Nov27_Nov08", "2026-10-27", "2026-11-08",
         "2026-10-01", 117),
    ]

    reference_path = (
        ROOT / "v9_historical_climatology"
        / "walkforward_20261109/comparison.csv"
    )
    reference = pd.read_csv(reference_path)

    results = []
    predictions = []

    for horizon in (30, 45):
        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        for fold, start, end, train_end, expected in folds:
            train = frame[
                frame["target_day"].between(
                    "2026-07-01", "2026-09-30"
                )
                & (frame["target_day"] < train_end)
            ]
            test = frame[
                frame["target_day"].between(start, end)
            ]

            if len(train) < 200 or len(test) != expected:
                raise RuntimeError(
                    f"Bad coverage: {horizon}m {fold}: "
                    f"train={len(train)} test={len(test)}"
                )

            if (
                train["target_time"].max()
                >= test["target_time"].min()
            ):
                raise RuntimeError(
                    "Chronology violation"
                )

            residuals = (
                train["target"].to_numpy(float)
                - train["v7_pred"].to_numpy(float)
            )

            actual = test["target"].to_numpy(float)
            baseline = test["v7_pred"].to_numpy(float)

            candidates = {"V7_baseline": baseline}

            for variant, features in variants.items():
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=40.0),
                )
                model.fit(train[features], residuals)
                correction = np.clip(
                    model.predict(test[features]),
                    -1.5, 1.5,
                )
                candidates[variant] = baseline + correction

            v8_pred = candidates["V8_reference"]
            v8_mae = metric(actual, v8_pred)[0]

            # Check V8 reproduces the preceding V9 study.
            prior = reference[
                (reference["horizon"] == horizon)
                & (reference["fold"] == fold)
                & (
                    reference["variant"]
                    == "v8_gfs_delta_bias"
                )
            ]

            if (
                len(prior) != 1
                or abs(
                    float(prior.iloc[0]["mae_f"]) - v8_mae
                ) > 1e-6
            ):
                raise RuntimeError(
                    f"V8 reference parity failed: "
                    f"{horizon}m {fold}"
                )

            print(
                f"\n{horizon}m {fold}: "
                f"train={len(train)}, test={len(test)} "
                f"(V8 MAE={v8_mae:.5f}F)"
            )

            for variant, prediction in candidates.items():
                mae, within1, above2 = metric(
                    actual, prediction
                )
                low, high = ci_by_day(
                    test["target_day"].to_numpy(),
                    actual, v8_pred, prediction
                )

                gain = v8_mae - mae
                print(
                    f"  {variant:20s} "
                    f"MAE={mae:.5f} "
                    f"gain_vs_V8={gain:+.5f} "
                    f"CI=[{low:+.5f},{high:+.5f}]"
                )

                results.append({
                    "horizon": horizon,
                    "fold": fold,
                    "variant": variant,
                    "train_n": len(train),
                    "test_n": len(test),
                    "mae_f": mae,
                    "within1_pct": within1,
                    "above2_n": above2,
                    "gain_vs_v8_f": gain,
                    "ci_low_f": low,
                    "ci_high_f": high,
                })

                predictions.append(pd.DataFrame({
                    "horizon": horizon,
                    "fold": fold,
                    "variant": variant,
                    "target_time":
                        test["target_time"].to_numpy(),
                    "target_day":
                        test["target_day"].to_numpy(),
                    "actual_f": actual,
                    "predicted_f": prediction,
                }))

    for filename in ("comparison.csv", "predictions.csv"):
        if (OUT / filename).exists():
            raise RuntimeError(
                f"Refusing to overwrite {OUT / filename}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(
        OUT / "comparison.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / "predictions.csv", index=False
    )

    print("\nLONG-HISTORY V9 EXPERIMENT COMPLETE")
    print("Saved:", OUT)
    print("No existing models or trading code changed.")
    print("All evaluation periods are exploratory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["collect", "evaluate"]
    )
    args = parser.parse_args()

    if args.mode == "collect":
        collect()
    else:
        evaluate()
