#!/usr/bin/env python3
"""Historical-temperature V9 research. No orders or model overwrites."""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/research/nyc_observation_20260701_20261026")
TAG = "knyc5_airports6"
SRC = Path("src/temp-prediction")
HOLD = ROOT / "v8_unseen_holdout_20261027_20261108"
OUT = ROOT / "v9_historical_climatology/walkforward_20261109"
KEYS = ["horizon", "target_time"]

VARIANTS = {
    "v8_gfs_delta_bias": [
        "gfs_delta_f", "gfs_bias_f"
    ],
    "v9_climatology_only": [
        "clim_anomaly_f", "clim_expected_change_f"
    ],
    "v9_plus_anomaly": [
        "gfs_delta_f", "gfs_bias_f", "clim_anomaly_f"
    ],
    "v9_plus_seasonal_change": [
        "gfs_delta_f", "gfs_bias_f", "clim_expected_change_f"
    ],
    "v9_plus_both": [
        "gfs_delta_f", "gfs_bias_f",
        "clim_anomaly_f", "clim_expected_change_f"
    ],
}


def load(name, alias):
    spec = importlib.util.spec_from_file_location(
        alias, SRC / name
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def utc(values):
    return pd.to_datetime(
        values, utc=True, errors="coerce", format="mixed"
    )


def seasonal_day(timestamp):
    return pd.Timestamp(
        "2000-" + timestamp.strftime("%m-%d")
    ).dayofyear


def add_climatology(grid, v7):
    history = pd.read_csv(
        v7.EXT / "NYC_20240101_20261026.csv",
        usecols=["valid", "metar"],
        low_memory=False,
    )

    history["observed"] = utc(history["valid"])
    history = history.dropna(subset=["observed"])
    history = history[
        history["observed"].dt.year.isin([2024, 2025])
        & history["observed"].dt.minute.eq(51)
    ].copy()

    history["precise_f"] = history["metar"].map(
        v7.precise_f
    )
    history = history.dropna(subset=["precise_f"])

    local = history["observed"].dt.tz_convert(
        "America/New_York"
    )
    history["hour"] = local.dt.hour
    history["season_day"] = local.apply(seasonal_day)

    by_hour = {
        h: (
            part["season_day"].to_numpy(int),
            part["precise_f"].to_numpy(float),
        )
        for h, part in history.groupby("hour")
    }

    print(
        "Historical 2024–2025 precise observations:",
        len(history),
        flush=True,
    )

    def historical_mean(hour, season, window=14):
        days, temps = by_hour.get(
            hour, (
                np.array([], dtype=int),
                np.array([], dtype=float),
            )
        )
        distance = np.abs(days - season)
        distance = np.minimum(distance, 366 - distance)
        selected = temps[distance <= window]
        if len(selected) < 20:
            return np.nan, len(selected)
        return float(selected.mean()), len(selected)

    anomalies = []
    historical_changes = []
    counts = []

    for r in grid.itertuples(index=False):
        target_local = pd.Timestamp(
            r.target_time
        ).tz_convert("America/New_York")

        # V7's station join exposes NYC_age_min rather
        # than NYC_observed. Recover the latest observation
        # timestamp using information available at decision.
        if not np.isfinite(r.NYC_age_min):
            raise RuntimeError(
                "Missing KNYC observation age"
            )

        observed_time = (
            pd.Timestamp(r.decision)
            - pd.Timedelta(
                minutes=float(r.NYC_age_min)
            )
        )

        observed_local = observed_time.tz_convert(
            "America/New_York"
        )

        target_mean, n1 = historical_mean(
            target_local.hour,
            seasonal_day(target_local),
        )
        current_mean, n2 = historical_mean(
            observed_local.hour,
            seasonal_day(observed_local),
        )

        anomalies.append(r.NYC_temp - current_mean)
        historical_changes.append(
            target_mean - current_mean
        )
        counts.append(min(n1, n2))

    grid = grid.copy()
    grid["clim_anomaly_f"] = anomalies
    grid["clim_expected_change_f"] = historical_changes
    grid["clim_observations"] = counts

    return grid


def add_baselines(grid):
    oof = pd.read_csv(
        ROOT / "v6_c2_empirical_distributions"
        / TAG / "chronological_oof_residuals.csv"
    )
    oof["target_time"] = utc(oof["target_time"])
    oof = oof[
        oof["horizon"].isin([30, 45])
    ][KEYS + ["predicted_f", "actual_f"]].rename(
        columns={
            "predicted_f": "oof_pred",
            "actual_f": "oof_actual",
        }
    )

    sep = pd.read_csv(
        ROOT / "v8_gfs/training_results"
        / TAG / "predictions.csv"
    )
    sep["target_time"] = utc(sep["target_time"])
    sep = sep[
        (sep["period"] == "September")
        & (sep["variant"] == "v7_baseline")
    ][KEYS + ["predicted_f", "actual_f"]].rename(
        columns={
            "predicted_f": "sep_pred",
            "actual_f": "sep_actual",
        }
    )

    octv = pd.read_csv(
        ROOT / "v7_extended_history/training_results"
        / TAG / "predictions.csv"
    )
    octv["target_time"] = utc(octv["target_time"])
    octv = octv[
        octv["variant"] == "V7_full_dedicated"
    ][KEYS + ["prediction_f"]].rename(
        columns={"prediction_f": "oct_pred"}
    )

    nov = pd.read_csv(
        HOLD / "evaluation_frozen/predictions.csv"
    )
    nov["target_time"] = utc(nov["target_time"])
    nov = nov[
        nov["variant"] == "v7_baseline"
    ][KEYS + ["predicted_f", "actual_f"]].rename(
        columns={
            "predicted_f": "nov_pred",
            "actual_f": "nov_actual",
        }
    )

    for part, name in [
        (oof, "OOF"), (sep, "September"),
        (octv, "October"), (nov, "November")
    ]:
        if part.duplicated(KEYS).any():
            raise RuntimeError(
                f"Duplicate {name} forecasts"
            )
        grid = grid.merge(
            part, on=KEYS,
            how="left", validate="one_to_one"
        )

    day = grid["target_day"]
    masks = [
        day <= "2026-08-31",
        day.between("2026-09-01", "2026-09-30"),
        day.between("2026-10-01", "2026-10-26"),
        day.between("2026-10-27", "2026-11-08"),
    ]

    grid["v7_pred"] = np.select(
        masks,
        [
            grid["oof_pred"],
            grid["sep_pred"],
            grid["oct_pred"],
            grid["nov_pred"],
        ],
        default=np.nan,
    )

    if grid["v7_pred"].isna().any():
        raise RuntimeError("Missing V7 predictions")

    for name, mask, column in [
        ("July-August", masks[0], "oof_actual"),
        ("September", masks[1], "sep_actual"),
        ("November", masks[3], "nov_actual"),
    ]:
        difference = (
            grid.loc[mask, "target"]
            - grid.loc[mask, column]
        ).abs()

        if (
            difference.isna().any()
            or difference.max() > 0.001
        ):
            raise RuntimeError(
                f"{name} target mismatch"
            )

    return grid


def add_gfs(grid, v8):
    historical = v8.GFSArchive(
        ROOT / "v8_gfs/gfs_hourly_runs.csv", 6
    )
    recent = v8.GFSArchive(
        HOLD / "gfs_hourly_runs.csv", 6
    )

    values = []

    for r in grid.itertuples(index=False):
        archive = (
            recent
            if r.target_day >= "2026-10-27"
            else historical
        )

        features = archive.features(
            r.decision, r.target_time, r.NYC_temp
        )

        if features is None:
            values.append((np.nan, np.nan))
            continue

        run = pd.Timestamp(
            features["gfs_run_init_utc"]
        )
        if run.tzinfo is None:
            run = run.tz_localize("UTC")

        if run > r.decision - pd.Timedelta(hours=6):
            raise RuntimeError(
                "GFS as-of guard failed"
            )

        values.append((
            features["gfs_temp_change_f"],
            features["gfs_bias_current_f"],
        ))

    result = grid.copy()
    result[
        ["gfs_delta_f", "gfs_bias_f"]
    ] = pd.DataFrame(
        values, index=result.index
    )
    return result


def metrics(actual, predicted):
    errors = np.abs(actual - predicted)
    return (
        float(errors.mean()),
        float(100 * (errors <= 1).mean()),
        int((errors > 2).sum()),
    )


def bootstrap(day, y, reference, candidate):
    gains = (
        np.abs(y - reference)
        - np.abs(y - candidate)
    )

    daily = pd.DataFrame({
        "day": np.asarray(day),
        "gain": gains,
    }).groupby("day")["gain"].agg(
        ["sum", "count"]
    )

    rng = np.random.default_rng(20261109)
    n = len(daily)
    draws = rng.integers(
        0, n, size=(5000, n)
    )
    sums = daily["sum"].to_numpy()
    counts = daily["count"].to_numpy()

    boot = (
        sums[draws].sum(axis=1)
        / counts[draws].sum(axis=1)
    )

    return np.quantile(boot, [0.025, 0.975])


def main():
    v7 = load(
        "train_knyc_v7_history.py", "v7_v9"
    )
    v8 = load(
        "train_knyc_v8_gfs.py", "v8_v9"
    )
    scorer = load(
        "v8_unseen_holdout_score.py", "scorer_v9"
    )

    print("Rebuilding historical features...")
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

    grid = add_baselines(grid)
    grid = add_climatology(grid, v7)
    grid = add_gfs(grid, v8)

    required = sorted(
        set(sum(VARIANTS.values(), []))
    ) + ["target", "v7_pred"]

    if grid[required].isna().any().any():
        print(grid[required].isna().sum())
        raise RuntimeError(
            "Incomplete features; stopping"
        )

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

    results = []
    predictions = []

    for horizon in (30, 45):
        data = grid[
            grid["horizon"] == horizon
        ].copy()

        for fold, start, end, train_end, expected in folds:
            train = data[
                data["target_day"].between(
                    "2026-07-01", "2026-09-30"
                )
                & (data["target_day"] < train_end)
            ]

            test = data[
                data["target_day"].between(start, end)
            ].copy()

            if (
                len(train) < 200
                or len(test) != expected
            ):
                raise RuntimeError(
                    f"{horizon}m {fold}: "
                    f"train={len(train)}, test={len(test)}"
                )

            if (
                train["target_time"].max()
                >= test["target_time"].min()
            ):
                raise RuntimeError(
                    "Training/test chronology violation"
                )

            residuals = (
                train["target"].to_numpy(float)
                - train["v7_pred"].to_numpy(float)
            )

            actual = test["target"].to_numpy(float)
            baseline = test["v7_pred"].to_numpy(float)

            candidates = {"v7_baseline": baseline}

            for name, columns in VARIANTS.items():
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=40.0),
                )
                model.fit(
                    train[columns], residuals
                )
                correction = np.clip(
                    model.predict(test[columns]),
                    -1.5, 1.5,
                )
                candidates[name] = (
                    baseline + correction
                )

            print(
                f"\n{horizon}m {fold} | "
                f"train={len(train)} test={len(test)}",
                flush=True,
            )

            v8_reference = candidates[
                "v8_gfs_delta_bias"
            ]
            v8_mae = metrics(
                actual, v8_reference
            )[0]

            for name, predicted in candidates.items():
                mae, within1, above2 = metrics(
                    actual, predicted
                )

                gain_v8 = v8_mae - mae
                low, high = bootstrap(
                    test["target_day"],
                    actual,
                    v8_reference,
                    predicted,
                )

                print(
                    f"  {name:26s} "
                    f"MAE={mae:.5f}F "
                    f"within1={within1:.2f}% "
                    f">2F={above2} "
                    f"gain_vs_V8={gain_v8:+.5f}F "
                    f"CI=[{low:+.5f},{high:+.5f}]"
                )

                results.append({
                    "horizon": horizon,
                    "fold": fold,
                    "variant": name,
                    "train_n": len(train),
                    "test_n": len(test),
                    "mae_f": mae,
                    "within1_pct": within1,
                    "above2_n": above2,
                    "gain_vs_v8_f": gain_v8,
                    "gain_ci_low_f": low,
                    "gain_ci_high_f": high,
                })

                predictions.append(pd.DataFrame({
                    "horizon": horizon,
                    "fold": fold,
                    "variant": name,
                    "target_time":
                        test["target_time"].to_numpy(),
                    "target_day":
                        test["target_day"].to_numpy(),
                    "actual_f": actual,
                    "predicted_f": predicted,
                }))

    OUT.mkdir(parents=True, exist_ok=True)

    files = [
        ("comparison.csv", pd.DataFrame(results)),
        (
            "predictions.csv",
            pd.concat(predictions, ignore_index=True),
        ),
    ]

    if any((OUT / name).exists() for name, _ in files):
        raise RuntimeError(
            "Previous V9 results exist; "
            "refusing to overwrite"
        )

    for filename, data in files:
        data.to_csv(OUT / filename, index=False)

    print("\nV9 HISTORICAL EXPERIMENT COMPLETE")
    print("Results:", OUT)
    print("Existing V7/V8 models unchanged.")
    print("Prior evaluation dates are exploratory.")


if __name__ == "__main__":
    main()
