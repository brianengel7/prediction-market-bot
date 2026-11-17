#!/usr/bin/env python3
"""Frozen V7 vs V8 unseen holdout evaluation. Research only."""

import importlib.util
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("data/research/nyc_observation_20260701_20261026")
TAG = "knyc5_airports6"
HOLD = ROOT / "v8_unseen_holdout_20261027_20261108"
SRC = Path("src/temp-prediction")

STATIONS = ("NYC", "LGA", "EWR", "TEB")
HORIZONS = (30, 45)
TEST_START = "2026-10-27"
TEST_END = "2026-11-08"
GFS_LAG_HOURS = 6


def load_module(filename, alias):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def to_utc(values):
    return pd.to_datetime(values, utc=True, errors="coerce")


def build_extended_grid(v7, v3):
    """Reuse V7 station transformations and V3 as-of joins."""

    fresh = pd.read_csv(
        HOLD / "iem_station_observations.csv",
        low_memory=False,
    )
    fresh["valid_utc"] = to_utc(fresh["valid"])

    cutoff = pd.Timestamp("2026-10-27", tz="UTC")
    fresh = fresh[fresh["valid_utc"] >= cutoff]

    reports = {}
    fields = [
        "station", "valid", "tmpf", "dwpf",
        "drct", "sknt", "metar",
    ]

    with tempfile.TemporaryDirectory(
        prefix="v8_holdout_"
    ) as temporary:
        for station in STATIONS:
            old_path = (
                v7.EXT /
                f"{station}_20240101_20261026.csv"
            )

            old = pd.read_csv(
                old_path, low_memory=False
            )
            old["valid_utc"] = to_utc(old["valid"])
            old = old[old["valid_utc"] < cutoff]

            additional = fresh[
                fresh["request_station"] == station
            ].copy()

            for field in fields:
                if field not in old or field not in additional:
                    raise RuntimeError(
                        f"{station}: missing {field}"
                    )

            combined = pd.concat(
                [old[fields], additional[fields]],
                ignore_index=True,
            )
            combined["valid"] = to_utc(combined["valid"])

            # Reject conflicting duplicate temperatures.
            conflicts = pd.DataFrame({
                "time": combined["valid"],
                "temperature": pd.to_numeric(
                    combined["tmpf"], errors="coerce"
                ),
            }).groupby("time")["temperature"].nunique()

            if (conflicts > 1).any():
                raise RuntimeError(
                    f"{station}: conflicting reports"
                )

            temporary_path = (
                Path(temporary) / f"{station}.csv"
            )
            combined.to_csv(
                temporary_path, index=False
            )

            reports[station] = v7.load_station(
                temporary_path,
                station,
                5 if station == "NYC" else 6,
            )

            print(
                f"{station}: "
                f"{len(reports[station])} usable observations",
                flush=True,
            )

    knyc = reports["NYC"]

    targets = knyc[
        knyc["observed"].dt.minute == 51
    ].copy()

    targets["precise_target"] = (
        targets["metar"].map(v7.precise_f)
    )
    targets = targets.dropna(
        subset=["precise_target"]
    )

    if targets["observed"].duplicated().any():
        raise RuntimeError("Duplicate KNYC targets")

    labels = targets[
        ["observed", "precise_target"]
    ].rename(columns={"observed": "target_time"})

    recent = knyc[
        knyc["observed"] >=
        pd.Timestamp("2026-07-01", tz="UTC")
    ].copy()

    grid = v3.make_grid(recent)
    grid = grid[
        grid["horizon"].isin(HORIZONS)
    ].copy()

    for station in STATIONS:
        grid = pd.concat(
            [
                grid,
                v3.join_station(
                    grid, reports[station], station
                ),
            ],
            axis=1,
        )

    grid = grid[
        grid["NYC_temp"].notna()
        & grid[
            [f"{s}_temp" for s in STATIONS[1:]]
        ].notna().any(axis=1)
    ].copy()

    for station in STATIONS[1:]:
        grid[f"{station}_minus_nyc"] = (
            grid[f"{station}_temp"] - grid["NYC_temp"]
        )

    grid = grid.merge(
        labels,
        on="target_time",
        how="inner",
        validate="many_to_one",
    ).reset_index(drop=True)

    grid["target"] = grid["precise_target"]
    grid = grid[
        grid["target_day"].between(
            "2026-07-01", TEST_END
        )
    ].copy()

    if grid.duplicated(
        ["horizon", "target_time"]
    ).any():
        raise RuntimeError("Duplicate forecast targets")

    return grid


def attach_training_predictions(grid):
    """Use genuine earlier V7 predictions, not fitted values."""

    keys = ["horizon", "target_time"]

    oof = pd.read_csv(
        ROOT / "v6_c2_empirical_distributions"
        / TAG / "chronological_oof_residuals.csv"
    )
    oof["target_time"] = to_utc(oof["target_time"])

    oof = oof[
        oof["horizon"].isin(HORIZONS)
    ][keys + ["predicted_f", "actual_f"]].rename(
        columns={
            "predicted_f": "oof_f",
            "actual_f": "oof_actual",
        }
    )

    september = pd.read_csv(
        ROOT / "v8_gfs/training_results"
        / TAG / "predictions.csv"
    )
    september["target_time"] = to_utc(
        september["target_time"]
    )

    september = september[
        (september["period"] == "September")
        & (september["variant"] == "v7_baseline")
    ][keys + ["predicted_f", "actual_f"]].rename(
        columns={
            "predicted_f": "sep_f",
            "actual_f": "sep_actual",
        }
    )

    if (
        oof.duplicated(keys).any()
        or september.duplicated(keys).any()
    ):
        raise RuntimeError("Duplicate training predictions")

    grid = grid.merge(
        oof, on=keys, how="left",
        validate="one_to_one",
    )
    grid = grid.merge(
        september, on=keys, how="left",
        validate="one_to_one",
    )

    july_aug = grid["target_day"].between(
        "2026-07-01", "2026-08-31"
    )
    sep = grid["target_day"].between(
        "2026-09-01", "2026-09-30"
    )

    for label, mask, pred, actual, expected in [
        ("July-August OOF", july_aug,
         "oof_f", "oof_actual", 558),
        ("September pre-fit", sep,
         "sep_f", "sep_actual", 269),
    ]:
        for horizon in HORIZONS:
            part = grid[
                mask & (grid["horizon"] == horizon)
            ]

            if (
                len(part) != expected
                or part[
                    [pred, actual, "target"]
                ].isna().any().any()
            ):
                raise RuntimeError(
                    f"{label} {horizon}m coverage failed"
                )

            difference = (
                part[actual] - part["target"]
            ).abs().max()

            if difference > 0.001:
                raise RuntimeError(
                    f"{label}: target mismatch"
                )

        print(
            f"Verified {label}: {expected} per horizon"
        )

    grid["train_v7_f"] = np.where(
        july_aug,
        grid["oof_f"],
        np.where(sep, grid["sep_f"], np.nan),
    )

    return grid


def add_gfs(frame, archive):
    features = []

    for record in frame.itertuples(index=False):
        values = archive.features(
            record.decision,
            record.target_time,
            record.NYC_temp,
        )

        if values is None:
            features.append((np.nan, np.nan, pd.NaT))
            continue

        run = pd.to_datetime(
            values["gfs_run_init_utc"], utc=True
        )

        if run > (
            record.decision -
            pd.Timedelta(hours=GFS_LAG_HOURS)
        ):
            raise RuntimeError(
                "Future GFS run detected"
            )

        features.append((
            values["gfs_temp_change_f"],
            values["gfs_bias_current_f"],
            run,
        ))

    result = frame.copy()

    result[
        ["gfs_delta_f", "gfs_bias_f", "gfs_run_utc"]
    ] = pd.DataFrame(
        features, index=result.index
    )

    return result


def metrics(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    absolute = np.abs(error)

    return {
        "mae_f": float(absolute.mean()),
        "within1_pct": float(
            100 * (absolute <= 1).mean()
        ),
        "over2_n": int((absolute > 2).sum()),
        "bias_f": float(error.mean()),
    }


def bootstrap(times, actual, baseline, challenger):
    gains = (
        np.abs(actual - baseline)
        - np.abs(actual - challenger)
    )

    days = pd.DatetimeIndex(times).tz_convert(
        "America/New_York"
    ).strftime("%Y-%m-%d")

    daily = pd.DataFrame({
        "day": days,
        "gain": gains,
    }).groupby("day")["gain"].agg(
        ["sum", "count"]
    )

    rng = np.random.default_rng(20261109)

    n = len(daily)
    samples = rng.integers(
        0, n, size=(10000, n)
    )

    sums = daily["sum"].to_numpy()
    counts = daily["count"].to_numpy()

    simulated = (
        sums[samples].sum(axis=1)
        / counts[samples].sum(axis=1)
    )

    low, high = np.quantile(
        simulated, [0.025, 0.975]
    )

    return {
        "gain_f": float(gains.mean()),
        "ci_low_f": float(low),
        "ci_high_f": float(high),
        "days_improved": int(
            (daily["sum"] > 0).sum()
        ),
        "days_total": n,
    }


def main():
    spec = json.loads(
        (HOLD / "locked_spec.json").read_text()
    )

    assert spec["primary_candidate"] == "gfs_delta_only"
    assert spec["secondary_candidate"] == "gfs_delta_plus_bias"
    assert spec["ridge_alpha"] == 40.0
    assert spec["correction_clip_f"] == 1.5
    assert spec["gfs_assumed_publication_lag_hours"] == 6

    v7 = load_module(
        "train_knyc_v7_history.py", "v7_holdout"
    )
    v8 = load_module(
        "train_knyc_v8_gfs.py", "v8_holdout"
    )
    v3 = v7.import_v3()

    print("Building extended historical feature grid...")
    grid = build_extended_grid(v7, v3)
    grid = attach_training_predictions(grid)

    frozen = joblib.load(
        ROOT / "v7_extended_history/training_results"
        / TAG / "research_models.joblib"
    )

    original = pd.read_csv(
        ROOT / "v7_extended_history/training_results"
        / TAG / "predictions.csv"
    )
    original["target_time"] = to_utc(
        original["target_time"]
    )
    original = original[
        original["variant"] == "V7_full_dedicated"
    ]

    historical_gfs = v8.GFSArchive(
        ROOT / "v8_gfs/gfs_hourly_runs.csv", 6
    )
    holdout_gfs = v8.GFSArchive(
        HOLD / "gfs_hourly_runs.csv", 6
    )

    summaries = []
    predictions = []

    for horizon in HORIZONS:
        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        entry = frozen["models"][
            f"{horizon}m_full_dedicated"
        ]

        missing = (
            set(entry["features"]) - set(frame.columns)
        )

        if missing:
            raise RuntimeError(
                f"Missing V7 features: {missing}"
            )

        frame["frozen_v7_f"] = (
            frame["NYC_temp"]
            + entry["model"].predict(
                frame[entry["features"]]
            )
        )

        # Verify October forecasts before scoring new data.
        parity = frame[
            frame["target_day"].between(
                "2026-10-24", "2026-10-26"
            )
        ]

        reference = original[
            original["horizon"] == horizon
        ][["target_time", "prediction_f"]]

        comparison = parity.merge(
            reference,
            on="target_time",
            how="left",
            validate="one_to_one",
        )

        if (
            len(comparison) != 27
            or comparison["prediction_f"].isna().any()
        ):
            raise RuntimeError(
                f"{horizon}m V7 parity coverage failed"
            )

        deviation = (
            comparison["frozen_v7_f"]
            - comparison["prediction_f"]
        ).abs().max()

        print(
            f"{horizon}m V7 parity: "
            f"{len(comparison)} forecasts, "
            f"max difference {deviation:.9f}F"
        )

        if deviation > 0.0001:
            raise RuntimeError(
                "Frozen V7 parity failed. "
                "Refusing holdout evaluation."
            )

        train = frame[
            frame["target_day"].between(
                "2026-07-01", "2026-09-30"
            )
        ].copy()

        test = frame[
            frame["target_day"].between(
                TEST_START, TEST_END
            )
        ].copy()

        train = add_gfs(train, historical_gfs)
        test = add_gfs(test, holdout_gfs)

        print(
            f"\n{horizon}m available rows: "
            f"train={len(train)} test={len(test)}"
        )

        for name, part in [
            ("Training", train), ("Holdout", test)
        ]:
            missing_gfs = part[
                ["gfs_delta_f", "gfs_bias_f"]
            ].isna().any(axis=1).sum()

            if missing_gfs:
                print(
                    f"WARNING: {name} missing GFS "
                    f"{missing_gfs}/{len(part)}"
                )

        train = train.dropna(subset=[
            "train_v7_f", "target",
            "gfs_delta_f", "gfs_bias_f",
        ])

        test = test.dropna(subset=[
            "frozen_v7_f", "target",
            "gfs_delta_f", "gfs_bias_f",
        ])

        if len(train) != 827 or len(test) < 100:
            raise RuntimeError(
                f"Unexpected coverage: "
                f"{horizon}m train={len(train)} "
                f"test={len(test)}"
            )

        if (
            train["target_time"].max()
            >= test["target_time"].min()
        ):
            raise RuntimeError(
                "Training/test chronology violated"
            )

        residuals = (
            train["target"].to_numpy(float)
            - train["train_v7_f"].to_numpy(float)
        )

        actual = test["target"].to_numpy(float)
        baseline = test["frozen_v7_f"].to_numpy(float)

        candidates = {
            "v7_baseline": baseline,
            "constant_bias": baseline + residuals.mean(),
        }

        for name, features in [
            ("gfs_delta_only", ["gfs_delta_f"]),
            (
                "gfs_delta_plus_bias",
                ["gfs_delta_f", "gfs_bias_f"],
            ),
        ]:
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=40.0),
            )

            model.fit(
                train[features], residuals
            )

            correction = np.clip(
                model.predict(test[features]),
                -1.5, 1.5,
            )

            candidates[name] = baseline + correction

        print(
            f"\n{'=' * 65}\n"
            f"{horizon}-MINUTE UNSEEN HOLDOUT | "
            f"N={len(test)} | "
            f"Days={test['target_day'].nunique()}\n"
            f"{'=' * 65}"
        )

        for name, predicted in candidates.items():
            result = metrics(actual, predicted)
            paired = bootstrap(
                test["target_time"],
                actual,
                baseline,
                predicted,
            )

            print(
                f"{name:22s} "
                f"MAE={result['mae_f']:.5f}F "
                f"within1={result['within1_pct']:.2f}% "
                f"above2={result['over2_n']} "
                f"gain={paired['gain_f']:+.5f}F "
                f"CI=[{paired['ci_low_f']:+.5f},"
                f"{paired['ci_high_f']:+.5f}] "
                f"days={paired['days_improved']}/"
                f"{paired['days_total']}"
            )

            summaries.append({
                "horizon": horizon,
                "variant": name,
                "train_n": len(train),
                "test_n": len(test),
                **result,
                **paired,
            })

            predictions.append(pd.DataFrame({
                "target_time": test["target_time"].to_numpy(),
                "target_day": test["target_day"].to_numpy(),
                "horizon": horizon,
                "variant": name,
                "actual_f": actual,
                "predicted_f": predicted,
                "gfs_delta_f": test["gfs_delta_f"].to_numpy(),
                "gfs_bias_f": test["gfs_bias_f"].to_numpy(),
                "gfs_run_utc": test["gfs_run_utc"].to_numpy(),
            }))

    output = HOLD / "evaluation_frozen"
    output.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("comparison.csv", pd.DataFrame(summaries)),
        (
            "predictions.csv",
            pd.concat(predictions, ignore_index=True),
        ),
    ]:
        path = output / name

        if path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing evaluation: "
                f"{path}"
            )

        data.to_csv(path, index=False)

    print("\nUNSEEN HOLDOUT EVALUATION COMPLETE")
    print("Saved:", output)
    print("Existing models and trading code unchanged.")


if __name__ == "__main__":
    main()
