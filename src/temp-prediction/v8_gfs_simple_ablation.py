#!/usr/bin/env python3
"""Research-only GFS temperature-change experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
BASE = Path("data/research/nyc_observation_20260701_20261026")
TAG = "knyc5_airports6"

RUNS = BASE / "v8_gfs/gfs_hourly_runs.csv"
OOF = (
    BASE / "v6_c2_empirical_distributions"
    / TAG / "chronological_oof_residuals.csv"
)
EXISTING = (
    BASE / "v8_gfs/training_results"
    / TAG / "predictions.csv"
)
OUT = (
    BASE / "v8_gfs/training_results"
    / TAG / "simple_ablation"
)


def load_file(filename, alias):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metric(y, predicted):
    error = np.abs(
        np.asarray(y, float)
        - np.asarray(predicted, float)
    )
    return {
        "mae_f": float(error.mean()),
        "within_1f_pct": float(100 * (error <= 1).mean()),
        "over_2f_n": int((error > 2).sum()),
    }


def main():
    v7 = load_file(
        "train_knyc_v7_history.py",
        "v7_simple_ablation",
    )
    v8 = load_file(
        "train_knyc_v8_gfs.py",
        "v8_simple_ablation",
    )

    archive = v8.GFSArchive(RUNS, lag_hours=6)

    print("Building historical V7 features...", flush=True)
    grid = v7.build_grid(v7.import_v3(), 5, 6)

    grid["target_time"] = pd.to_datetime(
        grid["target_time"], utc=True
    )
    grid["decision"] = pd.to_datetime(
        grid["decision"], utc=True
    )
    grid["local_day"] = (
        grid["target_time"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    grid = grid[
        grid["local_day"].between(
            "2026-07-01", "2026-10-26"
        )
    ].copy()

    if grid.duplicated(["horizon", "target_time"]).any():
        raise RuntimeError("Duplicate V7 prediction times")

    oof = pd.read_csv(OOF)
    oof["target_time"] = pd.to_datetime(
        oof["target_time"], utc=True
    )
    if oof.duplicated(["horizon", "target_time"]).any():
        raise RuntimeError("Duplicate V7 OOF prediction times")

    oof = oof[
        ["horizon", "target_time", "predicted_f", "actual_f"]
    ].rename(columns={
        "predicted_f": "v7_oof_f",
        "actual_f": "oof_actual_f",
    })

    grid = grid.merge(
        oof,
        on=["horizon", "target_time"],
        how="left",
        validate="one_to_one",
    )

    existing = pd.read_csv(EXISTING)
    existing["target_time"] = pd.to_datetime(
        existing["target_time"], utc=True
    )
    existing = existing[
        existing["variant"].isin([
            "v7_baseline", "gfs_ridge"
        ])
    ]

    if existing.duplicated([
        "horizon", "target_time", "variant"
    ]).any():
        raise RuntimeError("Duplicate existing V8 predictions")

    lookup = existing.pivot(
        index=["horizon", "target_time"],
        columns="variant",
        values="predicted_f",
    ).reset_index()

    actual = existing[
        existing["variant"] == "v7_baseline"
    ][["horizon", "target_time", "actual_f"]]

    lookup = lookup.merge(
        actual,
        on=["horizon", "target_time"],
        validate="one_to_one",
    )

    grid = grid.merge(
        lookup,
        on=["horizon", "target_time"],
        how="left",
        validate="one_to_one",
    )

    # Retrieve only the two GFS features being tested.
    gfs_features = []
    for record in grid.itertuples(index=False):
        f = archive.features(
            record.decision,
            record.target_time,
            record.NYC_temp,
        )
        gfs_features.append({
            "gfs_delta_f": (
                np.nan if f is None
                else f["gfs_temp_change_f"]
            ),
            "gfs_bias_f": (
                np.nan if f is None
                else f["gfs_bias_current_f"]
            ),
            "gfs_run": (
                None if f is None
                else f["gfs_run_init_utc"]
            ),
        })

    grid = pd.concat(
        [
            grid.reset_index(drop=True),
            pd.DataFrame(gfs_features),
        ],
        axis=1,
    )

    configurations = {
        "gfs_delta_only": ["gfs_delta_f"],
        "gfs_bias_only": ["gfs_bias_f"],
        "gfs_delta_plus_bias": [
            "gfs_delta_f", "gfs_bias_f"
        ],
    }

    rows = []
    forecasts = []

    for horizon in (30, 45):
        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        train = frame[
            frame["local_day"].between(
                "2026-07-01", "2026-08-31"
            )
        ].dropna(subset=[
            "v7_oof_f",
            "oof_actual_f",
            "gfs_delta_f",
            "gfs_bias_f",
        ]).copy()

        if len(train) < 400:
            raise RuntimeError(
                f"{horizon}m: only {len(train)} training rows"
            )

        residual = (
            train["oof_actual_f"].to_numpy(float)
            - train["v7_oof_f"].to_numpy(float)
        )

        print(
            f"\n{horizon}m training rows: {len(train)}",
            flush=True,
        )

        models = {}

        for name, columns in configurations.items():
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=40.0),
            )
            model.fit(train[columns], residual)
            models[name] = model

        for period, start, end in [
            ("September", "2026-09-01", "2026-09-30"),
            ("October", "2026-10-01", "2026-10-26"),
        ]:
            test = frame[
                frame["local_day"].between(start, end)
            ].dropna(subset=[
                "v7_baseline",
                "gfs_ridge",
                "actual_f",
                "gfs_delta_f",
                "gfs_bias_f",
            ]).copy()

            if len(test) < 150:
                raise RuntimeError(
                    f"{horizon}m {period}: "
                    f"only {len(test)} test rows"
                )

            y = test["actual_f"].to_numpy(float)
            baseline = test["v7_baseline"].to_numpy(float)

            candidates = {
                "v7_baseline": baseline,
                "constant_bias": baseline + residual.mean(),
                "full_gfs_ridge": (
                    test["gfs_ridge"].to_numpy(float)
                ),
            }

            for name, columns in configurations.items():
                correction = np.clip(
                    models[name].predict(test[columns]),
                    -1.5,
                    1.5,
                )
                candidates[name] = baseline + correction

            print(f"\n{horizon}m {period} | N={len(test)}")

            for name, prediction in candidates.items():
                result = metric(y, prediction)

                rows.append({
                    "horizon": horizon,
                    "period": period,
                    "variant": name,
                    "n": len(test),
                    **result,
                })

                print(
                    f"  {name:21s} "
                    f"MAE={result['mae_f']:.5f}F  "
                    f"within1={result['within_1f_pct']:.2f}%  "
                    f"above2={result['over_2f_n']}"
                )

                forecasts.append(pd.DataFrame({
                    "target_time": test["target_time"].to_numpy(),
                    "horizon": horizon,
                    "period": period,
                    "variant": name,
                    "actual_f": y,
                    "predicted_f": prediction,
                    "gfs_run": test["gfs_run"].to_numpy(),
                }))

    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        OUT / "comparison.csv",
        index=False,
    )
    pd.concat(
        forecasts, ignore_index=True
    ).to_csv(
        OUT / "predictions.csv",
        index=False,
    )

    print("\nExperiment complete.")
    print("Saved to:", OUT)
    print("V7 and original V8 models unchanged.")
    print("October remains an exploratory evaluation.")


if __name__ == "__main__":
    main()
