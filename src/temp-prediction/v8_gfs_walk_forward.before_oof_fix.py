
#!/usr/bin/env python3
"""Expanding-window GFS versus V7 backtest. Research only."""

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)
TAG = "knyc5_airports6"
SRC = Path(__file__).resolve().parent
OUT = (
    ROOT / "v8_gfs/training_results"
    / TAG / "walk_forward_monthly"
)

CONFIGS = {
    "gfs_delta_only": ["gfs_delta_f"],
    "gfs_bias_only": ["gfs_bias_f"],
    "gfs_delta_plus_bias": [
        "gfs_delta_f", "gfs_bias_f"
    ],
}


def load_module(filename, name):
    path = SRC / filename
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    absolute = np.abs(error)
    return {
        "mae_f": float(absolute.mean()),
        "within_1f_pct": float(
            100 * (absolute <= 1).mean()
        ),
        "above_2f_n": int((absolute > 2).sum()),
        "bias_f": float(error.mean()),
    }


def bootstrap(times, actual, baseline, candidate):
    gain = (
        np.abs(actual - baseline)
        - np.abs(actual - candidate)
    )
    days = pd.DatetimeIndex(times).tz_convert(
        "America/New_York"
    ).date

    daily = pd.DataFrame({
        "day": days,
        "gain": gain,
    }).groupby("day")["gain"].agg(["sum", "count"])

    sums = daily["sum"].to_numpy()
    counts = daily["count"].to_numpy()

    rng = np.random.default_rng(20261109)
    n = len(daily)
    draws = rng.integers(0, n, size=(5000, n))
    samples = (
        sums[draws].sum(axis=1)
        / counts[draws].sum(axis=1)
    )

    lo, hi = np.quantile(samples, [.025, .975])

    return (
        float(lo),
        float(hi),
        int((daily["sum"] > 0).sum()),
        n,
    )


def main():
    v7 = load_module(
        "train_knyc_v7_history.py",
        "v7_walkforward",
    )
    v8 = load_module(
        "train_knyc_v8_gfs.py",
        "v8_walkforward",
    )

    archive = v8.GFSArchive(
        ROOT / "v8_gfs/gfs_hourly_runs.csv",
        6,
    )

    print("Building historical V7 features...", flush=True)
    grid = v7.build_grid(v7.import_v3(), 5, 6)

    grid["target_time"] = pd.to_datetime(
        grid["target_time"], utc=True
    )
    grid["decision"] = pd.to_datetime(
        grid["decision"], utc=True
    )
    grid["day"] = (
        grid["target_time"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    grid = grid[
        grid["day"].between("2026-07-01", "2026-10-26")
    ].copy()

    keys = ["horizon", "target_time"]

    if grid.duplicated(keys).any():
        raise RuntimeError("Duplicate V7 target times")

    # V7 predictions produced out of sample,
    # using chronological historical training folds.
    oof = pd.read_csv(
        ROOT / "v6_c2_empirical_distributions"
        / TAG / "chronological_oof_residuals.csv"
    )
    oof["target_time"] = pd.to_datetime(
        oof["target_time"], utc=True
    )
    oof = oof[
        keys + ["predicted_f", "actual_f"]
    ].rename(columns={
        "predicted_f": "v7_oof_f",
        "actual_f": "oof_actual_f",
    })

    if oof.duplicated(keys).any():
        raise RuntimeError("Duplicate V7 OOF predictions")

    # October uses the existing frozen V7 model,
    # originally trained using data through September.
    frozen = pd.read_csv(
        ROOT / "v7_extended_history"
        / "training_results" / TAG
        / "predictions.csv"
    )
    frozen["target_time"] = pd.to_datetime(
        frozen["target_time"], utc=True
    )
    frozen = frozen[
        frozen["variant"] == "V7_full_dedicated"
    ][keys + ["prediction_f"]].rename(columns={
        "prediction_f": "v7_october_f"
    })

    if frozen.duplicated(keys).any():
        raise RuntimeError("Duplicate frozen V7 predictions")

    grid = grid.merge(
        oof, on=keys, how="left",
        validate="one_to_one"
    )
    grid = grid.merge(
        frozen, on=keys, how="left",
        validate="one_to_one"
    )

    early = grid["day"] < "2026-10-01"

    target_difference = (
        grid.loc[early, "target"]
        - grid.loc[early, "oof_actual_f"]
    ).abs()

    if (
        target_difference.isna().any()
        or target_difference.max() > 0.001
    ):
        raise RuntimeError(
            "Missing or mismatched V7 OOF observations"
        )

    grid["v7_f"] = np.where(
        early,
        grid["v7_oof_f"],
        grid["v7_october_f"],
    )

    if grid["v7_f"].isna().any():
        raise RuntimeError(
            "Missing V7 predictions in requested period"
        )

    # Recreate GFS features using only model runs
    # assumed available before each decision time.
    extra = []

    for record in grid.itertuples(index=False):
        gfs = archive.features(
            record.decision,
            record.target_time,
            record.NYC_temp,
        )

        extra.append({
            "gfs_delta_f": (
                np.nan if gfs is None
                else gfs["gfs_temp_change_f"]
            ),
            "gfs_bias_f": (
                np.nan if gfs is None
                else gfs["gfs_bias_current_f"]
            ),
            "gfs_run": (
                None if gfs is None
                else gfs["gfs_run_init_utc"]
            ),
        })

    grid = pd.concat([
        grid.reset_index(drop=True),
        pd.DataFrame(extra),
    ], axis=1)

    grid = grid.dropna(subset=[
        "gfs_delta_f",
        "gfs_bias_f",
        "v7_f",
        "target",
    ])

    folds = [
        ("August", "2026-08-01", "2026-08-31"),
        ("September", "2026-09-01", "2026-09-30"),
        ("October", "2026-10-01", "2026-10-26"),
    ]

    summary = []
    predictions = []

    for horizon in (30, 45):
        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        for month, start, end in folds:
            train = frame[
                (frame["day"] >= "2026-07-01")
                & (frame["day"] < start)
            ].copy()

            test = frame[
                frame["day"].between(start, end)
            ].copy()

            if len(train) < 200 or len(test) < 150:
                raise RuntimeError(
                    f"Insufficient {horizon}m {month} data: "
                    f"train={len(train)}, test={len(test)}"
                )

            if (
                train["target_time"].max()
                >= test["target_time"].min()
            ):
                raise RuntimeError(
                    "Train/test chronology violation"
                )

            # Only earlier residuals are used.
            residual = (
                train["target"].to_numpy(float)
                - train["v7_f"].to_numpy(float)
            )

            actual = test["target"].to_numpy(float)
            baseline = test["v7_f"].to_numpy(float)

            candidates = {
                "v7_baseline": baseline,
                "constant_bias":
                    baseline + residual.mean(),
            }

            for name, features in CONFIGS.items():
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=40.0),
                )
                model.fit(
                    train[features],
                    residual,
                )

                correction = np.clip(
                    model.predict(test[features]),
                    -1.5,
                    1.5,
                )
                candidates[name] = baseline + correction

            print(
                f"\n{horizon}m {month}: "
                f"train={len(train)} test={len(test)}"
            )

            for name, predicted in candidates.items():
                score = metrics(actual, predicted)

                low, high, days_won, days = bootstrap(
                    test["target_time"],
                    actual,
                    baseline,
                    predicted,
                )

                gain = (
                    np.abs(actual - baseline).mean()
                    - score["mae_f"]
                )

                print(
                    f"  {name:21s} "
                    f"MAE={score['mae_f']:.5f}F "
                    f"within1={score['within_1f_pct']:.2f}% "
                    f"gain={gain:+.5f}F "
                    f"CI=[{low:+.5f},{high:+.5f}] "
                    f"days={days_won}/{days}"
                )

                summary.append({
                    "horizon": horizon,
                    "month": month,
                    "variant": name,
                    "train_n": len(train),
                    "test_n": len(test),
                    "gain_vs_v7_f": gain,
                    "gain_ci_low_f": low,
                    "gain_ci_high_f": high,
                    "days_improved": days_won,
                    "days_total": days,
                    **score,
                })

                predictions.append(pd.DataFrame({
                    "target_time":
                        test["target_time"].to_numpy(),
                    "horizon": horizon,
                    "month": month,
                    "variant": name,
                    "actual_f": actual,
                    "predicted_f": predicted,
                    "gfs_run": test["gfs_run"].to_numpy(),
                }))

    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary).to_csv(
        OUT / "comparison.csv", index=False
    )
    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        OUT / "predictions.csv", index=False
    )

    print("\nWalk-forward test complete.")
    print("Results:", OUT)
    print("Existing V7/V8 models unchanged.")
    print(
        "Retrospective evaluation: September and October "
        "have already influenced earlier model research."
    )


if __name__ == "__main__":
    main()
