
#!/usr/bin/env python3
"""V6: Frozen V3/V5 baselines versus V3/V5 + archived HRRR.

Train July-August, select September, evaluate October.
Research only. No live trades or changes to V3/V5 models.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)
HRRR_PATH = (
    ROOT / "v6_hrrr_research" / "hourly_hrrr_runs.csv"
)
HORIZONS = (30, 45)
STATIONS = ("NYC", "LGA", "EWR", "TEB")


def load_module(filename, name):
    path = Path(__file__).with_name(filename)

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_grid(v3, v5, nyc_lag, airport_lag):
    reports = {
        s: v3.load_station(
            s,
            nyc_lag if s == "NYC" else airport_lag,
        )
        for s in STATIONS
    }

    grid = v3.make_grid(reports["NYC"])

    for station in STATIONS:
        features = v3.join_station(
            grid, reports[station], station
        )
        grid = pd.concat(
            [grid, features], axis=1
        )

    airport_cols = [
        f"{s}_temp" for s in STATIONS[1:]
    ]

    grid = grid[
        grid["NYC_temp"].notna()
        & grid[airport_cols].notna().any(axis=1)
    ].copy().reset_index(drop=True)

    for station in STATIONS[1:]:
        grid[f"{station}_minus_nyc"] = (
            grid[f"{station}_temp"]
            - grid["NYC_temp"]
        )

    labels = pd.read_csv(
        ROOT / "v5_inputs"
        / "knyc_precise_targets.csv"
    )

    labels["observed_utc"] = pd.to_datetime(
        labels["observed_utc"],
        utc=True,
        errors="coerce",
    )

    labels["precise_f"] = pd.to_numeric(
        labels["precise_f"],
        errors="coerce",
    )

    labels = labels.dropna(
        subset=["observed_utc", "precise_f"]
    )

    if labels["observed_utc"].duplicated().any():
        raise ValueError(
            "Conflicting precise KNYC targets"
        )

    grid = grid.merge(
        labels[["observed_utc", "precise_f"]],
        left_on="target_time",
        right_on="observed_utc",
        how="inner",
        validate="many_to_one",
    )

    grid["target"] = grid["precise_f"]

    grid = grid.drop(
        columns=["observed_utc", "precise_f"]
    ).reset_index(drop=True)

    solar = v5.solar_features(
        grid["decision"]
    )

    grid = pd.concat(
        [grid, solar.reset_index(drop=True)],
        axis=1,
    )

    clouds = pd.read_csv(
        ROOT / "v5_inputs"
        / "hourly_metar_clouds.csv",
        low_memory=False,
    )

    for station in STATIONS:
        lag = (
            nyc_lag if station == "NYC"
            else airport_lag
        )

        cloud_columns = v5.cloud_features(
            grid, clouds, station, lag
        )

        grid = pd.concat(
            [grid, cloud_columns], axis=1
        )

    return grid


def interpolate_at(times, values, requests):
    valid = np.isfinite(values)

    if valid.sum() < 2:
        return np.full(
            len(requests), np.nan
        )

    return np.interp(
        requests,
        times[valid],
        values[valid],
        left=np.nan,
        right=np.nan,
    )


def add_hrrr(grid, path, publication_lag):
    df = pd.read_csv(path)

    required = {
        "run_utc", "valid_utc",
        "temp_f", "cloud_pct", "wind_mph",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing HRRR columns: "
            f"{required - set(df.columns)}"
        )

    for col in ("run_utc", "valid_utc"):
        df[col] = pd.to_datetime(
            df[col],
            utc=True,
            errors="coerce",
        )

    for col in (
        "temp_f", "cloud_pct", "wind_mph"
    ):
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(
        subset=["run_utc", "valid_utc"]
    ).sort_values(
        ["run_utc", "valid_utc"]
    )

    if df.duplicated(
        ["run_utc", "valid_utc"]
    ).any():
        raise ValueError(
            "Duplicate HRRR forecast slots"
        )

    available = (
        df[["run_utc"]]
        .drop_duplicates()
        .copy()
    )

    available["run_available_utc"] = (
        available["run_utc"]
        + pd.Timedelta(
            minutes=publication_lag
        )
    )

    available = available.sort_values(
        "run_available_utc"
    )

    left = grid[["decision"]].copy()
    left["_id"] = np.arange(len(grid))

    matched = pd.merge_asof(
        left.sort_values("decision"),
        available,
        left_on="decision",
        right_on="run_available_utc",
        direction="backward",
    )

    matched = (
        matched.sort_values("_id")
        .reset_index(drop=True)
    )

    result = grid.copy().reset_index(
        drop=True
    )

    result["H_run_utc"] = matched["run_utc"]

    result["H_run_age_min"] = (
        result["decision"]
        - matched["run_utc"]
    ).dt.total_seconds() / 60

    # Reconstruct the latest eligible KNYC report
    # from V3's causally aligned observation age.
    last_observed = (
        result["decision"]
        - pd.to_timedelta(
            result["NYC_age_min"],
            unit="m",
        )
    )

    feature_names = [
        "H_delta_from_knyc_obs_f",
        "H_delta_from_decision_f",
        "H_temp_at_target_f",
        "H_temp_at_decision_f",
        "H_cloud_target_pct",
        "H_wind_target_mph",
        "H_cloud_change_pct",
        "H_wind_change_mph",
    ]

    for col in feature_names:
        result[col] = np.nan

    groups = result[
        result["H_run_utc"].notna()
    ].groupby("H_run_utc").groups

    for run, indices in groups.items():
        forecast = df[
            df["run_utc"] == run
        ]

        times = (
            forecast["valid_utc"]
            .map(lambda t: t.timestamp())
            .to_numpy(float)
        )

        ids = np.asarray(
            list(indices), dtype=int
        )

        targets = (
            result.loc[ids, "target_time"]
            .map(lambda t: t.timestamp())
            .to_numpy(float)
        )

        decisions = (
            result.loc[ids, "decision"]
            .map(lambda t: t.timestamp())
            .to_numpy(float)
        )

        previous = (
            last_observed.iloc[ids]
            .map(
                lambda t:
                    t.timestamp()
                    if pd.notna(t)
                    else np.nan
            )
            .to_numpy(float)
        )

        temp = forecast[
            "temp_f"
        ].to_numpy(float)

        cloud = forecast[
            "cloud_pct"
        ].to_numpy(float)

        wind = forecast[
            "wind_mph"
        ].to_numpy(float)

        temp_target = interpolate_at(
            times, temp, targets
        )

        temp_decision = interpolate_at(
            times, temp, decisions
        )

        temp_previous = interpolate_at(
            times, temp, previous
        )

        cloud_target = interpolate_at(
            times, cloud, targets
        )

        cloud_decision = interpolate_at(
            times, cloud, decisions
        )

        wind_target = interpolate_at(
            times, wind, targets
        )

        wind_decision = interpolate_at(
            times, wind, decisions
        )

        values = {
            "H_delta_from_knyc_obs_f":
                temp_target - temp_previous,

            "H_delta_from_decision_f":
                temp_target - temp_decision,

            "H_temp_at_target_f":
                temp_target,

            "H_temp_at_decision_f":
                temp_decision,

            "H_cloud_target_pct":
                cloud_target,

            "H_wind_target_mph":
                wind_target,

            "H_cloud_change_pct":
                cloud_target - cloud_decision,

            "H_wind_change_mph":
                wind_target - wind_decision,
        }

        for col, data in values.items():
            result.loc[ids, col] = data

    # Ensure no forecast run becomes available
    # after the simulated decision timestamp.
    valid = result["H_run_utc"].notna()

    assert (
        result.loc[valid, "H_run_utc"]
        + pd.Timedelta(
            minutes=publication_lag
        )
        <= result.loc[valid, "decision"]
    ).all()

    return result


def metrics(actual, predicted):
    errors = np.abs(
        np.asarray(actual, float)
        - np.asarray(predicted, float)
    )

    return {
        "n": len(errors),

        "mae_f": round(
            float(errors.mean()), 4
        ),

        "within_half_pct": round(
            float((errors <= 0.5).mean() * 100),
            1,
        ),

        "within_one_pct": round(
            float((errors <= 1).mean() * 100),
            1,
        ),

        "errors_gt_2": int(
            (errors > 2).sum()
        ),
    }


def make_model(leaves):
    return HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=leaves,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )


def self_test():
    times = np.array(
        [0.0, 3600.0, 7200.0]
    )

    temps = np.array(
        [60.0, 62.0, 64.0]
    )

    result = interpolate_at(
        times,
        temps,
        np.array([
            1800.0, -1.0, 7201.0
        ]),
    )

    assert abs(
        result[0] - 61
    ) < 1e-9

    assert np.isnan(result[1])
    assert np.isnan(result[2])

    print("SELF-TEST PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--airport-lag",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--hrrr-publication-lag",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    lags = [
        args.knyc_lag,
        args.airport_lag,
        args.hrrr_publication_lag,
    ]

    if min(lags) < 0:
        parser.error(
            "Lags must be nonnegative"
        )

    if not HRRR_PATH.exists():
        raise FileNotFoundError(
            f"Collect HRRR runs first: {HRRR_PATH}"
        )

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    saved_v5 = (
        ROOT / "v5_precise_clouds"
        / tag / "research_models.joblib"
    )

    if not saved_v5.exists():
        raise FileNotFoundError(
            f"Missing frozen models: {saved_v5}"
        )

    v3 = load_module(
        "train_knyc_v3_30_45.py",
        "v3_readonly_v6",
    )

    v5 = load_module(
        "train_knyc_v5_precise_clouds.py",
        "v5_readonly_v6",
    )

    frozen = joblib.load(saved_v5)[
        "models"
    ]

    print(
        "Reconstructing V3/V5 features...",
        flush=True,
    )

    grid = build_grid(
        v3, v5,
        args.knyc_lag,
        args.airport_lag,
    )

    print(
        "Rows by horizon:",
        grid.groupby("horizon")
        .size().to_dict(),
        flush=True,
    )

    grid = add_hrrr(
        grid,
        HRRR_PATH,
        args.hrrr_publication_lag,
    )

    grid["split"] = pd.cut(
        grid["target_day"],
        bins=[
            "2026-06-30",
            "2026-08-31",
            "2026-09-30",
            "2026-10-27",
        ],
        labels=[
            "train", "validation", "October"
        ],
    )

    coverage = grid.groupby(
        ["horizon", "split"],
        observed=True,
    )["H_delta_from_knyc_obs_f"].agg(
        ["size", "count"]
    )

    print(
        "\nHRRR coverage:\n",
        coverage.to_string(),
        flush=True,
    )

    # Same events for existing models and V6.
    matched = grid[
        grid["H_delta_from_knyc_obs_f"].notna()
    ].copy()

    train = matched[
        matched["target_day"] < "2026-09-01"
    ].copy()

    valid = matched[
        matched["target_day"].between(
            "2026-09-01", "2026-09-30"
        )
    ].copy()

    test = matched[
        matched["target_day"].between(
            "2026-10-01", "2026-10-26"
        )
    ].copy()

    print(
        "Matched train/September/October:",
        len(train), len(valid), len(test),
        flush=True,
    )

    if (
        len(train) < 1000
        or len(valid) < 100
        or len(test) < 100
    ):
        raise RuntimeError(
            "Insufficient HRRR coverage. "
            "Check missing runs before training."
        )

    variants = {
        "v6_v3_delta": (
            "v3_precise",
            [
                "H_delta_from_knyc_obs_f",
                "H_run_age_min",
            ],
        ),

        "v6_v3_weather": (
            "v3_precise",
            [
                "H_delta_from_knyc_obs_f",
                "H_delta_from_decision_f",
                "H_cloud_target_pct",
                "H_wind_target_mph",
                "H_run_age_min",
            ],
        ),

        "v6_v5_delta": (
            "plus_solar_clouds",
            [
                "H_delta_from_knyc_obs_f",
                "H_run_age_min",
            ],
        ),

        "v6_v5_weather": (
            "plus_solar_clouds",
            [
                "H_delta_from_knyc_obs_f",
                "H_delta_from_decision_f",
                "H_cloud_target_pct",
                "H_wind_target_mph",
                "H_run_age_min",
            ],
        ),
    }

    results = []
    predictions = []
    models = {}

    for horizon in HORIZONS:
        val_h = valid[
            valid["horizon"] == horizon
        ]

        test_h = test[
            test["horizon"] == horizon
        ]

        print(
            f"\n=== {horizon}m forecasts ===",
            f"September={len(val_h)}",
            f"October={len(test_h)}",
            flush=True,
        )

        if (
            len(val_h) < 30
            or len(test_h) < 30
        ):
            raise RuntimeError(
                "Insufficient matched target events"
            )

        # Frozen baseline evaluation.
        for name in (
            "v3_precise",
            "plus_clouds",
            "plus_solar_clouds",
        ):
            saved = frozen[
                f"{horizon}m_{name}"
            ]

            columns = saved["features"]

            missing = [
                c for c in columns
                if c not in grid.columns
            ]

            if missing:
                raise RuntimeError(
                    f"Missing V5 features: {missing}"
                )

            model = saved["model"]

            val_prediction = (
                val_h["NYC_temp"].to_numpy()
                + model.predict(val_h[columns])
            )

            test_prediction = (
                test_h["NYC_temp"].to_numpy()
                + model.predict(test_h[columns])
            )

            val_error = np.abs(
                val_h["target"].to_numpy()
                - val_prediction
            )

            results.append({
                "horizon": horizon,
                "variant": "FROZEN_" + name,
                "sep_mae": round(
                    float(val_error.mean()), 4
                ),
                "sep_within_one_pct": round(
                    float(
                        (val_error <= 1).mean()
                        * 100
                    ),
                    1,
                ),
                "leaves": saved["leaves"],
                **metrics(
                    test_h["target"],
                    test_prediction,
                ),
            })

            predictions.append(
                pd.DataFrame({
                    "target_time":
                        test_h["target_time"],
                    "horizon": horizon,
                    "actual_precise_f":
                        test_h["target"],
                    "variant": "FROZEN_" + name,
                    "prediction_f":
                        test_prediction,
                })
            )

        # Train V6 variants.
        for name, (base, extras) in variants.items():
            candidates = list(dict.fromkeys(
                frozen[
                    f"{horizon}m_{base}"
                ]["features"] + extras
            ))

            features = []

            for col in candidates:
                values = (
                    pd.to_numeric(
                        train[col],
                        errors="coerce",
                    )
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                )

                if (
                    values.notna().sum() >= 2
                    and values.nunique(
                        dropna=True
                    ) > 1
                ):
                    features.append(col)

            if (
                "H_delta_from_knyc_obs_f"
                not in features
            ):
                raise RuntimeError(
                    "HRRR delta unavailable "
                    "in training data"
                )

            for frame in (
                train, val_h, test_h
            ):
                frame.loc[:, features] = (
                    frame[features]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                )

            best = None

            for leaves in (7, 15, 31):
                model = make_model(leaves)

                model.fit(
                    train[features],
                    train["target"]
                    - train["NYC_temp"],
                )

                val_prediction = (
                    val_h["NYC_temp"].to_numpy()
                    + model.predict(
                        val_h[features]
                    )
                )

                val_error = np.abs(
                    val_h["target"].to_numpy()
                    - val_prediction
                )

                val_mae = float(
                    val_error.mean()
                )

                val_hit = float(
                    (val_error <= 1).mean()
                    * 100
                )

                # Prioritize +/-1°F accuracy.
                # Break ties using lower MAE.
                ranking = (-val_hit, val_mae)

                if (
                    best is None
                    or ranking < best[0]
                ):
                    best = (
                        ranking,
                        val_mae,
                        val_hit,
                        leaves,
                        model,
                    )

            (
                _, val_mae, val_hit,
                leaves, model
            ) = best

            prediction = (
                test_h["NYC_temp"].to_numpy()
                + model.predict(
                    test_h[features]
                )
            )

            test_metrics = metrics(
                test_h["target"],
                prediction,
            )

            results.append({
                "horizon": horizon,
                "variant": name,
                "sep_mae": round(
                    val_mae, 4
                ),
                "sep_within_one_pct": round(
                    val_hit, 1
                ),
                "leaves": leaves,
                **test_metrics,
            })

            predictions.append(
                pd.DataFrame({
                    "target_time":
                        test_h["target_time"],
                    "horizon": horizon,
                    "actual_precise_f":
                        test_h["target"],
                    "variant": name,
                    "prediction_f": prediction,
                })
            )

            models[f"{horizon}m_{name}"] = {
                "model": model,
                "features": features,
                "sep_mae": val_mae,
                "leaves": leaves,
                "publication_lag_minutes":
                    args.hrrr_publication_lag,
            }

            print(
                f"  {name}: "
                f"Sep ±1°F={val_hit:.1f}% "
                f"Sep MAE={val_mae:.4f} "
                f"Oct ±1°F="
                f"{test_metrics['within_one_pct']:.1f}%",
                flush=True,
            )

    # Verify existing model predictions were
    # reproduced without changing their features.
    earlier = pd.read_csv(
        ROOT / "v5_precise_clouds"
        / tag / "predictions.csv"
    )

    earlier["target_time"] = pd.to_datetime(
        earlier["target_time"], utc=True
    )

    generated = pd.concat(
        predictions, ignore_index=True
    )

    generated["target_time"] = pd.to_datetime(
        generated["target_time"], utc=True
    )

    for name in (
        "v3_precise",
        "plus_clouds",
        "plus_solar_clouds",
    ):
        rebuilt = generated[
            generated["variant"]
            == "FROZEN_" + name
        ]

        saved = earlier[
            earlier["variant"] == name
        ]

        check = rebuilt.merge(
            saved,
            on=["target_time", "horizon"],
            suffixes=("_rebuilt", "_saved"),
            validate="one_to_one",
        )

        if len(check) < 50:
            raise RuntimeError(
                "Cannot verify frozen-model parity"
            )

        difference = float(
            (
                check["prediction_f_rebuilt"]
                - check["prediction_f_saved"]
            ).abs().max()
        )

        print(
            f"Frozen {name} parity: "
            f"{len(check)} events, "
            f"max difference={difference:.7f}°F",
            flush=True,
        )

        if difference > 0.01:
            raise RuntimeError(
                "V3/V5 feature reconstruction "
                "does not match saved predictions."
            )

    out = (
        ROOT / "v6_hrrr_ml"
        / tag
        / f"publication_lag_"
          f"{args.hrrr_publication_lag}m"
    )

    out.mkdir(
        parents=True, exist_ok=True
    )

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        out / "comparison.csv",
        index=False,
    )

    generated.to_csv(
        out / "predictions.csv",
        index=False,
    )

    selected = {}

    for horizon in HORIZONS:
        options = comparison[
            (comparison["horizon"] == horizon)
            & comparison["variant"].str.startswith(
                "v6_"
            )
        ]

        winner = options.sort_values(
            [
                "sep_within_one_pct",
                "sep_mae",
            ],
            ascending=[False, True],
        ).iloc[0]

        selected[horizon] = winner["variant"]

    joblib.dump(
        {
            "models": models,
            "selected_by_september": selected,
            "knyc_lag": args.knyc_lag,
            "airport_lag": args.airport_lag,
            "hrrr_publication_lag_minutes":
                args.hrrr_publication_lag,
        },
        out / "research_models.joblib",
    )

    print(
        "\n=== V6 HRRR MODEL COMPARISON "
        "(OCTOBER EXPLORATORY) ==="
    )

    print(
        comparison.to_string(index=False)
    )

    print(
        "\nSelected by September:",
        selected,
    )

    print(
        "Saved to:",
        out.resolve(),
    )


if __name__ == "__main__":
    main()
