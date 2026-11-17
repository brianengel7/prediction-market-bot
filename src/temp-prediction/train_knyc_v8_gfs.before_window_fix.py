
#!/usr/bin/env python3
"""V8: GFS correction of frozen V7 KNYC forecasts.

Research only.

Training:
    Chronological V7 out-of-sample residuals,
    April through August 2026.

Selection:
    September 2026.

Exploratory evaluation:
    October 2026.

Existing V7 models and trading code are unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from bisect import bisect_right
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(
    "data/research/"
    "nyc_observation_20260701_20261026"
)

GFS_REQUIRED = (
    "run_init_utc",
    "valid_utc",
    "gfs_temp_f",
)

GFS_OPTIONAL = (
    "gfs_rh_pct",
    "gfs_cloud_pct",
    "gfs_wind_mph",
    "gfs_wind_dir_deg",
    "gfs_shortwave_wm2",
)

FEATURES = (
    "gfs_temp_now_f",
    "gfs_temp_target_f",
    "gfs_temp_change_f",
    "gfs_bias_current_f",
    "gfs_rh_target_pct",
    "gfs_cloud_target_pct",
    "gfs_wind_u_target_mph",
    "gfs_wind_v_target_mph",
    "gfs_shortwave_target_wm2",
    "gfs_run_age_hours",
    "NYC_trend1",
    "NYC_dew_depression",
    "LGA_minus_nyc",
    "EWR_minus_nyc",
    "TEB_minus_nyc",
)


def import_local(filename, name):
    path = Path(__file__).with_name(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Expected existing V7 source: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name, path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class GFSArchive:
    def __init__(self, csv_path, lag_hours):
        if lag_hours < 6.0:
            raise ValueError(
                "Require at least 6-hour "
                "conservative GFS publication delay"
            )

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing GFS CSV: {csv_path}"
            )

        df = pd.read_csv(
            csv_path, low_memory=False
        )

        missing = sorted(
            set(GFS_REQUIRED) - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Missing GFS columns: "
                + ", ".join(missing)
                + ". Inspect your existing GFS "
                  "data before training."
            )

        for field in GFS_OPTIONAL:
            if field not in df.columns:
                df[field] = np.nan

        for field in (
            "run_init_utc",
            "valid_utc",
        ):
            df[field] = pd.to_datetime(
                df[field],
                utc=True,
                errors="coerce",
            )

            if df[field].isna().any():
                raise ValueError(
                    f"Invalid timestamps: {field}"
                )

        for col in (
            "gfs_temp_f",
        ) + GFS_OPTIONAL:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        df = df.dropna(
            subset=["gfs_temp_f"]
        )

        df = df.drop_duplicates(
            ["run_init_utc", "valid_utc"],
            keep="last",
        )

        if df.empty:
            raise ValueError(
                "No numeric GFS temperatures"
            )

        self.by_run = {}

        for init, part in df.groupby(
            "run_init_utc", sort=True
        ):
            part = part.sort_values(
                "valid_utc"
            )

            if len(part) < 8:
                continue

            speed = part[
                "gfs_wind_mph"
            ].to_numpy(float)

            direction = np.deg2rad(
                part[
                    "gfs_wind_dir_deg"
                ].to_numpy(float)
            )

            self.by_run[init] = {
                "time":
                    part["valid_utc"]
                    .dt.as_unit("ns")
                    .astype("int64")
                    .to_numpy(),

                "temp":
                    part["gfs_temp_f"]
                    .to_numpy(float),

                "rh":
                    part["gfs_rh_pct"]
                    .to_numpy(float),

                "cloud":
                    part["gfs_cloud_pct"]
                    .to_numpy(float),

                "u":
                    -speed * np.sin(direction),

                "v":
                    -speed * np.cos(direction),

                "solar":
                    part["gfs_shortwave_wm2"]
                    .to_numpy(float),
            }

        self.keys = sorted(
            self.by_run
        )

        self.available = [
            init + pd.Timedelta(
                hours=lag_hours
            )
            for init in self.keys
        ]

        if not self.keys:
            raise ValueError(
                "No usable GFS runs"
            )

        self.lag_hours = lag_hours

        print(
            "GFS archive:", csv_path,
            "runs:", len(self.keys),
            "earliest:", self.keys[0],
            "latest:", self.keys[-1],
        )

    @staticmethod
    def sample(data, when):
        times = data["time"]
        ns = pd.Timestamp(when).value

        if (
            ns < times[0]
            or ns > times[-1]
        ):
            return None

        result = {}

        for name in (
            "temp", "rh", "cloud",
            "u", "v", "solar",
        ):
            values = data[name]
            finite = np.isfinite(values)

            if finite.sum() < 2:
                result[name] = np.nan
                continue

            valid_times = times[finite]
            valid_values = values[finite]

            right = np.searchsorted(
                valid_times, ns
            )

            if right == 0:
                result[name] = (
                    float(valid_values[0])
                    if ns == valid_times[0]
                    else np.nan
                )

            elif right == len(valid_times):
                result[name] = (
                    float(valid_values[-1])
                    if ns == valid_times[-1]
                    else np.nan
                )

            elif (
                valid_times[right]
                - valid_times[right - 1]
            ) > 3 * 3600 * 1e9:
                result[name] = np.nan

            else:
                result[name] = float(
                    np.interp(
                        ns,
                        valid_times,
                        valid_values,
                    )
                )

        return result

    def features(
        self, decision, target, nyc_temp
    ):
        decision = pd.Timestamp(decision)
        target = pd.Timestamp(target)

        if (
            decision.tzinfo is None
            or target.tzinfo is None
        ):
            raise ValueError(
                "UTC-aware timestamps required"
            )

        if target <= decision:
            raise ValueError(
                "Target must follow decision"
            )

        idx = (
            bisect_right(
                self.available,
                decision,
            ) - 1
        )

        # Search newest eligible model runs.
        for j in range(
            idx,
            max(-1, idx - 6),
            -1,
        ):
            init = self.keys[j]

            available_at = (
                init + pd.Timedelta(
                    hours=self.lag_hours
                )
            )

            if available_at > decision:
                raise RuntimeError(
                    "GFS look-ahead guard failed"
                )

            arr = self.by_run[init]

            before = self.sample(
                arr, decision
            )

            after = self.sample(
                arr, target
            )

            if (
                before is None
                or after is None
            ):
                continue

            if (
                not np.isfinite(before["temp"])
                or not np.isfinite(after["temp"])
            ):
                continue

            return {
                "gfs_run_init_utc":
                    init.isoformat(),

                "gfs_available_assumed_utc":
                    available_at.isoformat(),

                "gfs_temp_now_f":
                    before["temp"],

                "gfs_temp_target_f":
                    after["temp"],

                "gfs_temp_change_f":
                    after["temp"] - before["temp"],

                "gfs_bias_current_f":
                    float(nyc_temp) - before["temp"],

                "gfs_rh_target_pct":
                    after["rh"],

                "gfs_cloud_target_pct":
                    after["cloud"],

                "gfs_wind_u_target_mph":
                    after["u"],

                "gfs_wind_v_target_mph":
                    after["v"],

                "gfs_shortwave_target_wm2":
                    after["solar"],

                "gfs_run_age_hours":
                    (
                        decision - init
                    ).total_seconds() / 3600,
            }

        return None


def self_test():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as folder:
        file = Path(folder) / "run.csv"

        records = []

        for init in (
            "2026-07-10T00:00:00Z",
            "2026-07-10T06:00:00Z",
        ):
            init_ts = pd.Timestamp(init)

            for h in range(30):
                records.append({
                    "run_init_utc": init,
                    "valid_utc": (
                        init_ts
                        + pd.Timedelta(hours=h)
                    ).isoformat(),
                    "gfs_temp_f": 65 + h * 0.1,
                    "gfs_cloud_pct": 50,
                    "gfs_rh_pct": 70,
                    "gfs_wind_mph": 5,
                    "gfs_wind_dir_deg": 270,
                    "gfs_shortwave_wm2": 100,
                })

        pd.DataFrame(records).to_csv(
            file, index=False
        )

        archive = GFSArchive(file, 6)

        early = archive.features(
            pd.Timestamp(
                "2026-07-10T11:21:00Z"
            ),
            pd.Timestamp(
                "2026-07-10T11:51:00Z"
            ),
            65,
        )

        assert early[
            "gfs_run_init_utc"
        ].startswith(
            "2026-07-10T00:00"
        )

        late = archive.features(
            pd.Timestamp(
                "2026-07-10T12:21:00Z"
            ),
            pd.Timestamp(
                "2026-07-10T12:51:00Z"
            ),
            65,
        )

        assert late[
            "gfs_run_init_utc"
        ].startswith(
            "2026-07-10T06:00"
        )

        assert abs(
            late["gfs_temp_change_f"] - 0.05
        ) < 1e-6

        assert archive.features(
            pd.Timestamp(
                "2026-07-10T05:00:00Z"
            ),
            pd.Timestamp(
                "2026-07-10T05:30:00Z"
            ),
            65,
        ) is None

    print(
        "SELF-TEST PASS: GFS as-of "
        "selection and interpolation"
    )


def append_gfs(grid, archive):
    rows = []

    for row in grid.itertuples(
        index=False
    ):
        values = archive.features(
            row.decision,
            row.target_time,
            row.NYC_temp,
        )

        rows.append(
            values if values is not None else {}
        )

    extra = pd.DataFrame(
        rows, index=grid.index
    )

    required = (
        "gfs_run_init_utc",
        "gfs_available_assumed_utc",
    ) + tuple(
        f for f in FEATURES
        if f.startswith("gfs_")
    )

    for col in required:
        if col not in extra:
            extra[col] = np.nan

    return pd.concat(
        [grid, extra], axis=1
    )


def scores(actual, predicted):
    errors = (
        np.asarray(actual, float)
        - np.asarray(predicted, float)
    )

    return {
        "n": len(errors),

        "mae_f": round(
            float(np.mean(np.abs(errors))),
            5,
        ),

        "rmse_f": round(
            float(
                np.sqrt(
                    np.mean(errors ** 2)
                )
            ),
            5,
        ),

        "within_half_pct": round(
            float(
                100 * np.mean(
                    np.abs(errors) <= 0.5
                )
            ),
            2,
        ),

        "within_one_pct": round(
            float(
                100 * np.mean(
                    np.abs(errors) <= 1
                )
            ),
            2,
        ),

        "above_two_n": int(
            np.sum(
                np.abs(errors) > 2
            )
        ),

        "bias_f": round(
            float(np.mean(errors)),
            5,
        ),
    }


def bootstrap(
    data, actual, base, other,
    iterations=4000,
):
    frame = pd.DataFrame({
        "day":
            pd.to_datetime(
                data["target_time"],
                utc=True,
            ).dt.tz_convert(
                "America/New_York"
            ).dt.date,

        "gain":
            np.abs(actual - base)
            - np.abs(actual - other),
    })

    grouped = frame.groupby(
        "day"
    )["gain"].agg(
        ["sum", "size"]
    )

    sums = grouped[
        "sum"
    ].to_numpy(float)

    counts = grouped[
        "size"
    ].to_numpy(int)

    rng = np.random.default_rng(
        20261030
    )

    selections = rng.integers(
        0,
        len(sums),
        size=(iterations, len(sums)),
    )

    samples = (
        sums[selections].sum(axis=1)
        / counts[selections].sum(axis=1)
    )

    return [
        round(float(x), 5)
        for x in np.quantile(
            samples, [0.025, 0.975]
        )
    ]


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gfs-csv",
        type=Path,
        default=(
            ROOT / "v8_gfs"
            / "gfs_hourly_runs.csv"
        ),
    )

    p.add_argument(
        "--knyc-lag",
        type=int,
        default=5,
    )

    p.add_argument(
        "--airport-lag",
        type=int,
        default=6,
    )

    p.add_argument(
        "--gfs-publication-lag-hours",
        type=float,
        default=6,
    )

    p.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.85,
    )

    p.add_argument(
        "--self-test",
        action="store_true",
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    if args.gfs_publication_lag_hours < 6:
        p.error(
            "GFS publication lag must be >= 6h"
        )

    if not (
        0 < args.minimum_coverage <= 1
    ):
        p.error(
            "Invalid minimum coverage"
        )

    archive = GFSArchive(
        args.gfs_csv,
        args.gfs_publication_lag_hours,
    )

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    v7dir = (
        ROOT / "v7_extended_history"
        / "training_results" / tag
    )

    oof_file = (
        ROOT
        / "v6_c2_empirical_distributions"
        / tag
        / "chronological_oof_residuals.csv"
    )

    v7 = import_local(
        "train_knyc_v7_history.py",
        "v7_for_v8",
    )

    bundle = joblib.load(
        v7dir / "research_models.joblib"
    )

    oof = pd.read_csv(oof_file)

    oof["target_time"] = pd.to_datetime(
        oof["target_time"],
        utc=True,
    )

    if oof.duplicated(
        ["horizon", "target_time"]
    ).any():
        raise RuntimeError(
            "Duplicate OOF timestamps"
        )

    print(
        "Constructing original V7 features...",
        flush=True,
    )

    grid = v7.build_grid(
        v7.import_v3(),
        args.knyc_lag,
        args.airport_lag,
    )

    grid["target_time"] = pd.to_datetime(
        grid["target_time"],
        utc=True,
    )

    grid["decision"] = pd.to_datetime(
        grid["decision"],
        utc=True,
    )

    if grid.duplicated(
        ["horizon", "target_time"]
    ).any():
        raise RuntimeError(
            "Duplicate V7 grid target times"
        )

    for col in FEATURES:
        if (
            col not in grid
            and not col.startswith("gfs_")
        ):
            grid[col] = np.nan

    # Preserve earlier rows for a genuine
    # pre-September point-model fit.
    full_history = grid.copy()

    grid = grid[
        grid["target_day"].between(
            "2026-04-02",
            "2026-10-26",
        )
    ].copy()

    grid = append_gfs(
        grid, archive
    )

    grid["gfs_ok"] = (
        grid["gfs_temp_target_f"].notna()
        & grid["gfs_temp_now_f"].notna()
    )

    output = (
        ROOT / "v8_gfs"
        / "training_results" / tag
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_columns = [
        "target_time",
        "decision",
        "horizon",
        "gfs_run_init_utc",
        "gfs_available_assumed_utc",
        "gfs_temp_target_f",
        "gfs_temp_change_f",
        "gfs_ok",
    ]

    grid[audit_columns].to_csv(
        output / "gfs_alignment_audit.csv",
        index=False,
    )

    metrics = []
    predictions = []
    decisions = []
    artifacts = {}

    saved = pd.read_csv(
        v7dir / "predictions.csv"
    )

    saved["target_time"] = pd.to_datetime(
        saved["target_time"],
        utc=True,
    )

    for horizon in (30, 45):
        entry = bundle["models"][
            f"{horizon}m_full_dedicated"
        ]

        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        # Clone the exact V7 architecture, but
        # fit it without September observations.
        historical = full_history[
            (full_history["horizon"] == horizon)
            & (full_history["target_day"] < "2026-09-01")
        ].copy()

        sep_model = clone(entry["model"])
        sep_model.fit(
            historical[entry["features"]],
            historical["target"].to_numpy(float)
            - historical["NYC_temp"].to_numpy(float),
        )

        frame["v7_frozen"] = (
            frame["NYC_temp"].to_numpy(float)
            + entry["model"].predict(
                frame[entry["features"]]
            )
        )

        # September predictions use only the
        # pre-September refitted model.
        sep_mask = frame["target_day"].between(
            "2026-09-01", "2026-09-30"
        )

        frame.loc[sep_mask, "v7_frozen"] = (
            frame.loc[sep_mask, "NYC_temp"].to_numpy(float)
            + sep_model.predict(
                frame.loc[sep_mask, entry["features"]]
            )
        )

        print(
            f"{horizon}m September as-of V7: "
            f"trained on {len(historical)} earlier rows",
            flush=True,
        )

        frame["actual_f"] = (
            frame["target"].to_numpy(float)
        )

        reference = saved[
            (saved["horizon"] == horizon)
            & (
                saved["variant"]
                == "V7_full_dedicated"
            )
        ][
            ["target_time", "prediction_f"]
        ]

        october_parity = frame[
            frame["target_day"].between(
                "2026-10-01",
                "2026-10-26",
            )
        ][
            ["target_time", "v7_frozen"]
        ].merge(
            reference,
            on="target_time",
            validate="one_to_one",
        )

        if (
            len(october_parity) != len(reference)
            or len(october_parity) < 200
        ):
            raise RuntimeError(
                f"V7 event mismatch: {horizon}m"
            )

        delta = float(
            (
                october_parity["v7_frozen"]
                - october_parity["prediction_f"]
            ).abs().max()
        )

        if (
            not np.isfinite(delta)
            or delta > 0.00001
        ):
            raise RuntimeError(
                f"V7 parity failed: {delta}"
            )

        print(
            f"Frozen {horizon}m V7 "
            f"parity PASS: {delta:.9f}F"
        )

        part_oof = oof[
            oof["horizon"] == horizon
        ][
            [
                "target_time",
                "predicted_f",
                "actual_f",
            ]
        ].rename(
            columns={
                "predicted_f": "v7_oof",
                "actual_f": "oof_actual_f",
            }
        )

        frame = frame.merge(
            part_oof,
            on="target_time",
            how="left",
            validate="one_to_one",
        )

        train_all = frame[
            frame["target_day"].between(
                "2026-04-02",
                "2026-08-31",
            )
        ]

        validation_all = frame[
            frame["target_day"].between(
                "2026-09-01",
                "2026-09-30",
            )
        ]

        october_all = frame[
            frame["target_day"].between(
                "2026-10-01",
                "2026-10-26",
            )
        ]

        for label, raw in (
            ("train", train_all),
            ("September", validation_all),
            ("October", october_all),
        ):
            coverage = (
                float(raw["gfs_ok"].mean())
                if len(raw)
                else 0
            )

            print(
                f"{horizon}m {label}: "
                f"{len(raw)} rows, "
                f"GFS matched {coverage:.1%}"
            )

            if (
                coverage
                < args.minimum_coverage
            ):
                raise RuntimeError(
                    f"{horizon}m {label} "
                    f"coverage {coverage:.1%} "
                    f"below minimum. "
                    "Collect additional GFS runs."
                )

        train = train_all[
            train_all["gfs_ok"]
            & train_all["v7_oof"].notna()
        ].copy()

        sep = validation_all[
            validation_all["gfs_ok"]
        ].copy()

        octo = october_all[
            october_all["gfs_ok"]
        ].copy()

        if (
            len(train) < 450
            or len(sep) < 150
            or len(octo) < 150
        ):
            raise RuntimeError(
                f"Insufficient data: "
                f"{horizon}m "
                f"{len(train)}/"
                f"{len(sep)}/"
                f"{len(octo)}"
            )

        train_y = (
            train["oof_actual_f"].to_numpy(float)
            - train["v7_oof"].to_numpy(float)
        )

        x_train = train[
            list(FEATURES)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        models = {
            "gfs_ridge": make_pipeline(
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
                StandardScaler(),
                Ridge(alpha=40.0),
            ),

            "gfs_boost":
                HistGradientBoostingRegressor(
                    max_iter=110,
                    learning_rate=0.04,
                    max_leaf_nodes=7,
                    min_samples_leaf=60,
                    l2_regularization=15.0,
                    random_state=42,
                ),
        }

        for name, model in models.items():
            model.fit(
                x_train, train_y
            )

        candidates = {
            "v7_baseline": None,
            **models,
        }

        for period, data in (
            ("September", sep),
            ("October", octo),
        ):
            truth = (
                data["actual_f"].to_numpy(float)
            )

            base = (
                data["v7_frozen"].to_numpy(float)
            )

            for variant, model in candidates.items():
                if model is None:
                    correction = np.zeros(
                        len(data)
                    )
                else:
                    x = data[
                        list(FEATURES)
                    ].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )

                    correction = np.clip(
                        model.predict(x),
                        -1.5,
                        1.5,
                    )

                forecast = base + correction

                metrics.append({
                    "horizon": horizon,
                    "period": period,
                    "variant": variant,
                    **scores(truth, forecast),
                })

                predictions.append(
                    pd.DataFrame({
                        "target_time":
                            data[
                                "target_time"
                            ].to_numpy(),

                        "horizon": horizon,
                        "period": period,
                        "variant": variant,
                        "baseline_f": base,
                        "predicted_f": forecast,
                        "actual_f": truth,

                        "gfs_run_init_utc":
                            data[
                                "gfs_run_init_utc"
                            ].to_numpy(),
                    })
                )

        sep_metrics = {
            m["variant"]: m
            for m in metrics
            if (
                m["horizon"] == horizon
                and m["period"] == "September"
            )
        }

        baseline_mae = sep_metrics[
            "v7_baseline"
        ]["mae_f"]

        baseline_hit = sep_metrics[
            "v7_baseline"
        ]["within_one_pct"]

        qualified = [
            name
            for name in models
            if (
                baseline_mae
                - sep_metrics[name]["mae_f"]
                >= 0.015
            )
            and (
                sep_metrics[name][
                    "within_one_pct"
                ]
                >= baseline_hit - 1.0
            )
        ]

        chosen = (
            min(
                qualified,
                key=lambda name:
                    sep_metrics[name]["mae_f"],
            )
            if qualified
            else "v7_baseline"
        )

        oct_metrics = {
            m["variant"]: m
            for m in metrics
            if (
                m["horizon"] == horizon
                and m["period"] == "October"
            )
        }

        base_pred = (
            octo["v7_frozen"].to_numpy(float)
        )

        if chosen == "v7_baseline":
            winner = base_pred.copy()

        else:
            x_oct = octo[
                list(FEATURES)
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            winner = base_pred + np.clip(
                models[chosen].predict(x_oct),
                -1.5,
                1.5,
            )

        ci = bootstrap(
            octo,
            octo["actual_f"].to_numpy(float),
            base_pred,
            winner,
        )

        print(
            f"\n{horizon}m September selection: "
            f"{chosen}"
        )

        print(
            "October baseline:",
            oct_metrics["v7_baseline"]
        )

        print(
            "October selected:",
            oct_metrics[chosen]
        )

        print(
            "Paired bootstrap MAE gain interval:",
            ci
        )

        decisions.append({
            "horizon": horizon,
            "selected_by_september": chosen,
            "september_baseline_mae":
                baseline_mae,
            "september_selected_mae":
                sep_metrics[chosen]["mae_f"],
            "october_baseline_mae":
                oct_metrics[
                    "v7_baseline"
                ]["mae_f"],
            "october_selected_mae":
                oct_metrics[chosen]["mae_f"],
            "october_mae_gain_ci95_f": ci,
            "oof_training_rows": len(train),
        })

        artifacts[horizon] = {
            "models": models,
            "features": list(FEATURES),
            "selected": chosen,
            "v7_model_reference":
                f"{horizon}m_full_dedicated",
        }

    comparison = pd.DataFrame(
        metrics
    )

    comparison.to_csv(
        output / "comparison.csv",
        index=False,
    )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_csv(
        output / "predictions.csv",
        index=False,
    )

    (
        output / "selection.json"
    ).write_text(
        json.dumps(
            decisions, indent=2
        ),
        encoding="utf-8",
    )

    joblib.dump(
        artifacts,
        output / "research_models.joblib",
    )

    print("\n=== V8 GFS COMPARISON ===")

    print(
        comparison.to_string(index=False)
    )

    print(
        "\nOutput:", output.resolve()
    )

    print(
        "October is exploratory. "
        "V7 and production trading models "
        "remain unchanged."
    )


if __name__ == "__main__":
    main()
