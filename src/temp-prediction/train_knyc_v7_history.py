
#!/usr/bin/env python3
"""V7: Extended-history KNYC model comparison.

Uses established V3 features and precise KNYC targets.
Compares training windows and dedicated forecast horizons.

Training ends August 2026.
September selects models.
October is exploratory evaluation only.

Never modifies V3/V5 or the live bot.
"""

import argparse
import importlib.util
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

EXT = ROOT / "v7_extended_history"

STATIONS = ("NYC", "LGA", "EWR", "TEB")
HORIZONS = (30, 45)

T_RE = re.compile(
    r"(?<!\S)T([01])(\d{3})([01])(\d{3})(?=\s|$)"
)


def import_v3():
    path = Path(__file__).with_name(
        "train_knyc_v3_30_45.py"
    )

    spec = importlib.util.spec_from_file_location(
        "v3_helper_v7", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_station(path, station, lag):
    """Reuse V3's feature calculations on longer history."""

    r = pd.read_csv(
        path, low_memory=False
    )

    if "station" in r:
        r = r[
            r["station"].astype(str).str.upper()
            .isin([station, "K" + station])
        ].copy()

    r["observed"] = pd.to_datetime(
        r["valid"], utc=True, errors="coerce"
    )

    for source, name in [
        ("tmpf", "temp"),
        ("dwpf", "dew"),
        ("sknt", "wind"),
        ("drct", "direction"),
    ]:
        r[name] = pd.to_numeric(
            r[source], errors="coerce"
        )

    r = r.dropna(
        subset=["observed", "temp"]
    ).sort_values("observed")

    inconsistent = (
        r.groupby("observed")["temp"].nunique()
    )

    bad = inconsistent[
        inconsistent > 1
    ].index

    r = r[
        ~r["observed"].isin(bad)
    ]

    r = (
        r.drop_duplicates("observed")
        .copy()
        .reset_index(drop=True)
    )

    r["prev_temp"] = r["temp"].shift(1)
    r["prev2_temp"] = r["temp"].shift(2)

    r["prev_time"] = r["observed"].shift(1)
    r["prev2_time"] = r["observed"].shift(2)

    r["prev_wind"] = r["wind"].shift(1)
    r["prev_direction"] = (
        r["direction"].shift(1)
    )

    r["available"] = (
        r["observed"]
        + pd.Timedelta(minutes=lag)
    )

    dt1 = (
        r["observed"] - r["prev_time"]
    ).dt.total_seconds() / 3600

    dt2 = (
        r["observed"] - r["prev2_time"]
    ).dt.total_seconds() / 3600

    r["trend1"] = (
        (r["temp"] - r["prev_temp"]) / dt1
    ).where(dt1.between(0.1, 2.5))

    r["trend2"] = (
        (r["temp"] - r["prev2_temp"]) / dt2
    ).where(dt2.between(0.3, 4.5))

    r["acceleration"] = (
        r["trend1"] - r["trend2"]
    )

    r["wind_change"] = (
        r["wind"] - r["prev_wind"]
    )

    theta = np.deg2rad(r["direction"])
    old_theta = np.deg2rad(r["prev_direction"])

    r["wind_u"] = (
        -r["wind"] * np.sin(theta)
    )

    r["wind_v"] = (
        -r["wind"] * np.cos(theta)
    )

    r["wind_u_change"] = (
        r["wind_u"]
        + r["prev_wind"] * np.sin(old_theta)
    )

    r["wind_v_change"] = (
        r["wind_v"]
        + r["prev_wind"] * np.cos(old_theta)
    )

    tc = (r["temp"] - 32) * 5 / 9
    dc = (r["dew"] - 32) * 5 / 9

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        rh = 100 * np.exp(
            17.625 * dc / (243.04 + dc)
            - 17.625 * tc / (243.04 + tc)
        )

    r["relative_humidity"] = rh.clip(0, 100)

    r["dew_depression"] = (
        r["temp"] - r["dew"]
    )

    return r


def precise_f(raw):
    match = (
        T_RE.search(raw)
        if isinstance(raw, str)
        else None
    )

    if not match:
        return np.nan

    sign = (
        -1 if match.group(1) == "1"
        else 1
    )

    return (
        32 + sign * int(match.group(2)) / 10 * 1.8
    )


def build_grid(v3, knyc_lag, airport_lag):
    reports = {}

    for station in STATIONS:
        path = (
            EXT
            / f"{station}_20240101_20261026.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Run history collector first: {path}"
            )

        reports[station] = load_station(
            path,
            station,
            knyc_lag
            if station == "NYC"
            else airport_lag,
        )

    knyc = reports["NYC"]

    targets = knyc[
        knyc["observed"].dt.minute == 51
    ].copy()

    targets["precise_target"] = (
        targets["metar"].map(precise_f)
    )

    print("\nPrecise targets by year:")

    print(
        targets.groupby(
            targets["observed"].dt.year
        )["precise_target"]
        .agg(["size", "count"])
        .to_string(),
        flush=True,
    )

    targets = targets.dropna(
        subset=["precise_target"]
    )

    labels = targets[
        ["observed", "precise_target"]
    ].rename(
        columns={"observed": "target_time"}
    )

    if labels["target_time"].duplicated().any():
        raise ValueError(
            "Conflicting precise KNYC targets"
        )

    grid = v3.make_grid(knyc)

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
            [
                f"{s}_temp"
                for s in STATIONS[1:]
            ]
        ].notna().any(axis=1)
    ].copy()

    for station in STATIONS[1:]:
        grid[f"{station}_minus_nyc"] = (
            grid[f"{station}_temp"]
            - grid["NYC_temp"]
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
            "2024-01-01", "2026-10-26"
        )
    ]

    return grid


def feature_names(recipe):
    base = [
        "horizon",
        "hour_sin", "hour_cos",
        "day_sin", "day_cos",
    ]

    for station in STATIONS:
        base += [
            f"{station}_{name}"
            for name in (
                "temp", "dew", "wind",
                "direction", "prev_temp",
                "trend1", "age_min",
            )
        ]

    base += [
        f"{s}_minus_nyc"
        for s in STATIONS[1:]
    ]

    groups = {
        "humidity": [
            f"{s}_{name}"
            for s in STATIONS
            for name in (
                "relative_humidity",
                "dew_depression",
            )
        ],
        "wind_vectors": [
            f"{s}_{name}"
            for s in STATIONS
            for name in (
                "wind_u", "wind_v",
                "wind_change",
                "wind_u_change",
                "wind_v_change",
            )
        ],
        "two_hour_trends": [
            f"{s}_{name}"
            for s in STATIONS
            for name in (
                "trend2", "acceleration"
            )
        ],
    }

    recipes = {
        "V2_like": [],
        "humidity": groups["humidity"],
        "wind_vectors": groups["wind_vectors"],
        "two_hour_trends": groups[
            "two_hour_trends"
        ],
        "V3_combined": sum(
            groups.values(), []
        ),
    }

    return base + recipes[recipe]


def score(actual, predicted):
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


def fit_window(grid, horizon, features,
               window, leaves):

    train = grid[
        grid["target_day"] < "2026-09-01"
    ]

    if window == "short":
        train = train[
            train["target_day"].between(
                "2026-07-01", "2026-08-31"
            )
        ]

    elif window == "summer":
        train = train[
            train["target_day"]
            .str[5:7].isin(["07", "08"])
        ]

    elif window == "trailing_year":
        train = train[
            train["target_day"] >= "2025-09-01"
        ]

    elif window in (
        "full_pooled",
        "full_dedicated",
    ):
        pass

    else:
        raise ValueError(window)

    if window == "full_dedicated":
        train = train[
            train["horizon"] == horizon
        ]

    model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=leaves,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )

    model.fit(
        train[features],
        train["target"] - train["NYC_temp"],
    )

    return model, len(train)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag", type=int, default=5
    )
    parser.add_argument(
        "--airport-lag", type=int, default=6
    )
    parser.add_argument(
        "--self-test", action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        assert abs(
            precise_f(
                "METAR KNYC 010000Z T01610050"
            ) - 60.98
        ) < 1e-8

        assert np.isnan(
            precise_f("METAR KNYC 010000Z")
        )

        print("SELF-TEST PASS")
        return

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    v3 = import_v3()

    frozen_v3 = joblib.load(
        ROOT / "v3_30_45" / tag
        / "research_models.joblib"
    )

    frozen_v5 = joblib.load(
        ROOT / "v5_precise_clouds" / tag
        / "research_models.joblib"
    )

    saved_v5 = pd.read_csv(
        ROOT / "v5_precise_clouds" / tag
        / "predictions.csv"
    )

    saved_v5["target_time"] = pd.to_datetime(
        saved_v5["target_time"], utc=True
    )

    print("Building extended historical features...")

    grid = build_grid(
        v3,
        args.knyc_lag,
        args.airport_lag,
    )

    print(
        "Rows per horizon:",
        grid.groupby("horizon")
        .size().to_dict(),
    )

    sep = grid[
        grid["target_day"].between(
            "2026-09-01", "2026-09-30"
        )
    ]

    october = grid[
        grid["target_day"].between(
            "2026-10-01", "2026-10-26"
        )
    ]

    windows = [
        "short",
        "summer",
        "trailing_year",
        "full_pooled",
        "full_dedicated",
    ]

    results = []
    predictions = []
    trained = {}

    for horizon in HORIZONS:
        recipe = frozen_v3[
            "selected_by_horizon"
        ][horizon]

        features = feature_names(recipe)

        validation = sep[
            sep["horizon"] == horizon
        ]

        test = october[
            october["horizon"] == horizon
        ]

        print(
            f"\n=== {horizon}m — V3 recipe "
            f"{recipe} ===",
            flush=True,
        )

        # Verify exact parity with previous
        # V3-precise predictions before comparing.
        baseline = frozen_v5["models"][
            f"{horizon}m_v3_precise"
        ]

        old_prediction = (
            test["NYC_temp"].to_numpy()
            + baseline["model"].predict(
                test[baseline["features"]]
            )
        )

        earlier = saved_v5[
            (saved_v5["horizon"] == horizon)
            & (saved_v5["variant"] == "v3_precise")
        ][["target_time", "prediction_f"]]

        parity = pd.DataFrame({
            "target_time": test["target_time"],
            "new": old_prediction,
        }).merge(
            earlier,
            on="target_time",
            validate="one_to_one",
        )

        if len(parity) != len(test):
            raise RuntimeError(
                "Frozen baseline event mismatch"
            )

        difference = (
            parity["new"] - parity["prediction_f"]
        ).abs().max()

        print(
            "Frozen V3 parity maximum difference:",
            f"{difference:.7f}°F",
            flush=True,
        )

        if difference > 0.01:
            raise RuntimeError(
                "V3 feature reconstruction "
                "does not match previous results."
            )

        results.append({
            "horizon": horizon,
            "variant": "FROZEN_v3_precise",
            "sep_mae": np.nan,
            "sep_within_one_pct": np.nan,
            "train_n": np.nan,
            **score(test["target"], old_prediction),
        })

        for name in (
            "plus_clouds",
            "plus_solar_clouds",
        ):
            earlier = saved_v5[
                (saved_v5["horizon"] == horizon)
                & (saved_v5["variant"] == name)
            ]

            match = test[
                ["target_time", "target"]
            ].merge(
                earlier[
                    ["target_time", "prediction_f"]
                ],
                on="target_time",
                validate="one_to_one",
            )

            if len(match) != len(test):
                raise RuntimeError(
                    "Frozen V5 target mismatch"
                )

            results.append({
                "horizon": horizon,
                "variant": "FROZEN_" + name,
                "sep_mae": np.nan,
                "sep_within_one_pct": np.nan,
                "train_n": np.nan,
                **score(
                    match["target"],
                    match["prediction_f"],
                ),
            })

        for window in windows:
            winner = None

            for leaves in (7, 15, 31):
                model, train_n = fit_window(
                    grid,
                    horizon,
                    features,
                    window,
                    leaves,
                )

                val_prediction = (
                    validation["NYC_temp"].to_numpy()
                    + model.predict(
                        validation[features]
                    )
                )

                val_score = score(
                    validation["target"],
                    val_prediction,
                )

                ranking = (
                    -val_score["within_one_pct"],
                    val_score["mae_f"],
                )

                if (
                    winner is None
                    or ranking < winner[0]
                ):
                    winner = (
                        ranking, leaves,
                        model, train_n, val_score,
                    )

            (
                _, leaves, model,
                train_n, val_score
            ) = winner

            prediction = (
                test["NYC_temp"].to_numpy()
                + model.predict(test[features])
            )

            test_score = score(
                test["target"],
                prediction,
            )

            print(
                f"{window:16} "
                f"train={train_n:6} "
                f"Sep ±1°F="
                f"{val_score['within_one_pct']:5.1f}% "
                f"Oct ±1°F="
                f"{test_score['within_one_pct']:5.1f}% "
                f"Oct MAE="
                f"{test_score['mae_f']:.4f}°F",
                flush=True,
            )

            results.append({
                "horizon": horizon,
                "variant": "V7_" + window,
                "train_n": train_n,
                "sep_mae": val_score["mae_f"],
                "sep_within_one_pct":
                    val_score["within_one_pct"],
                "leaves": leaves,
                **test_score,
            })

            predictions.append(
                pd.DataFrame({
                    "target_time": test["target_time"],
                    "horizon": horizon,
                    "variant": "V7_" + window,
                    "actual_precise_f": test["target"],
                    "prediction_f": prediction,
                })
            )

            trained[f"{horizon}m_{window}"] = {
                "model": model,
                "features": features,
                "leaves": leaves,
                "validation": val_score,
                "train_window": window,
                "horizon": horizon,
            }

    out = (
        EXT / "training_results" / tag
    )

    out.mkdir(
        parents=True, exist_ok=True
    )

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        out / "comparison.csv",
        index=False,
    )

    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        out / "predictions.csv",
        index=False,
    )

    selected = {}

    for horizon in HORIZONS:
        options = comparison[
            (comparison["horizon"] == horizon)
            & comparison["variant"]
            .str.startswith("V7_")
        ]

        best = options.sort_values(
            ["sep_within_one_pct", "sep_mae"],
            ascending=[False, True],
        ).iloc[0]

        selected[horizon] = best["variant"]

    joblib.dump(
        {
            "models": trained,
            "selected_by_september": selected,
            "knyc_lag": args.knyc_lag,
            "airport_lag": args.airport_lag,
        },
        out / "research_models.joblib",
    )

    print(
        "\n=== V7 EXPANDED HISTORY COMPARISON ==="
    )

    print(
        comparison.to_string(index=False)
    )

    print(
        "\nSelected by September:",
        selected,
    )

    print("Saved:", out.resolve())

    print(
        "October evaluation is exploratory; "
        "historical reporting delays are simulated."
    )


if __name__ == "__main__":
    main()
