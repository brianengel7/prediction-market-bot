#!/usr/bin/env python3
"""V5 ablation: V3 features vs +sunlight vs +METAR cloud conditions.

Train: July-Aug 2026; select leaf count on September; report October as
EXPLORATORY (October was repeatedly inspected in earlier research).
Uses only reports deemed available at each decision, not future :51 readings.
Writes exclusively to new v5_precise_clouds output paths; no network/trades.

Run from the prediction-market-bot repository root:
python3 src/temp-prediction/train_knyc_v5_precise_clouds.py --knyc-lag 5 --airport-lag 6
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("data/research/nyc_observation_20260701_20261026")
STATIONS = ("NYC", "LGA", "EWR", "TEB")
HORIZONS = (30, 45)
COVER = {"CLR": 0.0, "SKC": 0.0, "FEW": 0.125,
         "SCT": 0.375, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}


def get_v3():
    path = Path(__file__).with_name("train_knyc_v3_30_45.py")
    if not path.exists():
        raise FileNotFoundError(f"Required existing V3 source not found: {path}")
    spec = importlib.util.spec_from_file_location("v3_readonly", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solar_features(decisions):
    """Approximate NOAA solar elevation for KNYC at *decision* timestamps."""
    t = pd.DatetimeIndex(decisions).tz_convert("UTC")
    day = t.dayofyear.to_numpy(dtype=float)
    fractional_hour = (t.hour + t.minute / 60 + t.second / 3600).to_numpy(dtype=float)
    gamma = 2 * np.pi / 365 * (day - 1 + (fractional_hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
                       - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2 * gamma)
                       - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    lat, lon = np.deg2rad(40.779), -73.969  # KNYC / Central Park
    solar_minutes = (fractional_hour * 60 + eqtime + 4 * lon) % 1440
    hour_angle = np.deg2rad(solar_minutes / 4 - 180)
    sin_elevation = (np.sin(lat) * np.sin(decl)
                     + np.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    elevation = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
    return pd.DataFrame({
        "sun_elevation_deg": elevation,
        "sun_heating_proxy": np.maximum(0, sin_elevation),
    })


def cloud_features(grid, clouds, station, lag):
    obs = clouds[clouds["station_name"] == station].copy()
    if obs.empty:
        raise ValueError(f"No cloud reports for station {station}")
    obs["observed"] = pd.to_datetime(obs["observed_utc"], utc=True, errors="coerce")
    obs = obs.dropna(subset=["observed"]).sort_values("observed")
    cover_cols = ["skyc1", "skyc2", "skyc3", "skyc4"]
    obs["cloud_fraction"] = obs[cover_cols].apply(
        lambda col: col.astype(str).str.upper().str.strip().map(COVER)
    ).max(axis=1, skipna=True)
    # A lower broken/overcast/vertical-visibility layer is a ceiling.
    ceiling_candidates = []
    for i in range(1, 5):
        kind = obs[f"skyc{i}"].astype(str).str.upper().str.strip()
        height = pd.to_numeric(obs[f"skyl{i}"], errors="coerce")
        ceiling_candidates.append(height.where(kind.isin(["BKN", "OVC", "VV"])))
    obs["ceiling_ft"] = pd.concat(ceiling_candidates, axis=1).min(axis=1)
    # Log-transform limits outlier influence; missing remains NaN.
    obs["log_ceiling_ft"] = np.log1p(obs["ceiling_ft"].clip(lower=0))
    # Skip duplicate timestamps whose cloud reports disagree, rather than
    # choosing a revision that could have arrived after the decision.
    conflicts = obs.groupby("observed")[["cloud_fraction", "log_ceiling_ft"]].nunique(dropna=True)
    ambiguous = conflicts.index[(conflicts > 1).any(axis=1)]
    obs = obs[~obs["observed"].isin(ambiguous)].drop_duplicates("observed")
    obs["available"] = obs["observed"] + pd.Timedelta(minutes=lag)
    right = obs[["available", "observed", "cloud_fraction", "log_ceiling_ft"]]
    left = grid[["decision"]].sort_values("decision")
    joined = pd.merge_asof(left, right.sort_values("available"),
                           left_on="decision", right_on="available",
                           direction="backward", tolerance=pd.Timedelta(hours=4))
    joined.index = left.index
    joined = joined.reindex(grid.index)
    age = (joined["decision"] - joined["observed"]).dt.total_seconds() / 60
    fresh = age.between(0, 120)
    return pd.DataFrame({
        f"C_{station}_fraction": joined["cloud_fraction"].where(fresh).to_numpy(),
        f"C_{station}_log_ceiling_ft": joined["log_ceiling_ft"].where(fresh).to_numpy(),
        f"C_{station}_age_min": age.where(fresh).to_numpy(),
    }, index=grid.index)


def score(actual, predicted):
    error = np.abs(np.asarray(actual, float) - np.asarray(predicted, float))
    return {"n": len(error), "mae_f": round(float(error.mean()), 4),
            "within_half_pct": round(float(np.mean(error <= 0.5) * 100), 1),
            "within_one_pct": round(float(np.mean(error <= 1.0) * 100), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knyc-lag", type=int, default=5)
    ap.add_argument("--airport-lag", type=int, default=6)
    args = ap.parse_args()
    if min(args.knyc_lag, args.airport_lag) < 0:
        ap.error("Report lags must be nonnegative")

    v3 = get_v3()
    tag = f"knyc{args.knyc_lag}_airports{args.airport_lag}"
    saved_v3 = ROOT / "v3_30_45" / tag / "research_models.joblib"
    if not saved_v3.exists():
        raise FileNotFoundError(f"Need September-selected V3 settings: {saved_v3}")
    selected = joblib.load(saved_v3)["selected_by_horizon"]
    print("V3 feature recipes selected on September:", selected, flush=True)

    reports = {s: v3.load_station(s, args.knyc_lag if s == "NYC" else args.airport_lag)
               for s in STATIONS}
    grid = v3.make_grid(reports["NYC"])
    for station in STATIONS:
        grid = pd.concat([grid, v3.join_station(grid, reports[station], station)], axis=1)
    grid = grid[grid.NYC_temp.notna() & grid[[f"{s}_temp" for s in STATIONS[1:]]].notna().any(axis=1)].copy()
    grid = grid.reset_index(drop=True)
    for s in STATIONS[1:]:
        grid[f"{s}_minus_nyc"] = grid[f"{s}_temp"] - grid.NYC_temp

    labels = pd.read_csv(ROOT / "v5_inputs" / "knyc_precise_targets.csv")
    labels["observed_utc"] = pd.to_datetime(labels["observed_utc"], utc=True, errors="coerce")
    labels["precise_f"] = pd.to_numeric(labels["precise_f"], errors="coerce")
    labels = labels.dropna(subset=["observed_utc", "precise_f"])
    if labels["observed_utc"].duplicated().any():
        raise ValueError("Ambiguous precise targets in v5_inputs")
    grid = grid.merge(labels[["observed_utc", "precise_f"]],
                      left_on="target_time", right_on="observed_utc",
                      how="inner", validate="many_to_one")
    grid["target"] = grid["precise_f"]
    grid = grid.drop(columns=["observed_utc", "precise_f"])
    print("Matched prediction rows:", len(grid),
          "| per horizon:", grid.horizon.value_counts().sort_index().to_dict(), flush=True)

    sun = solar_features(grid["decision"])
    grid = pd.concat([grid.reset_index(drop=True), sun.reset_index(drop=True)], axis=1)
    clouds = pd.read_csv(ROOT / "v5_inputs" / "hourly_metar_clouds.csv", low_memory=False)
    for station in STATIONS:
        lag = args.knyc_lag if station == "NYC" else args.airport_lag
        grid = pd.concat([grid, cloud_features(grid, clouds, station, lag)], axis=1)

    base = ["horizon", "hour_sin", "hour_cos", "day_sin", "day_cos"]
    for s in STATIONS:
        base += [f"{s}_{x}" for x in ["temp", "dew", "wind", "direction",
                                       "prev_temp", "trend1", "age_min"]]
    base += [f"{s}_minus_nyc" for s in STATIONS[1:]]
    groups = {
        "humidity": [f"{s}_{x}" for s in STATIONS for x in ["relative_humidity", "dew_depression"]],
        "wind_vectors": [f"{s}_{x}" for s in STATIONS for x in ["wind_u", "wind_v", "wind_change", "wind_u_change", "wind_v_change"]],
        "two_hour_trends": [f"{s}_{x}" for s in STATIONS for x in ["trend2", "acceleration"]],
    }
    recipes = {"V2_like": [], "humidity": groups["humidity"],
               "wind_vectors": groups["wind_vectors"],
               "two_hour_trends": groups["two_hour_trends"],
               "V3_combined": sum(groups.values(), [])}
    solar_cols = ["sun_elevation_deg", "sun_heating_proxy"]
    cloud_cols = [f"C_{s}_{suffix}" for s in STATIONS
                  for suffix in ["fraction", "log_ceiling_ft", "age_min"]]
    additions = {"v3_precise": [], "plus_solar": solar_cols,
                 "plus_clouds": cloud_cols,
                 "plus_solar_clouds": solar_cols + cloud_cols}

    train = grid[grid.target_day < "2026-09-01"].copy()
    valid = grid[grid.target_day.between("2026-09-01", "2026-09-30")].copy()
    test = grid[grid.target_day >= "2026-10-01"].copy()
    print("Train/validation/exploratory test rows:", len(train), len(valid), len(test), flush=True)

    output = ROOT / "v5_precise_clouds" / tag
    output.mkdir(parents=True, exist_ok=True)
    results, predictions, models = [], [], {}

    for horizon in HORIZONS:
        recipe = selected.get(horizon, selected.get(str(horizon)))
        if recipe not in recipes:
            raise ValueError(f"No recognized V3 feature recipe for {horizon}m: {recipe}")
        focus_valid = valid[valid.horizon.eq(horizon)]
        focus_test = test[test.horizon.eq(horizon)]
        print(f"\n{horizon}m: frozen V3 recipe {recipe}; "
              f"validation N={len(focus_valid)}; October N={len(focus_test)}", flush=True)
        for name, extra in additions.items():
            feature_candidates = base + recipes[recipe] + extra
            # Remove empty or invariant columns based only on July-Aug training.
            feature_cols = [col for col in feature_candidates
                            if train[col].notna().sum() >= 2
                            and train[col].nunique(dropna=True) > 1]
            if not feature_cols:
                raise RuntimeError("No usable model features")
            best = None
            for leaves in (7, 15, 31):
                model = HistGradientBoostingRegressor(
                    max_iter=180, learning_rate=0.05,
                    max_leaf_nodes=leaves, min_samples_leaf=30,
                    l2_regularization=2.0, random_state=42,
                )
                model.fit(train[feature_cols], train["target"] - train["NYC_temp"])
                pred_valid = focus_valid["NYC_temp"].to_numpy() + model.predict(focus_valid[feature_cols])
                val_mae = float(np.abs(focus_valid["target"].to_numpy() - pred_valid).mean())
                if best is None or val_mae < best[0]:
                    best = (val_mae, leaves, model)
            val_mae, leaves, model = best
            pred = focus_test["NYC_temp"].to_numpy() + model.predict(focus_test[feature_cols])
            results.append({"horizon": horizon, "variant": name, "v3_recipe": recipe,
                            "train_n": len(train), "val_n": len(focus_valid),
                            "sep_mae": round(val_mae, 4), "leaves": leaves,
                            **score(focus_test["target"], pred)})
            predictions.append(pd.DataFrame({"target_time": focus_test["target_time"],
                                             "decision": focus_test["decision"],
                                             "horizon": horizon, "variant": name,
                                             "actual_precise_f": focus_test["target"],
                                             "prediction_f": pred}))
            models[f"{horizon}m_{name}"] = {"model": model, "features": feature_cols,
                                           "v3_recipe": recipe, "sep_mae": val_mae,
                                           "leaves": leaves}
            print(f"  {name}: Sep MAE={val_mae:.4f}F; "
                  f"Oct MAE={np.abs(focus_test['target'].to_numpy()-pred).mean():.4f}F; "
                  f"features={len(feature_cols)}", flush=True)

    comparison = pd.DataFrame(results)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    joblib.dump({"models": models, "selected_v3_recipes": selected,
                 "knyc_lag": args.knyc_lag, "airport_lag": args.airport_lag},
                output / "research_models.joblib")
    print("\n=== V5 PRECISE TARGET + SUN / CLOUD ABLATION ===")
    print(comparison.to_string(index=False))
    print("\nWrote:", output.resolve())
    print("Caution: October results are exploratory; publication delays are simulated.")


if __name__ == "__main__":
    main()
