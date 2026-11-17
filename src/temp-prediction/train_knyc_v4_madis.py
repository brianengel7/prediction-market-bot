
#!/usr/bin/env python3
"""V4: KNYC hourly-only vs hourly + historical MADIS.

Train: July-August 2026
Validation: September 2026
Exploratory test: October 2026

No trading, no network calls, no production changes.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

AIRPORTS = ("LGA", "TEB", "EWR", "JRB", "JFK")
HORIZONS = (30, 45)


def clean_observations(raw, station):
    df = raw.copy()

    df["observed"] = pd.to_datetime(
        df["valid"], utc=True, errors="coerce"
    )
    df["temp"] = pd.to_numeric(
        df["tmpf"], errors="coerce"
    )

    for col, source in [
        ("dew", "dwpf"),
        ("wind", "sknt"),
        ("direction", "drct"),
    ]:
        df[col] = (
            pd.to_numeric(df[source], errors="coerce")
            if source in df else np.nan
        )

    if "station" in df:
        df = df[
            df["station"].astype(str).str.upper()
            .isin([station, "K" + station])
        ]

    df = df.dropna(
        subset=["observed", "temp"]
    ).copy()

    conflict = df.groupby("observed")["temp"].nunique()
    bad = conflict[conflict > 1].index

    return (
        df[~df["observed"].isin(bad)]
        .sort_values("observed")
        .drop_duplicates("observed")
        .reset_index(drop=True)
    )


def make_targets(raw):
    df = clean_observations(raw, "NYC")
    df = df[df["observed"].dt.minute == 51]

    df = df.rename(columns={
        "observed": "target_time",
        "temp": "actual",
    })

    parts = []

    for horizon in HORIZONS:
        part = df[["target_time", "actual"]].copy()
        part["horizon"] = horizon
        part["decision"] = (
            part["target_time"]
            - pd.Timedelta(minutes=horizon)
        )
        parts.append(part)

    grid = pd.concat(parts, ignore_index=True)

    local = grid["decision"].dt.tz_convert(
        "America/New_York"
    )
    local_target = grid["target_time"].dt.tz_convert(
        "America/New_York"
    )

    grid = grid[
        local.dt.hour.between(8, 16)
        & local_target.dt.hour.between(8, 16)
    ].copy()

    local = grid["decision"].dt.tz_convert(
        "America/New_York"
    )

    hour = local.dt.hour + local.dt.minute / 60
    day = local.dt.dayofyear

    grid["hour_sin"] = np.sin(2*np.pi*hour/24)
    grid["hour_cos"] = np.cos(2*np.pi*hour/24)
    grid["day_sin"] = np.sin(2*np.pi*day/365.25)
    grid["day_cos"] = np.cos(2*np.pi*day/365.25)

    grid["target_day"] = (
        grid["target_time"]
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )

    return (
        grid.sort_values("decision")
        .reset_index(drop=True)
    )


def hourly_features(grid, station, raw, lag):
    obs = clean_observations(raw, station)

    obs["prev_temp"] = obs["temp"].shift(1)

    elapsed = (
        obs["observed"].diff().dt.total_seconds()
        / 3600
    )

    obs["trend"] = (
        (obs["temp"] - obs["prev_temp"]) / elapsed
    ).where(elapsed.between(0.3, 2.5))

    obs["available"] = (
        obs["observed"]
        + pd.Timedelta(minutes=lag)
    )

    merged = pd.merge_asof(
        grid[["decision"]],
        obs.sort_values("available")[
            [
                "available", "observed", "temp",
                "dew", "wind", "direction", "trend",
            ]
        ],
        left_on="decision",
        right_on="available",
        direction="backward",
    )

    age = (
        merged["decision"] - merged["observed"]
    ).dt.total_seconds() / 60

    valid = age.between(0, 120)

    result = pd.DataFrame(index=grid.index)

    for column in [
        "temp", "dew", "wind", "direction", "trend"
    ]:
        result[f"{station}_{column}"] = (
            merged[column].where(valid).to_numpy()
        )

    result[f"{station}_age"] = (
        age.where(valid).to_numpy()
    )

    result[f"{station}_observed"] = (
        merged["observed"].where(valid).to_numpy()
    )

    return result


def load_madis(path):
    df = pd.read_csv(path)

    required = {
        "station", "observation_utc", "temperature_f"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing MADIS columns: "
            f"{required - set(df.columns)}"
        )

    df["observed"] = pd.to_datetime(
        df["observation_utc"],
        utc=True,
        errors="coerce",
    )

    df["temp"] = pd.to_numeric(
        df["temperature_f"], errors="coerce"
    )

    df = df[
        df["station"].isin(
            ["K" + s for s in AIRPORTS]
        )
    ].dropna(
        subset=["observed", "temp"]
    ).copy()

    df = df[df["temp"].between(-80, 140)]

    conflict = (
        df.groupby(["station", "observed"])["temp"]
        .agg(lambda x: x.max() - x.min())
    )
    bad = conflict[conflict > 0.2].index

    if len(bad):
        key = pd.MultiIndex.from_frame(
            df[["station", "observed"]]
        )
        df = df[~key.isin(bad)]

    return (
        df.sort_values("observed")
        .drop_duplicates(["station", "observed"])
        .reset_index(drop=True)
    )


def madis_features(grid, station, madis, lag):
    df = madis[
        madis["station"] == "K" + station
    ][["observed", "temp"]].copy()

    out = pd.DataFrame(index=grid.index)

    names = [
        "temp", "age",
        "trend10", "trend20", "trend30",
        "relative_hourly", "newer_hourly",
    ]

    if df.empty:
        for name in names:
            out[f"M_{station}_{name}"] = np.nan
        return out

    df["available"] = (
        df["observed"]
        + pd.Timedelta(minutes=lag)
    )
    df = df.sort_values("available")

    def as_of(decisions):
        left = pd.DataFrame({"decision": decisions})

        return pd.merge_asof(
            left,
            df[["available", "observed", "temp"]],
            left_on="decision",
            right_on="available",
            direction="backward",
        )

    current = as_of(grid["decision"])

    age = (
        grid["decision"] - current["observed"]
    ).dt.total_seconds() / 60

    current_temp = current["temp"].where(
        age.between(0, 90)
    )

    out[f"M_{station}_temp"] = current_temp
    out[f"M_{station}_age"] = age.where(
        current_temp.notna()
    )

    for minutes in [10, 20, 30]:
        past_decision = (
            grid["decision"]
            - pd.Timedelta(minutes=minutes)
        )

        past = as_of(past_decision)

        past_age = (
            past_decision - past["observed"]
        ).dt.total_seconds() / 60

        past_temp = past["temp"].where(
            past_age.between(0, 90)
        )

        out[f"M_{station}_trend{minutes}"] = (
            (current_temp - past_temp)
            / (minutes / 60)
        )

    if f"{station}_temp" in grid:
        out[f"M_{station}_relative_hourly"] = (
            current_temp - grid[f"{station}_temp"]
        )

        out[f"M_{station}_newer_hourly"] = (
            current["observed"]
            .gt(grid[f"{station}_observed"])
            .where(
                current_temp.notna()
                & grid[f"{station}_observed"].notna()
            )
            .astype(float)
        )
    else:
        # Existing hourly archive lacks JRB/JFK.
        out[f"M_{station}_relative_hourly"] = np.nan
        out[f"M_{station}_newer_hourly"] = np.nan

    return out


def metrics(actual, predicted):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    errors = np.abs(y - p)

    return {
        "n": len(errors),
        "mae_f": round(float(errors.mean()), 4),
        "within_half_pct": round(
            float(np.mean(errors <= 0.5) * 100), 1
        ),
        "within_one_pct": round(
            float(np.mean(errors <= 1.0) * 100), 1
        ),
    }


def fit_select(train, valid, features):
    best = None

    for leaves in [7, 15, 31]:
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
            train["actual"] - train["NYC_temp"],
        )

        predicted = (
            valid["NYC_temp"].to_numpy()
            + model.predict(valid[features])
        )

        mae = float(np.mean(
            np.abs(
                valid["actual"].to_numpy()
                - predicted
            )
        ))

        if best is None or mae < best[0]:
            best = (mae, leaves, model)

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag", type=int, default=5
    )
    parser.add_argument(
        "--airport-lag", type=int, default=6
    )
    parser.add_argument(
        "--madis-lags",
        type=int,
        nargs="+",
        default=[15, 25, 30],
    )

    args = parser.parse_args()

    if min(
        args.knyc_lag,
        args.airport_lag,
        *args.madis_lags,
    ) < 0:
        parser.error("Delays cannot be negative")

    output = (
        ROOT / "v4_madis_ml"
        / f"nyc{args.knyc_lag}_airports{args.airport_lag}"
    )
    output.mkdir(parents=True, exist_ok=True)

    knyc = pd.read_csv(
        ROOT / "knyc_high_frequency_observations.csv"
    )

    grid = make_targets(knyc)

    hourly = {"NYC": knyc}

    for station in AIRPORTS:
        path = (
            ROOT
            / "live_feasible_research"
            / "airport_metars"
            / f"{station}_20260701_20261026.csv"
        )
        hourly[station] = (
            pd.read_csv(path)
            if path.exists() else None
        )

    for station, raw in hourly.items():
        if raw is None:
            continue

        lag = (
            args.knyc_lag if station == "NYC"
            else args.airport_lag
        )

        grid = pd.concat(
            [
                grid,
                hourly_features(
                    grid, station, raw, lag
                ),
            ],
            axis=1,
        )

    grid = grid[
        grid["NYC_temp"].notna()
    ].reset_index(drop=True)

    for station in ("LGA", "EWR", "TEB"):
        grid[f"{station}_minus_nyc"] = (
            grid[f"{station}_temp"]
            - grid["NYC_temp"]
        )

    grid = grid[
        grid[
            ["LGA_temp", "EWR_temp", "TEB_temp"]
        ].notna().any(axis=1)
    ].reset_index(drop=True)

    base_features = [
        "hour_sin", "hour_cos",
        "day_sin", "day_cos",
    ]

    for column in grid:
        if (
            column.startswith(
                ("NYC_", "LGA_", "EWR_", "TEB_")
            )
            and not column.endswith("_observed")
            and pd.api.types.is_numeric_dtype(
                grid[column]
            )
        ):
            base_features.append(column)

    base_features = list(
        dict.fromkeys(base_features)
    )

    print(
        "Hourly events:",
        grid.groupby("horizon").size().to_dict(),
    )

    madis = load_madis(
        ROOT / "madis_historical"
        / "observations_20260701_20261026.csv"
    )

    print(
        "MADIS rows:", len(madis),
        "Stations:",
        madis.groupby("station").size().to_dict(),
    )

    results = []
    predictions = []
    models = {}

    for horizon in HORIZONS:
        for lag in [
            None, *dict.fromkeys(args.madis_lags)
        ]:
            data = grid[
                grid["horizon"] == horizon
            ].copy().reset_index(drop=True)

            if lag is not None:
                for station in AIRPORTS:
                    data = pd.concat(
                        [
                            data,
                            madis_features(
                                data, station, madis, lag
                            ),
                        ],
                        axis=1,
                    )

                temperature_cols = [
                    f"M_{s}_temp" for s in AIRPORTS
                ]
                trend_cols = [
                    f"M_{s}_trend20" for s in AIRPORTS
                ]

                data["M_regional_temp"] = (
                    data[temperature_cols].median(
                        axis=1
                    )
                )
                data["M_regional_trend20"] = (
                    data[trend_cols].median(axis=1)
                )
                data["M_stations_present"] = (
                    data[temperature_cols]
                    .notna().sum(axis=1)
                )
                data["M_regional_minus_nyc"] = (
                    data["M_regional_temp"]
                    - data["NYC_temp"]
                )

            features = base_features.copy()

            if lag is not None:
                features += [
                    c for c in data
                    if c.startswith("M_")
                    and pd.api.types.is_numeric_dtype(
                        data[c]
                    )
                ]

            train = data[
                data["target_day"] < "2026-09-01"
            ]
            valid = data[
                data["target_day"].between(
                    "2026-09-01", "2026-09-30"
                )
            ]
            test = data[
                data["target_day"] >= "2026-10-01"
            ].copy()

            if min(
                len(train), len(valid), len(test)
            ) < 30:
                raise RuntimeError(
                    f"Insufficient samples for {horizon}m"
                )

            usable_features = []
            excluded_features = []

            for column in features:
                values = pd.to_numeric(
                    train[column], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

                if (
                    values.notna().sum() >= 2
                    and values.nunique(dropna=True) >= 2
                ):
                    usable_features.append(column)
                else:
                    excluded_features.append(column)

            features = usable_features

            if not features:
                raise RuntimeError(
                    f"No usable features for "
                    f"horizon={horizon}, lag={lag}"
                )

            # Clean any infinite values across all evaluation splits.
            for frame in [train, valid, test]:
                frame.loc[:, features] = (
                    frame[features]
                    .replace([np.inf, -np.inf], np.nan)
                )

            print(
                f"Training {horizon}m, MADIS lag={lag}: "
                f"{len(features)} features, "
                f"{len(excluded_features)} excluded",
                flush=True,
            )

            if excluded_features:
                print(
                    "Excluded:",
                    excluded_features,
                    flush=True,
                )

            val_mae, leaves, model = fit_select(
                train, valid, features
            )

            test["predicted"] = (
                test["NYC_temp"].to_numpy()
                + model.predict(test[features])
            )

            name = (
                "hourly_only"
                if lag is None
                else f"madis_lag_{lag}m"
            )

            results.append({
                "horizon": horizon,
                "variant": name,
                "train_n": len(train),
                "val_n": len(valid),
                "sep_mae": round(val_mae, 4),
                "leaves": leaves,
                **metrics(
                    test["actual"],
                    test["predicted"],
                ),
            })

            predictions.append(
                test[
                    [
                        "target_time", "decision",
                        "target_day", "horizon",
                        "actual", "predicted",
                    ]
                ].assign(variant=name)
            )

            models[f"{horizon}m_{name}"] = {
                "model": model,
                "features": features,
                "validation_mae": val_mae,
                "madis_lag": lag,
            }

    comparison = pd.DataFrame(results)

    selected = {}

    for horizon in HORIZONS:
        group = comparison[
            comparison["horizon"] == horizon
        ]

        selected[horizon] = group.loc[
            group["sep_mae"].idxmin(),
            "variant",
        ]

    comparison.to_csv(
        output / "comparison.csv",
        index=False,
    )

    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        output / "predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "models": models,
            "selected_by_september": selected,
            "knyc_lag": args.knyc_lag,
            "airport_lag": args.airport_lag,
        },
        output / "research_models.joblib",
    )

    print("\n=== V4 PAIRED OCTOBER EVALUATION ===")
    print(comparison.to_string(index=False))
    print("\nSelected by September:", selected)
    print("Saved to:", output.resolve())

    print(
        "\nCAUTION: Historical availability is "
        "simulated; October is exploratory."
    )


if __name__ == "__main__":
    main()
