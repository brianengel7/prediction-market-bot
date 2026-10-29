
#!/usr/bin/env python3
"""KNYC forecasting using historically public METAR-type feeds.

No archived one-minute KNYC inputs.
No Kalshi orders, production DB writes, or repo API access.
Historical publication availability is approximated with a lag.
"""

from pathlib import Path
import io
import time

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor


BASE = Path(
    "data/research/nyc_observation_20260701_20261026"
)
OUT = BASE / "live_feasible_research"
AIRPORT_DIR = OUT / "airport_metars"

HORIZONS = [5, 10, 15, 20, 30, 45, 60]

STATIONS = {
    "LGA": "NY_ASOS",
    "EWR": "NJ_ASOS",
    "TEB": "NJ_ASOS",
}

# Conservative, ASSUMED delay after measurement.
# Historical first-publication times are unavailable.
LAG_MIN = 20
MAX_AGE_MIN = 120


def get_airport(station, network):
    AIRPORT_DIR.mkdir(parents=True, exist_ok=True)

    dest = AIRPORT_DIR / (
        f"{station}_20260701_20261026.csv"
    )

    if dest.exists():
        data = pd.read_csv(dest)
        print(
            f"{station}: cached {len(data):,} reports",
            flush=True,
        )
        return data

    url = (
        "https://mesonet.agron.iastate.edu/"
        "cgi-bin/request/asos.py"
    )

    blocks = []

    boundaries = [
        "2026-07-01",
        "2026-08-01",
        "2026-09-01",
        "2026-10-01",
        "2026-10-27",
    ]

    for start_day, end_day in zip(
        boundaries[:-1], boundaries[1:]
    ):
        start = pd.Timestamp(
            start_day, tz="America/New_York"
        ).tz_convert("UTC")

        end = pd.Timestamp(
            end_day, tz="America/New_York"
        ).tz_convert("UTC")

        for attempt in range(4):
            try:
                response = requests.get(
                    url,
                    params={
                        "station": station,
                        "network": network,
                        "data": [
                            "tmpf", "dwpf",
                            "drct", "sknt",
                        ],
                        "report_type": ["3", "4"],
                        "sts": start.isoformat(),
                        "ets": end.isoformat(),
                        "tz": "UTC",
                        "format": "onlycomma",
                        "missing": "M",
                        "latlon": "no",
                        "elev": "no",
                    },
                    timeout=120,
                )
                response.raise_for_status()

                frame = pd.read_csv(
                    io.StringIO(response.text),
                    comment="#",
                )

                required = {
                    "station", "valid", "tmpf"
                }

                if not required.issubset(frame):
                    raise ValueError(
                        f"Unexpected columns: {frame.columns.tolist()}"
                    )

                blocks.append(frame)

                print(
                    f"{station}: {start_day} -> {end_day}: "
                    f"{len(frame):,} reports",
                    flush=True,
                )
                break

            except (
                requests.RequestException,
                ValueError,
            ) as error:
                if attempt == 3:
                    raise RuntimeError(
                        f"{station} download failed: {error}"
                    ) from error

                time.sleep(3 * (attempt + 1))

        time.sleep(0.5)

    data = pd.concat(blocks, ignore_index=True)

    if data.empty:
        raise RuntimeError(
            f"No airport observations for {station}"
        )

    data.to_csv(dest, index=False)
    return data


def normalize(raw, station):
    df = raw.copy()

    df["observed"] = pd.to_datetime(
        df["valid"],
        utc=True,
        errors="coerce",
    )

    for name, column in [
        ("temp", "tmpf"),
        ("dew", "dwpf"),
        ("wind", "sknt"),
        ("direction", "drct"),
    ]:
        df[name] = (
            pd.to_numeric(df[column], errors="coerce")
            if column in df
            else np.nan
        )

    df = df.dropna(
        subset=["observed", "temp"]
    )

    if "station" in df:
        df = df[
            df["station"].astype(str).str.upper()
            .isin([station, "K" + station])
        ]

    conflicts = df.groupby(
        "observed"
    )["temp"].nunique()

    bad = conflicts[conflicts > 1].index

    df = df[
        ~df["observed"].isin(bad)
    ]

    df = (
        df.sort_values("observed")
        .drop_duplicates("observed")
    )

    # The observation cannot be used until the
    # assumed publication delay has elapsed.
    df["available"] = (
        df["observed"]
        + pd.Timedelta(minutes=LAG_MIN)
    )

    df["prior_temp"] = df["temp"].shift(1)
    df["prior_observed"] = df["observed"].shift(1)

    return df[[
        "available",
        "observed",
        "temp",
        "dew",
        "wind",
        "direction",
        "prior_temp",
        "prior_observed",
    ]].reset_index(drop=True)


def features_at_decision(grid, station, obs):
    right = obs.rename(
        columns={
            c: f"{station}_{c}"
            for c in obs.columns
            if c != "available"
        }
    )

    ordered = grid[
        ["decision"]
    ].sort_values("decision")

    joined = pd.merge_asof(
        ordered,
        right.sort_values("available"),
        left_on="decision",
        right_on="available",
        direction="backward",
        tolerance=pd.Timedelta(
            minutes=MAX_AGE_MIN + LAG_MIN
        ),
    )

    joined.index = ordered.index

    age = (
        joined["decision"]
        - joined[f"{station}_observed"]
    ).dt.total_seconds() / 60

    joined[f"{station}_age_min"] = age

    stale = age > MAX_AGE_MIN

    columns = [
        f"{station}_{name}"
        for name in [
            "temp",
            "dew",
            "wind",
            "direction",
            "prior_temp",
            "prior_observed",
        ]
    ]

    joined.loc[stale, columns] = np.nan
    joined.loc[
        stale, f"{station}_age_min"
    ] = np.nan

    delta = (
        joined[f"{station}_temp"]
        - joined[f"{station}_prior_temp"]
    )

    elapsed = (
        joined[f"{station}_observed"]
        - joined[f"{station}_prior_observed"]
    ).dt.total_seconds() / 3600

    joined[f"{station}_trend_per_hour"] = (
        (delta / elapsed)
        .where(elapsed.between(0.1, 2.5))
    )

    joined = joined.drop(
        columns=[
            f"{station}_prior_observed",
            f"{station}_observed",
            "available",
            "decision",
        ]
    )

    return joined.reindex(grid.index)


def score(actual, predicted):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    rounded = np.rint(p)

    return {
        "N": len(y),
        "MAE": round(
            float(np.abs(y - p).mean()), 3
        ),
        "exact_rounded": round(
            float(np.mean(rounded == y)), 3
        ),
        "within_1F": round(
            float(np.mean(np.abs(rounded - y) <= 1)),
            3,
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # KNYC official observations, NOT 1-minute archive.
    raw = pd.read_csv(
        BASE / "knyc_high_frequency_observations.csv"
    )

    knyc = normalize(raw, "NYC")

    labels = raw.copy()

    labels["target_time"] = pd.to_datetime(
        labels["valid"],
        utc=True,
        errors="coerce",
    )

    labels["target"] = pd.to_numeric(
        labels["tmpf"],
        errors="coerce",
    )

    labels = labels.dropna(
        subset=["target_time", "target"]
    )

    # Routine hourly reports are normally at :51.
    labels = labels[
        labels["target_time"].dt.minute == 51
    ]

    if "station" in labels:
        labels = labels[
            labels["station"].astype(str).str.upper()
            .isin(["NYC", "KNYC"])
        ]

    conflicts = labels.groupby(
        "target_time"
    )["target"].nunique()

    labels = labels[
        ~labels["target_time"].isin(
            conflicts[conflicts > 1].index
        )
    ]

    labels = (
        labels.sort_values("target_time")
        .drop_duplicates("target_time")
    )

    # Every target gets predictions at each horizon.
    grid = pd.concat(
        [
            labels[["target_time", "target"]]
            .assign(horizon=h)
            for h in HORIZONS
        ],
        ignore_index=True,
    )

    grid["decision"] = (
        grid["target_time"]
        - pd.to_timedelta(
            grid["horizon"], unit="min"
        )
    )

    local_decision = grid[
        "decision"
    ].dt.tz_convert("America/New_York")

    local_target = grid[
        "target_time"
    ].dt.tz_convert("America/New_York")

    # Both prediction and target must be inside
    # the 8 AM to 5 PM ET information session.
    eligible = (
        local_decision.dt.hour.between(8, 16)
        & local_target.dt.hour.between(8, 16)
    )

    grid = grid[eligible].copy()

    grid["target_day"] = (
        local_target.loc[grid.index]
        .dt.strftime("%Y-%m-%d")
    )

    grid["hour"] = (
        local_decision.loc[grid.index].dt.hour
    )

    grid["minute"] = (
        local_decision.loc[grid.index].dt.minute
    )

    hour = grid["hour"] + grid["minute"] / 60

    grid["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    grid["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    doy = local_decision.loc[
        grid.index
    ].dt.dayofyear

    grid["day_sin"] = np.sin(
        2 * np.pi * doy / 365.25
    )
    grid["day_cos"] = np.cos(
        2 * np.pi * doy / 365.25
    )

    grid = grid.reset_index(drop=True)

    # Use information only after assumed availability.
    for station, network in [
        ("NYC", None),
        *STATIONS.items(),
    ]:
        obs = (
            knyc
            if station == "NYC"
            else normalize(
                get_airport(station, network),
                station,
            )
        )

        station_features = features_at_decision(
            grid, station, obs
        )

        grid = pd.concat(
            [grid, station_features],
            axis=1,
        )

    airport_columns = [
        f"{s}_temp"
        for s in STATIONS
    ]

    has_airport = grid[
        airport_columns
    ].notna().any(axis=1)

    # Require a prior KNYC report and at least
    # one available airport observation.
    grid = grid[
        grid["NYC_temp"].notna()
        & has_airport
    ].copy()

    if grid.empty:
        raise RuntimeError(
            "No usable samples. Check timestamps and lags."
        )

    for station in STATIONS:
        grid[f"{station}_minus_nyc"] = (
            grid[f"{station}_temp"]
            - grid["NYC_temp"]
        )

    features = [
        "horizon",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    features += [
        c for c in grid.columns
        if c.startswith(
            ("NYC_", "LGA_", "EWR_", "TEB_")
        )
        and pd.api.types.is_numeric_dtype(grid[c])
    ]

    features = list(dict.fromkeys(features))

    train = grid[
        grid["target_day"] < "2026-09-01"
    ]

    valid = grid[
        grid["target_day"].between(
            "2026-09-01", "2026-09-30"
        )
    ]

    test = grid[
        grid["target_day"] >= "2026-10-01"
    ].copy()

    if min(
        len(train), len(valid), len(test)
    ) < 50:
        raise RuntimeError(
            "Insufficient samples: "
            f"{len(train)}, {len(valid)}, {len(test)}"
        )

    # Predict the change relative to last
    # available official KNYC temperature.
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
            train["target"] - train["NYC_temp"],
        )

        predictions = (
            valid["NYC_temp"].to_numpy()
            + model.predict(valid[features])
        )

        mae = float(
            np.mean(
                np.abs(
                    predictions
                    - valid["target"].to_numpy()
                )
            )
        )

        if best is None or mae < best[0]:
            best = (mae, leaves, model)

    validation_mae, leaves, model = best

    test["prediction"] = (
        test["NYC_temp"].to_numpy()
        + model.predict(test[features])
    )

    test["previous_nyc_prediction"] = (
        test["NYC_temp"]
    )

    # Simple non-ML baseline based on airport trends.
    airport_trend = (
        test[
            [
                f"{s}_trend_per_hour"
                for s in STATIONS
            ]
        ]
        .mean(axis=1)
        .fillna(0)
        .clip(-6, 6)
    )

    test["airport_trend_prediction"] = (
        test["NYC_temp"]
        + airport_trend
        * (test["NYC_age_min"] / 60)
        .clip(0, 1.5)
    )

    results = []

    for horizon, group in test.groupby(
        "horizon", sort=True
    ):
        for name, column in [
            (
                "prior_KNYC_METAR",
                "previous_nyc_prediction",
            ),
            (
                "airport_trend_proxy",
                "airport_trend_prediction",
            ),
            (
                "boosted_live_feasible",
                "prediction",
            ),
        ]:
            results.append({
                "horizon_min": horizon,
                "model": name,
                **score(
                    group["target"],
                    group[column],
                ),
            })

    comparison = pd.DataFrame(results)

    test.to_csv(
        OUT / "october_predictions_live_feasible.csv",
        index=False,
    )

    comparison.to_csv(
        OUT / "october_horizon_comparison.csv",
        index=False,
    )

    print("\n=== LIVE-FEASIBLE RETEST ===")
    print(
        "Assumed availability lag:",
        LAG_MIN, "minutes",
    )
    print("Stations: KNYC, KLGA, KEWR, KTEB")
    print("Horizons:", HORIZONS)
    print(
        "Train / Validation / October:",
        len(train), len(valid), len(test),
    )
    print("Selected tree leaves:", leaves)
    print(
        "September validation MAE:",
        round(validation_mae, 3),
    )

    print("\n=== OCTOBER PERFORMANCE ===")
    print(comparison.to_string(index=False))

    print("\nSaved to:", OUT.resolve())
    print(
        "\nCAUTION: Historical report arrival times "
        "are unknown. This is a lag-assumed backtest."
    )


if __name__ == "__main__":
    main()
