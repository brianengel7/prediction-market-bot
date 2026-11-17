
#!/usr/bin/env python3
"""One-command Miami hourly Weather Index temperature forecast.

Research only. No orders or trading.
Uses existing train_miami_synoptic_hourly.py features.
"""

import argparse
import json
import math
import os
import runpy
import sys
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/temp-prediction/miami_hourly_index"

TRAINER = ROOT / "src/temp-prediction/train_miami_synoptic_hourly.py"
READER = ROOT / "src/temp-prediction/synoptic_city_temp.py"
CACHE = DATA / "miami_live_forecast_models.joblib"

UTC = timezone.utc
ET = ZoneInfo("America/New_York")

STATIONS = ("KMIA", "KOPF", "KFLL", "KFXE", "KPMP")
HORIZONS = (20, 15, 10, 5)

INDEX_DELAY = 6
SYN_DELAY = 15


def dt(s):
    if isinstance(s, (int, float)):
        factor = 1000 if s > 1e11 else 1
        return datetime.fromtimestamp(s / factor, UTC)

    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))

    if d.tzinfo is None:
        raise ValueError("Timestamp without timezone")

    return d.astimezone(UTC)


def local(t):
    return t.astimezone(ET).strftime(
        "%Y-%m-%d %I:%M:%S %p %Z"
    )


def load_modules():
    if not TRAINER.is_file() or not READER.is_file():
        raise RuntimeError(
            "Missing existing trainer or Synoptic reader"
        )

    trainer = runpy.run_path(str(TRAINER))
    reader = runpy.run_path(str(READER))

    required_t = (
        "features",
        "make_series",
        "load_csv",
        "collect_targets",
    )

    required_r = (
        "read_token",
        "fetch_synoptic",
        "extract_observations",
    )

    for name in required_t:
        if not callable(trainer.get(name)):
            raise RuntimeError(
                "Trainer is missing function: " + name
            )

    for name in required_r:
        if not callable(reader.get(name)):
            raise RuntimeError(
                "Synoptic reader is missing function: " + name
            )

    return trainer, reader


def fingerprint():
    paths = [
        TRAINER,
        READER,
        DATA / "miami_index_minutes.csv",
        DATA / "miami_hourly_targets.csv",
        DATA / "miami_iem_observations.csv",
        DATA / "miami_iem_average_5m.csv",
    ]

    return tuple(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns)
        for p in paths
    )


def history(trainer):
    index = trainer["make_series"](
        trainer["load_csv"](
            DATA / "miami_index_minutes.csv"
        )
    )

    all_stations = trainer["load_csv"](
        DATA / "miami_iem_observations.csv"
    )

    syn = {
        sid: trainer["make_series"]([
            r for r in all_stations
            if r["stid"] == sid
        ])
        for sid in STATIONS
    }

    avg_rows = trainer["load_csv"](
        DATA / "miami_iem_average_5m.csv"
    )

    avg = trainer["make_series"]([
        {
            "obs_utc": r["minute_utc"],
            "temp_f": r["average_f"],
        }
        for r in avg_rows
        if r.get("average_f", "").strip()
        and r.get("stations_used") == "5"
    ])

    targets = trainer["collect_targets"]()

    if len(targets) < 150:
        raise RuntimeError(
            "Insufficient hourly targets for training"
        )

    return index, syn, avg, targets


def boost():
    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=150,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=10.0,
        random_state=42,
    )


def trained_models(trainer, refresh=False):
    signature = (
        fingerprint(),
        INDEX_DELAY,
        SYN_DELAY,
        "feature-v1",
    )

    if not refresh and CACHE.exists():
        try:
            # Only load this locally generated model file.
            saved = joblib.load(CACHE)

            if saved.get("signature") == signature:
                return (
                    saved["models"],
                    saved["n"],
                    "cache",
                )

        except (OSError, ValueError, KeyError, EOFError):
            pass

    index, syn, avg, targets = history(trainer)

    models = {}
    counts = {}

    # 15m: station + averaged trends
    # 20m: individual station features
    for horizon, col in ((15, 3), (20, 2)):
        x = []
        y = []

        for hour, target in targets:
            f = trainer["features"](
                hour,
                horizon,
                index,
                syn,
                avg,
                INDEX_DELAY,
                SYN_DELAY,
            )

            if f is None:
                continue

            base = f[0]
            vector = f[col]

            x.append(vector)
            y.append(target - base)

        if len(x) < 150:
            raise RuntimeError(
                f"Insufficient training data for "
                f"{horizon}m: {len(x)}"
            )

        model = boost().fit(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )

        models[horizon] = model
        counts[horizon] = len(x)

    saved = {
        "signature": signature,
        "models": models,
        "n": counts,
    }

    CACHE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = CACHE.with_suffix(".joblib.tmp")
    joblib.dump(saved, temporary)
    os.replace(temporary, CACHE)

    return models, counts, "retrained"


def kalshi_live():
    url = (
        "https://external-api.kalshi.com/"
        "trade-api/v2/live_data/weather/miami?"
        + urlencode({
            "last_sec": 10800,
            "limit": 2000,
        })
    )

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "miami-hourly-research/1.0",
        },
    )

    try:
        with urlopen(request, timeout=25) as res:
            payload = json.load(res)

    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Kalshi index request failed: "
            + type(exc).__name__
        ) from exc

    records = []

    for point in payload.get("timeseries", []):
        if not isinstance(point, dict):
            continue

        if str(point.get("status", "")).lower() in (
            "incomplete",
            "pending",
            "missing",
            "provisional",
        ):
            continue

        try:
            observed = dt(point["t"])
            temp = float(point["v"])

        except (
            KeyError,
            ValueError,
            TypeError,
            OverflowError,
        ):
            continue

        if (
            math.isfinite(temp)
            and observed <= datetime.now(UTC)
        ):
            records.append({
                "obs_utc": observed.isoformat(),
                "temp_f": temp,
            })

    if not records:
        raise RuntimeError(
            "Kalshi returned no completed recent "
            "Weather Index records"
        )

    return records


def synoptic_live(reader):
    token = reader["read_token"]()

    if not token:
        raise RuntimeError(
            "SYNOPTIC_API_TOKEN is missing"
        )

    response = reader["fetch_synoptic"](
        token,
        "miami",
        180,
    )

    results = []
    counts = {}

    for station in response.get("STATION", []):
        sid = str(
            station.get("STID", "")
        ).upper()

        if sid not in STATIONS:
            continue

        if str(station.get("MNET_ID")) != "1":
            continue

        readings = reader["extract_observations"](
            station
        )

        counts[sid] = len(readings)

        for observed, temp_c in readings:
            results.append({
                "stid": sid,
                "obs_utc": observed.isoformat(),
                "temp_f": temp_c * 1.8 + 32,
            })

    return results, counts


def live_avg(syn, trainer, now):
    """Reconstruct IEM-style five-minute averages."""

    newest = now.replace(
        second=0,
        microsecond=0,
    )

    newest -= timedelta(
        minutes=newest.minute % 5
    )

    first = newest - timedelta(minutes=150)

    averaged = []
    t = first

    while t <= newest:
        values = []

        for sid in STATIONS:
            times, vals = syn[sid]
            i = bisect_right(times, t) - 1

            if (
                i < 0
                or t - times[i] > timedelta(minutes=15)
            ):
                break

            values.append(vals[i])

        if len(values) == 5:
            averaged.append({
                "obs_utc": t.isoformat(),
                "temp_f": sum(values) / 5,
            })

        t += timedelta(minutes=5)

    return trainer["make_series"](averaged)


def choose_hour(now):
    hour = now.replace(
        minute=0,
        second=0,
        microsecond=0,
    ) + timedelta(hours=1)

    remaining = (
        hour - now
    ).total_seconds() / 60

    return hour, remaining


def forecast(now, models, trainer, reader):
    hour, remaining = choose_hour(now)

    live_index = trainer["make_series"](
        kalshi_live()
    )

    last_index = live_index[1][-1]
    last_index_time = live_index[0][-1]

    if (
        now - last_index_time
    ).total_seconds() > 20 * 60:
        raise RuntimeError(
            "Kalshi index is over 20 minutes old; "
            "not forecasting"
        )

    print()
    print(
        f"MIAMI | Next Weather Index settlement: "
        f"{local(hour)}"
    )

    print(
        f"Current time: {local(now)} | "
        f"{remaining:.1f} minutes remaining"
    )

    print(
        f"Latest Kalshi index: "
        f"{last_index:.2f} F "
        f"at {local(last_index_time)}"
    )

    if remaining > 20.5:
        print(
            "EARLY INDICATION "
            "(not a trained forecast): "
            f"{last_index:.2f} F"
        )

        print(
            "20-minute model becomes applicable at "
            f"{local(hour - timedelta(minutes=20))}"
        )
        return

    try:
        raw, station_counts = synoptic_live(reader)

    except RuntimeError as exc:
        raw, station_counts = [], {}
        print(f"Synoptic warning: {exc}")

    syn = {
        sid: trainer["make_series"]([
            r for r in raw
            if r["stid"] == sid
        ])
        for sid in STATIONS
    }

    avg = live_avg(
        syn,
        trainer,
        now,
    )

    latest_temps = []

    for sid in STATIONS:
        times, vals = syn[sid]

        if not times:
            print(f"  {sid}: MISSING")
            continue

        age = (
            now - times[-1]
        ).total_seconds() / 60

        print(
            f"  {sid}: {vals[-1]:.2f} F "
            f"| age {age:.1f} min"
        )

        if 0 <= age <= 25:
            latest_temps.append(vals[-1])

    if len(latest_temps) == 5:
        print(
            "Latest independent Synoptic average: "
            f"{sum(latest_temps) / 5:.2f} F"
        )
    else:
        print(
            "Synoptic average unavailable: "
            f"only {len(latest_temps)}/5 recent stations"
        )

    # Choose the nearest trained forecast horizon.
    horizon = min(
        HORIZONS,
        key=lambda h: abs(h - remaining),
    )

    # Existing training features calculate:
    # decision = hour - horizon.
    # Set the decision to NOW without future observations.
    pseudo_hour = now + timedelta(
        minutes=horizon
    )

    f = trainer["features"](
        pseudo_hour,
        horizon,
        live_index,
        syn,
        avg,
        INDEX_DELAY,
        SYN_DELAY,
    )

    if f is None:
        if horizon in (5, 10):
            # Persistence doesn't require station data.
            cutoff = now - timedelta(
                minutes=INDEX_DELAY
            )

            times, vals = live_index
            i = bisect_right(times, cutoff) - 1

            if (
                i >= 0
                and cutoff - times[i]
                <= timedelta(minutes=20)
            ):
                print(
                    f"Model: {horizon}-minute "
                    "persistence (stations unavailable)"
                )
                print(
                    "PREDICTED NEXT SETTLEMENT: "
                    f"{vals[i]:.2f} F"
                )
                return

        print(
            "Cannot assemble complete "
            "timestamp-valid live features."
        )
        print(
            "No ML prediction produced. "
            "Check airport freshness and Synoptic response."
        )
        return

    (
        base,
        small,
        station_features,
        avg_features,
        count,
        decision,
        asof,
    ) = f

    # Restore real target-hour time-of-day features.
    phase = (
        2 * math.pi
        * hour.astimezone(ET).hour / 24
    )

    for v in (
        small,
        station_features,
        avg_features,
    ):
        v[1] = math.sin(phase)
        v[2] = math.cos(phase)

    if horizon in (5, 10):
        pred = base
        method = "persistence"

    else:
        vector = (
            avg_features
            if horizon == 15
            else station_features
        )

        correction = float(
            models[horizon].predict(
                np.asarray(
                    [vector],
                    dtype=float,
                )
            )[0]
        )

        pred = base + correction
        method = (
            "gradient boosting + "
            "IEM-trained station features"
        )

    print(
        f"Synoptic latest-history stations: "
        f"{len(station_counts)}/5"
    )

    print(
        "Weather features usable at assumed cutoff: "
        f"{count}/5"
    )

    print(
        "Index cutoff: "
        f"{local(now - timedelta(minutes=INDEX_DELAY))}"
    )

    print(
        f"Airport cutoff: {local(asof)}"
    )

    print(
        f"Model: {horizon}-minute {method}"
    )

    if abs(remaining - horizon) > 0.75:
        print(
            "CAUTION: Actual time remaining "
            f"({remaining:.1f}m) differs from "
            f"validated {horizon}m horizon"
        )

    print()
    print(
        "PREDICTED NEXT SETTLEMENT: "
        f"{pred:.2f} F"
    )

    print(
        "Change vs as-of index: "
        f"{pred - base:+.2f} F "
        f"(reference {base:.2f} F)"
    )

    print(
        "No trades placed. Data availability delays "
        "are assumptions, not measured receipt times."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--retrain",
        action="store_true",
        help="Ignore cached ML models",
    )

    ap.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Print forecasts at :40, :45, "
            ":50, :55 each hour"
        ),
    )

    args = ap.parse_args()

    trainer, reader = load_modules()

    models, counts, state = trained_models(
        trainer,
        args.retrain,
    )

    print(
        f"Miami models: {state}; "
        f"training samples: "
        f"15m={counts[15]}, 20m={counts[20]}"
    )

    if not args.watch:
        forecast(
            datetime.now(UTC),
            models,
            trainer,
            reader,
        )
        return

    print(
        "Watching for hourly :40/:45/:50/:55 "
        "ET gates. Ctrl+C to stop.",
        flush=True,
    )

    last_gate = None

    while True:
        now = datetime.now(UTC)
        local_now = now.astimezone(ET)

        if local_now.minute in (40, 45, 50, 55):
            gate = (
                local_now.strftime("%Y%m%d%H"),
                local_now.minute,
            )

            if gate != last_gate:
                try:
                    forecast(
                        now,
                        models,
                        trainer,
                        reader,
                    )

                except RuntimeError as exc:
                    print(
                        "Forecast unavailable:",
                        exc,
                        flush=True,
                    )

                last_gate = gate

        time.sleep(10)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nStopped")

    except (
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        sys.exit("ERROR: " + str(exc))
