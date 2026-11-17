
#!/usr/bin/env python3
"""Collect untouched KNYC V8 holdout data. Research only."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = Path(
    "data/research/nyc_observation_20260701_20261026"
)
OUT = BASE / "v8_unseen_holdout_20261027_20261108"
GFS_RAW = OUT / "raw_gfs_runs"
IEM_RAW = OUT / "raw_iem"

GFS_API = "https://single-runs-api.open-meteo.com/v1/forecast"
IEM_API = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

UTC = timezone.utc
STATIONS = {
    "NYC": "NY_ASOS",
    "LGA": "NY_ASOS",
    "EWR": "NJ_ASOS",
    "TEB": "NJ_ASOS",
}

SPEC = {
    "holdout_start_local": "2026-10-27",
    "holdout_end_local": "2026-11-08",
    "training_cutoff_local": "2026-10-26",
    "horizons_minutes": [30, 45],
    "primary_candidate": "gfs_delta_only",
    "secondary_candidate": "gfs_delta_plus_bias",
    "baseline": "frozen V7",
    "ridge_alpha": 40.0,
    "correction_clip_f": 1.5,
    "gfs_assumed_publication_lag_hours": 6,
    "primary_metric": "paired MAE versus V7",
    "secondary_metrics": [
        "within_1F",
        "above_2F",
        "day_bootstrap_CI",
    ],
    "rule": "No fitting or selecting on holdout data",
}


def request(url, params):
    """Retry transient connection and server failures."""
    max_attempts = 8

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(60, 120),
            )

            # These status codes may indicate temporary failures.
            if response.status_code in (
                429, 500, 502, 503, 504
            ):
                response.raise_for_status()

            # Preserve original missing-run handling.
            if response.status_code in (400, 404):
                return response

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            print(
                f"HTTP attempt {attempt}/{max_attempts} "
                f"failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

            if attempt == max_attempts:
                raise

            delay = min(5 * attempt, 40)

            print(
                f"Retrying in {delay} seconds...",
                flush=True,
            )
            time.sleep(delay)

def initialize():
    GFS_RAW.mkdir(parents=True, exist_ok=True)
    IEM_RAW.mkdir(parents=True, exist_ok=True)

    path = OUT / "locked_spec.json"

    if path.exists():
        if json.loads(path.read_text()) != SPEC:
            raise RuntimeError(
                "Locked experiment changed; refusing to continue."
            )
    else:
        path.write_text(json.dumps(SPEC, indent=2) + "\n")


def collect_gfs():
    print("\n=== HISTORICAL GFS RUNS ===", flush=True)

    start = datetime(2026, 10, 26, tzinfo=UTC)
    end = datetime(2026, 11, 8, 18, tzinfo=UTC)

    unavailable = []
    downloaded = 0
    current = start

    while current <= end:
        name = current.strftime("%Y%m%dT%HZ") + ".json"
        cache = GFS_RAW / name

        if not cache.exists():
            params = {
                "latitude": 40.77898,
                "longitude": -73.96925,
                "models": "gfs_global",
                "run": current.strftime("%Y-%m-%dT%H:%M"),
                "forecast_days": 2,
                "timezone": "UTC",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "hourly": ",".join([
                    "temperature_2m",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "shortwave_radiation",
                ]),
            }

            response = request(GFS_API, params)

            if response.status_code in (400, 404):
                unavailable.append({
                    "run": current.isoformat(),
                    "message": response.text[:300],
                })
                print("Unavailable:", name, flush=True)
                current += timedelta(hours=6)
                continue

            payload = response.json()

            if "hourly" not in payload:
                raise RuntimeError(
                    f"Invalid GFS response for {name}: {payload}"
                )

            temporary = cache.with_suffix(".part")
            temporary.write_text(json.dumps(payload))
            temporary.replace(cache)

            downloaded += 1

            if downloaded % 8 == 0:
                print(
                    f"New runs downloaded: {downloaded}",
                    flush=True,
                )

            time.sleep(0.5)

        current += timedelta(hours=6)

    records = []

    for file in sorted(GFS_RAW.glob("*.json")):
        payload = json.loads(file.read_text())
        hourly = payload["hourly"]

        run_time = datetime.strptime(
            file.stem, "%Y%m%dT%HZ"
        ).replace(tzinfo=UTC)

        for i, valid in enumerate(hourly["time"]):
            records.append({
                "run_init_utc": run_time.isoformat(),
                "valid_utc": pd.to_datetime(
                    valid, utc=True
                ).isoformat(),
                "gfs_temp_f": hourly["temperature_2m"][i],
                "gfs_rh_pct": hourly["relative_humidity_2m"][i],
                "gfs_cloud_pct": hourly["cloud_cover"][i],
                "gfs_wind_mph": hourly["wind_speed_10m"][i],
                "gfs_wind_dir_deg":
                    hourly["wind_direction_10m"][i],
                "gfs_shortwave_wm2":
                    hourly["shortwave_radiation"][i],
            })

    if not records:
        raise RuntimeError("No GFS runs downloaded")

    df = pd.DataFrame(records).sort_values(
        ["run_init_utc", "valid_utc"]
    )

    df.to_csv(OUT / "gfs_hourly_runs.csv", index=False)

    (OUT / "unavailable_gfs_runs.json").write_text(
        json.dumps(unavailable, indent=2) + "\n"
    )

    print(
        f"GFS runs: {df['run_init_utc'].nunique()}",
        f"Hourly records: {len(df)}",
        f"Unavailable: {len(unavailable)}",
    )


def collect_observations():
    print("\n=== HISTORICAL STATION OBSERVATIONS ===")

    tables = []

    for station, network in STATIONS.items():
        cache = IEM_RAW / f"{station}_observations.csv"

        if not cache.exists():
            params = [
                ("network", network),
                ("station", station),
                ("data", "all"),
                ("year1", 2026),
                ("month1", 10),
                ("day1", 24),
                ("year2", 2026),
                ("month2", 11),
                ("day2", 10),
                ("tz", "Etc/UTC"),
                ("format", "onlycomma"),
                ("latlon", "no"),
                ("elev", "no"),
                ("missing", "M"),
                ("trace", "T"),
                ("direct", "no"),
                ("report_type", 3),
                ("report_type", 4),
            ]

            response = request(IEM_API, params)

            if response.status_code != 200:
                raise RuntimeError(
                    f"IEM {station}: HTTP {response.status_code}"
                )

            if response.text.lstrip().startswith("<"):
                raise RuntimeError(
                    f"IEM {station} returned HTML instead of CSV"
                )

            cache.write_text(response.text)
            time.sleep(0.5)

        df = pd.read_csv(
            cache, comment="#", low_memory=False
        )

        if not {"valid", "tmpf"}.issubset(df.columns):
            raise RuntimeError(
                f"Unexpected IEM columns for {station}: "
                f"{list(df.columns)}"
            )

        df["valid_utc"] = pd.to_datetime(
            df["valid"], utc=True, errors="coerce"
        )

        df["temperature_f"] = pd.to_numeric(
            df["tmpf"], errors="coerce"
        )

        df["request_station"] = station

        df = df.dropna(subset=["valid_utc"])

        relevant = df[
            (df["valid_utc"] >= "2026-10-27")
            & (df["valid_utc"] < "2026-11-10")
        ]

        at_51 = relevant[
            (relevant["valid_utc"].dt.minute == 51)
            & relevant["temperature_f"].notna()
        ]

        print(
            f"{station}: {len(df)} total observations, "
            f"{len(at_51)} temperature reports at :51"
        )

        tables.append(df)

    combined = pd.concat(tables, ignore_index=True)

    combined.to_csv(
        OUT / "iem_station_observations.csv",
        index=False,
    )

    print("\nObservation rows:", len(combined))


def main():
    initialize()
    collect_gfs()
    collect_observations()

    print("\nHOLDOUT COLLECTION COMPLETE")
    print("Output:", OUT)
    print("Original research models and files unchanged.")
    print("No trading activity.")


if __name__ == "__main__":
    main()
