
#!/usr/bin/env python3
"""Collect historical GFS single runs via Open-Meteo."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import (
    date, datetime, timedelta, timezone
)
from pathlib import Path

import requests


BASE = (
    Path(
        "data/research/"
        "nyc_observation_20260701_20261026"
    )
    / "v8_gfs"
)

ENDPOINT = (
    "https://single-runs-api.open-meteo.com/"
    "v1/forecast"
)

HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]

COLS = [
    "run_init_utc",
    "valid_utc",
    "gfs_temp_f",
    "gfs_rh_pct",
    "gfs_cloud_pct",
    "gfs_wind_mph",
    "gfs_wind_dir_deg",
    "gfs_shortwave_wm2",
]


def runs(start: date, end: date):
    day = start

    while day <= end:
        for hour in (0, 6, 12, 18):
            yield datetime(
                day.year,
                day.month,
                day.day,
                hour,
                tzinfo=timezone.utc,
            )

        day += timedelta(days=1)


def file_for(directory: Path, init: datetime):
    return directory / f"{init:%Y%m%dT%HZ}.json"


def normalized_rows(path: Path):
    init = datetime.strptime(
        path.stem, "%Y%m%dT%HZ"
    ).replace(tzinfo=timezone.utc)

    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    hourly = obj.get("hourly", {})
    units = obj.get("hourly_units", {})

    if units.get("temperature_2m") not in (
        "°F", "F"
    ):
        raise ValueError(
            f"{path}: expected Fahrenheit, "
            f"got {units.get('temperature_2m')}"
        )

    if units.get("wind_speed_10m") not in (
        "mp/h", "mph"
    ):
        raise ValueError(
            f"{path}: expected mph, "
            f"got {units.get('wind_speed_10m')}"
        )

    valid = hourly.get("time", [])

    if (
        not valid
        or len(hourly.get("temperature_2m", []))
        != len(valid)
    ):
        raise ValueError(
            f"{path}: missing hourly data"
        )

    for i, text in enumerate(valid):
        when = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )

        if when.tzinfo is None:
            when = when.replace(
                tzinfo=timezone.utc
            )

        when = when.astimezone(timezone.utc)

        row = [
            init.isoformat().replace(
                "+00:00", "Z"
            ),
            when.isoformat().replace(
                "+00:00", "Z"
            ),
        ]

        for key in HOURLY:
            values = hourly.get(key, [])

            row.append(
                values[i]
                if len(values) == len(valid)
                else None
            )

        yield row


def rebuild(directory: Path, csv_path: Path):
    paths = sorted(
        directory.glob(
            "????????T??Z.json"
        )
    )

    if not paths:
        print(
            "No archived GFS JSON runs "
            "to normalize."
        )
        return

    csv_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    temp = csv_path.with_suffix(".tmp")

    total = 0
    failed = 0

    with temp.open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(COLS)

        for path in paths:
            try:
                rows = list(
                    normalized_rows(path)
                )

                writer.writerows(rows)
                total += len(rows)

            except (
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                failed += 1

                print(
                    "SKIP INVALID CACHE",
                    path.name,
                    str(exc)[:150],
                )

    os.replace(temp, csv_path)

    print(
        "Normalized runs:",
        len(paths) - failed,
        "rows:", total,
        "invalid:", failed,
        "file:", csv_path,
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--start",
        default="2026-04-02",
    )

    p.add_argument(
        "--end",
        default="2026-10-26",
    )

    p.add_argument(
        "--max-new",
        type=int,
        default=8,
        help=(
            "Maximum new downloads per invocation. "
            "0 means unlimited."
        ),
    )

    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--out",
        type=Path,
        default=BASE,
    )

    p.add_argument(
        "--normalize-only",
        action="store_true",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = p.parse_args()

    start = date.fromisoformat(
        args.start
    )

    end = date.fromisoformat(
        args.end
    )

    if start < date(2026, 4, 2):
        p.error(
            "GFS Single Runs archive "
            "generally begins April 2, 2026."
        )

    if end < start:
        p.error(
            "End date is earlier than start date"
        )

    if (
        args.max_new < 0
        or args.sleep_seconds < 0
    ):
        p.error(
            "Limits must be nonnegative"
        )

    raw = (
        args.out / "raw_gfs_runs"
    )

    csv_path = (
        args.out / "gfs_hourly_runs.csv"
    )

    wanted = list(
        runs(start, end)
    )

    already = sum(
        file_for(raw, t).exists()
        for t in wanted
    )

    print(
        "GFS runs requested:", len(wanted),
        "cached:", already,
        "missing:", len(wanted) - already,
    )

    if args.dry_run:
        shown = 0

        for init in wanted:
            if not file_for(raw, init).exists():
                print(
                    "MISSING",
                    init.isoformat()
                )

                shown += 1

                if shown >= 12:
                    break

        return

    if not args.normalize_only:
        raw.mkdir(
            parents=True,
            exist_ok=True
        )

        session = requests.Session()

        session.headers.update({
            "User-Agent":
                "KNYC-GFS-V8-research/1.0"
        })

        made = 0

        for init in wanted:
            path = file_for(raw, init)

            if path.exists():
                continue

            if (
                args.max_new
                and made >= args.max_new
            ):
                break

            params = {
                "latitude": 40.77898,
                "longitude": -73.96925,
                "run": init.strftime(
                    "%Y-%m-%dT%H:%M"
                ),
                "models": "gfs_global",
                "hourly": ",".join(HOURLY),
                "timezone": "UTC",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "forecast_days": 2,
            }

            print(
                "DOWNLOAD",
                params["run"],
                flush=True,
            )

            try:
                response = session.get(
                    ENDPOINT,
                    params=params,
                    timeout=60,
                )

                if response.status_code == 429:
                    print(
                        "RATE LIMITED: stopping"
                    )
                    break

                response.raise_for_status()
                payload = response.json()

                if payload.get("error"):
                    raise RuntimeError(
                        str(
                            payload.get(
                                "reason", payload
                            )
                        )[:200]
                    )

                if (
                    not isinstance(
                        payload.get("hourly"),
                        dict,
                    )
                    or not payload[
                        "hourly"
                    ].get("time")
                ):
                    raise ValueError(
                        "Missing hourly forecast"
                    )

                temp = path.with_suffix(
                    ".tmp"
                )

                temp.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

                os.replace(temp, path)
                made += 1

            except (
                requests.RequestException,
                ValueError,
                RuntimeError,
            ) as exc:
                print(
                    "FAILED",
                    type(exc).__name__,
                    str(exc)[:250],
                )
                break

            time.sleep(
                args.sleep_seconds
            )

        print(
            "Downloaded new runs:",
            made
        )

    rebuild(raw, csv_path)


if __name__ == "__main__":
    main()
