
#!/usr/bin/env python3
"""Download historical MADIS five-minute station temperatures.

Stores extracted station records, not nationwide NetCDF files.
Does not modify existing models or trading code.
"""

import argparse
import gzip
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, chartostring


BASE_URL = (
    "https://madis-data.ncep.noaa.gov/"
    "madisPublic1/data/archive/"
)

ROOT = Path(
    "data/research/nyc_observation_20260701_20261026/"
    "madis_historical"
)

STATIONS = {
    "KLGA", "KTEB", "KEWR", "KJRB", "KJFK"
}

LOCAL_TZ = ZoneInfo("America/New_York")

COLUMNS = [
    "station",
    "observation_utc",
    "temperature_f",
    "madis_received_utc",
    "source_file",
]


def parse_gz(payload, filename):
    raw = gzip.decompress(payload)

    with Dataset("inmemory", memory=raw) as ds:
        ids = np.asarray(
            chartostring(
                ds.variables["stationId"][:]
            )
        ).astype(str)

        ids = np.char.strip(ids)

        indices = np.flatnonzero(
            np.isin(ids, list(STATIONS))
        )

        if not len(indices):
            return pd.DataFrame(columns=COLUMNS)

        temperatures = (
            ds.variables["temperature"][indices]
        )
        observed = (
            ds.variables["observationTime"][indices]
        )

        received = (
            ds.variables["receivedTime"][indices]
            if "receivedTime" in ds.variables
            else None
        )

        records = []

        for j, i in enumerate(indices):
            if (
                np.ma.is_masked(temperatures[j])
                or np.ma.is_masked(observed[j])
            ):
                continue

            kelvin = float(temperatures[j])
            timestamp = float(observed[j])

            if not (
                200 < kelvin < 335
                and 1_500_000_000
                < timestamp
                < 2_000_000_000
            ):
                continue

            reported = ""

            if (
                received is not None
                and not np.ma.is_masked(received[j])
            ):
                r = float(received[j])

                if 1_500_000_000 < r < 2_000_000_000:
                    reported = datetime.fromtimestamp(
                        r, timezone.utc
                    ).isoformat()

            records.append({
                "station": str(ids[i]),
                "observation_utc":
                    datetime.fromtimestamp(
                        timestamp, timezone.utc
                    ).isoformat(),
                "temperature_f": round(
                    (kelvin - 273.15) * 1.8 + 32,
                    4,
                ),
                "madis_received_utc": reported,
                "source_file": filename,
            })

    return pd.DataFrame.from_records(
        records, columns=COLUMNS
    )


def wanted_hours(start, end):
    day = start

    while day <= end:
        # Include early hours for trend history.
        # Covers the 8 AM–5 PM ET research session.
        for hour in range(6, 18):
            local = datetime(
                day.year,
                day.month,
                day.day,
                hour,
                tzinfo=LOCAL_TZ,
            )

            yield local.astimezone(timezone.utc)

        day += timedelta(days=1)


def fetch(session, url):
    for trial in range(4):
        try:
            response = session.get(
                url,
                timeout=(15, 80),
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.content

        except (requests.RequestException, OSError):
            if trial == 3:
                raise

            time.sleep(2 ** trial * 2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2026-07-01",
    )
    parser.add_argument(
        "--end",
        default="2026-10-26",
    )

    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if end < start:
        parser.error("--end must be >= --start")

    cache = ROOT / "hourly_extracted"
    cache.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "PolyQuant-historical-temperature-research/1.0"
    })

    frames = []
    failures = []
    hours = list(wanted_hours(start, end))

    print(
        f"MADIS historical: {len(hours)} hourly files "
        f"requested ({start} to {end})",
        flush=True,
    )

    print(
        "Extracting five nearby stations. "
        "Nationwide files will not be retained.",
        flush=True,
    )

    for n, hour in enumerate(hours, 1):
        filename = hour.strftime(
            "%Y%m%d_%H00.gz"
        )

        path = cache / (filename + ".csv")

        if path.exists():
            frame = pd.read_csv(path)

        else:
            url = (
                BASE_URL
                + hour.strftime("%Y/%m/%d")
                + "/LDAD/hfmetar/netCDF/"
                + filename
            )

            try:
                raw = fetch(session, url)

                if raw is None:
                    failures.append(
                        (filename, "404")
                    )
                    continue

                frame = parse_gz(
                    raw, filename
                )

                frame.to_csv(
                    path, index=False
                )

            except Exception as exc:
                failures.append(
                    (filename, str(exc)[:180])
                )

                print(
                    f"FAILED {filename}: {exc}",
                    flush=True,
                )
                continue

            time.sleep(0.15)

        frames.append(frame)

        if (
            n == 1
            or n % 24 == 0
            or n == len(hours)
        ):
            print(
                f"Processed {n}/{len(hours)} hours; "
                f"current {filename}; "
                f"failures={len(failures)}",
                flush=True,
            )

    if not frames:
        raise SystemExit(
            "No data downloaded. Check archive connectivity."
        )

    df = pd.concat(
        frames, ignore_index=True
    )

    df["observation_utc"] = pd.to_datetime(
        df["observation_utc"],
        utc=True,
        errors="coerce",
    )

    df["temperature_f"] = pd.to_numeric(
        df["temperature_f"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "observation_utc",
            "temperature_f",
        ]
    )

    # Exclude conflicting archive records.
    conflict = df.groupby(
        ["station", "observation_utc"]
    )["temperature_f"].agg(
        lambda x: x.max() - x.min()
    )

    bad = conflict[
        conflict > 0.2
    ].index

    if len(bad):
        key = pd.MultiIndex.from_frame(
            df[["station", "observation_utc"]]
        )

        df = df[
            ~key.isin(bad)
        ]

    df = (
        df.sort_values([
            "station",
            "observation_utc",
        ])
        .drop_duplicates([
            "station",
            "observation_utc",
        ])
    )

    output = ROOT / (
        f"observations_"
        f"{start:%Y%m%d}_"
        f"{end:%Y%m%d}.csv"
    )

    df.to_csv(output, index=False)

    df["local_day"] = (
        df["observation_utc"]
        .dt.tz_convert(LOCAL_TZ)
        .dt.date
    )

    coverage = (
        df.groupby(["local_day", "station"])
        .agg(
            n=("temperature_f", "size"),
            first=("observation_utc", "min"),
            last=("observation_utc", "max"),
        )
        .reset_index()
    )

    coverage.to_csv(
        ROOT / (
            f"coverage_{start:%Y%m%d}_"
            f"{end:%Y%m%d}.csv"
        ),
        index=False,
    )

    pd.DataFrame(
        failures,
        columns=["file", "error"],
    ).to_csv(
        ROOT / (
            f"failures_{start:%Y%m%d}_"
            f"{end:%Y%m%d}.csv"
        ),
        index=False,
    )

    print("\n=== SUMMARY ===")
    print("Total observations:", len(df))

    print(
        "By station:\n",
        df.groupby("station").size().to_string(),
    )

    print(
        "Days with observations:\n",
        df.groupby("station")[
            "local_day"
        ].nunique().to_string(),
    )

    print("Failed hours:", len(failures))
    print("Saved:", output.resolve())

    print(
        "CAUTION: Historical MADIS receipt timestamps "
        "do not establish first public availability."
    )


if __name__ == "__main__":
    main()
