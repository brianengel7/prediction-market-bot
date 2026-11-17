
import csv
import gzip
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from netCDF4 import Dataset, chartostring

BASE = (
    "https://madis-data.ncep.noaa.gov/"
    "madisPublic1/data/LDAD/hfmetar/netCDF/"
)

STATIONS = {"KLGA", "KTEB", "KJRB", "KEWR", "KJFK"}

OUTPUT = Path(
    "data/research/nyc_observation_20260701_20261026/"
    "live_latency/madis_five_minute.csv"
)

POLL_SECONDS = 120

FIELDS = [
    "station",
    "observation_utc",
    "first_seen_utc",
    "delay_minutes",
    "temp_f",
    "baseline",
    "source_file",
    "file_last_modified",
    "madis_received_utc",
]


def utcnow():
    return datetime.now(timezone.utc)


def utcstr(ts):
    return ts.isoformat(timespec="milliseconds")


def epoch(value):
    if np.ma.is_masked(value):
        return None
    try:
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            return None
        return datetime.fromtimestamp(number, timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def read_existing():
    seen = set()

    if OUTPUT.exists():
        with OUTPUT.open(newline="") as file:
            for row in csv.DictReader(file):
                seen.add((
                    row["station"],
                    row["observation_utc"],
                ))

    return seen


def write_rows(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT.exists()

    with OUTPUT.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


def get_latest_files(session):
    response = session.get(BASE, timeout=30)
    response.raise_for_status()

    filenames = sorted(set(re.findall(
        r"20\d{6}_\d{4}\.gz", response.text
    )))

    return filenames[-2:]


def parse_file(session, filename):
    response = session.get(
        BASE + filename,
        timeout=60,
    )
    response.raise_for_status()

    collected = utcnow()
    modified = response.headers.get("Last-Modified")

    raw = gzip.decompress(response.content)
    records = []

    with Dataset("memory", memory=raw) as ds:
        ids = [
            str(value).strip().upper()
            for value in chartostring(
                ds.variables["stationId"][:]
            ).tolist()
        ]

        temperatures = ds.variables["temperature"][:]
        obs_times = ds.variables["observationTime"][:]
        recv_times = ds.variables.get("receivedTime")

        for index, station in enumerate(ids):
            if station not in STATIONS:
                continue

            observed = epoch(obs_times[index])

            if observed is None:
                continue

            value = temperatures[index]

            if np.ma.is_masked(value):
                continue

            kelvin = float(value)

            if not np.isfinite(kelvin) or not 200 < kelvin < 335:
                continue

            temp_f = (kelvin - 273.15) * 1.8 + 32

            received = (
                epoch(recv_times[index])
                if recv_times is not None
                else None
            )

            records.append({
                "station": station,
                "observed": observed,
                "first_seen": collected,
                "temp_f": round(temp_f, 3),
                "file": filename,
                "modified": modified,
                "madis_received": (
                    utcstr(received)
                    if received else ""
                ),
            })

    return records


def main():
    session = requests.Session()
    session.headers["User-Agent"] = (
        "PolyQuant weather latency research/1.0"
    )

    seen = read_existing()
    first_cycle = True

    print("MADIS FIVE-MINUTE LATENCY MONITOR")
    print("Stations:", ", ".join(sorted(STATIONS)))
    print("Poll interval:", POLL_SECONDS, "seconds")
    print("Saving:", OUTPUT)
    print(
        "Startup records are BASELINE; "
        "their arrival times are unknown."
    )
    print("Press Ctrl+C to stop.\n", flush=True)

    try:
        while True:
            start = time.monotonic()
            new_rows = []
            latest = {}
            files = []

            try:
                files = get_latest_files(session)

                for filename in files:
                    try:
                        records = parse_file(
                            session, filename
                        )
                    except Exception as exc:
                        print(
                            "FILE ERROR:",
                            filename,
                            str(exc)[:150],
                            flush=True,
                        )
                        continue

                    for record in records:
                        station = record["station"]
                        observed = record["observed"]

                        if (
                            station not in latest
                            or observed > latest[station]["observed"]
                        ):
                            latest[station] = record

                        key = (station, utcstr(observed))

                        if key in seen:
                            continue

                        seen.add(key)

                        delay = (
                            record["first_seen"] - observed
                        ).total_seconds() / 60

                        row = {
                            "station": station,
                            "observation_utc": utcstr(observed),
                            "first_seen_utc": utcstr(
                                record["first_seen"]
                            ),
                            "delay_minutes": round(delay, 2),
                            "temp_f": record["temp_f"],
                            "baseline": first_cycle,
                            "source_file": record["file"],
                            "file_last_modified": (
                                record["modified"]
                            ),
                            "madis_received_utc": (
                                record["madis_received"]
                            ),
                        }

                        new_rows.append(row)

                        if not first_cycle:
                            print(
                                f"NEW {station} "
                                f"observed={utcstr(observed)} "
                                f"first_seen={row['first_seen_utc']} "
                                f"delay={delay:.1f}m "
                                f"temp={record['temp_f']:.2f}F",
                                flush=True,
                            )

                if new_rows:
                    write_rows(new_rows)

                checked = utcnow()

                print(
                    f"\nCHECK {utcstr(checked)} "
                    f"files={files} "
                    f"new_records={len(new_rows)}"
                )

                for station in sorted(STATIONS):
                    item = latest.get(station)

                    if item is None:
                        print(station, "NO DATA")
                        continue

                    age = (
                        checked - item["observed"]
                    ).total_seconds() / 60

                    print(
                        f"{station}: latest observation "
                        f"{utcstr(item['observed'])} "
                        f"age={age:.1f}m "
                        f"temp={item['temp_f']:.2f}F"
                    )

                print("", flush=True)
                first_cycle = False

            except Exception as exc:
                print(
                    "POLL ERROR:",
                    type(exc).__name__,
                    str(exc)[:200],
                    flush=True,
                )

            time.sleep(max(
                0,
                POLL_SECONDS - (
                    time.monotonic() - start
                )
            ))

    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        print("Results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
