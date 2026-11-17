
#!/usr/bin/env python3
"""Backfill Miami airport temperatures from IEM. Research only."""

import csv
import io
import re
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "temp-prediction" / "miami_hourly_index"

OUTPUT = DATA / "miami_iem_observations.csv"
TARGETS = DATA / "miami_hourly_targets.csv"

STATIONS = {
    "MIA": "KMIA",
    "OPF": "KOPF",
    "FLL": "KFLL",
    "FXE": "KFXE",
    "PMP": "KPMP",
}

UTC = timezone.utc
API = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def parse_time(s):
    s = str(s).strip().replace(" ", "T")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)

def metar_temperature_f(raw):
    """Extract air temperature from a raw METAR report."""

    raw = (raw or "").upper()

    # Prefer the precise METAR temperature remarks group.
    # Example: T02400230 means +24.0 C.
    match = re.search(
        r"(?:^|\s)T([01])(\d{3})[01]\d{3}(?=\s|$)",
        raw,
    )

    if match:
        sign = -1 if match.group(1) == "1" else 1
        celsius = sign * int(match.group(2)) / 10
        return celsius * 1.8 + 32

    # Fallback: conventional whole-Celsius temperature.
    # Examples: 24/23 or M02/M05.
    body = raw.split(" RMK ", 1)[0]

    for token in body.split():
        match = re.fullmatch(
            r"(M?)(\d{2})/(?:M?\d{2}|//)",
            token,
        )

        if match:
            sign = -1 if match.group(1) else 1
            celsius = sign * int(match.group(2))
            return celsius * 1.8 + 32

    return None


def download(sid, start, end):
    params = [
        ("station", sid),
        ("data", "tmpf"),
        ("data", "metar"),
        ("sts", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("ets", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("missing", "M"),
        ("direct", "no"),
        ("report_type", "1"),  # Five-minute HF-METAR
        ("report_type", "3"),  # Routine METAR
        ("report_type", "4"),  # Special METAR
    ]

    req = Request(
        API + "?" + urlencode(params),
        headers={
            "User-Agent": "MiamiHourlyIndexResearch/1.0"
        },
    )

    try:
        with urlopen(req, timeout=120) as response:
            content = response.read().decode("utf-8-sig")

    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"IEM request for {sid} failed: {exc}"
        ) from exc

    lines = (
        line for line in io.StringIO(content)
        if line.strip()
        and not line.lstrip().startswith("#")
    )

    reader = csv.DictReader(lines)

    required = {"station", "valid", "tmpf"}

    if not reader.fieldnames or not required.issubset(
        reader.fieldnames
    ):
        raise RuntimeError(
            f"Unexpected IEM CSV for {sid}: "
            f"{content[:180]!r}"
        )

    collected = {}
    harvested = datetime.now(UTC).isoformat(
        timespec="seconds"
    )

    for row in reader:
        try:
            obs = parse_time(row["valid"])

        except (ValueError, TypeError, KeyError):
            continue

        # First attempt IEM's standard temperature.
        try:
            temp = float(row.get("tmpf", ""))

        except (ValueError, TypeError):
            temp = None

        # HF-METAR temperatures are often intentionally blanked
        # in tmpf, but retained inside the original METAR text.
        if temp is None or not math.isfinite(temp):
            temp = metar_temperature_f(
                row.get("metar", "")
            )

        if temp is None:
            continue

        if not (
            math.isfinite(temp)
            and -40 <= temp <= 130
            and start <= obs < end
        ):
            continue

        returned_sid = str(
            row.get("station", "")
        ).strip().upper()

        if returned_sid not in {sid, "K" + sid}:
            continue

        key = obs.isoformat(timespec="seconds")

        collected[key] = {
            "stid": STATIONS[sid],
            "obs_utc": key,
            "temp_f": f"{temp:.4f}",
            "retrieved_at_utc": harvested,
        }

    return list(collected.values())


def write_averaged_history(rows, start, end):
    """Build a 5-minute Miami average from all five stations.

    Uses only observations at or before each timestamp.
    Requires all five stations for a complete average.
    """
    from bisect import bisect_right

    indexed = {}

    for sid in STATIONS.values():
        observations = {
            parse_time(row["obs_utc"]): float(row["temp_f"])
            for row in rows
            if row["stid"] == sid
        }

        times = sorted(observations)
        indexed[sid] = (
            times,
            [observations[t] for t in times],
        )

    output = DATA / "miami_iem_average_5m.csv"

    fields = [
        "minute_utc",
        "average_f",
        "stations_used",
    ]

    for sid in STATIONS.values():
        fields.extend([
            sid + "_f",
            sid + "_age_min",
        ])

    results = []

    # Align timestamps to UTC five-minute boundaries.
    minute = start.replace(second=0, microsecond=0)

    if minute < start:
        minute += timedelta(minutes=1)

    minute += timedelta(minutes=(-minute.minute) % 5)

    while minute < end:
        record = {
            "minute_utc": minute.isoformat(),
            "average_f": "",
            "stations_used": 0,
        }

        temps = []

        for sid in STATIONS.values():
            times, values = indexed[sid]
            pos = bisect_right(times, minute) - 1

            temp = None
            age = None

            if pos >= 0:
                age = (
                    minute - times[pos]
                ).total_seconds() / 60

                # Prevent stale weather observations from
                # being carried forward indefinitely.
                if 0 <= age <= 15:
                    temp = values[pos]

            record[sid + "_f"] = (
                f"{temp:.4f}" if temp is not None else ""
            )
            record[sid + "_age_min"] = (
                f"{age:.1f}" if temp is not None else ""
            )

            if temp is not None:
                temps.append(temp)

        record["stations_used"] = len(temps)

        if len(temps) == 5:
            record["average_f"] = (
                f"{sum(temps) / 5:.4f}"
            )

        results.append(record)
        minute += timedelta(minutes=5)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    complete = sum(
        bool(r["average_f"]) for r in results
    )

    print(f"\nHistorical Miami average: {output}")
    print(f"Five-minute timestamps: {len(results):,}")
    print(f"Complete five-station averages: {complete:,}")
    print(
        f"Coverage: {complete / len(results):.1%}"
        if results else "Coverage: 0%"
    )



def main():
    if not TARGETS.exists():
        raise RuntimeError(
            f"Missing Kalshi targets: {TARGETS}"
        )

    with TARGETS.open(newline="") as f:
        hours = [
            parse_time(r["settlement_hour_utc"])
            for r in csv.DictReader(f)
        ]

    if not hours:
        raise RuntimeError("No Kalshi target hours")

    start = min(hours) - timedelta(hours=3)

    end = min(
        max(hours) + timedelta(hours=1),
        datetime.now(UTC),
    )

    print(
        f"IEM UTC window: {start.isoformat()} "
        f"through {end.isoformat()}",
        flush=True,
    )

    rows = []

    for station in STATIONS:
        found = download(station, start, end)

        print(
            f"{station}: {len(found):,} observations",
            flush=True,
        )

        rows.extend(found)

        # Respect IEM's documented per-IP throttle.
        time.sleep(1.2)

    if not rows:
        raise RuntimeError(
            "IEM returned no usable temperatures; "
            "no file was replaced"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = OUTPUT.with_suffix(".tmp")

    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stid",
                "obs_utc",
                "temp_f",
                "retrieved_at_utc",
            ],
        )

        writer.writeheader()

        writer.writerows(
            sorted(
                rows,
                key=lambda r: (
                    r["obs_utc"],
                    r["stid"],
                ),
            )
        )

    tmp.replace(OUTPUT)
    write_averaged_history(rows, start, end)

    counts = Counter(
        r["stid"] for r in rows
    )

    print()
    print(
        f"Saved {len(rows):,} observations "
        f"to {OUTPUT}"
    )

    if len(counts) != len(STATIONS):
        print(
            "WARNING: Some stations have no "
            "IEM observations"
        )

    print(
        "NOTE: Archive retrieval timestamps "
        "are not original live receipt times."
    )


if __name__ == "__main__":
    try:
        main()

    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
