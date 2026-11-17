
#!/usr/bin/env python3
"""Synoptic-only 5-minute airport temperature reader."""

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

LOCAL_TZ = {
    "miami": ZoneInfo("America/New_York"),
    "nyc": ZoneInfo("America/New_York"),
    "la": ZoneInfo("America/Los_Angeles"),
    "la-coastal": ZoneInfo("America/Los_Angeles"),
}

STATIONS = {
    "miami": (
        "KMIA", "KOPF", "KFLL", "KFXE", "KPMP"
    ),
    "nyc": (
        "KLGA", "KEWR", "KTEB", "KCDW",
        "KHPN", "KFRG", "KSMQ", "KISP"
    ),
    "la": (
        "KHHR", "KLAX", "KSMO", "KLGB", "KSNA"
    ),
    "la-coastal": (
        "KHHR", "KLAX", "KSMO", "KLGB", "KSNA"
    ),
}

API_URL = (
    "https://api.synopticdata.com/v2/"
    "stations/timeseries"
)


def read_token():
    token = (
        os.getenv("SYNOPTIC_API_TOKEN")
        or os.getenv("SYNOPTIC_TOKEN")
    )

    if token:
        return token.strip()

    paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue

        for line in path.read_text().splitlines():
            line = line.strip()

            if line.startswith("export "):
                line = line[7:].strip()

            key, sep, value = line.partition("=")

            if sep and key.strip() in (
                "SYNOPTIC_API_TOKEN",
                "SYNOPTIC_TOKEN",
            ):
                return value.strip().strip("\"'")

    return None


def fetch_synoptic(token, city, lookback):
    params = {
        "token": token,
        "stid": ",".join(STATIONS[city]),
        "network": 1,
        "recent": lookback,
        "vars": "air_temp",
        "hfmetars": 1,
        "units": "metric",
        "obtimezone": "UTC",
        "showemptystations": 1,
    }

    request = Request(
        API_URL + "?" + urlencode(params),
        headers={
            "Accept": "application/json",
            "User-Agent": "synoptic-temp-research/1.0",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = json.load(response)

    except HTTPError as exc:
        raise RuntimeError(
            f"Synoptic HTTP {exc.code}"
        ) from exc

    except (URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Synoptic request failed: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected API response")

    summary = data.get("SUMMARY") or {}

    if str(summary.get("RESPONSE_CODE")) != "1":
        message = str(
            summary.get("RESPONSE_MESSAGE", "Unknown")
        ).replace(token, "[REDACTED]")

        raise RuntimeError(
            f"Synoptic response "
            f"{summary.get('RESPONSE_CODE')}: "
            f"{message}"
        )

    return data


def parse_time(value):
    dt = datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        raise ValueError("Timestamp has no timezone")

    return dt.astimezone(timezone.utc)


def extract_observations(station):
    """Extract all valid air temperature observations."""

    obs = station.get("OBSERVATIONS") or {}
    dates = obs.get("date_time") or []

    readings = {}

    for key, values in obs.items():
        if not re.fullmatch(
            r"air_temp_set_\d+", key
        ):
            continue

        if not isinstance(values, list):
            continue

        if len(values) != len(dates):
            continue

        for stamp, value in zip(dates, values):
            try:
                temp_c = float(value)
                dt = parse_time(stamp)

            except (
                ValueError,
                TypeError,
                OverflowError,
            ):
                continue

            if (
                math.isfinite(temp_c)
                and -80 <= temp_c <= 60
            ):
                readings[dt] = temp_c

    return sorted(readings.items())


def latest_five_minute(readings, now, max_age):
    """
    Find latest observation participating in an
    approximately five-minute reporting sequence.

    Cadence inference is not definitive source metadata.
    """

    eligible = []

    for i, (stamp, temp_c) in enumerate(readings):
        age = (
            now - stamp
        ).total_seconds() / 60

        if not 0 <= age <= max_age:
            continue

        before = False
        after = False

        if i > 0:
            interval = (
                stamp - readings[i - 1][0]
            ).total_seconds() / 60

            before = 3.5 <= interval <= 6.5

        if i + 1 < len(readings):
            interval = (
                readings[i + 1][0] - stamp
            ).total_seconds() / 60

            after = 3.5 <= interval <= 6.5

        if before or after:
            eligible.append(
                (stamp, temp_c, age)
            )

    if not eligible:
        return None

    return eligible[-1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "city",
        choices=STATIONS,
    )

    parser.add_argument(
        "--max-age-minutes",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=120,
    )

    args = parser.parse_args()

    if not 5 <= args.max_age_minutes <= 240:
        parser.error(
            "--max-age-minutes must be 5-240"
        )

    if not 30 <= args.lookback_minutes <= 360:
        parser.error(
            "--lookback-minutes must be 30-360"
        )

    token = read_token()

    if not token:
        parser.error(
            "SYNOPTIC_API_TOKEN missing"
        )

    data = fetch_synoptic(
        token,
        args.city,
        args.lookback_minutes,
    )

    returned = {
        str(s["STID"]).upper(): s
        for s in data.get("STATION", [])
        if isinstance(s, dict) and s.get("STID")
    }

    now = datetime.now(timezone.utc)
    temperatures = []

    print()
    print(
        f"{args.city.upper()} | DIRECT SYNOPTIC "
        f"| NETWORK 1"
    )
    print(
        "HF-METAR enabled: approximately "
        "5-minute observations"
    )
    print(
        "Retrieved:",
        now.astimezone(LOCAL_TZ[args.city]).isoformat(
            timespec="seconds"
        ),
    )
    print(
        "Maximum accepted age:",
        args.max_age_minutes,
        "minutes",
    )

    print("-" * 85)

    print(
        f"{'Station':<10}"
        f"{'Temp F':>10}"
        f"{'Age min':>10}  "
        f"{'Observation ET':<23}"
        f"Status"
    )

    for sid in STATIONS[args.city]:
        station = returned.get(sid)

        if station is None:
            print(
                f"{sid:<10}"
                f"{'--':>10}"
                f"{'--':>10}  "
                f"{'--':<23}"
                "NOT RETURNED"
            )
            continue

        if str(station.get("MNET_ID")) != "1":
            print(
                f"{sid:<10}"
                f"{'--':>10}"
                f"{'--':>10}  "
                f"{'--':<23}"
                "WRONG NETWORK"
            )
            continue

        readings = extract_observations(station)

        latest = latest_five_minute(
            readings,
            now,
            args.max_age_minutes,
        )

        if latest is None:
            if readings:
                stamp = readings[-1][0]
                age = (
                    now - stamp
                ).total_seconds() / 60

                print(
                    f"{sid:<10}"
                    f"{'--':>10}"
                    f"{age:>10.1f}  "
                    f"{stamp.astimezone(LOCAL_TZ[args.city]).strftime('%Y-%m-%d %H:%M'):<23}"
                    "NO RECENT 5-MIN CADENCE"
                )
            else:
                print(
                    f"{sid:<10}"
                    f"{'--':>10}"
                    f"{'--':>10}  "
                    f"{'--':<23}"
                    "NO TEMPERATURE DATA"
                )
            continue

        stamp, temp_c, age = latest

        temp_f = temp_c * 1.8 + 32
        temperatures.append(temp_f)

        print(
            f"{sid:<10}"
            f"{temp_f:>10.2f}"
            f"{age:>10.1f}  "
            f"{stamp.astimezone(LOCAL_TZ[args.city]).strftime('%Y-%m-%d %H:%M'):<23}"
            "5-MIN CADENCE"
        )

    print("-" * 85)

    total = len(STATIONS[args.city])

    if temperatures:
        average = (
            sum(temperatures) /
            len(temperatures)
        )

        print(
            f"\nLatest temperature average: "
            f"{average:.2f} F"
        )

        print(
            f"Stations used: "
            f"{len(temperatures)}/{total}"
        )

        if len(temperatures) < total:
            print(
                "WARNING: Partial average. "
                "Not all stations have recent "
                "5-minute observations."
            )

    else:
        print(
            "\nAverage: UNAVAILABLE"
        )
        print(
            "No stations with recent "
            "5-minute reporting cadence."
        )

    print(
        "\nSOURCE: SYNOPTIC ONLY"
    )
    print(
        "Five-minute reporting is inferred "
        "from observation intervals."
    )
    print(
        "This is not the official "
        "Kalshi settlement index."
    )


if __name__ == "__main__":
    try:
        main()

    except RuntimeError as exc:
        sys.exit(f"ERROR: {exc}")
