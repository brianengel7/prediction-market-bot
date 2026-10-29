
#!/usr/bin/env python3
"""Live NYC weather-feed latency measurement.

Polls public AWC and NWS endpoints and records first-seen
timestamps. Does not trade, access GitHub, or modify
existing research datasets.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


OUTPUT = Path(
    "data/research/nyc_observation_20260701_20261026/"
    "live_latency/feed_events.jsonl"
)

AWC_URL = "https://aviationweather.gov/api/data/metar"
NWS_URL = (
    "https://api.weather.gov/stations/KNYC/"
    "observations/latest"
)

STATIONS = {"KNYC", "KLGA", "KEWR", "KTEB"}


def now():
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    )


def normalize_timestamp(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(
            value, timezone.utc
        ).isoformat()

    try:
        value = str(value).replace("Z", "+00:00")
        ts = datetime.fromisoformat(value)

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return ts.astimezone(timezone.utc).isoformat()

    except (ValueError, TypeError, OverflowError):
        return None


def existing_reports():
    seen = set()

    if not OUTPUT.exists():
        return seen

    for line in OUTPUT.read_text().splitlines():
        try:
            row = json.loads(line)

            if row.get("kind") == "observation":
                seen.add((
                    row["feed"],
                    row["station"],
                    row["observed_at_utc"],
                ))
        except (ValueError, KeyError):
            continue

    return seen


def save(row):
    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def fetch_awc(session):
    requested = now()

    response = session.get(
        AWC_URL,
        params={
            "ids": ",".join(sorted(STATIONS)),
            "format": "json",
            "hours": 2,
        },
        timeout=20,
    )

    received = now()

    if response.status_code == 204:
        return [], requested, received, response.headers

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError("Unexpected AWC response format")

    records = []

    for item in payload:
        station = str(
            item.get("icaoId") or ""
        ).upper()

        if station not in STATIONS:
            continue

        observed = normalize_timestamp(
            item.get("obsTime")
            or item.get("reportTime")
        )

        if observed is None:
            continue

        temp_c = item.get("temp")

        try:
            temp_f = (
                round(float(temp_c) * 1.8 + 32, 3)
                if temp_c is not None
                else None
            )
        except (ValueError, TypeError):
            temp_f = None

        records.append({
            "feed": "AWC",
            "station": station,
            "observed_at_utc": observed,
            "temperature_f": temp_f,
            "raw": item.get("rawOb"),
        })

    return (
        records,
        requested,
        received,
        response.headers,
    )


def fetch_nws(session):
    requested = now()

    response = session.get(
        NWS_URL,
        headers={
            "Accept": "application/geo+json"
        },
        timeout=20,
    )

    received = now()
    response.raise_for_status()

    payload = response.json()
    props = payload.get("properties") or {}

    observed = normalize_timestamp(
        props.get("timestamp")
    )

    if not observed:
        raise ValueError(
            "NWS observation timestamp missing"
        )

    temp_c = (
        props.get("temperature") or {}
    ).get("value")

    try:
        temp_f = (
            round(float(temp_c) * 1.8 + 32, 3)
            if temp_c is not None
            else None
        )
    except (ValueError, TypeError):
        temp_f = None

    records = [{
        "feed": "NWS",
        "station": "KNYC",
        "observed_at_utc": observed,
        "temperature_f": temp_f,
        "raw": props.get("rawMessage"),
    }]

    return (
        records,
        requested,
        received,
        response.headers,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch each source once, then exit",
    )

    parser.add_argument(
        "--minutes",
        type=int,
        default=0,
        help="Duration; 0 means run until Ctrl+C",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between polls; minimum 60",
    )

    args = parser.parse_args()

    if args.interval < 60 or args.minutes < 0:
        parser.error(
            "Interval must be >=60 and duration >=0"
        )

    seen = existing_reports()
    initialized = set()

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "NYC-weather-latency-research/1.0 "
            "(user-operated research)"
        ),
        "Accept": "application/json",
    })

    print("LIVE WEATHER LATENCY MONITOR")
    print("Logging:", OUTPUT.resolve())
    print("Polling interval:", args.interval, "seconds")
    print(
        "Initial observations are BASELINE, "
        "not verified publication events."
    )
    print("Press Ctrl+C to stop.\n", flush=True)

    deadline = (
        time.monotonic() + args.minutes * 60
        if args.minutes
        else None
    )

    try:
        while True:
            cycle_start = time.monotonic()

            for name, fetch in [
                ("AWC", fetch_awc),
                ("NWS", fetch_nws),
            ]:
                try:
                    records, requested, received, headers = (
                        fetch(session)
                    )

                    new_count = 0

                    for record in records:
                        key = (
                            record["feed"],
                            record["station"],
                            record["observed_at_utc"],
                        )

                        if key in seen:
                            continue

                        baseline = (
                            (
                                record["feed"],
                                record["station"],
                            )
                            not in initialized
                        )

                        seen.add(key)
                        new_count += 1

                        row = {
                            **record,
                            "kind": "observation",
                            "first_seen_utc": received,
                            "requested_at_utc": requested,
                            "baseline": baseline,
                            "http_age": headers.get("Age"),
                            "http_cache_control": (
                                headers.get("Cache-Control")
                            ),
                            "http_date": headers.get("Date"),
                        }

                        save(row)

                        print(
                            f"{received} "
                            f"{record['feed']:3s} "
                            f"{record['station']} "
                            f"OBS={record['observed_at_utc']} "
                            f"TEMP={record['temperature_f']}F "
                            f"{'BASELINE' if baseline else 'NEW'}",
                            flush=True,
                        )

                    initialized.update(
                        (r["feed"], r["station"])
                        for r in records
                    )

                    if new_count == 0:
                        print(
                            f"{received} {name}: "
                            f"{len(records)} reports, "
                            "no new observations",
                            flush=True,
                        )

                except (
                    requests.RequestException,
                    ValueError,
                    KeyError,
                ) as error:

                    save({
                        "kind": "error",
                        "feed": name,
                        "at_utc": now(),
                        "error": str(error),
                    })

                    print(
                        f"{now()} {name} ERROR: {error}",
                        flush=True,
                    )

            if args.once:
                break

            if (
                deadline
                and time.monotonic() >= deadline
            ):
                break

            elapsed = time.monotonic() - cycle_start
            wait = max(0, args.interval - elapsed)

            if deadline:
                wait = min(
                    wait,
                    max(0, deadline - time.monotonic()),
                )

            if wait:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("Saved:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
