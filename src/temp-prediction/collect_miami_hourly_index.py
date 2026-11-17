
#!/usr/bin/env python3
"""Collect Miami Kalshi Weather Index history and test a prediction baseline.

Read-only research. Never places trades.
Historical observations do not establish their actual live publication times.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "temp-prediction" / "miami_hourly_index"

URL = (
    "https://external-api.kalshi.com/trade-api/v2/"
    "live_data/weather/miami"
)

STATIONS = (
    "KMIA1M",
    "KOPF1M",
    "KFLL1M",
    "KFXE1M",
    "KPMP1M",
)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

HORIZONS = (5, 10, 15, 20)

# Conservative assumption:
# An index minute becomes usable 6 minutes after observation.
DELAY_MINUTES = 6


def iso(dt):
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def parse_time(value):
    if isinstance(value, (float, int)):
        factor = 1000 if value > 1e11 else 1
        return datetime.fromtimestamp(value / factor, UTC)

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if dt.tzinfo is None:
        raise ValueError("Timestamp has no timezone")

    return dt.astimezone(UTC)


def fetch(start, end):
    params = {
        "from": int(start.timestamp() * 1000),
        "to": int(end.timestamp() * 1000),
        "detailed": "true",
        "limit": 10000,
    }

    req = Request(
        URL + "?" + urlencode(params),
        headers={
            "Accept": "application/json",
            "User-Agent": "miami-index-research/1.0",
        },
    )

    for attempt in range(4):
        try:
            with urlopen(req, timeout=30) as response:
                data = json.load(response)

            if not isinstance(data, dict):
                raise RuntimeError("Unexpected Kalshi response")

            if not isinstance(data.get("timeseries"), list):
                raise RuntimeError(
                    "Kalshi response missing timeseries array"
                )

            return data

        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"Kalshi HTTP {exc.code}"
                ) from exc

            if attempt == 3:
                raise RuntimeError(
                    f"Kalshi HTTP {exc.code} after retries"
                ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    f"Kalshi connection failed: {exc}"
                ) from exc

        time.sleep(2 ** attempt)


def record(point, version, fetched_at):
    try:
        minute = parse_time(point["t"]).replace(
            second=0,
            microsecond=0,
        )

        value = float(point["v"])
        status = str(
            point.get("status", "unknown")
        ).lower()

        if not math.isfinite(value):
            return None

        if status in (
            "incomplete",
            "missing",
            "provisional",
            "pending",
        ):
            return None

        row = {
            "minute_utc": iso(minute),
            "index_f": f"{value:.4f}",
            "status": status,
            "contributors": point.get("contributors", ""),
            "response_config_version": version,
            "retrieved_at_utc": iso(fetched_at),
        }

        for sid in STATIONS:
            row[f"{sid}_f"] = ""
            row[f"{sid}_qc"] = ""

        for member in point.get("stations") or []:
            sid = str(
                member.get("station_id", "")
            ).upper()

            if sid not in STATIONS:
                if sid + "1M" in STATIONS:
                    sid += "1M"

            if sid not in STATIONS:
                continue

            temp = member.get("temp_f")

            if temp is not None:
                row[f"{sid}_f"] = temp

            row[f"{sid}_qc"] = member.get("code", "")

        return row

    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
    ):
        return None


FIELDS = [
    "minute_utc",
    "index_f",
    "status",
    "contributors",
    "response_config_version",
    "retrieved_at_utc",
]

for sid in STATIONS:
    FIELDS.extend([
        f"{sid}_f",
        f"{sid}_qc",
    ])


def read_rows(path):
    if not path.exists():
        return {}

    with path.open(newline="") as f:
        return {
            row["minute_utc"]: row
            for row in csv.DictReader(f)
        }


def write_rows(path, fields, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp, path)


def collect(days, chunk_hours, existing, live=False):
    now = datetime.now(UTC)

    # Avoid collecting minutes that haven't finalized.
    end = now - timedelta(minutes=7)

    if live:
        start = end - timedelta(minutes=20)
    else:
        start = end - timedelta(days=days)

    step = timedelta(hours=chunk_hours)

    added = 0
    updated = 0
    batches = 0

    cursor = start

    while cursor < end:
        boundary = min(cursor + step, end)

        fetched_at = datetime.now(UTC)
        data = fetch(cursor, boundary)

        batches += 1
        seen = 0

        for point in data["timeseries"]:
            row = record(
                point,
                data.get("config_version", ""),
                fetched_at,
            )

            if row is None:
                continue

            minute = parse_time(row["minute_utc"])

            # Never accidentally treat a recent API response
            # as historical data from the requested window.
            if not cursor <= minute <= boundary:
                continue

            seen += 1
            key = row["minute_utc"]

            if key not in existing:
                existing[key] = row
                added += 1
            else:
                old = existing[key]

                if old.get("retrieved_at_utc"):
                    row["retrieved_at_utc"] = min(
                        old["retrieved_at_utc"],
                        row["retrieved_at_utc"],
                    )

                # Keep previously saved station details if a
                # later response omits them.
                has_stations = any(
                    row[f"{sid}_f"] != ""
                    for sid in STATIONS
                )

                if not has_stations:
                    for sid in STATIONS:
                        row[f"{sid}_f"] = old.get(
                            f"{sid}_f", ""
                        )
                        row[f"{sid}_qc"] = old.get(
                            f"{sid}_qc", ""
                        )

                existing[key] = row
                updated += 1

        print(
            f"{iso(cursor)} to {iso(boundary)}: "
            f"{seen} valid minutes",
            flush=True,
        )

        cursor = boundary + timedelta(milliseconds=1)

        if cursor < end:
            time.sleep(0.15)

    return added, updated, batches


def build_targets_and_baseline(rows):
    parsed = sorted(
        (
            parse_time(row["minute_utc"]),
            float(row["index_f"]),
        )
        for row in rows.values()
    )

    if len(parsed) < 90:
        return [], [], {}

    minutes = [p[0] for p in parsed]
    values = [p[1] for p in parsed]

    earliest = minutes[0]
    last = minutes[-1]

    hour = earliest.replace(
        minute=0,
        second=0,
        microsecond=0,
    ) + timedelta(hours=2)

    targets = []
    predictions = []

    while (
        hour <= last
        and hour <= datetime.now(UTC) - timedelta(minutes=7)
    ):
        # Settlement: latest completed index minute at or
        # before the hour, within a maximum 60-minute lookback.
        j = bisect_right(minutes, hour) - 1

        if j >= 0 and (
            timedelta(0)
            <= hour - minutes[j]
            <= timedelta(minutes=60)
        ):
            actual = values[j]

            targets.append({
                "settlement_hour_utc": iso(hour),
                "settlement_hour_et": hour.astimezone(
                    ET
                ).isoformat(timespec="seconds"),
                "settlement_minute_utc": iso(minutes[j]),
                "settlement_index_f": f"{actual:.4f}",
                "settlement_status": rows[
                    iso(minutes[j])
                ]["status"],
            })

            for horizon in HORIZONS:
                decision = hour - timedelta(
                    minutes=horizon
                )

                # Crucial: only use information that would
                # have been finalized by decision time.
                asof = decision - timedelta(
                    minutes=DELAY_MINUTES
                )

                k = bisect_right(minutes, asof) - 1

                if k < 0:
                    continue

                # Skip unusually stale readings.
                if (
                    decision - minutes[k]
                    > timedelta(minutes=20)
                ):
                    continue

                predicted = values[k]
                error = abs(predicted - actual)

                same_bucket = int(
                    math.floor(predicted)
                    == math.floor(actual)
                )

                predictions.append({
                    "settlement_hour_utc": iso(hour),
                    "horizon_min": horizon,
                    "decision_time_utc": iso(decision),
                    "latest_available_minute_utc": iso(
                        minutes[k]
                    ),
                    "predicted_index_f": f"{predicted:.4f}",
                    "actual_index_f": f"{actual:.4f}",
                    "abs_error_f": f"{error:.4f}",
                    "same_one_degree_bucket": same_bucket,
                })

        hour += timedelta(hours=1)

    report = {}

    for horizon in HORIZONS:
        group = [
            r for r in predictions
            if r["horizon_min"] == horizon
        ]

        if not group:
            continue

        n = len(group)

        mae = sum(
            float(r["abs_error_f"])
            for r in group
        ) / n

        accuracy = sum(
            r["same_one_degree_bucket"]
            for r in group
        ) / n

        report[horizon] = (
            n,
            mae,
            accuracy,
        )

    return targets, predictions, report


def save_and_report(rows):
    if not rows:
        raise RuntimeError(
            "No usable minutes returned. "
            "Check API historical coverage."
        )

    write_rows(
        OUT / "miami_index_minutes.csv",
        FIELDS,
        (rows[key] for key in sorted(rows)),
    )

    targets, predictions, report = (
        build_targets_and_baseline(rows)
    )

    write_rows(
        OUT / "miami_hourly_targets.csv",
        [
            "settlement_hour_utc",
            "settlement_hour_et",
            "settlement_minute_utc",
            "settlement_index_f",
            "settlement_status",
        ],
        targets,
    )

    write_rows(
        OUT / "miami_persistence_baseline.csv",
        [
            "settlement_hour_utc",
            "horizon_min",
            "decision_time_utc",
            "latest_available_minute_utc",
            "predicted_index_f",
            "actual_index_f",
            "abs_error_f",
            "same_one_degree_bucket",
        ],
        predictions,
    )

    print()
    print(f"Stored {len(rows):,} minute records")
    print(f"Hourly settlement targets: {len(targets)}")
    print(f"Output folder: {OUT}")

    print()
    print("PERSISTENCE PREDICTION BASELINE")
    print("Assumed finalization: minute + 6 minutes")
    print()
    print("Horizon     N       MAE (F)     Same 1F bucket")
    print("-" * 54)

    for horizon in HORIZONS:
        if horizon in report:
            n, mae, accuracy = report[horizon]

            print(
                f"{horizon:>3} min  "
                f"{n:>5}       "
                f"{mae:>5.3f}          "
                f"{accuracy:>6.1%}"
            )
        else:
            print(
                f"{horizon:>3} min      0"
                "          --              --"
            )

    print()
    print(
        "NOTE: Historical API records do not prove "
        "actual real-time receipt times."
    )
    print(
        "This is a temperature baseline, "
        "not a price-adjusted trading backtest."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of history to request",
    )

    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=6,
        help="Hours per historical API request",
    )

    parser.add_argument(
        "--poll",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Continue collecting periodically",
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing CSV without API requests",
    )

    args = parser.parse_args()

    if not 1 <= args.days <= 180:
        parser.error("--days must be 1–180")

    if not 1 <= args.chunk_hours <= 24:
        parser.error("--chunk-hours must be 1–24")

    if args.poll and args.poll < 30:
        parser.error("--poll must be at least 30 seconds")

    rows = read_rows(
        OUT / "miami_index_minutes.csv"
    )

    if not args.evaluate_only:
        added, updated, batches = collect(
            args.days,
            args.chunk_hours,
            rows,
        )

        print(
            f"Backfill: {batches} requests, "
            f"{added} new, {updated} refreshed"
        )

    save_and_report(rows)

    if args.poll:
        print(
            f"\nPolling every {args.poll} seconds. "
            "Ctrl+C to stop."
        )

        while True:
            time.sleep(args.poll)

            try:
                added, updated, _ = collect(
                    args.days,
                    args.chunk_hours,
                    rows,
                    live=True,
                )

                print(
                    f"Live: {added} new, "
                    f"{updated} refreshed"
                )

                save_and_report(rows)

            except RuntimeError as exc:
                print(
                    f"Poll error: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nStopped")

    except (RuntimeError, ValueError, OSError) as exc:
        sys.exit(f"ERROR: {exc}")
