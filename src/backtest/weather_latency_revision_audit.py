"""Audit Miami weather-index publication and revisions."""
import argparse
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def utc(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Naive timestamp: {value}")
    return dt.astimezone(timezone.utc)


def audit(rows, start, end):
    start_dt, end_dt = utc(start), utc(end)
    grouped = defaultdict(list)

    for first_seen, index_ms, status, value, station_last_ms in rows:
        grouped[int(index_ms)].append(
            (utc(first_seen), str(status), value, station_last_ms)
        )

    print(f"Versions in query window: {len(rows)}")
    print(f"Status counts: {dict(Counter(str(x[2]) for x in rows))}")
    print(f"Distinct index minutes: {len(grouped)}")

    first_value_delays = []
    new_minutes = 0
    usable = 0

    for index_ms in sorted(grouped):
        index_dt = datetime.fromtimestamp(index_ms / 1000, tz=timezone.utc)
        records = sorted(grouped[index_ms], key=lambda x: x[0])

        if not start_dt <= index_dt <= end_dt:
            continue

        new_minutes += 1
        first_seen, initial_status, initial_value, _ = records[0]
        numerical = [r for r in records if r[2] is not None]

        print(f"\nINDEX MINUTE {index_dt:%H:%M}Z | {len(records)} observed versions")
        print(
            f"  First observed: {first_seen:%H:%M:%S.%f}Z; "
            f"status={initial_status}, value={initial_value}"
        )

        if numerical:
            usable += 1
            first_value = numerical[0]
            delay = (first_value[0] - index_dt).total_seconds()
            first_value_delays.append(delay)
            print(
                f"  First numerical value: {first_value[2]} "
                f"at {first_value[0]:%H:%M:%S.%f}Z "
                f"(+{delay:.1f}s from index minute)"
            )
        else:
            print("  First numerical value: NOT OBSERVED during collection")

        for seen, status, value, receipt_ms in records:
            receipt_age = (
                seen.timestamp() - int(receipt_ms) / 1000
                if receipt_ms is not None else None
            )
            lag = f"{receipt_age:.1f}s" if receipt_age is not None else "unknown"

            print(
                f"    {seen:%H:%M:%S.%f}Z | {status:12} "
                f"| value={str(value):>8} | station receipt age={lag}"
            )

    print("\nSUMMARY FOR NEW INDEX MINUTES")
    print(f"New minutes: {new_minutes}")
    print(f"With numerical value observed: {usable}")
    print(f"Without numerical value: {new_minutes - usable}")

    if first_value_delays:
        print(
            f"First usable value delay: "
            f"median={median(first_value_delays):.1f}s, "
            f"range={min(first_value_delays):.1f}"
            f" to {max(first_value_delays):.1f}s"
        )

    print("Note: 20-second polling limits publication-time precision.")
    print("Later revisions may fall outside this collection window.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-10-08T20:21:22Z")
    parser.add_argument("--end", default="2026-10-08T20:31:52Z")
    args = parser.parse_args()

    if utc(args.start) >= utc(args.end):
        parser.error("--start must precede --end")

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    if not os.getenv("PREDICTION_DATABASE_URL"):
        raise RuntimeError("PREDICTION_DATABASE_URL required")

    from src.database.db import get_connection

    with get_connection() as db:
        rows = db.execute("""
            SELECT first_seen_utc, index_minute_ms, status,
                   value_f, station_receipt_last_ms
            FROM hourly_weather_index_versions
            WHERE city = ?
              AND first_seen_utc >= ?
              AND first_seen_utc <= ?
            ORDER BY index_minute_ms, first_seen_utc
        """, (
            "miami",
            utc(args.start).isoformat(timespec="milliseconds"),
            utc(args.end).isoformat(timespec="milliseconds")
        )).fetchall()

    audit(rows, args.start, args.end)


if __name__ == "__main__":
    main()
