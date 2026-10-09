"""Read-only weather latency diagnostics."""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def utc(text):
    d = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if not d.tzinfo:
        raise ValueError("Timestamp needs a UTC offset")
    return d.astimezone(timezone.utc)


def report(versions, polls, books, start, end):
    begin, finish = utc(start), utc(end)

    print("\nWEATHER LATENCY ANALYSIS")
    print("=" * 65)
    print("Weather index versions:", len(versions))
    print("Weather polls:", len(polls))
    print("Order-book samples:", len(books))
    print("Weather poll statuses:", dict(Counter(r[0] for r in polls)))

    minutes = defaultdict(list)
    for seen, minute, status, value, station_last in versions:
        minutes[int(minute)].append(
            (utc(seen), status, value, station_last)
        )

    new_minutes, lag, station_lag = [], [], []
    value_changes = status_changes = 0

    for m, entries in sorted(minutes.items()):
        entries.sort(key=lambda r: r[0])
        value_changes += len({str(r[2]) for r in entries}) > 1
        status_changes += len({r[1] for r in entries}) > 1

        observed, status, value, station_last = entries[0]
        minute_time = datetime.fromtimestamp(m / 1000, tz=timezone.utc)

        if begin <= minute_time <= finish:
            new_minutes.append((minute_time, observed, status, value))
            lag.append((observed - minute_time).total_seconds())

            if station_last is not None:
                station_lag.append(
                    observed.timestamp() - int(station_last) / 1000
                )

    print("\nINDEX ANALYSIS")
    print("-" * 65)
    print("Distinct index minutes:", len(minutes))
    print("Minutes with value changes:", value_changes)
    print("Minutes with status changes:", status_changes)

    for label, values in (
        ("Index minute to first seen", lag),
        ("Station receipt to first seen", station_lag),
    ):
        if values:
            print(
                f"{label}: median={median(values):.1f}s, "
                f"range={min(values):.1f} to {max(values):.1f}s"
            )
        else:
            print(f"{label}: no measurements")

    history = defaultdict(list)

    for stamp, ticker, payload in books:
        row = json.loads(payload) if isinstance(payload, str) else payload
        prices = tuple(
            row.get(k)[0] if row.get(k) else None
            for k in ("yes_bid", "yes_ask", "no_bid", "no_ask")
        )
        history[ticker].append((utc(stamp), prices))

    changed = []
    skipped_gaps = 0

    for ticker, observations in history.items():
        observations.sort()

        for (prior, old), (current, new) in zip(
            observations, observations[1:]
        ):
            if old != new:
                if (current - prior).total_seconds() <= 30:
                    changed.append((current, ticker))
                else:
                    skipped_gaps += 1

    changed.sort()

    print("\nORDER-BOOK ANALYSIS")
    print("-" * 65)
    print("Book tickers:", len(history))
    print("Sampled price changes:", len(changed))
    print("Excluded price changes across gaps >30s:", skipped_gaps)

    print("\nINDEX UPDATES VS ORDER-BOOK MOVEMENTS")
    print("-" * 65)

    for minute, observed, status, value in new_minutes:
        forward = [
            (d - observed).total_seconds()
            for d, ticker in changed
            if 0 <= (d - observed).total_seconds() <= 60
        ]

        delay = f"{min(forward):.1f}s" if forward else "none observed"

        print(
            f"index={minute:%H:%M}Z "
            f"seen={observed:%H:%M:%S}Z "
            f"value={value} "
            f"status={status} "
            f"book_changes_next_60s={len(forward)} "
            f"first={delay}"
        )

    print("\nLIMITATIONS")
    print("Timing overlap does not establish causation.")
    print("Book samples are event-driven, not fixed-interval.")
    print("Contract expiry matching and execution are not tested.")
    print("No independent earlier station feed is included.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-10-08T20:21:22Z")
    parser.add_argument("--end", default="2026-10-08T20:31:52Z")
    args = parser.parse_args()

    a = utc(args.start).isoformat(timespec="milliseconds")
    b = utc(args.end).isoformat(timespec="milliseconds")

    if utc(args.start) >= utc(args.end):
        parser.error("start must precede end")

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    if not os.environ.get("PREDICTION_DATABASE_URL"):
        raise RuntimeError("PREDICTION_DATABASE_URL required")

    from src.database.db import get_connection

    with get_connection() as db:
        versions = db.execute("""
            SELECT first_seen_utc, index_minute_ms,
                   status, value_f, station_receipt_last_ms
            FROM hourly_weather_index_versions
            WHERE city = ?
              AND first_seen_utc >= ?
              AND first_seen_utc <= ?
            ORDER BY first_seen_utc
        """, ("miami", a, b)).fetchall()

        polls = db.execute("""
            SELECT status
            FROM hourly_weather_index_polls
            WHERE city = ?
              AND request_started_utc >= ?
              AND request_started_utc <= ?
        """, ("miami", a, b)).fetchall()

        books = db.execute("""
            SELECT received_at_utc, ticker, payload_json
            FROM latency_scanner_events
            WHERE kind = 'BOOK_SAMPLE'
              AND ticker LIKE 'KXTEMPMIAH-%%'
              AND received_at_utc >= ?
              AND received_at_utc <= ?
            ORDER BY received_at_utc
        """, (a, b)).fetchall()

    report(versions, polls, books, args.start, args.end)


if __name__ == "__main__":
    main()

