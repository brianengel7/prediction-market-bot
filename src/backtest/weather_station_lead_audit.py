
"""Compare preliminary Kalshi station readings to the published index.

Research only. Reads existing Supabase observations.
Does not place orders or modify database records.
"""

import argparse
import json
import math
import os

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


def as_utc(value):
    result = datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )

    if result.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")

    return result.astimezone(timezone.utc)


def instant(ms):
    return datetime.fromtimestamp(
        int(ms) / 1000,
        tz=timezone.utc
    )


def selected_calibration(configs, minute, desired_version):
    eligible = [
        c for c in configs
        if int(c["effective_at_ms"]) <= minute
    ]

    if desired_version:
        matched = [
            c for c in eligible
            if c.get("config_version") == desired_version
        ]
        if matched:
            return max(
                matched,
                key=lambda c: int(c["effective_at_ms"])
            )

    return (
        max(eligible, key=lambda c: int(c["effective_at_ms"]))
        if eligible else None
    )


def modeled_temp(point, calibration, include_pending):
    """Experimental weighted, offset-adjusted index proxy.

    Assumes a minimum of four stations and 80% base weight.
    Does not reproduce every Kalshi QC/fallback rule.
    """
    if not calibration:
        return None, []

    members = {
        s["station_id"]: s
        for s in calibration.get("stations", [])
    }

    included = []
    seen = set()

    for station in point.get("stations", []):
        if not isinstance(station, dict):
            continue

        sid = station.get("station_id")

        if sid in seen or sid not in members:
            continue

        code = station.get("code")
        if code != "ok" and not (
            include_pending and code == "pending"
        ):
            continue

        try:
            temp = float(station["temp_f"])
            weight = float(members[sid]["weight"])
            offset = float(members[sid]["offset_c"])
        except (KeyError, ValueError, TypeError):
            continue

        if (
            not all(math.isfinite(x) for x in (temp, weight, offset))
            or weight <= 0
        ):
            continue

        included.append(
            (sid, temp, weight, offset, station)
        )
        seen.add(sid)

    total_weight = sum(x[2] for x in included)

    if len(included) < 4 or total_weight < 0.8 - 1e-9:
        return None, included

    reference = float(calibration["city_reference_c"])

    center_c = (
        sum(
            ((temp - 32) * 5 / 9 - offset) * weight
            for _, temp, weight, offset, _ in included
        ) / total_weight
        + reference
    )

    return round(center_c * 9 / 5 + 32, 2), included


def audit(records, configs, verbose=False):
    grouped = defaultdict(list)

    for minute, first_seen, config_version, raw in records:
        point = (
            json.loads(raw)
            if isinstance(raw, str)
            else raw
        )

        if int(point.get("t", -1)) != int(minute):
            raise ValueError(
                f"Stored minute disagrees with payload for {minute}"
            )

        grouped[int(minute)].append(
            (as_utc(first_seen), config_version, point)
        )

    compared = 0
    leads = []
    errs = []
    absent = 0
    incomplete = 0

    print(
        f"Index minutes: {len(grouped)} | "
        f"stored versions: {len(records)}"
    )
    print(
        "PRELIMINARY STATION PROXY "
        "VS FIRST PUBLISHED NUMERIC INDEX"
    )

    for minute, versions in sorted(grouped.items()):
        versions.sort(key=lambda x: x[0])
        minute_time = instant(minute)

        cal = selected_calibration(
            configs, minute, versions[0][1]
        )

        if cal is None:
            absent += 1
            print(
                f"{minute_time:%H:%M}Z "
                "SKIP: missing historical calibration"
            )
            continue

        first_numeric = next(
            (
                (t, point)
                for t, _, point in versions
                if point.get("v") is not None
            ),
            None
        )

        earliest = None

        for first_seen, _, point in versions:
            if (
                first_numeric
                and first_seen >= first_numeric[0]
            ):
                break

            candidate, stations = modeled_temp(
                point, cal, include_pending=True
            )

            if candidate is not None:
                earliest = (
                    first_seen, candidate, stations, point
                )
                break

        if earliest is None:
            incomplete += 1
            suffix = (
                "; finalized reading present"
                if first_numeric
                else "; no final reading"
            )
            print(
                f"{minute_time:%H:%M}Z "
                "No 4-station pre-final proxy observed" + suffix
            )
            continue

        first_seen, estimate, used, source = earliest

        pending = sum(
            x[4].get("code") == "pending"
            for x in used
        )

        ok_only, _ = modeled_temp(
            source, cal, include_pending=False
        )

        if first_numeric:
            t_final, finished = first_numeric

            lead = (
                t_final - first_seen
            ).total_seconds()

            error = estimate - float(finished["v"])

            leads.append(lead)
            errs.append(abs(error))
            compared += 1

            print(
                f"{minute_time:%H:%M}Z "
                f"proxy={estimate:.2f}F "
                f"final={float(finished['v']):.2f}F "
                f"error={error:+.2f}F "
                f"lead={lead:.1f}s "
                f"pending={pending}/{len(used)} "
                f"ok_only={ok_only}"
            )
        else:
            print(
                f"{minute_time:%H:%M}Z "
                f"proxy={estimate:.2f}F "
                f"pending={pending}/{len(used)} "
                "final=NOT OBSERVED"
            )

        if verbose:
            for sid, temp, _, _, station in used:
                receipt = station.get("received_at_ms")

                age = (
                    (first_seen - instant(receipt))
                    .total_seconds()
                    if receipt is not None
                    else None
                )

                age_text = (
                    f"{age:.1f}s"
                    if age is not None
                    else "unknown"
                )

                print(
                    f"    {sid:7s} {temp:5.1f}F "
                    f"code={station.get('code')} "
                    f"source={station.get('source')} "
                    f"Kalshi-receipt-age={age_text}"
                )

    print("\nSUMMARY")
    print(
        "Compared before first numeric publication:",
        compared
    )
    print(
        f"Insufficient early station data: {incomplete} "
        f"| Missing calibration: {absent}"
    )

    if compared:
        print(
            f"Median absolute error: {median(errs):.2f}F "
            f"| max absolute error: {max(errs):.2f}F "
            f"| median observed lead: {median(leads):.1f}s"
        )

    print(
        "CAUTION: This is Kalshi-origin preliminary data, "
        "not upstream access."
    )
    print(
        "Lead measures provisional vs first observed numerical "
        "index publication."
    )
    print(
        "It does NOT establish a tradable edge "
        "or validate pending stations."
    )


def self_test():
    minute = 1791480000000

    cal = {
        "config_version": "cal1",
        "effective_at_ms": minute - 1,
        "city_reference_c": 0,
        "stations": [
            {
                "station_id": f"S{i}",
                "weight": 0.2,
                "offset_c": 0
            }
            for i in range(5)
        ]
    }

    early = {
        "t": minute,
        "status": "incomplete",
        "v": None,
        "stations": [
            {
                "station_id": f"S{i}",
                "code": "pending",
                "temp_f": 86.0
            }
            for i in range(4)
        ]
    }

    final = {
        "t": minute,
        "status": "normal",
        "v": 86.0,
        "stations": [
            {
                "station_id": f"S{i}",
                "code": "ok",
                "temp_f": 86.0
            }
            for i in range(5)
        ]
    }

    assert modeled_temp(early, cal, True)[0] == 86.0
    assert modeled_temp(early, cal, False)[0] is None
    assert modeled_temp(final, cal, False)[0] == 86.0
    assert selected_calibration(
        [cal], minute, "cal1"
    ) == cal

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()

    with redirect_stdout(buf):
        audit([
            (
                minute,
                "2026-10-08T20:02:00Z",
                "cal1",
                json.dumps(early)
            ),
            (
                minute,
                "2026-10-08T20:05:00Z",
                "cal1",
                json.dumps(final)
            )
        ], [cal])

    assert "lead=180.0s" in buf.getvalue()
    assert (
        "Compared before first numeric publication: 1"
        in buf.getvalue()
    )

    print(
        "PASS: calibration selection, station quorum, "
        "pending treatment, historical comparison"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--city", default="miami")
    p.add_argument(
        "--start", default="2026-10-08T20:21:22Z"
    )
    p.add_argument(
        "--end", default="2026-10-08T20:31:52Z"
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--self-test", action="store_true")

    args = p.parse_args()

    if args.self_test:
        return self_test()

    start = as_utc(args.start)
    end = as_utc(args.end)

    if start >= end:
        p.error("--start must precede --end")

    from dotenv import load_dotenv

    load_dotenv(
        dotenv_path=Path(__file__).resolve().parents[2] / ".env"
    )

    if not os.environ.get("PREDICTION_DATABASE_URL"):
        raise RuntimeError(
            "PREDICTION_DATABASE_URL required; "
            "refusing local database fallback"
        )

    from src.database.db import get_connection

    with get_connection() as con:
        configs = [
            json.loads(r[0])
            for r in con.execute("""
                SELECT raw_json
                FROM hourly_weather_index_calibrations
                WHERE city = ?
                ORDER BY effective_at_ms
            """, (args.city,)).fetchall()
        ]

        versions = con.execute("""
            SELECT index_minute_ms, first_seen_utc,
                   config_version, raw_json
            FROM hourly_weather_index_versions
            WHERE city = ?
              AND first_seen_utc >= ?
              AND first_seen_utc <= ?
            ORDER BY index_minute_ms, first_seen_utc
        """, (
            args.city,
            start.isoformat(timespec="milliseconds"),
            end.isoformat(timespec="milliseconds")
        )).fetchall()

    audit(versions, configs, args.verbose)


if __name__ == "__main__":
    main()
