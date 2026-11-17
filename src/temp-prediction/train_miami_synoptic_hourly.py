
#!/usr/bin/env python3
"""Backfill Synoptic network-1 METAR/HF-METAR and test Miami hourly forecasts.

Research only: read-only APIs, no orders. Historical observation timestamps are
NOT verified API receipt timestamps. Delays are configurable assumptions.
"""

import argparse
import csv
import json
import math
import runpy
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
DATA = ROOT / "data" / "temp-prediction" / "miami_hourly_index"
READER = ROOT / "src" / "temp-prediction" / "synoptic_city_temp.py"
SYN_CSV = DATA / "miami_iem_observations.csv"
TEST_CSV = DATA / "miami_synoptic_model_test.csv"

STATIONS = ("KMIA", "KOPF", "KFLL", "KFXE", "KPMP")
HORIZONS = (5, 10, 15, 20)
UTC = timezone.utc
ET = ZoneInfo("America/New_York")


def parse_time(s):
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone: {s}")
    return dt.astimezone(UTC)


def stamped(t):
    return t.astimezone(UTC).isoformat(timespec="seconds")


def load_csv(path):
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def request_synoptic(token, stations, start, end):
    url = "https://api.synopticdata.com/v2/stations/timeseries"

    query = urlencode({
        "token": token,
        "stid": ",".join(stations),
        "network": 1,
        "start": start.strftime("%Y%m%d%H%M"),
        "end": end.strftime("%Y%m%d%H%M"),
        "vars": "air_temp",
        "hfmetars": 1,
        "units": "metric",
        "obtimezone": "UTC",
        "showemptystations": 1,
    })

    req = Request(
        url + "?" + query,
        headers={"Accept": "application/json"},
    )

    for attempt in range(4):
        try:
            with urlopen(req, timeout=45) as response:
                payload = json.load(response)

            summary = payload.get("SUMMARY") or {}
            code = str(summary.get("RESPONSE_CODE"))

            if code not in ("1", "2"):
                raise RuntimeError(
                    "Synoptic response " + code + ": " +
                    str(
                        summary.get("RESPONSE_MESSAGE", "unknown")
                    ).replace(token, "[REDACTED]")
                )

            return payload.get("STATION") or []

        except HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"Synoptic HTTP {e.code}") from None

        except (URLError, TimeoutError, OSError) as e:
            if attempt == 3:
                raise RuntimeError(
                    f"Synoptic request failed: {type(e).__name__}"
                ) from e

        time.sleep(2 ** attempt)

    raise RuntimeError("Synoptic retries exhausted")


def backfill(targets, chunk_hours):
    # Reuse the parser from our existing Synoptic-only reader.
    if not READER.exists():
        raise SystemExit(f"Missing Synoptic reader: {READER}")

    reader = runpy.run_path(str(READER))
    token = reader["read_token"]()

    if not token:
        raise SystemExit(
            "SYNOPTIC_API_TOKEN missing from environment or .env"
        )

    extract = reader["extract_observations"]

    start = targets[0][0] - timedelta(hours=3)
    end = min(
        targets[-1][0],
        datetime.now(UTC) - timedelta(minutes=5),
    )

    start = start.replace(second=0, microsecond=0)
    end = end.replace(second=0, microsecond=0)

    cache = {}

    if SYN_CSV.exists():
        for row in load_csv(SYN_CSV):
            cache[(row["stid"], row["obs_utc"])] = row

    cursor = start
    batches = 0
    new = 0

    while cursor < end:
        stop = min(
            cursor + timedelta(hours=chunk_hours),
            end,
        )

        stations = request_synoptic(
            token, STATIONS, cursor, stop
        )

        added = 0

        for station in stations:
            sid = str(
                station.get("STID", "")
            ).upper()

            if sid not in STATIONS:
                continue

            if str(station.get("MNET_ID")) != "1":
                continue

            for observed, temp_c in extract(station):
                if not cursor <= observed <= stop:
                    continue

                key = (sid, stamped(observed))

                if key not in cache:
                    added += 1

                cache[key] = {
                    "stid": sid,
                    "obs_utc": stamped(observed),
                    "temp_f": f"{temp_c * 1.8 + 32:.4f}",
                    "retrieved_at_utc": stamped(
                        datetime.now(UTC)
                    ),
                }

        batches += 1
        new += added

        print(
            f"{stamped(cursor)} -> {stamped(stop)}: "
            f"{added} new observations",
            flush=True,
        )

        cursor = stop
        time.sleep(0.15)

    write_csv(
        SYN_CSV,
        [
            "stid",
            "obs_utc",
            "temp_f",
            "retrieved_at_utc",
        ],
        [
            cache[k]
            for k in sorted(
                cache,
                key=lambda k: (k[1], k[0]),
            )
        ],
    )

    print(
        f"Synoptic: {batches} requests, "
        f"{new:,} new; {len(cache):,} stored -> {SYN_CSV}"
    )


def make_series(data):
    records = []

    for row in data:
        try:
            t = parse_time(
                row["obs_utc"]
                if "obs_utc" in row
                else row["minute_utc"]
            )

            v = float(
                row["temp_f"]
                if "temp_f" in row
                else row["index_f"]
            )

            if math.isfinite(v):
                records.append((t, v))

        except (KeyError, TypeError, ValueError):
            continue

    uniq = dict(records)
    values = sorted(uniq.items())

    return (
        [p[0] for p in values],
        [p[1] for p in values],
    )


def at(series, cutoff, max_age):
    times, values = series
    idx = bisect_right(times, cutoff) - 1

    if idx < 0:
        return None

    if cutoff - times[idx] > timedelta(minutes=max_age):
        return None

    return times[idx], values[idx]


def collect_targets():
    result = {}

    for row in load_csv(
        DATA / "miami_hourly_targets.csv"
    ):
        try:
            hour = parse_time(
                row["settlement_hour_utc"]
            )
            source_minute = parse_time(
                row["settlement_minute_utc"]
            )
            y = float(row["settlement_index_f"])

            if (
                math.isfinite(y)
                and timedelta(0)
                <= hour - source_minute
                <= timedelta(minutes=2)
            ):
                result[hour] = y

        except (KeyError, ValueError):
            continue

    return sorted(result.items())


def average_trend_features(series, asof, reference):
    """Historical five-station averages, never after asof."""
    import numpy as np

    current = at(series, asof, 10)

    if current is None:
        return [np.nan] * 7

    observed_at, current_avg = current

    result = [
        current_avg - reference,
        (asof - observed_at).total_seconds() / 60,
    ]

    for lag in (5, 10, 15, 20, 30):
        previous = at(
            series,
            observed_at - timedelta(minutes=lag),
            1,
        )
        result.append(
            current_avg - previous[1]
            if previous else np.nan
        )

    return result


def features(
    hour,
    horizon,
    index,
    syn,
    average_series,
    delay_index,
    delay_syn,
):
    import numpy as np

    decision = hour - timedelta(minutes=horizon)

    index_asof = decision - timedelta(
        minutes=delay_index
    )
    syn_asof = decision - timedelta(
        minutes=delay_syn
    )

    cur = at(index, index_asof, 20)

    if not cur:
        return None

    base = cur[1]
    local = hour.astimezone(ET)
    phase = 2 * math.pi * local.hour / 24

    shared = [
        base,
        math.sin(phase),
        math.cos(phase),
        (index_asof - cur[0]).total_seconds() / 60,
    ]

    # Recent official index changes.
    for lag in (5, 10, 20, 30):
        past = at(
            index,
            index_asof - timedelta(minutes=lag),
            4,
        )

        shared.append(
            base - past[1]
            if past else np.nan
        )

    extra = []
    available = []

    # Independent Synoptic station features.
    for sid in STATIONS:
        series = syn.get(sid, ([], []))
        recent = at(series, syn_asof, 35)

        if recent:
            available.append(recent[1])
            extra += [
                recent[1] - base,
                (
                    syn_asof - recent[0]
                ).total_seconds() / 60,
            ]
        else:
            extra += [np.nan, np.nan]

        for lag in (10, 20, 30):
            prev = at(
                series,
                syn_asof - timedelta(minutes=lag),
                20,
            )

            extra.append(
                recent[1] - prev[1]
                if recent and prev
                else np.nan
            )

    if len(available) < 3:
        return None

    extra += [
        len(available),
        max(available) - min(available),
        sum(available) / len(available) - base,
    ]

    return (
        base,
        shared,
        shared + extra,
        shared + extra + average_trend_features(
            average_series, syn_asof, base
        ),
        len(available),
        decision,
        syn_asof,
    )


def score(y, pred):
    import numpy as np

    mae = float(
        np.mean(np.abs(y - pred))
    )
    bucket = float(
        np.mean(
            np.floor(y) == np.floor(pred)
        )
    )

    return mae, bucket


def run_model(
    targets,
    index,
    syn,
    average_series,
    delay_index,
    delay_syn,
):
    import numpy as np

    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
    )

    hours = [
        hour for hour, _ in targets
    ]

    if len(hours) < 150:
        raise SystemExit(
            f"Only {len(hours)} clean hourly targets; "
            "need >=150"
        )

    cut1 = hours[int(len(hours) * 0.60)]
    cut2 = hours[int(len(hours) * 0.80)]

    print(
        "Chronological train/val/test cutoffs:",
        stamped(cut1),
        "/",
        stamped(cut2),
    )

    all_output = []

    def new_model():
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=10.0,
            random_state=42,
        )

    for horizon in HORIZONS:
        rows = []

        for hour, actual in targets:
            f = features(
                hour,
                horizon,
                index,
                syn,
                average_series,
                delay_index,
                delay_syn,
            )

            if f:
                (
                    base, small, large, with_avg_trends,
                    count, decision, asof
                ) = f

                rows.append((
                    hour, actual, base,
                    small, large, with_avg_trends,
                    count, decision, asof,
                ))

        # Chronological split with one-hour embargo.
        train = [
            r for r in rows
            if r[0] < cut1 - timedelta(hours=1)
        ]

        valid = [
            r for r in rows
            if (
                cut1 + timedelta(hours=1)
                <= r[0]
                < cut2 - timedelta(hours=1)
            )
        ]

        test = [
            r for r in rows
            if r[0] >= cut2 + timedelta(hours=1)
        ]

        print(
            f"\n{horizon} min: "
            f"train={len(train)}, "
            f"valid={len(valid)}, "
            f"test={len(test)}"
        )

        if min(
            len(train),
            len(valid),
            len(test),
        ) < 25:
            print(
                "  Insufficient matched hours; skipping"
            )
            continue

        def arrays(items, which):
            return (
                np.asarray(
                    [r[which] for r in items],
                    dtype=float,
                ),
                np.asarray(
                    [r[1] for r in items],
                    dtype=float,
                ),
                np.asarray(
                    [r[2] for r in items],
                    dtype=float,
                ),
            )

        scores = {
            "persistence": float(
                np.mean([
                    abs(r[1] - r[2])
                    for r in valid
                ])
            )
        }

        for name, column in (
            ("index_only", 3),
            ("index_plus_synoptic", 4),
            ("index_plus_avg_trends", 5),
        ):
            Xtr, ytr, btr = arrays(
                train, column
            )
            Xv, yv, bv = arrays(
                valid, column
            )

            fitted = new_model().fit(
                Xtr,
                ytr - btr,
            )

            scores[name] = float(
                np.mean(
                    abs(
                        yv - (
                            bv + fitted.predict(Xv)
                        )
                    )
                )
            )

        chosen = min(
            scores,
            key=scores.get,
        )

        print(
            "  Validation MAE:",
            ", ".join(
                f"{k}={v:.3f}F"
                for k, v in scores.items()
            ),
        )

        print("  Selected:", chosen)

        _, yt, bt = arrays(test, 3)

        if chosen == "persistence":
            predicted = bt.copy()

        else:
            column = {
                "index_only": 3,
                "index_plus_synoptic": 4,
                "index_plus_avg_trends": 5,
            }[chosen]

            combined = train + valid

            Xcv, ycv, bcv = arrays(
                combined, column
            )
            Xt, _, _ = arrays(
                test, column
            )

            fitted = new_model().fit(
                Xcv,
                ycv - bcv,
            )

            predicted = (
                bt + fitted.predict(Xt)
            )

        def report(label, indices):
            if not indices:
                return

            subset = np.asarray(
                indices,
                dtype=int,
            )

            ys = yt[subset]
            b = bt[subset]
            p = predicted[subset]

            # Hypothetical nearest integer threshold.
            strike = np.floor(b + 0.5)

            b_accuracy = np.mean(
                (b >= strike) == (ys >= strike)
            )
            p_accuracy = np.mean(
                (p >= strike) == (ys >= strike)
            )

            print(
                f"  {label} n={len(subset)}: "
                f"persistence={score(ys, b)[0]:.3f}F, "
                f"selected={score(ys, p)[0]:.3f}F | "
                f"nearest-strike correct: "
                f"{b_accuracy:.1%} -> "
                f"{p_accuracy:.1%}"
            )

        report(
            "Previously inspected test",
            list(range(len(test))),
        )

        report(
            "Daytime 08-17 ET",
            [
                i for i, r in enumerate(test)
                if (
                    8 <= r[0].astimezone(ET).hour < 17
                )
            ],
        )

        for r, pred in zip(test, predicted):
            (
                hour, actual, base,
                _, _, _, count, decision, asof,
            ) = r

            strike = math.floor(base + 0.5)

            all_output.append({
                "hour_utc": stamped(hour),
                "horizon_min": horizon,
                "decision_utc": stamped(decision),
                "synoptic_asof_utc": stamped(asof),
                "stations_usable": count,
                "selected_model": chosen,
                "persistence_f": f"{base:.4f}",
                "forecast_f": f"{float(pred):.4f}",
                "actual_f": f"{actual:.4f}",
                "hypothetical_strike_f": strike,
                "persistence_yes": int(base >= strike),
                "forecast_yes": int(pred >= strike),
                "actual_yes": int(actual >= strike),
            })

    if not all_output:
        raise SystemExit(
            "No model results; inspect Synoptic "
            "coverage and delays"
        )

    write_csv(
        TEST_CSV,
        list(all_output[0]),
        all_output,
    )

    print(
        f"\nSaved {len(all_output):,} "
        f"test predictions to {TEST_CSV}"
    )

    print(
        "NOTICE: Historical availability is "
        "approximated, not proven by receipt logs."
    )
    print(
        "Strikes are hypothetical, not confirmed "
        "Kalshi market listings; no trades simulated."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--collect-only",
        action="store_true",
    )
    ap.add_argument(
        "--train-only",
        action="store_true",
    )
    ap.add_argument(
        "--chunk-hours",
        type=int,
        default=24,
    )
    ap.add_argument(
        "--synoptic-delay",
        type=int,
        default=15,
    )
    ap.add_argument(
        "--index-delay",
        type=int,
        default=6,
    )

    args = ap.parse_args()

    if args.collect_only and args.train_only:
        ap.error(
            "Choose only one of --collect-only "
            "or --train-only"
        )

    if not 1 <= args.chunk_hours <= 24:
        ap.error(
            "--chunk-hours must be between 1 and 24"
        )

    if (
        not 0 <= args.synoptic_delay <= 60
        or not 0 <= args.index_delay <= 30
    ):
        ap.error("Invalid delay")

    targets = collect_targets()

    if not targets:
        raise SystemExit(
            "No Miami hourly targets available"
        )

    if not args.train_only:
        backfill(
            targets,
            args.chunk_hours,
        )

    if args.collect_only:
        return

    index = make_series(
        load_csv(
            DATA / "miami_index_minutes.csv"
        )
    )

    syn_data = load_csv(SYN_CSV)

    syn = {
        sid: make_series([
            r for r in syn_data
            if r["stid"] == sid
        ])
        for sid in STATIONS
    }

    avg_rows = load_csv(
        DATA / "miami_iem_average_5m.csv"
    )

    average_series = make_series([
        {
            "obs_utc": row["minute_utc"],
            "temp_f": row["average_f"],
        }
        for row in avg_rows
        if row.get("average_f", "").strip()
        and row.get("stations_used") == "5"
    ])

    print(
        "Complete IEM average points:",
        len(average_series[0]),
    )

    print(
        "IEM station observation counts:",
        ", ".join(
            f"{sid}={len(syn[sid][0]):,}"
            for sid in STATIONS
        ),
    )

    print(
        "Assumed availability delays: "
        f"Synoptic +{args.synoptic_delay}min, "
        f"Kalshi index +{args.index_delay}min"
    )

    run_model(
        targets,
        index,
        syn,
        average_series,
        args.index_delay,
        args.synoptic_delay,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
