
#!/usr/bin/env python3
"""NYC eight-station hourly Weather Index forecast. Research only."""

import argparse
import csv
import io
import json
import math
import os
import re
import time
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/temp-prediction/nyc_hourly_index"

STATIONS = (
    "KLGA", "KEWR", "KTEB", "KCDW",
    "KHPN", "KFRG", "KSMQ", "KISP",
)
HORIZONS = (5, 10, 15, 20)
KALSHI_DELAY = 6
SYNOPTIC_DELAY = 15

INDEX_FILE = DATA / "nyc_index_minutes.csv"
IEM_FILE = DATA / "nyc_iem_observations.csv"
AVERAGE_FILE = DATA / "nyc_iem_average_5m.csv"
MODEL_FILE = DATA / "nyc_hourly_models.joblib"

KALSHI_URL = (
    "https://external-api.kalshi.com/"
    "trade-api/v2/live_data/weather/nyc"
)
IEM_URL = (
    "https://mesonet.agron.iastate.edu/"
    "cgi-bin/request/asos.py"
)


def stamp(value):
    if isinstance(value, (float, int)):
        divisor = 1000 if value > 1e11 else 1
        return datetime.fromtimestamp(value / divisor, UTC)

    d = datetime.fromisoformat(
        str(value).replace("Z", "+00:00").replace(" ", "T")
    )
    return (
        d.replace(tzinfo=UTC)
        if d.tzinfo is None
        else d.astimezone(UTC)
    )


def label(t):
    return t.astimezone(ET).strftime(
        "%Y-%m-%d %I:%M %p %Z"
    )


def get_json(url, params):
    req = Request(
        url + "?" + urlencode(params),
        headers={
            "Accept": "application/json",
            "User-Agent": "NYCHourlyResearch/1.0",
        },
    )

    for n in range(4):
        try:
            with urlopen(req, timeout=45) as res:
                return json.load(res)

        except HTTPError as e:
            if n == 3 or e.code not in (
                429, 500, 502, 503, 504
            ):
                raise RuntimeError(
                    f"API HTTP {e.code}: {url.split('/')[2]}"
                ) from None

        except (URLError, TimeoutError, OSError) as e:
            if n == 3:
                raise RuntimeError(
                    f"API unavailable: {type(e).__name__}"
                ) from e

        time.sleep(2 ** n)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp, path)


def read_csv(path):
    if not path.exists():
        raise RuntimeError(
            f"Missing {path}. Run --setup first."
        )

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def kalshi_points(data):
    points = []

    for p in data.get("timeseries", []):
        try:
            t = stamp(p["t"]).replace(
                second=0, microsecond=0
            )
            v = float(p["v"])
            status = str(
                p.get("status", "")
            ).lower()

            if (
                math.isfinite(v)
                and status not in {
                    "missing", "pending",
                    "incomplete", "provisional",
                }
            ):
                points.append((t, v))

        except (
            TypeError, ValueError,
            KeyError, OverflowError
        ):
            continue

    return points


def collect_index(days):
    end = datetime.now(UTC) - timedelta(minutes=10)
    start = end - timedelta(days=days)

    existing = (
        {
            stamp(r["t"]): float(r["index_f"])
            for r in read_csv(INDEX_FILE)
        }
        if INDEX_FILE.exists()
        else {}
    )

    cursor = start

    while cursor < end:
        stop = min(
            cursor + timedelta(hours=6), end
        )

        payload = get_json(
            KALSHI_URL,
            {
                "from": int(cursor.timestamp() * 1000),
                "to": int(stop.timestamp() * 1000),
                "limit": 10000,
            },
        )

        pts = [
            (t, v)
            for t, v in kalshi_points(payload)
            if cursor <= t <= stop
        ]

        existing.update(pts)

        print(
            f"Kalshi {cursor:%Y-%m-%d %H:%M} UTC: "
            f"{len(pts)} minutes",
            flush=True,
        )

        cursor = stop + timedelta(milliseconds=1)
        time.sleep(0.12)

    write_csv(
        INDEX_FILE,
        ["t", "index_f"],
        [
            {"t": t.isoformat(), "index_f": v}
            for t, v in sorted(existing.items())
        ],
    )

    print(
        f"Saved {len(existing):,} NYC index minutes"
    )


def raw_metar_f(raw):
    raw = (raw or "").upper()

    m = re.search(
        r"(?:^|\s)T([01])(\d{3})[01]\d{3}(?=\s|$)",
        raw,
    )

    if m:
        c = (
            (-1 if m[1] == "1" else 1)
            * int(m[2]) / 10
        )
        return c * 1.8 + 32

    body = raw.split(" RMK ", 1)[0]

    for token in body.split():
        m = re.fullmatch(
            r"(M?)(\d{2})/(?:M?\d{2}|//)",
            token,
        )
        if m:
            c = (
                (-1 if m[1] else 1)
                * int(m[2])
            )
            return c * 1.8 + 32

    return None


def collect_iem():
    index = load_index()

    if not index:
        raise RuntimeError(
            "No Kalshi NYC index minutes available"
        )

    start = index[0][0] - timedelta(hours=3)
    end = index[-1][0] + timedelta(hours=1)
    rows = {}

    for sid in STATIONS:
        short = sid[1:]

        params = [
            ("station", short),
            ("data", "tmpf"),
            ("data", "metar"),
            ("sts", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("ets", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("missing", "M"),
            ("report_type", "1"),
            ("report_type", "3"),
            ("report_type", "4"),
        ]

        req = Request(
            IEM_URL + "?" + urlencode(params),
            headers={
                "User-Agent": "NYCHourlyResearch/1.0"
            },
        )

        with urlopen(req, timeout=120) as res:
            content = res.read().decode("utf-8-sig")

        lines = (
            s for s in io.StringIO(content)
            if s.strip()
            and not s.lstrip().startswith("#")
        )

        reader = csv.DictReader(lines)

        required = {
            "station", "valid", "tmpf", "metar"
        }

        if (
            not reader.fieldnames
            or not required.issubset(reader.fieldnames)
        ):
            raise RuntimeError(
                f"Unexpected IEM format for {sid}: "
                f"{content[:120]!r}"
            )

        count = 0

        for r in reader:
            if str(
                r.get("station", "")
            ).upper() not in (short, sid):
                continue

            try:
                t = stamp(r["valid"])
            except (ValueError, TypeError, KeyError):
                continue

            if not start <= t <= end:
                continue

            try:
                v = float(r["tmpf"])
                if not math.isfinite(v):
                    v = None
            except (ValueError, TypeError, KeyError):
                v = None

            if v is None:
                v = raw_metar_f(
                    r.get("metar", "")
                )

            if v is None or not -40 <= v <= 130:
                continue

            rows[(sid, t)] = v
            count += 1

        print(
            f"IEM {sid}: {count:,} temperature records",
            flush=True,
        )
        time.sleep(1.1)

    if not rows:
        raise RuntimeError(
            "IEM returned no usable NYC temperatures"
        )

    write_csv(
        IEM_FILE,
        ["stid", "obs_utc", "temp_f"],
        [
            {
                "stid": sid,
                "obs_utc": t.isoformat(),
                "temp_f": v,
            }
            for (sid, t), v in sorted(
                rows.items(),
                key=lambda x: (x[0][1], x[0][0]),
            )
        ],
    )

    print(
        f"Saved {len(rows):,} IEM observations"
    )

    _, station_series = load_weather()
    avg = make_averages(station_series, start, end)

    write_csv(
        AVERAGE_FILE,
        [
            "minute_utc",
            "average_f",
            "stations_used",
        ],
        [
            {
                "minute_utc": t.isoformat(),
                "average_f": f"{v:.4f}",
                "stations_used": n,
            }
            for t, v, n in avg
        ],
    )

    complete = sum(
        n == 8 for _, _, n in avg
    )

    print(
        f"Five-minute NYC averages: {len(avg):,}, "
        f"complete 8/8: {complete:,}"
    )


def series(items):
    values = sorted(dict(items).items())
    return (
        [t for t, _ in values],
        [v for _, v in values],
    )


def pick(s, cutoff, age=35):
    ts, vs = s
    j = bisect_right(ts, cutoff) - 1

    if (
        j < 0
        or cutoff - ts[j] > timedelta(minutes=age)
    ):
        return None

    return ts[j], vs[j]


def load_index():
    return sorted(
        (
            stamp(r["t"]),
            float(r["index_f"]),
        )
        for r in read_csv(INDEX_FILE)
    )


def load_weather():
    data = read_csv(IEM_FILE)
    idx = {sid: [] for sid in STATIONS}

    for r in data:
        if r["stid"] in idx:
            idx[r["stid"]].append(
                (
                    stamp(r["obs_utc"]),
                    float(r["temp_f"]),
                )
            )

    return data, {
        sid: series(idx[sid])
        for sid in STATIONS
    }


def make_averages(stations, start, end):
    t = start.replace(
        second=0, microsecond=0
    )
    t += timedelta(
        minutes=(-t.minute) % 5
    )

    output = []

    while t <= end:
        recent = [
            pick(stations[sid], t, 15)
            for sid in STATIONS
        ]

        vals = [
            r[1] for r in recent
            if r is not None
        ]

        if len(vals) >= 7:
            output.append(
                (
                    t,
                    sum(vals) / len(vals),
                    len(vals),
                )
            )

        t += timedelta(minutes=5)

    return output


def targets_from_index(records):
    idx = series(records)

    start = records[0][0].replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)

    end = records[-1][0].replace(
        minute=0, second=0, microsecond=0
    )

    out = []
    t = start

    while t <= end:
        current = pick(idx, t, 2)

        if (
            current is not None
            and t - current[0]
            <= timedelta(minutes=2)
        ):
            out.append((t, current[1]))

        t += timedelta(hours=1)

    return out


def vector(
    hour, decision, index, stations, averages
):
    """Index, station, and averaged trend features."""
    import numpy as np

    index_asof = decision - timedelta(
        minutes=KALSHI_DELAY
    )
    syn_asof = decision - timedelta(
        minutes=SYNOPTIC_DELAY
    )

    cur = pick(index, index_asof, 20)

    if cur is None:
        return None

    base = cur[1]
    phase = (
        2 * math.pi
        * hour.astimezone(ET).hour / 24
    )

    shared = [
        base,
        math.sin(phase),
        math.cos(phase),
        (
            index_asof - cur[0]
        ).total_seconds() / 60,
    ]

    for lag in (5, 10, 20, 30):
        p = pick(
            index,
            index_asof - timedelta(minutes=lag),
            4,
        )
        shared.append(
            base - p[1] if p else np.nan
        )

    other = []
    values = []

    for sid in STATIONS:
        s = stations[sid]
        newest = pick(s, syn_asof, 35)

        if newest:
            values.append(newest[1])
            other += [
                newest[1] - base,
                (
                    syn_asof - newest[0]
                ).total_seconds() / 60,
            ]
        else:
            other += [np.nan, np.nan]

        for lag in (10, 20, 30):
            previous = pick(
                s,
                syn_asof - timedelta(minutes=lag),
                20,
            )

            other.append(
                newest[1] - previous[1]
                if newest and previous
                else np.nan
            )

    if len(values) < 7:
        return None

    other += [
        len(values),
        max(values) - min(values),
        sum(values) / len(values) - base,
    ]

    avg_cur = pick(
        averages, syn_asof, 10
    )

    avg_features = [np.nan] * 7

    if avg_cur:
        avg_features = [
            avg_cur[1] - base,
            (
                syn_asof - avg_cur[0]
            ).total_seconds() / 60,
        ]

        for lag in (5, 10, 15, 20, 30):
            past = pick(
                averages,
                avg_cur[0] - timedelta(minutes=lag),
                1,
            )
            avg_features.append(
                avg_cur[1] - past[1]
                if past else np.nan
            )

    return (
        base,
        (
            shared,
            shared + other,
            shared + other + avg_features,
        ),
        len(values),
        index_asof,
        syn_asof,
    )


def boost():
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
    )

    return HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_iter=150,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=10,
        random_state=42,
    )


def train():
    import numpy as np
    import joblib

    ix = load_index()
    _, stations = load_weather()

    avgs = make_averages(
        stations,
        ix[0][0],
        ix[-1][0],
    )

    averages = series([
        (t, v)
        for t, v, n in avgs
        if n >= 7
    ])

    targets = targets_from_index(ix)

    if len(targets) < 150:
        raise RuntimeError(
            f"Only {len(targets)} clean NYC "
            "settlement targets; need >=150"
        )

    cut1 = targets[
        int(0.6 * len(targets))
    ][0]
    cut2 = targets[
        int(0.8 * len(targets))
    ][0]

    index = series(ix)
    models = {}
    report = {}

    print(
        f"NYC historical hours: {len(targets):,}; "
        f"validation begins {label(cut1)}; "
        f"test begins {label(cut2)}"
    )

    for h in HORIZONS:
        samples = []

        for hour, actual in targets:
            f = vector(
                hour,
                hour - timedelta(minutes=h),
                index,
                stations,
                averages,
            )

            if f:
                base, variants, *_ = f
                samples.append(
                    (
                        hour,
                        actual,
                        base,
                        variants,
                    )
                )

        tr = [
            r for r in samples
            if r[0] < cut1 - timedelta(hours=1)
        ]
        va = [
            r for r in samples
            if (
                cut1 + timedelta(hours=1)
                <= r[0]
                < cut2 - timedelta(hours=1)
            )
        ]
        te = [
            r for r in samples
            if r[0] >= cut2 + timedelta(hours=1)
        ]

        if min(map(len, (tr, va, te))) < 25:
            print(
                f"{h}m: too few matched hours: "
                f"{len(tr)}/{len(va)}/{len(te)}"
            )
            continue

        scores = {
            "persistence": (
                sum(
                    abs(x[1] - x[2])
                    for x in va
                ) / len(va)
            )
        }

        kinds = (
            ("index_only", 0),
            ("index_stations", 1),
            ("index_avg_trends", 2),
        )

        for name, i in kinds:
            m = boost().fit(
                np.array([
                    x[3][i] for x in tr
                ]),
                np.array([
                    x[1] - x[2] for x in tr
                ]),
            )

            p = (
                np.array([x[2] for x in va])
                + m.predict(
                    np.array([
                        x[3][i] for x in va
                    ])
                )
            )

            scores[name] = float(
                np.mean(
                    np.abs(
                        p - [x[1] for x in va]
                    )
                )
            )

        chosen = min(
            scores, key=scores.get
        )
        col = dict(kinds).get(chosen)

        if chosen == "persistence":
            test_predictions = np.array([
                x[2] for x in te
            ])
        else:
            combined = tr + va

            fitted = boost().fit(
                np.array([
                    x[3][col] for x in combined
                ]),
                np.array([
                    x[1] - x[2] for x in combined
                ]),
            )

            test_predictions = (
                np.array([x[2] for x in te])
                + fitted.predict(
                    np.array([
                        x[3][col] for x in te
                    ])
                )
            )

        test_y = np.array([
            x[1] for x in te
        ])

        mae = float(
            np.mean(
                np.abs(
                    test_predictions - test_y
                )
            )
        )

        base_mae = float(
            np.mean(
                np.abs(
                    np.array([
                        x[2] for x in te
                    ]) - test_y
                )
            )
        )

        print(
            f"{h}m | train={len(tr)} "
            f"val={len(va)} test={len(te)} "
            f"| selected={chosen} "
            f"| test MAE {mae:.3f}F "
            f"vs persistence {base_mae:.3f}F"
        )

        # Refit using all historical data for live inference.
        if col is None:
            full = None
        else:
            full = boost().fit(
                np.array([
                    x[3][col] for x in samples
                ]),
                np.array([
                    x[1] - x[2] for x in samples
                ]),
            )

        models[h] = {
            "kind": chosen,
            "column": col,
            "model": full,
        }

        report[h] = {
            "test_mae": mae,
            "persistence_mae": base_mae,
            "test_n": len(te),
        }

    if len(models) != len(HORIZONS):
        raise RuntimeError(
            "Not all four NYC horizons were trainable; "
            "check station coverage"
        )

    DATA.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        {
            "models": models,
            "report": report,
            "inputs": [
                (
                    str(p),
                    p.stat().st_mtime_ns,
                )
                for p in (
                    INDEX_FILE,
                    IEM_FILE,
                )
            ],
        },
        MODEL_FILE,
    )

    print(
        f"Saved four NYC models: {MODEL_FILE}"
    )


def live_index():
    payload = get_json(
        KALSHI_URL,
        {
            "last_sec": 10800,
            "limit": 2000,
        },
    )
    return series(kalshi_points(payload))


def live_weather():
    import runpy

    reader = runpy.run_path(
        str(
            ROOT / "src/temp-prediction/"
            "synoptic_city_temp.py"
        )
    )

    token = reader["read_token"]()

    if not token:
        raise RuntimeError(
            "Missing SYNOPTIC_API_TOKEN in .env"
        )

    payload = reader["fetch_synoptic"](
        token,
        "nyc",
        180,
    )

    collection = {
        sid: [] for sid in STATIONS
    }

    for station in payload.get("STATION", []):
        sid = str(
            station.get("STID", "")
        ).upper()

        if (
            sid in collection
            and str(
                station.get("MNET_ID", "1")
            ) == "1"
        ):
            collection[sid] = [
                (t, c * 1.8 + 32)
                for t, c in reader[
                    "extract_observations"
                ](station)
            ]

    return {
        sid: series(collection[sid])
        for sid in STATIONS
    }


def forecast(saved):
    import numpy as np

    now = datetime.now(UTC)

    hour = now.replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)

    left = (
        hour - now
    ).total_seconds() / 60

    ix = live_index()

    print(
        f"\nNYC hourly Weather Index "
        f"| next settlement {label(hour)} "
        f"| {left:.1f} min left"
    )

    if left > 21:
        last = pick(
            ix,
            now - timedelta(
                minutes=KALSHI_DELAY
            ),
            20,
        )

        print(
            "Next trained horizon begins at :40 ET. "
            "Latest as-of index:",
            f"{last[1]:.2f}F"
            if last else "unavailable",
        )
        return

    h = min(
        HORIZONS,
        key=lambda h: abs(h - left),
    )

    try:
        stations = live_weather()

    except (RuntimeError, OSError) as exc:
        if (
            saved["models"][h]["kind"]
            == "persistence"
        ):
            reading = pick(
                ix,
                now - timedelta(
                    minutes=KALSHI_DELAY
                ),
                20,
            )

            if reading:
                print(
                    "PREDICTED NEXT NYC SETTLEMENT "
                    "(persistence): "
                    f"{reading[1]:.2f} F"
                )
                print(
                    "Synoptic feed unavailable:", exc
                )
                return
        raise

    for sid in STATIONS:
        recent = pick(
            stations[sid],
            now,
            25,
        )

        if recent:
            print(
                f"{sid}: {recent[1]:.2f} F, "
                f"observed {label(recent[0])}"
            )
        else:
            print(
                f"{sid}: MISSING/STALE"
            )

    avgs = make_averages(
        stations,
        now - timedelta(hours=2),
        now,
    )

    averages = series([
        (t, v)
        for t, v, n in avgs
        if n >= 7
    ])

    f = vector(
        hour,
        now,
        ix,
        stations,
        averages,
    )

    if not f:
        if (
            saved["models"][h]["kind"]
            == "persistence"
        ):
            reading = pick(
                ix,
                now - timedelta(
                    minutes=KALSHI_DELAY
                ),
                20,
            )

            if reading:
                print(
                    "PREDICTED NEXT NYC SETTLEMENT "
                    "(persistence): "
                    f"{reading[1]:.2f} F"
                )
                print(
                    "Station features unavailable; "
                    "index-only forecast"
                )
                return

        print(
            "NO FORECAST: missing or stale data "
            "for required seven-station features"
        )
        return

    base, vectors, n, index_asof, syn_asof = f
    winner = saved["models"][h]

    if winner["kind"] == "persistence":
        pred = base
    else:
        correction = float(
            winner["model"].predict(
                np.asarray(
                    [
                        vectors[
                            winner["column"]
                        ]
                    ],
                    dtype=float,
                )
            )[0]
        )
        pred = base + correction

    print(
        f"\nSelected: {h}-minute "
        f"{winner['kind']} model "
        f"| stations available as-of: {n}/8"
    )

    print(
        f"Index cutoff: {label(index_asof)} "
        f"| Synoptic cutoff: {label(syn_asof)}"
    )

    if abs(h - left) > 0.75:
        print(
            f"CAUTION: {left:.1f} minutes remain; "
            f"model trained for {h} minutes"
        )

    print(
        f"PREDICTED NEXT NYC SETTLEMENT: "
        f"{pred:.2f} F"
    )

    print(
        "Change from available Kalshi index: "
        f"{pred - base:+.2f} F"
    )

    print(
        "No trades. Historical receipt delays "
        "are assumptions, not measured timestamps."
    )


def main():
    import joblib

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        help=(
            "Backfill NYC history "
            "and train four models"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=55,
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Refit using existing NYC data",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Forecast at :40, :45, "
            ":50 and :55 ET"
        ),
    )

    args = parser.parse_args()

    if not 7 <= args.days <= 180:
        parser.error(
            "--days must be 7..180"
        )

    if args.setup:
        collect_index(args.days)
        collect_iem()
        train()
    elif args.retrain:
        train()

    if not MODEL_FILE.exists():
        raise RuntimeError(
            "No NYC models yet. Run: "
            "python3 src/temp-prediction/"
            "nyc_next_hour.py --setup"
        )

    # Only load locally generated, trusted model files.
    saved = joblib.load(MODEL_FILE)

    if not args.watch:
        forecast(saved)
        return

    print(
        "Watching :40/:45/:50/:55 ET; "
        "Ctrl+C to stop",
        flush=True,
    )

    last_gate = None

    while True:
        now = datetime.now(ET)
        gate = (
            now.date(),
            now.hour,
            now.minute,
        )

        if (
            now.minute in (40, 45, 50, 55)
            and gate != last_gate
        ):
            try:
                forecast(saved)

            except RuntimeError as e:
                print(
                    "Forecast unavailable:",
                    e,
                    flush=True,
                )

            last_gate = gate

        time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped")
    except (
        RuntimeError, OSError,
        ValueError, KeyError
    ) as e:
        raise SystemExit(
            "ERROR: " + str(e)
        )
