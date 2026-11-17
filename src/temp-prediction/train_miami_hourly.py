
#!/usr/bin/env python3
"""Train Miami hourly-index prediction models. Research only."""

import argparse
import csv
import math
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import numpy as np
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Install: python3 -m pip install numpy scikit-learn joblib"
    ) from exc

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "temp-prediction" / "miami_hourly_index"

STATIONS = (
    "KMIA1M", "KOPF1M", "KFLL1M", "KFXE1M", "KPMP1M"
)
HORIZONS = (5, 10, 15, 20)
UTC = timezone.utc
ET = ZoneInfo("America/New_York")


def dt(s):
    value = datetime.fromisoformat(
        str(s).replace("Z", "+00:00")
    )
    if value.tzinfo is None:
        raise ValueError(f"Missing timezone in {s}")
    return value.astimezone(UTC)


def number(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_minutes(path):
    with path.open(newline="") as f:
        raw = list(csv.DictReader(f))

    good = []
    for row in raw:
        t = dt(row["minute_utc"])
        index = number(row["index_f"])
        if math.isfinite(index):
            good.append((t, index, row))

    good.sort(key=lambda x: x[0])
    by_time = {
        t: (t, index, row)
        for t, index, row in good
    }
    records = sorted(
        by_time.values(), key=lambda x: x[0]
    )

    if not records:
        raise SystemExit("No valid minute records")

    return records, [r[0] for r in records]


def nearest(records, times, cutoff, max_stale=4):
    i = bisect_right(times, cutoff) - 1

    if i < 0:
        return None

    if cutoff - times[i] > timedelta(minutes=max_stale):
        return None

    return records[i]


def station_value(record, sid):
    if record is None:
        return float("nan")

    row = record[2]
    code = str(
        row.get(sid + "_qc", "")
    ).strip().lower()

    if code not in ("", "ok", "valid", "included"):
        return float("nan")

    return number(row.get(sid + "_f", ""))


def features(records, times, cutoff, hour):
    current = nearest(
        records, times, cutoff, max_stale=10
    )
    if current is None:
        return None

    base = current[1]

    feat = [
        base,
        (cutoff - current[0]).total_seconds() / 60,
    ]

    # Historical index changes.
    for lag in (5, 10, 20, 30):
        past = nearest(
            records,
            times,
            cutoff - timedelta(minutes=lag),
        )
        feat.append(
            base - past[1]
            if past else float("nan")
        )

    current_stations = [
        station_value(current, sid)
        for sid in STATIONS
    ]
    present = [
        v for v in current_stations
        if math.isfinite(v)
    ]

    # Require actual station-level history.
    if len(present) < 3:
        return None

    feat += [
        float(len(present)),
        max(present) - min(present),
    ]

    for sid, value in zip(STATIONS, current_stations):
        feat.append(
            value - base
            if math.isfinite(value)
            else float("nan")
        )

        for lag in (5, 10, 20):
            past = nearest(
                records,
                times,
                cutoff - timedelta(minutes=lag),
            )
            prior = station_value(past, sid)

            if math.isfinite(value) and math.isfinite(prior):
                feat.append(value - prior)
            else:
                feat.append(float("nan"))

    # Cyclical local time-of-day features.
    local = hour.astimezone(ET)
    phase = 2 * math.pi * (
        local.hour + local.minute / 60
    ) / 24

    feat += [
        math.sin(phase),
        math.cos(phase),
    ]

    return base, feat, len(present), current[0]


def model(kind):
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=30.0),
        )

    if kind == "boosted":
        return HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=10.0,
            random_state=42,
        )

    raise ValueError(kind)


def scores(pred, truth):
    error = np.abs(pred - truth)

    return (
        float(error.mean()),
        float(np.mean(
            np.floor(pred) == np.floor(truth)
        )),
        float(np.mean(error <= 1.0)),
    )


def describe(label, pred, truth):
    mae, bucket, within1 = scores(pred, truth)

    print(
        f"  {label:<15} "
        f"MAE={mae:.3f}F  "
        f"same_1F={bucket:.1%}  "
        f"within_1F={within1:.1%}"
    )

    return mae


def write_predictions(path, entries):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if not entries:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(entries[0]),
        )
        writer.writeheader()
        writer.writerows(entries)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__
    )
    ap.add_argument(
        "--delay-minutes",
        type=int,
        default=6,
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
    )
    args = ap.parse_args()

    if not 0 <= args.delay_minutes <= 30:
        ap.error(
            "--delay-minutes must be between 0 and 30"
        )

    src = args.data_dir

    records, times = load_minutes(
        src / "miami_index_minutes.csv"
    )

    with (
        src / "miami_hourly_targets.csv"
    ).open(newline="") as f:
        target_rows = list(csv.DictReader(f))

    targets = []

    for row in target_rows:
        hour = dt(row["settlement_hour_utc"])
        used = dt(row["settlement_minute_utc"])
        val = number(row["settlement_index_f"])

        # Exclude stale outage fallback targets.
        if (
            math.isfinite(val)
            and timedelta(0)
            <= hour - used
            <= timedelta(minutes=2)
        ):
            targets.append((hour, val))

    targets = sorted(set(targets))
    unique_hours = sorted({
        h for h, _ in targets
    })

    if len(unique_hours) < 150:
        raise SystemExit(
            f"Only {len(unique_hours)} clean targets; "
            "need at least 150"
        )

    cut1 = unique_hours[
        int(0.60 * len(unique_hours))
    ]
    cut2 = unique_hours[
        int(0.80 * len(unique_hours))
    ]

    print(
        f"Available minutes: {len(records):,}"
    )
    print(
        f"Clean hourly targets: {len(unique_hours):,}"
    )
    print(
        f"Train cutoff: {cut1.isoformat()}"
    )
    print(
        f"Test cutoff: {cut2.isoformat()}"
    )
    print(
        f"Assumed data delay: {args.delay_minutes} min"
    )

    output = []
    saved = {}

    for horizon in HORIZONS:
        samples = []

        for hour, actual in targets:
            decision = hour - timedelta(
                minutes=horizon
            )
            cutoff = decision - timedelta(
                minutes=args.delay_minutes
            )

            info = features(
                records, times, cutoff, hour
            )

            if info is None:
                continue

            base, feat, count, observation = info

            samples.append((
                hour, actual, base, feat,
                count, observation, decision
            ))

        # 1-hour embargo between periods.
        train = [
            x for x in samples
            if x[0] < cut1 - timedelta(hours=1)
        ]
        valid = [
            x for x in samples
            if cut1 + timedelta(hours=1)
            <= x[0]
            < cut2 - timedelta(hours=1)
        ]
        test = [
            x for x in samples
            if x[0] >= cut2 + timedelta(hours=1)
        ]

        print(
            f"\n{horizon}-minute horizon: "
            f"train={len(train)} "
            f"val={len(valid)} "
            f"test={len(test)}"
        )

        if min(
            len(train), len(valid), len(test)
        ) < 25:
            print(
                "Insufficient station samples; skipping"
            )
            continue

        def arrays(items):
            X = np.asarray(
                [x[3] for x in items],
                dtype=float,
            )
            y = np.asarray(
                [x[1] for x in items],
                dtype=float,
            )
            base = np.asarray(
                [x[2] for x in items],
                dtype=float,
            )
            return X, y, base

        Xtr, ytr, btr = arrays(train)
        Xv, yv, bv = arrays(valid)
        Xt, yt, bt = arrays(test)

        print(" Validation:")

        validation_mae = {
            "persistence": describe(
                "persistence", bv, yv
            )
        }

        for kind in ("ridge", "boosted"):
            fitted = model(kind).fit(
                Xtr, ytr - btr
            )
            validation_mae[kind] = describe(
                kind,
                bv + fitted.predict(Xv),
                yv,
            )

        winner = min(
            validation_mae,
            key=validation_mae.get,
        )
        print(
            " Selected on validation:", winner
        )

        # Refit selected model using training
        # and validation, without test labels.
        if winner == "persistence":
            test_pred = bt
            saved[horizon] = {
                "winner": winner,
                "delay_minutes": args.delay_minutes,
            }
        else:
            Xtv = np.concatenate([Xtr, Xv])
            residuals = np.concatenate([
                ytr - btr,
                yv - bv,
            ])

            fitted = model(winner).fit(
                Xtv, residuals
            )
            test_pred = bt + fitted.predict(Xt)

            saved[horizon] = {
                "winner": winner,
                "model": fitted,
                "delay_minutes": args.delay_minutes,
            }

        print(" UNSEEN TEST:")

        describe("persistence", bt, yt)
        describe(
            winner + " chosen",
            test_pred,
            yt,
        )

        daytime = [
            i for i, x in enumerate(test)
            if 8 <= x[0].astimezone(ET).hour < 17
        ]

        if daytime:
            print(
                f" Daytime 08-17 ET (n={len(daytime)}):"
            )
            describe(
                "persistence",
                bt[daytime],
                yt[daytime],
            )
            describe(
                winner + " chosen",
                test_pred[daytime],
                yt[daytime],
            )

        for row, pred, baseline in zip(
            test, test_pred, bt
        ):
            (
                hour, actual, _, _,
                count, observation, decision
            ) = row

            output.append({
                "hour_utc": hour.isoformat(),
                "horizon_min": horizon,
                "decision_utc": decision.isoformat(),
                "asof_obs_utc": observation.isoformat(),
                "station_count": count,
                "model": winner,
                "predicted_f": round(
                    float(pred), 4
                ),
                "persistence_f": round(
                    float(baseline), 4
                ),
                "actual_f": round(actual, 4),
                "model_error_f": round(
                    float(abs(pred - actual)), 4
                ),
                "baseline_error_f": round(
                    float(abs(baseline - actual)), 4
                ),
            })

    if not output:
        raise SystemExit(
            "No trainable horizons. Confirm "
            "historical CSV has populated "
            "station *_f columns."
        )

    out_csv = (
        src / "miami_model_test_predictions.csv"
    )
    write_predictions(out_csv, output)

    model_path = (
        src / "miami_station_models.joblib"
    )
    joblib.dump(saved, model_path)

    print(
        f"\nSaved predictions: {out_csv}"
    )
    print(
        f"Saved research models: {model_path}"
    )
    print(
        "NOTE: Fixed publication delay is assumed. "
        "These are not verified live trading results."
    )


if __name__ == "__main__":
    main()
