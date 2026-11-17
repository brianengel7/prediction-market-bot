#!/usr/bin/env python3
"""Independent V6-C3 forward-only KNYC probability shadow.

Research only. No trading endpoints. Reuses *read-only* the existing
live_knyc_shadow.py METAR parser, first-seen guard, and feature builder.

Run from project root: python3 -u src/temp-prediction/v6_c3_live_shadow.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from v6_c3_probability_engine import ProbabilityEngine

ROOT = Path("data/research/nyc_observation_20260701_20261026")
DEFAULT_OUT = ROOT / "v6_c3_live_forward"
UTC = timezone.utc
CONFIGS = {"primary": (5, 6), "fast": (3, 4)}


def old_shadow():
    path = Path(__file__).with_name("live_knyc_shadow.py")
    spec = importlib.util.spec_from_file_location("knyc_legacy_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_datetime(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")
    return result.astimezone(UTC)


def stamp(value):
    return value.astimezone(UTC).isoformat(timespec="seconds")


def append_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, allow_nan=False, separators=(",", ":")) + "\n")
        f.flush()


def read_jsonl(path):
    if not path.exists():
        return []
    output = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupt JSONL at {path}:{number}") from exc
    return output


def forecast_key(row):
    return (row["target_utc"], int(row["horizon_minutes"]), row["configuration"])


def artifact_id(tag):
    """Fingerprint the three inputs whose alteration would change output."""
    paths = [
        ROOT / "v7_extended_history" / "training_results" / tag / "research_models.joblib",
        ROOT / "v6_c2_empirical_distributions" / tag / "chronological_oof_residuals.csv",
        ROOT / "v6_c3_kde_recalibration" / tag / "selected_parameters.csv",
    ]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()[:20]


class ForwardState:
    def __init__(self, folder):
        self.out = Path(folder)
        self.out.mkdir(parents=True, exist_ok=True)
        self.metar_file = self.out / "metars.jsonl"
        self.forecast_file = self.out / "forecasts.jsonl"
        self.score_file = self.out / "scores.jsonl"
        self.observations = {}
        self.forecasts = {}
        self.scored = set()
        for row in read_jsonl(self.metar_file):
            self.observations[(row["station"], row["observed_utc"])] = row
        for row in read_jsonl(self.forecast_file):
            key = forecast_key(row)
            if key in self.forecasts:
                raise RuntimeError(f"Duplicate forecast in journal: {key}")
            self.forecasts[key] = row
        for row in read_jsonl(self.score_file):
            key = forecast_key(row)
            if key in self.scored:
                raise RuntimeError(f"Duplicate score in journal: {key}")
            self.scored.add(key)

    def add_metars(self, rows):
        count = 0
        for row in rows:
            key = (row["station"], row["observed_utc"])
            if key in self.observations:
                # Never replace the original first-seen timestamp or fields.
                continue
            self.observations[key] = row
            append_json(self.metar_file, row)
            count += 1
        return count

    def add_forecast(self, record):
        key = forecast_key(record)
        if key in self.forecasts:
            return False
        append_json(self.forecast_file, record)
        self.forecasts[key] = record
        return True

    def score_ready(self):
        count = 0
        for key, forecast in list(self.forecasts.items()):
            if key in self.scored:
                continue
            target = utc_datetime(forecast["target_utc"])
            observation = self.observations.get(("NYC", stamp(target)))
            if observation is None:
                # Compatible with legacy ISO UTC format if different.
                observation = next((
                    row for (station, _), row in self.observations.items()
                    if station == "NYC" and utc_datetime(row["observed_utc"]) == target
                ), None)
            if observation is None:
                continue
            actual = observation.get("precise_f")
            if actual is None or not math.isfinite(float(actual)):
                # Do not replace missing T-group target with coarser temperature.
                continue
            actual = float(actual)
            if utc_datetime(observation["first_seen_utc"]) <= utc_datetime(forecast["forecast_at_utc"]):
                raise RuntimeError("Actual target was already visible at decision: " + str(key))
            bucket = math.floor(actual + 0.5)
            first_bucket = int(forecast["full_bucket_start_f"])
            all_probs = forecast["full_bucket_probs"]
            index = bucket - first_bucket
            if not (0 <= index < len(all_probs)):
                print("WARNING target outside saved probability grid:", bucket, flush=True)
                p = 1e-12
            else:
                p = max(float(all_probs[index]), 1e-12)
            prediction = float(forecast["point_forecast_f"])
            hit1 = float(abs(actual - prediction) <= 1.0)
            p1 = float(forecast["probability_within_1f"])
            record = {
                "target_utc": forecast["target_utc"],
                "horizon_minutes": forecast["horizon_minutes"],
                "configuration": forecast["configuration"],
                "artifact_id": forecast["artifact_id"],
                "forecast_at_utc": forecast["forecast_at_utc"],
                "target_first_seen_utc": observation["first_seen_utc"],
                "actual_precise_f": actual,
                "actual_bucket_f": bucket,
                "predicted_f": prediction,
                "error_f": actual - prediction,
                "absolute_error_f": abs(actual - prediction),
                "hit_within_one_f": hit1,
                "p_actual_bucket": p,
                "bucket_logloss": -math.log(p),
                "bucket_brier": 1.0 - 2.0 * p + float(forecast["sum_p_squared"]),
                "within_one_brier": (p1 - hit1) ** 2,
                "cover_50": float(forecast["intervals"]["50"]["lower_f"] <= actual <= forecast["intervals"]["50"]["upper_f"]),
                "cover_80": float(forecast["intervals"]["80"]["lower_f"] <= actual <= forecast["intervals"]["80"]["upper_f"]),
                "cover_90": float(forecast["intervals"]["90"]["lower_f"] <= actual <= forecast["intervals"]["90"]["upper_f"]),
                "scored_at_utc": stamp(datetime.now(UTC)),
            }
            append_json(self.score_file, record)
            self.scored.add(key)
            count += 1
            print(f"SCORED {record['configuration']} {record['horizon_minutes']}m "
                  f"actual={actual:.2f}F forecast={prediction:.2f}F "
                  f"logloss={record['bucket_logloss']:.3f}", flush=True)
        return count


def valid_sources(helper, latest, at):
    nyc = latest.get("NYC")
    airports = [latest.get(s) for s in ("LGA", "EWR", "TEB")]
    if nyc is None or not any(r is not None for r in airports):
        raise RuntimeError("No eligible KNYC or airport METAR; cannot forecast")
    source = {}
    for station, item in latest.items():
        if item is None:
            source[station] = None
            continue
        seen = utc_datetime(item["first_seen_utc"])
        observed = utc_datetime(item["observed_utc"])
        if seen > at or observed > at:
            raise RuntimeError("Future-dated source slipped into model features")
        source[station] = {
            "observed_utc": item["observed_utc"],
            "first_seen_utc": item["first_seen_utc"],
            "age_minutes": round((at - observed).total_seconds() / 60, 2),
            "temp_f": item.get("temp"),
        }
    return source


def generate_forecast(helper, engine, history, at, target, horizon, artifacts, mode):
    feature, used = helper.input_features(
        history, at, horizon,
        engine.knyc_lag, engine.airport_lag,
    )
    source = valid_sources(helper, used, at)
    report = engine.predict_features(horizon, feature, stamp(target))
    # Save a full distribution NOW; scoring will not depend on future model files.
    grid = np.arange(-30, 131)
    probs = engine.distributions[horizon]["adjusted"].bucket_probs(
        float(report["point_forecast_f"]), grid
    )
    if not (np.isfinite(probs).all() and (probs >= 0).all()):
        raise RuntimeError("Invalid probability distribution")
    if float(probs.sum()) < 0.99999:
        raise RuntimeError("Uncaptured forecast probability outside -30..130F")
    record = dict(report)
    record.update({
        "mode": mode,
        "forecast_at_utc": stamp(at),
        "target_utc": stamp(target),
        "configuration": engine.tag,
        "artifact_id": artifacts[engine.tag],
        "source_observations": source,
        "full_bucket_start_f": int(grid[0]),
        "full_bucket_probs": probs.tolist(),
        "sum_p_squared": float(np.sum(probs * probs)),
        "actual_lead_minutes": round((target - at).total_seconds() / 60, 3),
    })
    return record


def print_forecast(row):
    print(f"FORECAST {row['configuration']} {row['horizon_minutes']}m "
          f"target={row['target_utc']} "
          f"point={row['point_forecast_f']:.3f}F "
          f"modal={row['most_likely_temperature_f']}F "
          f"p={row['most_likely_probability']:.1%} "
          f"within1={row['probability_within_1f']:.1%}", flush=True)


def summary(out):
    scores = read_jsonl(Path(out) / "scores.jsonl")
    print(f"Saved forecasts: {len(read_jsonl(Path(out) / 'forecasts.jsonl'))}")
    print(f"Scored forecasts: {len(scores)}")
    if not scores:
        print("No scored forecasts yet.")
        return
    df = pd.DataFrame(scores)
    table = df.groupby(["configuration", "horizon_minutes"]).agg(
        n=("bucket_logloss", "size"),
        mae_f=("absolute_error_f", "mean"),
        within_one_pct=("hit_within_one_f", lambda s: s.mean() * 100),
        bucket_logloss=("bucket_logloss", "mean"),
        bucket_brier=("bucket_brier", "mean"),
        within_one_brier=("within_one_brier", "mean"),
        cover_50_pct=("cover_50", lambda s: s.mean() * 100),
        cover_80_pct=("cover_80", lambda s: s.mean() * 100),
        cover_90_pct=("cover_90", lambda s: s.mean() * 100),
    )
    print(table.round(4).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", choices=("primary", "fast", "both"), default="both")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--decision-grace-seconds", type=int, default=90)
    parser.add_argument("--forever", action="store_true", help="Repeat every weekday/day until stopped")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        summary(args.out)
        return
    if args.poll_seconds < 10 or not 0 <= args.decision_grace_seconds <= 180:
        parser.error("--poll-seconds >=10 and grace 0..180 seconds required")

    helper = old_shadow()
    chosen = ("primary", "fast") if args.configs == "both" else (args.configs,)
    engines = {name: ProbabilityEngine(*CONFIGS[name]) for name in chosen}
    artifacts = {engine.tag: artifact_id(engine.tag) for engine in engines.values()}
    state = ForwardState(args.out)
    session = requests.Session()
    session.headers.update({"User-Agent": "PolyQuant-KNYC-V6C3-forward-research/1.0"})
    print("V6-C3 FORWARD SHADOW | NO TRADES | config:", list(engines),
          "| history:", len(state.observations), "| saved forecasts:", len(state.forecasts), flush=True)
    print("Uses AWC reception time; decisions at :06 (45m) and :21 (30m) ET, 08:00-16:59.", flush=True)
    next_poll = datetime.now(UTC)
    last_error = None
    try:
        while True:
            now = datetime.now(UTC)
            local = now.astimezone(helper.ET)
            cutoff = local.replace(hour=17, minute=15, second=0, microsecond=0)
            if not args.forever and local >= cutoff:
                print("End of research session (17:15 ET).", flush=True)
                break
            if args.forever and (local.hour < 7 or local.hour >= 18):
                time.sleep(60)
                continue

            if now >= next_poll:
                try:
                    fresh = helper.fetch_metars(session)
                    added = state.add_metars(fresh)
                    state.score_ready()
                    print(f"METARS received={len(fresh)} new={added} at={stamp(datetime.now(UTC))}", flush=True)
                except (requests.RequestException, ValueError, RuntimeError) as exc:
                    if str(exc) != last_error:
                        print("METAR WARNING:", str(exc)[:200], flush=True)
                        last_error = str(exc)
                else:
                    last_error = None
                next_poll = datetime.now(UTC) + timedelta(seconds=args.poll_seconds)

            # Take a fresh decision timestamp after the fetch has completed.
            at = datetime.now(UTC)
            local = at.astimezone(helper.ET)
            if 8 <= local.hour <= 16 and local.minute in (6, 21):
                scheduled = local.replace(second=0, microsecond=0)
                lateness = (local - scheduled).total_seconds()
                if lateness <= args.decision_grace_seconds:
                    horizon = 45 if local.minute == 6 else 30
                    target = local.replace(minute=51, second=0, microsecond=0).astimezone(UTC)
                    for name, engine in engines.items():
                        key = (stamp(target), horizon, engine.tag)
                        if key in state.forecasts:
                            continue
                        try:
                            record = generate_forecast(
                                helper, engine, state.observations,
                                at, target, horizon, artifacts, "scheduled_shadow",
                            )
                            if state.add_forecast(record):
                                print_forecast(record)
                        except (RuntimeError, ValueError, KeyError) as exc:
                            print(f"SKIPPED {name} {horizon}m: {exc}", flush=True)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    finally:
        session.close()
        summary(args.out)


if __name__ == "__main__":
    main()
