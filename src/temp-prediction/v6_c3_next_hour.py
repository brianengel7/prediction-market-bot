#!/usr/bin/env python3
"""One-shot KNYC next-:51 precise temperature distribution scanner.

Research only. Uses the SAME frozen V6-C3 engine and METAR feature builder
as v6_c3_live_shadow.py. This file does not schedule trades or change models.

Run: python3 -u src/temp-prediction/v6_c3_next_hour.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from v6_c3_probability_engine import ProbabilityEngine
from v6_c3_live_shadow import (
    CONFIGS, DEFAULT_OUT, append_json, artifact_id,
    generate_forecast, old_shadow, print_forecast,
)


def next_target(now, zone):
    local = now.astimezone(zone)
    target_local = local.replace(minute=51, second=0, microsecond=0)
    if target_local <= local:
        target_local += timedelta(hours=1)
    return target_local.astimezone(timezone.utc)


def choose_horizon(at, target, requested):
    lead = (target - at).total_seconds() / 60
    if requested != "auto":
        return int(requested), lead
    return (30 if abs(lead - 30) <= abs(lead - 45) else 45), lead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", choices=("primary", "fast", "both"), default="primary")
    parser.add_argument("--horizon", choices=("auto", "30", "45"), default="auto")
    parser.add_argument("--strict", action="store_true", help="Only forecast within 2m of validated scheduled decision")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "on_demand_forecasts.jsonl")
    args = parser.parse_args()

    helper = old_shadow()
    session = requests.Session()
    session.headers.update({"User-Agent": "PolyQuant-KNYC-V6C3-on-demand-research/1.0"})
    print("Fetching latest AWC METAR history (6h) ...", flush=True)
    try:
        obs = helper.fetch_metars(session)
    finally:
        session.close()
    if not obs:
        raise RuntimeError("AWC returned no usable METARs")

    # Always use the actual first-reception timestamps, never observation
    # publication times alone. No historical look-ahead is introduced.
    history = {(r["station"], r["observed_utc"]): r for r in obs}
    at = datetime.now(timezone.utc)
    target = next_target(at, helper.ET)
    horizon, lead = choose_horizon(at, target, args.horizon)
    local = at.astimezone(helper.ET)
    scheduled = (
        8 <= local.hour <= 16
        and abs(lead - horizon) <= 2.0
        and ((local.minute in (6, 7, 8) and horizon == 45)
             or (local.minute in (21, 22, 23) and horizon == 30))
    )
    # A near-scheduled forecast at :05 or :20 is also considered off-schedule:
    # the model was trained at :06/:21, so do not silently extend its validity.
    if args.strict and not scheduled:
        raise RuntimeError(
            f"Outside validated :06/:21 08:00-16:59 ET schedule. "
            f"Actual lead={lead:.1f}m; selected model={horizon}m. "
            "Run without --strict for an explicitly experimental forecast."
        )

    mode = "scheduled_like_on_demand" if scheduled else "EXPERIMENTAL_OFF_SCHEDULE"
    print(f"\nAt {local:%Y-%m-%d %H:%M:%S %Z}; next KNYC target "
          f"{target.astimezone(helper.ET):%Y-%m-%d %H:%M %Z}")
    print(f"Actual time to reading: {lead:.1f} minutes; selected frozen {horizon}m model")
    if not scheduled:
        print("WARNING: OUTSIDE VALIDATED DECISION TIME. The reported bucket "
              "probabilities are exploratory, not calibrated for this lead/hours.")

    selected = ("primary", "fast") if args.configs == "both" else (args.configs,)
    for name in selected:
        engine = ProbabilityEngine(*CONFIGS[name])
        artifacts = {engine.tag: artifact_id(engine.tag)}
        record = generate_forecast(
            helper, engine, history, at, target, horizon, artifacts, mode,
        )
        print("\n" + "=" * 65)
        print_forecast(record)
        print("Forecast mode:", mode)
        print("Latest source observations (never post-decision):")
        for station, source in record["source_observations"].items():
            if source:
                print(f"  {station}: temp={source['temp_f']}F "
                      f"age={source['age_minutes']:.1f}m "
                      f"observed={source['observed_utc']}")
            else:
                print(f"  {station}: no eligible report")
        print("\nWhole-degree KNYC temperature probabilities:")
        for row in record["temperature_buckets"]:
            if row["probability"] >= 0.0005:
                print(f"  {row['temperature_f']:>3}F  {row['probability']:>7.2%}")
        if record["probability_outside_display"] > 0.0005:
            print(f"  Outside display {record['probability_outside_display']:.2%}")
        print("\nCentral prediction intervals:")
        for level, bound in record["intervals"].items():
            print(f"  {level}%: {bound['lower_f']:.2f}F .. {bound['upper_f']:.2f}F")
        append_json(args.out, record)

    print(f"\nLogged this one-shot result to {args.out}")
    print("No orders placed. The historical validation applies to :06/:21 ET decisions only.")


if __name__ == "__main__":
    main()
