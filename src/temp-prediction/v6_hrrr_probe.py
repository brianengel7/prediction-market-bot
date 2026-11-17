
#!/usr/bin/env python3
"""V6 Stage 1: Historical HRRR run collection and KNYC comparison.

Research only. No modifications to V3/V5 or trading code.
Historical forecast publication delays are simulated, not verified.
"""

import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

OUT = ROOT / "v6_hrrr_research"
CACHE = OUT / "single_runs_cache"

API = (
    "https://single-runs-api.open-meteo.com/v1/forecast"
)

MODEL = "ncep_hrrr_conus"

VARIABLES = (
    "temperature_2m,cloud_cover,wind_speed_10m"
)

ET = ZoneInfo("America/New_York")


def run_times(start, end, step):
    first = (
        pd.Timestamp(start, tz="UTC")
        - pd.Timedelta(days=1)
    )

    last = (
        pd.Timestamp(end, tz="UTC")
        + pd.Timedelta(days=1)
    )

    return pd.date_range(
        first,
        last,
        freq=f"{step}h",
        inclusive="left",
    )


def parse_run(run, payload):
    hourly = payload.get("hourly") or {}
    raw_times = hourly.get("time") or []

    if not raw_times:
        raise ValueError("Missing hourly.time")

    times = pd.to_datetime(
        raw_times,
        utc=True,
        errors="coerce",
    )

    if times.isna().any():
        raise ValueError("Invalid forecast timestamps")

    units = payload.get("hourly_units") or {}

    temp_unit = units.get("temperature_2m")

    if temp_unit not in (
        "°F", "F", "fahrenheit"
    ):
        raise ValueError(
            f"Unexpected temperature unit: {temp_unit}"
        )

    fields = {
        "temp_f": "temperature_2m",
        "cloud_pct": "cloud_cover",
        "wind_mph": "wind_speed_10m",
    }

    frame = pd.DataFrame({
        "valid_utc": times,
    })

    for output_name, source_name in fields.items():
        values = hourly.get(source_name)

        if (
            not isinstance(values, list)
            or len(values) != len(times)
        ):
            raise ValueError(
                f"Missing or invalid {source_name}"
            )

        frame[output_name] = pd.to_numeric(
            pd.Series(values),
            errors="coerce",
        )

    frame["run_utc"] = pd.Timestamp(run)

    frame = (
        frame.sort_values("valid_utc")
        .drop_duplicates("valid_utc")
    )

    if frame["temp_f"].notna().sum() < 2:
        raise ValueError(
            "Insufficient valid forecast temperatures"
        )

    return frame


def request_run(session, run):
    CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    key = run.strftime("%Y%m%d_%H%M")

    path = CACHE / f"{MODEL}_{key}.json"

    if path.exists():
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
        parse_run(run, payload)
        return payload, True

    params = {
        "latitude": 40.779,
        "longitude": -73.969,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "models": MODEL,
        "hourly": VARIABLES,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "GMT",
        "forecast_hours": 18,
    }

    last_error = None

    for attempt in range(3):
        try:
            response = session.get(
                API,
                params=params,
                timeout=(15, 60),
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("error"):
                raise ValueError(
                    payload.get(
                        "reason",
                        "API returned an error",
                    )
                )

            parse_run(run, payload)

            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            time.sleep(0.15)

            return payload, False

        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            last_error = exc

            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"{run.isoformat()}: {last_error}"
    )


def interpolate(frame, column, at):
    values = (
        frame[["valid_utc", column]]
        .dropna()
        .sort_values("valid_utc")
    )

    if len(values) < 2:
        return math.nan

    # Convert both timestamps to Unix seconds.
    # This works regardless of whether pandas uses
    # nanosecond or microsecond datetime precision.
    xs = (
        pd.to_datetime(
            values["valid_utc"], utc=True
        )
        .map(lambda t: t.timestamp())
        .to_numpy(dtype=float)
    )

    x = pd.Timestamp(at).timestamp()

    # Do not extrapolate outside the forecast window.
    if x < xs[0] or x > xs[-1]:
        return math.nan

    return float(
        np.interp(
            x,
            xs,
            values[column].to_numpy(dtype=float),
        )
    )


def collect(start, end, cycle_hours):
    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "PolyQuant-HRRR-V6-research/1.0"
    })

    times = list(
        run_times(start, end, cycle_hours)
    )

    frames = []
    errors = []

    print(
        f"Requesting {len(times)} HRRR runs",
        flush=True,
    )

    for i, run in enumerate(times, 1):
        try:
            payload, cached = request_run(
                session, run
            )

            frames.append(
                parse_run(run, payload)
            )

            if (
                i <= 3
                or i % 20 == 0
                or i == len(times)
            ):
                print(
                    f"[{i}/{len(times)}] "
                    f"{run.isoformat()} "
                    f"{'cached' if cached else 'downloaded'} "
                    f"failures={len(errors)}",
                    flush=True,
                )

        except Exception as exc:
            errors.append({
                "run_utc": run.isoformat(),
                "error": str(exc)[:250],
            })

            print(
                f"FAILED {run.isoformat()}: "
                f"{str(exc)[:180]}",
                flush=True,
            )

            if len(errors) >= 5 and not frames:
                raise RuntimeError(
                    "Five failures before obtaining "
                    "any valid run. Inspect the API."
                )

    if not frames:
        raise RuntimeError(
            "No HRRR forecast runs collected"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        errors,
        columns=["run_utc", "error"],
    ).to_csv(
        OUT / "failed_runs.csv",
        index=False,
    )

    result = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["run_utc", "valid_utc"])
    )

    output = OUT / "hourly_hrrr_runs.csv"

    result.to_csv(output, index=False)

    print(
        f"\nCollected {len(frames)}/{len(times)} "
        f"runs and {len(result)} forecast slots",
        flush=True,
    )

    print("Saved:", output.resolve())

    return result


def load_targets(start, end):
    path = (
        ROOT / "v5_inputs"
        / "knyc_precise_targets.csv"
    )

    df = pd.read_csv(path)

    required = {
        "observed_utc", "precise_f"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: "
            f"{required - set(df.columns)}"
        )

    df["target_utc"] = pd.to_datetime(
        df["observed_utc"],
        utc=True,
        errors="coerce",
    )

    df["actual_f"] = pd.to_numeric(
        df["precise_f"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["target_utc", "actual_f"]
    )

    df["target_et"] = (
        df["target_utc"].dt.tz_convert(ET)
    )

    df = df[
        (df["target_et"].dt.date >= start)
        & (df["target_et"].dt.date <= end)
        & df["target_et"].dt.hour.between(8, 16)
        & (df["target_et"].dt.minute == 51)
    ].sort_values("target_utc")

    if df["target_utc"].duplicated().any():
        raise ValueError(
            "Duplicate KNYC target timestamps"
        )

    return df


def latest_knyc(labels, decision, lag):
    eligible = labels[
        labels["target_utc"]
        + pd.Timedelta(minutes=lag)
        <= decision
    ]

    if eligible.empty:
        return None

    latest = eligible.iloc[-1]

    age = (
        decision - latest["target_utc"]
    ).total_seconds() / 60

    if age > 120:
        return None

    return latest


def metrics(actual, predicted):
    errors = (actual - predicted).abs()

    return (
        f"MAE={errors.mean():.3f}F "
        f"within±0.5F="
        f"{(errors <= 0.5).mean()*100:.1f}% "
        f"within±1F="
        f"{(errors <= 1).mean()*100:.1f}%"
    )


def evaluate(
    runs,
    start,
    end,
    publication_lag,
    knyc_lag,
):
    labels = load_targets(
        start - timedelta(days=1),
        end,
    )

    targets = load_targets(start, end)

    if targets.empty:
        raise RuntimeError(
            "No precise KNYC observations found"
        )

    runs = runs.copy()

    runs["run_utc"] = pd.to_datetime(
        runs["run_utc"], utc=True
    )

    runs["valid_utc"] = pd.to_datetime(
        runs["valid_utc"], utc=True
    )

    grouped = {
        run: frame.sort_values("valid_utc")
        for run, frame in runs.groupby("run_utc")
    }

    known_runs = sorted(grouped)

    predictions = []

    for target in targets.itertuples(index=False):
        for horizon in (30, 45):

            decision = (
                target.target_utc
                - pd.Timedelta(minutes=horizon)
            )

            eligible_runs = [
                run for run in known_runs
                if (
                    run + pd.Timedelta(
                        minutes=publication_lag
                    ) <= decision
                )
            ]

            if not eligible_runs:
                continue

            chosen_run = eligible_runs[-1]
            frame = grouped[chosen_run]

            hrrr_target = interpolate(
                frame,
                "temp_f",
                target.target_utc,
            )

            if not math.isfinite(hrrr_target):
                continue

            previous = latest_knyc(
                labels,
                decision,
                knyc_lag,
            )

            if previous is None:
                continue

            hrrr_previous = interpolate(
                frame,
                "temp_f",
                previous["target_utc"],
            )

            if not math.isfinite(hrrr_previous):
                continue

            last_actual = float(
                previous["actual_f"]
            )

            # HRRR predicts the change between the
            # last known reading and target time.
            predicted_change = (
                hrrr_target - hrrr_previous
            )

            predictions.append({
                "target_utc":
                    target.target_utc.isoformat(),

                "target_et_date":
                    target.target_et.strftime(
                        "%Y-%m-%d"
                    ),

                "horizon": horizon,

                "decision_utc":
                    decision.isoformat(),

                "run_utc":
                    chosen_run.isoformat(),

                "assumed_publication_utc": (
                    chosen_run
                    + pd.Timedelta(
                        minutes=publication_lag
                    )
                ).isoformat(),

                "run_age_at_decision_min": (
                    decision - chosen_run
                ).total_seconds() / 60,

                "knyc_observed_utc":
                    previous["target_utc"].isoformat(),

                "knyc_last_f": last_actual,

                "actual_f":
                    float(target.actual_f),

                "hrrr_absolute_f":
                    hrrr_target,

                "hrrr_at_last_observation_f":
                    hrrr_previous,

                "hrrr_delta_f":
                    predicted_change,

                "hrrr_knyc_adjusted_f":
                    last_actual + predicted_change,

                "hrrr_cloud_pct": interpolate(
                    frame,
                    "cloud_pct",
                    target.target_utc,
                ),

                "hrrr_wind_mph": interpolate(
                    frame,
                    "wind_mph",
                    target.target_utc,
                ),
            })

    result = pd.DataFrame(predictions)

    if result.empty:
        print(
            "No matched predictions. Check archive "
            "coverage and publication assumptions."
        )
        return

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = OUT / (
        f"paired_hrrr_pub{publication_lag}m_"
        f"knyc{knyc_lag}m.csv"
    )

    result.to_csv(output, index=False)

    print(
        "\n=== V6 INITIAL HRRR DIAGNOSTIC ==="
    )

    print(
        f"HRRR publication lag: "
        f"{publication_lag} minutes (assumed)"
    )

    for horizon in (30, 45):
        subset = result[
            result["horizon"] == horizon
        ]

        if subset.empty:
            continue

        actual = subset["actual_f"]

        print(
            f"\n{horizon}-minute forecast:"
            f" {len(subset)} observations, "
            f"{subset['target_et_date'].nunique()} days"
        )

        variants = {
            "KNYC persistence":
                "knyc_last_f",

            "Raw HRRR":
                "hrrr_absolute_f",

            "KNYC + HRRR temperature change":
                "hrrr_knyc_adjusted_f",
        }

        for name, column in variants.items():
            print(
                f"  {name}: "
                f"{metrics(actual, subset[column])}"
            )

    print("\nSaved:", output.resolve())

    print(
        "\nThis is an initial forecast comparison, "
        "not the final trained V6 model."
    )


def probe():
    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "PolyQuant-HRRR-V6-research/1.0"
    })

    run = pd.Timestamp(
        "2026-10-25T12:00:00Z"
    )

    payload, cached = request_run(
        session, run
    )

    frame = parse_run(run, payload)

    print("\n=== HRRR API PROBE ===")

    print("Model:", MODEL)
    print("Run:", run.isoformat())
    print("Cached:", cached)
    print(
        "Units:",
        payload.get("hourly_units", {})
    )

    print(
        "\nFirst forecast rows:\n",
        frame.head(4).to_string(index=False),
    )

    print("\nPROBE PASS")


def self_test():
    run = pd.Timestamp(
        "2026-10-25T12:00:00Z"
    )

    payload = {
        "hourly_units": {
            "temperature_2m": "°F"
        },
        "hourly": {
            "time": [
                "2026-10-25T12:00",
                "2026-10-25T13:00",
                "2026-10-25T14:00",
            ],
            "temperature_2m": [60, 62, 64],
            "cloud_cover": [10, 30, 50],
            "wind_speed_10m": [5, 6, 7],
        },
    }

    frame = parse_run(run, payload)

    halfway = pd.Timestamp(
        "2026-10-25T12:30:00Z"
    )

    assert abs(
        interpolate(
            frame, "temp_f", halfway
        ) - 61
    ) < 1e-9

    assert math.isnan(
        interpolate(
            frame,
            "temp_f",
            pd.Timestamp(
                "2026-10-25T15:00:00Z"
            ),
        )
    )

    print("SELF-TEST PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--start",
        default="2026-10-01",
    )

    parser.add_argument(
        "--end",
        default="2026-10-03",
    )

    parser.add_argument(
        "--cycle-hours",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--publication-lag-min",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--knyc-lag-min",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--reuse-downloaded",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.probe:
        probe()
        return

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if (
        end < start
        or args.cycle_hours not in (1, 2, 3, 6)
        or min(
            args.publication_lag_min,
            args.knyc_lag_min,
        ) < 0
    ):
        parser.error(
            "Invalid dates, cycle hours or delays"
        )

    if args.reuse_downloaded:
        runs = pd.read_csv(
            OUT / "hourly_hrrr_runs.csv"
        )
    else:
        runs = collect(
            start,
            end,
            args.cycle_hours,
        )

    evaluate(
        runs,
        start,
        end,
        args.publication_lag_min,
        args.knyc_lag_min,
    )


if __name__ == "__main__":
    main()
