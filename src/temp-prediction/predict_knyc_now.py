
#!/usr/bin/env python3
"""Generate an immediate KNYC forecast from frozen V3/V5 models."""

import argparse
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

spec = importlib.util.spec_from_file_location(
    "live_knyc_shadow",
    HERE / "live_knyc_shadow.py",
)
shadow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)


def target_time(value, now):
    if value:
        chosen = datetime.strptime(
            value, "%H:%M"
        )
        return now.replace(
            hour=chosen.hour,
            minute=chosen.minute,
            second=0,
            microsecond=0,
        )

    upcoming = now.replace(
        minute=51, second=0, microsecond=0
    )

    if upcoming <= now:
        upcoming += timedelta(hours=1)

    return upcoming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        help="Target time in ET, e.g. 12:51",
    )
    args = parser.parse_args()

    now = shadow.now_utc()
    local_now = now.astimezone(ET)
    target = target_time(args.target, local_now)

    remaining = (
        target - local_now
    ).total_seconds() / 60

    print(
        "\n=== KNYC ON-DEMAND FORECAST ===",
        flush=True,
    )
    print("Current time:", local_now)
    print("Target time:", target)
    print(
        "Minutes until target:",
        round(remaining, 1),
    )

    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent":
            "PolyQuant-KNYC-on-demand-research/1.0"
    })

    reports = shadow.fetch_metars(session)

    history = {
        (
            report["station"],
            report["observed_utc"],
        ): report
        for report in reports
    }

    now = shadow.now_utc()

    local_now = now.astimezone(ET)

    remaining = (
        target - local_now
    ).total_seconds() / 60

    print("\nLatest available observations:")

    for station in shadow.STATIONS:
        matching = sorted(
            (
                r for r in reports
                if r["station"] == station
            ),
            key=lambda r: r["observed_utc"],
        )

        if matching:
            record = matching[-1]
            print(
                f"  {station}: "
                f"{record['observed_utc']} "
                f"temp={record['temp']}F"
            )
        else:
            print(f"  {station}: missing")

    if remaining <= 0:
        print(
            "\nTarget has already occurred."
            "\nThis command will not generate "
            "a backdated prediction."
        )

        matching = sorted(
            (
                r for r in reports
                if r["station"] == "NYC"
                and r["observed_utc"]
                == shadow.iso(
                    target.astimezone(
                        shadow.timezone.utc
                    )
                )
            ),
            key=lambda r: r["observed_utc"],
        )

        if matching:
            actual = matching[-1]
            print(
                "Observed temperature:",
                actual["precise_f"],
                "F (precise METAR)"
            )
        else:
            print(
                "The target observation is "
                "not yet available from AWC."
            )
        return

    # Show both selected models every time.
    BEST_MODELS = {
        45: ("knyc5_airports6", "V5_plus_clouds"),
        30: ("knyc3_airports4", "V5_plus_solar_clouds"),
    }

    print("\n=== SELECTED MODEL PREDICTIONS ===")

    specs = shadow.load_models()
    forecasts = []

    for (
        tag, trained_horizon, variant,
        model, columns
    ) in specs:

        if trained_horizon not in BEST_MODELS:
            continue

        selected_config, selected_variant = (
            BEST_MODELS[trained_horizon]
        )

        if (tag, variant) != (
            selected_config, selected_variant
        ):
            continue

        nyc_lag, airport_lag = (
            (5, 6)
            if tag == "knyc5_airports6"
            else (3, 4)
        )

        features, used = shadow.input_features(
            history,
            now,
            trained_horizon,
            nyc_lag,
            airport_lag,
        )

        if not (
            pd.notna(features["NYC_temp"])
            and any(
                pd.notna(features[f"{s}_temp"])
                for s in ("LGA", "EWR", "TEB")
            )
        ):
            print(
                "SKIPPED",
                trained_horizon,
                "minute model - insufficient observations"
            )
            continue

        try:
            X = pd.DataFrame([{
                feature: features[feature]
                for feature in columns
            }])

            prediction = (
                features["NYC_temp"]
                + float(model.predict(X)[0])
            )

            forecasts.append({
                "model": f"{trained_horizon}-minute model",
                "forecast_f": round(prediction, 3),
                "minutes_until_target": round(remaining, 1),
                "scenario": tag,
                "knyc_report": (
                    used["NYC"]["observed_utc"]
                    if used["NYC"]
                    else "missing"
                ),
            })

        except Exception as exc:
            print(
                "MODEL ERROR",
                trained_horizon,
                variant,
                str(exc)[:240]
            )

    forecasts.sort(
        key=lambda x: -int(x["model"].split("-")[0])
    )

    if forecasts and any(
        abs(remaining - h) > 2
        for h in BEST_MODELS
    ):
        print(
            "NOTE: Both models are shown, but forecasts "
            "made outside their trained horizons are "
            "experimental estimates."
        )

    if forecasts:
        print("\n=== FORECASTS (°F) ===")
        print(
            pd.DataFrame(forecasts).to_string(
                index=False
            )
        )
    else:
        print(
            "\nNo model forecasts generated. "
            "Check the missing observations "
            "or model errors above."
        )


if __name__ == "__main__":
    main()
