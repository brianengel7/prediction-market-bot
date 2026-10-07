import argparse
import math
import os

import pandas as pd

from src.backtest.actuals import get_actual_highs
from src.database.db import (
    get_connection,
    save_weather_model_backtest,
)
from src.weather.nbm import get_nbm_high_forecast


STATION = "KNYC"
HORIZON = "D0_12Z"


def backfill(start_date, end_date):
    # db.py loads your existing project .env.
    # Require the remote database instead of falling back to SQLite.
    if not os.environ.get("PREDICTION_DATABASE_URL"):
        raise RuntimeError(
            "PREDICTION_DATABASE_URL is missing. "
            "Set it to your existing Supabase connection string."
        )

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()

    if start > end:
        raise ValueError("--start must be on or before --end.")

    today = pd.Timestamp.now(tz="America/New_York").date()
    if end >= today:
        raise ValueError(
            "--end must be before today because this backfill "
            "requires completed daily actuals."
        )

    # Read existing rows first so reruns resume missing dates.
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT target_date, model_run
            FROM weather_model_backtests
            WHERE station = ?
              AND source = ?
              AND product = ?
              AND forecast_horizon = ?
              AND target_date >= ?
              AND target_date <= ?
            """,
            (
                STATION,
                "NBM",
                "NBS",
                HORIZON,
                start.isoformat(),
                end.isoformat(),
            ),
        ).fetchall()

    existing = {
        (
            pd.Timestamp(day).date(),
            pd.to_datetime(run, utc=True),
        )
        for day, run in rows
    }

    # Same historical actuals source as your original NBM backtest.
    actuals = get_actual_highs(start, end, station=STATION)
    actuals_by_date = {
        pd.Timestamp(row["date"]).date(): float(row["actual_high"])
        for row in actuals
    }

    saved = skipped = missing_actuals = failed = 0

    for timestamp in pd.date_range(start, end, freq="D"):
        day = timestamp.date()
        run_time = pd.Timestamp(
            f"{day.isoformat()} 12:00:00",
            tz="UTC",
        )

        if (day, run_time) in existing:
            skipped += 1
            print(f"SKIP {day}: already stored", flush=True)
            continue

        actual_high = actuals_by_date.get(day)
        if actual_high is None:
            missing_actuals += 1
            print(f"SKIP {day}: no actual high available", flush=True)
            continue

        try:
            # Existing collector and parser, with SAME-DAY run time.
            forecast = get_nbm_high_forecast(
                run_time=run_time,
                target_date=day,
                station=STATION,
            )

            forecast_high = float(forecast["forecast_high"])
            if not (
                math.isfinite(forecast_high)
                and math.isfinite(actual_high)
            ):
                raise ValueError("Non-finite forecast or actual high.")

            forecast_std = forecast.get("forecast_std")
            if forecast_std is not None:
                forecast_std = float(forecast_std)
                if not math.isfinite(forecast_std):
                    forecast_std = None

        except Exception as error:
            failed += 1
            print(f"FAILED {day}: {error}", flush=True)
            continue

        # Database errors stop the run rather than silently losing data.
        save_weather_model_backtest(
            target_date=day,
            source="NBM",
            product="NBS",
            model_run=forecast["run_time"],
            forecast_horizon=HORIZON,
            forecast_value=forecast_high,
            forecast_std=forecast_std,
            actual_high=actual_high,
            station=STATION,
            extraction_method="station_text",
            model_version="NBM-v5.0-NBS",
        )

        saved += 1
        print(
            f"SAVED {day} | run={run_time:%Y-%m-%d %HZ} | "
            f"NBM={forecast_high:.1f}F | "
            f"actual={actual_high:.1f}F | "
            f"error={forecast_high - actual_high:+.1f}F",
            flush=True,
        )

    with get_connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(DISTINCT target_date)
            FROM weather_model_backtests
            WHERE station = ?
              AND source = ?
              AND product = ?
              AND forecast_horizon = ?
              AND target_date >= ?
              AND target_date <= ?
            """,
            (
                STATION,
                "NBM",
                "NBS",
                HORIZON,
                start.isoformat(),
                end.isoformat(),
            ),
        ).fetchone()[0]

    print(
        f"\nFinished: saved={saved}, already_stored={skipped}, "
        f"missing_actuals={missing_actuals}, failed={failed}"
    )
    print(f"Database read-back: {count} morning NBM dates in this range.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill same-day 12Z NBM forecasts into Supabase."
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    backfill(args.start, args.end)