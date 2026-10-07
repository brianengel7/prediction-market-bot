import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from src.database.db import get_connection, read_dataframe
from src.weather.fair_distribution import probability_for_market
from src.backtest.probability_calibration import (
    MIN_TRAIN_DAYS,
    MIN_RESIDUAL_DAYS,
)
from src.backtest.v2_nbm_strategy import (
    build_daily_candidates,
    select_daily_trade,
    summarize_strategy,
)


NY = "America/New_York"
EDGE = 0.025

TIMINGS = {
    "previous_day": {
        "horizon": "D-1_12Z",
        "days_before": 1,
        "clock": "16:05",
    },
    "morning": {
        "horizon": "D0_12Z",
        "days_before": 0,
        "clock": "10:00",
    },
}


def utc(values):
    return pd.to_datetime(values, utc=True, format="mixed")


def decision_time(day, timing):
    settings = TIMINGS[timing]
    local_day = day - pd.Timedelta(days=settings["days_before"])
    return pd.Timestamp(
        f"{local_day:%Y-%m-%d} {settings['clock']}",
        tz=NY,
    ).tz_convert("UTC")


def latest_versions(frame, keys):
    frame = frame.copy()
    frame["downloaded_at_utc"] = utc(frame["downloaded_at_utc"])
    return frame.sort_values(
        ["downloaded_at_utc", "payload_hash"]
    ).drop_duplicates(keys, keep="last")


def load_weather(start, end):
    with get_connection() as connection:
        data = read_dataframe(
            """
            SELECT target_date, forecast_horizon, model_run,
                   forecast_value, actual_high
            FROM weather_model_backtests
            WHERE station = ?
              AND source = ?
              AND product = ?
              AND target_date >= ?
              AND target_date <= ?
              AND forecast_horizon IN (?, ?)
            """,
            connection,
            params=(
                "KNYC", "NBM", "NBS", start, end,
                "D-1_12Z", "D0_12Z",
            ),
        )

    if data.empty:
        raise RuntimeError("No NBM forecast history found.")

    data["target_date"] = pd.to_datetime(
        data["target_date"]
    ).dt.normalize()
    data["model_run"] = utc(data["model_run"])

    for column in ("forecast_value", "actual_high"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data[
        np.isfinite(
            data[["forecast_value", "actual_high"]].to_numpy(
                dtype=float
            )
        ).all(axis=1)
    ].copy()

    by_timing = {}

    for timing, settings in TIMINGS.items():
        frame = data[
            data["forecast_horizon"].eq(settings["horizon"])
        ].copy()

        expected_run = (
            frame["target_date"].dt.tz_localize("UTC")
            - pd.Timedelta(days=settings["days_before"])
            + pd.Timedelta(hours=12)
        )
        frame = frame[frame["model_run"].eq(expected_run)].copy()

        if frame["target_date"].duplicated().any():
            raise RuntimeError(
                f"Duplicate exact-cycle forecasts for {timing}."
            )

        by_timing[timing] = frame.set_index("target_date").sort_index()
        print(f"{timing}: {len(frame)} valid forecast dates")

    # Both arms use precisely the same historical target dates.
    shared = sorted(
        set(by_timing["previous_day"].index)
        & set(by_timing["morning"].index)
    )

    if not shared:
        raise RuntimeError(
            "No shared forecast dates. Both D-1_12Z and D0_12Z "
            "NBM forecasts are required."
        )

    for timing in by_timing:
        by_timing[timing] = by_timing[timing].loc[shared].copy()

    if not np.allclose(
        by_timing["previous_day"]["actual_high"],
        by_timing["morning"]["actual_high"],
        rtol=0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Actual temperatures disagree between forecast horizons. "
            "Resolve those records before comparing."
        )

    print(f"Shared forecast dates: {len(shared)}")
    return by_timing


def calibrate(frame):
    """
    Same algorithm for each forecast horizon.

    1. Fit expanding bias from eligible earlier forecasts.
    2. Produce a genuinely historical corrected point forecast.
    3. Estimate uncertainty from earlier out-of-sample point errors.

    Both arms exclude target_date - 1 from training, so they use
    the same information cutoff for daily actuals.
    """
    history = frame.copy()
    history["raw_error"] = (
        history["forecast_value"] - history["actual_high"]
    )

    past_residuals = []
    calibrated = {}

    for day, row in history.iterrows():
        cutoff = day - pd.Timedelta(days=1)
        prior = history[history.index < cutoff]

        if len(prior) < MIN_TRAIN_DAYS:
            continue

        bias = float(prior["raw_error"].mean())
        point = float(row["forecast_value"]) - bias

        eligible_residuals = [
            residual
            for residual_day, residual in past_residuals
            if residual_day < cutoff
        ]

        if len(eligible_residuals) >= MIN_RESIDUAL_DAYS:
            residual_mean = float(np.mean(eligible_residuals))
            residual_std = float(
                np.std(eligible_residuals, ddof=1)
            )

            if not np.isfinite(residual_std) or residual_std <= 0:
                raise RuntimeError(
                    f"Invalid residual spread for {day.date()}."
                )

            calibrated[day] = {
                "point": point,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "model_run": row["model_run"],
                "actual_high": float(row["actual_high"]),
                "bias_dates": len(prior),
                "residual_dates": len(eligible_residuals),
            }

        # Append only after constructing this day's prediction.
        # It becomes eligible for later predictions via cutoff.
        past_residuals.append(
            (day, float(row["actual_high"]) - point)
        )

    return calibrated


def load_quotes(start, end):
    with get_connection() as connection:
        contracts = read_dataframe(
            """
            SELECT ticker, target_date, yes_payout, raw_market,
                   downloaded_at_utc, payload_hash
            FROM weather_market_contract_history
            WHERE target_date >= ? AND target_date <= ?
              AND ticker LIKE ?
            """,
            connection,
            params=(start, end, "KXHIGHNY-%"),
        )

        candles = read_dataframe(
            """
            SELECT ticker, target_date, candle_end_utc,
                   yes_bid, yes_ask, downloaded_at_utc, payload_hash
            FROM weather_market_candle_history
            WHERE target_date >= ? AND target_date <= ?
              AND interval_minutes = 1
              AND ticker LIKE ?
            """,
            connection,
            params=(start, end, "KXHIGHNY-%"),
        )

    if contracts.empty or candles.empty:
        raise RuntimeError(
            "Missing contract history or one-minute candles. "
            "Use hourly_market_backfill with --interval-minutes 1."
        )

    contracts = latest_versions(contracts, ["ticker"])
    candles = latest_versions(
        candles, ["ticker", "candle_end_utc"]
    )

    for frame in (contracts, candles):
        frame["target_date"] = pd.to_datetime(
            frame["target_date"]
        ).dt.normalize()

    candles["quote_time"] = utc(candles["candle_end_utc"])

    raw = contracts["raw_market"].map(
        lambda value: (
            json.loads(value) if isinstance(value, str) else value
        )
    )
    contracts["strike_type"] = raw.map(
        lambda value: value.get("strike_type")
    )
    for column in ("floor_strike", "cap_strike"):
        contracts[column] = pd.to_numeric(
            raw.map(lambda value: value.get(column)),
            errors="coerce",
        )

    contracts["outcome"] = pd.to_numeric(
        contracts["yes_payout"], errors="coerce"
    )

    metadata = contracts[
        [
            "ticker", "target_date", "strike_type",
            "floor_strike", "cap_strike", "outcome",
        ]
    ]

    expected = contracts.groupby("target_date")["ticker"].nunique()
    result = {}

    for timing in TIMINGS:
        target_times = candles["target_date"].map(
            lambda day: decision_time(day, timing)
        )

        selected = candles[
            candles["quote_time"].eq(target_times)
        ].copy()

        selected = selected.merge(
            metadata,
            on=["ticker", "target_date"],
            how="inner",
            validate="many_to_one",
        )

        for column in ("yes_bid", "yes_ask"):
            selected[column] = pd.to_numeric(
                selected[column], errors="coerce"
            )

        selected["no_ask"] = 1.0 - selected["yes_bid"]
        result[timing] = {}

        for day, group in selected.groupby("target_date"):
            group = group.sort_values("ticker").reset_index(drop=True)

            valid = (
                group["yes_bid"].between(0, 1)
                & group["yes_ask"].between(0, 1)
                & group["yes_bid"].le(group["yes_ask"])
            )

            if (
                len(group) != 6
                or group["ticker"].nunique() != 6
                or expected.get(day, 0) != 6
                or not valid.all()
                or not group["outcome"].isin([0.0, 1.0]).all()
                or group["outcome"].sum() != 1
            ):
                continue

            result[timing][day] = group

        print(
            f"{timing}: {len(result[timing])} dates with "
            "complete exact-time minute quotes"
        )

    return result


def evaluate(group, calibration):
    probabilities = np.array(
        [
            probability_for_market(
                strike_type=row.strike_type,
                floor_strike=row.floor_strike,
                cap_strike=row.cap_strike,
                point_forecast=calibration["point"],
                residual_mean=calibration["residual_mean"],
                residual_std=calibration["residual_std"],
            )
            for row in group.itertuples(index=False)
        ],
        dtype=float,
    )

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-6)
    ):
        raise ValueError("Invalid weather bucket distribution.")

    outcomes = group["outcome"].to_numpy(dtype=float)
    winner = int(np.argmax(outcomes))

    candidates = build_daily_candidates(
        group,
        yes_probabilities=probabilities,
    )
    candidates = candidates[
        candidates["entry_price"].gt(0)
        & candidates["entry_price"].lt(1)
    ]

    trade = select_daily_trade(
        candidates=candidates,
        minimum_edge=EDGE,
    )

    metrics = {
        "bucket_hit": int(np.argmax(probabilities)) == winner,
        "brier": float(np.square(probabilities - outcomes).sum()),
        "log_loss": float(
            -np.log(max(probabilities[winner], 1e-6))
        ),
        "temperature_error": (
            calibration["point"]
            + calibration["residual_mean"]
            - calibration["actual_high"]
        ),
    }

    return metrics, trade


def run(args):
    if not os.environ.get("PREDICTION_DATABASE_URL", "").strip():
        raise RuntimeError("PREDICTION_DATABASE_URL is missing.")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    score_start = pd.Timestamp(args.score_start)

    if not start <= score_start <= end:
        raise ValueError("Require start <= score-start <= end.")

    weather = load_weather(args.start, args.end)
    calibrated = {
        timing: calibrate(frame)
        for timing, frame in weather.items()
    }
    quotes = load_quotes(args.score_start, args.end)

    metrics = {timing: [] for timing in TIMINGS}
    trades = {timing: [] for timing in TIMINGS}
    exclusions = Counter()
    paired_dates = []

    for day in pd.date_range(score_start, end, freq="D"):
        if any(day not in calibrated[timing] for timing in TIMINGS):
            exclusions["missing forecast or calibration warmup"] += 1
            continue

        if any(day not in quotes[timing] for timing in TIMINGS):
            exclusions["missing complete exact-time quotes"] += 1
            continue

        available = all(
            calibrated[timing][day]["model_run"]
            + pd.Timedelta(hours=args.publication_lag_hours)
            <= decision_time(day, timing)
            for timing in TIMINGS
        )
        if not available:
            exclusions["forecast availability assumption"] += 1
            continue

        previous = quotes["previous_day"][day]
        morning = quotes["morning"][day]

        # Both snapshots must refer to identical contract definitions
        # and settlement labels in the same order.
        identity = [
            "ticker", "strike_type", "floor_strike",
            "cap_strike", "outcome",
        ]
        if not previous[identity].equals(morning[identity]):
            exclusions["contract mismatch"] += 1
            continue

        day_results = {}
        for timing in TIMINGS:
            try:
                day_results[timing] = evaluate(
                    quotes[timing][day],
                    calibrated[timing][day],
                )
            except ValueError as error:
                raise RuntimeError(
                    f"{day.date()} {timing}: {error}"
                ) from error

        paired_dates.append(day)

        for timing, (day_metrics, trade) in day_results.items():
            metrics[timing].append(
                {"target_date": day, **day_metrics}
            )
            if trade is not None:
                trades[timing].append(trade)

    print("\nMATCHED NBM WEATHER COMPARISON")
    print("Previous day 16:05 ET vs same day 10:00 ET")
    print("Direct weather probabilities; no beta or alpha.")
    print(
        f"Calibration minimums: {MIN_TRAIN_DAYS} bias dates, "
        f"{MIN_RESIDUAL_DAYS} prior out-of-sample residuals."
    )
    print("Both arms exclude the immediately preceding target date.")
    print(f"Matched scored dates: {len(paired_dates)}")

    for reason, count in sorted(exclusions.items()):
        print(f"Excluded — {reason}: {count}")

    if not paired_dates:
        print("\nNo matched dates. Review forecast and quote coverage.")
        return

    print(
        f"Scored range: {paired_dates[0]:%Y-%m-%d} "
        f"through {paired_dates[-1]:%Y-%m-%d}"
    )

    summaries = []

    for timing in TIMINGS:
        scored = pd.DataFrame(metrics[timing])
        trade_frame = pd.DataFrame(trades[timing])
        summary = summarize_strategy(trade_frame, EDGE)

        summaries.append({
            "timing": timing,
            "days": len(scored),
            "trades": summary["trades"],
            "passes": len(scored) - summary["trades"],
            "win_rate": summary["win_rate"],
            "net_pnl": summary["net_pnl"],
            "roi": summary["roi"],
            "max_drawdown": summary["max_drawdown"],
            "bucket_hit": scored["bucket_hit"].mean(),
            "brier": scored["brier"].mean(),
            "log_loss": scored["log_loss"].mean(),
            "temperature_mae": scored["temperature_error"].abs().mean(),
        })

        print(f"\n{timing} trades:")
        if trade_frame.empty:
            print("No qualifying trades.")
        else:
            print(trade_frame[
                [
                    "target_date", "ticker", "side",
                    "entry_price", "fee", "fair_probability",
                    "net_edge", "won", "net_pnl",
                ]
            ].to_string(index=False))

    print("\nCOMPARISON — rates below are decimals")
    print(
        pd.DataFrame(summaries).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-06-13")
    parser.add_argument("--end", required=True)
    parser.add_argument("--score-start", default="2026-08-01")
    parser.add_argument(
        "--publication-lag-hours", type=float, default=2.0
    )
    args = parser.parse_args()

    if args.publication_lag_hours < 0:
        parser.error("--publication-lag-hours must be nonnegative.")

    run(args)