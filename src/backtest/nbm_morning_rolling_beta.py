import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from src.backtest.nbm_same_day_strategy import (
    load_morning_weather,
    load_morning_markets,
)
from src.weather.calibrated_ensemble import MIN_TRAIN_DAYS
from src.weather.fair_distribution import probability_for_market
from src.backtest.v2_source_signal_test import (
    BETA_GRID,
    fit_beta,
    adjusted_probabilities,
)
from src.backtest.v2_nbm_strategy import (
    build_daily_candidates,
    select_daily_trade,
    summarize_strategy,
)


NY = "America/New_York"
MINIMUM_EDGE = 0.025


def build_examples(weather, markets, args):
    """
    Build each historical morning's NBM probabilities using only
    weather outcomes from dates strictly before that morning.

    These examples are later used to fit beta.
    """
    examples = {}
    excluded = Counter()

    market_days = {
        day: group.copy()
        for day, group in markets.groupby("target_date")
    }

    for current in weather.itertuples(index=False):
        day = current.target_date

        decision = pd.Timestamp(
            f"{day:%Y-%m-%d} {args.hour:02d}:00:00",
            tz=NY,
        ).tz_convert("UTC")

        available = (
            current.model_run
            + pd.Timedelta(hours=args.publication_lag_hours)
        )
        if available > decision:
            excluded["forecast unavailable by decision"] += 1
            continue

        prior = weather[weather["target_date"].lt(day)]

        if len(prior) < MIN_TRAIN_DAYS:
            excluded["weather calibration warmup"] += 1
            continue

        bias = float(prior["error"].mean())
        spread = float(prior["error"].std(ddof=1))
        corrected = float(current.forecast_value) - bias

        if not np.isfinite(spread) or spread <= 0:
            excluded["invalid weather error spread"] += 1
            continue

        group = market_days.get(day)
        if group is None:
            excluded["missing morning quotes"] += 1
            continue

        group = group.sort_values("ticker").reset_index(drop=True)

        if (
            len(group) != 6
            or group["ticker"].nunique() != 6
            or not group["expected_contracts"].eq(6).all()
            or not group["decision_time_utc"].eq(decision).all()
            or not group["valid_quote"].all()
        ):
            excluded["incomplete or invalid quote set"] += 1
            continue

        if (
            not group["outcome"].isin([0.0, 1.0]).all()
            or group["outcome"].sum() != 1
        ):
            excluded["missing or invalid settlement"] += 1
            continue

        midpoint_sum = float(group["market_mid"].sum())
        if midpoint_sum <= 0:
            excluded["invalid market distribution"] += 1
            continue

        group["market_probability"] = (
            group["market_mid"] / midpoint_sum
        )

        group["NBM_yes"] = [
            probability_for_market(
                strike_type=row.strike_type,
                floor_strike=row.floor_strike,
                cap_strike=row.cap_strike,
                point_forecast=corrected,
                residual_mean=0.0,
                residual_std=spread,
            )
            for row in group.itertuples(index=False)
        ]

        probabilities = group["NBM_yes"].to_numpy(dtype=float)

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.isclose(
                probabilities.sum(), 1.0, atol=1e-6
            )
        ):
            excluded["invalid NBM distribution"] += 1
            continue

        examples[day] = {
            "group": group,
            "corrected_forecast": corrected,
            "actual_high": float(current.actual_high),
        }

    return examples, excluded


def run(args):
    if not os.environ.get("PREDICTION_DATABASE_URL", "").strip():
        raise RuntimeError("PREDICTION_DATABASE_URL is missing.")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    score_start = pd.Timestamp(args.score_start)

    if not start <= score_start <= end:
        raise ValueError("Require start <= score-start <= end.")

    # Load earlier market dates too: beta needs historical
    # morning quotes, not just quotes from the scoring period.
    weather = load_morning_weather(args.start, args.end)
    markets = load_morning_markets(
        args.start, args.end, args.hour
    )

    if weather.empty:
        raise RuntimeError("No morning NBM forecasts found.")

    examples, excluded = build_examples(weather, markets, args)
    dates = sorted(examples)

    decisions = []
    trades = []
    beta_warmup = 0

    for day in dates:
        if day < score_start:
            continue

        # Most recent AVAILABLE eligible dates, not necessarily
        # consecutive calendar days.
        prior_dates = [date for date in dates if date < day]
        training_dates = prior_dates[-args.lookback:]

        if len(training_dates) < args.lookback:
            beta_warmup += 1
            continue

        training = pd.concat(
            [examples[date]["group"] for date in training_dates],
            ignore_index=True,
        )

        # Existing beta grid and log-loss fitting method.
        # Today's outcomes are not passed to fit_beta().
        beta, training_loss = fit_beta(
            training_data=training,
            signal_column="NBM_yes",
        )

        example = examples[day]
        group = example["group"]

        probabilities = adjusted_probabilities(
            group=group,
            signal_column="NBM_yes",
            beta=beta,
        )
        outcome = group["outcome"].to_numpy(dtype=float)
        winner = int(np.argmax(outcome))

        decision = {
            "target_date": day,
            "beta": beta,
            "training_start": training_dates[0],
            "training_end": training_dates[-1],
            "training_dates": len(training_dates),
            "training_log_loss": training_loss,
            "bucket_hit": int(np.argmax(probabilities)) == winner,
            "brier": float(np.square(probabilities - outcome).sum()),
            "log_loss": float(
                -np.log(max(probabilities[winner], 1e-6))
            ),
            "temperature_error": (
                example["corrected_forecast"]
                - example["actual_high"]
            ),
            "action": "PASS",
        }

        # Original YES/NO trade logic and fees, using today's beta.
        candidates = build_daily_candidates(group, beta=beta)

        candidates = candidates[
            candidates["entry_price"].gt(0)
            & candidates["entry_price"].lt(1)
        ]

        trade = select_daily_trade(
            candidates=candidates,
            minimum_edge=MINIMUM_EDGE,
        )

        if trade is not None:
            trade["beta"] = beta
            trades.append(trade)
            decision["action"] = "TRADE"

        decisions.append(decision)

    print("\nMORNING NBM — ROLLING BETA")
    print(f"Decision time: {args.hour:02d}:00 Eastern")
    print(f"Forecast: same-day 12Z NBM")
    print(f"Beta lookback: {args.lookback} eligible dates")
    print(
        f"Beta grid: {min(BETA_GRID):+.2f} "
        f"to {max(BETA_GRID):+.2f}"
    )
    print(f"Minimum net edge: {MINIMUM_EDGE:.2%}")
    print(f"Morning forecasts loaded: {len(weather)}")
    print(f"Eligible historical examples: {len(examples)}")

    print("\nExample exclusions across the full input range:")
    for reason, count in sorted(excluded.items()):
        print(f"  {reason}: {count}")

    requested = pd.date_range(score_start, end, freq="D")
    unavailable = sum(day not in examples for day in requested)

    print(f"\nRequested scoring dates: {len(requested)}")
    print(f"Scoring dates without an eligible example: {unavailable}")
    print(f"Scoring dates awaiting beta warmup: {beta_warmup}")

    if not decisions:
        print(
            "\nNo scored dates. You need "
            f"{args.lookback} earlier eligible morning examples "
            "after the weather-calibration warmup."
        )
        return

    results = pd.DataFrame(decisions)
    trade_data = pd.DataFrame(trades)
    summary = summarize_strategy(trade_data, MINIMUM_EDGE)

    print(f"\nScored dates: {len(results)}")
    print(f"Passes: {results['action'].eq('PASS').sum()}")
    print(
        f"Actual scored range: "
        f"{results['target_date'].min():%Y-%m-%d} through "
        f"{results['target_date'].max():%Y-%m-%d}"
    )

    print("\nRecalculated beta:")
    print(f"Mean:   {results['beta'].mean():+.3f}")
    print(f"Median: {results['beta'].median():+.3f}")
    print(f"Min:    {results['beta'].min():+.3f}")
    print(f"Max:    {results['beta'].max():+.3f}")

    boundary = (
        np.isclose(results["beta"], min(BETA_GRID))
        | np.isclose(results["beta"], max(BETA_GRID))
    )
    print(f"Dates at grid boundary: {int(boundary.sum())}")

    print("\nWeather strategy probability accuracy:")
    print(f"Bucket hit rate: {results['bucket_hit'].mean():.2%}")
    print(f"Brier:          {results['brier'].mean():.4f}")
    print(f"Log loss:       {results['log_loss'].mean():.4f}")
    print(
        "Corrected NBM temperature MAE: "
        f"{results['temperature_error'].abs().mean():.3f} F"
    )

    print("\nTrading simulation — one contract per trade:")
    print(f"Trades:       {summary['trades']}")
    print(f"Wins:         {summary['wins']}")
    print(f"Win rate:     {summary['win_rate']:.2%}")
    print(f"Average edge: {summary['avg_edge']:.2%}")
    print(f"Total cost:   ${summary['capital']:.2f}")
    print(f"Net P&L:      ${summary['net_pnl']:.2f}")
    print(f"ROI on cost:  {summary['roi']:.2%}")
    print(f"Max drawdown: ${summary['max_drawdown']:.2f}")

    print("\nDaily beta and calibration window:")
    print(results[
        [
            "target_date", "beta", "training_start",
            "training_end", "training_dates",
            "training_log_loss", "action",
        ]
    ].to_string(index=False))

    if not trade_data.empty:
        print("\nTrades:")
        print(trade_data[
            [
                "target_date", "beta", "ticker", "side",
                "entry_price", "fee", "fair_probability",
                "net_edge", "won", "net_pnl",
            ]
        ].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-06-13")
    parser.add_argument("--end", required=True)
    parser.add_argument("--score-start", default="2026-09-27")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument(
        "--hour", type=int, choices=range(7, 12), default=10
    )
    parser.add_argument(
        "--publication-lag-hours", type=float, default=2.0
    )
    args = parser.parse_args()

    if args.lookback < 1:
        parser.error("--lookback must be positive.")
    if args.publication_lag_hours < 0:
        parser.error("--publication-lag-hours must be nonnegative.")

    run(args)