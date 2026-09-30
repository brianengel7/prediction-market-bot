import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.backtest.hourly_dataset import (
    build_hourly_dataset,
    latest,
    utc,
)
from src.database.db import get_connection, read_dataframe
from src.kalshi.settlements import parse_market_settlement

START = "2026-07-02"
END = "2026-09-29"
TRAIN_END = "2026-08-28"
VALID_START = "2026-08-31"
VALID_END = "2026-09-12"
TEST_START = "2026-09-15"

FIT_TIME = pd.Timestamp(
    VALID_START, tz="America/New_York"
).tz_convert("UTC")

KEYS = ["target_date", "decision_time_utc", "ticker"]
HOUR_KEYS = ["target_date", "decision_time_utc"]
LAGS = (30, 60)


def settlement_time(value):
    if value is None or pd.isna(value) or value == "":
        return pd.NaT
    if isinstance(value, (int, float)):
        return pd.to_datetime(
            value, unit="s", utc=True, errors="coerce"
        )
    return pd.to_datetime(value, utc=True, errors="coerce")


def label_ready_times():
    with get_connection() as connection:
        contracts = read_dataframe(
            """
            SELECT ticker, target_date, raw_market,
                   payload_hash, downloaded_at_utc
            FROM weather_market_contract_history
            WHERE target_date >= ? AND target_date <= ?
            """,
            connection,
            params=(START, TRAIN_END),
        )

    contracts = latest(contracts, ["ticker"])
    parsed = (
        contracts["raw_market"]
        .map(json.loads)
        .map(parse_market_settlement)
    )

    if not parsed.map(lambda x: x["is_settled"]).all():
        raise RuntimeError(
            "Some training contracts are not finalized."
        )

    contracts["settled_at_utc"] = utc(
        parsed.map(
            lambda x: settlement_time(x["settlement_time"])
        )
    )

    clock = contracts.groupby(
        "target_date"
    )["settled_at_utc"].agg(["max", "count", "size"])

    # Require timestamps for every contract belonging to the date.
    return clock["max"].where(
        clock["count"].eq(clock["size"])
    )


def prepare(frame):
    if frame.empty:
        raise RuntimeError(
            "A training or validation sample is empty."
        )

    frame = frame.sort_values(KEYS).reset_index(drop=True)
    types = frame["strike_type"]

    if not types.isin(["less", "between", "greater"]).all():
        raise RuntimeError(
            "Unexpected strike types in the hourly dataset."
        )

    capped = types.isin(["less", "between"])

    if (capped & frame["cap_strike"].isna()).any():
        raise RuntimeError(
            "A capped contract has no upper strike."
        )

    groups = frame.groupby(
        HOUR_KEYS, sort=False
    ).ngroup().to_numpy(dtype=int)

    hours = frame[HOUR_KEYS].drop_duplicates()
    hour_days, dates = pd.factorize(
        hours["target_date"], sort=False
    )
    n_hours = len(hours)
    y = frame["yes_payout"].to_numpy(dtype=float)

    if not frame["yes_payout"].isin([0.0, 1.0]).all():
        raise RuntimeError("Only binary labels are supported.")

    winners = np.bincount(
        groups, weights=y, minlength=n_hours
    )
    if not np.allclose(winners, 1.0):
        raise RuntimeError(
            "Every hourly group must contain one winner."
        )

    overshoot = np.where(
        capped.to_numpy(),
        frame["observed_high_minus_cap"]
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float),
        0.0,
    )

    if not np.isfinite(overshoot).all():
        raise RuntimeError(
            "Nonfinite temperature or strike features."
        )

    # Each date has equal total weight, regardless of its hour count.
    weights = 1.0 / (
        len(dates) * np.bincount(hour_days)[hour_days]
    )

    return {
        "log_market": np.log(
            frame["market_probability"]
            .clip(1e-12, 1)
            .to_numpy(dtype=float)
        ),
        "overshoot": overshoot,
        "y": y,
        "groups": groups,
        "n_hours": n_hours,
        "hour_days": hour_days,
        "dates": dates,
        "weights": weights,
    }


def log_probabilities(sample, alpha, beta=0.0):
    # alpha calibrates market probabilities.
    # beta applies a soft weather penalty to capped contracts.
    z = (
        alpha * sample["log_market"]
        - beta * sample["overshoot"]
    )

    groups = sample["groups"]
    maximum = np.full(sample["n_hours"], -np.inf)
    np.maximum.at(maximum, groups, z)

    total = np.bincount(
        groups,
        weights=np.exp(z - maximum[groups]),
        minlength=sample["n_hours"],
    )

    return z - (maximum + np.log(total))[groups]


def hourly_losses(sample, alpha, beta=0.0):
    logp = log_probabilities(sample, alpha, beta)
    groups = sample["groups"]
    n = sample["n_hours"]

    logloss = np.bincount(
        groups,
        weights=-sample["y"] * logp,
        minlength=n,
    )
    brier = np.bincount(
        groups,
        weights=(np.exp(logp) - sample["y"]) ** 2,
        minlength=n,
    )

    return logloss, brier


def objective(sample, alpha, beta=0.0):
    logloss, _ = hourly_losses(sample, alpha, beta)
    return float(logloss @ sample["weights"])


def fit_parameter(function, lower, upper, reference):
    result = minimize_scalar(
        function,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-6, "maxiter": 300},
    )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"Parameter fitting failed: {result.message}"
        )

    # Include the unchanged model and both endpoints.
    choices = [lower, upper, reference, float(result.x)]
    return float(min(choices, key=function))


def scores(sample, alpha, beta=0.0):
    logloss, brier = hourly_losses(sample, alpha, beta)

    return {
        "dates": len(sample["dates"]),
        "hours": sample["n_hours"],
        "log_loss": float(logloss @ sample["weights"]),
        "multiclass_brier": float(brier @ sample["weights"]),
    }


def daily_logloss(sample, alpha, beta=0.0):
    logloss, _ = hourly_losses(sample, alpha, beta)
    day_codes = sample["hour_days"]

    means = np.bincount(
        day_codes,
        weights=logloss,
        minlength=len(sample["dates"]),
    ) / np.bincount(day_codes)

    return pd.Series(means, index=sample["dates"])


def main():
    print(
        "Building both publication-delay scenarios...",
        flush=True,
    )

    data = {
        lag: build_hourly_dataset(
            START, END, observation_lag_minutes=lag
        )
        for lag in LAGS
    }

    # Every model is evaluated on the same eligible hours.
    common = data[30].loc[
        data[30]["weather_usable"], KEYS
    ].merge(
        data[60].loc[data[60]["weather_usable"], KEYS],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if common.empty:
        raise RuntimeError(
            "No weather-usable hours are shared by both scenarios."
        )

    data = {
        lag: common.merge(
            frame, on=KEYS, validate="one_to_one"
        ).sort_values(KEYS).reset_index(drop=True)
        for lag, frame in data.items()
    }

    for column in ("market_probability", "yes_payout"):
        if not np.allclose(data[30][column], data[60][column]):
            raise RuntimeError(
                f"The scenarios disagree on {column}."
            )

    ready = label_ready_times()
    train_dates = data[30]["target_date"].le(TRAIN_END)

    settled_before_fit = utc(
        data[30]["target_date"].map(ready)
    ).lt(FIT_TIME)

    training = train_dates & settled_before_fit
    validation = data[30]["target_date"].between(
        VALID_START, VALID_END
    )

    training_dates = data[30].loc[
        training, "target_date"
    ].nunique()

    if training_dates < 30:
        raise RuntimeError(
            f"Only {training_dates} training dates have verified "
            "settlement timestamps before the fit time. "
            "Inspect the archived settlement_ts fields."
        )

    train = {
        lag: prepare(frame.loc[training])
        for lag, frame in data.items()
    }
    valid = {
        lag: prepare(frame.loc[validation])
        for lag, frame in data.items()
    }

    alpha = fit_parameter(
        lambda a: objective(train[30], a),
        lower=0.25,
        upper=4.0,
        reference=1.0,
    )

    # Fix market calibration while fitting one weather parameter.
    betas = {
        lag: fit_parameter(
            lambda b: objective(train[lag], alpha, b),
            lower=0.0,
            upper=2.0,
            reference=0.0,
        )
        for lag in LAGS
    }

    shared_hours = len(
        common[HOUR_KEYS].drop_duplicates()
    )
    print(f"Shared weather-usable hours: {shared_hours}")
    print(
        f"Training: {START} -> {TRAIN_END} "
        f"({training_dates} dates)"
    )

    excluded = data[30].loc[
        train_dates & ~settled_before_fit, "target_date"
    ].nunique()
    print(
        f"Training dates excluded by settlement timing: {excluded}"
    )
    print(f"Validation: {VALID_START} -> {VALID_END}")
    print(
        f"Reserved test: {TEST_START} -> {END} (not scored)"
    )
    print(
        f"Market alpha: {alpha:.6f} [allowed: 0.25 -> 4.0]"
    )

    for lag in LAGS:
        print(
            f"Weather beta, {lag}m lag: {betas[lag]:.6f} "
            "[allowed: 0 -> 2.0]"
        )

    rows = []

    for name, samples in (
        ("training", train),
        ("validation", valid),
    ):
        models = [
            ("raw_market", 30, 1.0, 0.0),
            ("calibrated_market", 30, alpha, 0.0),
            ("weather_30m", 30, alpha, betas[30]),
            ("weather_60m", 60, alpha, betas[60]),
        ]

        for model, lag, a, b in models:
            rows.append({
                "sample": name,
                "model": model,
                **scores(samples[lag], a, b),
            })

    print(
        "\nProbability scores: lower is better; "
        "each date has equal weight."
    )
    print(pd.DataFrame(rows).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    baseline = daily_logloss(valid[30], alpha)
    daily = pd.DataFrame({"calibrated_market": baseline})

    for lag in LAGS:
        weather = daily_logloss(
            valid[lag], alpha, betas[lag]
        )
        daily[f"weather_{lag}m"] = weather
        daily[f"improvement_{lag}m"] = baseline - weather

    print(
        "\nValidation by date: "
        "positive improvement favors weather."
    )
    print(daily.to_string(
        float_format=lambda x: f"{x:.6f}"
    ))

    print("\nParameters to retain for the reserved test:")
    print(json.dumps({
        "alpha": alpha,
        "betas": {
            str(lag): betas[lag] for lag in LAGS
        },
        "train_end": TRAIN_END,
        "validation_start": VALID_START,
        "validation_end": VALID_END,
        "test_start": TEST_START,
        "test_end": END,
    }, indent=2))


if __name__ == "__main__":
    main()