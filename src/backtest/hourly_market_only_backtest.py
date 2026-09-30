import json

import numpy as np
import pandas as pd

from src.backtest.hourly_dataset import build_hourly_dataset, utc
from src.backtest.hourly_probability_backtest import (
    END, FIT_TIME, START, TEST_START, TRAIN_END, VALID_END, VALID_START,
    fit_parameter, label_ready_times,
)

MATCHED_ALPHA = 1.1116257546521748
HOUR_KEYS = ["target_date", "decision_time_utc"]
KEYS = HOUR_KEYS + ["ticker"]


def prepare_market(frame):
    if frame.empty:
        raise ValueError("No quote hours in this sample.")
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate contract-hour rows.")

    groups = frame.groupby(HOUR_KEYS, sort=False).ngroup().to_numpy(int)
    hours = frame[HOUR_KEYS].drop_duplicates()
    hour_days, dates = pd.factorize(hours["target_date"], sort=False)
    n_hours = len(hours)

    probabilities = frame["market_probability"].to_numpy(float)
    y = frame["yes_payout"].to_numpy(float)

    if not (
        np.isfinite(probabilities).all()
        and ((probabilities > 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("Invalid market probabilities.")

    if not np.allclose(
        np.bincount(groups, weights=probabilities),
        1.0,
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Market probabilities do not sum to one.")

    if not (
        np.isin(y, [0.0, 1.0]).all()
        and np.all(np.bincount(groups, weights=y) == 1)
    ):
        raise ValueError("Each quote hour must have one winning contract.")

    weights = 1.0 / (len(dates) * np.bincount(hour_days)[hour_days])

    return {
        "log_market": np.log(probabilities),
        "y": y,
        "groups": groups,
        "n_hours": n_hours,
        "dates": dates,
        "weights": weights,
    }


def losses(sample, alpha):
    groups = sample["groups"]
    z = alpha * sample["log_market"]

    maximum = np.full(sample["n_hours"], -np.inf)
    np.maximum.at(maximum, groups, z)
    shifted = z - maximum[groups]
    total = np.bincount(
        groups,
        weights=np.exp(shifted),
        minlength=sample["n_hours"],
    )
    logp = shifted - np.log(total)[groups]

    log_loss = np.bincount(groups, weights=-sample["y"] * logp)
    brier = np.bincount(
        groups,
        weights=(np.exp(logp) - sample["y"]) ** 2,
    )
    return log_loss, brier


def score(sample, alpha):
    log_loss, brier = losses(sample, alpha)
    return {
        "dates": len(sample["dates"]),
        "hours": sample["n_hours"],
        "log_loss": float(log_loss @ sample["weights"]),
        "multiclass_brier": float(brier @ sample["weights"]),
    }


def main():
    full = build_hourly_dataset(START, END, observation_lag_minutes=30)

    # Keep every complete quote hour and select only market inputs and labels.
    frame = full[KEYS + ["market_probability", "yes_payout"]].copy()

    ready = label_ready_times()
    train_dates = frame["target_date"].le(TRAIN_END)
    ready_before_fit = utc(frame["target_date"].map(ready)).lt(FIT_TIME)

    train_mask = train_dates & ready_before_fit
    valid_mask = frame["target_date"].between(VALID_START, VALID_END)

    excluded_dates = (
        frame.loc[train_dates, "target_date"].nunique()
        - frame.loc[train_mask, "target_date"].nunique()
    )
    if frame.loc[train_mask, "target_date"].nunique() < 30:
        raise ValueError(
            "Fewer than 30 training dates have verified settlement times."
        )

    train = prepare_market(frame.loc[train_mask])
    valid = prepare_market(frame.loc[valid_mask])

    alpha = fit_parameter(
        lambda a: float(losses(train, a)[0] @ train["weights"]),
        0.25,
        4.0,
        1.0,
    )

    print(f"Complete quote hours: {len(frame[HOUR_KEYS].drop_duplicates())}")
    print(f"Training dates excluded by settlement timing: {excluded_dates}")
    print(f"Training: {START} -> {TRAIN_END}")
    print(f"Validation: {VALID_START} -> {VALID_END}")
    print(f"Reserved test: {TEST_START} -> {END} (not scored)")
    print(f"Previous matched-hour alpha: {MATCHED_ALPHA:.6f}")
    print(f"Alpha fitted on all training quote hours: {alpha:.6f}")

    models = [
        ("raw_market", 1.0),
        ("previous_matched_alpha", MATCHED_ALPHA),
        ("all_hours_calibrated_market", alpha),
    ]

    rows = []
    for sample_name, sample in [("training", train), ("validation", valid)]:
        for model, parameter in models:
            rows.append({
                "sample": sample_name,
                "model": model,
                **score(sample, parameter),
            })

    print("\nMarket-only scores: lower is better; each date has equal weight.")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nAdditional parameter to retain:")
    print(json.dumps({"all_hours_alpha": alpha}, indent=2))


if __name__ == "__main__":
    main()