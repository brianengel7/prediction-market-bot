from decimal import Decimal, ROUND_CEILING

import numpy as np
import pandas as pd

from src.backtest.hourly_dataset import build_hourly_dataset
from src.backtest.hourly_market_only_backtest import (
    HOUR_KEYS, KEYS, prepare_market,
)
from src.backtest.hourly_probability_backtest import (
    VALID_END, VALID_START, log_probabilities,
)

ALPHA = 1.1202868518598814
MIN_NET_EDGE = 0.025  # 2.5 cents per contract after costs


def screening_fee(price):
    # One contract, multiplier 1, conservative whole-cent rounding.
    p = Decimal(str(price))
    return float(
        (Decimal("0.07") * p * (1 - p)).quantize(
            Decimal("0.01"), rounding=ROUND_CEILING
        )
    )


def candidates(frame, sample, alpha):
    probabilities = np.exp(log_probabilities(sample, alpha, 0.0))
    parts = []

    for side in ["YES", "NO"]:
        part = frame[KEYS].copy()
        part["side"] = side
        part["model_probability"] = (
            probabilities if side == "YES" else 1 - probabilities
        )
        part["ask"] = (
            frame["yes_ask"] if side == "YES" else 1 - frame["yes_bid"]
        )
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    result = result.loc[
        result["ask"].between(0, 1, inclusive="neither")
    ].copy()
    result["fee"] = result["ask"].map(screening_fee)
    result["net_edge"] = (
        result["model_probability"] - result["ask"] - result["fee"]
    )
    return result


def best_per_hour(frame):
    return (
        frame.sort_values(
            HOUR_KEYS + ["net_edge", "ticker", "side"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(HOUR_KEYS)
        .reset_index(drop=True)
    )


def main():
    full = build_hourly_dataset(
        VALID_START, VALID_END, observation_lag_minutes=30
    )
    frame = (
        full[
            KEYS + ["market_probability", "yes_payout", "yes_bid", "yes_ask"]
        ]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    sample = prepare_market(frame)

    # No weather adjustment enters these probabilities.
    sample["overshoot"] = np.zeros(len(frame))

    print(f"Validation: {VALID_START} -> {VALID_END}")
    print(
        f"Dates: {len(sample['dates'])}; "
        f"complete quote hours: {sample['n_hours']}"
    )
    print("One-contract cost screen; current multiplier 1 assumed.")
    print("Whole-cent fee rounding; additional cost buffers: 0c and 1c.")
    print("Reserved test remains unscored.")

    rows = []
    calibrated_best = None

    for model, alpha in [("raw_market", 1.0), ("calibrated_market", ALPHA)]:
        base = candidates(frame, sample, alpha)

        for buffer in [0.0, 0.01]:
            adjusted = base.copy()
            adjusted["net_edge"] -= buffer
            best = best_per_hour(adjusted)
            eligible = best.loc[
                best["net_edge"].ge(MIN_NET_EDGE - 1e-12)
            ]

            rows.append({
                "model": model,
                "extra_cost_cents": buffer * 100,
                "hours_net_positive": int(
                    best["net_edge"].gt(1e-12).sum()
                ),
                "hours_net_at_least_2_5c": len(eligible),
                "dates_net_at_least_2_5c": eligible["target_date"].nunique(),
                "max_net_edge_cents": (
                    float(best["net_edge"].max() * 100)
                    if not best.empty else np.nan
                ),
            })

            if model == "calibrated_market" and buffer == 0:
                calibrated_best = best

    print("\nBest candidate per hour, based on predicted net edge:")
    print(pd.DataFrame(rows).to_string(index=False))

    first = (
        calibrated_best.loc[
            calibrated_best["net_edge"].ge(MIN_NET_EDGE - 1e-12)
        ]
        .drop_duplicates("target_date", keep="first")
        .copy()
    )
    first["hour_local"] = (
        pd.to_datetime(first["decision_time_utc"], utc=True, format="mixed")
        .dt.tz_convert("America/New_York")
        .dt.hour
    )

    print("\nFirst qualifying calibrated opportunity per date, with 0c buffer:")
    columns = [
        "target_date", "hour_local", "ticker", "side",
        "model_probability", "ask", "fee", "net_edge",
    ]
    print(
        first[columns].to_string(index=False)
        if not first.empty else "None"
    )


if __name__ == "__main__":
    main()