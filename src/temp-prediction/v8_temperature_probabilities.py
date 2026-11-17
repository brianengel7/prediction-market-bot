#!/usr/bin/env python3
"""V8 probabilistic temperature forecasts. Research only."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

WALK = (
    ROOT / "v8_gfs/training_results/"
    "knyc5_airports6/walk_forward_monthly/predictions.csv"
)

HOLDOUT = (
    ROOT / "v8_unseen_holdout_20261027_20261108/"
    "evaluation_frozen/predictions.csv"
)

# Fixed initial smoothing assumption, in Fahrenheit.
# This must be evaluated before using probabilities to trade.
BANDWIDTH = 0.30


def residuals(horizon, variant):
    df = pd.read_csv(WALK)

    subset = df[
        (df["horizon"] == horizon)
        & (df["variant"] == variant)
        & df["month"].isin([
            "August", "September", "October"
        ])
    ]

    errors = (
        subset["actual_f"].to_numpy(float)
        - subset["predicted_f"].to_numpy(float)
    )

    errors = errors[np.isfinite(errors)]

    if len(errors) < 500:
        raise RuntimeError(
            f"Insufficient historical errors: {len(errors)}"
        )

    return errors


def cdf(temperature, forecast, errors):
    """P(precise temperature < temperature)."""
    centers = forecast + errors
    return float(
        ndtr((temperature - centers) / BANDWIDTH).mean()
    )


def rounded_probability(degree, forecast, errors):
    """P(temperature rounds to this whole Fahrenheit degree)."""
    return (
        cdf(degree + 0.5, forecast, errors)
        - cdf(degree - 0.5, forecast, errors)
    )


def at_least_probability(degree, forecast, errors):
    """P(temperature rounded to nearest F >= degree)."""
    return 1 - cdf(degree - 0.5, forecast, errors)


def show_forecast(forecast, horizon, at_least):
    errors = residuals(
        horizon, "gfs_delta_plus_bias"
    )

    print("\n" + "=" * 58)
    print("V8 KNYC TEMPERATURE PROBABILITY FORECAST")
    print("=" * 58)

    print(f"Horizon: {horizon} minutes")
    print(f"V8 point forecast: {forecast:.3f}F")
    print(f"Historical predictions: {len(errors)}")
    print(f"Historical signed error: {errors.mean():+.3f}F")
    print(f"Historical MAE: {np.abs(errors).mean():.3f}F")

    q10, q90 = np.quantile(errors, [0.10, 0.90])

    print(
        f"\nEmpirical 80% predictive interval: "
        f"{forecast + q10:.2f}F to {forecast + q90:.2f}F"
    )

    first = int(np.floor(forecast)) - 5
    last = first + 11

    print("\nPROBABILITY OF EACH ROUNDED TEMPERATURE")
    print("-" * 45)

    shown = 0.0

    for degree in range(first, last):
        probability = rounded_probability(
            degree, forecast, errors
        )
        shown += probability

        bar = "█" * int(round(probability * 100))
        print(
            f"{degree:3d}F | {probability * 100:6.2f}% | {bar}"
        )

    print(f"Other temperatures: {(1-shown)*100:.2f}%")

    if at_least is not None:
        probability = at_least_probability(
            at_least, forecast, errors
        )
        print(
            f"\nP(rounded temperature >= {at_least}F): "
            f"{probability * 100:.2f}%"
        )

    print("\nDistribution uses historical signed V8 errors.")
    print("Probabilities are preliminary until validated.")


def validate():
    """Evaluate fixed historical error distributions on holdout."""
    hold = pd.read_csv(HOLDOUT)

    print("\nOCT 27–NOV 8 PROBABILITY VALIDATION")
    print("Bandwidth:", BANDWIDTH, "F")
    print("Lower log loss is better.")

    for horizon in (30, 45):
        print(f"\n{horizon}-minute horizon")

        for variant in (
            "v7_baseline",
            "gfs_delta_plus_bias",
        ):
            errors = residuals(horizon, variant)

            test = hold[
                (hold["horizon"] == horizon)
                & (hold["variant"] == variant)
            ].copy()

            if len(test) != 117:
                raise RuntimeError(
                    f"Unexpected holdout count: {len(test)}"
                )

            probabilities = []
            covered = []

            low_error, high_error = np.quantile(
                errors, [0.10, 0.90]
            )

            for row in test.itertuples(index=False):
                actual_degree = int(
                    np.floor(row.actual_f + 0.5)
                )

                probability = rounded_probability(
                    actual_degree,
                    row.predicted_f,
                    errors,
                )

                probabilities.append(
                    max(probability, 1e-12)
                )

                covered.append(
                    row.predicted_f + low_error
                    <= row.actual_f
                    <= row.predicted_f + high_error
                )

            log_loss = -np.log(probabilities).mean()
            coverage = np.mean(covered) * 100

            print(
                f"  {variant:22s} "
                f"rounded log loss={log_loss:.4f} "
                f"80% interval coverage={coverage:.1f}%"
            )

    print("\nRead only: no model fitting or file writes.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--forecast", type=float
    )
    parser.add_argument(
        "--horizon", type=int, choices=[30, 45],
        default=30
    )
    parser.add_argument(
        "--at-least", type=int
    )
    parser.add_argument(
        "--validate", action="store_true"
    )

    args = parser.parse_args()

    if args.validate:
        validate()
    else:
        if args.forecast is None:
            parser.error(
                "Provide --forecast or use --validate"
            )

        show_forecast(
            args.forecast,
            args.horizon,
            args.at_least,
        )


if __name__ == "__main__":
    main()
