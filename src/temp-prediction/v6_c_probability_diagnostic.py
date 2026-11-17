
#!/usr/bin/env python3
"""V6-C initial probability calibration diagnostic.

Frozen expanded-history model.
September 2026 calibrates uncertainty.
October 2026 evaluates probabilities.

Research only. No changes to live trading.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

EXT = ROOT / "v7_extended_history"


def load_v7():
    path = Path(__file__).with_name(
        "train_knyc_v7_history.py"
    )

    spec = importlib.util.spec_from_file_location(
        "v7_for_v6c", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def bucket_probability(mean, sigma, bucket):
    """Probability of a rounded Fahrenheit bucket."""

    upper = ndtr(
        (bucket + 0.5 - mean) / sigma
    )

    lower = ndtr(
        (bucket - 0.5 - mean) / sigma
    )

    return np.maximum(
        upper - lower, 1e-12
    )


def calibrate(september):
    error = (
        september["actual_f"].to_numpy(float)
        - september["predicted_f"].to_numpy(float)
    )

    if (
        len(error) < 100
        or not np.isfinite(error).all()
    ):
        raise ValueError(
            "Insufficient valid September residuals"
        )

    overall_variance = float(
        np.mean(error ** 2)
    )

    global_sigma = max(
        0.10, np.sqrt(overall_variance)
    )

    changing = (
        september["NYC_trend1"]
        .abs()
        .fillna(0)
        .to_numpy(float)
        >= 1.5
    )

    regime_sigma = {}

    # Shrink smaller groups toward the global
    # September variance to reduce overfitting.
    for group in (False, True):
        subset = error[changing == group]

        variance = (
            np.sum(subset ** 2)
            + 40 * overall_variance
        ) / (len(subset) + 40)

        regime_sigma[group] = max(
            0.10, np.sqrt(variance)
        )

    return global_sigma, regime_sigma


def evaluate(october, sigma, name):
    actual = october[
        "actual_f"
    ].to_numpy(float)

    predicted = october[
        "predicted_f"
    ].to_numpy(float)

    sigma = np.broadcast_to(
        np.asarray(sigma, dtype=float),
        predicted.shape,
    )

    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(sigma).all()
        or (sigma <= 0).any()
    ):
        raise ValueError(
            "Invalid predictions or uncertainty"
        )

    rounded = np.floor(
        actual + 0.5
    ).astype(int)

    p_actual_bucket = bucket_probability(
        predicted, sigma, rounded
    )

    logloss = float(
        -np.mean(np.log(p_actual_bucket))
    )

    # Multiclass Brier score over integer
    # Fahrenheit outcomes.
    buckets = np.arange(
        -50, 151, dtype=float
    )

    upper = ndtr(
        (
            buckets[None, :] + 0.5
            - predicted[:, None]
        ) / sigma[:, None]
    )

    lower = ndtr(
        (
            buckets[None, :] - 0.5
            - predicted[:, None]
        ) / sigma[:, None]
    )

    probabilities = upper - lower

    if probabilities.sum(axis=1).min() < 0.999:
        raise ValueError(
            "Actual temperature outside scoring range"
        )

    multiclass_brier = float(
        np.mean(
            (probabilities ** 2).sum(axis=1)
            + 1
            - 2 * p_actual_bucket
        )
    )

    p_within_one = (
        2 * ndtr(1 / sigma) - 1
    )

    hit_within_one = (
        np.abs(actual - predicted) <= 1
    ).astype(float)

    hit_brier = float(
        np.mean(
            (p_within_one - hit_within_one) ** 2
        )
    )

    coverage = {}

    for label, z in (
        ("50", 0.67448975),
        ("80", 1.28155157),
        ("90", 1.64485363),
    ):
        coverage[label] = round(
            float(
                np.mean(
                    np.abs(actual - predicted)
                    <= z * sigma
                ) * 100
            ),
            1,
        )

    metrics = {
        "calibration": name,
        "n": len(actual),
        "sigma_mean_f": round(
            float(sigma.mean()), 4
        ),
        "point_mae_f": round(
            float(
                np.mean(
                    np.abs(actual - predicted)
                )
            ),
            4,
        ),
        "point_within_one_pct": round(
            float(hit_within_one.mean() * 100),
            1,
        ),
        "bucket_logloss": round(
            logloss, 4
        ),
        "bucket_brier": round(
            multiclass_brier, 4
        ),
        "within_one_brier": round(
            hit_brier, 4
        ),
        "interval_50_coverage_pct":
            coverage["50"],
        "interval_80_coverage_pct":
            coverage["80"],
        "interval_90_coverage_pct":
            coverage["90"],
    }

    predictions = october[
        [
            "target_time",
            "horizon",
            "actual_f",
            "predicted_f",
            "NYC_trend1",
        ]
    ].copy()

    predictions["calibration"] = name
    predictions["sigma_f"] = sigma

    predictions["p_within_one_f"] = (
        p_within_one
    )

    predictions["realized_within_one"] = (
        hit_within_one.astype(int)
    )

    predictions["rounded_target_f"] = rounded

    predictions["p_true_rounded_bucket"] = (
        p_actual_bucket
    )

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag", type=int, default=5
    )

    parser.add_argument(
        "--airport-lag", type=int, default=6
    )

    parser.add_argument(
        "--self-test", action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        p = bucket_probability(
            np.array([60.0]),
            np.array([1.0]),
            np.array([60.0]),
        )

        assert abs(
            float(p[0]) - 0.3829249225480262
        ) < 1e-9

        print("SELF-TEST PASS")
        return

    if min(
        args.knyc_lag, args.airport_lag
    ) < 0:
        parser.error(
            "Lags must be nonnegative"
        )

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    source = (
        EXT / "training_results" / tag
    )

    bundle = joblib.load(
        source / "research_models.joblib"
    )

    saved = pd.read_csv(
        source / "predictions.csv"
    )

    saved["target_time"] = pd.to_datetime(
        saved["target_time"], utc=True
    )

    v7 = load_v7()

    grid = v7.build_grid(
        v7.import_v3(),
        args.knyc_lag,
        args.airport_lag,
    )

    results = []
    output = []

    for horizon in (30, 45):
        entry = bundle["models"][
            f"{horizon}m_full_dedicated"
        ]

        model = entry["model"]
        features = entry["features"]

        eligible = grid[
            grid["horizon"] == horizon
        ].copy()

        eligible["actual_f"] = (
            eligible["target"].astype(float)
        )

        eligible["predicted_f"] = (
            eligible["NYC_temp"]
            .to_numpy(float)
            + model.predict(
                eligible[features]
            )
        )

        september = eligible[
            eligible["target_day"].between(
                "2026-09-01",
                "2026-09-30",
            )
        ].copy()

        october = eligible[
            eligible["target_day"].between(
                "2026-10-01",
                "2026-10-26",
            )
        ].copy()

        if (
            len(september) < 100
            or len(october) < 100
        ):
            raise RuntimeError(
                "Insufficient calibration samples"
            )

        # Confirm identical point predictions
        # against the frozen expanded-history model.
        prior = saved[
            (
                saved["horizon"]
                == horizon
            )
            & (
                saved["variant"]
                == "V7_full_dedicated"
            )
        ][
            ["target_time", "prediction_f"]
        ]

        check = october[
            ["target_time", "predicted_f"]
        ].merge(
            prior,
            on="target_time",
            validate="one_to_one",
        )

        if len(check) != len(october):
            raise RuntimeError(
                "Frozen model target mismatch"
            )

        difference = float(
            (
                check["predicted_f"]
                - check["prediction_f"]
            ).abs().max()
        )

        if difference > 0.01:
            raise RuntimeError(
                "Frozen model parity failed"
            )

        global_sigma, regime_sigma = calibrate(
            september
        )

        changing = (
            october["NYC_trend1"]
            .abs()
            .fillna(0)
            .to_numpy(float)
            >= 1.5
        )

        conditional_sigma = np.where(
            changing,
            regime_sigma[True],
            regime_sigma[False],
        )

        print(
            f"\n=== {horizon}m "
            f"V6-C CALIBRATION ==="
        )

        print(
            "Frozen parity difference:",
            f"{difference:.9f}F"
        )

        print(
            "September/October:",
            len(september), len(october)
        )

        print(
            "Global sigma:",
            f"{global_sigma:.4f}F"
        )

        print(
            "Stable/changing sigma:",
            f"{regime_sigma[False]:.4f}F / "
            f"{regime_sigma[True]:.4f}F"
        )

        for name, sigma in (
            ("global_gaussian", global_sigma),
            ("regime_gaussian", conditional_sigma),
        ):
            metrics, predictions = evaluate(
                october, sigma, name
            )

            metrics["horizon"] = horizon

            results.append(metrics)
            output.append(predictions)

    out = (
        ROOT
        / "v6_probability_calibration"
        / tag
    )

    out.mkdir(
        parents=True, exist_ok=True
    )

    summary = pd.DataFrame(results)

    summary.to_csv(
        out / "comparison.csv",
        index=False,
    )

    pd.concat(
        output, ignore_index=True
    ).to_csv(
        out / "predictions.csv",
        index=False,
    )

    print(
        "\n=== V6-C OCTOBER "
        "PROBABILITY EVALUATION ==="
    )

    print(
        summary.to_string(index=False)
    )

    print("Saved:", out)

    print(
        "Calibrated on September only. "
        "October evaluation remains exploratory."
    )


if __name__ == "__main__":
    main()
