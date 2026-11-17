
#!/usr/bin/env python3
"""V6-C2: Chronological OOF errors and empirical distributions.

Research only. Reuses V7 features and frozen point models.
Does not change existing models or submit trades.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

FOLDS = [
    ("2025-01-01", "2025-04-01"),
    ("2025-04-01", "2025-07-01"),
    ("2025-07-01", "2025-10-01"),
    ("2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-04-01"),
    ("2026-04-01", "2026-07-01"),
    ("2026-07-01", "2026-09-01"),
]

BANDWIDTHS = (0.20, 0.35, 0.50, 0.75)


def load_v7():
    path = Path(__file__).with_name(
        "train_knyc_v7_history.py"
    )

    spec = importlib.util.spec_from_file_location(
        "v7_for_v6c2", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def make_point_model(leaves):
    return HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=int(leaves),
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )


def predict(model, data, features):
    return (
        data["NYC_temp"].to_numpy(float)
        + model.predict(data[features])
    )


def make_oof(frame, features, leaves, horizon):
    pieces = []

    for begin, end in FOLDS:
        history = frame[
            frame["target_day"] < begin
        ]

        holdout = frame[
            (frame["target_day"] >= begin)
            & (frame["target_day"] < end)
        ]

        if len(history) < 300 or len(holdout) == 0:
            raise RuntimeError(
                f"Missing chronological fold {begin}"
            )

        model = make_point_model(leaves)

        model.fit(
            history[features],
            history["target"].to_numpy(float)
            - history["NYC_temp"].to_numpy(float),
        )

        forecast = predict(
            model, holdout, features
        )

        observed = (
            holdout["target"].to_numpy(float)
        )

        fold = pd.DataFrame({
            "target_time":
                holdout["target_time"].to_numpy(),
            "decision":
                holdout["decision"].to_numpy(),
            "horizon": horizon,
            "fold_start": begin,
            "training_examples": len(history),
            "actual_f": observed,
            "predicted_f": forecast,
            "error_f": observed - forecast,
        })

        pieces.append(fold)

        print(
            f"  fold {begin} to {end}: "
            f"train={len(history)} "
            f"OOF={len(holdout)} "
            f"MAE={np.abs(observed - forecast).mean():.4f}F",
            flush=True,
        )

    oof = pd.concat(
        pieces, ignore_index=True
    )

    if oof["target_time"].duplicated().any():
        raise RuntimeError(
            "Duplicate OOF target times"
        )

    if not np.isfinite(
        oof["error_f"].to_numpy(float)
    ).all():
        raise RuntimeError(
            "Invalid OOF residuals"
        )

    return oof


class ResidualDistribution:
    def __init__(
        self,
        name,
        errors,
        bandwidth=None,
        centered=False,
    ):
        self.name = name

        errors = np.asarray(
            errors, dtype=float
        )

        if (
            not np.isfinite(errors).all()
            or len(errors) < 100
        ):
            raise ValueError(
                "Need at least 100 finite residuals"
            )

        self.errors = errors
        self.bandwidth = bandwidth
        self.centered = centered

        if bandwidth is None:
            self.location = (
                float(errors.mean())
                if centered else 0.0
            )

            if centered:
                self.scale = max(
                    float(errors.std()), 0.1
                )
            else:
                self.scale = max(
                    float(
                        np.sqrt(
                            np.mean(errors ** 2)
                        )
                    ),
                    0.1,
                )

        else:
            self.location = np.nan
            self.scale = np.nan

            self.grid = np.linspace(
                -30.0, 30.0, 3001
            )

            residuals = (
                errors - errors.mean()
                if centered else errors
            )

            self.grid_cdf = np.empty(
                len(self.grid), dtype=float
            )

            for start in range(
                0, len(self.grid), 96
            ):
                xs = self.grid[start:start + 96]

                self.grid_cdf[
                    start:start + len(xs)
                ] = ndtr(
                    (
                        xs[:, None]
                        - residuals[None, :]
                    ) / bandwidth
                ).mean(axis=1)

            self.grid_cdf = (
                np.maximum.accumulate(
                    self.grid_cdf
                )
            )

    def cdf(self, delta):
        x = np.asarray(
            delta, dtype=float
        )

        if self.bandwidth is None:
            return ndtr(
                (x - self.location)
                / self.scale
            )

        return np.interp(
            x,
            self.grid,
            self.grid_cdf,
            left=0.0,
            right=1.0,
        )

    def bucket_probs(
        self, predicted, integers
    ):
        integers = np.asarray(
            integers, dtype=float
        )

        upper = self.cdf(
            integers + 0.5 - predicted
        )

        lower = self.cdf(
            integers - 0.5 - predicted
        )

        return np.maximum(
            upper - lower, 0.0
        )


def evaluate(distribution, data):
    actual = data[
        "actual_f"
    ].to_numpy(float)

    predicted = data[
        "predicted_f"
    ].to_numpy(float)

    residual = actual - predicted

    rounded = np.floor(
        actual + 0.5
    ).astype(int)

    actual_p = np.maximum(
        distribution.bucket_probs(
            predicted, rounded
        ),
        1e-12,
    )

    logloss = float(
        -np.log(actual_p).mean()
    )

    squared_sums = []
    captured_mass = []

    for pred, outcome in zip(
        predicted, rounded
    ):
        center = int(
            np.floor(pred + 0.5)
        )

        buckets = np.arange(
            min(center - 20, outcome),
            max(center + 20, outcome) + 1,
        )

        probabilities = (
            distribution.bucket_probs(
                pred, buckets
            )
        )

        squared_sums.append(
            float(
                np.sum(probabilities ** 2)
            )
        )

        captured_mass.append(
            float(probabilities.sum())
        )

    brier = float(
        np.mean(
            1.0
            - 2.0 * actual_p
            + np.asarray(squared_sums)
        )
    )

    p1 = float(
        distribution.cdf(1.0)
        - distribution.cdf(-1.0)
    )

    hit1 = (
        np.abs(residual) <= 1
    ).astype(float)

    one_brier = float(
        np.mean(
            (p1 - hit1) ** 2
        )
    )

    # Probability integral transform.
    pit = distribution.cdf(residual)

    coverage = {}

    for interval in (50, 80, 90):
        alpha = (
            100 - interval
        ) / 200.0

        coverage[
            f"cover_{interval}_pct"
        ] = float(
            100.0 * np.mean(
                (pit >= alpha)
                & (pit <= 1.0 - alpha)
            )
        )

    return {
        "distribution": distribution.name,
        "n": int(len(data)),
        "bucket_logloss": round(
            logloss, 5
        ),
        "bucket_brier": round(
            brier, 5
        ),
        "within_1_brier": round(
            one_brier, 5
        ),
        "pred_within_1_pct": round(
            100.0 * p1, 2
        ),
        "actual_within_1_pct": round(
            100.0 * hit1.mean(), 2
        ),
        "mean_captured_bucket_mass": round(
            float(
                np.mean(captured_mass)
            ),
            6,
        ),
        **{
            key: round(value, 2)
            for key, value in coverage.items()
        },
    }


def verify_parity(
    test,
    point_model,
    features,
    existing,
    horizon,
):
    reconstructed = pd.DataFrame({
        "target_time": test["target_time"],
        "reconstructed": predict(
            point_model, test, features
        ),
    })

    saved = existing[
        (existing["horizon"] == horizon)
        & (
            existing["variant"]
            == "V7_full_dedicated"
        )
    ][
        ["target_time", "prediction_f"]
    ]

    matched = reconstructed.merge(
        saved,
        on="target_time",
        validate="one_to_one",
    )

    if (
        len(matched) != len(test)
        or len(saved) != len(test)
    ):
        raise RuntimeError(
            "Saved point prediction timestamps mismatch"
        )

    maximum_difference = float(
        np.max(
            np.abs(
                matched["reconstructed"]
                - matched["prediction_f"]
            )
        )
    )

    print(
        "Frozen October parity: "
        f"max difference={maximum_difference:.9f}F"
    )

    if (
        not np.isfinite(maximum_difference)
        or maximum_difference > 0.01
    ):
        raise RuntimeError(
            "Frozen prediction parity failed"
        )


def residual_diagnostics(errors):
    x = np.asarray(
        errors, float
    )

    return {
        "n": int(len(x)),
        "mean_bias_f": round(
            float(np.mean(x)), 4
        ),
        "std_f": round(
            float(np.std(x)), 4
        ),
        "mae_f": round(
            float(np.mean(np.abs(x))), 4
        ),
        "skew": round(
            float(skew(x)), 4
        ),
        "excess_kurtosis": round(
            float(kurtosis(x)), 4
        ),
        "pct_error_below_minus_2": round(
            float(100 * (x < -2).mean()),
            2,
        ),
        "pct_error_above_plus_2": round(
            float(100 * (x > 2).mean()),
            2,
        ),
        "q05_f": round(
            float(np.quantile(x, 0.05)),
            4,
        ),
        "median_f": round(
            float(np.median(x)), 4
        ),
        "q95_f": round(
            float(np.quantile(x, 0.95)),
            4,
        ),
    }


def self_test():
    errors = np.array(
        [0.0, 1.0, -1.0] * 50
    )

    gaussian = ResidualDistribution(
        "test_gaussian", errors
    )

    empirical = ResidualDistribution(
        "test_kde",
        errors,
        bandwidth=0.35,
    )

    for distribution in (
        gaussian, empirical
    ):
        assert abs(
            float(
                distribution.cdf(0)
            ) - 0.5
        ) < 0.01

        masses = (
            distribution.bucket_probs(
                65.0,
                np.arange(45, 86),
            )
        )

        assert abs(
            float(masses.sum()) - 1.0
        ) < 0.0001

        assert np.all(
            masses >= 0
        )

    assert abs(
        float(
            gaussian.bucket_probs(
                65.0, 65
            )
        ) - 0.4597
    ) < 0.02

    print("SELF-TEST PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--airport-lag",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if min(
        args.knyc_lag,
        args.airport_lag,
    ) < 0:
        parser.error(
            "Lags must be nonnegative"
        )

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    source = (
        ROOT
        / "v7_extended_history"
        / "training_results"
        / tag
    )

    bundle = joblib.load(
        source / "research_models.joblib"
    )

    existing = pd.read_csv(
        source / "predictions.csv"
    )

    existing["target_time"] = (
        pd.to_datetime(
            existing["target_time"],
            utc=True,
        )
    )

    v7 = load_v7()

    print(
        "Reconstructing expanded-history features...",
        flush=True,
    )

    grid = v7.build_grid(
        v7.import_v3(),
        args.knyc_lag,
        args.airport_lag,
    )

    output = (
        ROOT
        / "v6_c2_empirical_distributions"
        / tag
    )

    output.mkdir(
        parents=True, exist_ok=True
    )

    all_oof = []
    all_scores = []
    all_buckets = []
    diagnostics = []
    choices = {}

    for horizon in (30, 45):
        print(
            f"\n=== {horizon}-MINUTE "
            "OOF RESIDUALS ===",
            flush=True,
        )

        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        entry = bundle["models"][
            f"{horizon}m_full_dedicated"
        ]

        features = entry["features"]
        leaves = entry["leaves"]

        oct_frame = frame[
            frame["target_day"].between(
                "2026-10-01",
                "2026-10-26",
            )
        ]

        verify_parity(
            oct_frame,
            entry["model"],
            features,
            existing,
            horizon,
        )

        oof = make_oof(
            frame,
            features,
            leaves,
            horizon,
        )

        all_oof.append(oof)

        errors = (
            oof["error_f"].to_numpy(float)
        )

        description = {
            "horizon": horizon,
            **residual_diagnostics(errors),
        }

        diagnostics.append(description)

        print(
            "OOF residual diagnostics:",
            description,
            flush=True,
        )

        sep_frame = frame[
            frame["target_day"].between(
                "2026-09-01",
                "2026-09-30",
            )
        ]

        september = pd.DataFrame({
            "target_time":
                sep_frame["target_time"].to_numpy(),
            "actual_f":
                sep_frame["target"].to_numpy(float),
            "predicted_f":
                predict(
                    entry["model"],
                    sep_frame,
                    features,
                ),
        })

        october = pd.DataFrame({
            "target_time":
                oct_frame["target_time"].to_numpy(),
            "actual_f":
                oct_frame["target"].to_numpy(float),
            "predicted_f":
                predict(
                    entry["model"],
                    oct_frame,
                    features,
                ),
        })

        if (
            len(september) < 100
            or len(october) < 100
        ):
            raise RuntimeError(
                "Insufficient September or October cases"
            )

        candidates = [
            ResidualDistribution(
                "gaussian_oof_zero_mean",
                errors,
            ),
            ResidualDistribution(
                "gaussian_oof_with_bias",
                errors,
                centered=True,
            ),
        ]

        candidates.extend(
            ResidualDistribution(
                f"empirical_kde_bw_{bw:.2f}",
                errors,
                bandwidth=bw,
            )
            for bw in BANDWIDTHS
        )

        sep_scores = []

        for distribution in candidates:
            score = evaluate(
                distribution,
                september,
            )

            score.update({
                "horizon": horizon,
                "period": "September",
            })

            all_scores.append(score)
            sep_scores.append(score)

        best = min(
            sep_scores,
            key=lambda x: (
                x["bucket_logloss"],
                x["bucket_brier"],
            ),
        )

        winner = next(
            distribution
            for distribution in candidates
            if distribution.name
            == best["distribution"]
        )

        choices[horizon] = winner.name

        print(
            "September-selected distribution:",
            winner.name,
            flush=True,
        )

        # Previous V6-C Gaussian reference.
        # It uses September residuals and is not
        # eligible for September model selection.
        september_errors = (
            september["actual_f"].to_numpy()
            - september["predicted_f"].to_numpy()
        )

        reference = ResidualDistribution(
            "gaussian_september_reference",
            september_errors,
        )

        for distribution in (
            candidates + [reference]
        ):
            score = evaluate(
                distribution, october
            )

            score.update({
                "horizon": horizon,
                "period": "October",
            })

            all_scores.append(score)

        # Complete integer-temperature bucket
        # distributions for the selected candidate.
        for row in october.itertuples(
            index=False
        ):
            buckets = np.arange(
                -30, 131
            )

            probabilities = (
                winner.bucket_probs(
                    row.predicted_f,
                    buckets,
                )
            )

            all_buckets.append(
                pd.DataFrame({
                    "target_time":
                        row.target_time,
                    "horizon": horizon,
                    "point_forecast_f":
                        row.predicted_f,
                    "actual_precise_f":
                        row.actual_f,
                    "temperature_bucket_f":
                        buckets,
                    "probability":
                        probabilities,
                    "distribution":
                        winner.name,
                })
            )

    pd.concat(
        all_oof, ignore_index=True
    ).to_csv(
        output
        / "chronological_oof_residuals.csv",
        index=False,
    )

    pd.DataFrame(
        diagnostics
    ).to_csv(
        output / "residual_diagnostics.csv",
        index=False,
    )

    scores = pd.DataFrame(
        all_scores
    )

    scores.to_csv(
        output / "distribution_comparison.csv",
        index=False,
    )

    pd.concat(
        all_buckets, ignore_index=True
    ).to_csv(
        output
        / "october_integer_bucket_probabilities.csv",
        index=False,
    )

    with (
        output / "selected_distributions.json"
    ).open("w") as file:
        json.dump(
            {
                "selected_on_september": choices,
                "bandwidths_f": BANDWIDTHS,
                "notes":
                    "October exploratory; "
                    "frozen point predictors",
            },
            file,
            indent=2,
        )

    print(
        "\n=== SEPTEMBER MODEL SELECTION ==="
    )

    print(
        scores[
            scores["period"] == "September"
        ].sort_values(
            ["horizon", "bucket_logloss"]
        ).to_string(index=False)
    )

    print(
        "\n=== OCTOBER DISTRIBUTION COMPARISON ==="
    )

    print(
        scores[
            scores["period"] == "October"
        ].sort_values(
            ["horizon", "bucket_logloss"]
        ).to_string(index=False)
    )

    print(
        "\nSelected:",
        choices,
    )

    print(
        "Saved to:",
        output.resolve(),
    )

    print(
        "October is exploratory; "
        "no live or trading model was changed."
    )


if __name__ == "__main__":
    main()
