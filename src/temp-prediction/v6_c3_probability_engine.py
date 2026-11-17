
#!/usr/bin/env python3
"""Frozen V6-C3 KNYC temperature-probability engine.

Combines:
- V7 expanded-history point predictors
- V6-C2 chronological residual KDE
- V6-C3 September-selected KDE shift and scale

No trading, no network requests, no changes to existing models.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)


def import_local(filename, module_name):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProbabilityEngine:
    def __init__(self, knyc_lag=5, airport_lag=6):
        self.knyc_lag = knyc_lag
        self.airport_lag = airport_lag

        self.tag = (
            f"knyc{knyc_lag}"
            f"_airports{airport_lag}"
        )

        self.v7_dir = (
            ROOT / "v7_extended_history"
            / "training_results" / self.tag
        )
        self.c2_dir = (
            ROOT / "v6_c2_empirical_distributions"
            / self.tag
        )
        self.c3_dir = (
            ROOT / "v6_c3_kde_recalibration"
            / self.tag
        )

        self.c2 = import_local(
            "v6_c2_empirical_distribution.py",
            "v6_c2_engine",
        )
        self.c3 = import_local(
            "v6_c3_recalibrate_kde.py",
            "v6_c3_engine",
        )

        bundle = joblib.load(
            self.v7_dir / "research_models.joblib"
        )
        self.models = bundle["models"]

        self.oof = pd.read_csv(
            self.c2_dir
            / "chronological_oof_residuals.csv"
        )

        self.params = pd.read_csv(
            self.c3_dir / "selected_parameters.csv"
        )

        self.distributions = {}

        for horizon in (30, 45):
            errors = (
                self.oof.loc[
                    self.oof["horizon"] == horizon,
                    "error_f",
                ]
                .to_numpy(float)
            )

            if len(errors) < 100:
                raise RuntimeError(
                    f"Too few {horizon}m residuals"
                )

            rows = self.params[
                self.params["horizon"] == horizon
            ]

            if len(rows) != 1:
                raise RuntimeError(
                    f"Missing {horizon}m KDE settings"
                )

            shift = float(
                rows.iloc[0]["shift_f"]
            )
            scale = float(
                rows.iloc[0]["scale"]
            )

            base = self.c2.ResidualDistribution(
                f"{horizon}m_original_kde",
                errors,
                bandwidth=0.20,
            )

            adjusted = self.c3.AdjustedKDE(
                base,
                shift,
                scale,
            )

            self.distributions[horizon] = {
                "base": base,
                "adjusted": adjusted,
                "shift": shift,
                "scale": scale,
                "residual_count": len(errors),
            }

    def point_forecast(self, horizon, features):
        if horizon not in (30, 45):
            raise ValueError(
                "Horizon must be 30 or 45"
            )

        entry = self.models[
            f"{horizon}m_full_dedicated"
        ]

        columns = entry["features"]

        missing = [
            col for col in columns
            if col not in features
        ]

        if missing:
            raise ValueError(
                "Missing model features: "
                + ", ".join(missing)
            )

        x = pd.DataFrame([
            {
                col: features[col]
                for col in columns
            }
        ])

        x = x.apply(
            pd.to_numeric, errors="coerce"
        ).replace(
            [np.inf, -np.inf], np.nan
        )

        knyc = float(features["NYC_temp"])

        if not np.isfinite(knyc):
            raise ValueError(
                "No usable KNYC temperature"
            )

        predicted_change = float(
            entry["model"].predict(x)[0]
        )

        return knyc + predicted_change

    def probability_report(
        self,
        horizon,
        predicted_f,
        target_utc=None,
    ):
        if horizon not in self.distributions:
            raise ValueError(
                "Unsupported forecast horizon"
            )

        predicted_f = float(predicted_f)

        if not np.isfinite(predicted_f):
            raise ValueError(
                "Point forecast is not finite"
            )

        settings = self.distributions[horizon]
        dist = settings["adjusted"]
        base = settings["base"]

        # Probabilities represent nearest whole
        # Fahrenheit with half-degree boundaries.
        # This is a research convention.
        center = int(
            np.floor(
                predicted_f
                + settings["shift"]
                + 0.5
            )
        )

        buckets = np.arange(
            center - 6,
            center + 7,
        )

        probabilities = dist.bucket_probs(
            predicted_f,
            buckets,
        )

        total = float(probabilities.sum())

        if not 0 <= total <= 1.000001:
            raise RuntimeError(
                "Invalid probability mass"
            )

        bucket_rows = [
            {
                "temperature_f": int(temp),
                "probability": float(prob),
            }
            for temp, prob in zip(
                buckets, probabilities
            )
        ]

        # Distribution quantiles derived directly
        # from the fitted residual KDE.
        def quantile(q):
            value = float(np.interp(
                q,
                base.grid_cdf,
                base.grid,
            ))

            return (
                predicted_f
                + settings["shift"]
                + settings["scale"] * value
            )

        intervals = {}

        for level in (50, 80, 90):
            alpha = (100 - level) / 200

            intervals[str(level)] = {
                "lower_f": quantile(alpha),
                "upper_f": quantile(1 - alpha),
            }

        p_within_one = float(
            dist.cdf(1)
            - dist.cdf(-1)
        )

        most_likely = max(
            bucket_rows,
            key=lambda row: row["probability"],
        )

        return {
            "model": "V6-C3",
            "configuration": self.tag,
            "horizon_minutes": horizon,
            "target_utc": target_utc,
            "point_forecast_f": predicted_f,
            "most_likely_temperature_f":
                most_likely["temperature_f"],
            "most_likely_probability":
                most_likely["probability"],
            "probability_within_1f":
                p_within_one,
            "kde_bandwidth_f": 0.20,
            "kde_shift_f": settings["shift"],
            "kde_scale": settings["scale"],
            "residual_sample_n":
                settings["residual_count"],
            "intervals": intervals,
            "temperature_buckets": bucket_rows,
            "probability_outside_display":
                max(0.0, 1.0 - total),
        }

    def predict_features(
        self, horizon, features, target_utc=None
    ):
        predicted_f = self.point_forecast(
            horizon, features
        )

        return self.probability_report(
            horizon,
            predicted_f,
            target_utc,
        )

    def audit(self):
        """Reproduce frozen October predictions and C3 scores."""

        v7 = import_local(
            "train_knyc_v7_history.py",
            "v7_probability_audit",
        )

        grid = v7.build_grid(
            v7.import_v3(),
            self.knyc_lag,
            self.airport_lag,
        )

        stored = pd.read_csv(
            self.v7_dir / "predictions.csv"
        )
        stored["target_time"] = pd.to_datetime(
            stored["target_time"], utc=True
        )

        saved_scores = pd.read_csv(
            self.c3_dir / "comparison.csv"
        )

        for horizon in (30, 45):
            entry = self.models[
                f"{horizon}m_full_dedicated"
            ]

            october = grid[
                (grid["horizon"] == horizon)
                & grid["target_day"].between(
                    "2026-10-01",
                    "2026-10-26",
                )
            ].copy()

            pred = (
                october["NYC_temp"].to_numpy(float)
                + entry["model"].predict(
                    october[entry["features"]]
                )
            )

            calculated = pd.DataFrame({
                "target_time":
                    october["target_time"],
                "predicted_f": pred,
            })

            previous = stored[
                (stored["horizon"] == horizon)
                & (
                    stored["variant"]
                    == "V7_full_dedicated"
                )
            ][
                ["target_time", "prediction_f"]
            ]

            merged = calculated.merge(
                previous,
                on="target_time",
                validate="one_to_one",
            )

            if (
                len(merged) != len(october)
                or len(previous) != len(october)
            ):
                raise RuntimeError(
                    "Frozen point prediction "
                    "timestamps do not match"
                )

            max_difference = float(
                np.max(np.abs(
                    merged["predicted_f"]
                    - merged["prediction_f"]
                ))
            )

            if max_difference > 0.00001:
                raise RuntimeError(
                    "Point-model parity failure: "
                    f"{max_difference:.9f}F"
                )

            data = pd.DataFrame({
                "actual_f":
                    october["target"].to_numpy(float),
                "predicted_f": pred,
            })

            evaluated = self.c2.evaluate(
                self.distributions[horizon][
                    "adjusted"
                ],
                data,
            )

            settings = self.distributions[horizon]

            reference_name = (
                f"kde_shift_{settings['shift']:+.2f}"
                f"_scale_{settings['scale']:.2f}"
            )

            ref = saved_scores[
                (saved_scores["horizon"] == horizon)
                & (
                    saved_scores["distribution"]
                    == reference_name
                )
            ]

            if len(ref) != 1:
                raise RuntimeError(
                    "Saved C3 reference not found"
                )

            for metric in (
                "bucket_logloss",
                "bucket_brier",
                "within_1_brier",
            ):
                difference = abs(
                    float(evaluated[metric])
                    - float(ref.iloc[0][metric])
                )

                if difference > 0.0001:
                    raise RuntimeError(
                        f"{metric} parity failure: "
                        f"{difference}"
                    )

            print(
                f"{horizon}m AUDIT PASS "
                f"n={len(october)} "
                f"point_diff={max_difference:.9f}F "
                f"logloss={evaluated['bucket_logloss']:.5f} "
                f"brier={evaluated['bucket_brier']:.5f}",
                flush=True,
            )


def print_report(result):
    print("\n=== FROZEN V6-C3 PROBABILITY FORECAST ===")
    print("Configuration:", result["configuration"])
    print("Horizon:", result["horizon_minutes"], "minutes")

    if result["target_utc"]:
        print("Target:", result["target_utc"])

    print(
        "Point forecast:",
        f"{result['point_forecast_f']:.3f}F"
    )

    print(
        "Most likely whole degree:",
        f"{result['most_likely_temperature_f']}F",
        f"({result['most_likely_probability']:.1%})",
    )

    print(
        "Probability within +/-1F:",
        f"{result['probability_within_1f']:.1%}",
    )

    print("\nTemperature probabilities:")

    for row in result["temperature_buckets"]:
        if row["probability"] >= 0.0005:
            print(
                f"  {row['temperature_f']:>4}F: "
                f"{row['probability'] * 100:6.2f}%"
            )

    print("\nPrediction intervals:")

    for level, bounds in result["intervals"].items():
        print(
            f"  {level}%: "
            f"{bounds['lower_f']:.2f}F to "
            f"{bounds['upper_f']:.2f}F"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--knyc-lag", type=int, default=5
    )
    parser.add_argument(
        "--airport-lag", type=int, default=6
    )
    parser.add_argument(
        "--horizon", type=int, choices=(30, 45)
    )

    parser.add_argument(
        "--audit", action="store_true"
    )
    parser.add_argument(
        "--point-f", type=float
    )
    parser.add_argument(
        "--features-json", type=Path
    )

    args = parser.parse_args()

    if min(
        args.knyc_lag,
        args.airport_lag
    ) < 0:
        parser.error(
            "Observation lags must be nonnegative"
        )

    engine = ProbabilityEngine(
        args.knyc_lag,
        args.airport_lag,
    )

    if args.audit:
        engine.audit()
        return

    if args.horizon is None:
        parser.error(
            "Specify --horizon 30 or 45"
        )

    if args.features_json:
        payload = json.loads(
            args.features_json.read_text()
        )

        features = payload.get(
            "features", payload
        )

        result = engine.predict_features(
            args.horizon,
            features,
            payload.get("target_utc"),
        )

    elif args.point_f is not None:
        result = engine.probability_report(
            args.horizon,
            args.point_f,
        )

    else:
        parser.error(
            "Specify --audit, --point-f, "
            "or --features-json"
        )

    print_report(result)


if __name__ == "__main__":
    main()
