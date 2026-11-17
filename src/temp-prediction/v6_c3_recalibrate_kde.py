
#!/usr/bin/env python3
"""V6-C3: Recalibrate frozen empirical error distributions.

KDE learned from chronological OOF errors.
September selects shift and scale.
October is exploratory.
No live-model changes.
"""

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)


def load_script(filename, name):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdjustedKDE:
    def __init__(self, base, shift, scale):
        self.base = base
        self.shift = float(shift)
        self.scale = float(scale)
        self.name = (
            f"kde_shift_{shift:+.2f}"
            f"_scale_{scale:.2f}"
        )

    def cdf(self, delta):
        return self.base.cdf(
            (np.asarray(delta) - self.shift)
            / self.scale
        )

    def bucket_probs(self, prediction, bucket):
        return np.maximum(
            self.cdf(bucket + 0.5 - prediction)
            - self.cdf(bucket - 0.5 - prediction),
            0.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knyc-lag", type=int, default=5)
    parser.add_argument("--airport-lag", type=int, default=6)
    args = parser.parse_args()

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    v7 = load_script(
        "train_knyc_v7_history.py", "v7_c3"
    )
    c2 = load_script(
        "v6_c2_empirical_distribution.py", "c2_c3"
    )

    model_dir = (
        ROOT / "v7_extended_history"
        / "training_results" / tag
    )
    c2_dir = (
        ROOT / "v6_c2_empirical_distributions" / tag
    )

    bundle = joblib.load(
        model_dir / "research_models.joblib"
    )

    oof = pd.read_csv(
        c2_dir / "chronological_oof_residuals.csv"
    )

    grid = v7.build_grid(
        v7.import_v3(),
        args.knyc_lag,
        args.airport_lag,
    )

    rows = []
    configurations = []

    for horizon in (30, 45):
        entry = bundle["models"][
            f"{horizon}m_full_dedicated"
        ]

        data = grid[
            grid["horizon"] == horizon
        ].copy()

        data["actual_f"] = data["target"]
        data["predicted_f"] = (
            data["NYC_temp"].to_numpy(float)
            + entry["model"].predict(
                data[entry["features"]]
            )
        )

        september = data[
            data["target_day"].between(
                "2026-09-01", "2026-09-30"
            )
        ]

        october = data[
            data["target_day"].between(
                "2026-10-01", "2026-10-26"
            )
        ]

        residuals = oof.loc[
            oof["horizon"] == horizon,
            "error_f",
        ].to_numpy(float)

        base = c2.ResidualDistribution(
            "original_kde",
            residuals,
            bandwidth=0.20,
        )

        # This is the original unchanged KDE.
        unchanged = AdjustedKDE(
            base, 0.0, 1.0
        )

        # Tune only on September, never October.
        candidates = []

        for shift in (-0.20, -0.10, 0.0, 0.10, 0.20):
            for scale in (
                0.70, 0.80, 0.90, 1.00, 1.10
            ):
                dist = AdjustedKDE(
                    base, shift, scale
                )

                result = c2.evaluate(
                    dist, september
                )

                candidates.append(
                    (
                        result["bucket_logloss"],
                        result["bucket_brier"],
                        dist,
                    )
                )

        candidates.sort(
            key=lambda item: (item[0], item[1])
        )

        selected = candidates[0][2]

        sep_errors = (
            september["actual_f"].to_numpy(float)
            - september["predicted_f"].to_numpy(float)
        )

        gaussian = c2.ResidualDistribution(
            "september_gaussian",
            sep_errors,
        )

        for dist in (
            gaussian, unchanged, selected
        ):
            result = c2.evaluate(
                dist, october
            )
            result["horizon"] = horizon
            rows.append(result)

        configurations.append({
            "horizon": horizon,
            "shift_f": selected.shift,
            "scale": selected.scale,
            "september_bucket_logloss":
                candidates[0][0],
        })

        print(
            f"\n{horizon}m selected: "
            f"shift={selected.shift:+.2f}F "
            f"scale={selected.scale:.2f}",
            flush=True,
        )

    output = (
        ROOT / "v6_c3_kde_recalibration" / tag
    )
    output.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output / "comparison.csv",
        index=False,
    )

    pd.DataFrame(configurations).to_csv(
        output / "selected_parameters.csv",
        index=False,
    )

    print("\n=== V6-C3 OCTOBER COMPARISON ===")
    print(comparison.to_string(index=False))
    print("\nSaved:", output.resolve())


if __name__ == "__main__":
    main()
