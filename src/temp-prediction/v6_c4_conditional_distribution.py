
#!/usr/bin/env python3
"""V6-C4: Test weather-conditioned KDE uncertainty.

Uses chronological OOF residuals for regime width ratios,
September for selection, October for exploratory evaluation.
Does not change saved or live prediction models.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

STATIONS = ("LGA", "EWR", "TEB")
STRENGTHS = (0.0, 0.5, 1.0)


def import_local(filename, name):
    path = Path(__file__).with_name(filename)

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regime_id(frame):
    """Four regimes: rapid trend x regional disagreement.

    Missing observations are marked unclassified (-1).
    """

    speed = pd.to_numeric(
        frame["NYC_trend1"], errors="coerce"
    )

    airport = frame[
        [f"{s}_temp" for s in STATIONS]
    ].apply(pd.to_numeric, errors="coerce")

    spread = (
        airport.max(axis=1)
        - airport.min(axis=1)
    )

    valid = (
        speed.notna()
        & (airport.notna().sum(axis=1) >= 2)
    )

    rapid = speed.abs().ge(1.5)
    disagreement = spread.ge(3.0)

    code = (
        rapid.astype(int) * 2
        + disagreement.astype(int)
    ).astype(int)

    return code.where(
        valid, -1
    ).to_numpy(dtype=int)


def estimate_ratios(oof):
    errors = oof[
        "error_f"
    ].to_numpy(dtype=float)

    groups = oof[
        "regime"
    ].to_numpy(dtype=int)

    eligible = (
        np.isfinite(errors)
        & (groups >= 0)
    )

    if eligible.sum() < 500:
        raise RuntimeError(
            "Too few valid chronological OOF errors"
        )

    global_var = float(
        np.mean(errors[eligible] ** 2)
    )

    if (
        not np.isfinite(global_var)
        or global_var <= 0
    ):
        raise RuntimeError(
            "Invalid global OOF error variance"
        )

    ratios = {-1: 1.0}
    summary = []

    for group in range(4):
        e = errors[
            eligible & (groups == group)
        ]

        n = len(e)

        # Shrink uncertainty toward the global
        # OOF variance to avoid extreme estimates.
        prior_weight = 300

        variance = (
            float(np.sum(e ** 2))
            + prior_weight * global_var
        ) / (n + prior_weight)

        ratio = float(
            np.clip(
                np.sqrt(variance / global_var),
                0.75,
                1.35,
            )
        )

        ratios[group] = ratio

        summary.append({
            "regime": group,
            "rapid_trend": bool(group & 2),
            "airport_spread_ge_3f": bool(group & 1),
            "oof_n": n,
            "oof_mae_f": (
                round(float(np.mean(np.abs(e))), 4)
                if n else np.nan
            ),
            "conditional_width_ratio": round(
                ratio, 4
            ),
        })

    return ratios, pd.DataFrame(summary)


def evaluate(
    data, kde, shift, scale,
    ratios, strength, include_rows=False
):
    """Score a row-specific width adjustment to KDE."""

    actual = data[
        "actual_f"
    ].to_numpy(float)

    pred = data[
        "predicted_f"
    ].to_numpy(float)

    regimes = data[
        "regime"
    ].to_numpy(int)

    if not (
        np.isfinite(actual).all()
        and np.isfinite(pred).all()
    ):
        raise RuntimeError(
            "Nonfinite point forecasts or targets"
        )

    row_ratio = np.array([
        ratios.get(int(r), 1.0)
        for r in regimes
    ])

    row_scale = (
        float(scale)
        * np.power(
            row_ratio, float(strength)
        )
    )

    if (
        (row_scale <= 0).any()
        or not np.isfinite(row_scale).all()
    ):
        raise RuntimeError(
            "Invalid conditional KDE widths"
        )

    def cdf_residual(offset):
        return kde.cdf(
            (np.asarray(offset) - shift)
            / row_scale
        )

    residual = actual - pred

    realized_bucket = np.floor(
        actual + 0.5
    ).astype(int)

    correct_bucket_p = np.maximum(
        cdf_residual(
            realized_bucket + 0.5 - pred
        ) - cdf_residual(
            realized_bucket - 0.5 - pred
        ),
        1e-12,
    )

    bucket_logloss = float(
        -np.log(correct_bucket_p).mean()
    )

    buckets = np.arange(
        -30, 131, dtype=float
    )

    upper = kde.cdf(
        (
            buckets[None, :] + 0.5
            - pred[:, None] - shift
        ) / row_scale[:, None]
    )

    lower = kde.cdf(
        (
            buckets[None, :] - 0.5
            - pred[:, None] - shift
        ) / row_scale[:, None]
    )

    probs = np.clip(
        upper - lower, 0, 1
    )

    mass = probs.sum(axis=1)

    if (mass < 0.999).any():
        raise RuntimeError(
            "Temperature buckets omit probability mass"
        )

    bucket_brier = float(
        np.mean(
            1 - 2 * correct_bucket_p
            + np.sum(probs ** 2, axis=1)
        )
    )

    within1_p = (
        kde.cdf(
            (1 - shift) / row_scale
        ) - kde.cdf(
            (-1 - shift) / row_scale
        )
    )

    within1_actual = (
        np.abs(residual) <= 1
    ).astype(float)

    within1_brier = float(
        np.mean(
            (within1_p - within1_actual) ** 2
        )
    )

    pit = cdf_residual(residual)

    coverage = {}

    for pct in (50, 80, 90):
        alpha = (100 - pct) / 200

        coverage[
            f"cover_{pct}_pct"
        ] = round(
            100 * float(
                np.mean(
                    (pit >= alpha)
                    & (pit <= 1 - alpha)
                )
            ),
            2,
        )

    metrics = {
        "n": len(data),
        "bucket_logloss": round(
            bucket_logloss, 5
        ),
        "bucket_brier": round(
            bucket_brier, 5
        ),
        "within_1_brier": round(
            within1_brier, 5
        ),
        "pred_within_1_pct": round(
            100 * float(within1_p.mean()), 2
        ),
        "actual_within_1_pct": round(
            100 * float(within1_actual.mean()), 2
        ),
        **coverage,
    }

    if not include_rows:
        return metrics, None

    rows = pd.DataFrame({
        "target_time":
            data["target_time"].to_numpy(),
        "actual_f": actual,
        "point_forecast_f": pred,
        "regime": regimes,
        "effective_scale": row_scale,
        "probability_within_one_f": within1_p,
        "correct_temperature_bucket_probability":
            correct_bucket_p,
        "probability_integral_transform": pit,
    })

    return metrics, rows


def self_test():
    sample = pd.DataFrame({
        "NYC_trend1": [0, 2.0, 0, 2.0],
        "LGA_temp": [62, 62, 60, 60],
        "EWR_temp": [62, 62, 64, 64],
        "TEB_temp": [62, 62, 62, 62],
    })

    assert regime_id(
        sample
    ).tolist() == [0, 2, 1, 3]

    oof = pd.DataFrame({
        "regime": np.tile(
            np.arange(4), 250
        ),
        "error_f": np.tile(
            np.array([
                0.25, -0.5, 1.0, -1.5
            ]),
            250,
        ),
    })

    ratios, summary = estimate_ratios(oof)

    assert len(summary) == 4
    assert ratios[-1] == 1

    assert all(
        0.75 <= ratios[g] <= 1.35
        for g in range(4)
    )

    print("SELF-TEST PASS")


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
        self_test()
        return

    if min(
        args.knyc_lag, args.airport_lag
    ) < 0:
        parser.error(
            "Observation delays must be nonnegative"
        )

    tag = (
        f"knyc{args.knyc_lag}"
        f"_airports{args.airport_lag}"
    )

    c2_dir = (
        ROOT
        / "v6_c2_empirical_distributions"
        / tag
    )

    c3_dir = (
        ROOT
        / "v6_c3_kde_recalibration"
        / tag
    )

    model_dir = (
        ROOT
        / "v7_extended_history"
        / "training_results"
        / tag
    )

    needed = (
        c2_dir / "chronological_oof_residuals.csv",
        c3_dir / "selected_parameters.csv",
        c3_dir / "comparison.csv",
        model_dir / "research_models.joblib",
        model_dir / "predictions.csv",
    )

    for path in needed:
        if not path.exists():
            raise FileNotFoundError(path)

    v7 = import_local(
        "train_knyc_v7_history.py",
        "v7_for_c4",
    )

    c2 = import_local(
        "v6_c2_empirical_distribution.py",
        "c2_for_c4",
    )

    bundle = joblib.load(
        model_dir / "research_models.joblib"
    )

    previous = pd.read_csv(
        model_dir / "predictions.csv"
    )

    previous["target_time"] = pd.to_datetime(
        previous["target_time"], utc=True
    )

    oof_all = pd.read_csv(
        c2_dir / "chronological_oof_residuals.csv"
    )

    oof_all["target_time"] = pd.to_datetime(
        oof_all["target_time"], utc=True
    )

    params = pd.read_csv(
        c3_dir / "selected_parameters.csv"
    )

    c3_scores = pd.read_csv(
        c3_dir / "comparison.csv"
    )

    print(
        "Reconstructing frozen point-model features...",
        flush=True,
    )

    grid = v7.build_grid(
        v7.import_v3(),
        args.knyc_lag,
        args.airport_lag,
    )

    grid["target_time"] = pd.to_datetime(
        grid["target_time"], utc=True
    )

    grid["regime"] = regime_id(grid)

    output = (
        ROOT
        / "v6_c4_conditional_distribution"
        / tag
    )

    output.mkdir(
        parents=True, exist_ok=True
    )

    scores = []
    selections = []
    reports = []
    predictions = []

    for horizon in (30, 45):
        print(
            f"\n=== {horizon}-MINUTE "
            "WEATHER-CONDITIONED KDE ===",
            flush=True,
        )

        frame = grid[
            grid["horizon"] == horizon
        ].copy()

        entry = bundle["models"][
            f"{horizon}m_full_dedicated"
        ]

        frame["actual_f"] = (
            frame["target"].to_numpy(float)
        )

        frame["predicted_f"] = (
            frame["NYC_temp"].to_numpy(float)
            + entry["model"].predict(
                frame[entry["features"]]
            )
        )

        september = frame[
            frame["target_day"].between(
                "2026-09-01", "2026-09-30"
            )
        ].copy()

        october = frame[
            frame["target_day"].between(
                "2026-10-01", "2026-10-26"
            )
        ].copy()

        if (
            len(september) < 100
            or len(october) < 100
        ):
            raise RuntimeError(
                "Incomplete validation data"
            )

        # Confirm unchanged point predictions.
        prior = previous[
            (previous["horizon"] == horizon)
            & (
                previous["variant"]
                == "V7_full_dedicated"
            )
        ][
            ["target_time", "prediction_f"]
        ]

        parity = october[
            ["target_time", "predicted_f"]
        ].merge(
            prior,
            on="target_time",
            validate="one_to_one",
        )

        if len(parity) != len(october):
            raise RuntimeError(
                "Point forecast parity event mismatch"
            )

        difference = float(
            (
                parity["predicted_f"]
                - parity["prediction_f"]
            ).abs().max()
        )

        if (
            not np.isfinite(difference)
            or difference > 0.01
        ):
            raise RuntimeError(
                "Frozen point prediction parity failed"
            )

        print(
            "Frozen point prediction parity:",
            f"{difference:.9f}F",
            flush=True,
        )

        # Match historical OOF errors with
        # the weather conditions available
        # at each original decision time.
        historical = oof_all[
            oof_all["horizon"] == horizon
        ][
            ["target_time", "error_f"]
        ]

        residuals = historical.merge(
            frame[["target_time", "regime"]],
            on="target_time",
            how="left",
            validate="one_to_one",
        )

        if (
            len(residuals) != len(historical)
            or residuals["regime"].isna().any()
        ):
            raise RuntimeError(
                "Cannot align OOF residuals "
                "with causal weather conditions"
            )

        residuals["regime"] = (
            residuals["regime"].astype(int)
        )

        ratios, table = estimate_ratios(
            residuals
        )

        table["horizon"] = horizon
        reports.append(table)

        print(
            "\nOOF conditional width estimates:"
        )
        print(
            table.to_string(index=False),
            flush=True,
        )

        # Reuse exactly the shift and scale
        # selected by the previous V6-C3 study.
        chosen = params[
            params["horizon"] == horizon
        ]

        if len(chosen) != 1:
            raise RuntimeError(
                "Missing V6-C3 selected parameters"
            )

        shift = float(
            chosen.iloc[0]["shift_f"]
        )

        base_scale = float(
            chosen.iloc[0]["scale"]
        )

        kde = c2.ResidualDistribution(
            "original_kde",
            historical["error_f"].to_numpy(float),
            bandwidth=0.20,
        )

        reference_name = (
            f"kde_shift_{shift:+.2f}"
            f"_scale_{base_scale:.2f}"
        )

        reference = c3_scores[
            (c3_scores["horizon"] == horizon)
            & (
                c3_scores["distribution"]
                == reference_name
            )
        ]

        if len(reference) != 1:
            raise RuntimeError(
                "Missing exact V6-C3 reference"
            )

        baseline, _ = evaluate(
            october,
            kde, shift, base_scale,
            ratios, 0.0,
        )

        for metric in (
            "bucket_logloss",
            "bucket_brier",
            "within_1_brier",
        ):
            delta = abs(
                baseline[metric]
                - float(
                    reference.iloc[0][metric]
                )
            )

            if delta > 0.0001:
                raise RuntimeError(
                    "V6-C3 probability parity failed: "
                    f"{metric} difference={delta:.7f}"
                )

        print(
            "Frozen V6-C3 probability parity: PASS",
            flush=True,
        )

        candidates = []

        for strength in STRENGTHS:
            sep_metrics, _ = evaluate(
                september,
                kde, shift, base_scale,
                ratios, strength,
            )

            oct_metrics, oct_rows = evaluate(
                october,
                kde, shift, base_scale,
                ratios, strength,
                include_rows=True,
            )

            scores.append({
                "horizon": horizon,
                "period": "September",
                "strength": strength,
                **sep_metrics,
            })

            scores.append({
                "horizon": horizon,
                "period": "October",
                "strength": strength,
                **oct_metrics,
            })

            candidates.append((
                sep_metrics["bucket_logloss"],
                sep_metrics["bucket_brier"],
                strength,
            ))

            if strength == 0.0:
                predictions.append(
                    oct_rows.assign(
                        horizon=horizon,
                        strength=strength,
                    )
                )

        # Choose using September only.
        winner = min(candidates)[2]

        selections.append({
            "horizon": horizon,
            "selected_strength": winner,
            "shift_f": shift,
            "base_scale": base_scale,
        })

        print(
            "September-selected conditioning strength:",
            winner,
            flush=True,
        )

        if winner != 0.0:
            _, rows = evaluate(
                october,
                kde, shift, base_scale,
                ratios, winner,
                include_rows=True,
            )

            predictions.append(
                rows.assign(
                    horizon=horizon,
                    strength=winner,
                )
            )

    summary = pd.DataFrame(scores)

    summary.to_csv(
        output / "distribution_comparison.csv",
        index=False,
    )

    pd.concat(
        reports, ignore_index=True
    ).to_csv(
        output / "oof_regime_widths.csv",
        index=False,
    )

    pd.DataFrame(
        selections
    ).to_csv(
        output / "september_selection.csv",
        index=False,
    )

    pd.concat(
        predictions, ignore_index=True
    ).to_csv(
        output / "october_probabilities.csv",
        index=False,
    )

    print(
        "\n=== V6-C4 CONDITIONAL KDE COMPARISON ==="
    )

    print(
        summary.sort_values(
            ["horizon", "period", "bucket_logloss"]
        ).to_string(index=False)
    )

    print(
        "\nSeptember selections:",
        selections
    )

    print(
        "\nSaved to:", output.resolve()
    )

    print(
        "October exploratory only. "
        "No live or trading models changed."
    )


if __name__ == "__main__":
    main()
