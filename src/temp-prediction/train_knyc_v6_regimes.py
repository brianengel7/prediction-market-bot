
#!/usr/bin/env python3
"""V6-B: Weather-regime features versus frozen V7 baseline.

September selects complexity; October is exploratory.
Research only. Does not modify existing or live models.
"""

import argparse
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)

STATIONS = ("LGA", "EWR", "TEB")


def import_v7():
    path = Path(__file__).with_name(
        "train_knyc_v7_history.py"
    )

    spec = importlib.util.spec_from_file_location(
        "v7_helpers_regimes", path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)
    return module


def add_regimes(frame):
    out = frame.copy()

    temps = out[
        [f"{s}_temp" for s in STATIONS]
    ]

    trends = out[
        [f"{s}_trend1" for s in STATIONS]
    ]

    wind = out[
        [f"{s}_wind" for s in STATIONS]
    ]

    dew = out[
        [f"{s}_dew" for s in STATIONS]
    ]

    # Central Park compared with nearby airports.
    out["B_airport_mean_temp"] = (
        temps.mean(axis=1)
    )

    out["B_airport_temp_spread"] = (
        temps.max(axis=1)
        - temps.min(axis=1)
    )

    out["B_airport_minus_nyc"] = (
        out["B_airport_mean_temp"]
        - out["NYC_temp"]
    )

    out["B_airport_mean_trend"] = (
        trends.mean(axis=1)
    )

    out["B_airport_minus_nyc_trend"] = (
        out["B_airport_mean_trend"]
        - out["NYC_trend1"]
    )

    out["B_airport_mean_wind"] = (
        wind.mean(axis=1)
    )

    out["B_airport_minus_nyc_wind"] = (
        out["B_airport_mean_wind"]
        - out["NYC_wind"]
    )

    out["B_airport_minus_nyc_dew"] = (
        dew.mean(axis=1)
        - out["NYC_dew"]
    )

    bias_features = [
        "B_airport_mean_temp",
        "B_airport_temp_spread",
        "B_airport_minus_nyc",
        "B_airport_mean_trend",
        "B_airport_minus_nyc_trend",
        "B_airport_mean_wind",
        "B_airport_minus_nyc_wind",
        "B_airport_minus_nyc_dew",
    ]

    # All regimes use observations available
    # at the forecast decision timestamp.
    out["R_rapid_nyc_warming"] = (
        out["NYC_trend1"] >= 2
    ).astype(float)

    out["R_rapid_nyc_cooling"] = (
        out["NYC_trend1"] <= -2
    ).astype(float)

    out["R_rapid_regional_warming"] = (
        out["B_airport_mean_trend"] >= 2
    ).astype(float)

    out["R_rapid_regional_cooling"] = (
        out["B_airport_mean_trend"] <= -2
    ).astype(float)

    out["R_nyc_lagging_warming"] = (
        (out["B_airport_mean_trend"] >= 1)
        & (out["B_airport_minus_nyc_trend"] >= 1)
    ).astype(float)

    out["R_regional_trend_disagreement"] = (
        (
            out["NYC_trend1"]
            * out["B_airport_mean_trend"]
        ) < 0
    ).astype(float)

    out["R_nyc_near_saturation"] = (
        out["NYC_dew_depression"] <= 3
    ).astype(float)

    out["R_nyc_windy"] = (
        out["NYC_wind"] >= 10
    ).astype(float)

    out["R_nyc_wind_from_east"] = (
        out["NYC_direction"].between(45, 135)
    ).astype(float)

    out["R_nyc_wind_from_south"] = (
        out["NYC_direction"].between(135, 225)
    ).astype(float)

    out["R_nyc_wind_change"] = (
        out["NYC_wind_change"]
    )

    regime_features = [
        "R_rapid_nyc_warming",
        "R_rapid_nyc_cooling",
        "R_rapid_regional_warming",
        "R_rapid_regional_cooling",
        "R_nyc_lagging_warming",
        "R_regional_trend_disagreement",
        "R_nyc_near_saturation",
        "R_nyc_windy",
        "R_nyc_wind_from_east",
        "R_nyc_wind_from_south",
        "R_nyc_wind_change",
        "NYC_relative_humidity",
        "NYC_dew_depression",
    ]

    out = out.replace(
        [np.inf, -np.inf], np.nan
    )

    return out, bias_features, regime_features


def metrics(actual, predicted):
    error = np.abs(
        np.asarray(actual, float)
        - np.asarray(predicted, float)
    )

    return {
        "n": len(error),
        "mae_f": round(
            float(error.mean()), 4
        ),
        "within_half_pct": round(
            float(100 * (error <= 0.5).mean()),
            1,
        ),
        "within_one_pct": round(
            float(100 * (error <= 1).mean()),
            1,
        ),
        "errors_gt_2": int(
            (error > 2).sum()
        ),
    }


def model_for(leaves):
    return HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=leaves,
        min_samples_leaf=30,
        l2_regularization=2,
        random_state=42,
    )


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
        sample = pd.DataFrame({
            **{
                f"{s}_{key}": [value]
                for s in STATIONS
                for key, value in [
                    ("temp", 65),
                    ("trend1", 1),
                    ("wind", 5),
                    ("dew", 55),
                ]
            },
            "NYC_temp": [63],
            "NYC_trend1": [0],
            "NYC_wind": [4],
            "NYC_dew": [54],
            "NYC_dew_depression": [9],
            "NYC_direction": [90],
            "NYC_wind_change": [1],
            "NYC_relative_humidity": [70],
        })

        test, bias, regimes = add_regimes(
            sample
        )

        assert (
            test.loc[0, "B_airport_minus_nyc"]
            == 2
        )

        assert (
            test.loc[
                0,
                "B_airport_minus_nyc_trend",
            ] == 1
        )

        assert set(
            bias + regimes
        ).issubset(test.columns)

        print("SELF-TEST PASS")
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

    baseline_dir = (
        ROOT
        / "v7_extended_history"
        / "training_results"
        / tag
    )

    old_models = joblib.load(
        baseline_dir / "research_models.joblib"
    )["models"]

    old_predictions = pd.read_csv(
        baseline_dir / "predictions.csv"
    )

    old_predictions["target_time"] = (
        pd.to_datetime(
            old_predictions["target_time"],
            utc=True,
        )
    )

    v7 = import_v7()

    print(
        "Building identical extended-history data...",
        flush=True,
    )

    grid = v7.build_grid(
        v7.import_v3(),
        args.knyc_lag,
        args.airport_lag,
    )

    grid, bias_features, regime_features = (
        add_regimes(grid)
    )

    splits = {
        "train": grid[
            grid["target_day"] < "2026-09-01"
        ],
        "sep": grid[
            grid["target_day"].between(
                "2026-09-01",
                "2026-09-30",
            )
        ],
        "oct": grid[
            grid["target_day"].between(
                "2026-10-01",
                "2026-10-26",
            )
        ],
    }

    print(
        "Rows:",
        {
            key: len(value)
            for key, value in splits.items()
        },
        flush=True,
    )

    outputs = []
    predictions = []
    fitted = {}

    for horizon in (30, 45):
        a = {
            key: value[
                value["horizon"] == horizon
            ].copy()
            for key, value in splits.items()
        }

        baseline = old_models[
            f"{horizon}m_full_dedicated"
        ]

        base_features = baseline["features"]

        old_predict = (
            a["oct"]["NYC_temp"].to_numpy(float)
            + baseline["model"].predict(
                a["oct"][base_features]
            )
        )

        stored = old_predictions[
            (
                old_predictions["variant"]
                == "V7_full_dedicated"
            )
            & (
                old_predictions["horizon"]
                == horizon
            )
        ][
            ["target_time", "prediction_f"]
        ]

        check = pd.DataFrame({
            "target_time":
                a["oct"]["target_time"],
            "new": old_predict,
        }).merge(
            stored,
            on="target_time",
            validate="one_to_one",
            how="inner",
        )

        if len(check) != len(a["oct"]):
            raise RuntimeError(
                "Frozen V7 target mismatch"
            )

        difference = float(
            (
                check["new"]
                - check["prediction_f"]
            ).abs().max()
        )

        print(
            f"\n{horizon}m frozen "
            f"expanded-history parity: "
            f"{difference:.9f} F",
            flush=True,
        )

        if (
            not np.isfinite(difference)
            or difference > 0.01
        ):
            raise RuntimeError(
                "Frozen V7 feature parity failed"
            )

        outputs.append({
            "horizon": horizon,
            "variant": "FROZEN_full_dedicated",
            "sep_within_one_pct": np.nan,
            "sep_mae": np.nan,
            **metrics(
                a["oct"]["target"],
                old_predict,
            ),
        })

        for variant, extras in (
            ("V6B_bias", bias_features),
            (
                "V6B_bias_and_regimes",
                bias_features + regime_features,
            ),
        ):
            features = list(dict.fromkeys(
                base_features + extras
            ))

            missing = [
                f for f in features
                if f not in grid.columns
            ]

            if missing:
                raise RuntimeError(
                    f"Missing features: {missing}"
                )

            winner = None

            for leaves in (7, 15, 31):
                model = model_for(leaves)

                model.fit(
                    a["train"][features],
                    (
                        a["train"]["target"]
                        - a["train"]["NYC_temp"]
                    ),
                )

                sep_pred = (
                    a["sep"]["NYC_temp"]
                    .to_numpy(float)
                    + model.predict(
                        a["sep"][features]
                    )
                )

                sep_error = np.abs(
                    a["sep"]["target"]
                    .to_numpy(float)
                    - sep_pred
                )

                ranking = (
                    -float(
                        (sep_error <= 1).mean()
                    ),
                    float(sep_error.mean()),
                )

                if (
                    winner is None
                    or ranking < winner[0]
                ):
                    winner = (
                        ranking, leaves, model
                    )

            _, leaves, model = winner

            sep_pred = (
                a["sep"]["NYC_temp"]
                .to_numpy(float)
                + model.predict(
                    a["sep"][features]
                )
            )

            oct_pred = (
                a["oct"]["NYC_temp"]
                .to_numpy(float)
                + model.predict(
                    a["oct"][features]
                )
            )

            sep_m = metrics(
                a["sep"]["target"],
                sep_pred,
            )

            oct_m = metrics(
                a["oct"]["target"],
                oct_pred,
            )

            print(
                f"{variant}: "
                f"September ±1F="
                f"{sep_m['within_one_pct']:.1f}% "
                f"October ±1F="
                f"{oct_m['within_one_pct']:.1f}% "
                f"October MAE="
                f"{oct_m['mae_f']:.4f}F "
                f"large={oct_m['errors_gt_2']}",
                flush=True,
            )

            outputs.append({
                "horizon": horizon,
                "variant": variant,
                "sep_within_one_pct":
                    sep_m["within_one_pct"],
                "sep_mae": sep_m["mae_f"],
                "leaves": leaves,
                **oct_m,
            })

            predictions.append(
                pd.DataFrame({
                    "target_time":
                        a["oct"]["target_time"],
                    "horizon": horizon,
                    "actual_precise_f":
                        a["oct"]["target"],
                    "variant": variant,
                    "prediction_f": oct_pred,
                })
            )

            fitted[
                f"{horizon}m_{variant}"
            ] = {
                "model": model,
                "features": features,
                "leaves": leaves,
                "selection_metric":
                    "September within 1F; "
                    "tie by MAE",
            }

    out = (
        ROOT
        / "v6_weather_regimes"
        / tag
    )

    out.mkdir(
        parents=True, exist_ok=True
    )

    summary = pd.DataFrame(outputs)

    summary.to_csv(
        out / "comparison.csv",
        index=False,
    )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_csv(
        out / "predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "models": fitted,
            "knyc_lag": args.knyc_lag,
            "airport_lag": args.airport_lag,
        },
        out / "research_models.joblib",
    )

    print(
        "\n=== V6-B REGIME COMPARISON "
        "(OCTOBER EXPLORATORY) ==="
    )

    print(
        summary.to_string(index=False)
    )

    print("Saved:", out)


if __name__ == "__main__":
    main()
