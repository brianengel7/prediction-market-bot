import pandas as pd

from src.backtest.hrrr import (
    backtest_hrrr_date
)

from src.backtest.gefs import (
    backtest_gefs_date
)

from src.weather.nbm import (
    get_nbm_high_forecast
)

from src.weather.mos import (
    get_mos_high_forecast
)

from src.weather.calibrated_ensemble import (
    get_live_calibration
)

from src.weather.fair_distribution import (
    build_integer_distribution
)


FORECAST_HOUR = 12


def get_d1_12z_run_time(
    target_date
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_date = (
        pd.Timestamp(target_date)
        - pd.Timedelta(days=1)
    )

    return pd.Timestamp(
        year=run_date.year,
        month=run_date.month,
        day=run_date.day,
        hour=FORECAST_HOUR,
        tz="UTC"
    )


def get_live_forecast_bundle(
    target_date
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    run_time = get_d1_12z_run_time(
        target_date
    )

    print()
    print("=" * 70)
    print(
        f"LIVE D-1 12Z FORECAST BUNDLE"
    )
    print("=" * 70)

    print(
        f"Target date: "
        f"{target_date}"
    )

    print(
        f"Model run:   "
        f"{run_time:%Y-%m-%d %HZ}"
    )

    # HRRR
    hrrr = backtest_hrrr_date(
        target_date=target_date,
        actual_high=None
    )

    # GEFS
    gefs = backtest_gefs_date(
        target_date=target_date,
        actual_high=None
    )

    # NBM
    nbm = get_nbm_high_forecast(
        run_time=run_time,
        target_date=target_date
    )

    # GFS MOS
    mos = get_mos_high_forecast(
        run_time=run_time,
        target_date=target_date
    )

    raw_forecasts = {
        "GEFS":
            float(
                gefs["mean_high"]
            ),

        "GFS_MOS":
            float(
                mos["forecast_high"]
            ),

        "HRRR":
            float(
                hrrr["forecast_high"]
            ),

        "NBM":
            float(
                nbm["forecast_high"]
            )
    }

    calibration = (
        get_live_calibration(
            raw_forecasts=
                raw_forecasts,

            target_date=
                target_date
        )
    )

    distribution = (
        build_integer_distribution(
            point_forecast=
                calibration[
                    "point_forecast"
                ],

            residual_mean=
                calibration[
                    "residual_mean"
                ],

            residual_std=
                calibration[
                    "residual_std"
                ]
        )
    )

    return {
        "target_date":
            target_date,

        "run_time":
            run_time,

        "raw_forecasts":
            raw_forecasts,

        "calibration":
            calibration,

        "distribution":
            distribution,

        "gefs":
            gefs,

        "hrrr":
            hrrr,

        "nbm":
            nbm,

        "mos":
            mos
    }


def print_live_bundle(
    result
):
    calibration = result[
        "calibration"
    ]

    print()
    print("=" * 70)
    print(
        "RAW MODEL GUIDANCE"
    )
    print("-" * 70)

    for source, forecast in (
        result[
            "raw_forecasts"
        ].items()
    ):

        print(
            f"{source:10s}: "
            f"{forecast:.2f}°F"
        )

    print()
    print(
        "CALIBRATED MODEL GUIDANCE"
    )
    print("-" * 70)

    for source, corrected in (
        calibration[
            "corrected_forecasts"
        ].items()
    ):

        bias = calibration[
            "biases"
        ][source]

        weight = calibration[
            "weights"
        ][source]

        print(
            f"{source:10s}: "
            f"{corrected:6.2f}°F "
            f"bias={bias:+6.2f} "
            f"weight={weight:.3f}"
        )

    print()
    print(
        f"Fair point forecast: "
        f"{calibration['point_forecast']:.2f}°F"
    )

    print(
        f"Residual mean:       "
        f"{calibration['residual_mean']:+.3f}°F"
    )

    print(
        f"Residual std:        "
        f"{calibration['residual_std']:.3f}°F"
    )

    print()
    print(
        "FAIR SETTLEMENT DISTRIBUTION"
    )
    print("-" * 70)

    for (
        temperature,
        probability
    ) in result[
        "distribution"
    ].items():

        if probability >= 0.001:

            print(
                f"{temperature:3d}°F: "
                f"{probability * 100:6.2f}%"
            )


if __name__ == "__main__":

    result = get_live_forecast_bundle(
        "2026-09-24"
    )

    print_live_bundle(
        result
    )