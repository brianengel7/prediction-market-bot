from src.database.db import (
    save_weather_actual,
    save_weather_model_backtest
)

from src.backtest.actuals import (
    get_actual_highs
)

from src.backtest.hrrr import (
    backtest_hrrr_date
)

from src.backtest.gefs import (
    backtest_gefs_date
)

from src.backtest.nbm import (
    backtest_nbm_date
)

from src.backtest.mos import (
    backtest_mos_date
)


def backtest_weather_range(
    start_date,
    end_date
):
    actuals = get_actual_highs(
        start_date,
        end_date
    )

    results = []

    for actual in actuals:

        target_date = actual["date"]
        actual_high = actual["actual_high"]

        save_weather_actual(
            target_date=target_date,
            actual_high=actual_high,
            station="KNYC",
            settlement_source="NWS_CLI",
            product="CLI"
        )

        date_result = {
            "target_date": target_date,
            "actual_high": actual_high,
            "HRRR": False,
            "GEFS": False,
            "NBM": False,
            "GFS_MOS": False
        }

        # HRRR
        try:
            hrrr = backtest_hrrr_date(
                target_date,
                actual_high
            )

            save_weather_model_backtest(
                target_date=target_date,
                source="HRRR",
                product="sfc",
                model_run=hrrr["run_time"],
                forecast_horizon="D-1_12Z",
                forecast_value=hrrr["forecast_high"],
                actual_high=actual_high,
                station="KNYC",
                extraction_method="nearest",
                model_version="backtest-v1"
            )

            date_result["HRRR"] = True

        except Exception as error:
            print(
                f"HRRR FAILED {target_date}: "
                f"{error}"
            )

        # GEFS
        try:
            gefs = backtest_gefs_date(
                target_date,
                actual_high
            )

            save_weather_model_backtest(
                target_date=target_date,
                source="GEFS",
                product="atmos.25",
                model_run=gefs["run_time"],
                forecast_horizon="D-1_12Z",
                forecast_value=gefs["mean_high"],
                forecast_median=gefs["median_high"],
                forecast_std=gefs["std_high"],
                member_count=31,
                actual_high=actual_high,
                station="KNYC",
                extraction_method="nearest",
                model_version="backtest-v1"
            )

            date_result["GEFS"] = True

        except Exception as error:
            print(
                f"GEFS FAILED {target_date}: "
                f"{error}"
            )

        # NBM
        try:
            nbm = backtest_nbm_date(
                target_date,
                actual_high
            )

            save_weather_model_backtest(
                target_date=target_date,
                source="NBM",
                product="NBS",
                model_run=nbm["run_time"],
                forecast_horizon="D-1_12Z",
                forecast_value=nbm["forecast_high"],
                forecast_std=nbm["forecast_std"],
                actual_high=actual_high,
                station="KNYC",
                extraction_method="station_text",
                model_version="NBM-v5.0-NBS"
            )

            date_result["NBM"] = True

        except Exception as error:
            print(
                f"NBM FAILED {target_date}: "
                f"{error}"
            )

        # GFS MOS
        try:
            mos = backtest_mos_date(
                target_date,
                actual_high
            )

            save_weather_model_backtest(
                target_date=target_date,
                source="GFS_MOS",
                product="MAV",
                model_run=mos["run_time"],
                forecast_horizon="D-1_12Z",
                forecast_value=mos["forecast_high"],
                actual_high=actual_high,
                station="KNYC",
                extraction_method="station_guidance",
                model_version="GFS-MOS-MAV"
            )

            date_result["GFS_MOS"] = True

        except Exception as error:
            print(
                f"GFS MOS FAILED "
                f"{target_date}: {error}"
            )

        results.append(
            date_result
        )

    return results


if __name__ == "__main__":

    results = backtest_weather_range(
        "2026-06-11",
        "2026-09-10"
    )

    print()
    print(
        f"Completed "
        f"{len(results)} dates."
    )