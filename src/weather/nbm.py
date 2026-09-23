import requests
import pandas as pd


NBM_BASE_URL = (
    "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
)

NBM_STATION = "KNYC"

REQUEST_TIMEOUT = 30


def normalize_run_time(
    run_time
):
    run_time = pd.Timestamp(
        run_time
    )

    if run_time.tzinfo is None:
        run_time = run_time.tz_localize(
            "UTC"
        )
    else:
        run_time = run_time.tz_convert(
            "UTC"
        )

    return run_time


def get_nbm_nbs_bulletin(
    run_time
):
    """
    Download the historical NBM short-range
    text bulletin for one model cycle.
    """

    run_time = normalize_run_time(
        run_time
    )

    date_string = (
        run_time.strftime(
            "%Y%m%d"
        )
    )

    cycle = (
        run_time.strftime(
            "%H"
        )
    )

    url = (
        f"{NBM_BASE_URL}/"
        f"blend.{date_string}/"
        f"{cycle}/text/"
        f"blend_nbstx.t{cycle}z"
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    return {
        "run_time": run_time,
        "url": url,
        "text": response.text
    }


def get_station_block(
    bulletin_text,
    station=NBM_STATION,
    lines_after=35
):
    """
    Locate a station inside the full NBS
    bulletin and return enough lines for
    us to inspect its format.
    """

    lines = bulletin_text.splitlines()

    for index, line in enumerate(
        lines
    ):
        if line.strip().startswith(
            station
        ):
            end_index = min(
                index + lines_after,
                len(lines)
            )

            return "\n".join(
                lines[
                    index:end_index
                ]
            )

    return None


def inspect_nbm_station(
    run_time,
    station=NBM_STATION
):
    bulletin = get_nbm_nbs_bulletin(
        run_time
    )

    block = get_station_block(
        bulletin["text"],
        station=station
    )

    print()
    print("=" * 70)

    print(
        f"NBM NBS INSPECTION"
    )

    print(
        f"Run: "
        f"{bulletin['run_time']:%Y-%m-%d %HZ}"
    )

    print(
        f"Station: {station}"
    )

    print(
        f"URL: {bulletin['url']}"
    )

    print("=" * 70)

    if block is None:
        print(
            f"{station} not found."
        )

        return

    print(block)

def get_bulletin_row(
    station_block,
    row_name
):
    for line in station_block.splitlines():

        stripped = line.lstrip()

        if stripped.startswith(
            row_name
        ):
            return line

    return None


def parse_fixed_width_row(
    line,
    start=5,
    width=3
):
    """
    NBM station text rows use fixed
    3-character columns.

    Blank columns return None.
    """

    values = []

    for index in range(
        start,
        len(line),
        width
    ):
        field = line[
            index:index + width
        ].strip()

        if field == "":
            values.append(
                None
            )
        else:
            values.append(
                field
            )

    return values


def parse_nbm_max_temperatures(
    station_block,
    run_time
):
    """
    Parse TXN and XND into daily maximum
    temperature forecasts.

    For CONUS stations:
        TXN at 00Z = maximum temperature
        TXN at 12Z = minimum temperature

    A maximum reported at 00Z belongs to
    the preceding calendar day.
    """

    run_time = normalize_run_time(
        run_time
    )

    fhr_line = get_bulletin_row(
        station_block,
        "FHR"
    )

    txn_line = get_bulletin_row(
        station_block,
        "TXN"
    )

    xnd_line = get_bulletin_row(
        station_block,
        "XND"
    )

    if fhr_line is None:
        raise ValueError(
            "NBM bulletin missing FHR row."
        )

    if txn_line is None:
        raise ValueError(
            "NBM bulletin missing TXN row."
        )

    forecast_hours = (
        parse_fixed_width_row(
            fhr_line
        )
    )

    txn_values = (
        parse_fixed_width_row(
            txn_line
        )
    )

    if xnd_line is not None:

        xnd_values = (
            parse_fixed_width_row(
                xnd_line
            )
        )

    else:

        xnd_values = [
            None
            for _ in forecast_hours
        ]

    results = []

    for index, fhr_value in enumerate(
        forecast_hours
    ):

        if fhr_value is None:
            continue

        if index >= len(
            txn_values
        ):
            continue

        txn_value = txn_values[
            index
        ]

        if txn_value is None:
            continue

        forecast_hour = int(
            fhr_value
        )

        valid_time = (
            run_time
            + pd.Timedelta(
                hours=forecast_hour
            )
        )

        # For CONUS NBS guidance:
        # values valid at 00Z are Tmax.
        if valid_time.hour != 0:
            continue

        forecast_date = (
            valid_time
            - pd.Timedelta(
                days=1
            )
        ).date()

        forecast_std = None

        if index < len(
            xnd_values
        ):

            xnd_value = xnd_values[
                index
            ]

            if xnd_value is not None:

                forecast_std = float(
                    xnd_value
                )

        results.append(
            {
                "target_date":
                    forecast_date,

                "forecast_high":
                    float(txn_value),

                "forecast_std":
                    forecast_std,

                "forecast_hour":
                    forecast_hour,

                "valid_time":
                    valid_time,

                "run_time":
                    run_time
            }
        )

    return results

def get_nbm_high_forecast(
    run_time,
    target_date,
    station=NBM_STATION
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    bulletin = get_nbm_nbs_bulletin(
        run_time
    )

    station_block = get_station_block(
        bulletin["text"],
        station=station
    )

    if station_block is None:

        raise ValueError(
            f"NBM station {station} "
            f"not found."
        )

    forecasts = (
        parse_nbm_max_temperatures(
            station_block,
            bulletin["run_time"]
        )
    )

    for forecast in forecasts:

        if (
            forecast["target_date"]
            == target_date
        ):

            forecast[
                "station"
            ] = station

            forecast[
                "product"
            ] = "NBS"

            return forecast

    raise ValueError(
        f"No NBM maximum-temperature "
        f"forecast found for "
        f"{target_date}."
    )


if __name__ == "__main__":

    result = get_nbm_high_forecast(
        run_time="2026-09-09 12:00",
        target_date="2026-09-10"
    )

    print(result)