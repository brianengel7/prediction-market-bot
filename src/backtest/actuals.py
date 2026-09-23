import requests
import pandas as pd


IEM_CLI_URL = (
    "https://mesonet.agron.iastate.edu/"
    "json/cli.py"
)

CENTRAL_PARK_STATION = "KNYC"

REQUEST_TIMEOUT = 30


def get_cli_year(
    year,
    station=CENTRAL_PARK_STATION
):
    response = requests.get(
        IEM_CLI_URL,
        params={
            "station": station,
            "year": int(year),
            "fmt": "json"
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "results",
        []
    )


def parse_temperature(value):
    """
    Convert a CLI temperature value into float.

    Missing values such as M return None.
    """

    if value is None:
        return None

    if value in (
        "M",
        "T",
        ""
    ):
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return None


def get_actual_high(
    target_date,
    station=CENTRAL_PARK_STATION
):
    target_date = pd.Timestamp(
        target_date
    ).date()

    rows = get_cli_year(
        target_date.year,
        station=station
    )

    target_string = (
        target_date.isoformat()
    )

    for row in rows:

        if row.get("valid") != target_string:
            continue

        high = parse_temperature(
            row.get("high")
        )

        if high is None:
            return None

        return {
            "date": target_date,
            "station": station,
            "actual_high": high,
            "high_time": row.get(
                "high_time"
            ),
            "product": row.get(
                "product"
            ),
            "product_link": row.get(
                "link"
            )
        }

    return None

def get_actual_highs(
    start_date,
    end_date,
    station=CENTRAL_PARK_STATION
):
    start_date = pd.Timestamp(
        start_date
    ).date()

    end_date = pd.Timestamp(
        end_date
    ).date()

    results = []

    for year in range(
        start_date.year,
        end_date.year + 1
    ):

        rows = get_cli_year(
            year,
            station=station
        )

        for row in rows:

            valid = row.get(
                "valid"
            )

            if valid is None:
                continue

            date = pd.Timestamp(
                valid
            ).date()

            if not (
                start_date
                <= date
                <= end_date
            ):
                continue

            high = parse_temperature(
                row.get("high")
            )

            if high is None:
                continue

            results.append(
                {
                    "date": date,
                    "station": station,
                    "actual_high": high,
                    "high_time": row.get(
                        "high_time"
                    ),
                    "product": row.get(
                        "product"
                    ),
                    "product_link": row.get(
                        "link"
                    )
                }
            )

    results.sort(
        key=lambda result:
        result["date"]
    )

    return results


if __name__ == "__main__":

    actuals = get_actual_highs(
        "2026-09-01",
        "2026-09-20"
    )

    for actual in actuals:
        print(
            actual["date"],
            actual["actual_high"],
            actual["high_time"]
        )