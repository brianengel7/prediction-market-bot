import sqlite3
from pathlib import Path
import pandas as pd


CACHE_DIRECTORY = Path("data")
CACHE_DIRECTORY.mkdir(exist_ok=True)

DATABASE_PATH = CACHE_DIRECTORY / "weather_cache.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_cache():
    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS temperature_forecasts (
                model TEXT NOT NULL,
                run_time TEXT NOT NULL,
                forecast_hour INTEGER NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                valid_time TEXT NOT NULL,
                temperature_f REAL NOT NULL,

                PRIMARY KEY (
                    model,
                    run_time,
                    forecast_hour,
                    latitude,
                    longitude
                )
            )
            """
        )


def get_cached_temperature(
    model,
    run_time,
    forecast_hour,
    latitude,
    longitude
):
    run_time = pd.Timestamp(run_time)

    if run_time.tzinfo is None:
        run_time = run_time.tz_localize("UTC")
    else:
        run_time = run_time.tz_convert("UTC")

    run_time_string = run_time.isoformat()

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT valid_time, temperature_f
            FROM temperature_forecasts
            WHERE model = ?
              AND run_time = ?
              AND forecast_hour = ?
              AND latitude = ?
              AND longitude = ?
            """,
            (
                model,
                run_time_string,
                forecast_hour,
                latitude,
                longitude
            )
        )

        row = cursor.fetchone()

    if row is None:
        return None

    valid_time, temperature_f = row

    return {
        "forecast_hour": forecast_hour,
        "valid_time": pd.Timestamp(valid_time),
        "temperature": temperature_f
    }


def save_temperature(
    model,
    run_time,
    forecast_hour,
    latitude,
    longitude,
    valid_time,
    temperature_f
):
    run_time = pd.Timestamp(run_time)

    if run_time.tzinfo is None:
        run_time = run_time.tz_localize("UTC")
    else:
        run_time = run_time.tz_convert("UTC")

    valid_time = pd.Timestamp(valid_time)

    with get_connection() as connection:

        connection.execute(
            """
            INSERT OR REPLACE INTO temperature_forecasts (
                model,
                run_time,
                forecast_hour,
                latitude,
                longitude,
                valid_time,
                temperature_f
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model,
                run_time.isoformat(),
                forecast_hour,
                latitude,
                longitude,
                valid_time.isoformat(),
                temperature_f
            )
        )


initialize_cache()