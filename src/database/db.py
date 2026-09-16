import sqlite3
from pathlib import Path
import pandas as pd


DATABASE_PATH = Path(__file__).parent / "prediction_market.db"


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                model TEXT NOT NULL,
                member TEXT NOT NULL,
                run_time TEXT NOT NULL,
                forecast_hour INTEGER NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                valid_time TEXT NOT NULL,
                temperature_f REAL NOT NULL,

                PRIMARY KEY (
                    model,
                    member,
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
    longitude,
    member="deterministic"
):
    run_time = pd.Timestamp(run_time)

    if run_time.tzinfo is None:
        run_time = run_time.tz_localize("UTC")
    else:
        run_time = run_time.tz_convert("UTC")

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT valid_time, temperature_f
            FROM weather_forecasts
            WHERE model = ?
              AND member = ?
              AND run_time = ?
              AND forecast_hour = ?
              AND latitude = ?
              AND longitude = ?
            """,
            (
                model,
                member,
                run_time.isoformat(),
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
    temperature_f,
    member="deterministic"
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
            INSERT OR REPLACE INTO weather_forecasts (
                model,
                member,
                run_time,
                forecast_hour,
                latitude,
                longitude,
                valid_time,
                temperature_f
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model,
                member,
                run_time.isoformat(),
                forecast_hour,
                latitude,
                longitude,
                valid_time.isoformat(),
                temperature_f
            )
        )

initialize_database()