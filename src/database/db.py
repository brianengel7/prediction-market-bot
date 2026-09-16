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

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                snapshot_time TEXT NOT NULL,
                target_date TEXT NOT NULL,

                gefs_run_time TEXT,
                gefs_mean_high REAL,
                gefs_median_high REAL,
                gefs_std_high REAL,
                gefs_min_high REAL,
                gefs_max_high REAL,

                hrrr_latest_run TEXT,
                hrrr_latest_high REAL,
                hrrr_latest_change REAL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                snapshot_time TEXT NOT NULL,
                target_date TEXT NOT NULL,

                ticker TEXT NOT NULL,
                title TEXT,

                strike_type TEXT,
                floor_strike REAL,
                cap_strike REAL,

                raw_gefs_yes REAL,
                model_yes REAL,
                model_no REAL,
                bandwidth REAL,

                yes_bid REAL,
                yes_ask REAL,
                no_bid REAL,
                no_ask REAL,

                yes_edge REAL,
                no_edge REAL,

                best_side TEXT,
                best_edge REAL
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

def save_model_snapshot(
    target_date,
    gefs_results,
    hrrr_results
):
    snapshot_time = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    latest_hrrr = hrrr_results[-1]

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO model_snapshots (
                snapshot_time,
                target_date,

                gefs_run_time,
                gefs_mean_high,
                gefs_median_high,
                gefs_std_high,
                gefs_min_high,
                gefs_max_high,

                hrrr_latest_run,
                hrrr_latest_high,
                hrrr_latest_change
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_time,
                str(target_date),

                str(
                    gefs_results["run_time"]
                ),

                gefs_results[
                    "mean_high"
                ],

                gefs_results[
                    "median_high"
                ],

                gefs_results[
                    "std_high"
                ],

                gefs_results[
                    "min_high"
                ],

                gefs_results[
                    "max_high"
                ],

                str(
                    latest_hrrr[
                        "run_time"
                    ]
                ),

                latest_hrrr[
                    "max_temperature"
                ],

                latest_hrrr[
                    "change"
                ]
            )
        )

    return snapshot_time

def save_market_snapshots(
    snapshot_time,
    target_date,
    edge_results
):
    with get_connection() as connection:

        for result in edge_results:

            connection.execute(
                """
                INSERT INTO market_snapshots (
                    snapshot_time,
                    target_date,

                    ticker,
                    title,

                    strike_type,
                    floor_strike,
                    cap_strike,

                    raw_gefs_yes,
                    model_yes,
                    model_no,
                    bandwidth,

                    yes_bid,
                    yes_ask,
                    no_bid,
                    no_ask,

                    yes_edge,
                    no_edge,

                    best_side,
                    best_edge
                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    snapshot_time,
                    str(target_date),

                    result[
                        "ticker"
                    ],

                    result[
                        "title"
                    ],

                    result[
                        "strike_type"
                    ],

                    result[
                        "floor_strike"
                    ],

                    result[
                        "cap_strike"
                    ],

                    result[
                        "raw_gefs_yes"
                    ],

                    result[
                        "model_yes"
                    ],

                    result[
                        "model_no"
                    ],

                    result[
                        "bandwidth"
                    ],

                    result[
                        "yes_bid"
                    ],

                    result[
                        "yes_ask"
                    ],

                    result[
                        "no_bid"
                    ],

                    result[
                        "no_ask"
                    ],

                    result[
                        "yes_edge"
                    ],

                    result[
                        "no_edge"
                    ],

                    result[
                        "best_side"
                    ],

                    result[
                        "best_edge"
                    ]
                )
            )

initialize_database()

