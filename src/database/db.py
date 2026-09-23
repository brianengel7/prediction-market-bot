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
                product TEXT NOT NULL,
                member TEXT NOT NULL,
                run_time TEXT NOT NULL,
                forecast_hour INTEGER NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                valid_time TEXT NOT NULL,
                temperature_f REAL NOT NULL,

                PRIMARY KEY (
                    model,
                    product,
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
                hrrr_latest_change REAL,

                gefs_product TEXT,
                nws_high REAL,
                model_version TEXT NOT NULL
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
                best_edge REAL,

                model_version TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                target_date TEXT NOT NULL,
                forecast_horizon TEXT NOT NULL,

                actual_high REAL NOT NULL,

                hrrr_run_time TEXT,
                hrrr_high REAL,
                hrrr_error REAL,

                gefs_run_time TEXT,
                gefs_mean REAL,
                gefs_median REAL,
                gefs_std REAL,
                gefs_error REAL,

                created_at TEXT NOT NULL,

                UNIQUE (
                    target_date,
                    forecast_horizon
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_actuals (
                target_date TEXT NOT NULL,
                station TEXT NOT NULL,

                actual_high REAL NOT NULL,

                settlement_source TEXT NOT NULL,
                product TEXT,

                created_at TEXT NOT NULL,

                PRIMARY KEY (
                    target_date,
                    station,
                    settlement_source
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_model_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                target_date TEXT NOT NULL,
                station TEXT NOT NULL,

                source TEXT NOT NULL,
                product TEXT NOT NULL,
                model_run TEXT NOT NULL,

                forecast_horizon TEXT NOT NULL,

                forecast_value REAL NOT NULL,
                forecast_median REAL,
                forecast_std REAL,
                member_count INTEGER,

                extraction_method TEXT,
                model_version TEXT NOT NULL,

                actual_high REAL NOT NULL,

                error REAL NOT NULL,
                absolute_error REAL NOT NULL,
                squared_error REAL NOT NULL,

                created_at TEXT NOT NULL,

                UNIQUE (
                    target_date,
                    station,
                    source,
                    product,
                    model_run,
                    forecast_horizon
                )
            )
            """
        )


def get_cached_temperature(
    model,
    product,
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
              AND product = ?
              AND member = ?
              AND run_time = ?
              AND forecast_hour = ?
              AND latitude = ?
              AND longitude = ?
            """,
            (
                model,
                product,
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
    product,
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
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                model TEXT NOT NULL,
                product TEXT NOT NULL,
                member TEXT NOT NULL,

                run_time TEXT NOT NULL,
                forecast_hour INTEGER NOT NULL,

                latitude REAL NOT NULL,
                longitude REAL NOT NULL,

                valid_time TEXT NOT NULL,
                temperature_f REAL NOT NULL,

                PRIMARY KEY (
                    model,
                    product,
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
            INSERT OR REPLACE INTO weather_forecasts (
                model,
                product,
                member,
                run_time,
                forecast_hour,
                latitude,
                longitude,
                valid_time,
                temperature_f
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model,
                product,
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
    hrrr_results,
    nws_result,
    model_version
):
    snapshot_time = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    latest_hrrr = hrrr_results[-1]

    nws_high = None

    if nws_result is not None:
        nws_high = nws_result[
            "temperature"
        ]

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
                hrrr_latest_change,

                nws_high,
                model_version
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ],

                (
                nws_result["temperature"]
                    if nws_result is not None
                    else None
                ),

                model_version
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
                    best_edge,
                    model_version
                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
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
                    ],

                    result["model_version"]
                )
            )

def save_weather_backtest(
    target_date,
    actual_high,
    hrrr_result=None,
    gefs_result=None,
    forecast_horizon="D-1_12Z"
):
    snapshot_time = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    hrrr_run_time = None
    hrrr_high = None
    hrrr_error = None

    if hrrr_result is not None:
        hrrr_run_time = str(
            hrrr_result["run_time"]
        )

        hrrr_high = hrrr_result[
            "forecast_high"
        ]

        hrrr_error = hrrr_result[
            "error"
        ]

    gefs_run_time = None
    gefs_mean = None
    gefs_median = None
    gefs_std = None
    gefs_error = None

    if gefs_result is not None:
        gefs_run_time = str(
            gefs_result["run_time"]
        )

        gefs_mean = gefs_result[
            "mean_high"
        ]

        gefs_median = gefs_result[
            "median_high"
        ]

        gefs_std = gefs_result[
            "std_high"
        ]

        gefs_error = gefs_result[
            "mean_error"
        ]

    with get_connection() as connection:

        connection.execute(
            """
            INSERT OR REPLACE INTO weather_backtests (
                target_date,
                forecast_horizon,

                actual_high,

                hrrr_run_time,
                hrrr_high,
                hrrr_error,

                gefs_run_time,
                gefs_mean,
                gefs_median,
                gefs_std,
                gefs_error,

                created_at
            )

            VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                str(target_date),
                forecast_horizon,

                actual_high,

                hrrr_run_time,
                hrrr_high,
                hrrr_error,

                gefs_run_time,
                gefs_mean,
                gefs_median,
                gefs_std,
                gefs_error,

                snapshot_time
            )
        )

def save_weather_actual(
    target_date,
    actual_high,
    station="KNYC",
    settlement_source="NWS_CLI",
    product="CLI"
):
    created_at = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO weather_actuals (
                target_date,
                station,
                actual_high,
                settlement_source,
                product,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT (
                target_date,
                station,
                settlement_source
            )

            DO UPDATE SET
                actual_high = excluded.actual_high,
                product = excluded.product,
                created_at = excluded.created_at
            """,
            (
                str(target_date),
                station,
                float(actual_high),
                settlement_source,
                product,
                created_at
            )
        )

def save_weather_model_backtest(
    target_date,
    source,
    product,
    model_run,
    forecast_horizon,
    forecast_value,
    actual_high,

    station="KNYC",
    forecast_median=None,
    forecast_std=None,
    member_count=None,
    extraction_method=None,
    model_version="backtest-v1"
):
    created_at = pd.Timestamp.now(
        tz="UTC"
    ).isoformat()

    error = (
        float(forecast_value)
        - float(actual_high)
    )

    absolute_error = abs(
        error
    )

    squared_error = (
        error ** 2
    )

    model_run = str(
        model_run
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO weather_model_backtests (
                target_date,
                station,

                source,
                product,
                model_run,

                forecast_horizon,

                forecast_value,
                forecast_median,
                forecast_std,
                member_count,

                extraction_method,
                model_version,

                actual_high,

                error,
                absolute_error,
                squared_error,

                created_at
            )

            VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )

            ON CONFLICT (
                target_date,
                station,
                source,
                product,
                model_run,
                forecast_horizon
            )

            DO UPDATE SET
                forecast_value =
                    excluded.forecast_value,

                forecast_median =
                    excluded.forecast_median,

                forecast_std =
                    excluded.forecast_std,

                member_count =
                    excluded.member_count,

                extraction_method =
                    excluded.extraction_method,

                model_version =
                    excluded.model_version,

                actual_high =
                    excluded.actual_high,

                error =
                    excluded.error,

                absolute_error =
                    excluded.absolute_error,

                squared_error =
                    excluded.squared_error,

                created_at =
                    excluded.created_at
            """,
            (
                str(target_date),
                station,

                source,
                product,
                model_run,

                forecast_horizon,

                float(forecast_value),
                forecast_median,
                forecast_std,
                member_count,

                extraction_method,
                model_version,

                float(actual_high),

                error,
                absolute_error,
                squared_error,

                created_at
            )
        )

initialize_database()

