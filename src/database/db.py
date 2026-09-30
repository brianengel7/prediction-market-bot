import os
import sqlite3
from pathlib import Path

import pandas as pd


DATABASE_PATH = Path(__file__).parent / "prediction_market.db"


class PostgresConnection:
    def __init__(self, url):
        import psycopg

        self.raw = psycopg.connect(
            url,
            sslmode="require",
            connect_timeout=10,
            prepare_threshold=None,
        )
        self.raw.execute("SET search_path TO prediction_bot, public")

    def __enter__(self):
        self.raw.__enter__()
        return self

    def __exit__(self, *args):
        return self.raw.__exit__(*args)

    def execute(self, query, params=None):
        return self.raw.execute(
            query.replace("?", "%s"),
            params or (),
        )


def get_connection():
    url = os.environ.get("PREDICTION_DATABASE_URL")
    if url:
        return PostgresConnection(url)
    return sqlite3.connect(DATABASE_PATH)

def read_dataframe(query, connection, params=None):
    cursor = connection.execute(query, params or ())
    return pd.DataFrame.from_records(
        cursor.fetchall(),
        columns=[column[0] for column in cursor.description],
    )


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

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_edge_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                snapshot_time TEXT NOT NULL,
                target_date TEXT NOT NULL,

                ticker TEXT NOT NULL,

                yes_bid REAL,
                yes_ask REAL,
                no_bid REAL,
                no_ask REAL,

                model_yes REAL NOT NULL,
                model_no REAL NOT NULL,

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
            CREATE TABLE IF NOT EXISTS weather_fair_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,
                target_date TEXT NOT NULL,

                ticker TEXT NOT NULL,
                title TEXT,

                strike_type TEXT,
                floor_strike REAL,
                cap_strike REAL,

                model_yes REAL NOT NULL,
                model_no REAL NOT NULL,

                fair_point_forecast REAL NOT NULL,
                residual_mean REAL NOT NULL,
                residual_std REAL NOT NULL,

                model_version TEXT NOT NULL,

                UNIQUE (
                    target_date,
                    ticker,
                    model_version
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_market_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                target_date TEXT NOT NULL,
                event_ticker TEXT NOT NULL,

                ticker TEXT NOT NULL,
                title TEXT,

                strike_type TEXT,
                floor_strike REAL,
                cap_strike REAL,

                market_status TEXT,
                result TEXT,
                settlement_value REAL,

                decision_time TEXT NOT NULL,
                entry_time TEXT NOT NULL,

                yes_bid REAL,
                yes_ask REAL,
                no_bid REAL,
                no_ask REAL,

                market_source TEXT,
                candle_source TEXT,

                created_at TEXT NOT NULL,

                UNIQUE (
                    target_date,
                    ticker,
                    decision_time
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS v2_shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,
                target_date TEXT NOT NULL,

                ticker TEXT NOT NULL,
                side TEXT NOT NULL,

                decision_time TEXT NOT NULL,

                kalshi_probability REAL NOT NULL,
                nbm_probability REAL NOT NULL,
                adjusted_probability REAL NOT NULL,

                entry_price REAL NOT NULL,
                fee REAL NOT NULL,
                total_cost REAL NOT NULL,
                net_edge REAL NOT NULL,

                beta REAL NOT NULL,
                minimum_edge REAL NOT NULL,

                settled INTEGER NOT NULL DEFAULT 0,

                won INTEGER,
                payout REAL,
                net_pnl REAL,

                UNIQUE (
                    target_date
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS v2_nbm_probabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,
                target_date TEXT NOT NULL,

                ticker TEXT NOT NULL,

                strike_type TEXT,
                floor_strike REAL,
                cap_strike REAL,

                nbm_probability REAL NOT NULL,

                nbm_forecast REAL NOT NULL,
                residual_std REAL NOT NULL,

                UNIQUE (
                    target_date,
                    ticker
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
                v2_market_only_shadow_decisions (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,
                target_date TEXT NOT NULL,
                decision_time TEXT NOT NULL,

                training_start TEXT NOT NULL,
                training_end TEXT NOT NULL,
                lookback_days INTEGER NOT NULL,

                alpha REAL NOT NULL,
                raw_mid_sum REAL NOT NULL,

                trade_taken INTEGER NOT NULL,

                ticker TEXT NOT NULL,
                side TEXT NOT NULL,

                kalshi_probability REAL NOT NULL,
                adjusted_probability REAL NOT NULL,

                entry_price REAL NOT NULL,
                fee REAL NOT NULL,
                total_cost REAL NOT NULL,
                net_edge REAL NOT NULL,

                minimum_edge REAL NOT NULL,

                settled INTEGER NOT NULL DEFAULT 0,
                won INTEGER,
                payout REAL,
                net_pnl REAL,

                UNIQUE(target_date)
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_orders (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,

                strategy TEXT NOT NULL,
                target_date TEXT NOT NULL,

                ticker TEXT NOT NULL,
                side TEXT NOT NULL,

                decision_time TEXT NOT NULL,

                entry_price REAL NOT NULL,
                net_edge REAL NOT NULL,

                contracts REAL NOT NULL,

                client_order_id TEXT NOT NULL,
                kalshi_order_id TEXT,

                order_status TEXT NOT NULL,

                fill_count REAL,
                remaining_count REAL,

                average_fill_price REAL,
                average_fee_paid REAL,

                error_message TEXT,

                settled INTEGER NOT NULL DEFAULT 0,
                won INTEGER,
                payout REAL,
                net_pnl REAL,

                UNIQUE(client_order_id)
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
            INSERT INTO weather_forecasts (
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
            ON CONFLICT (
                model, product, member, run_time,
                forecast_hour, latitude, longitude
            ) DO UPDATE SET
                valid_time = excluded.valid_time,
                temperature_f = excluded.temperature_f
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

                    result["ticker"],
                    result.get("title"),

                    result.get("strike_type"),
                    result.get("floor_strike"),
                    result.get("cap_strike"),

                    result.get("raw_gefs_yes"),

                    result["model_yes"],
                    result["model_no"],

                    result.get("bandwidth"),

                    result.get("yes_bid"),
                    result.get("yes_ask"),
                    result.get("no_bid"),
                    result.get("no_ask"),

                    result.get("yes_edge"),
                    result.get("no_edge"),

                    result.get("best_side"),
                    result.get("best_edge"),

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
            INSERT INTO weather_backtests (
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
            ON CONFLICT (target_date, forecast_horizon) DO UPDATE SET
                actual_high = excluded.actual_high,
                hrrr_run_time = excluded.hrrr_run_time,
                hrrr_high = excluded.hrrr_high,
                hrrr_error = excluded.hrrr_error,
                gefs_run_time = excluded.gefs_run_time,
                gefs_mean = excluded.gefs_mean,
                gefs_median = excluded.gefs_median,
                gefs_std = excluded.gefs_std,
                gefs_error = excluded.gefs_error,
                created_at = excluded.created_at
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

def save_weather_fair_values(
    target_date,
    edge_results,
    calibration,
    model_version
):
    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    point_forecast = float(
        calibration[
            "point_forecast"
        ]
    )

    residual_mean = float(
        calibration[
            "residual_mean"
        ]
    )

    residual_std = float(
        calibration[
            "residual_std"
        ]
    )

    saved_count = 0

    with get_connection() as connection:

        for result in edge_results:

            connection.execute(
                """
                INSERT INTO weather_fair_values (
                    created_at,
                    target_date,

                    ticker,
                    title,

                    strike_type,
                    floor_strike,
                    cap_strike,

                    model_yes,
                    model_no,

                    fair_point_forecast,
                    residual_mean,
                    residual_std,

                    model_version
                )

                VALUES (
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?
                )

                ON CONFLICT (
                    target_date,
                    ticker,
                    model_version
                )

                DO UPDATE SET
                    created_at =
                        excluded.created_at,

                    title =
                        excluded.title,

                    strike_type =
                        excluded.strike_type,

                    floor_strike =
                        excluded.floor_strike,

                    cap_strike =
                        excluded.cap_strike,

                    model_yes =
                        excluded.model_yes,

                    model_no =
                        excluded.model_no,

                    fair_point_forecast =
                        excluded.fair_point_forecast,

                    residual_mean =
                        excluded.residual_mean,

                    residual_std =
                        excluded.residual_std
                """,
                (
                    created_at,
                    str(
                        target_date
                    ),

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

                    float(
                        result[
                            "model_yes"
                        ]
                    ),

                    float(
                        result[
                            "model_no"
                        ]
                    ),

                    point_forecast,
                    residual_mean,
                    residual_std,

                    model_version
                )
            )

            saved_count += 1

    return saved_count

def get_weather_fair_values(
    target_date,
    model_version
):
    with get_connection() as connection:

        query = """
        SELECT
            ticker,
            title,
            strike_type,
            floor_strike,
            cap_strike,
            model_yes,
            model_no,
            fair_point_forecast,
            residual_mean,
            residual_std,
            created_at

        FROM weather_fair_values

        WHERE target_date = ?
          AND model_version = ?
        """

        dataframe = read_dataframe(
            query,
            connection,
            params=(
                str(
                    target_date
                ),
                model_version
            )
        )

    return dataframe

def get_market_snapshots(
    target_date=None,
    model_version=None
):
    with get_connection() as connection:

        query = """
        SELECT
            snapshot_time,
            target_date,

            ticker,
            title,

            strike_type,
            floor_strike,
            cap_strike,

            model_yes,
            model_no,

            yes_bid,
            yes_ask,
            no_bid,
            no_ask,

            yes_edge,
            no_edge,

            best_side,
            best_edge,

            model_version

        FROM market_snapshots
        """

        conditions = []
        params = []

        if target_date is not None:

            conditions.append(
                "target_date = ?"
            )

            params.append(
                str(target_date)
            )

        if model_version is not None:

            conditions.append(
                "model_version = ?"
            )

            params.append(
                model_version
            )

        if conditions:

            query += (
                " WHERE "
                + " AND ".join(
                    conditions
                )
            )

        query += """
        ORDER BY
            snapshot_time ASC,
            ticker ASC
        """

        dataframe = read_dataframe(
            query,
            connection,
            params=params
        )

    if not dataframe.empty:

        dataframe[
            "snapshot_time"
        ] = pd.to_datetime(
            dataframe[
                "snapshot_time"
            ],
            utc=True
        )

    return dataframe

def save_historical_market_entries(
    entries
):

    if not entries:
        return 0

    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    saved_count = 0

    with get_connection() as connection:

        for entry in entries:

            settlement_value = (
                entry.get(
                    "settlement_value"
                )
            )

            if settlement_value in (
                None,
                ""
            ):
                settlement_value = None

            else:
                settlement_value = float(
                    settlement_value
                )

            connection.execute(
                """
                INSERT INTO historical_market_entries (
                    target_date,
                    event_ticker,

                    ticker,
                    title,

                    strike_type,
                    floor_strike,
                    cap_strike,

                    market_status,
                    result,
                    settlement_value,

                    decision_time,
                    entry_time,

                    yes_bid,
                    yes_ask,
                    no_bid,
                    no_ask,

                    market_source,
                    candle_source,

                    created_at
                )

                VALUES (
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?
                )

                ON CONFLICT (
                    target_date,
                    ticker,
                    decision_time
                )

                DO UPDATE SET
                    event_ticker =
                        excluded.event_ticker,

                    title =
                        excluded.title,

                    strike_type =
                        excluded.strike_type,

                    floor_strike =
                        excluded.floor_strike,

                    cap_strike =
                        excluded.cap_strike,

                    market_status =
                        excluded.market_status,

                    result =
                        excluded.result,

                    settlement_value =
                        excluded.settlement_value,

                    entry_time =
                        excluded.entry_time,

                    yes_bid =
                        excluded.yes_bid,

                    yes_ask =
                        excluded.yes_ask,

                    no_bid =
                        excluded.no_bid,

                    no_ask =
                        excluded.no_ask,

                    market_source =
                        excluded.market_source,

                    candle_source =
                        excluded.candle_source,

                    created_at =
                        excluded.created_at
                """,
                (
                    entry[
                        "target_date"
                    ],

                    entry[
                        "event_ticker"
                    ],

                    entry[
                        "ticker"
                    ],

                    entry.get(
                        "title"
                    ),

                    entry.get(
                        "strike_type"
                    ),

                    entry.get(
                        "floor_strike"
                    ),

                    entry.get(
                        "cap_strike"
                    ),

                    entry.get(
                        "market_status"
                    ),

                    entry.get(
                        "result"
                    ),

                    settlement_value,

                    str(
                        entry[
                            "decision_time"
                        ]
                    ),

                    str(
                        entry[
                            "entry_time"
                        ]
                    ),

                    entry.get(
                        "yes_bid"
                    ),

                    entry.get(
                        "yes_ask"
                    ),

                    entry.get(
                        "no_bid"
                    ),

                    entry.get(
                        "no_ask"
                    ),

                    entry.get(
                        "market_source"
                    ),

                    entry.get(
                        "candle_source"
                    ),

                    created_at
                )
            )

            saved_count += 1

    return saved_count

def get_historical_market_entries(
    start_date=None,
    end_date=None,
    target_date=None
):

    query = """
        SELECT *

        FROM historical_market_entries
    """

    conditions = []
    params = []

    if target_date is not None:

        conditions.append(
            "target_date = ?"
        )

        params.append(
            str(target_date)
        )

    if start_date is not None:

        conditions.append(
            "target_date >= ?"
        )

        params.append(
            str(start_date)
        )

    if end_date is not None:

        conditions.append(
            "target_date <= ?"
        )

        params.append(
            str(end_date)
        )

    if conditions:

        query += (
            " WHERE "
            + " AND ".join(
                conditions
            )
        )

    query += """
        ORDER BY
            target_date,
            entry_time,
            ticker
    """

    with get_connection() as connection:

        dataframe = read_dataframe(
            query,
            connection,
            params=params
        )

    if not dataframe.empty:

        dataframe[
            "decision_time"
        ] = pd.to_datetime(
            dataframe[
                "decision_time"
            ],
            utc=True
        )

        dataframe[
            "entry_time"
        ] = pd.to_datetime(
            dataframe[
                "entry_time"
            ],
            utc=True
        )

    return dataframe

def save_v2_nbm_probabilities(
    target_date,
    probabilities
):
    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    with get_connection() as connection:

        for row in probabilities:

            connection.execute(
                """
                INSERT INTO
                    v2_nbm_probabilities (
                        created_at,
                        target_date,
                        ticker,
                        strike_type,
                        floor_strike,
                        cap_strike,
                        nbm_probability,
                        nbm_forecast,
                        residual_std
                    )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT (target_date, ticker) DO UPDATE SET
                    created_at = excluded.created_at,
                    strike_type = excluded.strike_type,
                    floor_strike = excluded.floor_strike,
                    cap_strike = excluded.cap_strike,
                    nbm_probability = excluded.nbm_probability,
                    nbm_forecast = excluded.nbm_forecast,
                    residual_std = excluded.residual_std
                """,
                (
                    created_at,
                    str(target_date),

                    row[
                        "ticker"
                    ],

                    row.get(
                        "strike_type"
                    ),

                    row.get(
                        "floor_strike"
                    ),

                    row.get(
                        "cap_strike"
                    ),

                    float(
                        row[
                            "nbm_probability"
                        ]
                    ),

                    float(
                        row[
                            "nbm_forecast"
                        ]
                    ),

                    float(
                        row[
                            "residual_std"
                        ]
                    )
                )
            )

def get_v2_nbm_probabilities(
    target_date
):

    with get_connection() as connection:

        return read_dataframe(
            """
            SELECT *
            FROM v2_nbm_probabilities
            WHERE target_date = ?
            ORDER BY ticker
            """,
            connection,
            params=[
                str(target_date)
            ]
        )

def save_v2_shadow_trade(
    trade
):

    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO v2_shadow_trades (
                created_at,
                target_date,
                ticker,
                side,
                decision_time,
                kalshi_probability,
                nbm_probability,
                adjusted_probability,
                entry_price,
                fee,
                total_cost,
                net_edge,
                beta,
                minimum_edge
            )

            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                created_at,

                str(
                    trade[
                        "target_date"
                    ]
                ),

                trade[
                    "ticker"
                ],

                trade[
                    "side"
                ],

                str(
                    trade[
                        "decision_time"
                    ]
                ),

                float(
                    trade[
                        "kalshi_probability"
                    ]
                ),

                float(
                    trade[
                        "nbm_probability"
                    ]
                ),

                float(
                    trade[
                        "adjusted_probability"
                    ]
                ),

                float(
                    trade[
                        "entry_price"
                    ]
                ),

                float(
                    trade[
                        "fee"
                    ]
                ),

                float(
                    trade[
                        "total_cost"
                    ]
                ),

                float(
                    trade[
                        "net_edge"
                    ]
                ),

                float(
                    trade[
                        "beta"
                    ]
                ),

                float(
                    trade[
                        "minimum_edge"
                    ]
                )
            )
        )

def get_v2_shadow_trades():

    with get_connection() as connection:

        return read_dataframe(
            """
            SELECT *
            FROM v2_shadow_trades
            ORDER BY target_date
            """,
            connection
        )

def save_v2_market_only_shadow_decision(
    decision
):

    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    trade_taken = bool(
        decision[
            "trade_taken"
        ]
    )

    # PASS days are already complete decisions.
    settled = (
        0
        if trade_taken
        else 1
    )

    net_pnl = (
        None
        if trade_taken
        else 0.0
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO
                v2_market_only_shadow_decisions (

                created_at,
                target_date,
                decision_time,

                training_start,
                training_end,
                lookback_days,

                alpha,
                raw_mid_sum,

                trade_taken,

                ticker,
                side,

                kalshi_probability,
                adjusted_probability,

                entry_price,
                fee,
                total_cost,
                net_edge,

                minimum_edge,

                settled,
                net_pnl
            )

            VALUES (
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?,
                ?, ?
            )
            """,

            (
                created_at,

                str(
                    decision[
                        "target_date"
                    ]
                ),

                str(
                    decision[
                        "decision_time"
                    ]
                ),

                str(
                    decision[
                        "training_start"
                    ]
                ),

                str(
                    decision[
                        "training_end"
                    ]
                ),

                int(
                    decision[
                        "lookback_days"
                    ]
                ),

                float(
                    decision[
                        "alpha"
                    ]
                ),

                float(
                    decision[
                        "raw_mid_sum"
                    ]
                ),

                int(
                    trade_taken
                ),

                decision[
                    "ticker"
                ],

                decision[
                    "side"
                ],

                float(
                    decision[
                        "kalshi_probability"
                    ]
                ),

                float(
                    decision[
                        "adjusted_probability"
                    ]
                ),

                float(
                    decision[
                        "entry_price"
                    ]
                ),

                float(
                    decision[
                        "fee"
                    ]
                ),

                float(
                    decision[
                        "total_cost"
                    ]
                ),

                float(
                    decision[
                        "net_edge"
                    ]
                ),

                float(
                    decision[
                        "minimum_edge"
                    ]
                ),

                settled,
                net_pnl
            )
        )


def get_v2_market_only_shadow_decisions():

    with get_connection() as connection:

        return read_dataframe(
            """
            SELECT *
            FROM v2_market_only_shadow_decisions
            ORDER BY target_date
            """,

            connection
        )

def update_v2_shadow_trade_settlement(
    trade_id,
    won,
    payout,
    net_pnl
):

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE v2_shadow_trades

            SET
                settled = 1,
                won = ?,
                payout = ?,
                net_pnl = ?

            WHERE id = ?
            """,

            (
                int(bool(won)),
                float(payout),
                float(net_pnl),
                int(trade_id)
            )
        )


def update_v2_market_only_shadow_settlement(
    decision_id,
    won,
    payout,
    net_pnl
):

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE v2_market_only_shadow_decisions

            SET
                settled = 1,
                won = ?,
                payout = ?,
                net_pnl = ?

            WHERE id = ?
            """,

            (
                int(bool(won)),
                float(payout),
                float(net_pnl),
                int(decision_id)
            )
        )

def save_live_order(
    order
):

    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO live_orders (

                created_at,

                strategy,
                target_date,

                ticker,
                side,

                decision_time,

                entry_price,
                net_edge,

                contracts,

                client_order_id,
                kalshi_order_id,

                order_status,

                fill_count,
                remaining_count,

                average_fill_price,
                average_fee_paid,

                error_message
            )

            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )

            ON CONFLICT(client_order_id)

            DO UPDATE SET

                kalshi_order_id =
                    excluded.kalshi_order_id,

                order_status =
                    excluded.order_status,

                fill_count =
                    excluded.fill_count,

                remaining_count =
                    excluded.remaining_count,

                average_fill_price =
                    excluded.average_fill_price,

                average_fee_paid =
                    excluded.average_fee_paid,

                error_message =
                    excluded.error_message
            """,

            (
                created_at,

                order[
                    "strategy"
                ],

                str(
                    order[
                        "target_date"
                    ]
                ),

                order[
                    "ticker"
                ],

                order[
                    "side"
                ],

                str(
                    order[
                        "decision_time"
                    ]
                ),

                float(
                    order[
                        "entry_price"
                    ]
                ),

                float(
                    order[
                        "net_edge"
                    ]
                ),

                float(
                    order[
                        "contracts"
                    ]
                ),

                order[
                    "client_order_id"
                ],

                order.get(
                    "kalshi_order_id"
                ),

                order[
                    "order_status"
                ],

                order.get(
                    "fill_count"
                ),

                order.get(
                    "remaining_count"
                ),

                order.get(
                    "average_fill_price"
                ),

                order.get(
                    "average_fee_paid"
                ),

                order.get(
                    "error_message"
                )
            )
        )


def get_live_orders():

    with get_connection() as connection:

        return read_dataframe(
            """
            SELECT *
            FROM live_orders
            ORDER BY created_at
            """,
            connection
        )

def update_live_order_reconciliation(
    order_id,
    fill_count,
    average_fill_price,
    average_fee_paid,
    order_status
):

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE live_orders

            SET
                fill_count = ?,
                average_fill_price = ?,
                average_fee_paid = ?,
                order_status = ?,
                error_message = NULL

            WHERE id = ?
            """,

            (
                float(
                    fill_count
                ),

                float(
                    average_fill_price
                ),

                float(
                    average_fee_paid
                ),

                order_status,

                int(
                    order_id
                )
            )
        )

def update_live_order_settlement(
    order_id,
    won,
    payout,
    net_pnl
):

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE live_orders

            SET
                settled = 1,
                won = ?,
                payout = ?,
                net_pnl = ?,
                order_status = 'SETTLED'

            WHERE id = ?
            """,

            (
                int(
                    bool(won)
                ),

                float(
                    payout
                ),

                float(
                    net_pnl
                ),

                int(
                    order_id
                )
            )
        )

if not os.environ.get("PREDICTION_DATABASE_URL"):
    initialize_database()

