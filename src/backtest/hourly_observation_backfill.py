import argparse
import hashlib
import io
import json
import math
import os
from datetime import date, timedelta

import pandas as pd
import requests

from src.database.db import get_connection

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
SOURCE = "IEM_ASOS_ARCHIVE"


def backfill_observations(start_date, end_date):
    if not os.getenv("PREDICTION_DATABASE_URL", "").strip():
        raise RuntimeError("PREDICTION_DATABASE_URL is missing.")

    first_day = date.fromisoformat(start_date)
    last_day = date.fromisoformat(end_date)

    if last_day < first_day:
        raise ValueError("end-date must be on or after start-date.")

    start = pd.Timestamp(
        first_day, tz="America/New_York"
    ).tz_convert("UTC")

    end = pd.Timestamp(
        last_day + timedelta(days=1),
        tz="America/New_York",
    ).tz_convert("UTC")

    response = requests.get(
        IEM_URL,
        params={
            "station": "NYC",
            "network": "NY_ASOS",
            "data": [
                "tmpf", "dwpf", "drct",
                "sknt", "skyc1", "metar",
            ],
            "report_type": ["3", "4"],
            "sts": start.isoformat(),
            "ets": end.isoformat(),
            "tz": "UTC",
            "format": "onlycomma",
            "missing": "M",
            "latlon": "no",
            "elev": "no",
        },
        timeout=60,
    )
    response.raise_for_status()

    frame = pd.read_csv(
        io.StringIO(response.text),
        comment="#",
        dtype=str,
        na_values=["M"],
    )

    required = {"station", "valid", "tmpf"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            f"Unexpected IEM columns: {list(frame.columns)}"
        )

    timestamps = pd.to_datetime(
        frame["valid"], utc=True, errors="coerce"
    )

    if timestamps.isna().any():
        raise RuntimeError(
            "IEM returned an invalid observation timestamp."
        )

    downloaded_at = pd.Timestamp.now(tz="UTC").isoformat()
    rows = []

    for report, observed_at in zip(
        frame.to_dict("records"), timestamps
    ):
        if not start <= observed_at < end:
            continue

        if str(report["station"]).upper() not in {"NYC", "KNYC"}:
            raise RuntimeError("IEM returned an unexpected station.")

        temperature = (
            None
            if pd.isna(report["tmpf"])
            else float(report["tmpf"])
        )

        if temperature is not None and not math.isfinite(temperature):
            temperature = None

        raw = {
            key: None if pd.isna(value) else value
            for key, value in report.items()
        }

        payload = json.dumps(
            raw, sort_keys=True, allow_nan=False
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()

        rows.append((
            SOURCE,
            "KNYC",
            observed_at.isoformat(),
            temperature,
            digest,
            payload,
            downloaded_at,
        ))

    if not rows:
        raise RuntimeError(
            "No observations returned for this date range."
        )

    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS weather_observation_history (
                source TEXT NOT NULL,
                station TEXT NOT NULL,
                observed_at_utc TEXT NOT NULL,
                temperature_f DOUBLE PRECISION,
                payload_hash TEXT NOT NULL,
                raw_payload TEXT NOT NULL,
                available_at_utc TEXT,
                downloaded_at_utc TEXT NOT NULL,
                PRIMARY KEY (
                    source, station, observed_at_utc, payload_hash
                )
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_observation_history_time
            ON weather_observation_history (station, observed_at_utc)
        """)

        for offset in range(0, len(rows), 100):
            batch = rows[offset:offset + 100]
            placeholders = ",".join(
                ["(?, ?, ?, ?, ?, ?, ?)"] * len(batch)
            )

            connection.execute(
                f"""
                INSERT INTO weather_observation_history (
                    source, station, observed_at_utc, temperature_f,
                    payload_hash, raw_payload, downloaded_at_utc
                ) VALUES {placeholders}
                ON CONFLICT (
                    source, station, observed_at_utc, payload_hash
                ) DO NOTHING
                """,
                tuple(
                    value
                    for row in batch
                    for value in row
                ),
            )

        stored = connection.execute(
            """
            SELECT COUNT(*) FROM weather_observation_history
            WHERE source = ? AND station = ?
              AND observed_at_utc >= ? AND observed_at_utc < ?
            """,
            (
                SOURCE,
                "KNYC",
                start.isoformat(),
                end.isoformat(),
            ),
        ).fetchone()[0]

    print(f"UTC range: {start.isoformat()} -> {end.isoformat()}")
    print(f"Reports fetched in range: {len(rows)}")
    print(f"Reports stored in range: {stored}")
    print(
        "Historical publication times remain unknown "
        "(available_at_utc = NULL)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()

    backfill_observations(args.start_date, args.end_date)