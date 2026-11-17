#!/usr/bin/env python3
"""Multicity V8 research data collection. No orders."""

import argparse
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("data/research/v8_multicity")
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
GFS = "https://single-runs-api.open-meteo.com/v1/forecast"

# City:
# (name, primary station, state, latitude, longitude,
#  timezone, [(nearby station, state), ...])
#
# Initial registry; station assignments must be verified
# against current Kalshi rules before production use.

CITIES = {
    "NYC": ("New York City", "NYC", "NY", 40.77898, -73.96925,
            "America/New_York", [("LGA","NY"),("EWR","NJ"),("TEB","NJ")]),
    "CHI": ("Chicago", "MDW", "IL", 41.7868, -87.7522,
            "America/Chicago", [("ORD","IL"),("PWK","IL"),("GYY","IN")]),
    "LAX": ("Los Angeles", "LAX", "CA", 33.9416, -118.4085,
            "America/Los_Angeles", [("BUR","CA"),("LGB","CA"),("SMO","CA")]),
    "MIA": ("Miami", "MIA", "FL", 25.7954, -80.2900,
            "America/New_York", [("OPF","FL"),("FLL","FL"),("FXE","FL")]),
    "DEN": ("Denver", "DEN", "CO", 39.8561, -104.6737,
            "America/Denver", [("APA","CO"),("BJC","CO"),("BKF","CO")]),
    "PHL": ("Philadelphia", "PHL", "PA", 39.8744, -75.2424,
            "America/New_York", [("PNE","PA"),("ILG","DE"),("TTN","NJ")]),
    "AUS": ("Austin", "AUS", "TX", 30.1975, -97.6664,
            "America/Chicago", [("ATT","TX"),("EDC","TX"),("GTU","TX")]),
    "HOU": ("Houston", "HOU", "TX", 29.6454, -95.2789,
            "America/Chicago", [("IAH","TX"),("EFD","TX"),("DWH","TX")]),
    "DAL": ("Dallas", "DFW", "TX", 32.8998, -97.0403,
            "America/Chicago", [("DAL","TX"),("AFW","TX"),("ADS","TX")]),
    "PHX": ("Phoenix", "PHX", "AZ", 33.4342, -112.0116,
            "America/Phoenix", [("SDL","AZ"),("DVT","AZ"),("IWA","AZ")]),
    "LV": ("Las Vegas", "LAS", "NV", 36.0840, -115.1537,
           "America/Los_Angeles", [("HND","NV"),("VGT","NV"),("BVU","NV")]),
    "SEA": ("Seattle", "SEA", "WA", 47.4502, -122.3088,
            "America/Los_Angeles", [("BFI","WA"),("RNT","WA"),("PAE","WA")]),
    "SFO": ("San Francisco", "SFO", "CA", 37.6213, -122.3790,
            "America/Los_Angeles", [("OAK","CA"),("SQL","CA"),("SJC","CA")]),
    "SAN": ("San Diego", "SAN", "CA", 32.7338, -117.1933,
            "America/Los_Angeles", [("MYF","CA"),("SEE","CA"),("CRQ","CA")]),
    "SAT": ("San Antonio", "SAT", "TX", 29.5337, -98.4698,
            "America/Chicago", [("SSF","TX"),("SKF","TX"),("BAZ","TX")]),
    "BOS": ("Boston", "BOS", "MA", 42.3656, -71.0096,
            "America/New_York", [("BED","MA"),("OWD","MA"),("LWM","MA")]),
    "DC": ("Washington DC", "DCA", "VA", 38.8512, -77.0402,
           "America/New_York", [("IAD","VA"),("BWI","MD"),("HEF","VA")]),
    "ATL": ("Atlanta", "ATL", "GA", 33.6407, -84.4277,
            "America/New_York", [("PDK","GA"),("FTY","GA"),("AHN","GA")]),
    "MIN": ("Minneapolis", "MSP", "MN", 44.8848, -93.2223,
            "America/Chicago", [("STP","MN"),("FCM","MN"),("ANE","MN")]),
    "NOLA": ("New Orleans", "MSY", "LA", 29.9934, -90.2580,
             "America/Chicago", [("NEW","LA"),("ASD","LA"),("BTR","LA")]),
    "OKC": ("Oklahoma City", "OKC", "OK", 35.3931, -97.6007,
            "America/Chicago", [("PWA","OK"),("OUN","OK"),("GOK","OK")]),
    "SDF": ("Louisville", "SDF", "KY", 38.1744, -85.7360,
            "America/New_York", [("LOU","KY"),("JVY","IN"),("FTK","KY")]),
    "EWR": ("Newark", "EWR", "NJ", 40.6895, -74.1745,
            "America/New_York", [("TEB","NJ"),("LGA","NY"),("CDW","NJ")]),
    "TTN": ("Trenton", "TTN", "NJ", 40.2767, -74.8135,
            "America/New_York", [("PNE","PA"),("SMQ","NJ"),("EWR","NJ")]),
}


def select_city(value):
    if value.upper() == "REMAINING":
        return [
            code for code in CITIES
            if code not in ("NYC", "CHI")
        ]

    if value.upper() == "ALL":
        return list(CITIES)

    key = value.upper().replace(" ", "").replace("-", "")

    for code, config in CITIES.items():
        if key == code or key == config[0].upper().replace(" ", ""):
            return [code]

    raise SystemExit(f"Unknown city: {value}")


def request(url, params):
    """Retry network failures, invalid JSON, and transient HTTP errors."""

    max_attempts = 8

    for attempt in range(max_attempts):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(60, 120),
            )

            if response.status_code in (
                429, 500, 502, 503, 504
            ):
                response.raise_for_status()

            # A successful HTTP response does not guarantee
            # that Open-Meteo returned valid forecast data.
            if url == GFS and response.status_code == 200:

                try:
                    payload = response.json()

                except ValueError as exc:
                    preview = (
                        response.text[:160]
                        .replace("\n", " ")
                    )

                    raise ValueError(
                        "Invalid GFS JSON response: "
                        f"bytes={len(response.content)}, "
                        f"content_type={response.headers.get('Content-Type')}, "
                        f"preview={preview!r}"
                    ) from exc

                expected = len(
                    params["latitude"].split(",")
                )

                values = (
                    payload
                    if isinstance(payload, list)
                    else [payload]
                )

                if (
                    len(values) != expected
                    or any(
                        not isinstance(item, dict)
                        or not isinstance(item.get("hourly"), dict)
                        or "time" not in item["hourly"]
                        or "temperature_2m" not in item["hourly"]
                        for item in values
                    )
                ):
                    raise ValueError(
                        "GFS returned unexpected forecast structure "
                        f"for {expected} requested locations"
                    )

            return response

        except (requests.RequestException, ValueError) as exc:

            attempt_number = attempt + 1

            print(
                f"Request attempt "
                f"{attempt_number}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            if attempt_number == max_attempts:
                raise

            delay = min(
                5 * attempt_number,
                40,
            )

            print(
                f"Retrying in {delay} seconds...",
                flush=True,
            )

            time.sleep(delay)

def station_params(code, state, year):
    if year < 2026:
        end = (year + 1, 1, 1)
    else:
        end = (2026, 11, 9)

    return [
        ("network", f"{state}_ASOS"),
        ("station", code),
        ("data", "all"),
        ("year1", year),
        ("month1", 1),
        ("day1", 1),
        ("year2", end[0]),
        ("month2", end[1]),
        ("day2", end[2]),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "no"),
        ("elev", "no"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "no"),
        ("report_type", 3),
        ("report_type", 4),
    ]


def collect_stations(cities):
    stations = {}

    for city in cities:
        config = CITIES[city]
        stations[config[1]] = config[2]

        for neighbor, state in config[6]:
            if (
                neighbor in stations
                and stations[neighbor] != state
            ):
                raise RuntimeError(
                    f"Conflicting state for {neighbor}"
                )
            stations[neighbor] = state

    print(
        f"Collecting {len(stations)} unique stations "
        f"for {len(cities)} cities",
        flush=True,
    )

    for code, state in sorted(stations.items()):
        for year in (2024, 2025, 2026):
            path = (
                ROOT / "stations" / code / f"{year}.csv"
            )

            if path.exists() and path.stat().st_size > 200:
                continue

            response = request(
                IEM, station_params(code, state, year)
            )

            if (
                response.status_code != 200
                or response.text.lstrip().startswith("<")
            ):
                raise RuntimeError(
                    f"IEM {code}/{year}: "
                    f"HTTP {response.status_code}"
                )

            data = pd.read_csv(
                io.StringIO(response.text),
                comment="#",
                low_memory=False,
            )

            required = {
                "station", "valid", "tmpf", "dwpf",
                "drct", "sknt", "metar"
            }

            if not required.issubset(data.columns):
                raise RuntimeError(
                    f"Unexpected columns for {code}: "
                    f"{list(data.columns)}"
                )

            if len(data) < 1000:
                raise RuntimeError(
                    f"Sparse station archive: "
                    f"{code} {year} ({len(data)} reports)"
                )

            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            temporary = path.with_suffix(".part")
            temporary.write_text(response.text)
            temporary.replace(path)

            print(
                f"{code} {year}: {len(data)} reports",
                flush=True,
            )

            time.sleep(1)

    print("\nStation collection complete.")


def gfs_params(run, cities):
    return {
        "latitude": ",".join(
            str(CITIES[c][3]) for c in cities
        ),
        "longitude": ",".join(
            str(CITIES[c][4]) for c in cities
        ),
        "models": "gfs_global",
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": 2,
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
        ]),
    }


def collect_gfs(cities, start, end, batch_size, max_new):
    start = pd.Timestamp(start, tz="UTC")
    end = (
        pd.Timestamp(end, tz="UTC")
        + pd.Timedelta(hours=18)
    )

    batches = [
        cities[i:i + batch_size]
        for i in range(0, len(cities), batch_size)
    ]

    downloaded = 0

    for run in pd.date_range(start, end, freq="6h"):
        for batch in batches:
            fingerprint = hashlib.sha256(
                ",".join(batch).encode()
            ).hexdigest()[:12]

            path = (
                ROOT / "raw_gfs_batches"
                / fingerprint
                / f"{run:%Y%m%dT%HZ}.json"
            )

            if path.exists():
                continue

            if max_new and downloaded >= max_new:
                print(
                    f"Stopped after {downloaded} batches."
                )
                return

            response = request(
                GFS, gfs_params(run, batch)
            )

            if (
                response.status_code == 400
                and "not available"
                in response.text.lower()
            ):
                print(
                    f"GFS unavailable: {run}",
                    flush=True,
                )
                continue

            if response.status_code != 200:
                raise RuntimeError(
                    f"GFS {run}: HTTP "
                    f"{response.status_code} "
                    f"{response.text[:200]}"
                )

            payload = response.json()
            values = (
                payload if isinstance(payload, list)
                else [payload]
            )

            if (
                len(values) != len(batch)
                or any("hourly" not in p for p in values)
            ):
                raise RuntimeError(
                    f"Unexpected GFS response for {batch}"
                )

            document = {
                "cities": batch,
                "run_init_utc": run.isoformat(),
                "data": values,
            }

            path.parent.mkdir(
                parents=True, exist_ok=True
            )

            temporary = path.with_suffix(".part")
            temporary.write_text(json.dumps(document))
            temporary.replace(path)

            downloaded += 1

            if downloaded % 10 == 0:
                print(
                    f"Cached {downloaded} GFS batches "
                    f"through {run}",
                    flush=True,
                )

            time.sleep(0.8)

    print(
        f"GFS collection complete: "
        f"{downloaded} new batches."
    )


def normalize_gfs(cities):
    records = {c: [] for c in cities}

    folder = ROOT / "raw_gfs_batches"

    for path in sorted(folder.glob("*/*.json")):
        document = json.loads(path.read_text())

        for city, payload in zip(
            document["cities"], document["data"]
        ):
            if city not in records:
                continue

            hourly = payload["hourly"]

            mapping = {
                "gfs_temp_f": "temperature_2m",
                "gfs_rh_pct": "relative_humidity_2m",
                "gfs_cloud_pct": "cloud_cover",
                "gfs_wind_mph": "wind_speed_10m",
                "gfs_wind_dir_deg": "wind_direction_10m",
                "gfs_shortwave_wm2": "shortwave_radiation",
            }

            for i, valid in enumerate(hourly["time"]):
                row = {
                    "run_init_utc":
                        document["run_init_utc"],
                    "valid_utc":
                        pd.to_datetime(
                            valid, utc=True
                        ).isoformat(),
                }

                for column, source in mapping.items():
                    row[column] = hourly[source][i]

                records[city].append(row)

    for city, rows in records.items():
        if not rows:
            print(f"{city}: no GFS data")
            continue

        df = pd.DataFrame(rows)

        df = df.drop_duplicates(
            ["run_init_utc", "valid_utc"]
        ).sort_values(
            ["run_init_utc", "valid_utc"]
        )

        path = (
            ROOT / "gfs" / city
            / "gfs_hourly_runs.csv"
        )

        path.parent.mkdir(
            parents=True, exist_ok=True
        )
        df.to_csv(path, index=False)

        print(
            f"{city}: "
            f"{df['run_init_utc'].nunique()} runs, "
            f"{len(df)} hourly forecasts"
        )


def audit(cities):
    source = Path(
        "src/temp-prediction/train_knyc_v7_history.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v7_multicity_audit", source
    )
    v7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v7)

    for city in cities:
        name, station, _, _, _, tz, _ = CITIES[city]
        frames = []

        for year in (2024, 2025, 2026):
            path = (
                ROOT / "stations" / station
                / f"{year}.csv"
            )

            if not path.exists():
                print(f"{city}: missing {path}")
                break

            frames.append(
                pd.read_csv(
                    path,
                    usecols=["valid", "metar"],
                    low_memory=False,
                )
            )
        else:
            df = pd.concat(
                frames, ignore_index=True
            )

            df["observed"] = pd.to_datetime(
                df["valid"],
                utc=True,
                format="mixed",
                errors="coerce",
            )
            df["precise"] = (
                df["metar"].map(v7.precise_f)
            )
            df = df.dropna(
                subset=["observed", "precise"]
            )

            recent = df[
                df["observed"].dt.year == 2026
            ]

            counts = (
                recent.groupby(
                    recent["observed"].dt.minute
                ).size().sort_values(ascending=False)
            )

            if counts.empty:
                print(
                    f"{city}: no precise observations"
                )
                continue

            minute = int(counts.index[0])

            selected = df[
                df["observed"].dt.minute == minute
            ]

            print(
                f"{city:5s} "
                f"{name:17s} "
                f"K{station:3s} "
                f"routine~:{minute:02d} "
                f"precise={len(selected):6d} "
                f"2026_reports={int(counts.iloc[0])}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=[
            "list",
            "stations",
            "gfs",
            "normalize-gfs",
            "audit",
        ],
    )
    parser.add_argument("--city", default="ALL")
    parser.add_argument(
        "--start", default="2026-07-01"
    )
    parser.add_argument(
        "--end", default="2026-11-08"
    )
    parser.add_argument(
        "--batch-size", type=int, default=6
    )
    parser.add_argument(
        "--max-new", type=int, default=0
    )

    args = parser.parse_args()
    cities = select_city(args.city)

    if args.action == "list":
        for code in cities:
            config = CITIES[code]
            print(
                code,
                config[0],
                "K" + config[1],
                config[5],
            )

    elif args.action == "stations":
        collect_stations(cities)

    elif args.action == "gfs":
        if not 1 <= args.batch_size <= 12:
            parser.error(
                "--batch-size must be between 1 and 12"
            )

        collect_gfs(
            cities,
            args.start,
            args.end,
            args.batch_size,
            args.max_new,
        )

    elif args.action == "normalize-gfs":
        normalize_gfs(cities)

    elif args.action == "audit":
        audit(cities)


if __name__ == "__main__":
    main()
