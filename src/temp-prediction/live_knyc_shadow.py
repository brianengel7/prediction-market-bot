
#!/usr/bin/env python3
"""Live KNYC V3/V5 shadow evaluator and forecast-provider research.

Reads saved V3/V5 models. Records actual first-seen weather reports.
Tracks NWS, Open-Meteo, and optional Weather Company forecasts.
Research only: no Kalshi access, no trading, no model retraining.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests


ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)
OUT = ROOT / "live_shadow_20261029"
ET = ZoneInfo("America/New_York")

STATIONS = ("NYC", "LGA", "EWR", "TEB")

COVER = {
    "CLR": 0.0,
    "SKC": 0.0,
    "FEW": 0.125,
    "SCT": 0.375,
    "BKN": 0.75,
    "OVC": 1.0,
    "VV": 1.0,
}

T_RE = re.compile(
    r"(?<!\S)T([01])(\d{3})([01])(\d{3})(?=\s|$)"
)

SKY_RE = re.compile(
    r"(?<!\S)(CLR|SKC|FEW|SCT|BKN|OVC|VV)"
    r"(\d{3})?(?=\s|$)"
)


def now_utc():
    return datetime.now(timezone.utc)


def dt(value):
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(
            float(value), timezone.utc
        )

    return pd.to_datetime(
        value, utc=True
    ).to_pydatetime()


def iso(value):
    return value.astimezone(
        timezone.utc
    ).isoformat()


def number(value):
    try:
        result = float(value)
        return (
            result if math.isfinite(result)
            else math.nan
        )
    except (ValueError, TypeError):
        return math.nan


def write_csv(path, row):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(row)
        )

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(row)


def write_jsonl(path, row):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with path.open(
        "a", encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(
                row, allow_nan=False, default=str
            ) + "\n"
        )


def saved_keys(path, cols):
    if not path.exists():
        return set()

    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        return {
            tuple(row[c] for c in cols)
            for row in csv.DictReader(file)
        }


def parse_metar(item, received):
    station = str(
        item.get("icaoId", "")
    ).upper()

    if station not in {
        "K" + s for s in STATIONS
    }:
        return None

    observed = dt(item.get("obsTime"))

    if (
        observed is None
        or observed > received + timedelta(minutes=5)
    ):
        return None

    temp_c = number(item.get("temp"))

    if not math.isfinite(temp_c):
        return None

    raw = str(item.get("rawOb") or "")

    # KNYC precise temperature from METAR T-group.
    match = T_RE.search(raw)
    precise = math.nan

    if match:
        precise = (
            32
            + 1.8
            * int(match[2]) / 10
            * (-1 if match[1] == "1" else 1)
        )

    clouds = []

    for layer in item.get("clouds") or []:
        if isinstance(layer, dict):
            code = str(
                layer.get("cover", "")
            ).upper()

            if code in COVER:
                clouds.append((
                    code,
                    number(layer.get("base")),
                ))

    if not clouds:
        clouds = [
            (
                code,
                float(alt) * 100
                if alt else math.nan
            )
            for code, alt in SKY_RE.findall(raw)
        ]

    sky = max(
        (COVER[code] for code, _ in clouds),
        default=math.nan,
    )

    ceilings = [
        height
        for code, height in clouds
        if code in {"BKN", "OVC", "VV"}
        and math.isfinite(height)
    ]

    dew_c = number(item.get("dewp"))

    return {
        "station": station[1:],
        "observed_utc": iso(observed),
        "first_seen_utc": iso(received),

        # Maintain the whole-F input convention
        # used during historical V3/V5 training.
        "temp": float(np.rint(
            temp_c * 1.8 + 32
        )),

        "precise_f": (
            precise
            if math.isfinite(precise)
            else None
        ),

        "dew": (
            dew_c * 1.8 + 32
            if math.isfinite(dew_c)
            else None
        ),

        "wind": number(item.get("wspd")),
        "direction": number(item.get("wdir")),

        "cloud_fraction": (
            sky if math.isfinite(sky)
            else None
        ),

        "log_ceiling_ft": (
            math.log1p(min(ceilings))
            if ceilings else None
        ),

        "raw_metar": raw,
    }


def station_features(
    history, station, decision, lag
):
    names = [
        "temp", "dew", "wind",
        "direction", "prev_temp",
        "trend1", "trend2",
        "acceleration", "wind_change",
        "wind_u", "wind_v",
        "wind_u_change", "wind_v_change",
        "relative_humidity",
        "dew_depression", "age_min",
    ]

    out = {
        station + "_" + name: math.nan
        for name in names
    }

    out.update({
        "C_" + station + "_" + name: math.nan
        for name in (
            "fraction",
            "log_ceiling_ft",
            "age_min",
        )
    })

    # Require actual live reception before decision.
    # Also apply the model's simulated reporting lag.
    rows = sorted(
        (
            r for r in history.values()
            if r["station"] == station
            and dt(r["first_seen_utc"]) <= decision
            and (
                dt(r["observed_utc"])
                + timedelta(minutes=lag)
                <= decision
            )
        ),
        key=lambda r: r["observed_utc"],
    )

    if not rows:
        return out, None

    cur = rows[-1]

    age = (
        decision - dt(cur["observed_utc"])
    ).total_seconds() / 60

    if age > 120:
        return out, None

    prev = rows[-2] if len(rows) > 1 else None
    prev2 = rows[-3] if len(rows) > 2 else None

    def v(record, col):
        return (
            number(record.get(col))
            if record else math.nan
        )

    t, dew, w, direction = [
        v(cur, c)
        for c in (
            "temp", "dew",
            "wind", "direction"
        )
    ]

    pt, pw, pd = [
        v(prev, c)
        for c in (
            "temp", "wind", "direction"
        )
    ]

    p2t = v(prev2, "temp")

    hours1 = (
        (
            dt(cur["observed_utc"])
            - dt(prev["observed_utc"])
        ).total_seconds() / 3600
        if prev else math.nan
    )

    hours2 = (
        (
            dt(cur["observed_utc"])
            - dt(prev2["observed_utc"])
        ).total_seconds() / 3600
        if prev2 else math.nan
    )

    tr1 = (
        (t - pt) / hours1
        if 0.1 <= hours1 <= 2.5
        else math.nan
    )

    tr2 = (
        (t - p2t) / hours2
        if 0.3 <= hours2 <= 4.5
        else math.nan
    )

    direction_rad = math.radians(direction)
    previous_rad = math.radians(pd)

    u = -w * math.sin(direction_rad)
    vv = -w * math.cos(direction_rad)

    uc = u + pw * math.sin(previous_rad)
    vc = vv + pw * math.cos(previous_rad)

    tc = (t - 32) * 5 / 9
    dc = (dew - 32) * 5 / 9

    rh = (
        100 * math.exp(
            17.625 * dc / (243.04 + dc)
            - 17.625 * tc / (243.04 + tc)
        )
        if math.isfinite(dc)
        else math.nan
    )

    details = {
        "temp": t,
        "dew": dew,
        "wind": w,
        "direction": direction,
        "prev_temp": pt,
        "trend1": tr1,
        "trend2": tr2,
        "acceleration": tr1 - tr2,
        "wind_change": w - pw,
        "wind_u": u,
        "wind_v": vv,
        "wind_u_change": uc,
        "wind_v_change": vc,
        "relative_humidity": (
            min(100, max(0, rh))
            if math.isfinite(rh)
            else math.nan
        ),
        "dew_depression": t - dew,
        "age_min": age,
    }

    out.update({
        station + "_" + k: value
        for k, value in details.items()
    })

    out["C_" + station + "_fraction"] = v(
        cur, "cloud_fraction"
    )

    out["C_" + station + "_log_ceiling_ft"] = v(
        cur, "log_ceiling_ft"
    )

    out["C_" + station + "_age_min"] = age

    return out, cur


def sun(when):
    doy = when.timetuple().tm_yday

    hours = (
        when.hour
        + when.minute / 60
        + when.second / 3600
    )

    gamma = (
        2 * math.pi / 365
        * (doy - 1 + (hours - 12) / 24)
    )

    eq = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    ha = math.radians(
        (
            (hours * 60 + eq + 4 * -73.969)
            % 1440
        ) / 4 - 180
    )

    lat = math.radians(40.779)

    sine = (
        math.sin(lat) * math.sin(decl)
        + math.cos(lat)
        * math.cos(decl)
        * math.cos(ha)
    )

    return {
        "sun_elevation_deg": math.degrees(
            math.asin(
                max(-1, min(1, sine))
            )
        ),
        "sun_heating_proxy": max(0, sine),
    }


def input_features(
    history, at, horizon,
    nyc_lag, airport_lag
):
    local = at.astimezone(ET)

    hour = local.hour + local.minute / 60
    day = local.timetuple().tm_yday

    feat = {
        "horizon": horizon,
        "hour_sin": math.sin(
            2 * math.pi * hour / 24
        ),
        "hour_cos": math.cos(
            2 * math.pi * hour / 24
        ),
        "day_sin": math.sin(
            2 * math.pi * day / 365.25
        ),
        "day_cos": math.cos(
            2 * math.pi * day / 365.25
        ),
    }

    latest = {}

    for station in STATIONS:
        f, report = station_features(
            history,
            station,
            at,
            nyc_lag
            if station == "NYC"
            else airport_lag,
        )

        feat.update(f)
        latest[station] = report

    for station in STATIONS[1:]:
        feat[station + "_minus_nyc"] = (
            feat[station + "_temp"]
            - feat["NYC_temp"]
        )

    feat.update(sun(at))

    return feat, latest


def load_models():
    specs = []

    for nyc_lag, airport_lag in (
        (5, 6), (3, 4)
    ):
        tag = (
            f"knyc{nyc_lag}"
            f"_airports{airport_lag}"
        )

        v3 = joblib.load(
            ROOT / "v3_30_45"
            / tag / "research_models.joblib"
        )

        v5 = joblib.load(
            ROOT / "v5_precise_clouds"
            / tag / "research_models.joblib"
        )

        for horizon in (30, 45):
            recipe = (
                v3["selected_by_horizon"][horizon]
            )

            a = v3["trained"][
                (horizon, recipe)
            ]

            specs.append((
                tag, horizon,
                "V3_" + recipe,
                a["model"],
                a["features"],
            ))

            variants = (
                ("plus_solar_clouds",)
                if horizon == 30
                else (
                    "plus_clouds",
                    "plus_solar_clouds",
                )
            )

            for variant in variants:
                b = v5["models"][
                    f"{horizon}m_{variant}"
                ]

                specs.append((
                    tag, horizon,
                    "V5_" + variant,
                    b["model"],
                    b["features"],
                ))

    return specs


def fetch_metars(session):
    response = session.get(
        "https://aviationweather.gov/api/data/metar",
        params={
            "ids": ",".join(
                "K" + s for s in STATIONS
            ),
            "format": "json",
            "hours": 6,
        },
        timeout=20,
    )

    response.raise_for_status()
    result = response.json()

    if not isinstance(result, list):
        raise ValueError(
            "AWC response is not a JSON list"
        )

    received = now_utc()

    return [
        r
        for item in result
        if isinstance(item, dict)
        if (
            r := parse_metar(item, received)
        ) is not None
    ]


def nws_url(session):
    response = session.get(
        "https://api.weather.gov/points/"
        "40.779,-73.969",
        timeout=20,
    )

    response.raise_for_status()

    return response.json()[
        "properties"
    ]["forecastHourly"]


def fetch_provider(
    session, provider,
    nwshourly, twc_key
):
    if provider == "NWS":
        response = session.get(
            nwshourly,
            timeout=20,
            headers={
                "Accept": "application/geo+json"
            },
        )

        response.raise_for_status()
        body = response.json()

        values = []

        for period in body.get(
            "properties", {}
        ).get("periods", []):
            temperature = number(
                period.get("temperature")
            )

            if period.get(
                "temperatureUnit"
            ) == "C":
                temperature = (
                    temperature * 1.8 + 32
                )

            values.append((
                dt(period["startTime"]),
                temperature,
            ))

    elif provider == "OPEN_METEO":
        response = session.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 40.779,
                "longitude": -73.969,
                "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit",
                "timezone": "UTC",
                "forecast_days": 2,
            },
            timeout=20,
        )

        response.raise_for_status()
        body = response.json()

        hourly = body.get("hourly", {})

        values = [
            (
                dt(when + "Z"),
                number(temp),
            )
            for when, temp in zip(
                hourly.get("time", []),
                hourly.get("temperature_2m", []),
            )
        ]

    else:
        # Optional authorized Weather Company API.
        response = session.get(
            "https://api.weather.com/v3/wx/"
            "forecast/hourly/2day",
            params={
                "geocode": "40.779,-73.969",
                "units": "e",
                "language": "en-US",
                "format": "json",
                "apiKey": twc_key,
            },
            timeout=20,
        )

        response.raise_for_status()
        body = response.json()

        values = [
            (
                dt(when),
                number(temp),
            )
            for when, temp in zip(
                body.get("validTimeUtc", []),
                body.get("temperature", []),
            )
        ]

    return body, [
        (when, temp)
        for when, temp in values
        if (
            when is not None
            and math.isfinite(temp)
        )
    ]


def nearest_hourly(values, target):
    eligible = sorted([
        (
            abs(
                (when - target).total_seconds()
            ),
            when,
            temp,
        )
        for when, temp in values
        if abs(
            (when - target).total_seconds()
        ) <= 40 * 60
    ])

    if not eligible:
        return None

    _, when, temp = eligible[0]

    return temp, when


def score_new(history, out, scored):
    path = out / "predictions.csv"

    if not path.exists():
        return

    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        predictions = list(
            csv.DictReader(file)
        )

    for prediction in predictions:
        key = (
            prediction["target_utc"],
            prediction["horizon"],
            prediction["variant"],
            prediction["config"],
        )

        if key in scored:
            continue

        actual = history.get((
            "NYC",
            prediction["target_utc"],
        ))

        if (
            actual is None
            or not math.isfinite(
                number(actual["precise_f"])
            )
        ):
            continue

        error = (
            number(prediction["prediction_f"])
            - number(actual["precise_f"])
        )

        write_csv(
            out / "scores.csv",
            {
                "target_utc": prediction["target_utc"],
                "horizon": prediction["horizon"],
                "variant": prediction["variant"],
                "config": prediction["config"],
                "prediction_f":
                    prediction["prediction_f"],
                "actual_precise_f":
                    actual["precise_f"],
                "error_f": round(error, 5),
                "absolute_error_f":
                    round(abs(error), 5),
                "target_first_seen_utc":
                    actual["first_seen_utc"],
            },
        )

        scored.add(key)

        print(
            "SCORED",
            prediction["variant"],
            prediction["config"],
            "error=",
            round(error, 3),
            "F",
            flush=True,
        )


def summary(out):
    path = out / "scores.csv"

    if not path.exists():
        print(
            "No verified targets scored yet."
        )
        return

    df = pd.read_csv(path)

    df["absolute_error_f"] = pd.to_numeric(
        df["absolute_error_f"]
    )

    grouped = df.groupby([
        "horizon",
        "variant",
        "config",
    ]).agg(
        n=("absolute_error_f", "size"),
        mae=("absolute_error_f", "mean"),
        within_half=(
            "absolute_error_f",
            lambda x: 100 * (x <= 0.5).mean()
        ),
        within_one=(
            "absolute_error_f",
            lambda x: 100 * (x <= 1).mean()
        ),
        gt2=(
            "absolute_error_f",
            lambda x: int((x > 2).sum())
        ),
    )

    print(
        "\n=== LIVE PRECISE-TEMPERATURE SCORE ==="
    )

    print(
        grouped.round(3).to_string()
    )

    for key, part in df.groupby([
        "horizon",
        "variant",
        "config",
    ]):
        correlation = (
            part["prediction_f"].corr(
                part["actual_precise_f"]
            )
            if len(part) > 2
            else math.nan
        )

        bias = (
            part["prediction_f"]
            - part["actual_precise_f"]
        ).mean()

        print(
            key,
            "correlation=",
            round(correlation, 3),
            "bias=",
            round(bias, 3),
            "F",
        )

    print(
        "Small same-day samples do not establish "
        "statistically significant correlation."
    )


def run(args):
    out = Path(args.out)
    out.mkdir(
        parents=True, exist_ok=True
    )

    specs = load_models()

    print(
        "Loaded",
        len(specs),
        "frozen V3/V5 models",
        flush=True,
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "PolyQuant-KNYC-shadow-research/1.0"
    })

    forecasts = {}
    observations = {}

    history_file = out / "metars.jsonl"

    if history_file.exists():
        for line in history_file.read_text(
            encoding="utf-8"
        ).splitlines():
            item = json.loads(line)

            observations[(
                item["station"],
                item["observed_utc"],
            )] = item

    completed = saved_keys(
        out / "decisions.csv",
        ("target_utc", "horizon"),
    )

    predicted = saved_keys(
        out / "predictions.csv",
        (
            "target_utc",
            "horizon",
            "variant",
            "config",
        ),
    )

    scored = saved_keys(
        out / "scores.csv",
        (
            "target_utc",
            "horizon",
            "variant",
            "config",
        ),
    )

    key = os.environ.get(
        "TWC_API_KEY", ""
    ).strip()

    url = None

    end_local = now_utc().astimezone(ET).replace(
        hour=17,
        minute=15,
        second=0,
        microsecond=0,
    )

    print(
        "Shadow only; stop at",
        end_local,
        "| TWC:",
        bool(key),
        flush=True,
    )

    next_metar = now_utc()
    next_provider = now_utc()

    while now_utc() < end_local.astimezone(
        timezone.utc
    ):
        if now_utc() >= next_metar:
            try:
                fresh = fetch_metars(session)
                count = 0

                for row in fresh:
                    item = (
                        row["station"],
                        row["observed_utc"],
                    )

                    if item not in observations:
                        observations[item] = row

                        clean = {
                            k: (
                                None
                                if isinstance(v, float)
                                and not math.isfinite(v)
                                else v
                            )
                            for k, v in row.items()
                        }

                        write_jsonl(
                            out / "metars.jsonl",
                            clean,
                        )

                        count += 1

                if count:
                    print(
                        "New AWC reports:",
                        count,
                        flush=True,
                    )

            except Exception as exc:
                print(
                    "AWC ERROR:",
                    str(exc)[:130],
                    flush=True,
                )

            next_metar = (
                now_utc()
                + timedelta(seconds=45)
            )

        if now_utc() >= next_provider:
            provider_names = [
                "NWS",
                "OPEN_METEO",
            ]

            if key:
                provider_names.append("TWC")

            for name in provider_names:
                try:
                    if name == "NWS" and url is None:
                        url = nws_url(session)

                    body, values = fetch_provider(
                        session, name, url, key
                    )

                    snap = now_utc()

                    forecasts[name] = (
                        snap, values
                    )

                    path = (
                        out / "provider_raw"
                        / f"{name}_"
                        f"{snap:%Y%m%dT%H%M%SZ}.json"
                    )

                    path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    path.write_text(
                        json.dumps(body),
                        encoding="utf-8",
                    )

                    print(
                        name,
                        "snapshot:",
                        len(values),
                        "hourly slots",
                        flush=True,
                    )

                except Exception as exc:
                    print(
                        name,
                        "ERROR:",
                        (
                            str(exc)[:130]
                            if name != "TWC"
                            else "Check API entitlement/key"
                        ),
                        flush=True,
                    )

                    if name == "NWS":
                        url = None

            next_provider = (
                now_utc()
                + timedelta(minutes=10)
            )

        at = now_utc()
        local = at.astimezone(ET)

        if (
            8 <= local.hour <= 16
            and local.minute in (6, 21)
        ):
            horizon = (
                45 if local.minute == 6
                else 30
            )

            target = local.replace(
                minute=51,
                second=0,
                microsecond=0,
            ).astimezone(timezone.utc)

            decision_key = (
                iso(target),
                str(horizon),
            )

            if decision_key not in completed:
                write_csv(
                    out / "decisions.csv",
                    {
                        "target_utc": iso(target),
                        "horizon": horizon,
                        "forecast_at_utc": iso(at),
                    },
                )

                completed.add(decision_key)

                print(
                    "DECISION",
                    local.strftime("%H:%M:%S"),
                    horizon,
                    "min ahead",
                    flush=True,
                )

                # Run saved V3 and V5 models.
                for (
                    tag, h, variant,
                    model, columns
                ) in specs:
                    if h != horizon:
                        continue

                    nl, al = (
                        (5, 6)
                        if tag == "knyc5_airports6"
                        else (3, 4)
                    )

                    try:
                        feat, used = input_features(
                            observations,
                            at,
                            horizon,
                            nl,
                            al,
                        )

                        if (
                            not math.isfinite(
                                feat["NYC_temp"]
                            )
                            or not any(
                                math.isfinite(
                                    feat[s + "_temp"]
                                )
                                for s in STATIONS[1:]
                            )
                        ):
                            print(
                                " SKIPPED",
                                tag,
                                variant,
                                "missing KNYC/airport report",
                                flush=True,
                            )
                            continue

                        missing = [
                            column
                            for column in columns
                            if column not in feat
                        ]

                        if missing:
                            raise ValueError(
                                "Missing features "
                                + str(missing)
                            )

                        data = pd.DataFrame([{
                            c: feat[c]
                            for c in columns
                        }])

                        prediction = (
                            feat["NYC_temp"]
                            + float(
                                model.predict(data)[0]
                            )
                        )

                        row = {
                            "target_utc": iso(target),
                            "horizon": horizon,
                            "variant": variant,
                            "config": tag,
                            "forecast_at_utc": iso(at),
                            "prediction_f":
                                round(prediction, 5),
                            "valid_utc": "",
                            "knyc_used_utc": (
                                used["NYC"]["observed_utc"]
                                if used["NYC"]
                                else ""
                            ),
                        }

                        identifier = (
                            row["target_utc"],
                            str(horizon),
                            variant,
                            tag,
                        )

                        if identifier not in predicted:
                            write_csv(
                                out / "predictions.csv",
                                row,
                            )
                            predicted.add(identifier)

                        print(
                            " ",
                            tag,
                            variant,
                            round(prediction, 3),
                            "F",
                            flush=True,
                        )

                    except Exception as exc:
                        print(
                            " MODEL ERROR",
                            tag,
                            variant,
                            str(exc)[:150],
                            flush=True,
                        )

                # Record forecasts issued by providers
                # before the current decision.
                for name, (
                    snap, values
                ) in forecasts.items():
                    if snap > at:
                        continue

                    projected = nearest_hourly(
                        values, target
                    )

                    if projected is None:
                        continue

                    value, valid = projected

                    row = {
                        "target_utc": iso(target),
                        "horizon": horizon,
                        "variant": name,
                        "config": "provider_hourly",
                        "forecast_at_utc": iso(at),
                        "prediction_f":
                            round(value, 5),
                        "valid_utc": iso(valid),
                        "knyc_used_utc": "",
                    }

                    identifier = (
                        row["target_utc"],
                        str(horizon),
                        name,
                        "provider_hourly",
                    )

                    if identifier not in predicted:
                        write_csv(
                            out / "predictions.csv",
                            row,
                        )
                        predicted.add(identifier)

                    print(
                        " ",
                        name,
                        round(value, 2),
                        "F",
                        flush=True,
                    )

        score_new(
            observations, out, scored
        )

        time.sleep(2)

    print(
        "Session ended.",
        flush=True,
    )

    summary(out)


def preflight():
    print(
        "Frozen models:",
        len(load_models())
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "PolyQuant-KNYC-shadow-research/1.0"
    })

    reports = fetch_metars(session)

    stations = sorted({
        r["station"] for r in reports
    })

    print(
        "AWC stations:",
        stations,
    )

    if len(stations) < 4:
        raise RuntimeError(
            "Not all four AWC stations returned data"
        )

    url = nws_url(session)

    print(
        "NWS hourly slots:",
        len(fetch_provider(
            session, "NWS", url, ""
        )[1]),
    )

    print(
        "Open-Meteo hourly slots:",
        len(fetch_provider(
            session, "OPEN_METEO", url, ""
        )[1]),
    )

    if os.environ.get("TWC_API_KEY"):
        print(
            "TWC hourly slots:",
            len(fetch_provider(
                session,
                "TWC",
                url,
                os.environ["TWC_API_KEY"],
            )[1]),
        )

    print("PREFLIGHT PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default=str(OUT),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
    )

    args = parser.parse_args()

    if args.summary:
        summary(Path(args.out))
    elif args.preflight:
        preflight()
    else:
        run(args)
