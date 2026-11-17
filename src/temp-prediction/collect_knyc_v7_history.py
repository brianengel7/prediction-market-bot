
#!/usr/bin/env python3
"""Extend KNYC/LGA/EWR/TEB history from January 2024.

Downloads 2024-01 through 2026-06, then reuses
the exact July-October 2026 observations from V3.

Research only. Does not modify existing models.
"""

import argparse
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(
    "data/research/nyc_observation_20260701_20261026"
)
OUT = ROOT / "v7_extended_history"
CACHE = OUT / "monthly_cache"

API = (
    "https://mesonet.agron.iastate.edu/"
    "cgi-bin/request/asos.py"
)

NETWORKS = {
    "NYC": "NY_ASOS",
    "LGA": "NY_ASOS",
    "EWR": "NJ_ASOS",
    "TEB": "NJ_ASOS",
}

FIELDS = [
    "station", "valid", "tmpf",
    "dwpf", "drct", "sknt", "metar",
]

T_GROUP = re.compile(
    r"(?<!\S)T([01])(\d{3})([01])(\d{3})(?=\s|$)"
)


def precise_temp(raw):
    match = (
        T_GROUP.search(raw)
        if isinstance(raw, str)
        else None
    )

    if not match:
        return float("nan")

    sign = -1 if match.group(1) == "1" else 1

    return (
        32
        + sign * int(match.group(2)) / 10 * 1.8
    )


def clean(df, station):
    # Historical airport CSVs may not contain raw
    # METAR text. It is required only for KNYC.
    if station != "NYC" and "metar" not in df.columns:
        df = df.copy()
        df["metar"] = pd.NA

    missing = set(FIELDS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{station}: missing columns {missing}"
        )

    df = df[FIELDS].copy()

    df["station"] = (
        df["station"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["station"].isin(
            [station, "K" + station]
        )
    ].copy()

    df["station"] = station

    df["valid"] = pd.to_datetime(
        df["valid"], utc=True, errors="coerce"
    )

    df["tmpf"] = pd.to_numeric(
        df["tmpf"], errors="coerce"
    )

    df = df.dropna(
        subset=["valid", "tmpf"]
    )

    conflicts = (
        df.groupby("valid")["tmpf"].nunique()
    )

    bad = conflicts[conflicts > 1].index

    df = df[~df["valid"].isin(bad)]

    return (
        df.sort_values("valid")
        .drop_duplicates("valid")
        .reset_index(drop=True)
    )


def download(session, station, year, month):
    next_year = year + (month == 12)
    next_month = month % 12 + 1

    params = [
        ("station", station),
        ("network", NETWORKS[station]),
        ("data", "tmpf"),
        ("data", "dwpf"),
        ("data", "drct"),
        ("data", "sknt"),
        ("data", "metar"),
        ("report_type", "3"),
        ("report_type", "4"),
        ("year1", str(year)),
        ("month1", str(month)),
        ("day1", "1"),
        ("year2", str(next_year)),
        ("month2", str(next_month)),
        ("day2", "1"),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("missing", "M"),
        ("latlon", "no"),
        ("elev", "no"),
    ]

    error = None

    for attempt in range(3):
        try:
            response = session.get(
                API, params=params,
                timeout=(15, 100)
            )
            response.raise_for_status()

            if "station" not in response.text[:300]:
                raise ValueError(
                    "Unexpected IEM response: "
                    + response.text[:120]
                )

            df = pd.read_csv(
                StringIO(response.text),
                comment="#",
                low_memory=False,
            )

            df = clean(df, station)

            if df.empty:
                raise ValueError(
                    "No valid temperature reports"
                )

            return df

        except (
            requests.RequestException,
            ValueError,
            pd.errors.ParserError,
        ) as exc:
            error = exc

            if attempt < 2:
                time.sleep(3 * 2 ** attempt)

    raise RuntimeError(
        f"{station} {year}-{month:02d}: {error}"
    )


def months():
    for year in (2024, 2025, 2026):
        stop = 13 if year < 2026 else 7

        for month in range(1, stop):
            yield year, month


def existing_path(station):
    if station == "NYC":
        return (
            ROOT / "knyc_high_frequency_observations.csv"
        )

    return (
        ROOT
        / "live_feasible_research"
        / "airport_metars"
        / f"{station}_20260701_20261026.csv"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download January 2024 only",
    )

    args = parser.parse_args()

    CACHE.mkdir(
        parents=True, exist_ok=True
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "PolyQuant-historical-METAR-study/1.0"
    })

    requested = (
        [(2024, 1)]
        if args.sample
        else list(months())
    )

    failures = []
    summary = []

    for station in NETWORKS:
        pieces = []

        for index, (year, month) in enumerate(
            requested, 1
        ):
            cache_file = (
                CACHE / f"{station}_{year}{month:02d}.csv"
            )

            try:
                if cache_file.exists():
                    part = clean(
                        pd.read_csv(
                            cache_file, low_memory=False
                        ),
                        station,
                    )
                    status = "cached"

                else:
                    part = download(
                        session, station, year, month
                    )

                    part.to_csv(
                        cache_file, index=False
                    )

                    status = "downloaded"
                    time.sleep(0.2)

                pieces.append(part)

                print(
                    f"{station} "
                    f"[{index}/{len(requested)}] "
                    f"{year}-{month:02d}: "
                    f"{len(part)} reports ({status})",
                    flush=True,
                )

            except Exception as exc:
                failures.append(
                    (station, year, month, str(exc)[:200])
                )

                print(
                    f"FAILED {station} "
                    f"{year}-{month:02d}: {exc}",
                    flush=True,
                )

        if args.sample:
            continue

        if len(pieces) != len(requested):
            print(
                f"INCOMPLETE {station}: "
                f"{len(pieces)}/{len(requested)} months",
                flush=True,
            )
            continue

        original = existing_path(station)

        if not original.exists():
            raise FileNotFoundError(original)

        older = pd.concat(
            pieces, ignore_index=True
        )

        recent = clean(
            pd.read_csv(
                original, low_memory=False
            ),
            station,
        )

        cutoff = pd.Timestamp(
            "2026-07-01T00:00:00Z"
        )

        older = older[
            older["valid"] < cutoff
        ]

        recent = recent[
            recent["valid"] >= cutoff
        ]

        combined = clean(
            pd.concat(
                [older, recent],
                ignore_index=True,
            ),
            station,
        )

        output = (
            OUT
            / f"{station}_20240101_20261026.csv"
        )

        combined.to_csv(
            output, index=False
        )

        targets = (
            combined[
                combined["valid"].dt.minute == 51
            ]
            if station == "NYC"
            else combined.iloc[0:0]
        )

        summary.append({
            "station": station,
            "reports": len(combined),
            "days": (
                combined["valid"].dt.date.nunique()
            ),
            "knyc_51_reports": len(targets),
            "knyc_51_precise": int(
                targets["metar"]
                .map(precise_temp)
                .notna().sum()
            ),
        })

        print(
            "SAVED:", output,
            flush=True,
        )

    failures_path = (
        OUT / "download_failures.csv"
    )

    pd.DataFrame(
        failures,
        columns=[
            "station", "year", "month", "error"
        ],
    ).to_csv(
        failures_path, index=False
    )

    print("\n=== V7 HISTORY SUMMARY ===")

    if summary:
        print(
            pd.DataFrame(summary)
            .to_string(index=False)
        )

    print("Failed months:", len(failures))
    print("Failure log:", failures_path)

    if failures:
        print(
            "Resolve failed months before training. "
            "Successful monthly downloads are cached."
        )


if __name__ == "__main__":
    main()
