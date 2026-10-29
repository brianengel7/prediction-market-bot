
#!/usr/bin/env python3
"""Read-only NYC weather observation / Kalshi event study.

Uses:
  - Existing weather_observation_history database table
  - Previously collected Kalshi one-minute candle CSVs

Does not call APIs, place orders, or modify database records.

Publication timestamps are ESTIMATED, not historically verified.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


NY = ZoneInfo("America/New_York")

DEFAULT_BASE = Path(
    "data/research/nyc_observation_20260701_20261026"
)

EVENT_COLS = [
    "event_id",
    "local_date",
    "observed_at_utc",
    "observed_at_ny",
    "temperature_f",
    "previous_temperature_f",
    "temperature_change_f",
    "previous_observed_high_f",
    "new_observed_high",
    "report_type",
    "raw_metar",
    "source",
]

REACTION_COLS = [
    "event_id",
    "local_date",
    "observed_at_utc",
    "temperature_f",
    "temperature_change_f",
    "new_observed_high",
    "report_type",
    "assumed_publication_delay_min",
    "assumed_release_at_utc",
    "assumed_entry_at_utc",
    "assumed_exit_at_utc",
    "ticker",
    "entry_candle_end_utc",
    "exit_candle_end_utc",
    "entry_quote_age_sec",
    "exit_quote_delay_sec",
    "entry_yes_bid",
    "entry_yes_ask",
    "exit_yes_bid",
    "exit_yes_ask",
    "entry_mid",
    "exit_mid",
    "mid_change",
    "yes_buy_ask_sell_bid_gross",
    "no_buy_ask_sell_bid_gross",
    "entry_spread",
    "exit_spread",
]


def load_observations(args):
    if args.observations_csv:
        result = pd.read_csv(
            args.observations_csv, dtype=str
        )
    else:
        from src.database.db import (
            get_connection,
            read_dataframe,
        )

        start_utc = datetime.combine(
            args.start, datetime.min.time(), NY
        ).astimezone(timezone.utc).isoformat()

        end_utc = datetime.combine(
            args.end + timedelta(days=1),
            datetime.min.time(),
            NY,
        ).astimezone(timezone.utc).isoformat()

        with get_connection() as connection:
            result = read_dataframe(
                """
                SELECT
                    source,
                    station,
                    observed_at_utc,
                    temperature_f,
                    raw_payload,
                    payload_hash
                FROM weather_observation_history
                WHERE source = ?
                  AND station = ?
                  AND observed_at_utc >= ?
                  AND observed_at_utc < ?
                """,
                connection,
                params=(
                    "IEM_ASOS_ARCHIVE",
                    "KNYC",
                    start_utc,
                    end_utc,
                ),
            )

    required = {
        "observed_at_utc",
        "temperature_f",
    }

    missing = required - set(result.columns)

    if missing:
        raise ValueError(
            f"Weather columns missing: {missing}"
        )

    if "station" in result.columns:
        result = result[
            result["station"]
            .astype(str)
            .str.upper()
            .isin(("KNYC", "NYC"))
        ]

    result = result.copy()

    result["observed_at_utc"] = pd.to_datetime(
        result["observed_at_utc"],
        utc=True,
        errors="coerce",
    )

    result["temperature_f"] = pd.to_numeric(
        result["temperature_f"],
        errors="coerce",
    )

    result = result[
        result["observed_at_utc"].notna()
        & result["temperature_f"].notna()
    ]

    result = result[
        result["temperature_f"].map(math.isfinite)
    ]

    if result.empty:
        raise RuntimeError(
            "No usable KNYC observations found."
        )

    result["local_date"] = (
        result["observed_at_utc"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    result = result[
        (result["local_date"] >= args.start)
        & (result["local_date"] <= args.end)
    ]

    if result.empty:
        raise RuntimeError(
            "No observations in requested date range."
        )

    return result


def raw_report(row):
    payload = row.get("raw_payload", "")

    try:
        if isinstance(payload, str) and payload.strip():
            payload = json.loads(payload)
    except (TypeError, ValueError):
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    metar = str(
        payload.get("metar") or ""
    ).strip()

    upper = metar.upper()

    if upper.startswith("SPECI "):
        report_type = "SPECI"
    elif upper.startswith("METAR "):
        report_type = "METAR"
    else:
        report_type = "UNCLASSIFIED"

    return metar, report_type


def build_events(weather):
    events = []
    conflicts = 0

    prior_date = None
    prior_temp = None
    daily_high = None

    grouped = weather.groupby(
        "observed_at_utc",
        sort=True,
    )

    for ts, versions in grouped:

        # Exclude conflicting historical revisions.
        # We don't know which version was initially public.
        if versions["temperature_f"].nunique() > 1:
            conflicts += 1
            continue

        chosen = versions.iloc[0]
        val = float(chosen["temperature_f"])

        local = (
            ts.to_pydatetime()
            .astimezone(NY)
        )

        day = local.date()

        if day != prior_date:
            prior_date = day
            prior_temp = None
            daily_high = None

        is_new_high = (
            daily_high is None
            or val > daily_high
        )

        metar, report_type = raw_report(chosen)

        events.append({
            "event_id": (
                f"{day.isoformat()}_"
                f"{ts.strftime('%H%M%S')}"
            ),
            "local_date": day.isoformat(),
            "observed_at_utc": ts.isoformat(),
            "observed_at_ny": local.isoformat(),
            "temperature_f": val,
            "previous_temperature_f": prior_temp,
            "temperature_change_f": (
                None
                if prior_temp is None
                else round(val - prior_temp, 3)
            ),
            "previous_observed_high_f": daily_high,
            "new_observed_high": int(is_new_high),
            "report_type": report_type,
            "raw_metar": metar,
            "source": str(
                chosen.get(
                    "source",
                    "IEM_ASOS_ARCHIVE",
                )
            ),
        })

        prior_temp = val

        if daily_high is None:
            daily_high = val
        else:
            daily_high = max(daily_high, val)

    return events, conflicts


def read_contract_candles(csv_path):
    times = []
    bids = []
    asks = []

    with csv_path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):
            try:
                ts = int(row["end_period_ts"])
                bid = float(row["yes_bid_close"])
                ask = float(row["yes_ask_close"])

                valid = (
                    math.isfinite(bid)
                    and math.isfinite(ask)
                    and 0 <= bid <= ask <= 1
                )

                if not valid:
                    continue

            except (KeyError, TypeError, ValueError):
                continue

            times.append(ts)
            bids.append(bid)
            asks.append(ask)

    if (
        times != sorted(times)
        or len(times) != len(set(times))
    ):
        raise RuntimeError(
            f"Unsorted/duplicate candles: {csv_path}"
        )

    return times, bids, asks


def choose_quotes(
    candles,
    entry_ts,
    exit_ts,
    max_entry_age_sec,
    max_exit_delay_sec,
):
    times, bids, asks = candles

    # Entry: last completed candle at/before decision.
    entry_i = (
        bisect.bisect_right(
            times, entry_ts
        ) - 1
    )

    # Exit: first completed candle at/after exit.
    exit_i = bisect.bisect_left(
        times, exit_ts
    )

    if entry_i < 0 or exit_i >= len(times):
        return None

    entry_age = entry_ts - times[entry_i]
    exit_delay = times[exit_i] - exit_ts

    if entry_age < 0 or exit_delay < 0:
        raise AssertionError(
            "Look-ahead timing error."
        )

    if entry_age > max_entry_age_sec:
        return None

    if exit_delay > max_exit_delay_sec:
        return None

    return (
        times[entry_i],
        bids[entry_i],
        asks[entry_i],
        entry_age,
        times[exit_i],
        bids[exit_i],
        asks[exit_i],
        exit_delay,
    )


def iso_utc(timestamp):
    return datetime.fromtimestamp(
        timestamp,
        timezone.utc,
    ).isoformat()


def write_csv(path, headers, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=headers,
        )

        writer.writeheader()
        writer.writerows(rows)


def run(args):
    weather = load_observations(args)

    events, conflicted = build_events(
        weather
    )

    root = (
        args.base
        / "kalshi"
        / "candles"
    )

    output = args.output_dir

    if (
        output.resolve() == root.resolve()
        or output.resolve() == args.base.resolve()
    ):
        raise ValueError(
            "Output directory cannot overwrite inputs."
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        output / "observation_events.csv",
        EVENT_COLS,
        events,
    )

    events_by_day = defaultdict(list)

    for event in events:
        events_by_day[
            event["local_date"]
        ].append(event)

    reactions = []

    days_without_candle_files = []
    invalid_contracts = []

    available_contracts = Counter()
    eligible = Counter()

    delays = sorted(
        set(args.delays)
    )

    total_days = len(events_by_day)

    for i, day in enumerate(
        sorted(events_by_day),
        start=1,
    ):

        folder = root / day

        files = (
            sorted(
                folder.glob(
                    "KXHIGHNY-*.csv"
                )
            )
            if folder.exists()
            else []
        )

        if not files:
            days_without_candle_files.append(day)
            continue

        print(
            f"[{i}/{total_days}] "
            f"{day}: "
            f"{len(files)} buckets, "
            f"{len(events_by_day[day])} events",
            flush=True,
        )

        for file in files:
            ticker = file.stem

            available_contracts[day] += 1

            try:
                candles = (
                    read_contract_candles(file)
                )

            except (OSError, RuntimeError) as exc:
                invalid_contracts.append({
                    "date": day,
                    "ticker": ticker,
                    "reason": str(exc),
                })
                continue

            for ev in events_by_day[day]:

                observed_ts = int(
                    pd.Timestamp(
                        ev["observed_at_utc"]
                    ).timestamp()
                )

                for delay in delays:

                    release_ts = (
                        observed_ts
                        + delay * 60
                    )

                    entry_ts = (
                        release_ts
                        - args.entry_minutes * 60
                    )

                    exit_ts = (
                        release_ts
                        + args.exit_seconds
                    )

                    eligible[delay] += 1

                    pair = choose_quotes(
                        candles,
                        entry_ts,
                        exit_ts,
                        args.max_entry_age_seconds,
                        args.max_exit_delay_seconds,
                    )

                    if pair is None:
                        continue

                    (
                        ent_t,
                        eb,
                        ea,
                        age,
                        ex_t,
                        xb,
                        xa,
                        exit_delay,
                    ) = pair

                    entry_mid = (
                        eb + ea
                    ) / 2

                    exit_mid = (
                        xb + xa
                    ) / 2

                    reactions.append({
                        "event_id": ev["event_id"],
                        "local_date": day,
                        "observed_at_utc": (
                            ev["observed_at_utc"]
                        ),
                        "temperature_f": (
                            ev["temperature_f"]
                        ),
                        "temperature_change_f": (
                            ev["temperature_change_f"]
                        ),
                        "new_observed_high": (
                            ev["new_observed_high"]
                        ),
                        "report_type": (
                            ev["report_type"]
                        ),
                        "assumed_publication_delay_min": delay,
                        "assumed_release_at_utc": (
                            iso_utc(release_ts)
                        ),
                        "assumed_entry_at_utc": (
                            iso_utc(entry_ts)
                        ),
                        "assumed_exit_at_utc": (
                            iso_utc(exit_ts)
                        ),
                        "ticker": ticker,
                        "entry_candle_end_utc": (
                            iso_utc(ent_t)
                        ),
                        "exit_candle_end_utc": (
                            iso_utc(ex_t)
                        ),
                        "entry_quote_age_sec": age,
                        "exit_quote_delay_sec": (
                            exit_delay
                        ),
                        "entry_yes_bid": eb,
                        "entry_yes_ask": ea,
                        "exit_yes_bid": xb,
                        "exit_yes_ask": xa,
                        "entry_mid": round(
                            entry_mid, 6
                        ),
                        "exit_mid": round(
                            exit_mid, 6
                        ),
                        "mid_change": round(
                            exit_mid - entry_mid, 6
                        ),
                        "yes_buy_ask_sell_bid_gross": round(
                            xb - ea, 6
                        ),
                        "no_buy_ask_sell_bid_gross": round(
                            eb - xa, 6
                        ),
                        "entry_spread": round(
                            ea - eb, 6
                        ),
                        "exit_spread": round(
                            xa - xb, 6
                        ),
                    })

    write_csv(
        output / "market_reactions.csv",
        REACTION_COLS,
        reactions,
    )

    print("\nRESEARCH DATASET SUMMARY")
    print("Weather observations:", len(weather))
    print("Distinct observation events:", len(events))
    print(
        "Conflicting-temperature times excluded:",
        conflicted,
    )
    print(
        "Market reaction rows:",
        len(reactions),
    )
    print(
        "Days missing candle files:",
        len(days_without_candle_files),
    )
    print(
        "Invalid candle files:",
        len(invalid_contracts),
    )

    print(
        "\nASSUMED PUBLICATION DELAY SENSITIVITY"
        "\n(NOT VERIFIED RELEASE TIMES)"
    )

    summary = []

    for delay in delays:
        rows = [
            r
            for r in reactions
            if (
                r["assumed_publication_delay_min"]
                == delay
            )
        ]

        n_events = len({
            r["event_id"]
            for r in rows
        })

        n_days = len({
            r["local_date"]
            for r in rows
        })

        values = [
            abs(r["mid_change"])
            for r in rows
        ]

        yes_values = [
            r["yes_buy_ask_sell_bid_gross"]
            for r in rows
        ]

        new_high_values = [
            abs(r["mid_change"])
            for r in rows
            if r["new_observed_high"]
        ]

        record = {
            "assumed_delay_minutes": delay,
            "candidate_contract_event_pairs": (
                eligible[delay]
            ),
            "usable_contract_event_pairs": (
                len(rows)
            ),
            "usable_events": n_events,
            "usable_days": n_days,
            "mean_abs_mid_change": (
                round(
                    sum(values) / len(values),
                    6,
                )
                if values
                else None
            ),
            "mean_abs_mid_change_new_observed_high": (
                round(
                    sum(new_high_values)
                    / len(new_high_values),
                    6,
                )
                if new_high_values
                else None
            ),
            "mean_all_bucket_yes_roundtrip_gross": (
                round(
                    sum(yes_values)
                    / len(yes_values),
                    6,
                )
                if yes_values
                else None
            ),
        }

        summary.append(record)

        print(
            f"  lag={delay:2d}m "
            f"| events={n_events:4d} "
            f"| contract-event pairs={len(rows):6d} "
            f"| mean |mid Δ|="
            f"{record['mean_abs_mid_change']}"
        )

    if summary:
        write_csv(
            output / "delay_sensitivity.csv",
            list(summary[0]),
            summary,
        )

    audit = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "observation_rows_read": len(weather),
        "distinct_observation_events": len(events),
        "conflicting_observation_times_excluded": (
            conflicted
        ),
        "available_market_contracts_by_day": dict(
            available_contracts
        ),
        "days_without_candle_files": (
            days_without_candle_files
        ),
        "invalid_contract_files": invalid_contracts,
        "delay_sensitivity": summary,
        "assumptions": {
            "publication_time": (
                "Observation timestamp + assumed delay; "
                "actual historical first-seen time unknown"
            ),
            "entry": (
                f"{args.entry_minutes} minutes "
                "before assumed publication"
            ),
            "exit": (
                f"{args.exit_seconds} seconds "
                "after assumed publication"
            ),
            "entry_candle": (
                "Last completed close at/before entry; "
                f"maximum age {args.max_entry_age_seconds}s"
            ),
            "exit_candle": (
                "First completed close at/after exit; "
                f"maximum delay {args.max_exit_delay_seconds}s"
            ),
            "execution": (
                "YES ask to buy, YES bid to sell; "
                "NO synthetic ask/bid; "
                "no depth, fills, or fees"
            ),
            "selectivity": (
                "All observable buckets; "
                "no predictive trade-selection rule"
            ),
        },
    }

    (
        output / "audit.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(
        "\nOutputs saved in:",
        output.resolve(),
    )
    print(
        "WARNING: Timing assumptions and quote proxies "
        "do not establish a profitable strategy."
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    p.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2026, 7, 1),
    )

    p.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 10, 26),
    )

    p.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_BASE / "event_study"
        ),
    )

    p.add_argument(
        "--observations-csv",
        type=Path,
        help="Optional weather CSV instead of DB SELECT",
    )

    p.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[0, 5, 10],
    )

    p.add_argument(
        "--entry-minutes",
        type=int,
        default=5,
    )

    p.add_argument(
        "--exit-seconds",
        type=int,
        default=60,
    )

    p.add_argument(
        "--max-entry-age-seconds",
        type=int,
        default=120,
    )

    p.add_argument(
        "--max-exit-delay-seconds",
        type=int,
        default=120,
    )

    args = p.parse_args()

    if args.end < args.start:
        p.error("Invalid date range")

    if any(
        x < 0 or x > 60
        for x in args.delays
    ):
        p.error(
            "Release delays must be 0–60 minutes"
        )

    if args.entry_minutes <= 0:
        p.error(
            "Entry minutes must be positive"
        )

    if (
        args.exit_seconds < 0
        or args.max_entry_age_seconds < 0
        or args.max_exit_delay_seconds < 0
    ):
        p.error(
            "Timing parameters cannot be negative"
        )

    run(args)


if __name__ == "__main__":
    main()
