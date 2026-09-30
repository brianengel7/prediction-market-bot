import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.database.db import get_connection, read_dataframe

NY = "America/New_York"

# Use this explicit list when fitting models.
# Payouts and archive download times are excluded.
FEATURE_COLUMNS = [
    "market_probability", "market_spread", "market_mid_change_1h",
    "hour_local", "strike_type", "floor_strike", "cap_strike",
    "temperature_f", "observed_high_since_midnight_f",
    "temperature_trend_f_per_hour", "observation_age_minutes",
    "observations_so_far", "largest_observation_gap_minutes",
    "temperature_minus_floor", "temperature_minus_cap",
    "observed_high_minus_floor", "observed_high_minus_cap",
]


def utc(values):
    return pd.to_datetime(values, utc=True, format="mixed")


def latest(frame, keys):
    frame = frame.copy()
    frame["downloaded_at_utc"] = utc(frame["downloaded_at_utc"])
    return frame.sort_values(
        ["downloaded_at_utc", "payload_hash"]
    ).drop_duplicates(keys, keep="last")


def weather_at_hours(observations, hours, lag_minutes):
    observations = observations.copy()
    observations["observed_at_utc"] = utc(observations["observed_at_utc"])
    observations["temperature_f"] = pd.to_numeric(
        observations["temperature_f"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    observations["availability_assumed"] = (
        observations["available_at_utc"].isna()
    )
    observations["eligible_at_utc"] = utc(
        observations["available_at_utc"]
    ).fillna(
        observations["observed_at_utc"]
        + pd.Timedelta(minutes=lag_minutes)
    )

    if observations["eligible_at_utc"].lt(
        observations["observed_at_utc"]
    ).any():
        raise ValueError(
            "An availability timestamp precedes its observation."
        )

    # Unknown publication times prevent ordering conflicting revisions.
    versions = observations.groupby("observed_at_utc").agg(
        values=("temperature_f", lambda x: x.nunique(dropna=False)),
        assumed=("availability_assumed", "any"),
    )
    ambiguous = versions.index[
        versions["values"].gt(1) & versions["assumed"]
    ]
    observations = observations[
        ~observations["observed_at_utc"].isin(ambiguous)
        & observations["temperature_f"].notna()
    ].sort_values(
        ["observed_at_utc", "eligible_at_utc", "payload_hash"]
    )

    rows = []

    for hour in hours.itertuples(index=False):
        available = observations[
            observations["eligible_at_utc"].le(
                hour.decision_time_utc
            )
        ].drop_duplicates("observed_at_utc", keep="last")

        midnight = pd.Timestamp(
            hour.target_date, tz=NY
        ).tz_convert("UTC")

        today = available[
            available["observed_at_utc"].ge(midnight)
        ]

        row = {
            "target_date": hour.target_date,
            "decision_time_utc": hour.decision_time_utc,
            "observation_time_utc": pd.NaT,
            "observation_eligible_at_utc": pd.NaT,
            "temperature_f": np.nan,
            "observation_age_minutes": np.nan,
            "observed_high_since_midnight_f": (
                today["temperature_f"].max()
            ),
            "observations_so_far": len(today),
            "largest_observation_gap_minutes": (
                today["observed_at_utc"]
                .diff().dt.total_seconds().max() / 60
            ),
            "temperature_trend_f_per_hour": np.nan,
            "weather_availability_assumed": bool(
                available["availability_assumed"].any()
            ),
        }

        if not available.empty:
            current = available.iloc[-1]

            row.update({
                "observation_time_utc": current["observed_at_utc"],
                "observation_eligible_at_utc": (
                    current["eligible_at_utc"]
                ),
                "temperature_f": current["temperature_f"],
                "observation_age_minutes": (
                    hour.decision_time_utc
                    - current["observed_at_utc"]
                ).total_seconds() / 60,
            })

            earlier = available[
                available["observed_at_utc"].le(
                    current["observed_at_utc"]
                    - pd.Timedelta(hours=3)
                )
            ]

            if not earlier.empty:
                previous = earlier.iloc[-1]
                elapsed = (
                    current["observed_at_utc"]
                    - previous["observed_at_utc"]
                ).total_seconds() / 3600

                if 3 <= elapsed <= 4:
                    row["temperature_trend_f_per_hour"] = (
                        current["temperature_f"]
                        - previous["temperature_f"]
                    ) / elapsed

        rows.append(row)

    return pd.DataFrame(rows), len(ambiguous)


def build_hourly_dataset(
    start_date,
    end_date,
    observation_lag_minutes=30,
    max_observation_age_minutes=120,
):
    if not os.getenv("PREDICTION_DATABASE_URL", "").strip():
        raise RuntimeError("PREDICTION_DATABASE_URL is missing.")

    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must be on or before end_date.")

    if observation_lag_minutes < 0 or max_observation_age_minutes <= 0:
        raise ValueError(
            "Lag must be nonnegative; maximum age must be positive."
        )

    start = (
        pd.Timestamp(start_date, tz=NY) - pd.DateOffset(days=1)
    ).tz_convert("UTC")
    end = (
        pd.Timestamp(end_date, tz=NY) + pd.DateOffset(days=1)
    ).tz_convert("UTC")

    with get_connection() as connection:
        contracts = read_dataframe(
            """
            SELECT ticker, target_date, yes_payout, raw_market,
                   payload_hash, downloaded_at_utc
            FROM weather_market_contract_history
            WHERE target_date >= ? AND target_date <= ?
            """,
            connection,
            params=(start_date, end_date),
        )
        candles = read_dataframe(
            """
            SELECT ticker, target_date, candle_end_utc,
                   yes_bid, yes_ask, payload_hash, downloaded_at_utc
            FROM weather_market_candle_history
            WHERE target_date >= ? AND target_date <= ?
              AND interval_minutes = 60
            """,
            connection,
            params=(start_date, end_date),
        )
        observations = read_dataframe(
            """
            SELECT observed_at_utc, temperature_f,
                   available_at_utc, payload_hash
            FROM weather_observation_history
            WHERE source = ? AND station = ?
              AND observed_at_utc >= ? AND observed_at_utc < ?
            """,
            connection,
            params=(
                "IEM_ASOS_ARCHIVE", "KNYC",
                start.isoformat(), end.isoformat(),
            ),
        )

    if contracts.empty or candles.empty or observations.empty:
        raise RuntimeError("One or more hourly archives are empty.")

    contracts = latest(contracts, ["ticker"])
    contracts["yes_payout"] = pd.to_numeric(
        contracts["yes_payout"], errors="coerce"
    )

    if not contracts["yes_payout"].isin([0.0, 1.0]).all():
        raise RuntimeError(
            "This dataset requires settled binary payouts."
        )

    if not contracts.groupby(
        "target_date"
    )["yes_payout"].sum().eq(1).all():
        raise RuntimeError(
            "Expected one winning bracket per target date."
        )

    raw = contracts["raw_market"].map(json.loads)
    contracts["strike_type"] = raw.map(
        lambda x: x.get("strike_type")
    )

    for column in ("floor_strike", "cap_strike"):
        contracts[column] = pd.to_numeric(
            raw.map(lambda x: x.get(column)), errors="coerce"
        )

    candles = latest(candles, ["ticker", "candle_end_utc"])
    candles["decision_time_utc"] = utc(
        candles["candle_end_utc"]
    )

    for column in ("yes_bid", "yes_ask"):
        candles[column] = pd.to_numeric(
            candles[column], errors="coerce"
        )

    valid = (
        candles["yes_bid"].ge(0)
        & candles["yes_bid"].lt(1)
        & candles["yes_ask"].gt(0)
        & candles["yes_ask"].le(1)
        & candles["yes_bid"].le(candles["yes_ask"])
    )
    candles = candles[valid].sort_values(
        ["ticker", "decision_time_utc"]
    ).copy()

    candles["market_mid"] = (
        candles["yes_bid"] + candles["yes_ask"]
    ) / 2
    candles["market_spread"] = (
        candles["yes_ask"] - candles["yes_bid"]
    )

    previous_times = candles.groupby(
        "ticker"
    )["decision_time_utc"].shift()

    # A gap longer than one hour produces a missing momentum feature.
    candles["market_mid_change_1h"] = (
        candles.groupby("ticker")["market_mid"].diff().where(
            (
                candles["decision_time_utc"] - previous_times
            ).eq(pd.Timedelta(hours=1))
        )
    )

    local = candles["decision_time_utc"].dt.tz_convert(NY)
    candles["hour_local"] = local.dt.hour

    # Initial research scope: decisions during the target calendar day.
    candles = candles[
        local.dt.strftime("%Y-%m-%d").eq(
            candles["target_date"]
        )
    ].copy()

    keys = ["target_date", "decision_time_utc"]
    counts = candles.groupby(keys)["ticker"].transform("nunique")
    expected = candles["target_date"].map(
        contracts.groupby("target_date")["ticker"].nunique()
    )
    candles = candles[counts.eq(expected)].copy()

    if candles.empty:
        raise RuntimeError(
            "No complete intraday quote periods remain."
        )

    candles["midpoint_sum"] = candles.groupby(
        keys
    )["market_mid"].transform("sum")
    candles["market_probability"] = (
        candles["market_mid"] / candles["midpoint_sum"]
    )

    hours = candles[keys].drop_duplicates().sort_values(
        "decision_time_utc"
    )
    weather, ambiguous = weather_at_hours(
        observations, hours, observation_lag_minutes
    )

    metadata = contracts[[
        "ticker", "target_date", "strike_type",
        "floor_strike", "cap_strike", "yes_payout",
    ]]

    data = candles.merge(
        metadata,
        on=["ticker", "target_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        weather,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    if data["yes_payout"].isna().any():
        raise RuntimeError(
            "A candle could not be matched to its contract."
        )

    for bound in ("floor", "cap"):
        data[f"temperature_minus_{bound}"] = (
            data["temperature_f"] - data[f"{bound}_strike"]
        )
        data[f"observed_high_minus_{bound}"] = (
            data["observed_high_since_midnight_f"]
            - data[f"{bound}_strike"]
        )

    data["weather_usable"] = (
        data["observation_age_minutes"].between(
            0, max_observation_age_minutes
        )
        & data["observed_high_since_midnight_f"].notna()
    )
    data["observation_lag_minutes"] = observation_lag_minutes
    data["no_bid"] = 1 - data["yes_ask"]
    data["no_ask"] = 1 - data["yes_bid"]

    data = data.sort_values(
        keys + ["ticker"]
    ).reset_index(drop=True)
    data.attrs["ambiguous_observation_timestamps"] = ambiguous

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--observation-lag-minutes", type=int, default=30
    )
    parser.add_argument(
        "--max-observation-age-minutes", type=int, default=120
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    data = build_hourly_dataset(
        args.start_date,
        args.end_date,
        args.observation_lag_minutes,
        args.max_observation_age_minutes,
    )
    hours = data.drop_duplicates(
        ["target_date", "decision_time_utc"]
    )

    print(f"Target dates: {data['target_date'].nunique()}")
    print(f"Complete quote hours: {len(hours)}")
    print(f"Contract-hour rows: {len(data)}")
    print(
        f"Weather-usable hours: "
        f"{hours['weather_usable'].sum()}"
    )
    print(
        f"Hours with estimated availability: "
        f"{hours['weather_availability_assumed'].sum()}"
    )
    print(
        f"Ambiguous observation times excluded: "
        f"{data.attrs['ambiguous_observation_timestamps']}"
    )
    print(
        f"Hours without a valid 3-4h temperature trend: "
        f"{hours['temperature_trend_f_per_hour'].isna().sum()}"
    )

    print("\nWeather coverage by date:")
    coverage = hours.groupby(
        "target_date"
    )["weather_usable"].agg(["size", "sum"])
    coverage.columns = ["quote_hours", "weather_usable_hours"]
    print(coverage.agg(["min", "median", "max"]).to_string())

    print("\nSample:")
    print(data[[
        "target_date", "hour_local", "ticker",
        "market_probability", "temperature_f",
        "observed_high_since_midnight_f",
        "observation_age_minutes", "weather_usable", "yes_payout",
    ]].head(12).to_string(index=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()