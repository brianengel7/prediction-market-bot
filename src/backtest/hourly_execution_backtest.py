import os

import numpy as np
import pandas as pd
import argparse
from src.backtest.hourly_cost_screen import (
    ALPHA, MIN_NET_EDGE, best_per_hour, candidates, screening_fee,
)
from src.backtest.hourly_dataset import build_hourly_dataset, latest, utc
from src.backtest.hourly_market_only_backtest import (
    HOUR_KEYS, KEYS, prepare_market,
)
from src.backtest.hourly_probability_backtest import (
    END,
    TEST_START,
    VALID_END,
    VALID_START,
)
from src.database.db import get_connection, read_dataframe

COST_BUFFER = 0.01
ENTRY_DELAY = pd.Timedelta(minutes=1)

AUDIT_COLUMNS = [
    "target_date", "hour_local", "ticker", "side",
    "decision_time_utc", "entry_time_utc", "model_probability",
    "signal_ask", "signal_net_edge", "entry_ask", "entry_fee",
    "entry_net_edge", "status", "payout",
    "pnl_after_fees", "pnl_with_buffer",
]


def load_minutes(start_date, end_date):
    if not os.getenv("PREDICTION_DATABASE_URL", "").strip():
        raise RuntimeError("PREDICTION_DATABASE_URL is required.")

    with get_connection() as connection:
        minute = read_dataframe(
            """
            SELECT ticker, target_date, candle_end_utc, yes_bid, yes_ask,
                   payload_hash, downloaded_at_utc
            FROM weather_market_candle_history
            WHERE target_date >= ? AND target_date <= ?
              AND interval_minutes = 1
            """,
            connection,
            params=[start_date, end_date],
        )

    if minute.empty:
        raise ValueError("No one-minute candles in the requested date window.")

    minute["candle_end_utc"] = utc(minute["candle_end_utc"])
    minute = latest(
        minute, ["target_date", "ticker", "candle_end_utc"]
    )

    for column in ["yes_bid", "yes_ask"]:
        minute[column] = pd.to_numeric(minute[column], errors="coerce")

    minute = minute.rename(columns={
        "candle_end_utc": "entry_time_utc",
        "yes_bid": "entry_yes_bid",
        "yes_ask": "entry_yes_ask",
    })
    minute["minute_present"] = True

    return minute[
        ["target_date", "ticker", "entry_time_utc",
         "entry_yes_bid", "entry_yes_ask", "minute_present"]
    ]


def simulate(signals, minute, labels):
    if signals.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    selected = signals.copy()
    selected["entry_time_utc"] = (
        selected["decision_time_utc"] + ENTRY_DELAY
    )
    selected = selected.merge(
        minute,
        on=["target_date", "ticker", "entry_time_utc"],
        how="left",
        validate="many_to_one",
    ).merge(
        labels,
        on=KEYS,
        how="left",
        validate="one_to_one",
    ).sort_values(HOUR_KEYS)

    processed_hours = set()
    records = []

    for row in selected.itertuples(index=False):
        hour_key = row.decision_time_utc.floor("h")
        if hour_key in processed_hours:
            continue
        processed_hours.add(hour_key)

        record = {
            "target_date": row.target_date,
            "hour_local": (
                row.decision_time_utc
                .tz_convert("America/New_York").hour
            ),
            "ticker": row.ticker,
            "side": row.side,
            "decision_time_utc": row.decision_time_utc,
            "entry_time_utc": row.entry_time_utc,
            "model_probability": row.model_probability,
            "signal_ask": row.ask,
            "signal_net_edge": row.net_edge,
        }

        if pd.isna(row.minute_present):
            record["status"] = "unresolved_missing_quote"

        else:
            bid, ask = row.entry_yes_bid, row.entry_yes_ask
            valid = (
                np.isfinite(bid) and np.isfinite(ask)
                and 0 <= bid < 1
                and 0 < ask <= 1
                and bid <= ask
            )

            if not valid:
                record["status"] = "unresolved_invalid_quote"

            else:
                entry_ask = ask if row.side == "YES" else 1 - bid
                fee = screening_fee(entry_ask)
                cash_cost = entry_ask + fee
                edge = (
                    row.model_probability - cash_cost - COST_BUFFER
                )

                record.update({
                    "entry_ask": entry_ask,
                    "entry_fee": fee,
                    "entry_net_edge": edge,
                })

                if edge < MIN_NET_EDGE - 1e-12:
                    record["status"] = "edge_faded"

                else:
                    if row.yes_payout not in [0.0, 1.0]:
                        raise ValueError(
                            "Missing or invalid settlement label."
                        )

                    payout = (
                        row.yes_payout
                        if row.side == "YES"
                        else 1 - row.yes_payout
                    )
                    record.update({
                        "status": "simulated_entry",
                        "payout": payout,
                        "pnl_after_fees": payout - cash_cost,
                        "pnl_with_buffer": (
                            payout - cash_cost - COST_BUFFER
                        ),
                    })

        records.append(record)

    return pd.DataFrame(records, columns=AUDIT_COLUMNS)


def main(sample_name="validation"):
    ranges = {
        "validation": (VALID_START, VALID_END),
        "test": (TEST_START, END),
    }
    start_date, end_date = ranges[sample_name]

    full = build_hourly_dataset(
        start_date,
        end_date,
        observation_lag_minutes=30,
    )
    frame = (
        full[
            KEYS + ["market_probability", "yes_payout", "yes_bid", "yes_ask"]
        ]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    sample = prepare_market(frame)
    sample["overshoot"] = np.zeros(len(frame))
    minute = load_minutes(start_date, end_date)
    labels = frame[KEYS + ["yes_payout"]]

    print(f"{sample_name.capitalize()}: {start_date} -> {end_date}")
    print(f"Complete hourly quote sets: {sample['n_hours']}")
    print(f"Unique minute candle periods loaded: {len(minute)}")
    print(f"Frozen alpha: {ALPHA:.12f}")
    print("One contract; exact next-minute ask; at most one entry per hour.")
    print("Minimum edge: 2.5c after fees and a 1c cost buffer.")
    print("Missing/invalid next-minute quotes leave only that hour unresolved.")
    if sample_name == "test":
        print("Reserved test is being scored with frozen parameters.")
    else:
        print("Reserved test remains unscored.")

    summaries = []
    calibrated_audit = None

    for model, alpha in [
        ("raw_market", 1.0),
        ("calibrated_market", ALPHA),
    ]:
        base = candidates(frame, sample, alpha)
        base["net_edge"] -= COST_BUFFER
        best = best_per_hour(base)
        signals = best.loc[
            best["net_edge"].ge(MIN_NET_EDGE - 1e-12)
        ].copy()

        audit = simulate(signals, minute, labels)
        trades = audit.loc[audit["status"].eq("simulated_entry")]
        unresolved = audit.loc[
            audit["status"].str.startswith("unresolved", na=False)
        ]
        wins = int(trades["payout"].eq(1.0).sum())

        summaries.append({
            "model": model,
            "signal_hours": len(signals),
            "signal_dates": signals["target_date"].nunique(),
            "simulated_entries": len(trades),
            "unresolved_hours": len(unresolved),
            "edge_faded_attempts": int(
                audit["status"].eq("edge_faded").sum()
            ),
            "wins": wins,
            "losses": len(trades) - wins,
            "fees_only_pnl": float(trades["pnl_after_fees"].sum()),
            "fees_plus_buffer_pnl": float(
                trades["pnl_with_buffer"].sum()
            ),
            "max_entries_per_date": (
                int(trades.groupby("target_date").size().max())
                if not trades.empty else 0
            ),
            "entry_cost_dollars": float(
                (trades["entry_ask"] + trades["entry_fee"]).sum()
            ),
        })

        if model == "calibrated_market":
            calibrated_audit = audit

    print("\nQuote-based simulation: dollar P&L for one-contract entries.")
    print(pd.DataFrame(summaries).to_string(index=False))

    print("\nCalibrated entry attempts:")
    columns = [
        "target_date", "hour_local", "ticker", "side",
        "signal_net_edge", "entry_ask", "entry_net_edge", "status",
    ]
    print(
        calibrated_audit[columns].to_string(index=False)
        if not calibrated_audit.empty else "None"
    )

    trades = calibrated_audit.loc[
        calibrated_audit["status"].eq("simulated_entry")
    ]
    print("\nSettlement results for simulated entries:")
    columns = [
        "target_date", "ticker", "side", "entry_ask", "entry_fee",
        "payout", "pnl_after_fees", "pnl_with_buffer",
    ]
    print(
        trades[columns].to_string(index=False)
        if not trades.empty else "None"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        choices=["validation", "test"],
        default="validation",
    )
    main(parser.parse_args().sample)