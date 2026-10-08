"""Read-only walk-forward backtest for KXTEMPNYCHS.

Reads historical Supabase data.
No database writes, CSV output, or order submissions.
"""
import argparse
import json
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.database.db import get_connection, read_dataframe

load_dotenv(".env")

SERIES = "KXTEMPNYCHS-"
HORIZONS = (45, 30, 15, 5)
MAX_RESEARCH_DATE = "2026-10-02"


def taker_fee(price, rate):
    return math.ceil(
        100 * rate * price * (1 - price) - 1e-10
    ) / 100


def sharpen(midpoints, alpha):
    p = np.clip(
        np.asarray(midpoints, dtype=float),
        0.001, 0.999,
    )
    log_odds = np.log(p / (1 - p))
    return 1.0 / (
        1.0 + np.exp(
            -np.clip(alpha * log_odds, -40, 40)
        )
    )


def logloss(y, p):
    q = np.clip(np.asarray(p, dtype=float), 0.001, 0.999)
    y = np.asarray(y, dtype=float)
    return float(np.mean(
        -y * np.log(q) - (1 - y) * np.log(1 - q)
    ))


def fit_alpha(train):
    use = train[train["mid"].between(0.02, 0.98)]

    if len(use) < 100:
        return 1.0

    grid = np.arange(0.50, 2.001, 0.025)
    scores = [
        logloss(use["outcome"], sharpen(use["mid"], a))
        for a in grid
    ]
    return float(grid[int(np.argmin(scores))])


def load_history(research_end):
    if not os.environ.get(
        "PREDICTION_DATABASE_URL", ""
    ).strip():
        raise RuntimeError(
            "PREDICTION_DATABASE_URL required. "
            "Refusing SQLite fallback."
        )

    with get_connection() as con:
        markets = read_dataframe("""
            SELECT ticker, target_date, yes_payout,
                   raw_market, downloaded_at_utc
            FROM weather_market_contract_history
            WHERE ticker LIKE ?
              AND target_date <= ?
            ORDER BY downloaded_at_utc
        """, con, [SERIES + "%", research_end])

        candles = read_dataframe("""
            SELECT ticker, candle_end_utc,
                   yes_bid, yes_ask,
                   downloaded_at_utc, payload_hash
            FROM weather_market_candle_history
            WHERE ticker LIKE ?
              AND interval_minutes = 1
              AND target_date <= ?
            ORDER BY downloaded_at_utc
        """, con, [SERIES + "%", research_end])

    if markets.empty or candles.empty:
        raise RuntimeError(
            "Missing hourly temperature market history."
        )

    market_duplicates = (
        len(markets) - markets["ticker"].nunique()
    )
    candle_duplicates = (
        len(candles)
        - len(candles.drop_duplicates(
            ["ticker", "candle_end_utc"]
        ))
    )

    markets = markets.drop_duplicates(
        "ticker", keep="last"
    ).copy()

    markets["outcome"] = pd.to_numeric(
        markets["yes_payout"], errors="coerce"
    )

    metadata = markets["raw_market"].map(json.loads)

    markets["event"] = metadata.map(
        lambda m: m.get("event_ticker")
    )
    markets["close"] = pd.to_datetime(
        metadata.map(lambda m: m.get("close_time")),
        utc=True,
        errors="coerce",
    )
    markets["day"] = (
        markets["close"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )

    markets = markets[
        markets["outcome"].isin([0, 1])
        & markets["close"].notna()
        & markets["event"].notna()
    ][[
        "ticker", "event", "close", "day", "outcome"
    ]].copy()

    conflicts = candles.groupby(
        ["ticker", "candle_end_utc"]
    )[["yes_bid", "yes_ask"]].nunique(
        dropna=False
    )

    disputed = conflicts[
        (conflicts["yes_bid"] > 1)
        | (conflicts["yes_ask"] > 1)
    ].index

    candles = candles.drop_duplicates(
        ["ticker", "candle_end_utc"],
        keep="last",
    ).copy()

    if len(disputed):
        keys = pd.MultiIndex.from_frame(
            candles[["ticker", "candle_end_utc"]]
        )
        candles = candles.loc[
            ~keys.isin(disputed)
        ].copy()

    candles["stamp"] = pd.to_datetime(
        candles["candle_end_utc"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    candles["bid"] = pd.to_numeric(
        candles["yes_bid"], errors="coerce"
    )
    candles["ask"] = pd.to_numeric(
        candles["yes_ask"], errors="coerce"
    )

    candles = candles[
        candles["stamp"].notna()
        & candles["bid"].between(0, 1)
        & candles["ask"].between(0, 1)
        & (candles["bid"] <= candles["ask"])
    ][["ticker", "stamp", "bid", "ask"]].copy()

    markets["threshold"] = pd.to_numeric(
        markets["ticker"].str.extract(
            r"-T(\d+(?:\.\d+)?)$"
        )[0],
        errors="coerce",
    )

    bad_events = set()

    for event, group in markets.groupby("event"):
        group = group.sort_values("threshold")

        if (
            len(group) != 10
            or group["threshold"].isna().any()
            or (
                np.diff(
                    group["outcome"].to_numpy(dtype=float)
                ) > 0
            ).any()
        ):
            bad_events.add(event)

    markets = markets[
        ~markets["event"].isin(bad_events)
    ].copy()

    print("\nDATA INTEGRITY")
    print("Valid contracts:", len(markets))
    print("Valid minute quotes:", len(candles))
    print("Duplicate contract records:", market_duplicates)
    print("Duplicate candle timestamps:", candle_duplicates)
    print("Conflicting quote timestamps:", len(disputed))
    print("Invalid threshold events:", len(bad_events))

    return markets, candles


def make_snapshots(
    markets, candles, horizon,
    max_age, entry_delay, entry_wait,
):
    left = markets.copy()

    left["decision"] = (
        left["close"]
        - pd.to_timedelta(horizon, unit="min")
    )
    left["signal_cutoff"] = (
        left["decision"] - pd.Timedelta(minutes=1)
    )
    left["entry_after"] = (
        left["decision"]
        + pd.to_timedelta(entry_delay, unit="min")
    )

    right = candles.sort_values("stamp")

    snap = pd.merge_asof(
        left.sort_values("signal_cutoff"),
        right,
        left_on="signal_cutoff",
        right_on="stamp",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(minutes=max_age),
    )

    snap = snap.rename(columns={
        "stamp": "signal_quote_time",
        "bid": "signal_bid",
        "ask": "signal_ask",
    })

    snap = snap[
        snap["signal_bid"].notna()
        & snap["signal_ask"].notna()
    ].copy()

    snap["mid"] = (
        snap["signal_bid"] + snap["signal_ask"]
    ) / 2

    entry = pd.merge_asof(
        snap.sort_values("entry_after"),
        right,
        left_on="entry_after",
        right_on="stamp",
        by="ticker",
        direction="forward",
        tolerance=pd.Timedelta(minutes=entry_wait),
    ).rename(columns={
        "stamp": "entry_quote_time",
        "bid": "entry_bid",
        "ask": "entry_ask_yes",
    })

    entry["entry_possible"] = (
        entry["entry_quote_time"].notna()
        & (entry["entry_quote_time"] < entry["close"])
    )

    return entry


def select_trades(
    frame, alpha, rate,
    min_edge, min_roi, buffer,
):
    scored = frame.copy()
    scored["fair_yes"] = sharpen(
        scored["mid"], alpha
    )

    candidates = []

    for side in ("YES", "NO"):
        if side == "YES":
            fair = scored["fair_yes"]
            ask = scored["signal_ask"]
        else:
            fair = 1 - scored["fair_yes"]
            ask = 1 - scored["signal_bid"]

        fee = ask.map(
            lambda x: taker_fee(x, rate)
        )
        cost = ask + fee + buffer
        edge = fair - cost
        roi = edge / cost.replace(0, np.nan)

        subset = scored[
            (ask >= 0.05)
            & (ask <= 0.95)
            & (edge >= min_edge)
            & (roi >= min_roi)
        ].copy()

        subset["side"] = side
        subset["signal_edge"] = edge.loc[
            subset.index
        ]

        candidates.append(subset)

    all_candidates = pd.concat(
        candidates, ignore_index=True
    )

    if all_candidates.empty:
        return pd.DataFrame()

    return (
        all_candidates
        .sort_values(
            "signal_edge", ascending=False
        )
        .drop_duplicates("event", keep="first")
    )


def run_backtest(markets, candles, args):
    summaries = []
    all_trades = []

    for horizon in HORIZONS:
        snap = make_snapshots(
            markets,
            candles,
            horizon,
            args.max_quote_age,
            args.entry_delay,
            args.entry_wait,
        )

        days = sorted(snap["day"].unique())

        if not days:
            continue

        scored = []
        trades = []

        for day in days:
            eligible_days = [
                d for d in days
                if d <= day - timedelta(days=2)
            ]
            train_days = eligible_days[
                -args.train_days:
            ]

            if len(train_days) < args.min_train_days:
                continue

            train = snap[
                snap["day"].isin(train_days)
            ]
            test = snap[
                snap["day"] == day
            ]

            alpha = fit_alpha(train)
            calibrated = sharpen(
                test["mid"], alpha
            )

            scored.append({
                "date": day,
                "alpha": alpha,
                "n": len(test),
                "raw_logloss": logloss(
                    test["outcome"], test["mid"]
                ),
                "cal_logloss": logloss(
                    test["outcome"], calibrated
                ),
                "raw_brier": np.mean(
                    (test["mid"] - test["outcome"]) ** 2
                ),
                "cal_brier": np.mean(
                    (calibrated - test["outcome"]) ** 2
                ),
            })

            candidates = select_trades(
                test,
                alpha,
                args.fee_rate,
                args.min_edge,
                args.min_roi,
                args.buffer,
            )

            if candidates.empty:
                continue

            for row in candidates.itertuples(
                index=False
            ):
                entry_price = (
                    row.entry_ask_yes
                    if row.side == "YES"
                    else 1 - row.entry_bid
                )

                status = "missing_next_quote"
                pnl = float("nan")
                buffered_pnl = float("nan")

                if (
                    row.entry_possible
                    and pd.notna(entry_price)
                    and 0.05 <= entry_price <= 0.95
                ):
                    fee = taker_fee(
                        entry_price, args.fee_rate
                    )
                    fair = (
                        row.fair_yes
                        if row.side == "YES"
                        else 1 - row.fair_yes
                    )
                    cost = entry_price + fee
                    entry_edge = (
                        fair - cost - args.buffer
                    )

                    if (
                        entry_edge >= args.min_edge
                        and entry_edge / (
                            cost + args.buffer
                        ) >= args.min_roi
                    ):
                        payout = (
                            row.outcome
                            if row.side == "YES"
                            else 1 - row.outcome
                        )

                        pnl = payout - cost
                        buffered_pnl = (
                            pnl - args.buffer
                        )
                        status = "indicative_entry"
                    else:
                        status = "edge_faded"

                trades.append({
                    "horizon_min": horizon,
                    "day": str(day),
                    "event": row.event,
                    "ticker": row.ticker,
                    "side": row.side,
                    "alpha": alpha,
                    "entry_price": entry_price,
                    "status": status,
                    "pnl": pnl,
                    "pnl_after_buffer": buffered_pnl,
                })

        scored_df = pd.DataFrame(scored)
        trades_df = pd.DataFrame(trades)

        if trades_df.empty:
            executed = pd.DataFrame()
        else:
            executed = trades_df[
                trades_df["status"] == "indicative_entry"
            ]

        n = len(executed)

        if n:
            capital = executed["entry_price"].map(
                lambda price: price + taker_fee(
                    price, args.fee_rate
                )
            ).sum()
            pnl = executed["pnl"].sum()
            conservative_pnl = executed[
                "pnl_after_buffer"
            ].sum()
            wins = int(
                (executed["pnl"] > 0).sum()
            )
        else:
            capital = 0.0
            pnl = 0.0
            conservative_pnl = 0.0
            wins = 0

        def weighted(column):
            if scored_df.empty:
                return float("nan")
            return np.average(
                scored_df[column],
                weights=scored_df["n"],
            )

        summaries.append({
            "horizon_min": horizon,
            "research_days": (
                scored_df["date"].nunique()
                if not scored_df.empty else 0
            ),
            "snapshots": (
                int(scored_df["n"].sum())
                if not scored_df.empty else 0
            ),
            "signals": len(trades_df),
            "trades": n,
            "wins": wins,
            "win_rate_pct": (
                100 * wins / n
                if n else float("nan")
            ),
            "raw_logloss": weighted("raw_logloss"),
            "cal_logloss": weighted("cal_logloss"),
            "raw_brier": weighted("raw_brier"),
            "cal_brier": weighted("cal_brier"),
            "capital": capital,
            "pnl": pnl,
            "pnl_after_buffer": conservative_pnl,
            "roi_pct": (
                100 * pnl / capital
                if capital else float("nan")
            ),
        })

        all_trades.extend(trades)

        print(
            f"Horizon {horizon:2d}m | "
            f"research days={summaries[-1]['research_days']} | "
            f"signals={len(trades_df)} | "
            f"indicative trades={n}",
            flush=True,
        )

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(all_trades),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-days", type=int, default=14
    )
    parser.add_argument(
        "--min-train-days", type=int, default=10
    )
    parser.add_argument(
        "--max-quote-age", type=int, default=3
    )
    parser.add_argument(
        "--entry-delay", type=int, default=1
    )
    parser.add_argument(
        "--entry-wait", type=int, default=2
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.025
    )
    parser.add_argument(
        "--min-roi", type=float, default=0.10
    )
    parser.add_argument(
        "--buffer", type=float, default=0.01
    )
    parser.add_argument(
        "--fee-rate", type=float, default=0.07
    )
    parser.add_argument(
        "--research-end",
        default=MAX_RESEARCH_DATE,
    )

    args = parser.parse_args()

    if (
        args.train_days < args.min_train_days
        or args.min_train_days < 2
    ):
        parser.error("Invalid training window.")

    if args.research_end > MAX_RESEARCH_DATE:
        parser.error(
            "Dates after October 2 were reserved "
            "for later evaluation."
        )

    print("\nHOURLY NYC MARKET-ONLY BACKTEST")
    print("=" * 75)
    print("Series:", SERIES)
    print("Research ends:", args.research_end)
    print("Database access: SELECT only")

    markets, candles = load_history(
        args.research_end
    )

    summary, trades = run_backtest(
        markets, candles, args
    )

    print("\nWALK-FORWARD RESEARCH RESULTS")
    print("=" * 75)

    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    if not trades.empty:
        for horizon, group in trades.groupby(
            "horizon_min"
        ):
            picked = group[
                group["status"] == "indicative_entry"
            ]
            if picked.empty:
                continue

            print(
                f"\nHORIZON {horizon} MINUTES "
                "INDICATIVE ENTRIES"
            )
            print(picked.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            ))

    print("\nCOMPLETE")
    print("No orders submitted or results stored.")


if __name__ == "__main__":
    main()
