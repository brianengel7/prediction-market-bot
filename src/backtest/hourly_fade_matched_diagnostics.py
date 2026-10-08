"""Read-only matched-event and quote-quality audit.

Compares alpha-only against alpha + 2c fading.
Prints results only. No orders, output files, or database writes.
"""
from collections import Counter
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest import hourly_continuous_multi_backtest as model

MINUTE_NS = 60_000_000_000


def config(mode, cents):
    return SimpleNamespace(
        start=date(2026, 9, 10),
        end=date(2026, 10, 7),
        scan_start=30,
        scan_end=5,
        max_entries=1,
        max_event_cost=2.50,
        min_roi=5,
        min_winning_roi=30,
        momentum_mode=mode,
        momentum_cents=cents,
        momentum_lookback=3,
        return_trades=True,
    )


def quote_asof(book, cutoff_ns, max_age_min=1):
    if book is None:
        return None

    times, bids, asks = book

    i = int(
        np.searchsorted(
            times, cutoff_ns, side="right"
        )
    ) - 1

    if i < 0:
        return None

    age_ns = int(cutoff_ns - times[i])

    if (
        age_ns < 0
        or age_ns > max_age_min * MINUTE_NS
    ):
        return None

    bid = float(bids[i])
    ask = float(asks[i])

    return {
        "stamp_ns": int(times[i]),
        "age_seconds": age_ns / 1e9,
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "spread_c": 100 * (ask - bid),
    }


def verify_fade_quote(trade, book):
    """Reconstruct the exact historical momentum signal."""
    cutoff_ns = (
        int(trade.close_ns)
        - (int(trade.minutes_left) + 1) * MINUTE_NS
    )

    previous = quote_asof(
        book,
        cutoff_ns - 3 * MINUTE_NS,
        1,
    )

    current = quote_asof(
        book, cutoff_ns, 1
    )

    if previous is None or current is None:
        raise AssertionError(
            f"Missing pre-decision quote: {trade.event}"
        )

    sign = 1 if trade.side == "YES" else -1

    move_c = (
        sign * (current["mid"] - previous["mid"]) * 100
    )

    if move_c > -2 + 1e-8:
        raise AssertionError(
            f"Fade condition mismatch: "
            f"{trade.event}, move={move_c:.4f}c"
        )

    bid_move = sign * (
        current["bid"] - previous["bid"]
    ) * 100

    ask_move = sign * (
        current["ask"] - previous["ask"]
    ) * 100

    flags = []

    if abs(move_c) > 10:
        flags.append("LARGE_MOVE")

    if max(
        current["spread_c"],
        previous["spread_c"],
    ) > 10:
        flags.append("WIDE_SPREAD")

    if (
        previous["age_seconds"] >= 55
        or current["age_seconds"] >= 55
    ):
        flags.append("NEAR_AGE_LIMIT")

    if bid_move >= 0 or ask_move >= 0:
        flags.append("ONE_SIDED_MOVE")

    return {
        "day": str(trade.day),
        "event": trade.event,
        "minutes_left": int(trade.minutes_left),
        "ticker": trade.ticker,
        "side": trade.side,
        "move_c": move_c,
        "bid_move_c": bid_move,
        "ask_move_c": ask_move,
        "prev_spread_c": previous["spread_c"],
        "cur_spread_c": current["spread_c"],
        "prev_age_s": previous["age_seconds"],
        "cur_age_s": current["age_seconds"],
        "prev_bid": previous["bid"],
        "prev_ask": previous["ask"],
        "cur_bid": current["bid"],
        "cur_ask": current["ask"],
        "entry": float(trade.entry),
        "fair": float(trade.fair),
        "won": int(trade.win),
        "pnl": float(trade.buffered_pnl),
        "flags": ",".join(flags) if flags else "OK",
    }


def compare_events(alpha, fade, period):
    a = alpha[
        alpha.period == period
    ].copy()

    f = fade[
        fade.period == period
    ].copy()

    if (
        a.event.duplicated().any()
        or f.event.duplicated().any()
    ):
        raise AssertionError(
            "Expected at most one entry per event"
        )

    left = a.set_index("event")
    right = f.set_index("event")

    union = left.index.union(right.index)
    records = []

    for event in union:
        aa = (
            left.loc[event]
            if event in left.index
            else None
        )

        ff = (
            right.loc[event]
            if event in right.index
            else None
        )

        if aa is None:
            category = "FADE_ONLY"
        elif ff is None:
            category = "ALPHA_ONLY"
        elif (
            aa.ticker == ff.ticker
            and aa.side == ff.side
        ):
            category = "BOTH_SAME_LEG"
        else:
            category = "BOTH_DIFFERENT_LEG"

        a_pnl = (
            float(aa.buffered_pnl)
            if aa is not None else 0.0
        )
        f_pnl = (
            float(ff.buffered_pnl)
            if ff is not None else 0.0
        )

        records.append({
            "event": event,
            "day": str(
                ff.day if ff is not None else aa.day
            ),
            "category": category,
            "alpha_minutes": (
                int(aa.minutes_left)
                if aa is not None else None
            ),
            "fade_minutes": (
                int(ff.minutes_left)
                if ff is not None else None
            ),
            "alpha_pnl": a_pnl,
            "fade_pnl": f_pnl,
            "pnl_delta": f_pnl - a_pnl,
            "same_entry_time": (
                int(aa.entry_ns) == int(ff.entry_ns)
                if aa is not None and ff is not None
                else False
            ),
        })

    matched = pd.DataFrame(records)

    print("\n" + "=" * 95)
    print(period, "— MATCHED HOURLY EVENTS")
    print("=" * 95)

    for name, trades in (
        ("ALPHA_ONLY", a),
        ("FADE_2C", f),
    ):
        pnl = float(trades.buffered_pnl.sum())

        risked = float(
            (trades.capital + model.BUFFER).sum()
        )

        roi = (
            100 * pnl / risked
            if risked else float("nan")
        )

        print(
            f"{name:12} "
            f"trades={len(trades):2} "
            f"wins={int(trades.win.sum()):2} "
            f"pnl=${pnl:+.2f} "
            f"risked=${risked:.2f} "
            f"ROI={roi:+.2f}%"
        )

    print("\nMATCH CATEGORIES")

    summary = (
        matched.groupby(
            "category", as_index=False
        )
        .agg(
            events=("event", "size"),
            alpha_pnl=("alpha_pnl", "sum"),
            fade_pnl=("fade_pnl", "sum"),
            delta=("pnl_delta", "sum"),
        )
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.2f}",
        )
    )

    same = matched[
        matched.category == "BOTH_SAME_LEG"
    ]

    print(
        "\nSame-leg events with identical entry times:",
        int(same.same_entry_time.sum()),
        "of",
        len(same),
    )

    print("\nALL MATCHED / UNMATCHED EVENTS")

    print(
        matched.sort_values(["day", "event"])[[
            "day",
            "event",
            "category",
            "alpha_minutes",
            "fade_minutes",
            "alpha_pnl",
            "fade_pnl",
            "pnl_delta",
        ]].to_string(
            index=False,
            float_format=lambda x: f"{x:+.2f}",
        )
    )


def main():
    print("\nKXTEMPNYCHS — FADE MOMENTUM AUDIT")
    print("=" * 95)
    print("Alpha only vs alpha + 2c fade")
    print("30–5m scanning window")
    print("5% expected ROI, 30% winning ROI")
    print("One entry per event")
    print("Research ends Oct 2; viewed OOS Oct 3–7")
    print("Read-only; no results stored")

    original_load = model.load_history
    markets, candles = original_load("2026-10-07")

    # Reuse history in memory for both models.
    model.load_history = lambda _: (
        markets, candles
    )

    original_snapshots = model.make_snapshots
    snapshot_cache = {}

    def cached_snapshots(
        m, c, horizon, age, delay, wait
    ):
        key = (
            horizon, age, delay, wait
        )

        if key not in snapshot_cache:
            snapshot_cache[key] = original_snapshots(
                m, c, horizon, age, delay, wait
            )

        return snapshot_cache[key]

    model.make_snapshots = cached_snapshots

    try:
        with redirect_stdout(StringIO()):
            _, alpha, _ = model.main(
                config("none", 0)
            )

            _, fade, _ = model.main(
                config("fade", 2)
            )

    finally:
        model.load_history = original_load
        model.make_snapshots = original_snapshots

    a = alpha[
        alpha.period == "RESEARCH"
    ]

    f = fade[
        fade.period == "RESEARCH"
    ]

    ao = alpha[
        alpha.period == "VIEWED_OOS"
    ]

    fo = fade[
        fade.period == "VIEWED_OOS"
    ]

    if (
        len(a) != 37
        or len(f) != 12
        or len(ao) != 1
        or len(fo) != 1
    ):
        raise RuntimeError(
            "Baseline trade counts changed. "
            f"Research={len(a)}/{len(f)}, "
            f"viewed={len(ao)}/{len(fo)}"
        )

    if (
        abs(float(a.buffered_pnl.sum()) - 4.24) > 0.02
        or abs(float(f.buffered_pnl.sum()) - 1.72) > 0.02
    ):
        raise RuntimeError(
            "Historical P&L changed. "
            "Stop and investigate."
        )

    required = {"entry_ns", "close_ns"}

    if not required.issubset(fade.columns):
        raise RuntimeError(
            "Missing entry/close timestamps "
            "from continuous backtest."
        )

    print(
        "\nPASS: alpha=37 trades/$4.24; "
        "fade=12 trades/$1.72"
    )

    tickers = set(fade.ticker)
    books = {}

    selected_candles = candles[
        candles.ticker.isin(tickers)
    ]

    for ticker, group in selected_candles.groupby(
        "ticker"
    ):
        group = group.sort_values("stamp")

        books[ticker] = (
            group.stamp.dt.as_unit("ns")
            .astype("int64").to_numpy(),
            group.bid.to_numpy(dtype=float),
            group.ask.to_numpy(dtype=float),
        )

    audited = pd.DataFrame([
        verify_fade_quote(
            row, books.get(row.ticker)
        )
        for row in fade.itertuples(index=False)
    ])

    print("\n" + "=" * 95)
    print("FADE SIGNAL QUOTE QUALITY")
    print("=" * 95)

    print(
        audited[[
            "day",
            "minutes_left",
            "ticker",
            "side",
            "move_c",
            "bid_move_c",
            "ask_move_c",
            "prev_spread_c",
            "cur_spread_c",
            "prev_age_s",
            "cur_age_s",
            "won",
            "pnl",
            "flags",
        ]].to_string(
            index=False,
            float_format=lambda x: f"{x:+.2f}",
        )
    )

    print("\nQUOTE QUALITY SUMMARY")

    print(
        "Average directed move:",
        f"{audited.move_c.mean():+.2f}c"
    )

    print(
        "Median directed move:",
        f"{audited.move_c.median():+.2f}c"
    )

    print(
        "Median current spread:",
        f"{audited.cur_spread_c.median():.2f}c"
    )

    print(
        "Largest current spread:",
        f"{audited.cur_spread_c.max():.2f}c"
    )

    flags = Counter(
        flag
        for row in audited["flags"]
        for flag in row.split(",")
        if flag != "OK"
    )

    print("Quality flags:", dict(flags))

    compare_events(alpha, fade, "RESEARCH")
    compare_events(alpha, fade, "VIEWED_OOS")

    print("\nDIAGNOSTIC COMPLETE")
    print(
        "Minute candles are indicative quotes, "
        "not confirmed executable fills."
    )
    print("No orders or database writes.")


if __name__ == "__main__":
    main()
