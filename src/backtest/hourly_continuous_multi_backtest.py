"""Read-only, minute-by-minute, multi-entry walk-forward backtest.

Uses existing hourly historical candle/settlement tables.
Does not place orders, alter live traders, or write backtest results.
"""
import argparse
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env")

from src.backtest.hourly_index_market_backtest import (
    load_history,
    make_snapshots,
    fit_alpha,
    sharpen,
    taker_fee,
)
from src.backtest.hourly_contract_compatibility import compatible

MINUTE_NS = 60_000_000_000
MIN_EDGE = 0.025
MIN_ROI = 0.05
BUFFER = 0.01
FEE_RATE = 0.07

ALPHA_SWEEP_CACHE = {}
BUCKETS = tuple(range(1, 5)) + tuple(range(5, 61, 5))


def asof_quote(data, when, age=3):
    if data is None:
        return None

    times, bids, asks = data

    i = int(
        np.searchsorted(
            times, when, side="right"
        )
    ) - 1

    if i < 0 or when - times[i] > age * MINUTE_NS:
        return None

    return float(bids[i]), float(asks[i])


def first_later_quote(data, start, deadline):
    if data is None:
        return None

    times, bids, asks = data

    i = int(np.searchsorted(
        times, start, side="left"
    ))

    if i >= len(times) or times[i] > deadline:
        return None

    return (
        int(times[i]),
        float(bids[i]),
        float(asks[i]),
    )


def momentum_passes(
    delta_yes, side, mode, minimum_cents
):
    """Momentum in the proposed YES/NO purchase direction."""
    if mode == "none":
        return True

    if delta_yes is None:
        return False

    directed_cents = 100 * (
        delta_yes if side == "YES" else -delta_yes
    )

    if mode == "follow":
        return (
            directed_cents >= minimum_cents - 1e-9
        )

    if mode == "fade":
        return (
            directed_cents <= -minimum_cents + 1e-9
        )

    raise ValueError(f"Invalid momentum mode: {mode}")


def main(args):
    if args.start > args.end:
        raise ValueError(
            "Start date must not exceed end date"
        )

    markets, candles = load_history(
        args.end.isoformat()
    )

    if markets.empty or candles.empty:
        raise RuntimeError(
            "No usable hourly history"
        )

    print(
        "\nONE-MINUTE CONTINUOUS "
        "MULTI-ENTRY BACKTEST (READ ONLY)"
    )
    print("=" * 80)

    print(
        "Historical coverage:",
        markets.day.min(),
        "through",
        markets.day.max(),
    )

    print(
        f"Evaluate: {args.scan_start} to "
        f"{args.scan_end} minutes before close, "
        "once each minute"
    )

    print(
        "Rolling alpha: 14 prior available dates "
        "/ at least 10 / 1-day embargo"
    )

    print(
        f"Fee: 7% quadratic; edge >= 2.5c; "
        f"expected ROI >= {args.min_roi:g}%; "
        f"winning ROI >= {args.min_winning_roi:g}%"
    )

    print(
        "Signal quotes: max age 3m, cutoff -1m; "
        "entry: +1m to +3m"
    )
    print(
        f"Max compatible positions/event: "
        f"{args.max_entries}"
    )
    print(
        f"Max research cost/event: "
        f"${args.max_event_cost:.2f}"
    )

    # Each 5-minute horizon bucket has a separate alpha.
    # All training outcomes precede the decision day
    # by at least two calendar days.
    frames = {}
    dates = {}

    for bucket in BUCKETS:
        s = make_snapshots(
            markets,
            candles,
            bucket,
            3,
            1,
            2,
        )

        frames[bucket] = s
        dates[bucket] = (
            sorted(s.day.unique())
            if not s.empty else []
        )

    alphas = ALPHA_SWEEP_CACHE.setdefault(
        args.end.isoformat(), {}
    )

    def alpha_for(day, bucket):
        key = (day, bucket)

        if key not in alphas:
            eligible = [
                d for d in dates[bucket]
                if d <= day - timedelta(days=2)
            ]

            selected = eligible[-14:]

            if len(selected) < 10:
                alphas[key] = None
            else:
                train = frames[bucket][
                    frames[bucket].day.isin(selected)
                ]
                alphas[key] = fit_alpha(train)

        return alphas[key]

    # Build timestamped quote arrays.
    # Explicit nanosecond precision is important.
    books = {}

    for ticker, g in candles.groupby(
        "ticker", sort=False
    ):
        g = g.sort_values("stamp")

        books[ticker] = (
            g["stamp"]
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy(),
            g.bid.to_numpy(dtype=float),
            g.ask.to_numpy(dtype=float),
        )

    mode = getattr(
        args, "momentum_mode", "none"
    )
    minimum_cents = float(
        getattr(args, "momentum_cents", 0)
    )
    lookback = int(
        getattr(args, "momentum_lookback", 3)
    )

    if (
        mode not in ("none", "follow", "fade")
        or minimum_cents < 0
        or not 1 <= lookback <= 10
    ):
        raise ValueError(
            "Invalid momentum settings"
        )

    print(
        f"Momentum: {mode}, "
        f"threshold={minimum_cents:g}c, "
        f"lookback={lookback}m, "
        "quote-age<=1m"
    )

    trades = []
    event_rows = []

    for event, group in markets.groupby(
        "event", sort=False
    ):
        if len(group) != 10:
            continue

        day = group.day.iloc[0]

        if not args.start <= day <= args.end:
            continue

        close_ns = pd.Timestamp(
            group.close.iloc[0]
        ).value

        positions = []
        spent = 0.0
        signals = 0
        last_entry_ns = -1

        # Scan chronologically.
        # Never select an entry using a future quote.
        for minutes_left in range(
            args.scan_start,
            args.scan_end - 1,
            -1,
        ):
            if len(positions) >= args.max_entries:
                break

            decision_ns = (
                close_ns
                - minutes_left * MINUTE_NS
            )

            if decision_ns < last_entry_ns:
                continue

            bucket = (
                minutes_left
                if minutes_left < 5
                else min(
                    60,
                    5 * math.ceil(minutes_left / 5),
                )
            )

            alpha = alpha_for(day, bucket)

            if alpha is None:
                continue

            signal_cutoff = (
                decision_ns - MINUTE_NS
            )

            choices = []

            for row in group.itertuples(index=False):
                quote = asof_quote(
                    books.get(row.ticker),
                    signal_cutoff,
                )

                if quote is None:
                    continue

                bid, ask = quote
                delta_yes = None

                if mode != "none":
                    # Both quotes are available
                    # before the trading decision.
                    current = asof_quote(
                        books.get(row.ticker),
                        signal_cutoff,
                        age=1,
                    )

                    earlier = asof_quote(
                        books.get(row.ticker),
                        signal_cutoff
                        - lookback * MINUTE_NS,
                        age=1,
                    )

                    if (
                        current is None
                        or earlier is None
                    ):
                        continue

                    delta_yes = (
                        current[0] + current[1]
                        - earlier[0] - earlier[1]
                    ) / 2

                model_yes = float(
                    sharpen(
                        [(bid + ask) / 2],
                        alpha,
                    )[0]
                )

                for side, price, fair in (
                    ("YES", ask, model_yes),
                    ("NO", 1 - bid, 1 - model_yes),
                ):
                    if not momentum_passes(
                        delta_yes,
                        side,
                        mode,
                        minimum_cents,
                    ):
                        continue

                    if not 0.05 <= price <= 0.95:
                        continue

                    cost = (
                        price
                        + taker_fee(
                            price, FEE_RATE
                        )
                        + BUFFER
                    )

                    edge = fair - cost

                    if (
                        edge < MIN_EDGE
                        or edge / cost
                        < args.min_roi / 100
                        or (1 - cost) / cost
                        < args.min_winning_roi / 100
                    ):
                        continue

                    leg = {
                        "ticker": row.ticker,
                        "side": side,
                    }

                    if compatible(positions, leg):
                        directed = (
                            100 * (
                                delta_yes
                                if side == "YES"
                                else -delta_yes
                            )
                            if delta_yes is not None
                            else float("nan")
                        )

                        choices.append((
                            edge,
                            row,
                            leg,
                            fair,
                            directed,
                        ))

            signals += len(choices)

            if not choices:
                continue

            # Choose the strongest current edge.
            # Only then inspect the delayed entry price.
            _, row, leg, fair, signed_move = max(
                choices,
                key=lambda x: x[0],
            )

            future = first_later_quote(
                books.get(row.ticker),
                decision_ns + MINUTE_NS,
                min(
                    decision_ns + 3 * MINUTE_NS,
                    close_ns - 1,
                ),
            )

            if future is None:
                continue

            entry_ns, entry_bid, entry_ask = future

            price = (
                entry_ask
                if leg["side"] == "YES"
                else 1 - entry_bid
            )

            if not 0.05 <= price <= 0.95:
                continue

            fee = taker_fee(
                price, FEE_RATE
            )

            capital = price + fee
            buffered_edge = (
                fair - capital - BUFFER
            )

            if (
                buffered_edge < MIN_EDGE
                or buffered_edge / (
                    capital + BUFFER
                ) < args.min_roi / 100
                or (
                    1 - capital - BUFFER
                ) / (
                    capital + BUFFER
                ) < args.min_winning_roi / 100
            ):
                continue

            if (
                spent + capital
                > args.max_event_cost + 1e-9
            ):
                continue

            payout = int(
                row.outcome == 1
                if leg["side"] == "YES"
                else row.outcome == 0
            )

            pnl = (
                payout - capital - BUFFER
            )

            positions.append(leg)
            spent += capital
            last_entry_ns = entry_ns

            trades.append({
                "period": (
                    "RESEARCH"
                    if day <= date(2026, 10, 2)
                    else "VIEWED_OOS"
                ),
                "day": day,
                "event": event,
                "minutes_left": minutes_left,
                "ticker": row.ticker,
                "side": leg["side"],
                "entry": price,
                "capital": capital,
                "fair": fair,
                "win": payout,
                "buffered_pnl": pnl,
                "signal_momentum_cents": signed_move,
                "entry_ns": int(entry_ns),
                "close_ns": int(close_ns),
            })

        event_rows.append({
            "day": day,
            "event": event,
            "entries": len(positions),
            "signals": signals,
        })

    t = pd.DataFrame(trades)
    e = pd.DataFrame(event_rows)

    if e.empty:
        print(
            "No events available with current warmup."
        )
        return

    report = {}

    for period, day_filter in (
        (
            "RESEARCH",
            lambda d: d <= date(2026, 10, 2),
        ),
        (
            "VIEWED_OOS",
            lambda d: d >= date(2026, 10, 3),
        ),
    ):
        subset = e[
            e.day.map(day_filter)
        ]

        picks = (
            t[t.period == period]
            if not t.empty
            else pd.DataFrame()
        )

        capital = (
            float(picks.capital.sum())
            if not picks.empty else 0.0
        )
        pnl = (
            float(picks.buffered_pnl.sum())
            if not picks.empty else 0.0
        )
        win_count = (
            int(picks.win.sum())
            if not picks.empty else 0
        )

        daily_pnl = (
            picks.groupby("day")
            .buffered_pnl.sum()
            .sort_index()
            .to_numpy(dtype=float)
            if not picks.empty
            else np.array([], dtype=float)
        )

        equity = np.r_[
            0.0,
            np.cumsum(daily_pnl),
        ]

        drawdown = float(np.max(
            np.maximum.accumulate(equity)
            - equity
        ))

        report[period] = {
            "trades": len(picks),
            "wins": win_count,
            "capital": capital,
            "pnl": pnl,
            "roi": (
                100 * pnl / capital
                if capital else float("nan")
            ),
            "drawdown": drawdown,
        }

        print(
            f"\n{period} | "
            f"days={subset.day.nunique()} "
            f"events={len(subset)}"
        )

        print(
            f"Signals={int(subset.signals.sum())} "
            f"traded_events="
            f"{(subset.entries > 0).sum()} "
            f"events_with_2plus="
            f"{(subset.entries >= 2).sum()}"
        )

        print(
            f"Trades={len(picks)} "
            f"wins={win_count} "
            f"losses={len(picks) - win_count}"
        )

        if capital:
            print(
                f"Capital=${capital:.2f} "
                f"buffered_PnL=${pnl:+.2f} "
                f"ROI={pnl/capital:+.2%}"
            )
        else:
            print("No entries")

        if not picks.empty:
            distribution = (
                picks.minutes_left
                .floordiv(5)
                .mul(5)
                .value_counts()
                .sort_index()
                .to_dict()
            )

            print(
                "Entries by time-to-close:",
                distribution,
            )

    print(
        "\nNo real orders, no simulated-result "
        "database writes, no CSVs."
    )
    print(
        "Indicative historical prices: "
        "no depth or confirmed fills."
    )

    if getattr(args, "return_trades", False):
        return report, t.copy(), e.copy()

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2026, 9, 10),
    )
    p.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 10, 7),
    )
    p.add_argument(
        "--max-entries",
        type=int,
        default=3,
    )
    p.add_argument(
        "--max-event-cost",
        type=float,
        default=2.50,
    )
    p.add_argument(
        "--scan-start",
        type=int,
        default=60,
    )
    p.add_argument(
        "--scan-end",
        type=int,
        default=5,
    )
    p.add_argument(
        "--min-roi",
        type=float,
        default=5,
        help="Minimum expected ROI in percent",
    )
    p.add_argument(
        "--min-winning-roi",
        type=float,
        default=0,
        help="Minimum return if contract wins, in percent",
    )

    p.add_argument(
        "--momentum-mode",
        choices=("none", "follow", "fade"),
        default="none",
    )
    p.add_argument(
        "--momentum-cents",
        type=float,
        default=0,
    )
    p.add_argument(
        "--momentum-lookback",
        type=int,
        default=3,
    )

    a = p.parse_args()

    if not (
        0 < a.min_roi <= 100
        and 0 <= a.min_winning_roi <= 200
    ):
        p.error(
            "Invalid ROI thresholds"
        )

    if not (
        1 <= a.scan_end
        <= a.scan_start <= 60
    ):
        p.error(
            "Require 1 <= scan-end <= scan-start <= 60"
        )

    if (
        not 1 <= a.max_entries <= 10
        or not 0 < a.max_event_cost <= 10
    ):
        p.error(
            "Entries must be 1..10 and "
            "event cost (0, $10]"
        )

    main(a)
