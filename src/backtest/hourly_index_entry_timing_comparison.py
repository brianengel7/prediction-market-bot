"""Read-only KXTEMPNYCHS entry timing comparison."""

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.backtest.hourly_index_market_backtest import (
    load_history,
    make_snapshots,
    fit_alpha,
    sharpen,
    logloss,
    select_trades,
    taker_fee,
)

HORIZONS = (50, 45, 10, 5)
RESEARCH_END = date(2026, 10, 2)


def period_of(day):
    return (
        "RESEARCH"
        if day <= RESEARCH_END
        else "ALREADY_VIEWED_OOS"
    )


def summarize(days, trades, quality, horizon, period):
    d = days[
        (days.horizon == horizon)
        & (days.period == period)
    ]

    t = trades[
        (trades.horizon == horizon)
        & (trades.period == period)
    ]

    q = quality[
        (quality.horizon == horizon)
        & (quality.period == period)
    ]

    capital = float(t.cost.sum())
    pnl = float(t.pnl.sum())
    wins = int(t.won.sum())

    daily_pnl = (
        d.sort_values("day")
        .pnl.to_numpy(dtype=float)
    )

    equity = np.r_[0.0, np.cumsum(daily_pnl)]

    drawdown = float(np.max(
        np.maximum.accumulate(equity) - equity
    ))

    def weighted(column):
        if q.empty:
            return float("nan")

        return float(np.average(
            q[column],
            weights=q["n"],
        ))

    return {
        "period": period,
        "horizon": horizon,
        "days": len(d),
        "events": int(d.events.sum()),
        "signals": int(d.signals.sum()),
        "trades": len(t),
        "wins": wins,
        "win_pct": (
            100 * wins / len(t)
            if len(t) else np.nan
        ),
        "capital": capital,
        "pnl": pnl,
        "roi_pct": (
            100 * pnl / capital
            if capital else np.nan
        ),
        "drawdown": drawdown,
        "raw_ll": weighted("raw_ll"),
        "cal_ll": weighted("cal_ll"),
        "raw_brier": weighted("raw_brier"),
        "cal_brier": weighted("cal_brier"),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=date(2026, 6, 1),
    )

    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=date(2026, 10, 7),
    )

    args = parser.parse_args()

    if args.end > date(2026, 10, 7):
        parser.error(
            "This comparison is capped at October 7. "
            "Evaluate newer dates separately."
        )

    if args.start > args.end:
        parser.error("Start must be before end.")

    print("\nHOURLY NYC ENTRY TIMING COMPARISON")
    print("=" * 100)
    print("Series: KXTEMPNYCHS")
    print("Requested:", args.start, "through", args.end)
    print("Horizon comparison: 50m / 45m / 10m / 5m")
    print("Alpha: last 14 eligible dates, minimum 10")
    print("Outcome embargo: preceding calendar day")
    print("Quote age: 3 minutes maximum")
    print("Entry delay: 1 minute; wait: 2 minutes")
    print("Minimum net edge: 2.5c")
    print("Minimum expected ROI: 5%")
    print("Fee rate: 0.07; buffer: 1c")
    print("Supabase: SELECT only")

    markets, candles = load_history(
        args.end.isoformat()
    )

    print("\nDATA AVAILABILITY")
    print("Earliest event date:", markets.day.min())
    print("Latest event date:", markets.day.max())
    print("Hourly events:", markets.event.nunique())
    print("Threshold contracts:", len(markets))
    print("Valid minute quotes:", len(candles))

    daily = []
    trades = []
    quality = []

    for horizon in HORIZONS:
        print(
            f"\nProcessing {horizon}-minute strategy...",
            flush=True,
        )

        snap = make_snapshots(
            markets,
            candles,
            horizon,
            3,
            1,
            2,
        )

        if snap.empty:
            print("No valid quote snapshots.")
            continue

        days = sorted(snap.day.unique())

        for day in days:
            if not args.start <= day <= args.end:
                continue

            eligible = [
                d for d in days
                if d <= day - timedelta(days=2)
            ]

            train_days = eligible[-14:]

            if len(train_days) < 10:
                continue

            train = snap[
                snap.day.isin(train_days)
            ]

            today = snap[
                snap.day == day
            ]

            alpha = fit_alpha(train)

            calibrated = sharpen(
                today.mid,
                alpha,
            )

            period = period_of(day)

            quality.append({
                "period": period,
                "horizon": horizon,
                "day": day,
                "n": len(today),
                "alpha": alpha,
                "raw_ll": logloss(
                    today.outcome,
                    today.mid,
                ),
                "cal_ll": logloss(
                    today.outcome,
                    calibrated,
                ),
                "raw_brier": float(np.mean(
                    (today.mid - today.outcome) ** 2
                )),
                "cal_brier": float(np.mean(
                    (calibrated - today.outcome) ** 2
                )),
            })

            candidates = select_trades(
                today,
                alpha,
                0.07,
                0.025,
                0.05,
                0.01,
            )

            filled = 0
            daily_pnl = 0.0

            for row in candidates.itertuples(index=False):
                if not row.entry_possible:
                    continue

                price = (
                    row.entry_ask_yes
                    if row.side == "YES"
                    else 1 - row.entry_bid
                )

                if (
                    pd.isna(price)
                    or not 0.05 <= price <= 0.95
                ):
                    continue

                price = float(price)
                fee = taker_fee(price, 0.07)

                fair = float(
                    row.fair_yes
                    if row.side == "YES"
                    else 1 - row.fair_yes
                )

                cost = price + fee
                edge = fair - cost - 0.01
                expected_roi = edge / (cost + 0.01)

                if (
                    edge < 0.025
                    or expected_roi < 0.05
                ):
                    continue

                payout = float(
                    row.outcome
                    if row.side == "YES"
                    else 1 - row.outcome
                )

                pnl = payout - cost - 0.01

                trades.append({
                    "period": period,
                    "horizon": horizon,
                    "day": day,
                    "event": row.event,
                    "ticker": row.ticker,
                    "side": row.side,
                    "entry_at": row.entry_quote_time,
                    "price": price,
                    "fee": fee,
                    "cost": cost,
                    "fair": fair,
                    "edge": edge,
                    "won": int(payout == 1),
                    "pnl": pnl,
                })

                filled += 1
                daily_pnl += pnl

            daily.append({
                "period": period,
                "horizon": horizon,
                "day": day,
                "events": today.event.nunique(),
                "signals": len(candidates),
                "trades": filled,
                "pnl": daily_pnl,
            })

    d = pd.DataFrame(daily)

    t = pd.DataFrame(trades, columns=[
        "period", "horizon", "day", "event",
        "ticker", "side", "entry_at", "price",
        "fee", "cost", "fair", "edge",
        "won", "pnl",
    ])

    q = pd.DataFrame(quality)

    if d.empty:
        print("\nNo scored days available.")
        return

    summary = pd.DataFrame([
        summarize(d, t, q, horizon, period)
        for period in (
            "RESEARCH",
            "ALREADY_VIEWED_OOS",
        )
        for horizon in HORIZONS
    ])

    print("\n" + "=" * 100)
    print("INDIVIDUAL STRATEGY PERFORMANCE")
    print("=" * 100)

    print(summary[[
        "period",
        "horizon",
        "days",
        "events",
        "signals",
        "trades",
        "wins",
        "win_pct",
        "capital",
        "pnl",
        "roi_pct",
        "drawdown",
    ]].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    ))

    print("\nBASELINE REPRODUCTION CHECK")

    for horizon, prior in ((45, 30), (5, 26)):
        observed = int(summary[
            (summary.period == "RESEARCH")
            & (summary.horizon == horizon)
        ].trades.iloc[0])

        print(
            f"{horizon}m: current={observed}, "
            f"previous={prior}, "
            f"{'MATCH' if observed == prior else 'CHANGED'}"
        )

    print("\nPROBABILITY CALIBRATION")

    print(summary[[
        "period",
        "horizon",
        "raw_ll",
        "cal_ll",
        "raw_brier",
        "cal_brier",
    ]].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    ))

    if not t.empty:
        print("\n" + "=" * 100)
        print("MATCHED EVENT COMPARISON")
        print("=" * 100)

        for period in (
            "RESEARCH",
            "ALREADY_VIEWED_OOS",
        ):
            for earlier, existing in (
                (50, 45),
                (10, 5),
            ):
                a = t[
                    (t.period == period)
                    & (t.horizon == earlier)
                ].set_index("event")

                b = t[
                    (t.period == period)
                    & (t.horizon == existing)
                ].set_index("event")

                shared = a.index.intersection(b.index)

                print(
                    f"{period} | {earlier}m vs {existing}m "
                    f"| earlier trades={len(a)} "
                    f"| existing trades={len(b)} "
                    f"| shared events={len(shared)} "
                    f"| earlier-only={len(a)-len(shared)} "
                    f"| existing-only={len(b)-len(shared)}"
                )

                if len(shared):
                    pa = float(
                        a.loc[shared, "pnl"].sum()
                    )
                    pb = float(
                        b.loc[shared, "pnl"].sum()
                    )

                    print(
                        f"  Shared-event P&L: "
                        f"earlier=${pa:+.2f}, "
                        f"existing=${pb:+.2f}, "
                        f"difference=${pa-pb:+.2f}"
                    )

        print("\n" + "=" * 100)
        print("COMBINED STRATEGY PERFORMANCE")
        print("One indicative entry per hourly event")
        print("=" * 100)

        for period in (
            "RESEARCH",
            "ALREADY_VIEWED_OOS",
        ):
            for name, horizons in (
                ("Existing 45m + 5m", (45, 5)),
                ("Earlier 50m + 10m", (50, 10)),
            ):
                eligible = t[
                    (t.period == period)
                    & (t.horizon.isin(horizons))
                ]

                selected = (
                    eligible
                    .sort_values("entry_at")
                    .drop_duplicates(
                        "event",
                        keep="first",
                    )
                )

                capital = float(selected.cost.sum())
                pnl = float(selected.pnl.sum())

                roi = (
                    100 * pnl / capital
                    if capital else float("nan")
                )

                print(
                    f"{period:19} | {name:19} "
                    f"| trades={len(selected):3d} "
                    f"| P&L=${pnl:+.2f} "
                    f"| ROI={roi:+.2f}%"
                )

        print("\nNEW STRATEGY RESEARCH TRADES")

        new = t[
            (t.period == "RESEARCH")
            & (t.horizon.isin((50, 10)))
        ]

        if new.empty:
            print("No qualifying trades.")
        else:
            print(
                new[[
                    "day",
                    "horizon",
                    "ticker",
                    "side",
                    "price",
                    "fair",
                    "won",
                    "pnl",
                ]]
                .sort_values(["day", "horizon"])
                .to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    print("\nCOMPLETE")
    print("No live orders submitted.")
    print("No database writes or CSV files.")
    print("Historical asks are not guaranteed fills.")
    print(
        "October 3-7 is previously viewed data, "
        "not a fresh independent holdout."
    )


if __name__ == "__main__":
    main()
