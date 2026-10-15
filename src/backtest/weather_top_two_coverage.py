"""Read-only top-two YES weather basket study."""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.backtest.hourly_request_limits import hourly_backfill_requests
from src.kalshi.fees import calculate_taker_fee
from src.kalshi.historical_market import (
    get_event_markets,
    load_market_candles,
    parse_candle,
)

ET = "America/New_York"
ENTRY_TIMES = ("14:30", "15:00", "15:30", "16:00")


def valid_quotes(candles):
    quotes = {}
    conflicts = set()

    for candle in candles:
        try:
            q = parse_candle(candle)
            t = q["timestamp"]
            bid, ask = q["yes_bid"], q["yes_ask"]

            if bid is None or ask is None:
                continue
            if not 0 <= bid <= ask <= 1:
                continue

            if t in quotes:
                previous = quotes[t]
                if (
                    previous["yes_bid"], previous["yes_ask"]
                ) != (bid, ask):
                    conflicts.add(t)

            quotes[t] = q
        except (KeyError, TypeError, ValueError):
            continue

    for t in conflicts:
        quotes.pop(t, None)

    return sorted(quotes.values(), key=lambda q: q["timestamp"])


def quote_asof(quotes, decision, max_age):
    eligible = [
        q for q in quotes
        if decision - max_age <= q["timestamp"] <= decision
    ]
    return eligible[-1] if eligible else None


def summarize(groups, title):
    print("\n" + title)
    print(
        f"{'group':>12} {'days':>6} {'hit%':>8} "
        f"{'third%':>8} {'avg cost':>10} "
        f"{'net P&L':>10} {'ROI%':>9}"
    )

    for label, rows in groups:
        if not rows:
            print(f"{label:>12} {0:>6}")
            continue

        n = len(rows)
        hits = sum(r["hit"] for r in rows)
        cost = sum(r["cost"] for r in rows)
        pnl = sum(r["pnl"] for r in rows)

        print(
            f"{label:>12} {n:>6} "
            f"{100*hits/n:>7.1f}% "
            f"{100*(n-hits)/n:>7.1f}% "
            f"${cost/n:>9.3f} "
            f"${pnl:>9.2f} "
            f"{100*pnl/cost if cost else 0:>8.2f}%"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-09-01")
    parser.add_argument("--end", default="2026-10-14")
    parser.add_argument("--max-age-minutes", type=int, default=15)
    args = parser.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()

    if start > end or end >= pd.Timestamp.now(tz=ET).date():
        parser.error("Select past settled dates, start <= end")

    if not 1 <= args.max_age_minutes <= 60:
        parser.error("Max quote age must be 1-60 minutes")

    load_dotenv(Path.cwd() / ".env")

    max_age = pd.Timedelta(minutes=args.max_age_minutes)
    results = []
    skipped = Counter()

    print("READ ONLY: no orders or database writes")
    print(f"KXHIGHNY: {start} through {end}")

    with hourly_backfill_requests(
        interval_seconds=2, max_attempts=6
    ):
        for date in pd.date_range(start, end):
            day = date.strftime("%Y-%m-%d")

            decisions = {
                t: pd.Timestamp(
                    f"{day} {t}", tz=ET
                ).tz_convert("UTC")
                for t in ENTRY_TIMES
            }

            try:
                event = get_event_markets(day)
                markets = sorted(
                    event["markets"], key=lambda m: m["ticker"]
                )
                tickers = [m["ticker"] for m in markets]

                if (
                    len(markets) != 6
                    or len(set(tickers)) != 6
                    or not all(
                        t.startswith(event["event_ticker"] + "-")
                        for t in tickers
                    )
                ):
                    skipped["invalid_event"] += 4
                    print(f"{day}: incomplete event")
                    continue

                outcomes = {
                    m["ticker"]: str(
                        m.get("result", "")
                    ).lower().strip()
                    for m in markets
                }

                winners = [
                    t for t, result in outcomes.items()
                    if result == "yes"
                ]

                if (
                    len(winners) != 1
                    or not set(outcomes.values()) <= {"yes", "no"}
                ):
                    skipped["unsettled"] += 4
                    print(f"{day}: settlement unavailable")
                    continue

                winner = winners[0]
                earliest = min(decisions.values()) - max_age
                latest = max(decisions.values())

                history = {}
                for ticker in tickers:
                    response = load_market_candles(
                        ticker, earliest, latest
                    )
                    history[ticker] = valid_quotes(
                        response["candles"]
                    )

                eligible = 0

                for time_label, decision in decisions.items():
                    snapshot = []

                    for ticker in tickers:
                        q = quote_asof(
                            history[ticker], decision, max_age
                        )

                        if q is None:
                            break

                        snapshot.append({
                            "ticker": ticker,
                            "mid": (
                                q["yes_bid"] + q["yes_ask"]
                            ) / 2,
                            "ask": q["yes_ask"],
                            "bid": q["yes_bid"],
                            "quote_time_et": q["timestamp"].tz_convert(
                                ET
                            ).strftime("%H:%M:%S"),
                            "quote_age_min": round(
                                (decision - q["timestamp"]).total_seconds() / 60,
                                2,
                            ),
                        })

                    if len(snapshot) != 6:
                        skipped[f"missing_{time_label}"] += 1
                        continue

                    top2 = sorted(
                        snapshot,
                        key=lambda r: (-r["mid"], r["ticker"])
                    )[:2]

                    if (
                        top2[-1]["mid"] <= 0
                        or any(r["ask"] <= 0 for r in top2)
                    ):
                        skipped[f"unpriced_{time_label}"] += 1
                        continue

                    cost = sum(
                        r["ask"] + float(
                            calculate_taker_fee(
                                price=r["ask"],
                                contracts=1,
                                multiplier=1,
                            )
                        )
                        for r in top2
                    )

                    hit = int(
                        winner in {r["ticker"] for r in top2}
                    )

                    results.append({
                        "day": day,
                        "time": time_label,
                        "winner": winner,
                        "selected": [r["ticker"] for r in top2],
                        "legs": top2,
                        "cost": cost,
                        "hit": hit,
                        "pnl": hit - cost,
                    })

                    eligible += 1

                print(f"{day}: {eligible}/4 eligible snapshots")

            except Exception as exc:
                skipped["fetch_errors"] += 4
                print(
                    f"{day}: FETCH ERROR "
                    f"{type(exc).__name__}: {exc}"
                )

    summarize(
        [
            (t, [r for r in results if r["time"] == t])
            for t in ENTRY_TIMES
        ],
        "TOP-TWO COVERAGE BY ENTRY TIME",
    )

    bands = [
        ("<80c", lambda c: c < .80),
        ("80-90c", lambda c: .80 <= c < .90),
        ("90-95c", lambda c: .90 <= c < .95),
        ("95-100c", lambda c: .95 <= c < 1),
        (">=100c", lambda c: c >= 1),
    ]

    summarize(
        [
            (label, [r for r in results if test(r["cost"])])
            for label, test in bands
        ],
        "RESULTS BY COMBINED COST (ALL TIMES POOLED)",
    )

    print("\nTHIRD-OUTCOME LOSSES:")
    for r in results:
        if not r["hit"]:
            print(
                f"{r['day']} {r['time']} "
                f"winner={r['winner']} "
                f"selected={','.join(r['selected'])} "
                f"cost=${r['cost']:.3f}"
            )


    print("\nSUB-$1 BASKET OPPORTUNITIES")
    cheap = sorted(
        (r for r in results if r["cost"] < 1),
        key=lambda r: (r["day"], r["time"]),
    )

    for r in cheap:
        print(
            f"\n{r['day']} {r['time']} "
            f"cost=${r['cost']:.3f} "
            f"pnl=${r['pnl']:+.3f} "
            f"winner={r['winner']}"
        )

        for leg in r["legs"]:
            print(
                f"  {leg['ticker']} "
                f"bid={leg['bid']:.3f} "
                f"ask={leg['ask']:.3f} "
                f"mid={leg['mid']:.3f} "
                f"quote_ET={leg['quote_time_et']} "
                f"age={leg['quote_age_min']:.1f}m"
            )

    unique_days = len({r["day"] for r in results})
    cheap_days = len({r["day"] for r in cheap})

    # First qualifying entry each day, avoiding
    # counting repeated intraday snapshots as new bets.
    first_per_day = {}
    for r in cheap:
        first_per_day.setdefault(r["day"], r)

    trades = list(first_per_day.values())
    capital = sum(r["cost"] for r in trades)
    pnl = sum(r["pnl"] for r in trades)
    wins = sum(r["hit"] for r in trades)

    print("\nONE ENTRY MAXIMUM PER DAY")
    print(f"Total eligible settlement days: {unique_days}")
    print(f"Sub-$1 snapshots: {len(cheap)}")
    print(f"Unique sub-$1 dates: {cheap_days}")
    print(f"Trades: {len(trades)}")
    print(f"Wins: {wins}")
    print(f"Losses: {len(trades)-wins}")
    print(f"Capital: ${capital:.2f}")
    print(f"Net P&L: ${pnl:+.2f}")
    print(f"ROI: {100*pnl/capital if capital else 0:+.2f}%")

    print("\nSKIPPED:", dict(skipped))
    print(
        "Note: historical candle asks do not guarantee fills. "
        "ROI includes modeled taker fees, but not slippage."
    )


if __name__ == "__main__":
    main()
