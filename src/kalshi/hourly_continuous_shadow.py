"""Continuous NYC hourly temperature shadow trader.

Paper only. No order submission.
Scans 30–5 minutes before close, once per minute.
Stores forward observations in local JSONL.
"""
import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from src.backtest.hourly_index_market_backtest import (
    fit_alpha,
    load_history,
    make_snapshots,
    sharpen,
    taker_fee,
)
from src.kalshi import hourly_index_live_core as core
from src.database.hourly_shadow_journal import SupabaseJournal


MIN_EDGE = 0.025
MIN_EXPECTED_ROI = 0.05
MIN_WINNING_ROI = 0.30
BUFFER = 0.01
FEE_RATE = 0.07

BUCKETS = (5, 10, 15, 20, 25, 30)
STRATEGY = "HOURLY_CONTINUOUS_30_5_V1"


def now_utc():
    return datetime.now(timezone.utc)


def economics(fair, price):
    if not 0.05 <= price <= 0.95:
        return None

    fee = taker_fee(price, FEE_RATE)
    risked = price + fee + BUFFER
    edge = fair - risked

    return {
        "price": float(price),
        "fee": float(fee),
        "risked": float(risked),
        "fair": float(fair),
        "edge": float(edge),
        "expected_roi": float(edge / risked),
        "winning_roi": float((1 - risked) / risked),
    }


def rejection(metric):
    if metric is None:
        return "PRICE_OUT_OF_RANGE"
    if metric["edge"] < MIN_EDGE - 1e-9:
        return "LOW_EDGE"
    if metric["expected_roi"] < MIN_EXPECTED_ROI - 1e-9:
        return "LOW_EXPECTED_ROI"
    if metric["winning_roi"] < MIN_WINNING_ROI - 1e-9:
        return "LOW_WINNING_ROI"
    return "QUALIFIED"


def rank_candidates(quotes, alpha):
    evaluations = []
    passing = []

    for row in quotes.itertuples(index=False):
        ticker = str(row.ticker)
        bid = float(row.signal_bid)
        ask = float(row.signal_ask)

        if not 0 <= bid <= ask <= 1:
            continue

        fair_yes = float(
            sharpen([(bid + ask) / 2], alpha)[0]
        )

        for side, price, fair in (
            ("YES", ask, fair_yes),
            ("NO", 1 - bid, 1 - fair_yes),
        ):
            metric = economics(fair, price)
            reason = rejection(metric)

            entry = {
                "ticker": ticker,
                "side": side,
                "signal_bid": bid,
                "signal_ask": ask,
                "fair": fair,
                "reason": reason,
            }

            if metric is not None:
                entry.update(metric)

            evaluations.append(entry)

            if reason == "QUALIFIED":
                passing.append(entry)

    best = (
        max(passing, key=lambda x: x["edge"])
        if passing else None
    )

    return best, evaluations


def prepare_alphas(target_day):
    cutoff = target_day - timedelta(days=2)
    markets, candles = load_history(cutoff.isoformat())

    result = {}

    for bucket in BUCKETS:
        snapshots = make_snapshots(
            markets, candles, bucket, 3, 1, 2
        )

        if snapshots.empty:
            raise RuntimeError(
                f"No historical snapshots for {bucket}m"
            )

        eligible = sorted(
            d for d in snapshots.day.unique()
            if d <= cutoff
        )

        dates = eligible[-14:]

        if len(dates) < 10:
            raise RuntimeError(
                f"{bucket}m: only {len(dates)} calibration dates"
            )

        if dates[-1] < target_day - timedelta(days=3):
            raise RuntimeError(
                f"{bucket}m: stale calibration {dates[-1]}"
            )

        training = snapshots[
            snapshots.day.isin(dates)
        ]

        result[bucket] = {
            "alpha": float(fit_alpha(training)),
            "train_first": str(dates[0]),
            "train_last": str(dates[-1]),
            "train_dates": len(dates),
            "train_snapshots": len(training),
        }

    return result



def handle_pending(pending, journal, timestamp):
    for event, signal in list(pending.items()):
        if timestamp < signal["entry_after"]:
            continue

        expired = (
            timestamp > signal["deadline"]
            or (
                signal["close"] - timestamp
            ).total_seconds() <= 15
        )

        if expired:
            journal.record(
                "ENTRY_REJECTED",
                event=event,
                ticker=signal["ticker"],
                side=signal["side"],
                reason="DELAY_WINDOW_EXPIRED",
            )
            print("EXPIRED", event, flush=True)
            del pending[event]
            continue

        try:
            book = core.book_ask(
                signal["ticker"], signal["side"]
            )
        except Exception as exc:
            print(
                "BOOK_ERROR", event, str(exc),
                flush=True,
            )
            continue

        if book is None:
            if timestamp >= signal["next_no_book_log"]:
                journal.record(
                    "ENTRY_WAIT",
                    event=event,
                    ticker=signal["ticker"],
                    side=signal["side"],
                    reason="NO_BOOK_DEPTH",
                )
                signal["next_no_book_log"] = (
                    timestamp + timedelta(seconds=20)
                )
            continue

        price, depth = map(float, book)

        metric = economics(signal["fair"], price)
        reason = rejection(metric)

        if depth < 1:
            reason = "INSUFFICIENT_DEPTH"

        journal.record(
            "ENTRY_CHECK",
            event=event,
            ticker=signal["ticker"],
            side=signal["side"],
            observed_ask=price,
            depth=depth,
            economics=metric,
            result=reason,
        )

        if reason == "QUALIFIED":
            journal.record(
                "PAPER_ENTRY",
                event=event,
                ticker=signal["ticker"],
                side=signal["side"],
                decision_utc=signal["decision"].isoformat(),
                entry_observed_utc=timestamp.isoformat(),
                close_utc=signal["close"].isoformat(),
                minutes_left=signal["minutes_left"],
                bucket=signal["bucket"],
                alpha=signal["alpha"],
                signal_fair=signal["fair"],
                contracts=1,
                displayed_depth=depth,
                price=price,
                fee=metric["fee"],
                buffer=BUFFER,
                risked=metric["risked"],
                expected_roi=metric["expected_roi"],
                winning_roi=metric["winning_roi"],
                expected_edge=metric["edge"],
            )

            print(
                "PAPER_ENTRY", event,
                signal["ticker"], signal["side"],
                f"ask={price:.2f}",
                f"depth={depth:g}",
                f"expected_roi={metric['expected_roi']:.1%}",
                flush=True,
            )

        else:
            journal.record(
                "ENTRY_REJECTED",
                event=event,
                ticker=signal["ticker"],
                side=signal["side"],
                reason=reason,
                economics=metric,
            )
            print(
                "FADED", event, reason, flush=True
            )

        del pending[event]


def sellable_bid(ticker, side):
    """Read current bid for a position we hypothetically own."""
    response = core.get(
        f"/markets/{ticker}/orderbook",
        {"depth": 1},
    )

    book = response.get("orderbook_fp") or {}
    field = (
        "yes_dollars" if side == "YES"
        else "no_dollars"
    )
    levels = book.get(field) or []

    if not levels:
        return None

    bid, depth = max(
        levels, key=lambda x: float(x[0])
    )

    bid, depth = float(bid), float(depth)

    if not 0 < bid < 1 or depth < 1:
        return None

    return bid, depth


def observe_resale_bids(journal, timestamp):
    """Record possible exit prices. Never sell."""
    for event, position in list(
        journal.open_positions.items()
    ):
        if timestamp >= core.parse_time(
            position["close_utc"]
        ):
            continue

        try:
            quote = sellable_bid(
                position["ticker"], position["side"]
            )

            if quote is None:
                journal.record(
                    "RESALE_MARK",
                    event=event,
                    ticker=position["ticker"],
                    side=position["side"],
                    bid=None,
                    depth=0,
                    reason="NO_DISPLAYED_BID",
                )
                continue

            bid, depth = quote
            exit_fee = taker_fee(bid, FEE_RATE)

            entry_cost = (
                float(position["price"])
                + float(position["fee"])
            )

            indicative_pnl = (
                bid - exit_fee - entry_cost
            )

            journal.record(
                "RESALE_MARK",
                event=event,
                ticker=position["ticker"],
                side=position["side"],
                bid=bid,
                depth=depth,
                entry_cash_cost=entry_cost,
                estimated_exit_fee=exit_fee,
                indicative_exit_pnl=indicative_pnl,
                indicative_exit_roi=(
                    indicative_pnl / entry_cost
                ),
            )

        except Exception as exc:
            journal.record(
                "RESALE_MARK_ERROR",
                event=event,
                ticker=position["ticker"],
                error=str(exc),
            )


def check_settlements(journal, timestamp):
    for event, position in list(
        journal.open_positions.items()
    ):
        if timestamp < core.parse_time(
            position["close_utc"]
        ):
            continue

        try:
            market = core.get_market_settlement(
                position["ticker"]
            )

            result = str(
                market.get("result", "")
            ).upper()

            status = str(
                market.get("status", "")
            ).lower()

            if (
                result not in ("YES", "NO")
                or status not in ("settled", "finalized")
            ):
                continue

        except Exception as exc:
            print(
                "SETTLEMENT_ERROR", event, str(exc),
                flush=True,
            )
            continue

        won = result == position["side"]
        payout = 1 if won else 0

        risked = float(position["risked"])
        cash_cost = (
            float(position["price"])
            + float(position["fee"])
        )

        journal.record(
            "SETTLED",
            event=event,
            ticker=position["ticker"],
            side=position["side"],
            actual_result=result,
            won=won,
            payout=payout,
            risked=risked,
            buffered_pnl=payout - risked,
            cash_pnl=payout - cash_cost,
        )

        print(
            "SETTLED", event,
            "WIN" if won else "LOSS",
            f"buffered_pnl=${payout-risked:+.2f}",
            flush=True,
        )


def valid_events(events):
    for event, markets in events.items():
        if len(markets) != 10:
            continue

        closes = {
            market.get("close_time")
            for market in markets
        }

        if len(closes) != 1:
            continue

        close = core.parse_time(
            markets[0]["close_time"]
        )

        yield event, markets, close


def process_scans(
    events, pending, journal, seen, alphas, timestamp
):
    for event, markets, close in valid_events(events):
        if event in journal.traded or event in pending:
            continue

        remaining = (
            close - timestamp
        ).total_seconds() / 60

        minutes_left = math.ceil(remaining)

        if not 5 <= minutes_left <= 30:
            continue

        decision = (
            close - timedelta(minutes=minutes_left)
        )

        lateness = (
            timestamp - decision
        ).total_seconds()

        key = (event, minutes_left)

        if key in seen:
            continue

        seen.add(key)

        if not 0 <= lateness <= 30:
            journal.record(
                "MISSED_MINUTE",
                event=event,
                minutes_left=minutes_left,
                seconds_late=lateness,
            )
            continue

        day = close.astimezone(
            core.ET
        ).date()

        bucket = 5 * math.ceil(
            minutes_left / 5
        )

        info = alphas.get(day, {}).get(bucket)

        if info is None:
            journal.record(
                "SKIP",
                event=event,
                minutes_left=minutes_left,
                reason="ALPHA_UNAVAILABLE",
            )
            continue

        try:
            quotes = core.signal_quotes(
                markets, decision
            )

            if quotes.empty:
                best, evaluated = None, []
            else:
                best, evaluated = rank_candidates(
                    quotes, info["alpha"]
                )

        except Exception as exc:
            journal.record(
                "ERROR",
                event=event,
                minutes_left=minutes_left,
                operation="SIGNAL_QUOTES",
                error=str(exc),
            )
            print(
                "SIGNAL_ERROR", event, str(exc),
                flush=True,
            )
            continue

        journal.record(
            "SCAN",
            event=event,
            decision_utc=decision.isoformat(),
            minutes_left=minutes_left,
            bucket=bucket,
            alpha=info["alpha"],
            contracts_quoted=len(quotes),
            evaluated=evaluated,
            qualifying_sides=sum(
                x["reason"] == "QUALIFIED"
                for x in evaluated
            ),
            best=(
                {
                    "ticker": best["ticker"],
                    "side": best["side"],
                    "edge": best["edge"],
                }
                if best else None
            ),
        )

        if best is None:
            continue

        pending[event] = {
            "ticker": best["ticker"],
            "side": best["side"],
            "fair": best["fair"],
            "alpha": info["alpha"],
            "day": day,
            "bucket": bucket,
            "minutes_left": minutes_left,
            "decision": decision,
            "close": close,
            "entry_after": (
                decision + timedelta(minutes=1)
            ),
            "deadline": min(
                decision + timedelta(minutes=3),
                close - timedelta(seconds=1),
            ),
            "next_no_book_log": timestamp,
        }

        journal.record(
            "SIGNAL",
            event=event,
            ticker=best["ticker"],
            side=best["side"],
            decision_utc=decision.isoformat(),
            minutes_left=minutes_left,
            bucket=bucket,
            alpha=info["alpha"],
            fair=best["fair"],
            signal_edge=best["edge"],
            signal_expected_roi=best["expected_roi"],
            signal_winning_roi=best["winning_roi"],
        )

        print(
            "SIGNAL", event,
            best["ticker"], best["side"],
            f"edge={best['edge']:.3f}",
            f"expected_roi={best['expected_roi']:.1%}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--once", action="store_true"
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=5
    )
    parser.add_argument(
        "--mark-seconds", type=float, default=30
    )

    args = parser.parse_args()

    if not 2 <= args.poll_seconds <= 15:
        parser.error("poll-seconds must be 2..15")

    if not 10 <= args.mark_seconds <= 300:
        parser.error("mark-seconds must be 10..300")

    print(
        "\nHOURLY CONTINUOUS SHADOW — PAPER ONLY"
    )
    print(
        "30–5 min | edge 2.5c | expected ROI 5% "
        "| winning ROI 30%"
    )
    print(
        "One simulated contract and entry per event."
    )
    print("Paper observations: Supabase")

    core.fee_check()
    events = core.open_events()

    print("Open events:", len(events))

    if args.once:
        future = [
            (
                close.astimezone(core.ET).date(),
                event,
            )
            for event, markets, close
            in valid_events(events)
            if close > now_utc()
        ]

        if not future:
            print("No future complete events found.")
            return

        day, event = min(future)
        alphas = prepare_alphas(day)

        print("\nCALIBRATION CHECK", day, event)

        for bucket in BUCKETS:
            row = alphas[bucket]
            print(
                f"{bucket:2}m "
                f"alpha={row['alpha']:.3f} "
                f"training={row['train_first']}"
                f"..{row['train_last']} "
                f"dates={row['train_dates']}"
            )

        print("READ-ONLY CHECK COMPLETE")
        return

    journal = SupabaseJournal()

    print(
        "Recovered traded events:",
        len(journal.traded),
        "unsettled paper positions:",
        len(journal.open_positions),
    )

    pending = {}
    seen = set()
    alpha_cache = {}
    next_retry = {}

    last_discovery = 0.0
    last_settlement = 0.0
    last_mark = 0.0
    last_heartbeat = 0.0

    while True:
        stamp = now_utc()
        tick = time.monotonic()

        if tick - last_discovery >= 20:
            try:
                events = core.open_events()
            except Exception as exc:
                print(
                    "DISCOVERY_ERROR", str(exc),
                    flush=True,
                )
            last_discovery = tick

        days = sorted({
            close.astimezone(core.ET).date()
            for event, markets, close
            in valid_events(events)
            if stamp < close <= (
                stamp + timedelta(hours=3)
            )
        })

        for day in days:
            if (
                day in alpha_cache
                or tick < next_retry.get(day, 0)
            ):
                continue

            try:
                alpha_cache[day] = prepare_alphas(day)

                for bucket, info in (
                    alpha_cache[day].items()
                ):
                    journal.record(
                        "ALPHA",
                        day=str(day),
                        bucket=bucket,
                        **info,
                    )

                print(
                    "ALPHAS", day,
                    {
                        k: round(v["alpha"], 3)
                        for k, v in alpha_cache[day].items()
                    },
                    flush=True,
                )

            except Exception as exc:
                print(
                    "ALPHA_ERROR", day, str(exc),
                    flush=True,
                )
                journal.record(
                    "ALPHA_ERROR",
                    day=str(day),
                    error=str(exc),
                )
                next_retry[day] = tick + 900

        handle_pending(
            pending, journal, stamp
        )

        process_scans(
            events,
            pending,
            journal,
            seen,
            alpha_cache,
            stamp,
        )

        if tick - last_settlement >= 60:
            check_settlements(
                journal, stamp
            )
            last_settlement = tick

        if tick - last_mark >= args.mark_seconds:
            observe_resale_bids(
                journal, stamp
            )
            last_mark = tick

        if tick - last_heartbeat >= 300:
            print(
                "HEARTBEAT CONTINUOUS",
                stamp.isoformat(),
                f"pending={len(pending)}",
                f"paper_open={len(journal.open_positions)}",
                f"traded_events={len(journal.traded)}",
                flush=True,
            )
            last_heartbeat = tick

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
