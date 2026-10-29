#!/usr/bin/env python3
"""Read-only NYC KXHIGHNY historical 1-minute candlestick collector.

Standalone: does not import or modify the prediction-market-bot repository.
Requires: pip install requests

Usage:
  python3 collect_kalshi_nyc_1m.py --first-days 2
  python3 collect_kalshi_nyc_1m.py

Writes only local research data. Makes only HTTP GET requests (no auth/orders).
"""

import argparse
import csv
import gzip
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
NY = ZoneInfo("America/New_York")
SERIES = "KXHIGHNY"
PRICE_PARTS = ("yes_bid", "yes_ask", "price")
OHLC = ("open", "high", "low", "close")
FIELDS = [
    "event_date", "event_ticker", "ticker", "source", "end_period_ts",
    "end_utc", "end_ny", "volume", "open_interest",
] + [f"{part}_{key}" for part in PRICE_PARTS for key in OHLC] + [
    "price_mean", "price_previous"
]


def utc_stamp(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_gz(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(obj, handle, separators=(",", ":"))
    os.replace(tmp, path)


def get(session, path, params=None, delay=0.30):
    url = BASE + path
    for attempt in range(7):
        time.sleep(delay)
        try:
            response = session.get(url, params=params, timeout=40)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 6:
                raise
            print(f"  RETRY network {type(exc).__name__}: {path}", flush=True)
            time.sleep(min(2 ** attempt, 30))
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt == 6:
                response.raise_for_status()
            value = response.headers.get("Retry-After", "")
            try:
                wait = max(1, min(90, float(value)))
            except ValueError:
                wait = min(2 ** (attempt + 1), 60)
            print(f"  RETRY HTTP {response.status_code} after {wait:g}s", flush=True)
            time.sleep(wait)
            continue
        if response.status_code in (404, 410):
            return None
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Retries exhausted")


def paged_markets(session, event, historical, delay):
    endpoint = "/historical/markets" if historical else "/markets"
    markets = []
    cursor = ""
    seen_cursors = set()
    while True:
        params = {"event_ticker": event, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        payload = get(session, endpoint, params, delay)
        if payload is None:
            return markets
        batch = payload.get("markets", [])
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected market result for {event}")
        for market in batch:
            if market.get("event_ticker") != event:
                raise RuntimeError(f"Unexpected event: {market.get('event_ticker')} != {event}")
            ticker = market.get("ticker", "")
            if not ticker.startswith(event + "-"):
                raise RuntimeError(f"Unexpected ticker: {ticker}")
            markets.append(market)
        cursor = payload.get("cursor") or ""
        if not cursor:
            break
        if cursor in seen_cursors:
            raise RuntimeError(f"Repeated pagination cursor: {event}")
        seen_cursors.add(cursor)
    return markets


def event_markets(session, event, delay):
    # Query both partitions; recent and older settled markets live in different datasets.
    result = {}
    for historical, label in ((True, "historical"), (False, "live")):
        for market in paged_markets(session, event, historical, delay):
            ticker = market["ticker"]
            previous = result.get(ticker)
            if previous is not None and previous["source"] == "historical":
                continue
            result[ticker] = {"source": label, "market": market}
    return result


def price_value(part, field):
    if not isinstance(part, dict):
        return None
    value = part.get(field + "_dollars")
    if value is None:
        value = part.get(field)
    if value in (None, ""):
        return None
    # Legacy non-_dollars fields must be dollar strings; reject ambiguous integers.
    if not isinstance(value, str):
        raise ValueError(f"Unexpected numeric price {field}: {value!r}")
    price = float(value)
    if not 0 <= price <= 1:
        raise ValueError(f"Price outside [0,1]: {value!r}")
    return value


def row_for(day, event, ticker, source, candle):
    ts = int(candle["end_period_ts"])
    time_utc = datetime.fromtimestamp(ts, timezone.utc)
    row = {
        "event_date": day.isoformat(), "event_ticker": event,
        "ticker": ticker, "source": source, "end_period_ts": ts,
        "end_utc": time_utc.isoformat(),
        "end_ny": time_utc.astimezone(NY).isoformat(),
        "volume": candle.get("volume_fp", candle.get("volume")),
        "open_interest": candle.get("open_interest_fp", candle.get("open_interest")),
    }
    for part in PRICE_PARTS:
        values = candle.get(part) or {}
        for field in OHLC:
            row[f"{part}_{field}"] = price_value(values, field)
        if part == "price":
            for field in ("mean", "previous"):
                row[f"price_{field}"] = price_value(values, field)
    return row


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def append_error(root, day, ticker, error):
    with (root / "errors.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "date": day.isoformat(), "ticker": ticker, "error": str(error)
        }) + "\n")


def fetch_candles(session, ticker, source, start_ts, end_ts, delay):
    paths = (
        [("historical", f"/historical/markets/{ticker}/candlesticks"),
         ("live", f"/series/{SERIES}/markets/{ticker}/candlesticks")]
        if source == "historical" else
        [("live", f"/series/{SERIES}/markets/{ticker}/candlesticks"),
         ("historical", f"/historical/markets/{ticker}/candlesticks")]
    )
    for label, path in paths:
        payload = get(session, path, {
            "start_ts": start_ts, "end_ts": end_ts, "period_interval": 1,
        }, delay)
        if payload is not None:
            if payload.get("ticker", ticker) != ticker:
                raise RuntimeError(f"Unexpected candle ticker for {ticker}")
            if not isinstance(payload.get("candlesticks"), list):
                raise RuntimeError(f"Malformed candlestick payload for {ticker}")
            return label, payload
    raise RuntimeError(f"Neither candlestick endpoint found {ticker}")


def iso_day_bounds(day):
    begin = datetime(day.year, day.month, day.day, tzinfo=NY)
    following = day + timedelta(days=1)
    finish = datetime(following.year, following.month, following.day, tzinfo=NY)
    return int(begin.timestamp()), int(finish.timestamp())


def collect(args):
    first = date.fromisoformat(args.start)
    last = date.fromisoformat(args.end)
    if last < first:
        raise ValueError("--end precedes --start")
    days = (last - first).days + 1
    count_days = min(days, args.first_days) if args.first_days else days
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "NYC-observation-research-collector/1.0"})
    cutoff = get(session, "/historical/cutoff", delay=args.delay)
    atomic_json(root / "historical_cutoff.json", cutoff)
    print("Historical cutoff:", cutoff.get("market_settled_ts", "unknown"), flush=True)

    coverage = []
    missing_days = []
    for offset in range(count_days):
        day = first + timedelta(days=offset)
        event = SERIES + "-" + day.strftime("%y%b%d").upper()
        manifest_path = root / "markets" / f"{day}.json"
        if manifest_path.exists() and not args.refresh_markets:
            markets = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            try:
                markets = event_markets(session, event, args.delay)
                if markets:  # Don't permanently cache a failed/empty discovery.
                    atomic_json(manifest_path, markets)
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                print(f"ERROR discovering {event}: {exc}", flush=True)
                append_error(root, day, "MARKET_DISCOVERY", exc)
                missing_days.append(day.isoformat())
                continue
        if not markets:
            print(f"WARNING {event}: no markets returned", flush=True)
            missing_days.append(day.isoformat())
            continue
        print(f"[{offset + 1}/{count_days}] {event}: {len(markets)} buckets", flush=True)
        begin_ts, end_ts = iso_day_bounds(day)
        for j, (ticker, metadata) in enumerate(sorted(markets.items()), start=1):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", ticker):
                append_error(root, day, ticker, "Unsafe ticker format")
                continue
            path = root / "candles" / day.isoformat() / (ticker + ".csv")
            raw_path = root / "raw" / day.isoformat() / (ticker + ".json.gz")
            try:
                if path.exists() and raw_path.exists() and not args.overwrite:
                    with path.open(newline="", encoding="utf-8") as handle:
                        saved = list(csv.DictReader(handle))
                    source = metadata["source"]
                    rows = saved
                    skipped = " cached"
                else:
                    source, payload = fetch_candles(
                        session, ticker, metadata["source"], begin_ts, end_ts, args.delay
                    )
                    candles = payload["candlesticks"]
                    # End timestamps represent completed minute intervals.
                    candles = [c for c in candles if begin_ts < int(c["end_period_ts"]) <= end_ts]
                    candles.sort(key=lambda c: int(c["end_period_ts"]))
                    times = [int(c["end_period_ts"]) for c in candles]
                    if len(times) != len(set(times)):
                        raise RuntimeError("Duplicate candle end_period_ts")
                    rows = [row_for(day, event, ticker, source, c) for c in candles]
                    atomic_gz(raw_path, {"ticker": ticker, "source": source,
                                         "range": [begin_ts, end_ts], "candlesticks": candles})
                    save_csv(path, rows)
                    skipped = ""
                quoted = sum(bool(r.get("yes_bid_close")) and bool(r.get("yes_ask_close")) for r in rows)
                coverage.append({
                    "date": day.isoformat(), "ticker": ticker, "source": source,
                    "candles": len(rows), "two_sided_close": quoted,
                    "first_end_utc": rows[0]["end_utc"] if rows else "",
                    "last_end_utc": rows[-1]["end_utc"] if rows else "",
                })
                print(f"  {j}/{len(markets)} {ticker}: {len(rows)} minutes, {quoted} both-sided{skipped}", flush=True)
            except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
                print(f"  ERROR {ticker}: {exc}", flush=True)
                append_error(root, day, ticker, exc)
                continue
        # Coverage is overwritten after each date so interruption retains progress.
        with (root / "coverage.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "date", "ticker", "source", "candles", "two_sided_close",
                "first_end_utc", "last_end_utc"
            ])
            writer.writeheader()
            writer.writerows(coverage)

    print("\nCOLLECTION SUMMARY", flush=True)
    print("Dates requested:", count_days, flush=True)
    print("Contracts processed:", len(coverage), flush=True)
    print("Total one-minute candles:", sum(x["candles"] for x in coverage), flush=True)
    print("Missing-market dates:", ", ".join(missing_days) if missing_days else "none", flush=True)
    print("Files stored in:", root.resolve(), flush=True)
    if (root / "errors.jsonl").exists():
        print("Inspect errors.jsonl for failed requests; rerun the same command to retry.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-10-26")
    parser.add_argument("--first-days", type=int, default=None, help="Try just the first N days")
    parser.add_argument("--delay", type=float, default=0.30, help="Minimum seconds between GETs")
    parser.add_argument("--output", default="data/research/nyc_observation_20260701_20261026/kalshi")
    parser.add_argument("--overwrite", action="store_true", help="Re-download already saved candle files")
    parser.add_argument("--refresh-markets", action="store_true", help="Refresh saved market lists")
    args = parser.parse_args()
    if args.first_days is not None and args.first_days <= 0:
        parser.error("--first-days must be positive")
    if args.delay < 0.2:
        parser.error("--delay must be >=0.2 seconds")
    collect(args)


if __name__ == "__main__":
    main()
