"""Historical NBA sportsbook -> Kalshi convergence backtest.

Research only. No order placement.

Commands:
    python3 -m src.sports.research.nba_historical odds ...
    python3 -m src.sports.research.nba_historical kalshi
    python3 -m src.sports.research.nba_historical backtest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


ROOT = Path("data/research/sports_nba_lead_lag")

KALSHI = "https://external-api.kalshi.com/trade-api/v2"
ODDS = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"

SERIES = "KXNBAGAME"

TEAMS = {
    "Atlanta Hawks": ("ATL",),
    "Boston Celtics": ("BOS",),
    "Brooklyn Nets": ("BKN", "BRK"),
    "Charlotte Hornets": ("CHA",),
    "Chicago Bulls": ("CHI",),
    "Cleveland Cavaliers": ("CLE",),
    "Dallas Mavericks": ("DAL",),
    "Denver Nuggets": ("DEN",),
    "Detroit Pistons": ("DET",),
    "Golden State Warriors": ("GSW", "GS"),
    "Houston Rockets": ("HOU",),
    "Indiana Pacers": ("IND",),
    "Los Angeles Clippers": ("LAC",),
    "LA Clippers": ("LAC",),
    "Los Angeles Lakers": ("LAL",),
    "Memphis Grizzlies": ("MEM",),
    "Miami Heat": ("MIA",),
    "Milwaukee Bucks": ("MIL",),
    "Minnesota Timberwolves": ("MIN",),
    "New Orleans Pelicans": ("NOP", "NO"),
    "New York Knicks": ("NYK", "NY"),
    "Oklahoma City Thunder": ("OKC",),
    "Orlando Magic": ("ORL",),
    "Philadelphia 76ers": ("PHI",),
    "Phoenix Suns": ("PHX",),
    "Portland Trail Blazers": ("POR",),
    "Sacramento Kings": ("SAC",),
    "San Antonio Spurs": ("SAS", "SA"),
    "Toronto Raptors": ("TOR",),
    "Utah Jazz": ("UTA", "UT"),
    "Washington Wizards": ("WAS",),
}

ODDS_FIELDS = (
    "timestamp",
    "event_id",
    "start",
    "home",
    "away",
    "book",
    "book_updated",
    "home_fair",
    "away_fair",
    "home_odds",
    "away_odds",
)


# ============================================================
# SHARED HELPERS
# ============================================================

def dt(x):
    parsed = datetime.fromisoformat(
        str(x).replace("Z", "+00:00")
    )

    if parsed.tzinfo is None:
        raise ValueError(f"Timezone required: {x}")

    return parsed.astimezone(timezone.utc)


def stamp(x):
    return (
        x.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def read_csv(path):
    with Path(path).open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def american_prob(odds):
    odds = float(odds)

    if odds == 0:
        raise ValueError("Invalid American odds")

    if odds > 0:
        return 100 / (100 + odds)

    return -odds / (100 - odds)


def fair_pair(home, away):
    h = american_prob(home)
    a = american_prob(away)

    return h / (h + a), a / (h + a)


def fee(price, qty, multiplier):
    """Estimated aggregate Kalshi quadratic taker fee."""

    p = Decimal(str(price))

    if (
        not 0 < p < 1
        or qty < 1
        or not 0 <= multiplier <= 1
    ):
        raise ValueError("Invalid fee inputs")

    raw = (
        Decimal("0.07")
        * Decimal(str(multiplier))
        * qty
        * p
        * (1 - p)
    )

    return float(
        raw.quantize(
            Decimal("0.01"),
            rounding=ROUND_CEILING,
        )
    )


# ============================================================
# HISTORICAL SPORTSBOOK DATA
# ============================================================

def export_odds(root):
    rows = {}

    for path in sorted(
        (root / "odds_raw").glob("*.json")
    ):
        body = read_json(path)
        snapshot = body["timestamp"]

        for game in body.get("data", []):
            home = game.get("home_team")
            away = game.get("away_team")

            if not home or not away:
                continue

            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") != "h2h":
                        continue

                    outcomes = market.get(
                        "outcomes", []
                    )

                    by_name = {
                        o.get("name"): o.get("price")
                        for o in outcomes
                    }

                    if (
                        len(outcomes) != 2
                        or home not in by_name
                        or away not in by_name
                    ):
                        continue

                    updated = (
                        market.get("last_update")
                        or book.get("last_update")
                    )

                    if not updated:
                        continue

                    try:
                        if dt(updated) > dt(snapshot):
                            continue

                        home_fair, away_fair = fair_pair(
                            by_name[home],
                            by_name[away],
                        )

                    except (
                        ValueError,
                        TypeError,
                        ZeroDivisionError,
                    ):
                        continue

                    key = (
                        snapshot,
                        game["id"],
                        book.get("key", ""),
                    )

                    rows[key] = dict(
                        timestamp=snapshot,
                        event_id=game["id"],
                        start=game["commence_time"],
                        home=home,
                        away=away,
                        book=book.get("key", ""),
                        book_updated=updated,
                        home_fair=home_fair,
                        away_fair=away_fair,
                        home_odds=by_name[home],
                        away_odds=by_name[away],
                    )

    save_csv(
        root / "odds.csv",
        [rows[k] for k in sorted(rows)],
        ODDS_FIELDS,
    )

    print(
        f"Exported {len(rows)} bookmaker observations "
        f"-> {root / 'odds.csv'}"
    )


def collect_odds(a):
    start = dt(a.start)
    end = dt(a.end)

    if (
        start >= end
        or a.step < 5
        or a.step % 5
        or a.limit < 1
    ):
        raise SystemExit(
            "Require start < end, step multiple of 5, "
            "and positive limit"
        )

    pending = []
    t = start

    while t <= end:
        path = (
            a.dir
            / "odds_raw"
            / (
                t.strftime("%Y%m%dT%H%M%SZ")
                + ".json"
            )
        )

        if not path.exists():
            pending.append((t, path))

        t += timedelta(minutes=a.step)

    n = min(a.limit, len(pending))

    print(
        f"Uncached snapshots: {len(pending)}; "
        f"this run: {n}; "
        f"estimate: ~{n * 10} API credits"
    )

    if not a.execute:
        print(
            "DRY RUN. Add --execute to spend credits "
            "on a qualifying historical "
            "The Odds API plan."
        )
        return

    key = os.getenv("THE_ODDS_API_KEY")

    if not key:
        raise SystemExit(
            "Missing environment variable THE_ODDS_API_KEY"
        )

    session = requests.Session()

    for index, (when, path) in enumerate(
        pending[:a.limit], 1
    ):
        print(
            f"[{index}/{n}] {stamp(when)}",
            flush=True,
        )

        response = session.get(
            ODDS,
            params={
                "apiKey": key,
                "date": stamp(when),
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
            },
            timeout=35,
        )

        response.raise_for_status()

        body = response.json()

        if (
            "timestamp" not in body
            or "data" not in body
            or dt(body["timestamp"]) > when
        ):
            raise RuntimeError(
                "Unexpected historical snapshot format "
                "or future data"
            )

        save_json(path, body)

        print(
            f"  events={len(body['data'])}; "
            f"credits remaining="
            f"{response.headers.get('x-requests-remaining', '?')}"
        )

        time.sleep(0.15)

    export_odds(a.dir)


# ============================================================
# KALSHI HISTORICAL CLIENT
# ============================================================

class KalshiAPI:
    def __init__(self):
        self.session = requests.Session()

    def get(self, path, params=None):
        for i in range(6):
            try:
                r = self.session.get(
                    KALSHI + path,
                    params=params,
                    timeout=30,
                )

            except (
                requests.Timeout,
                requests.ConnectionError,
            ):
                if i == 5:
                    raise

                time.sleep(min(2 ** i, 12))
                continue

            if r.status_code in (
                429, 500, 502, 503, 504
            ):
                if i == 5:
                    r.raise_for_status()

                time.sleep(min(2 ** i, 12))
                continue

            r.raise_for_status()
            return r.json()

        raise RuntimeError("Kalshi retry limit")

    def markets(self):
        found = {}

        endpoints = (
            (
                "/markets",
                {"mve_filter": "exclude"},
            ),
            (
                "/historical/markets",
                {},
            ),
        )

        for route, extra in endpoints:
            cursor = None

            while True:
                params = {
                    "series_ticker": SERIES,
                    "limit": 1000,
                    **extra,
                }

                if cursor:
                    params["cursor"] = cursor

                response = self.get(
                    route,
                    params,
                )

                for market in response.get(
                    "markets", []
                ):
                    ticker = market.get(
                        "ticker", ""
                    )

                    if ticker.startswith(
                        SERIES + "-"
                    ):
                        found[ticker] = market

                cursor = response.get("cursor")

                if not cursor:
                    break

        return list(found.values())

    def candles(self, ticker, start, end):
        collected = {}
        moment = start

        while moment < end:
            stop = min(
                end,
                moment + timedelta(minutes=4500),
            )

            params = {
                "start_ts": int(moment.timestamp()),
                "end_ts": int(stop.timestamp()),
                "period_interval": 1,
            }

            body = None

            endpoints = (
                f"/historical/markets/"
                f"{ticker}/candlesticks",
                f"/series/{SERIES}/markets/"
                f"{ticker}/candlesticks",
            )

            for route in endpoints:
                try:
                    response = self.get(
                        route,
                        params,
                    )

                except requests.HTTPError as exc:
                    if (
                        exc.response is not None
                        and exc.response.status_code == 404
                    ):
                        continue

                    raise

                body = response

                if response.get("candlesticks"):
                    break

            if body is None:
                raise RuntimeError(
                    f"Missing candles for {ticker}"
                )

            for candle in body.get(
                "candlesticks", []
            ):
                epoch = candle.get(
                    "end_period_ts"
                )

                if epoch is None:
                    continue

                def price(side):
                    item = candle.get(side) or {}

                    value = item.get(
                        "close_dollars",
                        item.get("close"),
                    )

                    if value is None:
                        return ""

                    if (
                        "close_dollars" not in item
                        and isinstance(value, int)
                    ):
                        value /= 100

                    value = float(value)

                    if 0 < value < 1:
                        return value

                    return ""

                collected[int(epoch)] = dict(
                    ts=stamp(
                        datetime.fromtimestamp(
                            int(epoch),
                            timezone.utc,
                        )
                    ),
                    bid=price("yes_bid"),
                    ask=price("yes_ask"),
                )

            moment = stop + timedelta(seconds=1)

        return [
            collected[k]
            for k in sorted(collected)
        ]


# ============================================================
# MATCH NBA GAMES TO KALSHI CONTRACTS
# ============================================================

def match_game(game, markets):
    home_codes = TEAMS.get(game["home"])
    away_codes = TEAMS.get(game["away"])

    if not home_codes or not away_codes:
        return None

    datecode = (
        dt(game["start"])
        .astimezone(
            ZoneInfo("America/New_York")
        )
        .strftime("%y%b%d")
        .upper()
    )

    blobs = (
        {
            x + y
            for x in away_codes
            for y in home_codes
        }
        |
        {
            y + x
            for x in away_codes
            for y in home_codes
        }
    )

    matches = defaultdict(list)

    for m in markets:
        event = str(
            m.get("event_ticker", "")
        )

        if any(
            event == f"{SERIES}-{datecode}{blob}"
            for blob in blobs
        ):
            matches[event].append(m)

    if len(matches) != 1:
        return None

    event, candidates = next(
        iter(matches.items())
    )

    result = {
        "event_ticker": event,
    }

    for side, codes in (
        ("home", home_codes),
        ("away", away_codes),
    ):
        found = [
            m
            for m in candidates
            if any(
                str(
                    m.get("ticker", "")
                ).endswith("-" + c)
                for c in codes
            )
        ]

        if len(found) != 1:
            return None

        result[side + "_ticker"] = (
            found[0]["ticker"]
        )

        result[side + "_result"] = (
            found[0].get("result", "")
        )

    if (
        result["home_ticker"]
        == result["away_ticker"]
    ):
        return None

    return result


def collect_kalshi(a):
    odds = read_csv(
        a.dir / "odds.csv"
    )

    games = {}

    for r in odds:
        lead = (
            dt(r["start"])
            - dt(r["timestamp"])
        ).total_seconds() / 60

        if 0 < lead <= a.max_lead:
            games[r["event_id"]] = {
                k: r[k]
                for k in (
                    "event_id",
                    "start",
                    "home",
                    "away",
                )
            }

    games = sorted(
        games.values(),
        key=lambda g: g["start"],
    )

    if a.max_games:
        games = games[:a.max_games]

    api = KalshiAPI()

    cache = (
        a.dir / "kalshi_markets.json"
    )

    if (
        cache.exists()
        and not a.refresh_markets
    ):
        markets = read_json(cache)

    else:
        markets = api.markets()

    if (
        a.refresh_markets
        or not cache.exists()
    ):
        save_json(
            cache,
            markets,
        )

    matched = []
    unmatched = []

    for i, game in enumerate(
        games, 1
    ):
        info = match_game(
            game,
            markets,
        )

        if info is None:
            unmatched.append(game)

            print(
                f"[{i}/{len(games)}] "
                f"UNMATCHED "
                f"{game['away']} @ {game['home']}"
            )

            continue

        entry = dict(game, **info)
        matched.append(entry)

        start = dt(
            game["start"]
        )

        relevant = [
            dt(r["timestamp"])
            for r in odds
            if (
                r["event_id"] == game["event_id"]
                and 0 <
                (
                    start - dt(r["timestamp"])
                ).total_seconds() / 60
                <= a.max_lead
            )
        ]

        first = (
            min(relevant)
            - timedelta(minutes=5)
        )

        for side in (
            "home",
            "away",
        ):
            ticker = entry[
                side + "_ticker"
            ]

            path = (
                a.dir
                / "candles"
                / f"{ticker}.csv"
            )

            if (
                path.exists()
                and not a.refresh_candles
            ):
                print(
                    f"[{i}/{len(games)}] "
                    f"CACHED {ticker}"
                )
                continue

            candles = api.candles(
                ticker,
                first,
                start,
            )

            save_csv(
                path,
                candles,
                ("ts", "bid", "ask"),
            )

            print(
                f"[{i}/{len(games)}] "
                f"{ticker}: "
                f"{len(candles)} candles"
            )

    save_json(
        a.dir / "matched_games.json",
        matched,
    )

    save_json(
        a.dir / "unmatched_games.json",
        unmatched,
    )

    print(
        f"Matched {len(matched)}/{len(games)} games"
    )


# ============================================================
# HISTORICAL PRICE HELPERS
# ============================================================

def candle_rows(path):
    out = []

    for row in read_csv(path):
        def num(key):
            try:
                p = float(
                    row[key]
                )

                if 0 < p < 1:
                    return p

                return None

            except (
                ValueError,
                KeyError,
            ):
                return None

        out.append({
            "t": dt(row["ts"]),
            "bid": num("bid"),
            "ask": num("ask"),
        })

    return sorted(
        out,
        key=lambda r: r["t"],
    )


def nearest_before(candles, point, max_age):
    times = [
        c["t"]
        for c in candles
    ]

    i = (
        bisect_right(
            times,
            point,
        ) - 1
    )

    while (
        i >= 0
        and point - times[i]
        <= timedelta(minutes=max_age)
    ):
        if (
            candles[i]["ask"] is not None
            and candles[i]["bid"] is not None
        ):
            return candles[i]

        i -= 1

    return None


def nearest_after(
    candles,
    point,
    latest,
    max_lag,
):
    times = [
        c["t"]
        for c in candles
    ]

    i = bisect_left(
        times,
        point,
    )

    while (
        i < len(times)
        and times[i]
        <= min(
            latest,
            point + timedelta(minutes=max_lag),
        )
    ):
        if (
            candles[i]["ask"] is not None
            and candles[i]["bid"] is not None
        ):
            return candles[i]

        i += 1

    return None


def aggregate_books(
    rows,
    snapshot,
    max_age,
    min_books,
):
    unique = {}

    for r in rows:
        try:
            age = (
                snapshot
                - dt(r["book_updated"])
            ).total_seconds() / 60

            h = float(
                r["home_fair"]
            )

            away = float(
                r["away_fair"]
            )

            if (
                not 0 <= age <= max_age
                or abs(h + away - 1) > .005
            ):
                continue

            unique[r["book"]] = (
                h,
                away,
            )

        except (
            ValueError,
            KeyError,
        ):
            continue

    if len(unique) < min_books:
        return None

    values = list(
        unique.values()
    )

    return {
        "home": statistics.median(
            v[0] for v in values
        ),
        "away": statistics.median(
            v[1] for v in values
        ),
        "n": len(values),
    }


# ============================================================
# ENTRY SIGNALS
# ============================================================

def find_entry(
    odds,
    game,
    candles,
    a,
    signal,
):
    start = dt(
        game["start"]
    )

    snapshots = defaultdict(list)

    for r in odds:
        t = dt(
            r["timestamp"]
        )

        if (
            a.min_before
            <= (
                start - t
            ).total_seconds() / 60
            <= a.max_lead
        ):
            snapshots[t].append(r)

    previous = None

    for t in sorted(snapshots):
        current = aggregate_books(
            snapshots[t],
            t,
            a.max_book_age,
            a.min_books,
        )

        if current is None:
            continue

        options = []

        for side in (
            "home",
            "away",
        ):
            fair = current[side]

            prev = (
                previous[1][side]
                if previous
                else None
            )

            move = (
                fair - prev
                if prev is not None
                else 0
            )

            if signal == "lead_lag":
                if (
                    previous is None
                    or (
                        t - previous[0]
                    ) > timedelta(minutes=15)
                    or move < a.min_book_move
                ):
                    continue

            old = nearest_before(
                candles[side],
                t - timedelta(minutes=1),
                a.max_quote_age,
            )

            if old is None:
                continue

            if (
                old["ask"] - old["bid"]
                > a.max_spread
            ):
                continue

            expected_price = (
                old["ask"]
                + a.entry_slip
            )

            if not 0 < expected_price < 1:
                continue

            projected_cost = (
                expected_price
                + fee(
                    expected_price,
                    a.qty,
                    a.fee_multiplier,
                ) / a.qty
            )

            if (
                fair - projected_cost
                < a.min_edge
            ):
                continue

            execution = nearest_after(
                candles[side],
                t + timedelta(
                    minutes=a.entry_delay
                ),
                start - timedelta(
                    minutes=a.min_before
                ),
                a.max_quote_age,
            )

            if execution is None:
                continue

            if (
                execution["ask"]
                - execution["bid"]
                > a.max_spread
            ):
                continue

            buy = (
                execution["ask"]
                + a.entry_slip
            )

            if not 0 < buy < 1:
                continue

            cost = (
                a.qty * buy
                + fee(
                    buy,
                    a.qty,
                    a.fee_multiplier,
                )
            )

            edge = (
                fair
                - cost / a.qty
            )

            if edge < a.min_edge:
                continue

            options.append({
                "signal": signal,
                "event_id": game["event_id"],
                "ticker": game[side + "_ticker"],
                "side": side,
                "start": game["start"],
                "snapshot": stamp(t),
                "entry_time": stamp(
                    execution["t"]
                ),
                "entry_ask": buy,
                "entry_cost": cost,
                "fair": fair,
                "prior_fair": prev,
                "books": current["n"],
                "edge": edge,
                "result": game[side + "_result"],
            })

        previous = (
            t,
            current,
        )

        if options:
            return max(
                options,
                key=lambda x: x["edge"],
            )

    return None


# ============================================================
# EXIT STRATEGIES
# ============================================================

def simulate(
    entry,
    candles,
    a,
    rule,
    target=None,
    hold=None,
):
    row = dict(
        entry,
        rule=rule,
        target_cents=target,
        hold_minutes=hold,
        status="censored",
        reason="",
        exit_time="",
        exit_bid="",
        pnl="",
        roi="",
    )

    entry_time = dt(
        entry["entry_time"]
    )

    buy_cost = entry[
        "entry_cost"
    ]

    # -----------------------------------------
    # SETTLEMENT BENCHMARK
    # -----------------------------------------

    if rule == "settlement":
        result = str(
            entry["result"]
        ).lower()

        if result not in (
            "yes",
            "no",
        ):
            row["reason"] = "UNKNOWN_RESULT"
            return row

        pnl = (
            a.qty * (result == "yes")
            - buy_cost
        )

        row.update(
            status="completed",
            reason="SETTLEMENT",
            exit_time="settlement",
            pnl=pnl,
            roi=pnl / buy_cost,
        )

        return row

    # -----------------------------------------
    # PREGAME EXIT LIMIT
    # -----------------------------------------

    cutoff = (
        dt(entry["start"])
        - timedelta(minutes=1)
    )

    deadline = min(
        entry_time
        + timedelta(minutes=hold),
        cutoff,
    )

    if (
        entry_time
        + timedelta(minutes=1)
        > cutoff
    ):
        row["reason"] = (
            "NO_PREGAME_EXIT_WINDOW"
        )
        return row

    chosen = None

    # -----------------------------------------
    # CHRONOLOGICAL EXIT SIMULATION
    # -----------------------------------------

    for c in candles:
        if (
            c["t"] <= entry_time
            or c["t"] > cutoff
            or c["bid"] is None
        ):
            continue

        if (
            c["t"]
            > deadline
            + timedelta(
                minutes=a.max_quote_age
            )
        ):
            break

        sell = (
            c["bid"]
            - a.exit_slip
        )

        if not 0 < sell < 1:
            continue

        proceeds = (
            a.qty * sell
            - fee(
                sell,
                a.qty,
                a.fee_multiplier,
            )
        )

        pnl = (
            proceeds
            - buy_cost
        )

        # -------------------------------------
        # IMMEDIATE / TARGET PROFIT
        # -------------------------------------

        if (
            c["t"] <= deadline
            and rule == "take_profit"
            and pnl >= a.qty * (target / 100)
        ):
            chosen = (
                c,
                sell,
                pnl,
                "TARGET",
            )
            break

        # -------------------------------------
        # OPTIONAL STOP LOSS
        # -------------------------------------

        if (
            c["t"] <= deadline
            and a.stop_cents is not None
            and (
                entry["entry_ask"] - sell
            ) * 100 >= a.stop_cents
        ):
            chosen = (
                c,
                sell,
                pnl,
                "STOP",
            )
            break

        # -------------------------------------
        # TIMEOUT
        # -------------------------------------

        if c["t"] >= deadline:
            chosen = (
                c,
                sell,
                pnl,
                "DEADLINE",
            )
            break

    if chosen is None:
        row["reason"] = (
            "NO_FRESH_EXIT_BID"
        )
        return row

    c, sell, pnl, reason = chosen

    row.update(
        status="completed",
        reason=reason,
        exit_time=stamp(c["t"]),
        exit_bid=sell,
        pnl=pnl,
        roi=pnl / buy_cost,
    )

    return row


# ============================================================
# BACKTEST
# ============================================================

def backtest(a):
    odds = read_csv(
        a.dir / "odds.csv"
    )

    games = sorted(
        read_json(
            a.dir / "matched_games.json"
        ),
        key=lambda g: g["start"],
    )

    by_game = defaultdict(list)

    for r in odds:
        by_game[
            r["event_id"]
        ].append(r)

    all_signals = []
    trades = []

    has_test = (
        len(games) >= 10
    )

    split = int(
        len(games) * .7
    )

    for i, game in enumerate(games):
        paths = {
            side: (
                a.dir
                / "candles"
                / f"{game[side + '_ticker']}.csv"
            )
            for side in (
                "home",
                "away",
            )
        }

        if not all(
            path.exists()
            for path in paths.values()
        ):
            continue

        candles = {
            side: candle_rows(path)
            for side, path in paths.items()
        }

        for name in (
            "gap",
            "lead_lag",
        ):
            entry = find_entry(
                by_game[
                    game["event_id"]
                ],
                game,
                candles,
                a,
                name,
            )

            if entry is None:
                continue

            if has_test:
                entry["split"] = (
                    "train"
                    if i < split
                    else "test"
                )
            else:
                entry["split"] = "pilot"

            all_signals.append(entry)

            series = candles[
                entry["side"]
            ]

            # Immediate profit and fixed profit targets
            for target in (
                0, 1, 2, 3, 5
            ):
                for hold in (
                    5, 15, 30, 60
                ):
                    trades.append(
                        simulate(
                            entry,
                            series,
                            a,
                            "take_profit",
                            target,
                            hold,
                        )
                    )

            # Fixed-time exits
            for hold in (
                1, 3, 5, 15, 30
            ):
                trades.append(
                    simulate(
                        entry,
                        series,
                        a,
                        "timed",
                        None,
                        hold,
                    )
                )

            # Hold to final settlement
            trades.append(
                simulate(
                    entry,
                    series,
                    a,
                    "settlement",
                )
            )

    signal_fields = (
        "split",
        "signal",
        "event_id",
        "ticker",
        "side",
        "start",
        "snapshot",
        "entry_time",
        "entry_ask",
        "entry_cost",
        "fair",
        "prior_fair",
        "books",
        "edge",
        "result",
    )

    trade_fields = signal_fields + (
        "rule",
        "target_cents",
        "hold_minutes",
        "status",
        "reason",
        "exit_time",
        "exit_bid",
        "pnl",
        "roi",
    )

    save_csv(
        a.dir / "signals.csv",
        all_signals,
        signal_fields,
    )

    save_csv(
        a.dir / "trades.csv",
        trades,
        trade_fields,
    )

    # -----------------------------------------
    # PERFORMANCE SUMMARY
    # -----------------------------------------

    group = defaultdict(list)

    for trade in trades:
        key = (
            trade["split"],
            trade["signal"],
            trade["rule"],
            trade["target_cents"],
            trade["hold_minutes"],
        )

        group[key].append(trade)

    summary = []

    for key, rows in sorted(
        group.items(),
        key=lambda kv: str(kv[0]),
    ):
        completed = sorted(
            (
                r for r in rows
                if r["status"] == "completed"
            ),
            key=lambda r: r["entry_time"],
        )

        total_cost = sum(
            r["entry_cost"]
            for r in completed
        )

        profit = sum(
            r["pnl"]
            for r in completed
        )

        wins = sum(
            r["pnl"] > 0
            for r in completed
        )

        equity = 0.
        peak = 0.
        drawdown = 0.

        for r in completed:
            equity += r["pnl"]
            peak = max(peak, equity)

            drawdown = max(
                drawdown,
                peak - equity,
            )

        summary.append(
            dict(
                zip(
                    (
                        "split",
                        "signal",
                        "rule",
                        "target_cents",
                        "hold_minutes",
                    ),
                    key,
                ),
                candidates=len(rows),
                completed=len(completed),
                censored=(
                    len(rows) - len(completed)
                ),
                wins=wins,
                win_rate=(
                    wins / len(completed)
                    if completed else ""
                ),
                net_pnl=round(
                    profit, 4
                ),
                roi=(
                    profit / total_cost
                    if total_cost else ""
                ),
                max_drawdown=round(
                    drawdown, 4
                ),
            )
        )

    save_csv(
        a.dir / "summary.csv",
        summary,
        (
            "split",
            "signal",
            "rule",
            "target_cents",
            "hold_minutes",
            "candidates",
            "completed",
            "censored",
            "wins",
            "win_rate",
            "net_pnl",
            "roi",
            "max_drawdown",
        ),
    )

    print(
        f"Matched games={len(games)}, "
        f"signals={len(all_signals)}, "
        f"scenarios={len(trades)}"
    )

    print(
        f"Wrote {a.dir / 'signals.csv'}, "
        f"{a.dir / 'trades.csv'}, "
        f"{a.dir / 'summary.csv'}"
    )

    for r in summary:
        show = (
            r["rule"] == "settlement"
            or (
                r["rule"] == "take_profit"
                and (
                    r["target_cents"],
                    r["hold_minutes"],
                ) in (
                    (0, 15),
                    (2, 15),
                    (5, 30),
                )
            )
        )

        if not show:
            continue

        value = r["roi"]

        if isinstance(value, float):
            print(
                f"{r['split']:>5} "
                f"{r['signal']:>8} "
                f"{r['rule']:>12} "
                f"target={str(r['target_cents']):>4} "
                f"hold={str(r['hold_minutes']):>4} "
                f"completed={r['completed']:>4} "
                f"censored={r['censored']:>4} "
                f"ROI={value:+.2%}"
            )
        else:
            print(
                f"{r['split']} "
                f"{r['signal']} "
                f"{r['rule']}: "
                f"no completed trades"
            )

    print(
        "Quote-based simulations only: "
        "past bids/asks do not prove "
        "historical available depth."
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--dir",
        type=Path,
        default=ROOT,
    )

    sub = p.add_subparsers(
        dest="mode",
        required=True,
    )

    odds = sub.add_parser(
        "odds",
        help="Historical sportsbook snapshots",
    )

    odds.add_argument(
        "--start",
        required=True,
    )

    odds.add_argument(
        "--end",
        required=True,
    )

    odds.add_argument(
        "--step",
        type=int,
        default=5,
    )

    odds.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    odds.add_argument(
        "--execute",
        action="store_true",
    )

    sub.add_parser(
        "odds-export",
        help="Re-export cached odds JSON",
    )

    kalshi = sub.add_parser(
        "kalshi",
        help="Download historical Kalshi data",
    )

    kalshi.add_argument(
        "--max-lead",
        type=int,
        default=240,
    )

    kalshi.add_argument(
        "--max-games",
        type=int,
        default=0,
    )

    kalshi.add_argument(
        "--refresh-markets",
        action="store_true",
    )

    kalshi.add_argument(
        "--refresh-candles",
        action="store_true",
    )

    bt = sub.add_parser(
        "backtest",
        help="Run historical simulations",
    )

    bt.add_argument(
        "--qty",
        type=int,
        default=10,
    )

    bt.add_argument(
        "--fee-multiplier",
        type=float,
        default=1.,
    )

    bt.add_argument(
        "--entry-slip",
        type=float,
        default=.01,
    )

    bt.add_argument(
        "--exit-slip",
        type=float,
        default=.01,
    )

    bt.add_argument(
        "--min-edge",
        type=float,
        default=.03,
    )

    bt.add_argument(
        "--min-book-move",
        type=float,
        default=.02,
    )

    bt.add_argument(
        "--min-books",
        type=int,
        default=3,
    )

    bt.add_argument(
        "--max-book-age",
        type=int,
        default=60,
    )

    bt.add_argument(
        "--max-quote-age",
        type=int,
        default=2,
    )

    bt.add_argument(
        "--max-spread",
        type=float,
        default=.06,
    )

    bt.add_argument(
        "--max-lead",
        type=int,
        default=240,
    )

    bt.add_argument(
        "--min-before",
        type=int,
        default=10,
    )

    bt.add_argument(
        "--entry-delay",
        type=int,
        default=2,
    )

    bt.add_argument(
        "--stop-cents",
        type=int,
    )

    a = p.parse_args()

    if a.mode == "odds":
        collect_odds(a)

    elif a.mode == "odds-export":
        export_odds(a.dir)

    elif a.mode == "kalshi":
        collect_kalshi(a)

    elif a.mode == "backtest":
        if (
            a.qty < 1
            or a.entry_delay < 1
            or a.min_before < 2
            or not 0 <= a.fee_multiplier <= 1
        ):
            p.error(
                "Invalid backtest quantity, entry delay, "
                "cutoff or fee multiplier"
            )

        backtest(a)


if __name__ == "__main__":
    main()
