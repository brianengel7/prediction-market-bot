"""Read-only multi-city daily-high YES basket scanner."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal as D, ROUND_FLOOR
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import websockets

from src.backtest.weather_yes_basket_scanner import (
    Book, WS_URL, auth_headers, scan,
)
from src.kalshi.fees import calculate_taker_fee


REST = "https://external-api.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")

CITY = {
    "NYC": ("New York", "KXHIGHNY", "America/New_York"),
    "CHI": ("Chicago", "KXHIGHCHI", "America/Chicago"),
    "LAX": ("Los Angeles", "KXHIGHLAX", "America/Los_Angeles"),
    "MIA": ("Miami", "KXHIGHMIA", "America/New_York"),
    "DEN": ("Denver", "KXHIGHDEN", "America/Denver"),
    "AUS": ("Austin", "KXHIGHAUS", "America/Chicago"),
    "BOS": ("Boston", "KXHIGHTBOS", "America/New_York"),
    "PHL": ("Philadelphia", "KXHIGHPHIL", "America/New_York"),
    "PHX": ("Phoenix", "KXHIGHTPHX", "America/Phoenix"),
    "SEA": ("Seattle", "KXHIGHTSEA", "America/Los_Angeles"),
    'ATL': ('Atlanta', 'KXHIGHTATL', 'America/New_York'),
    'DAL': ('Dallas', 'KXHIGHTDAL', 'America/Chicago'),
    'HOU': ('Houston', 'KXHIGHTHOU', 'America/Chicago'),
    'LV': ('Las Vegas', 'KXHIGHTLV', 'America/Los_Angeles'),
    'SDF': ('Louisville', 'KXHIGHTSDF', 'America/New_York'),
    'MIN': ('Minneapolis', 'KXHIGHTMIN', 'America/Chicago'),
    'NOLA': ('New Orleans', 'KXHIGHTNOLA', 'America/Chicago'),
    'EWR': ('Newark', 'KXHIGHTEWR', 'America/New_York'),
    'OKC': ('Oklahoma City', 'KXHIGHTOKC', 'America/Chicago'),
    'SATX': ('San Antonio', 'KXHIGHTSATX', 'America/Chicago'),
    'SAN': ('San Diego', 'KXHIGHTSAN', 'America/Los_Angeles'),
    'SFO': ('San Francisco', 'KXHIGHTSFO', 'America/Los_Angeles'),
    'TTN': ('Trenton', 'KXHIGHTTTN', 'America/New_York'),
    'DC': ('Washington DC', 'KXHIGHTDC', 'America/New_York'),
}

DEFAULT_CITIES = 'NYC,CHI,LAX,MIA,DEN,AUS,BOS,PHL,PHX,SEA,ATL,DAL,HOU,LV,SDF,MIN,NOLA,EWR,OKC,SATX,SAN,SFO,TTN,DC'
DELAYS = (100, 250, 500, 1000, 2000, 5000)


def now_et():
    return datetime.now(ET).strftime("%H:%M:%S.%f")[:-3] + " ET"


def whole(q):
    return max(
        0,
        int(D(str(q)).to_integral_value(rounding=ROUND_FLOOR)),
    )


def fetch_event(code):
    name, series, timezone = CITY[code]
    local_date = datetime.now(ZoneInfo(timezone)).date()
    event_ticker = (
        f"{series}-{local_date.strftime('%y%b%d').upper()}"
    )
    url = (
        f"{REST}/events/{event_ticker}"
        "?with_nested_markets=true"
    )

    try:
        req = Request(
            url, headers={"Accept": "application/json"}
        )
        with urlopen(req, timeout=12) as response:
            payload = json.load(response)

    except HTTPError as exc:
        if exc.code == 404:
            return code, None, (
                f"Today's event {event_ticker} not listed"
            )
        return code, None, f"HTTP {exc.code}: {event_ticker}"

    except Exception as exc:
        return code, None, (
            f"Event fetch error: "
            f"{type(exc).__name__}: {exc}"
        )

    event = payload.get("event") or {}

    if event.get("event_ticker") != event_ticker:
        return code, None, "Event ticker mismatch"

    if event.get("mutually_exclusive") is False:
        return code, None, (
            "Event not mutually exclusive; skipping"
        )

    markets = (
        event.get("markets")
        or payload.get("markets")
        or []
    )

    tickers = sorted({
        m["ticker"]
        for m in markets
        if m.get("event_ticker", event_ticker) == event_ticker
        and m.get("status") in ("active", "open")
        and m.get("ticker")
    })

    if len(tickers) < 3:
        states = sorted({
            str(m.get("status")) for m in markets
        })
        return code, None, (
            f"{event_ticker}: {len(tickers)} active markets; "
            f"statuses={states}"
        )

    return code, (event_ticker, local_date, tickers), None


TOP_TWO_MIN = D("0.75")


def top_two_midpoint_share(books, tickers, legs):
    """Combined raw YES midpoints of selected contracts.

    Requires two-sided quotes only for selected legs.
    Other contracts do not enter the calculation.
    Returns None when a selected quote is unavailable.
    """
    if len(legs) != 2 or legs[0] == legs[1]:
        return None

    midpoint_total = D(0)

    for ticker in legs:
        book = books.get(ticker)

        if book is None:
            return None

        yes_bids = [
            D(str(price))
            for price, qty in book.levels["yes"].items()
            if D(str(qty)) > 0
        ]

        no_bids = [
            D(str(price))
            for price, qty in book.levels["no"].items()
            if D(str(qty)) > 0
        ]

        if not yes_bids or not no_bids:
            return None

        yes_bid = max(yes_bids)
        yes_ask = D(1) - max(no_bids)

        if not D(0) <= yes_bid <= yes_ask <= D(1):
            return None

        midpoint_total += (yes_bid + yes_ask) / D(2)

    return midpoint_total



def top_ask_and_depth(book):
    """Get executable YES ask from resting NO bids."""
    bids = [
        (D(str(p)), D(str(q)))
        for p, q in book.levels["no"].items()
        if D(str(q)) > 0
    ]

    if not bids:
        return None

    best_no = max(p for p, _ in bids)

    depth = whole(sum(
        q for p, q in bids if p == best_no
    ))

    return D(1) - best_no, depth


def fills_at_original_limit(book, max_yes_ask, contracts):
    """Hypothetical YES fill from available NO bids."""
    remaining = D(contracts)
    paid = D(0)

    for no_bid, depth in sorted(
        book.levels["no"].items(),
        reverse=True,
    ):
        price = D(1) - D(str(no_bid))

        if price > max_yes_ask:
            break

        qty = min(remaining, D(str(depth)))

        if qty > 0:
            paid += qty * price
            remaining -= qty

        if remaining <= 0:
            return paid / D(contracts)

    return None


def estimate_fees(a, b, n):
    return sum(
        (
            D(str(calculate_taker_fee(
                price=float(price),
                contracts=n,
                multiplier=1,
            )))
            for price in (a, b)
        ),
        D(0),
    )


def economics(p1, p2, n):
    """Economics conditional on either YES winning."""
    fee = estimate_fees(p1, p2, n)
    cash = (p1 + p2) * n + fee
    net_if_win = D(n) - cash
    all_in_each = cash / n

    return fee, cash, net_if_win, all_in_each


@dataclass
class Episode:
    id: int
    city: str
    start_ns: int
    legs: tuple[str, str]
    limits: dict[str, D]
    sizes: tuple[int, ...]
    initial_capacity: int
    first_leg: str
    ended_ns: int | None = None
    checked: set[int] = field(default_factory=set)
    seq_first: dict[int, bool] | None = None
    seq_completed: bool = False


@dataclass
class CityState:
    code: str
    event: str
    local_date: object
    tickers: list[str]
    books: dict[str, Book]
    updates: int = 0
    last_pair: str = "waiting for snapshots"
    active: Episode | None = None
    episodes: int = 0
    last_tier_signature: tuple | None = None
    last_capacity_print: float = 0.0
    stats: dict[int, list[int]] = field(
        default_factory=lambda: {
            d: [0, 0] for d in DELAYS
        }
    )
    sequential: dict[str, int] = field(
        default_factory=lambda: {
            "both": 0,
            "exposed": 0,
            "no_entry": 0,
        }
    )


class Tracker:
    def __init__(self, discovered, cfg):
        self.cfg = cfg

        self.city = {
            code: CityState(
                code,
                event,
                day,
                tickers,
                {t: Book() for t in tickers},
            )
            for code, (event, day, tickers)
            in discovered.items()
        }

        self.market_to_city = {
            t: code
            for code, c in self.city.items()
            for t in c.tickers
        }

        self.pending = []
        self.last_seq = {}
        self.total_updates = 0
        self.candidates = 0

    def apply(self, packet):
        sid = packet.get("sid")
        seq = packet.get("seq")

        if sid is not None and seq is not None:
            sid, seq = int(sid), int(seq)
            previous = self.last_seq.get(sid)

            if previous is not None and seq != previous + 1:
                raise RuntimeError(
                    f"Sequence gap sid={sid}: "
                    f"{previous}->{seq}"
                )

            self.last_seq[sid] = seq

        msg = packet["msg"]
        ticker = msg["market_ticker"]
        code = self.market_to_city.get(ticker)

        if code is None:
            raise RuntimeError(
                f"Unexpected market ticker: {ticker}"
            )

        state = self.city[code]

        if packet["type"] == "orderbook_snapshot":
            state.books[ticker].snapshot(msg)
        else:
            state.books[ticker].delta(msg)

        state.updates += 1
        self.total_updates += 1

        self.evaluate(state)

    def close(self, city, t_ns, why):
        ep = city.active

        if ep is None:
            return

        ep.ended_ns = t_ns
        duration = (t_ns - ep.start_ns) / 1e6

        print(
            f"[{now_et()}] END {city.code} #{ep.id} "
            f"lasted={duration:.0f}ms reason={why}",
            flush=True,
        )

        city.active = None
        city.last_tier_signature = None

    def evaluate(self, city):
        if not all(
            book.ready for book in city.books.values()
        ):
            return

        result = scan(city.books, city.tickers)
        now_ns = time.monotonic_ns()

        if result is None:
            city.last_pair = "insufficient quotes"
            self.close(
                city, now_ns, "incomplete quotes"
            )
            return

        a, b, quoted_sum, valid = result

        legs = tuple(sorted((a["ticker"], b["ticker"])))
        book1 = city.books[legs[0]]
        book2 = city.books[legs[1]]

        first = top_ask_and_depth(book1)
        second = top_ask_and_depth(book2)

        if first is None or second is None:
            city.last_pair = "no executable YES ask"
            self.close(city, now_ns, "missing ask")
            return

        p1, d1 = first
        p2, d2 = second

        price = p1 + p2
        capacity = min(d1, d2)

        city.last_pair = (
            f"{price*100:.2f}c max={capacity}"
        )

        qualifies = (
            bool(valid)
            and price < self.cfg.cap
            and capacity >= 1
        )

        if not qualifies:
            self.close(
                city,
                now_ns,
                "price/depth below entry criteria",
            )
            return

        top_two_share = top_two_midpoint_share(
            city.books,
            city.tickers,
            legs,
        )

        if top_two_share is None or top_two_share < TOP_TWO_MIN:
            share_text = (
                "unavailable"
                if top_two_share is None
                else f"{top_two_share * 100:.2f}%"
            )

            city.last_pair = (
                f"{price*100:.2f}c "
                f"REJECT top2={share_text}"
            )

            self.close(
                city,
                now_ns,
                f"top-two concentration below 75% "
                f"or unavailable ({share_text})",
            )
            return

        sizes = sorted(
            {
                q for q in self.cfg.sizes
                if q <= capacity
            }
            | {
                min(capacity, self.cfg.max_evaluate)
            }
        )

        qualifying_sizes = []

        for n in sizes:
            fee, cost, profit, all_in = economics(
                p1, p2, n
            )

            if profit > 0:
                qualifying_sizes.append(n)

        if not qualifying_sizes:
            self.close(
                city,
                now_ns,
                "no positive conditional payout after fees",
            )
            return

        if city.active and city.active.legs != legs:
            self.close(
                city,
                now_ns,
                "leading outcomes changed",
            )

        tier_signature = (
            legs,
            p1,
            p2,
            tuple(
                n for n in self.cfg.sizes
                if n in qualifying_sizes
            ),
        )

        now_s = time.monotonic()

        if city.active is None:
            city.episodes += 1

            ep = Episode(
                city.episodes,
                city.code,
                now_ns,
                legs,
                {
                    legs[0]: p1,
                    legs[1]: p2,
                },
                tuple(qualifying_sizes),
                capacity,
                legs[0] if d1 <= d2 else legs[1],
            )

            city.active = ep
            self.pending.append(ep)
            self.candidates += 1

            self.print_signal(
                city, ep, p1, p2,
                d1, d2, capacity,
                top_two_share,
            )

            city.last_tier_signature = tier_signature
            city.last_capacity_print = now_s

        elif (
            tier_signature != city.last_tier_signature
            and now_s - city.last_capacity_print
            >= self.cfg.capacity_log_seconds
        ):
            city.last_tier_signature = tier_signature
            city.last_capacity_print = now_s

            print(
                f"[{now_et()}] CAPACITY {city.code} "
                f"pair={price*100:.2f}c "
                f"max_exact={capacity} "
                f"sizes={qualifying_sizes}",
                flush=True,
            )

    def print_signal(
        self, city, ep, p1, p2, d1, d2,
        capacity, top_two_share,
    ):
        print("\n" + "=" * 72)
        print(
            f">>> PRICE/FILL CANDIDATE [{now_et()}] "
            f"{city.code} {city.event} #{ep.id}"
        )

        print(
            f"  1 BUY YES {ep.legs[0]} "
            f"<= {p1*100:.2f}c "
            f"best-ask whole depth={d1}"
        )
        print(
            f"  2 BUY YES {ep.legs[1]} "
            f"<= {p2*100:.2f}c "
            f"best-ask whole depth={d2}"
        )
        print(
            f"  Basket={(p1+p2)*100:.2f}c "
            f"(<{self.cfg.cap*100:.2f}c); "
            f"MAX EXACT-ASK PAIRS={capacity}"
        )
        print(
            f"  TOP-TWO CONCENTRATION: "
            f"{top_two_share*100:.2f}% "
            f"(minimum 75%)"
        )
        print(
            "  q | est total cost | fees | "
            "profit IF COVERED | break-even coverage"
        )

        for n in ep.sizes:
            fee, cost, profit, breakeven = economics(
                p1, p2, n
            )

            print(
                f"  {n:>3} | ${cost:>10.2f} "
                f"| ${fee:>5.2f} "
                f"| ${profit:>+9.2f} "
                f"| {breakeven*100:>7.2f}%"
            )

        if capacity > self.cfg.max_evaluate:
            print(
                f"  Depth exceeds calculation ceiling "
                f"{self.cfg.max_evaluate}; "
                "increase --max-evaluate if needed."
            )

        print(
            "  Conditional payout, NOT guaranteed profit "
            "or proof of positive expected value."
        )
        print("  READ ONLY: NO ORDERS SENT")
        print("=" * 72, flush=True)

    def check_delays(self):
        now_ns = time.monotonic_ns()

        for ep in self.pending[:]:
            city = self.city[ep.city]
            age_ms = (
                now_ns - ep.start_ns
            ) / 1e6

            for delay in DELAYS:
                if (
                    delay in ep.checked
                    or age_ms < delay
                ):
                    continue

                passed = []
                failed = []

                for n in ep.sizes:
                    ok = all(
                        fills_at_original_limit(
                            city.books[t],
                            ep.limits[t],
                            n,
                        ) is not None
                        for t in ep.legs
                    )

                    (
                        passed if ok else failed
                    ).append(n)

                city.stats[delay][0] += len(ep.sizes)
                city.stats[delay][1] += len(passed)

                ep.checked.add(delay)

                print(
                    f"[{now_et()}] {ep.city} #{ep.id} "
                    f"delay={delay}ms "
                    f"actual={age_ms:.0f}ms "
                    f"both-legs-available={passed} "
                    f"unavailable={failed}",
                    flush=True,
                )

            if (
                ep.seq_first is None
                and age_ms >= 250
            ):
                ep.seq_first = {
                    n: fills_at_original_limit(
                        city.books[ep.first_leg],
                        ep.limits[ep.first_leg],
                        n,
                    ) is not None
                    for n in ep.sizes
                }

            if (
                not ep.seq_completed
                and age_ms >= 500
            ):
                ep.seq_completed = True

                other = next(
                    t for t in ep.legs
                    if t != ep.first_leg
                )

                statuses = []

                for n in ep.sizes:
                    if not ep.seq_first[n]:
                        outcome = "no_entry"

                    elif fills_at_original_limit(
                        city.books[other],
                        ep.limits[other],
                        n,
                    ) is not None:
                        outcome = "both"

                    else:
                        outcome = "exposed"

                    city.sequential[outcome] += 1
                    statuses.append(
                        f"{n}:{outcome}"
                    )

                print(
                    f"[{now_et()}] "
                    f"SEQUENTIAL-SHADOW "
                    f"{ep.city} #{ep.id} "
                    f"first@250ms other@500ms "
                    + " ".join(statuses),
                    flush=True,
                )

            if (
                len(ep.checked) == len(DELAYS)
                and ep.seq_completed
            ):
                self.pending.remove(ep)

    def report(self):
        print(
            f"\n[{now_et()}] HEARTBEAT "
            f"updates={self.total_updates} "
            f"signals={self.candidates} "
            f"pending={len(self.pending)}",
            flush=True,
        )

        for c in self.city.values():
            ready = sum(
                book.ready
                for book in c.books.values()
            )

            active = (
                f"SIGNAL #{c.active.id}"
                if c.active else "PASS"
            )

            n500, k500 = c.stats[500]

            print(
                f"  {c.code:<3} "
                f"books={ready:>2}/{len(c.books):<2} "
                f"updates={c.updates:<6} "
                f"{c.last_pair:<22} "
                f"{active:<10} "
                f"checked500ms={k500}/{n500}",
                flush=True,
            )


def groups(items, n):
    for idx in range(0, len(items), n):
        yield items[idx:idx + n]


def activation_schedule(codes):
    """Identify cities eligible after 13:00 local time.

    This is called by the connection scheduler,
    never by the per-order-book evaluation path.
    """
    active = []
    waiting = []

    for code in codes:
        timezone = ZoneInfo(CITY[code][2])
        local_now = datetime.now(timezone)

        local_gate = local_now.replace(
            hour=14,
            minute=0,
            second=0,
            microsecond=0,
        )

        if local_now >= local_gate:
            active.append(code)
        else:
            seconds_remaining = max(
                0.0,
                local_gate.timestamp()
                - local_now.timestamp(),
            )

            waiting.append((
                seconds_remaining,
                code,
                local_gate.astimezone(ET).strftime(
                    "%I:%M %p ET"
                ),
            ))

    waiting.sort()
    return tuple(active), waiting


async def discover(codes):
    results = await asyncio.gather(
        *(
            asyncio.to_thread(fetch_event, code)
            for code in codes
        )
    )

    found = {}

    for code, result, error in results:
        if error:
            print(
                f"[{now_et()}] SKIP {code}: {error}",
                flush=True,
            )
            continue

        event, day, tickers = result
        found[code] = result

        print(
            f"[{now_et()}] {code}: "
            f"{event} "
            f"{len(tickers)} open markets",
            flush=True,
        )

    return found


async def run_session(found, cfg, next_gate_deadline=None):
    tracker = Tracker(found, cfg)
    tickers = list(tracker.market_to_city)

    batches = list(groups(
        tickers,
        cfg.markets_per_subscription,
    ))

    header_key = (
        "additional_headers"
        if "additional_headers" in
        inspect.signature(
            websockets.connect
        ).parameters
        else "extra_headers"
    )

    async with websockets.connect(
        WS_URL,
        **{header_key: auth_headers()},
        ping_interval=20,
        ping_timeout=20,
        max_queue=8192,
    ) as ws:

        for idx, batch in enumerate(batches, 1):
            await ws.send(json.dumps({
                "id": idx,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": batch,
                    "use_yes_price": False,
                },
            }))

        print(
            f"[{now_et()}] WebSocket connected; "
            f"{len(tickers)} markets "
            f"across {len(found)} cities, "
            f"{len(batches)} subscriptions",
            flush=True,
        )

        last_report = time.monotonic()
        start = last_report

        while True:
            try:
                raw = await asyncio.wait_for(
                    ws.recv(),
                    timeout=0.02,
                )

            except asyncio.TimeoutError:
                raw = None

            if raw is not None:
                packet = json.loads(raw)
                kind = packet.get("type")

                if kind == "error":
                    raise RuntimeError(
                        f"WS subscription error: "
                        f"{packet.get('msg')}"
                    )

                if kind in (
                    "orderbook_snapshot",
                    "orderbook_delta",
                ):
                    tracker.apply(packet)

                elif kind == "subscribed":
                    print(
                        f"[{now_et()}] "
                        f"WS subscription acknowledged: "
                        f"{packet.get('msg')}",
                        flush=True,
                    )

            tracker.check_delays()

            now = time.monotonic()

            if now - last_report >= cfg.heartbeat:
                tracker.report()
                last_report = now

                date_changed = any(
                    datetime.now(
                        ZoneInfo(CITY[code][2])
                    ).date() != state.local_date
                    for code, state in tracker.city.items()
                )

                if date_changed:
                    print(
                        f"[{now_et()}] "
                        "Local date changed; refreshing",
                        flush=True,
                    )
                    return

            if (
                next_gate_deadline is not None
                and now >= next_gate_deadline
            ):
                print(
                    f"[{now_et()}] "
                    "Next city reached 1 PM local; "
                    "refreshing subscriptions",
                    flush=True,
                )
                tracker.report()
                return

            if (
                now - start
                > cfg.refresh_minutes * 60
            ):
                print(
                    f"[{now_et()}] "
                    "Scheduled event/connection refresh",
                    flush=True,
                )
                tracker.report()
                return



def args_parser():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--cities",
        default=DEFAULT_CITIES,
        help="Comma-separated city codes",
    )
    parser.add_argument(
        "--cap-cents",
        type=str,
        default="96",
    )
    parser.add_argument(
        "--sizes",
        default="1,5,10,20,50,100",
    )
    parser.add_argument(
        "--max-evaluate",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=15,
    )
    parser.add_argument(
        "--markets-per-subscription",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--refresh-minutes",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--capacity-log-seconds",
        type=float,
        default=5,
    )

    cfg = parser.parse_args()

    cfg.cap = D(cfg.cap_cents) / D(100)

    cfg.sizes = tuple(sorted({
        int(x) for x in cfg.sizes.split(",")
    }))

    cfg.codes = tuple(dict.fromkeys(
        x.strip().upper()
        for x in cfg.cities.split(",")
    ))

    if (
        not cfg.codes
        or any(c not in CITY for c in cfg.codes)
    ):
        parser.error(
            f"Supported city codes: {', '.join(CITY)}"
        )

    if (
        any(n < 1 for n in cfg.sizes)
        or cfg.max_evaluate < 1
    ):
        parser.error(
            "Sizes must be positive whole contracts"
        )

    if not D("0.01") < cfg.cap < D(1):
        parser.error(
            "--cap-cents must be between 1 and 100"
        )

    if (
        cfg.heartbeat < 1
        or not 1 <= cfg.markets_per_subscription <= 100
    ):
        parser.error(
            "Heartbeat >= 1; "
            "markets-per-subscription 1..100"
        )

    if cfg.refresh_minutes < 1:
        parser.error(
            "Refresh minutes must be >= 1"
        )

    return cfg


async def main():
    cfg = args_parser()

    print("MULTI-CITY DAILY HIGH BASKET SHADOW SCANNER")
    print(
        f"Cities={','.join(cfg.codes)} "
        f"| cap < {cfg.cap*100:.2f}c before fees"
    )
    print(
        f"Sizes={cfg.sizes} "
        f"| dynamic maximum evaluation "
        f"<= {cfg.max_evaluate} pairs"
    )
    print(
        "Evaluates continuously on relevant "
        "WebSocket order-book updates"
    )
    print(
        "READ ONLY: NO ORDERS, DB WRITES OR CSV WRITES. "
        "Ctrl+C to stop."
    )

    backoff = 1

    while True:
        try:
            eligible, waiting = activation_schedule(
                cfg.codes
            )

            if not eligible:
                delay, next_city, next_time = waiting[0]

                print(
                    f"[{now_et()}] WAITING: "
                    f"no cities past 1 PM local. "
                    f"Next={next_city} at {next_time}",
                    flush=True,
                )

                await asyncio.sleep(max(0.05, delay))
                continue

            print(
                f"[{now_et()}] ACTIVE CITIES: "
                f"{','.join(eligible)}",
                flush=True,
            )

            if waiting:
                print(
                    f"[{now_et()}] WAITING CITIES: "
                    + ", ".join(
                        f"{code} ({when})"
                        for _, code, when in waiting
                    ),
                    flush=True,
                )

            # Only eligible cities reach event discovery.
            found = await discover(eligible)

            if not found:
                print(
                    f"[{now_et()}] "
                    "No eligible active events; "
                    "retrying discovery",
                    flush=True,
                )
                await asyncio.sleep(60)
                continue

            # Recompute after HTTP discovery, so a city
            # opening during discovery is not missed.
            updated_eligible, updated_waiting = (
                activation_schedule(cfg.codes)
            )

            if updated_eligible != eligible:
                continue

            next_gate_deadline = None

            if updated_waiting:
                next_gate_deadline = (
                    time.monotonic()
                    + updated_waiting[0][0]
                )

            await run_session(
                found,
                cfg,
                next_gate_deadline,
            )
            backoff = 1

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            print(
                f"[{now_et()}] ERROR "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print(
                f"[{now_et()}] Reconnecting after "
                f"{backoff}s with fresh books",
                flush=True,
            )

            await asyncio.sleep(backoff)
            backoff = min(30, backoff * 2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped (read-only).")
