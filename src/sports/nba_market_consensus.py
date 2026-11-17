"""Read-only, prospective NBA sportsbook consensus vs Kalshi order book research.

Usage:
  python -m src.sports.nba_market_consensus scan
  python -m src.sports.nba_market_consensus evaluate

No Kalshi credentials, positions, or order-placement calls are used. A quote-derived
edge is an estimate, NOT a guaranteed trade or a verified predictive advantage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import requests

from src.kalshi.fees import calculate_taker_fee

SPORTS_URL = "https://api.sportsgameodds.com/v2/events"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBAGAME"
NY = ZoneInfo("America/New_York")
OUT = Path("data/sports/nba_market_consensus_snapshots.jsonl")
HOME_ID = "points-home-game-ml-home"
AWAY_ID = "points-away-game-ml-away"
FULL_NAMES = dict(zip(
    ("Atlanta Hawks|Boston Celtics|Brooklyn Nets|Charlotte Hornets|Chicago Bulls|"
     "Cleveland Cavaliers|Dallas Mavericks|Denver Nuggets|Detroit Pistons|"
     "Golden State Warriors|Houston Rockets|Indiana Pacers|Los Angeles Clippers|"
     "Los Angeles Lakers|Memphis Grizzlies|Miami Heat|Milwaukee Bucks|"
     "Minnesota Timberwolves|New Orleans Pelicans|New York Knicks|"
     "Oklahoma City Thunder|Orlando Magic|Philadelphia 76ers|Phoenix Suns|"
     "Portland Trail Blazers|Sacramento Kings|San Antonio Spurs|Toronto Raptors|"
     "Utah Jazz|Washington Wizards").split("|"),
    "ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split()
))
TEAM_SET = frozenset(FULL_NAMES.values())
ALIASES = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP",
           "PHO": "PHX", "UTAH": "UTA", "BRK": "BKN", "WSH": "WAS"}


def canonical_team(raw: str) -> str:
    s = " ".join(str(raw).strip().split())
    if s in FULL_NAMES:
        return FULL_NAMES[s]
    code = ALIASES.get(s.upper(), s.upper())
    if code in TEAM_SET:
        return code
    # Team IDs supplied by SportsGameOdds can be e.g. BOSTON_CELTICS_NBA.
    return FULL_NAMES.get(s.replace("_", " ").title().removesuffix(" Nba"), "")


def parse_time(s: str) -> datetime:
    parsed = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return parsed.astimezone(timezone.utc)


def odds_probability(x) -> float:
    s = str(x).replace("+", "").strip()
    odds = float(s)
    if odds >= 100:
        return 100 / (100 + odds)
    if odds <= -100:
        return -odds / (100 - odds)
    raise ValueError(f"Invalid American odds: {x!r}")


def book_consensus(event: dict, now: datetime, max_quote_age: int,
                   min_books: int, whitelist: set[str] | None = None) -> dict | None:
    odds = event.get("odds") or {}
    h = (odds.get(HOME_ID) or {}).get("byBookmaker") or {}
    a = (odds.get(AWAY_ID) or {}).get("byBookmaker") or {}
    books = []
    for bookmaker in sorted(set(h) & set(a)):
        if whitelist and bookmaker.lower() not in whitelist:
            continue
        bh, ba = h[bookmaker], a[bookmaker]
        if not bh.get("available", False) or not ba.get("available", False):
            continue
        try:
            th, ta = parse_time(bh["lastUpdatedAt"]), parse_time(ba["lastUpdatedAt"])
            ages = [(now - t).total_seconds() for t in (th, ta)]
            if any(not 0 <= age <= max_quote_age for age in ages):
                continue
            ph, pa = odds_probability(bh["odds"]), odds_probability(ba["odds"])
            total = ph + pa
            if not 0.98 <= total <= 1.20:  # Reject anomalous/incorrect pairings.
                continue
            fair_home = ph / total
            books.append({"book": bookmaker, "home_odds": str(bh["odds"]),
                          "away_odds": str(ba["odds"]), "home_fair": fair_home,
                          "latest_quote_utc": max(th, ta).isoformat(),
                          "oldest_quote_utc": min(th, ta).isoformat()})
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    if len(books) < min_books:
        return None
    p = mean(b["home_fair"] for b in books)
    return {"home_probability": p, "away_probability": 1-p,
            "book_count": len(books), "book_range": max(b["home_fair"] for b in books)
            - min(b["home_fair"] for b in books), "books": books}


def event_teams(event: dict) -> tuple[str, str] | None:
    teams = event.get("teams") or {}
    found = []
    for side in ("home", "away"):
        team = teams.get(side) or {}
        names = team.get("names") or {}
        candidates = [names.get("long"), names.get("short"), team.get("teamID")]
        code = next((canonical_team(s) for s in candidates if s and canonical_team(s)), "")
        found.append(code)
    return tuple(found) if all(found) and found[0] != found[1] else None


def parse_market(m: dict):
    ticker = str(m.get("ticker") or "").upper()
    found = re.fullmatch(r"KXNBAGAME-(\d\d[A-Z]{3}\d\d)([A-Z]{4,9})-([A-Z]{2,5})", ticker)
    if not found:
        return None
    try:
        game_date = datetime.strptime(found[1], "%y%b%d").date()
    except ValueError:
        return None
    blob, yes_team = found[2], canonical_team(found[3])
    combos = {frozenset((canonical_team(blob[:i]), canonical_team(blob[i:])))
              for i in range(2, len(blob) - 1)
              if canonical_team(blob[:i]) in TEAM_SET and canonical_team(blob[i:]) in TEAM_SET
              and canonical_team(blob[:i]) != canonical_team(blob[i:])}
    if len(combos) != 1 or yes_team not in next(iter(combos)):
        return None
    return game_date, next(iter(combos)), yes_team


def best_bid(levels: list) -> tuple[float, float] | None:
    valid = []
    for level in levels:
        try:
            price, size = float(level[0]), float(level[1])
        except (TypeError, IndexError, ValueError):
            continue
        if 0 < price < 1 and size > 0:
            valid.append((price, size))
    return max(valid, key=lambda item: item[0]) if valid else None


def book_prices(data: dict) -> dict:
    """Kalshi order books publish bids only; asks are complements of opposite bids."""
    ob = data.get("orderbook_fp") or {}
    y = best_bid(ob.get("yes_dollars") or [])
    n = best_bid(ob.get("no_dollars") or [])
    return {"yes_bid": y[0] if y else None, "no_bid": n[0] if n else None,
            "yes_ask": round(1 - n[0], 4) if n else None,
            "yes_ask_depth": n[1] if n else 0,
            "no_ask": round(1 - y[0], 4) if y else None,
            "no_ask_depth": y[1] if y else 0}


def get_json(session: requests.Session, url: str, *, params=None, headers=None) -> dict:
    r = session.get(url, params=params, headers=headers, timeout=25)
    if not r.ok:
        # Never include an API key or the request URL in errors.
        raise RuntimeError(f"GET {url.split('?')[0].split('/v2/')[0]} HTTP {r.status_code}: {r.text[:300]}")
    obj = r.json()
    if not isinstance(obj, dict):
        raise ValueError("Unexpected non-object API response")
    return obj


def sportsbook_events(session: requests.Session, api_key: str, now: datetime,
                      hours_ahead: float) -> list[dict]:
    all_events, cursor, seen = [], None, set()
    for _ in range(20):
        params = {"leagueID": "NBA", "type": "match", "oddsAvailable": "true",
                  "oddID": f"{HOME_ID},{AWAY_ID}", "limit": 100,
                  "startsAfter": now.isoformat(),
                  "startsBefore": (now + timedelta(hours=hours_ahead)).isoformat()}
        if cursor:
            params["cursor"] = cursor
        result = get_json(session, SPORTS_URL, params=params,
                          headers={"x-api-key": api_key})
        if result.get("success") is not True or not isinstance(result.get("data"), list):
            raise RuntimeError(f"SportsGameOdds response invalid: {result.get('error')}")
        all_events.extend(result["data"])
        cursor = result.get("nextCursor")
        if not cursor:
            return all_events
        if cursor in seen:
            raise RuntimeError("SportsGameOdds repeated pagination cursor")
        seen.add(cursor)
    raise RuntimeError("SportsGameOdds exceeded 20 pages; refusing partial scan")


def kalshi_markets(session: requests.Session) -> list[dict]:
    all_markets, cursor, seen = [], None, set()
    for _ in range(20):
        params = {"series_ticker": SERIES, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        result = get_json(session, f"{KALSHI_URL}/markets", params=params)
        markets = result.get("markets")
        if not isinstance(markets, list):
            raise RuntimeError("Kalshi /markets returned invalid markets")
        all_markets.extend(markets)
        cursor = result.get("cursor")
        if not cursor:
            return all_markets
        if cursor in seen:
            raise RuntimeError("Kalshi repeated pagination cursor")
        seen.add(cursor)
    raise RuntimeError("Kalshi market pages exceeded 20; refusing partial scan")


def choose_route(team: str, fair: float, own: dict, opponent: dict,
                 quantity: int, multiplier: float) -> dict | None:
    """Compare BUY YES(team) and BUY NO(opponent), same NBA payout."""
    paths = []
    for kind, info, ask_col, depth_col in (
        ("YES", own, "yes_ask", "yes_ask_depth"),
        ("NO", opponent, "no_ask", "no_ask_depth"),
    ):
        ask, depth = info.get(ask_col), info.get(depth_col, 0)
        if ask is None or depth < quantity or not 0 < ask < 1:
            continue
        fee = calculate_taker_fee(ask, quantity, multiplier)
        cost = quantity * ask + fee
        ev = quantity * fair - cost
        paths.append({"team": team, "ticker": info["ticker"], "side": kind,
                      "ask": ask, "available_at_best": depth,
                      "quantity": quantity, "fee": fee,
                      "estimated_cost": round(cost, 4),
                      "estimated_ev": round(ev, 4),
                      "estimated_roi": round(ev/cost, 6)})
    return max(paths, key=lambda p: p["estimated_ev"]) if paths else None


def scan(args):
    try:
        from dotenv import load_dotenv  # Optional; many Codespaces use a .env file.
    except ImportError:
        pass
    else:
        load_dotenv()
    api_key = os.getenv("SPORTSGAMEODDS_API_KEY")
    if not api_key:
        raise SystemExit("Missing SPORTSGAMEODDS_API_KEY environment variable")
    if args.quantity < 1 or args.min_books < 1 or args.max_quote_age < 1:
        raise SystemExit("Invalid quantity, bookmaker minimum or quote freshness")
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "nba-consensus-research/1.0"})
    now = datetime.now(timezone.utc)
    events = sportsbook_events(session, api_key, now, args.hours_ahead)
    markets = kalshi_markets(session)
    series = get_json(session, f"{KALSHI_URL}/series/{SERIES}").get("series") or {}
    fee_type, raw_multiplier = series.get("fee_type"), series.get("fee_multiplier")
    if fee_type not in ("quadratic", "quadratic_with_maker_fees") or raw_multiplier is None:
        raise RuntimeError("Unrecognized current fee schedule; refusing to assume fees")
    multiplier = float(raw_multiplier)
    if not math.isfinite(multiplier) or not 0 <= multiplier <= 10:
        raise RuntimeError("Invalid Kalshi fee multiplier")
    by_key = defaultdict(list)
    for m in markets:
        parsed = parse_market(m)
        if parsed:
            d, teams, yes_team = parsed
            by_key[(d, teams)].append((m, yes_team))
    whitelist = {x.strip().lower() for x in args.bookmakers.split(",") if x.strip()} or None
    counts = Counter()
    saved = []
    for ev in events:
        teams = event_teams(ev)
        if not teams:
            counts["unknown_team"] += 1; continue
        home, away = teams
        try:
            start = parse_time(ev.get("startTime") or (ev.get("status") or {}).get("startsAt"))
        except (TypeError, ValueError):
            counts["missing_start"] += 1; continue
        if not args.min_minutes_before <= (start-now).total_seconds()/60 <= args.hours_ahead*60:
            counts["outside_window"] += 1; continue
        status = ev.get("status") or {}
        if any(status.get(x, False) for x in ("started", "live", "cancelled", "ended", "completed")):
            counts["already_started"] += 1; continue
        fair = book_consensus(ev, now, args.max_quote_age, args.min_books, whitelist)
        if not fair:
            counts["insufficient_fresh_books"] += 1; continue
        if fair["book_range"] > args.max_book_disagreement:
            counts["disagreement"] += 1; continue
        day = start.astimezone(NY).date()
        matched = by_key.get((day, frozenset((home, away))), [])
        if len({m.get("event_ticker") for m, _ in matched}) != 1 or not matched:
            counts["missing_or_ambiguous_kalshi_event"] += 1; continue
        if len(matched) != 2 or {t for _, t in matched} != {home, away}:
            counts["missing_kalshi_side"] += 1; continue
        quotes = {}
        for m, team in matched:
            ticker = str(m["ticker"])
            raw = get_json(session, f"{KALSHI_URL}/markets/{ticker}/orderbook")
            quotes[team] = {"ticker": ticker, **book_prices(raw)}
        # All quotes are read-only. Snapshot happens after both Kalshi requests.
        captured = datetime.now(timezone.utc)
        if captured >= start:
            counts["started_during_snapshot"] += 1; continue
        # Recheck quote freshness at the actual completion time, not just start of scan.
        if any(not 0 <= (captured - parse_time(b["oldest_quote_utc"])).total_seconds()
               <= args.max_quote_age for b in fair["books"]):
            counts["expired_during_snapshot"] += 1; continue
        options = []
        for team, p in ((home, fair["home_probability"]),
                        (away, fair["away_probability"])):
            selection = choose_route(team, p, quotes[team],
                                     quotes[away if team == home else home],
                                     args.quantity, multiplier)
            if selection:
                options.append(selection)
        best = max(options, key=lambda x: x["estimated_ev"]) if options else None
        if not best:
            counts["insufficient_depth"] += 1; continue
        if best["estimated_roi"] >= args.min_roi:
            counts["candidate"] += 1
        record = {"snapshot_utc": captured.isoformat(), "event_start_utc": start.isoformat(),
                  "sgo_event_id": str(ev.get("eventID") or ""),
                  "home": home, "away": away, "kalshi_event": matched[0][0]["event_ticker"],
                  "book_home": round(fair["home_probability"], 8),
                  "book_range": round(fair["book_range"], 8), "books": fair["books"],
                  "kalshi_home": quotes[home], "kalshi_away": quotes[away],
                  "kalshi_market_mid_home": (
                      round((quotes[home]["yes_bid"] + (quotes[home]["yes_ask"]))/2, 6)
                      if quotes[home]["yes_bid"] is not None
                      and quotes[home]["yes_ask"] is not None else None),
                  "best": best, "fee_type": fee_type, "fee_multiplier": multiplier,
                  "candidate_min_roi": args.min_roi,
                  "qualified": best["estimated_roi"] >= args.min_roi}
        saved.append(record)
        counts["matched"] += 1
        print(f"{away} @ {home}: books={len(fair['books'])} "
              f"sportsbook_home={fair['home_probability']:.1%} "
              f"Kalshi_mid={record['kalshi_market_mid_home']} "
              f"best={best['side']} {best['ticker']} @ {best['ask']:.3f} "
              f"ROI_est={best['estimated_roi']:+.1%}"
              f"{' [RESEARCH CANDIDATE]' if record['qualified'] else ''}")
    if saved:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as f:
            for item in saved:
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(f"Sportsbook games={len(events)}; Kalshi markets={len(markets)}; "
          f"snapshots_saved={len(saved)}; diagnostics={dict(counts)}")
    print(f"Append-only log: {args.output}; READ-ONLY, NO ORDERS PLACED")


def evaluate(args):
    """Evaluate strictly timed *prospective* observations after results exist."""
    if not args.output.exists():
        print("No snapshot log yet; run scan before evaluate")
        return
    from src.sports.nba_value_model import load_games
    games = load_games(args.games)
    outcomes = {(g["date"], frozenset((g["home"], g["away"]))): g for g in games}
    grouped = defaultdict(list)
    for line in args.output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        when = parse_time(r["snapshot_utc"])
        start = parse_time(r["event_start_utc"])
        cutoff = start - timedelta(minutes=args.decision_minutes_before)
        age = (cutoff - when).total_seconds()
        if not 0 <= age <= args.cutoff_window_seconds:
            continue
        key = (start.astimezone(NY).date(), frozenset((r["home"], r["away"])))
        grouped[key].append(r)
    n, model, market, pnl, paid, qualified = 0, [], [], 0., 0., 0
    for key, snaps in grouped.items():
        g = outcomes.get(key)
        if not g:
            continue
        row = min(snaps, key=lambda r: (parse_time(r["event_start_utc"]) -
                                       timedelta(minutes=args.decision_minutes_before) -
                                       parse_time(r["snapshot_utc"])).total_seconds())
        actual_start = g["start_time"]
        if abs((actual_start - parse_time(row["event_start_utc"])).total_seconds()) > 3600:
            continue  # Scheduled vs actual tipoff differs too much; avoid stale/cancelled games.
        if parse_time(row["snapshot_utc"]) > actual_start - timedelta(minutes=args.decision_minutes_before):
            continue
        y = int(g["home_score"] > g["away_score"])
        ph = float(row["book_home"])
        model.append((ph, y))
        if row.get("kalshi_market_mid_home") is not None:
            market.append((float(row["kalshi_market_mid_home"]), y))
        n += 1
        best = row["best"]
        if row.get("qualified") and best["estimated_roi"] >= args.min_roi:
            cost = float(best["estimated_cost"])
            win = y if best["team"] == row["home"] else 1-y
            pnl += float(best["quantity"])*win-cost
            paid += cost
            qualified += 1
    def scores(xs):
        if not xs: return "no outcomes"
        brier = mean((p-y)**2 for p,y in xs)
        ll = mean(-y*math.log(min(max(p,1e-8),1-1e-8)) -
                  (1-y)*math.log(min(max(1-p,1e-8),1-1e-8)) for p,y in xs)
        return f"N={len(xs)} Brier={brier:.4f} LogLoss={ll:.4f}"
    print(f"Games with outcomes={n}; sportsbook consensus: {scores(model)}")
    print(f"Kalshi midpoint same subset: {scores(market)}")
    print(f"Selected one decision per game, {args.decision_minutes_before} min before tipoff "
          f"(lookback window={args.cutoff_window_seconds}s)")
    print(f"Research selections={qualified}; indicative gross cost=${paid:.2f}; "
          f"indicative net P&L=${pnl:+.2f}; "
          f"ROI={pnl/paid:+.2%}" if paid else "No qualifying resolved selections")
    print("INDICATIVE ONLY: No fills verified, historical model data excluded, "
          "quotes remain subject to venue latency/settlement/fee rules.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    scan_ap = sub.add_parser("scan", help="Snapshot upcoming NBA, read-only")
    scan_ap.add_argument("--hours-ahead", type=float, default=48)
    scan_ap.add_argument("--min-minutes-before", type=float, default=10)
    scan_ap.add_argument("--max-quote-age", type=int, default=600)
    scan_ap.add_argument("--min-books", type=int, default=3)
    scan_ap.add_argument("--max-book-disagreement", type=float, default=0.15)
    scan_ap.add_argument("--quantity", type=int, default=10)
    scan_ap.add_argument("--min-roi", type=float, default=0.05)
    scan_ap.add_argument("--bookmakers", type=str, default="", help="Comma-separated provider IDs (default all)")
    for p in (scan_ap,):
        p.add_argument("--output", type=Path, default=OUT)
    eva = sub.add_parser("evaluate", help="Grade previously captured prospective quotes")
    eva.add_argument("--output", type=Path, default=OUT)
    eva.add_argument("--games", type=Path, default=Path("data/sports/nba_games.csv"))
    eva.add_argument("--decision-minutes-before", type=float, default=30)
    eva.add_argument("--cutoff-window-seconds", type=int, default=300)
    eva.add_argument("--min-roi", type=float, default=0.05)
    args = ap.parse_args()
    if args.command == "scan":
        scan(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
