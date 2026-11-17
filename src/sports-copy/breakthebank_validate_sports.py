#!/usr/bin/env python3
"""Audit every V8 sports market against Kalshi. PUBLIC GET ONLY."""

import argparse
import csv
import re
import runpy
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

V14 = runpy.run_path(
    "src/sports-copy/breakthebank_v14_kalshi_crossvenue_readonly.py",
    run_name="v14_library",
)
get = V14["get"]
KAL = V14["KAL"]
POLY = V14["POLY"]

PATTERN = re.compile(
    r"^(nba|nfl)-([a-z0-9]+)-([a-z0-9]+)-"
    r"(20\d{2}-\d{2}-\d{2})(?:-(.+))?$", re.I
)
SPREAD = re.compile(r"spread-(home|away)-(\d+)pt5$", re.I)
TOTAL = re.compile(r"total-(\d+)pt5$", re.I)

COLUMNS = [
    "poly_market", "poly_slug", "sport", "kind",
    "poly_question", "poly_token", "poly_outcome",
    "intents", "kalshi_ticker", "kalshi_event",
    "kalshi_title", "kalshi_rules", "match_status", "notes",
]


def classify(slug):
    match = PATTERN.fullmatch(slug or "")
    if not match:
        return "OTHER", "OTHER", None

    league, away, home, day, suffix = match.groups()
    league = league.lower()
    suffix = (suffix or "").lower()

    if not suffix:
        kind = "GAME"
    elif SPREAD.fullmatch(suffix):
        kind = "SPREAD"
    elif TOTAL.fullmatch(suffix):
        kind = "TOTAL"
    else:
        kind = "OTHER"

    return league.upper(), kind, (
        league, away.lower(), home.lower(), day, suffix
    )


def get_event_markets(league, away, home, day, kind, cache):
    base = "KXNBAGAME" if league == "nba" else "KXNFLGAME"
    series = base.replace("GAME", kind)
    found = {}

    for event, _, _ in V14["calc_tickers"](
        league, away, home, day, away
    ):
        event = event.replace(base, series, 1)

        if event not in cache:
            data = get(
                KAL,
                "/events/" + quote(event, safe=""),
                {"with_nested_markets": "true"},
                not_found=True,
            )
            markets = []
            if isinstance(data, dict):
                markets = data.get("markets", [])
                if not markets:
                    markets = (data.get("event") or {}).get(
                        "markets", []
                    )
            cache[event] = [
                m for m in markets
                if isinstance(m, dict)
                and m.get("event_ticker") == event
            ]
            time.sleep(0.04)

        for market in cache[event]:
            if market.get("ticker"):
                found[market["ticker"]] = market

    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path.home() /
        "sports_copy_research/breakthebank_v8/intents.csv",
    )
    parser.add_argument(
        "--trades", type=Path,
        default=Path.home() /
        "sports_copy_research/breakthebank_v1/all_trades.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path.home() /
        "sports_copy_research/breakthebank_v17/"
        "all_sports_market_audit.csv",
    )
    args = parser.parse_args()

    if args.self_test:
        assert classify(
            "nba-chi-mil-2026-10-23-spread-home-1pt5"
        )[:2] == ("NBA", "SPREAD")
        assert classify(
            "nfl-dal-gb-2026-10-19-total-53pt5"
        )[:2] == ("NFL", "TOTAL")
        assert classify(
            "nba-bos-det-2026-10-20"
        )[:2] == ("NBA", "GAME")
        print("PASS: classification tests; OFFLINE")
        return

    intents, source = V14["read_inputs"](
        args.manifest, args.trades
    )

    markets = []
    categories = Counter()

    for cid, orders in sorted(
        intents.items(),
        key=lambda pair: min(
            float(r["wallet_trade_ts_utc"])
            for r in pair[1]
        ),
    ):
        slug = V14["extract_slug"](source[cid])
        sport, kind, parsed = classify(slug)
        categories[(sport, kind)] += 1
        markets.append(
            (cid, orders, slug, sport, kind, parsed)
        )

    print(
        "TOTAL SOURCE SPORTS MARKETS:", len(markets),
        "| INTENTS:", sum(len(x[1]) for x in markets)
    )

    for (sport, kind), count in sorted(categories.items()):
        print(f"{sport:8} {kind:8} {count:4}")

    if args.inventory:
        print("\nOTHER CONTRACTS REQUIRING CLASSIFICATION:")
        for _, orders, slug, sport, kind, _ in markets:
            if kind == "OTHER":
                print(sport, len(orders), slug)
        print("\nOFFLINE — NO REQUESTS, WRITES OR ORDERS")
        return

    if args.out.exists():
        raise SystemExit("Output already exists; use a NEW --out path")
    if str(args.out.resolve()).startswith("/workspaces/"):
        raise SystemExit("Output must be outside repository")

    event_cache = {}
    kalshi_cache = {}
    output = []
    statuses = Counter()

    for index, (
        cid, orders, slug, sport, kind, parsed
    ) in enumerate(markets, 1):

        info = {}
        if parsed and sport in ("NBA", "NFL"):
            try:
                info = get(
                    POLY,
                    "/markets/" + quote(cid, safe=""),
                    not_found=True,
                ) or {}
                if str(
                    info.get("condition_id", "")
                ).lower() != cid:
                    info = {}
            except Exception as exc:
                print(
                    "CLOB ERROR", cid[:12],
                    type(exc).__name__, flush=True
                )

        if slug:
            try:
                gamma = get(
                    "https://gamma-api.polymarket.com",
                    "/markets", {"slug": slug},
                    not_found=True,
                )
                exact = [
                    m for m in gamma
                    if isinstance(m, dict)
                    and str(m.get("conditionId", "")).lower() == cid
                ] if isinstance(gamma, list) else []
                if len(exact) == 1:
                    info["question"] = (
                        exact[0].get("question")
                        or info.get("question", "")
                    )
            except Exception as exc:
                print(
                    "GAMMA ERROR", cid[:12],
                    type(exc).__name__, flush=True
                )

        token_map = {
            str(t.get("token_id")): str(t.get("outcome", ""))
            for t in info.get("tokens", [])
            if isinstance(t, dict)
        }
        source_tokens = {
            str(r["token_id"]) for r in orders
        }

        candidates = []
        notes = ""

        if not parsed or sport not in ("NBA", "NFL"):
            status = "OTHER_SPORT_REVIEW"
            notes = "Requires sport-specific Kalshi mapping"

        elif kind == "OTHER":
            status = "OTHER_CONTRACT_REVIEW"
            notes = "Contract type not yet safely classified"

        elif not token_map or not source_tokens.issubset(token_map):
            status = "POLY_TOKEN_UNVERIFIED"
            notes = "Original outcome tokens could not be verified"

        else:
            league, away, home, day, suffix = parsed

            if (
                away not in V14["TEAMS"][league]
                or home not in V14["TEAMS"][league]
            ):
                status = "UNKNOWN_TEAM"
                notes = "Team abbreviation requires review"

            else:
                try:
                    if kind == "GAME":
                        for token in source_tokens:
                            team = V14["team_from_outcome"](
                                token_map[token],
                                league, away, home
                            )
                            if team is None:
                                continue
                            matched = V14["find_kalshi"](
                                league, away, home, day,
                                team, kalshi_cache
                            )
                            for ticker, (market, _) in matched.items():
                                candidates.append((token, market))

                        if len(candidates) != len(source_tokens):
                            status = "NO_UNIQUE_GAME_MATCH"
                            notes = "Exact team/outcome match not found"
                        elif league == "nfl":
                            status = "NFL_TIE_RULE_REVIEW"
                            notes = "Exact team; verify tie payout rules"
                        else:
                            status = "NBA_GAME_RULE_REVIEW"
                            notes = "Exact team; verify settlement rules"

                    else:
                        available = get_event_markets(
                            league, away, home, day,
                            kind, event_cache
                        )
                        match = (
                            TOTAL.fullmatch(suffix)
                            if kind == "TOTAL"
                            else SPREAD.fullmatch(suffix)
                        )
                        threshold = int(
                            match.group(
                                1 if kind == "TOTAL" else 2
                            )
                        ) + 1

                        def kal_code(team):
                            return V14["KAL_CODES"][league].get(
                                team, team.upper()
                            )

                        if kind == "TOTAL":
                            endings = {str(threshold)}
                        else:
                            endings = {
                                kal_code(away) + str(threshold),
                                kal_code(home) + str(threshold),
                            }

                        matched = [
                            m for ticker, m in available.items()
                            if ticker.rsplit("-", 1)[-1] in endings
                        ]

                        for token in source_tokens:
                            for market in matched:
                                candidates.append((token, market))

                        if matched:
                            status = (
                                "SPREAD_LINE_REVIEW"
                                if kind == "SPREAD"
                                else "TOTAL_LINE_REVIEW"
                            )
                            notes = (
                                "Candidate strike only; verify handicap "
                                "sign, YES/NO direction and settlement"
                            )
                        else:
                            status = "NO_KALSHI_LINE_CANDIDATE"
                            notes = "No equivalent strike identified"

                except Exception as exc:
                    status = "LOOKUP_ERROR"
                    notes = (
                        type(exc).__name__ + ": " + str(exc)[:120]
                    )

        if not candidates:
            candidates = [("", {})]

        for token, market in candidates:
            relevant_orders = (
                [
                    r for r in orders
                    if str(r["token_id"]) == token
                ]
                if token else orders
            )

            output.append({
                "poly_market": cid,
                "poly_slug": slug or "",
                "sport": sport,
                "kind": kind,
                "poly_question": info.get("question", ""),
                "poly_token": token,
                "poly_outcome": token_map.get(token, ""),
                "intents": len(relevant_orders),
                "kalshi_ticker": market.get("ticker", ""),
                "kalshi_event": market.get("event_ticker", ""),
                "kalshi_title": market.get("title", ""),
                "kalshi_rules": market.get("rules_primary", ""),
                "match_status": status,
                "notes": notes,
            })

        statuses[status] += 1

        if index % 10 == 0 or index == len(markets):
            print(
                f"Checked {index}/{len(markets)}",
                dict(statuses),
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(output)

    print("\nFINAL MARKET STATUS COUNTS:")
    for key, count in sorted(statuses.items()):
        print(f"{key:35} {count}")

    print("\nAudit CSV:", args.out)
    print("CANDIDATE MATCHING ONLY — not trade certification.")
    print("PUBLIC GET ONLY. NO ORDERS OR CREDENTIALS.")


if __name__ == "__main__":
    main()
