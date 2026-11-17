import csv, json, math, re, runpy
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

root = Path.home() / "sports_copy_research"
v14 = runpy.run_path(
    "src/sports-copy/breakthebank_v14_kalshi_crossvenue_readonly.py",
    run_name="v14_lib"
)

def read(p):
    with p.open(newline="") as f:
        return list(csv.DictReader(f))

def payoff(result, side, settled):
    try:
        if settled not in ("", None):
            val = float(settled)
            if math.isfinite(val) and 0 <= val <= 1:
                return val if side == "YES" else 1 - val
    except (ValueError, TypeError):
        pass
    if result not in ("yes", "no"):
        return None
    val = float(result == "yes")
    return val if side == "YES" else 1 - val

def main():
    v17 = read(root / "breakthebank_v17/all_sports_market_audit.csv")
    v18 = read(root / "breakthebank_v18/direction_candidates.csv")

    bycid = defaultdict(list)
    for r in v17:
        bycid[r["poly_market"].lower()].append(r)

    chosen = {}
    for r in v18:
        if r["status"] != "DIRECTION_CANDIDATE":
            continue
        cid = r["poly_market"].lower()
        assert cid not in chosen and r["kalshi_side"] in ("YES", "NO")
        matches = [
            x for x in bycid[cid]
            if x["poly_token"] == r["poly_token"]
            and x["kalshi_ticker"] == r["kalshi_ticker"]
        ]
        assert len(matches) == 1, ("V17/V18 mismatch", cid)
        chosen[cid] = (matches[0], r["kalshi_side"])

    for cid, group in bycid.items():
        if cid in chosen:
            continue
        games = [
            r for r in group
            if r["match_status"] in (
                "NBA_GAME_RULE_REVIEW", "NFL_TIE_RULE_REVIEW"
            )
            and r["poly_token"] and r["kalshi_ticker"]
        ]
        if not games:
            continue
        assert len(games) == 1, ("ambiguous game mapping", cid)

        r = games[0]
        league = r["sport"].lower()
        m = re.fullmatch(
            r"(nba|nfl)-([a-z0-9]+)-([a-z0-9]+)-\d{4}-\d{2}-\d{2}",
            r["poly_slug"]
        )
        assert m and league == m.group(1)
        team = v14["team_from_outcome"](
            r["poly_outcome"], league, m.group(2), m.group(3)
        )
        assert team
        code = v14["KAL_CODES"][league].get(team, team.upper())
        assert r["kalshi_ticker"].endswith("-" + code)
        chosen[cid] = (r, "YES")

    assert len(chosen) == 128, (
        "Expected 73 spreads/totals + 55 game winners",
        len(chosen)
    )

    idx = defaultdict(set)
    with (root / "breakthebank_v1/all_trades.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            cid = str(x.get("condition_id", "")).lower()
            if cid in chosen:
                idx[(cid, str(x.get("token_id")))].add(
                    str(x.get("outcome_index"))
                )

    resolution = {}
    with (root / "breakthebank_v3/resolutions.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            cid = x["condition_id"].lower()
            for item in x["response"]:
                if str(item.get("status", "")).lower() != "resolved":
                    continue
                p = item.get("payouts")
                if isinstance(p, str):
                    p = json.loads(p)
                if not isinstance(p, list) or len(p) != 2:
                    continue
                p = tuple(float(y) / 1000000 for y in p)
                if abs(sum(p) - 1) < 1e-8:
                    resolution[cid] = p

    if not resolution:
        raise RuntimeError("Empty Polymarket resolution cache")

    cache = {}
    rows = []
    stats = Counter()

    for i, (cid, (r, side)) in enumerate(sorted(chosen.items()), 1):
        token = r["poly_token"]
        indices = idx.get((cid, token), set())
        pp = resolution.get(cid)
        poly = (
            pp[int(next(iter(indices)))]
            if pp is not None and len(indices) == 1
            and next(iter(indices)) in ("0", "1")
            else None
        )

        ticker = r["kalshi_ticker"]
        if ticker not in cache:
            try:
                market = None
                for prefix in ("/markets/", "/historical/markets/"):
                    payload = v14["get"](
                        v14["KAL"], prefix + quote(ticker, safe=""),
                        not_found=True
                    )
                    if isinstance(payload, dict) and isinstance(
                        payload.get("market"), dict
                    ):
                        market = payload["market"]
                        break
                if market and market.get("ticker") != ticker:
                    raise RuntimeError("Ticker mismatch")
                cache[ticker] = (market, "")
            except Exception as exc:
                cache[ticker] = (
                    None, type(exc).__name__ + ": " + str(exc)[:100]
                )

        market, error = cache[ticker]
        result = str(market.get("result") or "").lower() if market else ""
        settled = market.get("settlement_value_dollars") if market else ""
        kp = payoff(result, side, settled)

        if error:
            status = "KALSHI_API_ERROR"
        elif not market:
            status = "KALSHI_MARKET_MISSING"
        elif poly is None:
            status = "POLY_UNRESOLVED_OR_TOKEN_UNKNOWN"
        elif kp is None:
            status = "KALSHI_UNRESOLVED"
        elif poly not in (0., 1.) or kp not in (0., 1.):
            status = "FRACTIONAL_PAYOUT_REVIEW"
        elif abs(poly - kp) > 1e-8:
            status = "PAYOUT_MISMATCH"
        else:
            status = "PAYOUT_MATCH_RULES_UNCERTIFIED"

        live_rule = str(market.get("rules_primary") or "") if market else ""
        if (
            status == "PAYOUT_MATCH_RULES_UNCERTIFIED"
            and r["kalshi_rules"]
            and re.sub(r"\s+", " ", r["kalshi_rules"]).strip()
            != re.sub(r"\s+", " ", live_rule).strip()
        ):
            status = "RULE_TEXT_CHANGED_REVIEW"

        row = dict(
            poly_market=cid, slug=r["poly_slug"],
            kind=r["kind"], sport=r["sport"], token=token,
            poly_outcome=r["poly_outcome"],
            ticker=ticker, side=side,
            poly_payout=poly, kalshi_payout=kp,
            kalshi_result=result, status=status, error=error
        )
        rows.append(row)
        stats[status] += 1

        if i % 25 == 0 or i == len(chosen):
            print("Checked", i, "/", len(chosen), flush=True)

    out = root / "breakthebank_v19/settlement_parity.csv"
    if out.exists():
        raise RuntimeError("Output exists; no overwrite: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nV19 PAYOUT COMPARISON (NOT RULE CERTIFICATION)")
    print("Candidates:", len(rows))
    print("By type:", dict(Counter(
        (r["sport"], r["kind"]) for r in rows
    )))
    print("Kalshi sides:", dict(Counter(r["side"] for r in rows)))
    print("Payout statuses:", dict(stats))
    print("Still unmapped original markets:", 144 - len(rows))

    for r in rows:
        if r["status"] != "PAYOUT_MATCH_RULES_UNCERTIFIED":
            print(
                "REVIEW", r["slug"], r["side"], r["ticker"],
                r["status"], "Poly:", r["poly_payout"],
                "Kalshi:", r["kalshi_payout"]
            )

    print("CSV:", out)
    print("PUBLIC GET ONLY — NO ORDERS OR SECRETS")

if __name__ == "__main__":
    main()
