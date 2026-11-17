#!/usr/bin/env python3
"""Strict Kalshi spread/total direction audit; offline, no orders."""
import csv
import re
import runpy
from pathlib import Path
from collections import Counter, defaultdict

root = Path.home() / "sports_copy_research"
source = root / "breakthebank_v17/all_sports_market_audit.csv"
out = root / "breakthebank_v18/direction_candidates.csv"

if out.exists():
    raise SystemExit("Output already exists; refusing overwrite")

v14 = runpy.run_path(
    "src/sports-copy/breakthebank_v14_kalshi_crossvenue_readonly.py",
    run_name="v14_library"
)

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def team(label, league, away, home):
    return v14["team_from_outcome"](
        str(label), league, away, home
    )

def rule_matches(text, line, kind):
    if kind == "SPREAD":
        pattern = (
            r"win(?:s)?\s+by\s+"
            r"(?:more\s+than|over)\s+(\d+(?:\.5)?)"
        )
    else:
        pattern = (
            r"(?:over|more\s+than)\s+"
            r"(\d+(?:\.5)?)\s+(?:total\s+)?points"
        )

    values = re.findall(pattern, text or "", re.I)
    return any(abs(float(x) - line) < 1e-8 for x in values)

def evaluate(group):
    r = group[0]
    slug = r["poly_slug"].lower()
    kind = r["kind"]
    question = r["poly_question"]
    outcome = r["poly_outcome"]

    m = re.fullmatch(
        r"(nba|nfl)-([a-z0-9]+)-([a-z0-9]+)-"
        r"(\d{4}-\d{2}-\d{2})-(.+)",
        slug
    )

    if kind not in ("SPREAD", "TOTAL"):
        return "OTHER_REVIEW", "", ""
    if not m:
        return "UNSUPPORTED_SLUG", "", ""

    league, away, home, day, suffix = m.groups()

    if kind == "SPREAD":
        q = re.fullmatch(
            r"Spread:\s*(.+?)\s*"
            r"\(([+-])\s*(\d+(?:\.5)?)\)",
            question, re.I
        )
        sm = re.fullmatch(
            r"spread-(?:home|away)-(\d+)pt5", suffix
        )

        if not q or not sm:
            return "SOURCE_QUESTION_UNPARSED", "", ""

        named, sign, value = q.groups()
        line = float(value)

        if abs(line - (int(sm.group(1)) + .5)) > 1e-8:
            return "SOURCE_LINE_MISMATCH", "", ""

        named_team = team(named, league, away, home)
        outcome_team = team(outcome, league, away, home)

        if not named_team or not outcome_team:
            return "SOURCE_TEAM_UNVERIFIED", "", ""

        # A negative handicap means the named team must
        # win by MORE than the line.
        # A positive handicap corresponds to the opponent
        # NOT winning by more than the line.
        trigger_team = (
            named_team if sign == "-"
            else (home if named_team == away else away)
        )

        side = (
            "YES" if outcome_team == trigger_team
            else "NO"
        )

        code = v14["KAL_CODES"][league].get(
            trigger_team, trigger_team.upper()
        )
        target_suffix = code + str(int(line + .5))

    else:
        sm = re.fullmatch(r"total-(\d+)pt5", suffix)
        q = re.search(
            r"(?:O\s*/\s*U|Over\s*/\s*Under|"
            r"Total(?:\s+Points)?)\s*[:\-]?\s*"
            r"(\d+\.5)",
            question, re.I
        )

        if not sm or not q:
            return "SOURCE_QUESTION_UNPARSED", "", ""

        line = float(q.group(1))

        if abs(line - (int(sm.group(1)) + .5)) > 1e-8:
            return "SOURCE_LINE_MISMATCH", "", ""

        if norm(outcome) in ("over", "o"):
            side = "YES"
        elif norm(outcome) in ("under", "u"):
            side = "NO"
        else:
            return "SOURCE_OUTCOME_UNVERIFIED", "", ""

        target_suffix = str(int(line + .5))

    candidates = set()

    for c in group:
        ticker = c["kalshi_ticker"]
        rules = c["kalshi_rules"]

        if not ticker.endswith("-" + target_suffix):
            continue

        if not rule_matches(rules, line, kind):
            continue

        if kind == "SPREAD":
            aliases = v14["TEAMS"][league][
                trigger_team
            ].split("|")

            full = aliases[0]
            city = full.rsplit(" ", 1)[0]
            first = full.split()[0]

            aliases.append(city)
            if len(first) > 4:
                aliases.append(first)

            hit = re.search(
                r"\bif\s+(.+?)\s+wins?\s+by\b",
                rules, re.I
            )
            subject = hit.group(1) if hit else ""

            if not any(
                norm(name) == norm(subject)
                for name in aliases
            ):
                continue

        candidates.add(ticker)

    if len(candidates) != 1:
        return "LINE_OR_RULE_NOT_VERIFIED", "", ""

    return "DIRECTION_CANDIDATE", next(iter(candidates)), side

with source.open(newline="") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
for r in rows:
    groups[(r["poly_market"], r["poly_token"])].append(r)

output = []
counts = Counter()

for (market, token), group in groups.items():
    r = group[0]

    if not token:
        status, ticker, side = "NO_SOURCE_TOKEN", "", ""
    else:
        status, ticker, side = evaluate(group)

    output.append({
        "poly_market": market,
        "poly_slug": r["poly_slug"],
        "sport": r["sport"],
        "kind": r["kind"],
        "poly_token": token,
        "poly_outcome": r["poly_outcome"],
        "poly_question": r["poly_question"],
        "kalshi_ticker": ticker,
        "kalshi_side": side,
        "status": status,
    })
    counts[status] += 1

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=output[0].keys())
    w.writeheader()
    w.writerows(output)

print("\nBREAKTHEBANK V18 — SPREAD/TOTAL DIRECTION AUDIT")
print("Source markets:", len(set(r["poly_market"] for r in rows)))
print("Source outcome groups:", len(groups))

for status, count in counts.most_common():
    print(f"{status:35s} {count:4}")

print("\nVerified-direction candidates:")
for r in output:
    if r["status"] == "DIRECTION_CANDIDATE":
        print(
            r["poly_slug"], "|",
            r["poly_outcome"], "=>",
            r["kalshi_side"],
            r["kalshi_ticker"]
        )

print("\nUnverified spread/total examples:")
for r in [
    x for x in output
    if x["kind"] in ("SPREAD", "TOTAL")
    and x["status"] != "DIRECTION_CANDIDATE"
][:20]:
    print(
        r["poly_slug"], "|",
        r["poly_question"], "|",
        r["poly_outcome"], "|",
        r["status"]
    )

print("\nOutput:", out)
print("Direction audit only; settlement exceptions not certified.")
print("READ ONLY — NO NETWORK OR ORDERS")
