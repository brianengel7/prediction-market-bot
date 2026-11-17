#!/usr/bin/env python3
# Copy 1% of every eligible wallet BUY's dollar cost.
# Copy SELLs proportional to wallet inventory reductions.
# Historical candle proxy. No Kalshi orders.
import bisect, json, runpy
from collections import defaultdict, Counter
from decimal import Decimal, ROUND_DOWN, ROUND_CEILING
from pathlib import Path
from types import SimpleNamespace

R = Path.home() / "sports_copy_research"
D = lambda x: Decimal(str(x))
STEP = D(".01")

def fee(q, p):
    return (D(".07") * q * p * (1-p)).quantize(
        D(".01"), rounding=ROUND_CEILING
    )

def down(q):
    return q.quantize(STEP, rounding=ROUND_DOWN)

def purchase(budget, p):
    # Maximum contracts within target including modeled fees.
    lo, hi = 0, int(budget / (STEP*p))
    while lo < hi:
        n = (lo+hi+1)//2
        q = STEP*n
        if q*p + fee(q,p) <= budget:
            lo = n
        else:
            hi = n-1
    return STEP*lo

assert purchase(D("1"), D(".5")) == D("1.92")
assert purchase(D(".01"), D(".5")) == 0

v8 = runpy.run_path(
    "src/sports-copy/breakthebank_v8_book_replay_readonly.py",
    run_name="library"
)
v20 = runpy.run_path(
    "src/sports-copy/breakthebank_v20_full_kalshi.py",
    run_name="library"
)

a = SimpleNamespace(
    v1_input_dir=R/"breakthebank_v1",
    v3_input_dir=R/"breakthebank_v3",
    scale=.01,
    min_age=1,
    min_notional=0.,
    min_shares=0.,
    batch_window=60
)

orders, st, *_ = v8["source_orders"](a)
_, matches, unmatched, _ = v20["load"]()
assert st["discarded_wallet_fills"] == 0

eligible = [
    o for o in orders
    if o["market"].lower() in matches
]

print("Source eligible fills:", len(orders))
print("Matched fills:", len(eligible))
print("Unmapped fills:", len(orders)-len(eligible))
print("Source fills discarded by threshold:",
      st["discarded_wallet_fills"])

by_ticker = defaultdict(list)
for o in eligible:
    ticker = matches[o["market"].lower()]["ticker"]
    by_ticker[ticker].append(o)

books = {}
failures = {}

for i, (ticker, rows) in enumerate(
    sorted(by_ticker.items()), 1
):
    start = int(min(o["ts"] for o in rows)) + 5
    end = int(max(o["ts"] for o in rows)) + 60
    path = R/"breakthebank_v23/candles"/(ticker+".json")

    try:
        if path.exists():
            data = json.loads(path.read_text())
            if data["window"] != [start, end]:
                raise ValueError("Cache interval changed")
            quotes = {
                int(t): tuple(x)
                for t,x in data["quotes"].items()
            }
        else:
            _, quotes = v20["v14"]["candle_history"](
                ticker, start, end
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({
                "window": [start,end],
                "quotes": quotes
            }))
            temp.replace(path)

        books[ticker] = (sorted(quotes), quotes)

    except Exception as exc:
        failures[ticker] = str(exc)

    if i % 20 == 0 or i == len(by_ticker):
        print(
            "Markets checked:", i, "/",len(by_ticker),
            "data errors:",len(failures),
            flush=True
        )

wallet = defaultdict(lambda: D(0))
ours = defaultdict(lambda: D(0))
ledger = defaultdict(lambda: dict(
    cost=D(0), fee=D(0),
    proceeds=D(0), payout=D(0),
    buys=0, sells=0
))
reject = Counter()

for _,o in sorted(
    enumerate(eligible),
    key=lambda z:(z[1]["ts"],z[1]["seq"],z[0])
):
    cid = o["market"].lower()
    m = matches[cid]
    ticker = m["ticker"]
    size = D(o["wallet_qty"])
    wallet_price = D(o["wallet_price"])
    side = o["side"]

    previous = wallet[cid]
    wallet[cid] = (
        previous+size if side=="BUY"
        else max(D(0),previous-size)
    )

    if ticker not in books:
        reject["NO_CANDLES"] += 1
        continue

    times, quotes = books[ticker]
    order_ts = float(o["ts"])+5
    index = bisect.bisect_right(times,order_ts-10)-1

    if index < 0:
        reject["NO_ASOF_QUOTE"] += 1
        continue

    if order_ts-times[index] > 75:
        reject["STALE_QUOTE"] += 1
        continue

    bid, ask = quotes[times[index]]
    price = v20["price"](bid,ask,side,m["side"])

    if price is None or not 0 < price < 1:
        reject["NO_RELEVANT_BID_ASK"] += 1
        continue

    p = D(price)
    state = ledger[cid]

    if side == "BUY":
        # Exact requested rule:
        # $100 wallet wager -> $1 total Kalshi budget.
        budget = size*wallet_price*D(".01")
        quantity = purchase(budget,p)

        if quantity == 0:
            reject["BELOW_EXCHANGE_SIZE_OR_FEE"] += 1
            continue

        ours[cid] += quantity
        state["cost"] += quantity*p
        state["fee"] += fee(quantity,p)
        state["buys"] += 1

    else:
        if previous <= 0 or ours[cid] <= 0:
            reject["NO_MIRRORED_INVENTORY"] += 1
            continue

        # Sell the same fraction of our position
        # as the source wallet sells of its position.
        fraction = min(D(1),size/previous)
        quantity = down(ours[cid]*fraction)

        if quantity == 0:
            reject["SELL_BELOW_0.01_CONTRACT"] += 1
            continue

        ours[cid] -= quantity
        state["proceeds"] += quantity*p
        state["fee"] += fee(quantity,p)
        state["sells"] += 1

for cid,m in matches.items():
    ledger[cid]["payout"] = (
        ours[cid]*D(m["kalshi_payout"])
    )

cost = sum(
    (s["cost"]+s["fee"] for s in ledger.values()),
    D(0)
)
pnl = sum(
    (
        s["proceeds"]+s["payout"]
        -s["cost"]-s["fee"]
        for s in ledger.values()
    ),
    D(0)
)

by_kind = defaultdict(lambda:[D(0),D(0),0])
for cid,s in ledger.items():
    kind = matches[cid]["sport"]+"_"+matches[cid]["kind"]
    row = by_kind[kind]
    row[0] += s["proceeds"]+s["payout"]-s["cost"]-s["fee"]
    row[1] += s["cost"]+s["fee"]
    row[2] += s["buys"]

print("\nV23 — 1% PROPORTIONAL CAPITAL COPY")
print("Modeled buy executions:",
      sum(s["buys"] for s in ledger.values()))
print("Modeled sell executions:",
      sum(s["sells"] for s in ledger.values()))
print("Total purchase cost + fees: $",round(cost,2))
print("Net P&L: $",round(pnl,2))
print("ROI:",round(100*pnl/cost,3) if cost else None,"%")

print("\nBY SPORTS CATEGORY")
for kind,(profit,spent,buys) in sorted(by_kind.items()):
    roi = round(100*profit/spent,3) if spent else None
    print(kind,"buys:",buys,
          "PnL: $",round(profit,2),"ROI:",roi,"%")

print("\nUNFILLED SIGNALS")
print(dict(reject))
print("Unmatched markets:",len(unmatched))
print("Candle data failures:",failures)

print("\nNo order, market, daily, or category caps.")
print("No arbitrary 2-cent drift rejection.")
print("Historical candle prices; NOT guaranteed live fills.")
print("No account balance or peak-exposure constraint modeled.")
print("PUBLIC GET ONLY — NO ORDERS OR PRIVATE KEYS")
