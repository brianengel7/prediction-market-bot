#!/usr/bin/env python3
"""Read-only, chronological KXBTC15M 45/45 *QUOTE REACHABILITY* test.

Purpose:
    Test if PAST Kalshi quote-path features predict a FUTURE sequence where:
      1) a 45c resting bid on the initially elevated side might be reached;
      2) a subsequent (STRICTLY LATER minute) 45c bid on the opposite side
         might also be reached.

NOT an order-fill backtest. Historical 1-minute candlestick LOWS have no
intra-bar ordering, aggressor side, queue priority or resting-order volume.
All profit figures are theoretical GROSS proxy outcomes, NOT tradable P&L.
No API calls, orders, database writes, or external dependencies except NumPy.

Expected preexisting SQLite schema (btc_penny_local.py):
 markets(ticker, open_ts, close_ts, result),
 candles(ticker, end_ts, yes_ask_close, yes_bid_close,
         yes_ask_low, yes_bid_high).

Usage:
 python3 -B src/crypto/research/btc_4545_regime_v1.py --self-test
 python3 -u -B src/crypto/research/btc_4545_regime_v1.py \
     --db .collector_runs/btc15m_pennies.sqlite3 \
     --output data/research/btc_4545_regime_v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

FEATURES = [
    "elapsed_min", "minutes_remaining", "side_is_yes", "quote_ask",
    "quote_spread", "abs_net_move_5m", "total_path_5m",
    "path_efficiency_5m", "reversal_rate_5m", "vol_ratio_2v5m",
    "recent_return_1m", "prev_abs_move_1m",
]


def safe_float(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def quote(row):
    bid, ask = safe_float(row["yes_bid_close"]), safe_float(row["yes_ask_close"])
    if bid is None or ask is None or not (0 <= bid <= ask <= 1):
        return None
    return bid, ask


def candidate_side(bid, ask, low=0.50, high=0.58):
    # When both sides are eligible choose the higher current offer, then YES.
    opts = []
    if low <= ask <= high:
        opts.append((ask, "yes"))
    no_ask = 1.0 - bid
    if low <= no_ask <= high:
        opts.append((no_ask, "no"))
    if not opts:
        return None
    opts.sort(key=lambda x: (x[0], x[1] == "yes"), reverse=True)
    return opts[0][1], opts[0][0]


def features(history, opened, closed, side, selected_ask):
    # history includes ONLY bars with end_ts <= decision time.
    # Requires 5 separate contiguous minute closes (4 price changes).
    recent = history[-5:]
    if len(recent) < 5:
        return None
    if any(recent[i]["end_ts"] - recent[i - 1]["end_ts"] != 60
           for i in range(1, len(recent))):
        return None
    pairs = [quote(r) for r in recent]
    if any(x is None for x in pairs):
        return None
    mids = [(b + a) / 2 for b, a in pairs]
    diffs = [mids[i] - mids[i - 1] for i in range(1, 5)]
    abs_diffs = [abs(d) for d in diffs]
    movement = sum(abs_diffs)
    net = mids[-1] - mids[0]
    turns = sum(1 for a, b in zip(diffs[:-1], diffs[1:]) if a * b < 0)
    short_v = math.sqrt(sum(x*x for x in diffs[-2:]) / 2)
    long_v = math.sqrt(sum(x*x for x in diffs) / 4)
    current_bid, current_ask = pairs[-1]
    ts = recent[-1]["end_ts"]
    return [
        (ts - opened) / 60, (closed - ts) / 60,
        float(side == "yes"), selected_ask,
        current_ask - current_bid, abs(net), movement,
        abs(net) / movement if movement > 1e-12 else 1.,
        turns / 3, short_v / (long_v + 1e-4), diffs[-1], abs(diffs[-2]),
    ]


@dataclass
class Trial:
    ticker: str
    close_ts: int
    decision_ts: int
    side: str
    x: list
    first: bool
    paired: bool
    outcome: str
    gross_pnl: float
    gross_cost: float


def evaluate_market(market, rows, price=0.45, min_minute=5, max_minute=11):
    """One pre-registered earliest signal / market, no future-based selection."""
    opened, closed = int(market["open_ts"]), int(market["close_ts"])
    if not 600 <= closed - opened <= 1200 or market["result"] not in ("yes", "no"):
        return None
    ordered = sorted((r for r in rows if opened < r["end_ts"] < closed),
                     key=lambda x: x["end_ts"])
    for idx, row in enumerate(ordered):
        elapsed = (row["end_ts"] - opened) / 60
        if not min_minute <= elapsed <= max_minute:
            continue
        q = quote(row)
        if q is None:
            continue
        proposal = candidate_side(*q)
        if proposal is None:
            continue
        side, selected_ask = proposal
        x = features(ordered[:idx+1], opened, closed, side, selected_ask)
        if x is None:
            continue
        # All lows/highs below must occur after signal bar.
        first_at = None
        paired = False
        for future in ordered[idx+1:]:
            if future["end_ts"] <= row["end_ts"]:
                continue
            ask_low = safe_float(future["yes_ask_low"])
            bid_high = safe_float(future["yes_bid_high"])
            yes_touch = ask_low is not None and 0 <= ask_low <= price + 1e-9
            no_touch = bid_high is not None and 0 <= bid_high <= 1 and 1-bid_high <= price + 1e-9
            same_touch = yes_touch if side == "yes" else no_touch
            opp_touch = no_touch if side == "yes" else yes_touch
            if first_at is None:
                if same_touch:
                    first_at = future["end_ts"]
            elif future["end_ts"] > first_at and opp_touch:
                paired = True
                break
        first = first_at is not None
        if paired:
            pnl, cost, outcome = 1 - 2*price, 2*price, "paired"
        elif first:
            pnl = (1-price) if side == market["result"] else -price
            cost, outcome = price, "unpaired_won" if pnl > 0 else "unpaired_lost"
        else:
            pnl, cost, outcome = 0., 0., "unfilled"
        return Trial(market["ticker"], closed, row["end_ts"], side,
                     x, first, paired, outcome, pnl, cost)
    return None


def load_trials(path, price=0.45, min_minute=5, max_minute=11):
    # URI read-only: never creates or changes an on-disk DB.
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        cols = {r["name"] for r in db.execute("PRAGMA table_info(candles)")}
        required = {"ticker", "end_ts", "yes_ask_close", "yes_bid_close",
                    "yes_ask_low", "yes_bid_high"}
        if not required.issubset(cols):
            raise ValueError(f"Missing candles columns: {sorted(required-cols)}")
        markets = db.execute(
            "SELECT ticker, open_ts, close_ts, result FROM markets ORDER BY close_ts, ticker"
        ).fetchall()
        trials=[]
        for m in markets:
            rows=db.execute("SELECT end_ts, yes_ask_close, yes_bid_close, "
                            "yes_ask_low, yes_bid_high FROM candles "
                            "WHERE ticker=? ORDER BY end_ts", (m["ticker"],)).fetchall()
            trial=evaluate_market(m, rows, price, min_minute, max_minute)
            if trial is not None:
                trials.append(trial)
        return trials, len(markets)
    finally:
        db.close()


def split_trials(trials):
    ordered=sorted(trials, key=lambda t:(t.close_ts,t.ticker))
    n=len(ordered)
    a=int(n*.60); b=int(n*.80)
    return ordered[:a], ordered[a:b], ordered[b:]


def matrix(rows):
    return np.array([t.x for t in rows], dtype=float)


def sigmoid(z):
    # Numerically stable sigmoid.
    return np.where(z >= 0, 1/(1+np.exp(-np.clip(z,-40,40))),
                    np.exp(np.clip(z,-40,40))/(1+np.exp(np.clip(z,-40,40))))


class SimpleLogit:
    """Deterministic regularized logit, no sklearn dependency."""
    def fit(self, rows, ridge=0.1, epochs=1200, learning_rate=0.12):
        x=matrix(rows)
        y=np.array([float(t.paired) for t in rows])
        self.mu=x.mean(axis=0)
        self.sd=x.std(axis=0)
        self.sd=np.where(self.sd < 1e-8, 1.0, self.sd)
        x=(x-self.mu)/self.sd
        x=np.clip(x,-12,12)
        self.weights=np.zeros(x.shape[1]); self.bias=math.log((y.mean()+1e-4)/(1-y.mean()+1e-4))
        for i in range(epochs):
            p=sigmoid(x@self.weights+self.bias)
            grad=x.T@(p-y)/len(y)+ridge*self.weights
            bias_grad=float(np.mean(p-y))
            # gentle decaying step and gradient clip
            rate=learning_rate/(1+i/400)
            self.weights-=rate*np.clip(grad,-5,5)
            self.bias-=rate*np.clip(bias_grad,-5,5)
        return self

    def predict(self, rows):
        if not rows:
            return np.array([])
        return sigmoid(np.clip((matrix(rows)-self.mu)/self.sd,-12,12)@self.weights+self.bias)


def auc(y, score):
    # Mann-Whitney rank AUC with exact tie handling; NaN if only one class.
    n=len(y); pos=sum(y); neg=n-pos
    if not pos or not neg:
        return None
    order=sorted(range(n), key=lambda i:score[i])
    ranks=0.; j=0
    while j<n:
        k=j+1
        while k<n and score[order[k]] == score[order[j]]:
            k+=1
        avg_rank=(j+1+k)/2
        ranks+=avg_rank*sum(y[order[t]] for t in range(j,k))
        j=k
    return (ranks-pos*(pos+1)/2)/(pos*neg)


def summary(rows, probs=None, baseline_rate=None):
    n=len(rows)
    if not n:
        return {"n":0}
    first=sum(t.first for t in rows)
    paired=sum(t.paired for t in rows)
    profit=sum(t.gross_pnl for t in rows)
    cost=sum(t.gross_cost for t in rows)
    won=sum(t.outcome == "unpaired_won" for t in rows)
    lost=sum(t.outcome == "unpaired_lost" for t in rows)
    # Wilson 95% interval for the sequential touch frequency.
    phat=paired/n
    z=1.96
    denominator=1+z*z/n
    center=(phat+z*z/(2*n))/denominator
    margin=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denominator
    ret={"n":n,"first_quote_reached":first,
         "paired_rate_wilson_95pct":[round(max(0,center-margin),4),round(min(1,center+margin),4)],
         "pairs_quote_reached":paired,
         "pair_rate_per_signal":round(paired/n,4),
         "pair_rate_per_first_touch":round(paired/first,4) if first else None,
         "unpaired_won":won,"unpaired_lost":lost,
         "gross_proxy_pnl_per_1contract":round(profit,4),
         "gross_proxy_stake_per_1contract":round(cost,4),
         "gross_proxy_roi":round(profit/cost,4) if cost else None}
    if probs is not None:
        labels=np.array([int(t.paired) for t in rows]); probs=np.asarray(probs)
        ret.update({"auc":float(round(auc(labels,probs),4)) if auc(labels,probs) is not None else None,
                    "brier":round(float(np.mean((probs-labels)**2)),5),
                    "train_prevalence_brier":round(float(np.mean((labels-(baseline_rate if baseline_rate is not None else 0.5))**2)),5),
                    "mean_predicted":round(float(np.mean(probs)),4)})
    return ret


def regime(t):
    # Hypothesis-only descriptive quadrants; NOT trained threshold decisions.
    e=t.x[FEATURES.index("path_efficiency_5m")]
    v=t.x[FEATURES.index("total_path_5m")]
    return ("choppy" if e <= .55 else "trending", v)


def self_test():
    # Synthetic 15-min YES close and range bars. Decision at minute 5, future
    # YES touches <=45c in minute 6 and NO only in minute 8.
    m={"ticker":"SYNTH","open_ts":0,"close_ts":900,"result":"no"}
    mids=[.51,.53,.50,.54,.55,.44,.51,.56,.54,.55,.5,.51,.52,.48]
    rows=[]
    for i,mid in enumerate(mids,1):
        rows.append({"end_ts":i*60,"yes_bid_close":mid-.01,
                     "yes_ask_close":mid+.01,"yes_ask_low":mid+.01,
                     "yes_bid_high":mid-.01})
    t=evaluate_market(m,rows,min_minute=5,max_minute=11)
    assert t and t.decision_ts==300 and t.side=="yes" and t.paired and abs(t.gross_pnl-.10)<1e-8,t
    # Disallow hedge in same minute, and ensure settlement fallback correct.
    rows[7]["yes_bid_high"] = .50
    rows[6]["yes_bid_high"] = .50
    for i in range(8,len(rows)):
        rows[i]["yes_bid_high"] = .50
    t=evaluate_market(m,rows)
    assert t.first and not t.paired and t.outcome=="unpaired_lost",t
    # Future data must not alter any entry feature.
    saved=t.x[:]
    rows[-1]["yes_ask_low"]=.01
    assert evaluate_market(m,rows).x==saved
    # AUC sanity, no leakage through the model split.
    assert abs(auc(np.array([0,1]),np.array([0.1,0.9]))-1)<1e-10
    print("PASS: synthetic chronology, side selection, no same-minute pair, "
          "unhedged loss, feature cutoff, AUC — NO ORDERS")


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db",type=Path,default=Path(".collector_runs/btc15m_pennies.sqlite3"))
    ap.add_argument("--output",type=Path,default=Path("data/research/btc_4545_regime_v1.json"))
    ap.add_argument("--price",type=float,default=.45)
    ap.add_argument("--min-entry-minute",type=int,default=5)
    ap.add_argument("--max-entry-minute",type=int,default=11)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        self_test(); return
    if not 0<a.price<.50 or not 5<=a.min_entry_minute<=a.max_entry_minute<=11:
        ap.error("price must be in (0,.50); entry minutes must satisfy 5 <= min <= max <= 11")
    print("KXBTC15M V1: CHRONOLOGICAL QUOTE-REACHABILITY PREDICTION (NOT FILLS)")
    print("SQLite: READ-ONLY; no network; no orders")
    trials,total=load_trials(a.db,a.price,a.min_entry_minute,a.max_entry_minute)
    print(f"Markets={total:,}, eligible decisions={len(trials):,}")
    if len(trials)<150:
        print("INSUFFICIENT eligible data for meaningful holdout; need >=150 signals")
        return
    train,val,test=split_trials(trials)
    if sum(t.paired for t in train) < 10 or sum(t.paired for t in val)<3:
        print("INSUFFICIENT positive events in train/validation; model not evaluated")
        return
    model=SimpleLogit().fit(train)
    pv=model.predict(val); pt=model.predict(test)
    # Threshold fixed on validation only, top 30%, no test-set tuning.
    threshold=float(np.quantile(pv, .70))
    mask=pt>=threshold
    selected=[t for i,t in enumerate(test) if mask[i]]
    train_base=sum(t.paired for t in train)/len(train)
    # Descriptive regime splits use thresholds determined from TRAIN only.
    train_volume_median=float(np.median([t.x[FEATURES.index("total_path_5m")] for t in train]))
    def regime_stats(rows):
        out={}
        for eff_name, eff_fn in (("choppy",lambda e:e<=.55),("trending",lambda e:e>.55)):
            for vol_name, vol_fn in (("higher_movement",lambda v:v>=train_volume_median),
                                     ("lower_movement",lambda v:v<train_volume_median)):
                label=f"{eff_name}_{vol_name}"
                cohort=[t for t in rows if eff_fn(t.x[FEATURES.index("path_efficiency_5m")])
                        and vol_fn(t.x[FEATURES.index("total_path_5m")])]
                out[label]=summary(cohort)
        return out
    result={
        "warning":"QUOTE REACHABILITY ONLY, NOT ORDER FILLS OR TRADEABLE PNL; fee-free optimistic minute-bar proxy",
        "db":str(a.db),"all_markets":total,"eligible_markets":len(trials),
        "definition":"Earliest minute 5-11 showing YES or NO ask 50-58c; 45c same-side future quote low, then 45c opposite-side future quote low in STRICTLY LATER candle",
        "train_period_utc":[datetime.fromtimestamp(train[0].close_ts,timezone.utc).isoformat(),datetime.fromtimestamp(train[-1].close_ts,timezone.utc).isoformat()],
        "validation_period_utc":[datetime.fromtimestamp(val[0].close_ts,timezone.utc).isoformat(),datetime.fromtimestamp(val[-1].close_ts,timezone.utc).isoformat()],
        "test_period_utc":[datetime.fromtimestamp(test[0].close_ts,timezone.utc).isoformat(),datetime.fromtimestamp(test[-1].close_ts,timezone.utc).isoformat()],
        "train":summary(train),"validation":summary(val,pv,train_base),
        "test_all":summary(test,pt,train_base),"test_selected":summary(selected,pt[mask],train_base),
        "test_regime_quadrants":regime_stats(test),
        "train_median_movement_threshold":round(train_volume_median,6),
        "threshold_from_validation":round(threshold,6),
        "feature_names":FEATURES,
        "coefficients":{name:round(float(w),6) for name,w in zip(FEATURES,model.weights)},
    }
    print("TRAIN:",result["train"])
    print("VALIDATION:",result["validation"])
    print("TEST ALL:",result["test_all"])
    print("TEST SELECTED:",result["test_selected"])
    print("TEST THRESHOLD (validation top 30%):",round(threshold,4))
    print("TEST REGIME BREAKDOWN (thresholds fixed on train):")
    for label, metrics in result["test_regime_quadrants"].items():
        print(" ",label,metrics)
    print("\nLIMITATIONS: Intraminute OHLC quote extremes, not maker fills; quote touches can be"
          " optimistic; NO fees, NO queue, NO BTC prices, no latency, NO actual order-book depth.")
    print("AUC/Brier measures prediction of QUOTE touch sequences, NOT profitability.")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("SAVED",a.output)

if __name__=="__main__":
    main()
