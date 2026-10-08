"""Forward performance report for KXBTC15M disagreement shadow v1."""

import numpy as np
import pandas as pd

from src.database.db import get_connection


TABLE = "crypto_cross_exchange_shadow_v1"

EPS = 1e-9

MODELS = {
    "KALSHI": {
        "probability": "market_probability",
    },
    "CONTROL": {
        "probability": "control_probability",
        "side": "control_side",
        "cost": "control_cost",
        "edge": "control_edge",
        "pnl": "control_pnl",
    },
    "DISAGREEMENT": {
        "probability": "plus_probability",
        "side": "plus_side",
        "cost": "plus_cost",
        "edge": "plus_edge",
        "pnl": "plus_pnl",
    },
}


def load_data():

    with get_connection() as connection:

        cursor = connection.execute(
            f"""
            SELECT
                ticker,
                created_at,
                close_time,
                target_decision_time,
                quote_received_at,
                spot_candle_end,

                market_probability,
                control_probability,
                plus_probability,

                control_side,
                control_cost,
                control_edge,
                control_pnl,

                plus_side,
                plus_cost,
                plus_edge,
                plus_pnl,

                settlement_result

            FROM {TABLE}

            ORDER BY close_time, ticker
            """
        )

        rows = cursor.fetchall()

        columns = [
            column[0]
            for column in cursor.description
        ]

    data = pd.DataFrame(
        rows,
        columns=columns,
    )

    if data.empty:
        return data

    for column in [
        "created_at",
        "close_time",
        "target_decision_time",
        "quote_received_at",
        "spot_candle_end",
    ]:

        data[column] = pd.to_datetime(
            data[column],
            utc=True,
            errors="coerce",
        )

    for column in [
        "market_probability",
        "control_probability",
        "plus_probability",
        "control_cost",
        "control_edge",
        "control_pnl",
        "plus_cost",
        "plus_edge",
        "plus_pnl",
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data


def probability_scores(data, column):

    valid = data.dropna(
        subset=[column]
    )

    if valid.empty:
        return 0, np.nan, np.nan

    y = (
        valid["settlement_result"]
        .eq("yes")
        .astype(float)
        .to_numpy()
    )

    p = (
        valid[column]
        .clip(EPS, 1 - EPS)
        .to_numpy(dtype=float)
    )

    brier = float(
        np.mean((p - y) ** 2)
    )

    logloss = float(
        -np.mean(
            y * np.log(p)
            +
            (1 - y) * np.log1p(-p)
        )
    )

    return len(valid), brier, logloss


def max_drawdown(pnl_values):

    pnl = np.asarray(
        pnl_values,
        dtype=float,
    )

    if len(pnl) == 0:
        return 0.0

    equity = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(pnl),
        ]
    )

    running_peak = np.maximum.accumulate(
        equity
    )

    return float(
        np.max(running_peak - equity)
    )


def trading_summary(settled, name, config):

    side_col = config["side"]
    cost_col = config["cost"]
    edge_col = config["edge"]
    pnl_col = config["pnl"]

    selected = settled.loc[
        settled[side_col].notna()
    ].copy()

    complete = selected.dropna(
        subset=[cost_col, pnl_col]
    ).copy()

    if complete.empty:

        print(
            f"{name}: "
            f"selected={len(selected)}, "
            "settled_trades=0"
        )

        return

    complete = complete.sort_values(
        "close_time"
    )

    wins = (
        complete[side_col]
        .str.lower()
        .eq(
            complete["settlement_result"]
        )
    )

    capital = float(
        complete[cost_col].sum()
    )

    pnl = float(
        complete[pnl_col].sum()
    )

    roi = (
        pnl / capital
        if capital > 0
        else np.nan
    )

    avg_edge = float(
        complete[edge_col].mean()
    )

    drawdown = max_drawdown(
        complete[pnl_col].to_numpy()
    )

    print(
        f"{name}: "
        f"selected={len(selected)}, "
        f"settled_trades={len(complete)}, "
        f"wins={int(wins.sum())}, "
        f"win_rate={wins.mean():.2%}, "
        f"avg_modeled_edge={avg_edge:.2%}"
    )

    print(
        f"  capital=${capital:.2f}, "
        f"pnl=${pnl:+.2f}, "
        f"roi={roi:+.2%}, "
        f"max_drawdown=${drawdown:.2f}"
    )

    if len(selected) != len(complete):

        print(
            f"  WARNING: "
            f"{len(selected) - len(complete)} "
            "selected trades lack complete P&L"
        )


def main():

    print()
    print("=" * 110)
    print(
        "KXBTC15M — CROSS-EXCHANGE FORWARD SHADOW REPORT"
    )
    print("=" * 110)

    data = load_data()

    if data.empty:

        print(
            "No shadow decisions recorded."
        )

        return

    print()
    print(
        f"Total decisions: {len(data)}"
    )

    print(
        f"First: {data['created_at'].min()}"
    )

    print(
        f"Latest: {data['created_at'].max()}"
    )

    settled = data.loc[
        data["settlement_result"].isin(
            ["yes", "no"]
        )
    ].copy()

    pending = (
        len(data) - len(settled)
    )

    print(
        f"Settled: {len(settled)}"
    )

    print(
        f"Pending: {pending}"
    )

    # ========================================================
    # TIMING AUDIT
    # ========================================================

    print()
    print("=" * 110)
    print("TIMING AUDIT")
    print("=" * 110)

    quote_delay = (
        data["quote_received_at"]
        -
        data["target_decision_time"]
    ).dt.total_seconds()

    spot_lag = (
        data["target_decision_time"]
        -
        data["spot_candle_end"]
    ).dt.total_seconds()

    print(
        f"Quote delay: "
        f"median={quote_delay.median():.2f}s, "
        f"maximum={quote_delay.max():.2f}s"
    )

    bad_quotes = (
        (quote_delay < 0)
        |
        (quote_delay > 35)
        |
        quote_delay.isna()
    )

    bad_spot = (
        (spot_lag != 60)
        |
        spot_lag.isna()
    )

    print(
        f"Quotes outside permitted window: "
        f"{int(bad_quotes.sum())}"
    )

    print(
        f"Unexpected spot-candle alignment: "
        f"{int(bad_spot.sum())}"
    )

    # Potential collection gaps.
    # These are not proof of missing markets.

    ordered = data.sort_values(
        "target_decision_time"
    )

    gaps = (
        ordered["target_decision_time"]
        .diff()
    )

    long_gaps = gaps[
        gaps > pd.Timedelta(minutes=20)
    ]

    print(
        f"Decision gaps greater than 20 minutes: "
        f"{len(long_gaps)}"
    )

    # ========================================================
    # PROBABILITY METRICS
    # ========================================================

    print()
    print("=" * 110)
    print("FORWARD PROBABILITY ACCURACY")
    print("=" * 110)

    print(
        f"{'MODEL':<18} "
        f"{'ROWS':>7} "
        f"{'BRIER':>12} "
        f"{'LOG LOSS':>12}"
    )

    scores = {}

    for name, config in MODELS.items():

        n, brier, logloss = probability_scores(
            settled,
            config["probability"],
        )

        scores[name] = (
            brier,
            logloss,
        )

        print(
            f"{name:<18} "
            f"{n:>7} "
            f"{brier:>12.6f} "
            f"{logloss:>12.6f}"
        )

    control_brier, control_log = (
        scores["CONTROL"]
    )

    plus_brier, plus_log = (
        scores["DISAGREEMENT"]
    )

    if (
        np.isfinite(control_brier)
        and control_brier > 0
        and np.isfinite(plus_brier)
    ):

        print()
        print(
            "Disagreement Brier improvement "
            "vs control: "
            f"{(control_brier - plus_brier) / control_brier:+.3%}"
        )

    if (
        np.isfinite(control_log)
        and control_log > 0
        and np.isfinite(plus_log)
    ):

        print(
            "Disagreement log-loss improvement "
            "vs control: "
            f"{(control_log - plus_log) / control_log:+.3%}"
        )

    # ========================================================
    # EXECUTION
    # ========================================================

    print()
    print("=" * 110)
    print("FORWARD PAPER TRADING PERFORMANCE")
    print("=" * 110)

    if settled.empty:

        print(
            "No settlements available yet."
        )

    else:

        for name in (
            "CONTROL",
            "DISAGREEMENT",
        ):

            trading_summary(
                settled,
                name,
                MODELS[name],
            )

    # ========================================================
    # MONTHLY PERFORMANCE
    # ========================================================

    print()
    print("=" * 110)
    print("MONTHLY PAPER P&L")
    print("=" * 110)

    if settled.empty:

        print(
            "No settled contracts."
        )

        return

    settled["month"] = (
        settled["close_time"]
        .dt.strftime("%Y-%m")
    )

    for month, group in settled.groupby(
        "month",
        sort=True,
    ):

        control_trades = int(
            group["control_side"]
            .notna()
            .sum()
        )

        plus_trades = int(
            group["plus_side"]
            .notna()
            .sum()
        )

        control_pnl = float(
            group["control_pnl"]
            .sum()
        )

        plus_pnl = float(
            group["plus_pnl"]
            .sum()
        )

        print(
            f"{month}: "
            f"settled={len(group)}, "
            f"control_trades={control_trades}, "
            f"control_pnl=${control_pnl:+.2f}, "
            f"disagreement_trades={plus_trades}, "
            f"disagreement_pnl=${plus_pnl:+.2f}"
        )


if __name__ == "__main__":
    main()