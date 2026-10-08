"""
KXBTC15M cross-exchange disagreement walk-forward.

Compare two models:

CONTROL:
    Kalshi probability + Coinbase distance to strike

PLUS_DISAGREEMENT:
    Kalshi probability + Coinbase distance to strike
    + Bitstamp/Coinbase price disagreement

Both models use identical:
    - historical observations
    - training windows
    - entry quotes
    - transaction fees
    - minimum net-edge threshold

Development only: January 1 through August 7, 2026.
"""

import numpy as np
import pandas as pd

from src.database.db import get_connection

from src.crypto.backtest.cross_exchange_incremental_walkforward import (
    build_trade,
)


# ============================================================
# CONFIGURATION
# ============================================================

START = pd.Timestamp(
    "2026-01-01",
    tz="UTC",
)

END = pd.Timestamp(
    "2026-08-08",
    tz="UTC",
)

LOOKBACK_DAYS = 30

OUTCOME_EMBARGO_DAYS = 1

MIN_TRAIN_ROWS = 1000

# Fixed L2 regularization
L2 = 0.001

# Feature units are per 10 basis points.
FEATURE_SCALE_BPS = 10.0

EPS = 1e-9


# ============================================================
# MATHEMATICAL HELPERS
# ============================================================

def sigmoid(z):

    z = np.clip(
        np.asarray(z, dtype=float),
        -40.0,
        40.0,
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(-z)
        )
    )


def brier(y, p):

    y = np.asarray(
        y,
        dtype=float,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


def logloss(y, p):

    p = np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    return float(
        np.mean(
            -y * np.log(p)
            -
            (1.0 - y) * np.log1p(-p)
        )
    )


def score(y, p):

    return (
        brier(y, p),
        logloss(y, p),
    )


# ============================================================
# LOAD SYNCHRONIZED KALSHI AND EXCHANGE DATA
# ============================================================

def load_data(start=START, end=END):

    """
    Use the T-5 Kalshi quote.

    Coinbase and Bitstamp candles must have ended
    one full minute before the Kalshi quote timestamp.

    This avoids using a same-minute exchange close
    that may not have been available at decision time.
    """

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT

                m.ticker,

                mm.quote_time::timestamptz
                    AS quote_time,

                m.outcome,
                m.floor_strike,

                mm.midpoint,
                mm.yes_ask,
                mm.no_ask,

                cb.close
                    AS coinbase_close,

                bs.close
                    AS bitstamp_close

            FROM crypto_market_minutes mm

            JOIN crypto_markets m

                ON m.ticker = mm.ticker

            JOIN btc_spot_minutes cb

                ON cb.candle_end =
                    mm.quote_time::timestamptz
                    - interval '1 minute'

                AND cb.product_id = 'BTC-USD'

            JOIN cross_exchange_spot_minutes bs

                ON bs.candle_end =
                    mm.quote_time::timestamptz
                    - interval '1 minute'

                AND bs.exchange = 'BITSTAMP'

            WHERE

                m.series_ticker = 'KXBTC15M'

                AND mm.actionable = 1

                AND ABS(
                    mm.minutes_remaining - 5.0
                ) < 0.000001

                AND m.outcome IS NOT NULL

                AND mm.quote_time::timestamptz
                    >= ?::timestamptz

                AND mm.quote_time::timestamptz
                    < ?::timestamptz

            ORDER BY
                mm.quote_time::timestamptz
            """,
            (
                start.isoformat(),
                end.isoformat(),
            ),
        )

        data = pd.DataFrame(
            cursor.fetchall(),
            columns=[
                column[0]
                for column in cursor.description
            ],
        )

    data["quote_time"] = pd.to_datetime(
        data["quote_time"],
        utc=True,
    )

    numeric_columns = [
        "outcome",
        "floor_strike",
        "midpoint",
        "yes_ask",
        "no_ask",
        "coinbase_close",
        "bitstamp_close",
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "quote_time",
            "outcome",
            "floor_strike",
            "midpoint",
            "coinbase_close",
            "bitstamp_close",
        ]
    ).copy()

    # --------------------------------------------------------
    # Exclude corrupt strike metadata
    # --------------------------------------------------------

    reference = data[
        [
            "coinbase_close",
            "bitstamp_close",
        ]
    ].median(axis=1)

    data = data.loc[
        data["outcome"].isin([0, 1])
        &
        data["floor_strike"].gt(0)
        &
        reference.gt(0)
        &
        (
            (
                reference
                -
                data["floor_strike"]
            ).abs()
            /
            reference
        ).le(0.05)
        &
        data["midpoint"].gt(0)
        &
        data["midpoint"].lt(1)
    ].copy()

    # --------------------------------------------------------
    # Kalshi probability baseline
    # --------------------------------------------------------

    data["market_logit"] = np.log(
        data["midpoint"]
        /
        (1.0 - data["midpoint"])
    )

    # --------------------------------------------------------
    # Coinbase distance to strike
    # --------------------------------------------------------

    data["coinbase_x"] = (
        (
            data["coinbase_close"]
            -
            data["floor_strike"]
        )
        /
        data["floor_strike"]
        *
        10000.0
        /
        FEATURE_SCALE_BPS
    )

    # --------------------------------------------------------
    # Bitstamp - Coinbase disagreement
    #
    # Positive:
    #   Bitstamp is above Coinbase.
    #
    # Negative:
    #   Bitstamp is below Coinbase.
    #
    # This is the incremental feature we are testing.
    # --------------------------------------------------------

    data["disagreement_x"] = (
        (
            data["bitstamp_close"]
            -
            data["coinbase_close"]
        )
        /
        data["floor_strike"]
        *
        10000.0
        /
        FEATURE_SCALE_BPS
    )

    data["date"] = (
        data["quote_time"]
        .dt.floor("D")
    )

    assert not data.duplicated("ticker").any(), (
        "Duplicate T-5 contract rows detected."
    )

    return data.reset_index(drop=True)


# ============================================================
# FIT OFFSET LOGISTIC REGRESSION
# ============================================================

def fit_offset_logistic(
    training,
    columns,
):

    """
    Fits:

        logit(p) = Kalshi_logit + X * beta

    The Kalshi coefficient is fixed at 1.

    L2 regularization is applied to beta.
    """

    x = training[
        columns
    ].to_numpy(dtype=float)

    y = training[
        "outcome"
    ].to_numpy(dtype=float)

    offset = training[
        "market_logit"
    ].to_numpy(dtype=float)

    n, k = x.shape

    beta = np.zeros(k)

    def objective(weights):

        z = (
            offset
            +
            x @ weights
        )

        return float(
            np.mean(
                np.logaddexp(0.0, z)
                -
                y * z
            )
            +
            0.5
            *
            L2
            *
            (weights @ weights)
        )

    # --------------------------------------------------------
    # Newton optimization
    # --------------------------------------------------------

    for _ in range(80):

        p = sigmoid(
            offset
            +
            x @ beta
        )

        gradient = (
            x.T @ (p - y) / n
            +
            L2 * beta
        )

        variance = (
            p
            *
            (1.0 - p)
        )

        hessian = (
            (x.T * variance) @ x / n
            +
            L2 * np.eye(k)
        )

        step = np.linalg.solve(
            hessian,
            gradient,
        )

        prior_objective = objective(
            beta
        )

        rate = 1.0

        updated = False

        # Backtracking line search
        for _ in range(20):

            candidate = (
                beta
                -
                rate * step
            )

            candidate_objective = objective(
                candidate
            )

            if (
                np.isfinite(candidate_objective)
                and
                candidate_objective <= prior_objective
            ):

                updated = True
                break

            rate *= 0.5

        if not updated:
            break

        beta = candidate

        if np.max(
            np.abs(rate * step)
        ) < 1e-8:
            break

    return beta


# ============================================================
# ROLLING WALK-FORWARD
# ============================================================

def walk_forward(data):

    prediction_frames = []

    trade_rows = []

    coefficient_rows = []

    for date, test in data.groupby(
        "date",
        sort=True,
    ):

        # ----------------------------------------------------
        # Use 30 prior calendar days.
        #
        # Exclude immediately preceding day to maintain
        # a conservative settlement embargo.
        # ----------------------------------------------------

        train_end = (
            date
            -
            pd.Timedelta(
                days=OUTCOME_EMBARGO_DAYS
            )
        )

        train_start = (
            train_end
            -
            pd.Timedelta(
                days=LOOKBACK_DAYS
            )
        )

        train = data.loc[
            (data["date"] >= train_start)
            &
            (data["date"] < train_end)
        ]

        if len(train) < MIN_TRAIN_ROWS:
            continue

        # ----------------------------------------------------
        # CONTROL:
        #
        # Kalshi + Coinbase
        # ----------------------------------------------------

        base_coef = fit_offset_logistic(
            train,
            ["coinbase_x"],
        )

        # ----------------------------------------------------
        # AUGMENTED:
        #
        # Kalshi + Coinbase + Bitstamp disagreement
        # ----------------------------------------------------

        plus_coef = fit_offset_logistic(
            train,
            [
                "coinbase_x",
                "disagreement_x",
            ],
        )

        test = test.copy()

        # ----------------------------------------------------
        # Control predictions
        # ----------------------------------------------------

        test["control_probability"] = sigmoid(
            test["market_logit"].to_numpy()
            +
            test[
                ["coinbase_x"]
            ].to_numpy()
            @
            base_coef
        )

        # ----------------------------------------------------
        # Augmented predictions
        # ----------------------------------------------------

        test["plus_probability"] = sigmoid(
            test["market_logit"].to_numpy()
            +
            test[
                [
                    "coinbase_x",
                    "disagreement_x",
                ]
            ].to_numpy()
            @
            plus_coef
        )

        prediction_frames.append(
            test
        )

        # ----------------------------------------------------
        # Track coefficient stability
        # ----------------------------------------------------

        coefficient_rows.append(
            {
                "date":
                    date,

                "training_rows":
                    len(train),

                "control_beta_coinbase_per_10bps":
                    base_coef[0],

                "plus_beta_coinbase_per_10bps":
                    plus_coef[0],

                "plus_beta_disagreement_per_10bps":
                    plus_coef[1],
            }
        )

        # ----------------------------------------------------
        # Evaluate trades using existing executable logic
        # ----------------------------------------------------

        for _, row in test.iterrows():

            for model, field in [
                (
                    "CONTROL",
                    "control_probability",
                ),
                (
                    "PLUS_DISAGREEMENT",
                    "plus_probability",
                ),
            ]:

                trade = build_trade(
                    row,
                    fair_yes=float(
                        row[field]
                    ),
                )

                if trade is not None:

                    trade_rows.append(
                        {
                            "ticker":
                                row["ticker"],

                            "date":
                                date,

                            "model":
                                model,

                            **trade,
                        }
                    )

    if not prediction_frames:

        raise RuntimeError(
            "No dates passed the minimum "
            "training-row requirement."
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    trades = pd.DataFrame(
        trade_rows
    )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    return (
        predictions,
        trades,
        coefficients,
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    predictions,
    trades,
    coefficients,
):

    y = predictions[
        "outcome"
    ].to_numpy()

    raw = score(
        y,
        predictions["midpoint"],
    )

    control = score(
        y,
        predictions["control_probability"],
    )

    plus = score(
        y,
        predictions["plus_probability"],
    )

    # --------------------------------------------------------
    # Probability accuracy
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "IDENTICAL-ROW PROBABILITY SCORES"
    )
    print("=" * 110)

    print(
        f"{'model':<23} "
        f"{'rows':>7} "
        f"{'brier':>11} "
        f"{'logloss':>11}"
    )

    for name, values in [
        ("RAW_KALSHI", raw),
        ("CONTROL_COINBASE", control),
        ("PLUS_DISAGREEMENT", plus),
    ]:

        print(
            f"{name:<23} "
            f"{len(predictions):7d} "
            f"{values[0]:11.6f} "
            f"{values[1]:11.6f}"
        )

    print()

    print(
        "Incremental Brier improvement vs control: "
        f"{(control[0] - plus[0]) / control[0]:+.3%}"
    )

    print(
        "Incremental log-loss improvement vs control: "
        f"{(control[1] - plus[1]) / control[1]:+.3%}"
    )

    # --------------------------------------------------------
    # Beta stability
    # --------------------------------------------------------

    beta = coefficients[
        "plus_beta_disagreement_per_10bps"
    ]

    print()
    print("=" * 110)
    print(
        "DISAGREEMENT COEFFICIENT"
    )
    print("=" * 110)

    print(
        f"Fitted days: {len(coefficients)}"
    )

    print(
        "Bitstamp disagreement beta "
        "(per +10 bps): "
        f"mean={beta.mean():+.5f}, "
        f"median={beta.median():+.5f}, "
        f"positive_days={(beta > 0).mean():.1%}"
    )

    # --------------------------------------------------------
    # Executable performance
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "EXECUTABLE STRATEGIES"
    )
    print("=" * 110)

    print(
        "Same 2.5pp minimum edge and one entry taker fee."
    )

    for model in [
        "CONTROL",
        "PLUS_DISAGREEMENT",
    ]:

        if trades.empty:

            group = pd.DataFrame()

        else:

            group = trades[
                trades["model"] == model
            ]

        if group.empty:

            print(
                f"{model}: 0 trades"
            )

            continue

        capital = float(
            group["cost"].sum()
        )

        pnl = float(
            group["pnl"].sum()
        )

        wins = int(
            group["won"].sum()
        )

        win_rate = float(
            group["won"].mean()
        )

        avg_edge = float(
            group["edge"].mean()
        )

        roi = (
            pnl / capital
            if capital > 0
            else np.nan
        )

        print(
            f"{model}: "
            f"trades={len(group)}, "
            f"wins={wins}, "
            f"win_rate={win_rate:.2%}, "
            f"avg_edge={avg_edge:.2%}, "
            f"capital=${capital:.2f}, "
            f"pnl=${pnl:+.2f}, "
            f"roi={roi:+.2%}"
        )

    # --------------------------------------------------------
    # Compare trade selections
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "TRADES SELECTED BY EACH MODEL"
    )
    print("=" * 110)

    if not trades.empty:

        baseline = set(
            trades.loc[
                trades["model"] == "CONTROL",
                "ticker",
            ]
        )

        augmented = set(
            trades.loc[
                trades["model"] == "PLUS_DISAGREEMENT",
                "ticker",
            ]
        )

        print(
            f"Both models: {len(baseline & augmented)}"
        )

        print(
            f"Control only: {len(baseline - augmented)}"
        )

        print(
            f"Augmented only: {len(augmented - baseline)}"
        )

    # --------------------------------------------------------
    # Monthly probability score comparison
    # --------------------------------------------------------

    predictions["month"] = (
        predictions["date"]
        .dt.strftime("%Y-%m")
    )

    print()
    print("=" * 110)
    print(
        "MONTHLY PAIRED PROBABILITY SCORES"
    )
    print("=" * 110)

    print(
        "Positive improvement means disagreement helped."
    )

    for month, group in predictions.groupby(
        "month",
        sort=True,
    ):

        outcomes = group[
            "outcome"
        ].to_numpy()

        base_scores = score(
            outcomes,
            group["control_probability"],
        )

        plus_scores = score(
            outcomes,
            group["plus_probability"],
        )

        brier_improvement = (
            (base_scores[0] - plus_scores[0])
            /
            base_scores[0]
        )

        logloss_improvement = (
            (base_scores[1] - plus_scores[1])
            /
            base_scores[1]
        )

        print(
            f"{month}: "
            f"rows={len(group):4d} "
            f"Brier improvement={brier_improvement:+.3%} "
            f"log-loss improvement={logloss_improvement:+.3%}"
        )

    # --------------------------------------------------------
    # Monthly realized P&L
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "MONTHLY TRADE NET P&L"
    )
    print("=" * 110)

    if not trades.empty:

        trades["month"] = (
            trades["date"]
            .dt.strftime("%Y-%m")
        )

        monthly = trades.pivot_table(
            index="month",
            columns="model",
            values="pnl",
            aggfunc="sum",
            fill_value=0,
        )

        print(
            monthly.round(2).to_string()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 130)
    print(
        "KXBTC15M — CROSS-EXCHANGE DISAGREEMENT WALK-FORWARD"
    )
    print("=" * 130)

    print(
        "Development: 2026-01-01 -> 2026-08-07"
    )

    print(
        "Horizon: T-5"
    )

    print(
        "Rolling 30-day training window"
    )

    print(
        "One-calendar-day outcome embargo"
    )

    print(
        "Minimum executable net edge: 2.50%"
    )

    print(
        "Exchange inputs: minute candles ending "
        "one minute before Kalshi quote"
    )

    print(
        "Fixed market-logit offset; "
        "features per 10 bps; L2=0.001"
    )

    print()

    data = load_data()

    print(
        f"Usable synchronized contracts: {len(data)}"
    )

    print(
        f"Range: {data['quote_time'].min()} "
        f"-> {data['quote_time'].max()}"
    )

    predictions, trades, coefficients = walk_forward(
        data
    )

    print_results(
        predictions,
        trades,
        coefficients,
    )


if __name__ == "__main__":
    main()