import numpy as np
import pandas as pd

from src.database.db import get_connection
from src.kalshi.fees import calculate_taker_fee


# ============================================================
# SETTINGS
# ============================================================

SERIES_TICKER = "KXBTC15M"

HORIZON = 5

DEVELOPMENT_START = pd.Timestamp(
    "2026-01-01T00:00:00Z"
)

DEVELOPMENT_END = pd.Timestamp(
    "2026-08-08T00:00:00Z"
)

LOOKBACK_DAYS = 30

MINIMUM_TRAINING_ROWS = 1000

MIN_NET_EDGE = 0.025

EPSILON = 1e-6

PROXIES = {
    "COINBASE":
        "coinbase_close",

    "BITSTAMP":
        "bitstamp_close",

    "MEAN_3":
        "mean_3",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    with get_connection() as connection:

        cursor = connection.execute(
            """
            SELECT

                m.ticker,

                m.close_time::timestamptz
                    AS close_time,

                m.floor_strike,

                m.outcome,

                mm.quote_time::timestamptz
                    AS quote_time,

                mm.minutes_remaining,

                mm.yes_bid,
                mm.yes_ask,

                mm.no_bid,
                mm.no_ask,

                mm.midpoint,

                cb.close
                    AS coinbase_close,

                bs.close
                    AS bitstamp_close,

                cc.close
                    AS crypto_com_close

            FROM crypto_market_minutes mm

            JOIN crypto_markets m

                ON m.ticker =
                    mm.ticker

            JOIN btc_spot_minutes cb

                ON cb.candle_end =
                    mm.quote_time::timestamptz

                AND cb.product_id =
                    'BTC-USD'

            JOIN cross_exchange_spot_minutes bs

                ON bs.candle_end =
                    mm.quote_time::timestamptz

                AND bs.exchange =
                    'BITSTAMP'

            JOIN cross_exchange_spot_minutes cc

                ON cc.candle_end =
                    mm.quote_time::timestamptz

                AND cc.exchange =
                    'CRYPTO_COM'

            WHERE

                m.series_ticker = ?

                AND mm.actionable = 1

                AND ABS(
                    mm.minutes_remaining - ?
                ) < 0.000001

                AND m.outcome IS NOT NULL

                AND m.floor_strike IS NOT NULL

            ORDER BY
                mm.quote_time::timestamptz
            """,
            (
                SERIES_TICKER,
                HORIZON,
            ),
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

    data[
        "quote_time"
    ] = pd.to_datetime(
        data[
            "quote_time"
        ],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    numeric = [
        "floor_strike",
        "outcome",
        "minutes_remaining",

        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",

        "midpoint",

        "coinbase_close",
        "bitstamp_close",
        "crypto_com_close",
    ]

    for column in numeric:

        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ],
            errors="coerce",
        )

    data = (
        data.dropna(
            subset=[
                "quote_time",
                "floor_strike",
                "outcome",
                "midpoint",
                "coinbase_close",
                "bitstamp_close",
                "crypto_com_close",
            ]
        )
        .copy()
    )

    # --------------------------------------------------------
    # Period
    # --------------------------------------------------------

    data = (
        data[
            (
                data[
                    "quote_time"
                ]
                >=
                DEVELOPMENT_START
            )
            &
            (
                data[
                    "quote_time"
                ]
                <
                DEVELOPMENT_END
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Combined price
    # --------------------------------------------------------

    data[
        "mean_3"
    ] = (
        data[
            [
                "coinbase_close",
                "bitstamp_close",
                "crypto_com_close",
            ]
        ]
        .mean(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Remove corrupt strike metadata.
    #
    # A 5% bound is enormously wider than any legitimate
    # five-minute BTC move and catches the ~$7 strike corruption.
    # --------------------------------------------------------

    reference_price = (
        data[
            [
                "coinbase_close",
                "bitstamp_close",
                "crypto_com_close",
            ]
        ]
        .median(
            axis=1
        )
    )

    relative_strike_error = (
        (
            reference_price
            -
            data[
                "floor_strike"
            ]
        )
        .abs()
        /
        reference_price
    )

    data = (
        data[
            (
                data[
                    "floor_strike"
                ]
                >
                0
            )
            &
            (
                relative_strike_error
                <=
                0.05
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Market probability
    # --------------------------------------------------------

    data[
        "market_probability"
    ] = (
        data[
            "midpoint"
        ]
        .clip(
            0.01,
            0.99,
        )
    )

    data[
        "market_logit"
    ] = np.log(
        data[
            "market_probability"
        ]
        /
        (
            1.0
            -
            data[
                "market_probability"
            ]
        )
    )

    # --------------------------------------------------------
    # Proxy distances
    # --------------------------------------------------------

    for (
        proxy_name,
        proxy_column,
    ) in PROXIES.items():

        data[
            f"{proxy_name}_distance_bps"
        ] = (
            (
                data[
                    proxy_column
                ]
                -
                data[
                    "floor_strike"
                ]
            )
            /
            data[
                "floor_strike"
            ]
            *
            10000.0
        )

    data[
        "date"
    ] = (
        data[
            "quote_time"
        ]
        .dt.floor(
            "D"
        )
    )

    return (
        data.sort_values(
            "quote_time"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MATH
# ============================================================

def sigmoid(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    result = np.empty_like(
        values
    )

    positive = (
        values >= 0
    )

    result[
        positive
    ] = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -values[
                    positive
                ]
            )
        )
    )

    negative = ~positive

    exp_values = np.exp(
        values[
            negative
        ]
    )

    result[
        negative
    ] = (
        exp_values
        /
        (
            1.0
            +
            exp_values
        )
    )

    return result


def fit_beta(
    training,
    distance_column,
):

    x = (
        training[
            distance_column
        ]
        .to_numpy(
            dtype=float
        )
    )

    baseline = (
        training[
            "market_logit"
        ]
        .to_numpy(
            dtype=float
        )
    )

    outcome = (
        training[
            "outcome"
        ]
        .to_numpy(
            dtype=float
        )
    )

    beta = 0.0

    for _ in range(100):

        probability = sigmoid(
            baseline
            +
            beta
            *
            x
        )

        gradient = np.sum(
            x
            *
            (
                probability
                -
                outcome
            )
        )

        hessian = np.sum(
            (
                x
                ** 2
            )
            *
            probability
            *
            (
                1.0
                -
                probability
            )
        )

        # Tiny numerical stabilizer.
        hessian += 1e-9

        step = (
            gradient
            /
            hessian
        )

        new_beta = (
            beta
            -
            step
        )

        # Defensive guard against pathological data.
        new_beta = float(
            np.clip(
                new_beta,
                -5.0,
                5.0,
            )
        )

        if abs(
            new_beta
            -
            beta
        ) < 1e-10:

            beta = new_beta

            break

        beta = new_beta

    return float(
        beta
    )


# ============================================================
# SCORING
# ============================================================

def brier_score(
    outcome,
    probability,
):

    return float(
        np.mean(
            (
                probability
                -
                outcome
            )
            ** 2
        )
    )


def log_loss(
    outcome,
    probability,
):

    probability = np.clip(
        probability,
        EPSILON,
        1.0 - EPSILON,
    )

    return float(
        -np.mean(
            outcome
            *
            np.log(
                probability
            )
            +
            (
                1.0
                -
                outcome
            )
            *
            np.log(
                1.0
                -
                probability
            )
        )
    )


# ============================================================
# EXECUTION
# ============================================================

def build_trade(
    row,
    fair_yes,
):

    outcome = int(
        row[
            "outcome"
        ]
    )

    candidates = []

    # --------------------------------------------------------
    # YES
    # --------------------------------------------------------

    yes_ask = row[
        "yes_ask"
    ]

    if pd.notna(
        yes_ask
    ):

        yes_ask = float(
            yes_ask
        )

        if (
            0
            <
            yes_ask
            <
            1
        ):

            fee = float(
                calculate_taker_fee(
                    price=
                        yes_ask,

                    contracts=
                        1,

                    multiplier=
                        1,
                )
            )

            total_cost = (
                yes_ask
                +
                fee
            )

            edge = (
                fair_yes
                -
                total_cost
            )

            payout = (
                1.0
                if outcome == 1
                else 0.0
            )

            candidates.append(
                {
                    "side":
                        "YES",

                    "entry":
                        yes_ask,

                    "fee":
                        fee,

                    "cost":
                        total_cost,

                    "edge":
                        edge,

                    "won":
                        outcome == 1,

                    "pnl":
                        payout
                        -
                        total_cost,
                }
            )

    # --------------------------------------------------------
    # NO
    # --------------------------------------------------------

    no_ask = row[
        "no_ask"
    ]

    if pd.notna(
        no_ask
    ):

        no_ask = float(
            no_ask
        )

        if (
            0
            <
            no_ask
            <
            1
        ):

            fee = float(
                calculate_taker_fee(
                    price=
                        no_ask,

                    contracts=
                        1,

                    multiplier=
                        1,
                )
            )

            total_cost = (
                no_ask
                +
                fee
            )

            fair_no = (
                1.0
                -
                fair_yes
            )

            edge = (
                fair_no
                -
                total_cost
            )

            payout = (
                1.0
                if outcome == 0
                else 0.0
            )

            candidates.append(
                {
                    "side":
                        "NO",

                    "entry":
                        no_ask,

                    "fee":
                        fee,

                    "cost":
                        total_cost,

                    "edge":
                        edge,

                    "won":
                        outcome == 0,

                    "pnl":
                        payout
                        -
                        total_cost,
                }
            )

    if not candidates:

        return None

    best = max(
        candidates,
        key=lambda candidate:
            candidate[
                "edge"
            ],
    )

    if (
        best[
            "edge"
        ]
        <
        MIN_NET_EDGE
    ):

        return None

    return best


# ============================================================
# WALK-FORWARD
# ============================================================

def walk_forward(
    data,
    proxy_name,
    proxy_column,
):

    distance_column = (
        f"{proxy_name}_distance_bps"
    )

    dates = sorted(
        data[
            "date"
        ]
        .dropna()
        .unique()
    )

    predictions = []

    trades = []

    betas = []

    decision_days = 0

    for current_date in dates:

        current_date = pd.Timestamp(
            current_date
        )

        training_start = (
            current_date
            -
            pd.Timedelta(
                days=
                    LOOKBACK_DAYS
            )
        )

        training = (
            data[
                (
                    data[
                        "date"
                    ]
                    >=
                    training_start
                )
                &
                (
                    data[
                        "date"
                    ]
                    <
                    current_date
                )
            ]
            .dropna(
                subset=[
                    distance_column,
                    "market_logit",
                    "outcome",
                ]
            )
            .copy()
        )

        test = (
            data[
                data[
                    "date"
                ]
                ==
                current_date
            ]
            .dropna(
                subset=[
                    distance_column,
                    "market_logit",
                    "outcome",
                ]
            )
            .copy()
        )

        if (
            len(
                training
            )
            <
            MINIMUM_TRAINING_ROWS
            or
            test.empty
        ):

            continue

        beta = fit_beta(
            training=
                training,

            distance_column=
                distance_column,
        )

        decision_days += 1

        betas.append(
            {
                "date":
                    current_date,

                "beta":
                    beta,

                "training_rows":
                    len(
                        training
                    ),
            }
        )

        distance = (
            test[
                distance_column
            ]
            .to_numpy(
                dtype=float
            )
        )

        adjusted = sigmoid(
            test[
                "market_logit"
            ]
            .to_numpy(
                dtype=float
            )
            +
            beta
            *
            distance
        )

        test[
            "adjusted_probability"
        ] = adjusted

        for _, row in test.iterrows():

            predictions.append(
                {
                    "ticker":
                        row[
                            "ticker"
                        ],

                    "quote_time":
                        row[
                            "quote_time"
                        ],

                    "outcome":
                        int(
                            row[
                                "outcome"
                            ]
                        ),

                    "market_probability":
                        float(
                            row[
                                "market_probability"
                            ]
                        ),

                    "adjusted_probability":
                        float(
                            row[
                                "adjusted_probability"
                            ]
                        ),

                    "distance_bps":
                        float(
                            row[
                                distance_column
                            ]
                        ),

                    "beta":
                        beta,
                }
            )

            trade = build_trade(
                row=
                    row,

                fair_yes=
                    float(
                        row[
                            "adjusted_probability"
                        ]
                    ),
            )

            if trade is not None:

                trades.append(
                    {
                        "ticker":
                            row[
                                "ticker"
                            ],

                        "quote_time":
                            row[
                                "quote_time"
                            ],

                        "beta":
                            beta,

                        "distance_bps":
                            float(
                                row[
                                    distance_column
                                ]
                            ),

                        **trade,
                    }
                )

    return (
        pd.DataFrame(
            predictions
        ),
        pd.DataFrame(
            trades
        ),
        pd.DataFrame(
            betas
        ),
        decision_days,
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_model(
    predictions,
    trades,
    betas,
    decision_days,
):

    if predictions.empty:

        return {}

    outcome = (
        predictions[
            "outcome"
        ]
        .to_numpy(
            dtype=float
        )
    )

    raw = (
        predictions[
            "market_probability"
        ]
        .to_numpy(
            dtype=float
        )
    )

    adjusted = (
        predictions[
            "adjusted_probability"
        ]
        .to_numpy(
            dtype=float
        )
    )

    raw_brier = brier_score(
        outcome,
        raw,
    )

    adjusted_brier = brier_score(
        outcome,
        adjusted,
    )

    raw_log = log_loss(
        outcome,
        raw,
    )

    adjusted_log = log_loss(
        outcome,
        adjusted,
    )

    if trades.empty:

        trade_count = 0
        wins = 0
        win_rate = np.nan
        average_edge = np.nan
        capital = 0.0
        pnl = 0.0
        roi = np.nan

    else:

        trade_count = len(
            trades
        )

        wins = int(
            trades[
                "won"
            ]
            .sum()
        )

        win_rate = float(
            trades[
                "won"
            ]
            .mean()
        )

        average_edge = float(
            trades[
                "edge"
            ]
            .mean()
        )

        capital = float(
            trades[
                "cost"
            ]
            .sum()
        )

        pnl = float(
            trades[
                "pnl"
            ]
            .sum()
        )

        roi = (
            pnl
            /
            capital
            if capital > 0
            else np.nan
        )

    return {
        "days":
            decision_days,

        "test_rows":
            len(
                predictions
            ),

        "mean_beta":
            float(
                betas[
                    "beta"
                ]
                .mean()
            ),

        "median_beta":
            float(
                betas[
                    "beta"
                ]
                .median()
            ),

        "positive_beta_days":
            float(
                (
                    betas[
                        "beta"
                    ]
                    >
                    0
                )
                .mean()
            ),

        "raw_brier":
            raw_brier,

        "adjusted_brier":
            adjusted_brier,

        "brier_improvement":
            (
                raw_brier
                -
                adjusted_brier
            )
            /
            raw_brier,

        "raw_logloss":
            raw_log,

        "adjusted_logloss":
            adjusted_log,

        "logloss_improvement":
            (
                raw_log
                -
                adjusted_log
            )
            /
            raw_log,

        "trades":
            trade_count,

        "wins":
            wins,

        "win_rate":
            win_rate,

        "avg_edge":
            average_edge,

        "capital":
            capital,

        "pnl":
            pnl,

        "roi":
            roi,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 180)
    print(
        "KXBTC15M — CROSS-EXCHANGE INCREMENTAL INFORMATION WALK-FORWARD"
    )
    print("=" * 180)

    print(
        f"Development: "
        f"{DEVELOPMENT_START.date()} "
        f"-> "
        f"{(
            DEVELOPMENT_END
            -
            pd.Timedelta(days=1)
        ).date()}"
    )

    print(
        f"Horizon: T-{HORIZON}"
    )

    print(
        f"Rolling training window: "
        f"{LOOKBACK_DAYS} calendar days"
    )

    print(
        f"Minimum executable net edge: "
        f"{MIN_NET_EDGE:.2%}"
    )

    print()

    print(
        "Model:"
    )

    print(
        "  adjusted_logit = "
        "market_logit + beta * proxy_distance_bps"
    )

    print()

    data = load_dataset()

    print(
        f"Usable synchronized T-5 rows: "
        f"{len(data)}"
    )

    print(
        f"Range: "
        f"{data['quote_time'].min()} "
        f"-> "
        f"{data['quote_time'].max()}"
    )

    results = []

    for (
        proxy_name,
        proxy_column,
    ) in PROXIES.items():

        print()
        print(
            f"Running {proxy_name}..."
        )

        (
            predictions,
            trades,
            betas,
            decision_days,
        ) = walk_forward(
            data=data,
            proxy_name=
                proxy_name,
            proxy_column=
                proxy_column,
        )

        summary = summarize_model(
            predictions=
                predictions,
            trades=
                trades,
            betas=
                betas,
            decision_days=
                decision_days,
        )

        results.append(
            {
                "proxy":
                    proxy_name,

                **summary,
            }
        )

    result = pd.DataFrame(
        results
    )

    print()
    print("=" * 210)
    print(
        "WALK-FORWARD RESULTS"
    )
    print("=" * 210)

    print(
        result.to_string(
            index=False,
            formatters={
                "mean_beta":
                    lambda x:
                        f"{x:+.5f}",

                "median_beta":
                    lambda x:
                        f"{x:+.5f}",

                "positive_beta_days":
                    lambda x:
                        f"{x:.2%}",

                "raw_brier":
                    lambda x:
                        f"{x:.6f}",

                "adjusted_brier":
                    lambda x:
                        f"{x:.6f}",

                "brier_improvement":
                    lambda x:
                        f"{x:+.2%}",

                "raw_logloss":
                    lambda x:
                        f"{x:.6f}",

                "adjusted_logloss":
                    lambda x:
                        f"{x:.6f}",

                "logloss_improvement":
                    lambda x:
                        f"{x:+.2%}",

                "win_rate":
                    lambda x:
                        (
                            "n/a"
                            if pd.isna(x)
                            else
                            f"{x:.2%}"
                        ),

                "avg_edge":
                    lambda x:
                        (
                            "n/a"
                            if pd.isna(x)
                            else
                            f"{x:.2%}"
                        ),

                "capital":
                    lambda x:
                        f"${x:.2f}",

                "pnl":
                    lambda x:
                        f"${x:+.2f}",

                "roi":
                    lambda x:
                        (
                            "n/a"
                            if pd.isna(x)
                            else
                            f"{x:+.2%}"
                        ),
            },
        )
    )


if __name__ == "__main__":

    main()