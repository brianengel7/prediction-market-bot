import argparse

import numpy as np
import pandas as pd

from src.database.db import (
    get_v2_nbm_probabilities,
    save_v2_shadow_trade
)

from src.kalshi.market_logger import (
    get_current_weather_event
)

from src.kalshi.fees import (
    calculate_taker_fee
)

from src.kalshi.live_executor import (
    execute_live_trade
)


# ============================================================
# FROZEN STRATEGY
# ============================================================

BETA = -0.40
MINIMUM_EDGE = 0.025

EPSILON = 1e-6

DECISION_HOUR_UTC = 20
DECISION_MINUTE_UTC = 5

# Only allow a real forward save shortly after 20:05 UTC.
SAVE_WINDOW_MINUTES = 15


# ============================================================
# MARKET PROBABILITIES
# ============================================================

def calculate_market_probabilities(
    markets
):
    """
    Convert the six Kalshi YES midpoints into a normalized
    probability distribution.

    This matches the v2 backtest methodology.
    """

    rows = []

    for market in markets:

        yes_bid = market.get(
            "yes_bid"
        )

        yes_ask = market.get(
            "yes_ask"
        )

        if (
            yes_bid is None
            or
            yes_ask is None
        ):
            raise RuntimeError(
                f"Incomplete YES quote for "
                f"{market['ticker']}."
            )

        midpoint = (
            float(
                yes_bid
            )
            +
            float(
                yes_ask
            )
        ) / 2.0

        rows.append({
            "ticker":
                market[
                    "ticker"
                ],

            "market_mid":
                midpoint
        })

    total_mid = sum(
        row[
            "market_mid"
        ]
        for row in rows
    )

    if total_mid <= 0:
        raise RuntimeError(
            "Invalid Kalshi probability mass."
        )

    for row in rows:

        row[
            "market_probability"
        ] = (
            row[
                "market_mid"
            ]
            /
            total_mid
        )

    return rows


# ============================================================
# FROZEN NBM ADJUSTMENT
# ============================================================

def calculate_adjusted_probabilities(
    market_probabilities,
    nbm_probabilities
):
    """
    Apply the frozen market-conditioned NBM transformation:

    log(adjusted score)
        =
        log(Kalshi)
        +
        beta * (
            log(NBM)
            -
            log(Kalshi)
        )

    beta = -0.40
    """

    market = np.clip(
        np.asarray(
            market_probabilities,
            dtype=float
        ),
        EPSILON,
        1.0
    )

    nbm = np.clip(
        np.asarray(
            nbm_probabilities,
            dtype=float
        ),
        EPSILON,
        1.0
    )

    log_scores = (
        np.log(
            market
        )
        +
        BETA
        *
        (
            np.log(
                nbm
            )
            -
            np.log(
                market
            )
        )
    )

    # Numerical stability
    log_scores = (
        log_scores
        -
        np.max(
            log_scores
        )
    )

    scores = np.exp(
        log_scores
    )

    adjusted = (
        scores
        /
        scores.sum()
    )

    return adjusted


# ============================================================
# BUILD ALL EXECUTABLE CANDIDATES
# ============================================================

def build_trade_candidates(
    markets,
    nbm_data
):

    market_rows = (
        calculate_market_probabilities(
            markets
        )
    )

    market_lookup = {
        row[
            "ticker"
        ]:
        row
        for row in market_rows
    }

    nbm_lookup = {
        row[
            "ticker"
        ]:
        row
        for row in nbm_data.to_dict(
            orient="records"
        )
    }

    tickers = [
        market[
            "ticker"
        ]
        for market in markets
    ]

    if len(tickers) != 6:

        raise RuntimeError(
            f"Expected 6 Kalshi contracts, "
            f"found {len(tickers)}."
        )

    for ticker in tickers:

        if ticker not in nbm_lookup:

            raise RuntimeError(
                f"No saved NBM probability "
                f"for {ticker}."
            )

    market_probabilities = [
        market_lookup[
            ticker
        ][
            "market_probability"
        ]
        for ticker in tickers
    ]

    nbm_probabilities = [
        float(
            nbm_lookup[
                ticker
            ][
                "nbm_probability"
            ]
        )
        for ticker in tickers
    ]

    # Normalize NBM again defensively.
    nbm_total = sum(
        nbm_probabilities
    )

    nbm_probabilities = [
        probability
        /
        nbm_total
        for probability
        in nbm_probabilities
    ]

    adjusted_probabilities = (
        calculate_adjusted_probabilities(
            market_probabilities=
                market_probabilities,

            nbm_probabilities=
                nbm_probabilities
        )
    )

    candidates = []

    for (
        market,
        market_probability,
        nbm_probability,
        adjusted_yes
    ) in zip(
        markets,
        market_probabilities,
        nbm_probabilities,
        adjusted_probabilities
    ):

        ticker = market[
            "ticker"
        ]

        # ----------------------------------------------------
        # YES
        # ----------------------------------------------------

        yes_ask = market.get(
            "yes_ask"
        )

        if yes_ask is not None:

            yes_ask = float(
                yes_ask
            )

            yes_fee = float(
                calculate_taker_fee(
                    price=
                        yes_ask,

                    contracts=
                        1,

                    multiplier=
                        1
                )
            )

            yes_total_cost = (
                yes_ask
                +
                yes_fee
            )

            yes_edge = (
                float(
                    adjusted_yes
                )
                -
                yes_total_cost
            )

            candidates.append({

                "ticker":
                    ticker,

                "side":
                    "YES",

                "kalshi_probability":
                    float(
                        market_probability
                    ),

                "nbm_probability":
                    float(
                        nbm_probability
                    ),

                "adjusted_probability":
                    float(
                        adjusted_yes
                    ),

                "entry_price":
                    yes_ask,

                "fee":
                    yes_fee,

                "total_cost":
                    yes_total_cost,

                "net_edge":
                    yes_edge
            })

        # ----------------------------------------------------
        # NO
        # ----------------------------------------------------

        no_ask = market.get(
            "no_ask"
        )

        if no_ask is not None:

            no_ask = float(
                no_ask
            )

            no_fee = float(
                calculate_taker_fee(
                    price=
                        no_ask,

                    contracts=
                        1,

                    multiplier=
                        1
                )
            )

            no_total_cost = (
                no_ask
                +
                no_fee
            )

            adjusted_no = (
                1.0
                -
                float(
                    adjusted_yes
                )
            )

            no_edge = (
                adjusted_no
                -
                no_total_cost
            )

            candidates.append({

                "ticker":
                    ticker,

                "side":
                    "NO",

                "kalshi_probability":
                    float(
                        1.0
                        -
                        market_probability
                    ),

                "nbm_probability":
                    float(
                        1.0
                        -
                        nbm_probability
                    ),

                "adjusted_probability":
                    adjusted_no,

                "entry_price":
                    no_ask,

                "fee":
                    no_fee,

                "total_cost":
                    no_total_cost,

                "net_edge":
                    no_edge
            })

    candidates.sort(
        key=lambda row:
            row[
                "net_edge"
            ],

        reverse=True
    )

    return candidates


# ============================================================
# TIMING GUARD
# ============================================================

def inside_save_window():

    now = pd.Timestamp.now(
        tz="UTC"
    )

    start = now.normalize() + pd.Timedelta(
        hours=
            DECISION_HOUR_UTC,

        minutes=
            DECISION_MINUTE_UTC
    )

    end = (
        start
        +
        pd.Timedelta(
            minutes=
                SAVE_WINDOW_MINUTES
        )
    )

    return (
        start
        <=
        now
        <=
        end
    )


# ============================================================
# SHADOW DECISION
# ============================================================

def run_shadow_trader(
    save=False,
    live=False
):
    if live and not save:

        raise RuntimeError(
            "--live requires --save. "
            "A real order cannot be submitted "
            "without saving the shadow decision."
        )

    event_info = (
        get_current_weather_event()
    )

    target_date = (
        event_info[
            "target_date"
        ]
    )

    markets = (
        event_info[
            "markets"
        ]
    )

    nbm_data = (
        get_v2_nbm_probabilities(
            target_date
        )
    )

    if len(
        nbm_data
    ) != 6:

        raise RuntimeError(
            f"Expected 6 saved NBM probabilities "
            f"for {target_date}, "
            f"found {len(nbm_data)}.\n"
            f"Run main.py first."
        )

    candidates = (
        build_trade_candidates(
            markets=
                markets,

            nbm_data=
                nbm_data
        )
    )

    best = candidates[
        0
    ]

    decision_time = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

    print()
    print("=" * 80)

    print(
        "WEATHER V2 SHADOW TRADER"
    )

    print("=" * 80)

    print(
        f"Target date:     "
        f"{target_date}"
    )

    print(
        f"Decision time:   "
        f"{decision_time}"
    )

    print(
        f"Frozen beta:     "
        f"{BETA:+.2f}"
    )

    print(
        f"Minimum edge:    "
        f"{MINIMUM_EDGE:.2%}"
    )

    print()

    print(
        "TOP CANDIDATES"
    )

    print("-" * 80)

    for candidate in candidates[
        :6
    ]:

        print(
            f"{candidate['ticker']:<28} "
            f"{candidate['side']:<3} "
            f"Kalshi: "
            f"{candidate['kalshi_probability']:.2%}  "
            f"NBM: "
            f"{candidate['nbm_probability']:.2%}  "
            f"Adjusted: "
            f"{candidate['adjusted_probability']:.2%}  "
            f"Cost: "
            f"{candidate['total_cost']:.2%}  "
            f"Edge: "
            f"{candidate['net_edge']:+.2%}"
        )

    print()
    print("-" * 80)

    if (
        best[
            "net_edge"
        ]
        <
        MINIMUM_EDGE
    ):

        print(
            f"DECISION: PASS"
        )

        print(
            f"Best available edge: "
            f"{best['net_edge']:+.2%}"
        )

        return None

    print(
        f"DECISION: "
        f"{best['side']} "
        f"{best['ticker']}"
    )

    print(
        f"Adjusted probability: "
        f"{best['adjusted_probability']:.2%}"
    )

    print(
        f"Entry ask:            "
        f"{best['entry_price']:.2%}"
    )

    print(
        f"Fee:                  "
        f"{best['fee']:.2%}"
    )

    print(
        f"Total cost:           "
        f"{best['total_cost']:.2%}"
    )

    print(
        f"Net edge:             "
        f"{best['net_edge']:+.2%}"
    )

    best[
        "target_date"
    ] = target_date

    best[
        "decision_time"
    ] = decision_time

    best[
        "beta"
    ] = BETA

    best[
        "minimum_edge"
    ] = MINIMUM_EDGE

    if not save:

        print()
        print(
            "DRY RUN ONLY - "
            "trade was NOT saved."
        )

        return best

    if not inside_save_window():

        raise RuntimeError(
            "Refusing to save forward trade: "
            "current time is outside the "
            "20:05-20:20 UTC decision window."
        )

    save_v2_shadow_trade(
        best
    )

    print()
    print(
        "SHADOW TRADE SAVED."
    )

    # ========================================================
    # OPTIONAL REAL-MONEY EXECUTION
    # ========================================================

    if live:

        print()
        print("=" * 80)
        print(
            "LIVE EXECUTION REQUESTED"
        )
        print("=" * 80)

        live_result = (
            execute_live_trade(
                trade=
                    best,

                strategy=
                    "NBM",

                contracts=
                    1
            )
        )

        print()
        print(
            "REAL ORDER SUBMITTED."
        )

        print(
            f"Kalshi order ID: "
            f"{live_result.get('order_id')}"
        )

    return best


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Save the qualifying shadow trade. "
            "Only allowed near 20:05 UTC."
        )
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Submit a qualifying shadow trade "
            "as a real Kalshi order."
        )
    )

    args = parser.parse_args()

    run_shadow_trader(
        save=
            args.save,

        live=
            args.live
    )
