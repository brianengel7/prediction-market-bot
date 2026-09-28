import pandas as pd
import numpy as np

from src.backtest.probability_calibration import (
    run_probability_backtest
)

from src.weather.fair_distribution import (
    probability_for_market
)

from src.database.db import (
    get_historical_market_entries
)

from src.kalshi.fees import (
    calculate_taker_fee
)


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2026-07-19"
END_DATE = "2026-09-10"

EDGE_THRESHOLDS = (
    0.03,
    0.04,
    0.05,
    0.075,
    0.10
)


# ============================================================
# BUILD CONTRACT EVALUATIONS
# ============================================================

def build_contract_evaluations():

    # --------------------------------------------------------
    # 1. Get true walk-forward weather probabilities
    # --------------------------------------------------------

    scored, _ = (
        run_probability_backtest()
    )

    if scored.empty:

        raise RuntimeError(
            "Probability backtest "
            "returned no scored dates."
        )

    scored = scored.copy()

    scored[
        "target_date"
    ] = (
        pd.to_datetime(
            scored[
                "target_date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # --------------------------------------------------------
    # 2. Load historical Kalshi executable entries
    # --------------------------------------------------------

    markets = (
        get_historical_market_entries(
            start_date=
                START_DATE,

            end_date=
                END_DATE
        )
    )

    if markets.empty:

        raise RuntimeError(
            "No historical Kalshi "
            "market entries found."
        )

    markets = markets.copy()

    markets[
        "target_date"
    ] = (
        pd.to_datetime(
            markets[
                "target_date"
            ]
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # --------------------------------------------------------
    # 3. Build calibration lookup by date
    # --------------------------------------------------------

    calibration_lookup = (
        scored
        .set_index(
            "target_date"
        )
        .to_dict(
            orient="index"
        )
    )

    results = []

    # --------------------------------------------------------
    # 4. Evaluate every historical contract
    # --------------------------------------------------------

    for _, market in markets.iterrows():

        target_date = (
            market[
                "target_date"
            ]
        )

        calibration = (
            calibration_lookup.get(
                target_date
            )
        )

        # Probability backtest not ready yet
        # for this historical date.
        if calibration is None:
            continue

        point_forecast = float(
            calibration[
                "point_forecast"
            ]
        )

        residual_mean = float(
            calibration[
                "residual_mean"
            ]
        )

        residual_std = float(
            calibration[
                "residual_std"
            ]
        )

        # ----------------------------------------------------
        # Model fair probability
        # ----------------------------------------------------

        model_yes = (
            probability_for_market(
                strike_type=
                    market[
                        "strike_type"
                    ],

                floor_strike=
                    market[
                        "floor_strike"
                    ],

                cap_strike=
                    market[
                        "cap_strike"
                    ],

                point_forecast=
                    point_forecast,

                residual_mean=
                    residual_mean,

                residual_std=
                    residual_std
            )
        )

        model_no = (
            1.0
            - model_yes
        )

        # ----------------------------------------------------
        # Determine official payout
        # ----------------------------------------------------

        result = str(
            market.get(
                "result",
                ""
            )
        ).lower()

        settlement_value = (
            market.get(
                "settlement_value"
            )
        )

        if (
            settlement_value is not None
            and
            not pd.isna(
                settlement_value
            )
        ):

            yes_payout = float(
                settlement_value
            )

            no_payout = (
                1.0
                - yes_payout
            )

        elif result == "yes":

            yes_payout = 1.0
            no_payout = 0.0

        elif result == "no":

            yes_payout = 0.0
            no_payout = 1.0

        else:

            # Cannot grade this contract.
            continue

        # ----------------------------------------------------
        # Evaluate YES and NO independently
        # ----------------------------------------------------

        candidates = []

        yes_ask = market.get(
            "yes_ask"
        )

        if (
            yes_ask is not None
            and
            not pd.isna(
                yes_ask
            )
        ):

            yes_price = float(
                yes_ask
            )

            yes_fee = (
                calculate_taker_fee(
                    price=
                        yes_price,

                    contracts=
                        1
                )
            )

            yes_total_cost = (
                yes_price
                + yes_fee
            )

            yes_gross_edge = (
                model_yes
                - yes_price
            )

            yes_net_edge = (
                model_yes
                - yes_total_cost
            )

            candidates.append({

                "side":
                    "YES",

                "model_probability":
                    model_yes,

                "entry_price":
                    yes_price,

                "fee":
                    yes_fee,

                "total_cost":
                    yes_total_cost,

                "gross_edge":
                    yes_gross_edge,

                "net_edge":
                    yes_net_edge,

                "payout":
                    yes_payout
            })

        no_ask = market.get(
            "no_ask"
        )

        if (
            no_ask is not None
            and
            not pd.isna(
                no_ask
            )
        ):

            no_price = float(
                no_ask
            )

            no_fee = (
                calculate_taker_fee(
                    price=
                        no_price,

                    contracts=
                        1
                )
            )

            no_total_cost = (
                no_price
                + no_fee
            )

            no_gross_edge = (
                model_no
                - no_price
            )

            no_net_edge = (
                model_no
                - no_total_cost
            )

            candidates.append({

                "side":
                    "NO",

                "model_probability":
                    model_no,

                "entry_price":
                    no_price,

                "fee":
                    no_fee,

                "total_cost":
                    no_total_cost,

                "gross_edge":
                    no_gross_edge,

                "net_edge":
                    no_net_edge,

                "payout":
                    no_payout
            })

        if not candidates:
            continue

        # ----------------------------------------------------
        # Pick side with highest NET expected edge
        # ----------------------------------------------------

        best = max(
            candidates,
            key=lambda candidate:
                candidate[
                    "net_edge"
                ]
        )

        gross_pnl = (
            best["payout"]
            - best[
                "entry_price"
            ]
        )

        net_pnl = (
            best["payout"]
            - best[
                "total_cost"
            ]
        )

        net_return = (
            net_pnl
            / best[
                "total_cost"
            ]
        )

        won = (
            best[
                "payout"
            ] > 0
        )

        results.append({

            "target_date":
                target_date,

            "ticker":
                market[
                    "ticker"
                ],

            "title":
                market[
                    "title"
                ],

            "entry_time":
                market[
                    "entry_time"
                ],

            "side":
                best[
                    "side"
                ],

            "model_yes":
                model_yes,

            "model_no":
                model_no,

            "model_probability":
                best[
                    "model_probability"
                ],

            "entry_price":
                best[
                    "entry_price"
                ],

            "fee":
                best[
                    "fee"
                ],

            "total_cost":
                best[
                    "total_cost"
                ],

            "gross_edge":
                best[
                    "gross_edge"
                ],

            "net_edge":
                best[
                    "net_edge"
                ],

            "payout":
                best[
                    "payout"
                ],

            "won":
                won,

            "gross_pnl":
                gross_pnl,

            "net_pnl":
                net_pnl,

            "net_return":
                net_return,

            "point_forecast":
                point_forecast,

            "residual_mean":
                residual_mean,

            "residual_std":
                residual_std
        })

    return (
        pd.DataFrame(
            results
        ),
        scored,
        markets
    )


# ============================================================
# THRESHOLD BACKTEST
# ============================================================

def summarize_threshold(
    evaluations,
    threshold
):

    trades = evaluations[
        evaluations[
            "net_edge"
        ] >= threshold
    ].copy()

    if trades.empty:

        return {

            "threshold":
                threshold,

            "trades":
                0,

            "wins":
                0,

            "losses":
                0,

            "win_rate":
                None,

            "avg_net_edge":
                None,

            "avg_entry":
                None,

            "capital":
                0.0,

            "net_pnl":
                0.0,

            "roi":
                None,

            "dates":
                0
        }

    wins = int(
        trades[
            "won"
        ].sum()
    )

    losses = (
        len(trades)
        - wins
    )

    capital = float(
        trades[
            "total_cost"
        ].sum()
    )

    net_pnl = float(
        trades[
            "net_pnl"
        ].sum()
    )

    roi = (
        net_pnl
        / capital
        if capital > 0
        else None
    )

    return {

        "threshold":
            threshold,

        "trades":
            len(trades),

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            wins
            / len(trades),

        "avg_net_edge":
            trades[
                "net_edge"
            ].mean(),

        "avg_entry":
            trades[
                "entry_price"
            ].mean(),

        "capital":
            capital,

        "net_pnl":
            net_pnl,

        "roi":
            roi,

        "dates":
            trades[
                "target_date"
            ].nunique()
    }


# ============================================================
# DISPLAY
# ============================================================

def print_backtest_summary(
    evaluations,
    scored,
    markets
):

    print()
    print("=" * 100)
    print(
        "KALSHI WEATHER STRATEGY BACKTEST"
    )
    print("=" * 100)

    print(
        f"Historical market dates: "
        f"{markets['target_date'].nunique()}"
    )

    print(
        f"Probability dates:       "
        f"{scored['target_date'].nunique()}"
    )

    print(
        f"Overlap dates:           "
        f"{evaluations['target_date'].nunique()}"
    )

    print(
        f"Contracts evaluated:     "
        f"{len(evaluations)}"
    )

    print()
    print(
        "NET EDGE THRESHOLD COMPARISON"
    )
    print("-" * 100)

    print(
        f"{'Threshold':<12}"
        f"{'Trades':<10}"
        f"{'Wins':<8}"
        f"{'Losses':<8}"
        f"{'Win Rate':<12}"
        f"{'Avg Edge':<12}"
        f"{'Capital':<12}"
        f"{'Net P&L':<12}"
        f"{'ROI':<12}"
    )

    print("-" * 100)

    for threshold in EDGE_THRESHOLDS:

        summary = (
            summarize_threshold(
                evaluations,
                threshold
            )
        )

        if summary[
            "trades"
        ] == 0:

            print(
                f"{threshold:<12.2%}"
                f"{0:<10}"
                f"{0:<8}"
                f"{0:<8}"
                f"{'N/A':<12}"
                f"{'N/A':<12}"
                f"${0:<11.2f}"
                f"${0:<11.2f}"
                f"{'N/A':<12}"
            )

            continue

        print(
            f"{summary['threshold']:<12.2%}"
            f"{summary['trades']:<10}"
            f"{summary['wins']:<8}"
            f"{summary['losses']:<8}"
            f"{summary['win_rate']:<12.2%}"
            f"{summary['avg_net_edge']:<12.2%}"
            f"${summary['capital']:<11.2f}"
            f"${summary['net_pnl']:<11.2f}"
            f"{summary['roi']:<12.2%}"
        )

def print_diagnostics(
    evaluations
):

    print()
    print("=" * 100)
    print(
        "BACKTEST DIAGNOSTICS"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # 1. Probability mass check
    #
    # The six KXHIGHNY contracts are mutually exclusive
    # and exhaustive.
    #
    # Therefore:
    #
    # sum(model_yes) ≈ 1.00
    #
    # for a complete six-contract date.
    # --------------------------------------------------------

    counts = (
        evaluations
        .groupby(
            "target_date"
        )
        .size()
    )

    complete_dates = (
        counts[
            counts == 6
        ]
        .index
    )

    complete = evaluations[
        evaluations[
            "target_date"
        ].isin(
            complete_dates
        )
    ]

    probability_mass = (
        complete
        .groupby(
            "target_date"
        )[
            "model_yes"
        ]
        .sum()
    )

    print()
    print(
        "MODEL PROBABILITY MASS"
    )
    print("-" * 100)

    print(
        f"Complete dates: "
        f"{len(probability_mass)}"
    )

    if not probability_mass.empty:

        print(
            f"Mean total:    "
            f"{probability_mass.mean():.4f}"
        )

        print(
            f"Minimum total: "
            f"{probability_mass.min():.4f}"
        )

        print(
            f"Maximum total: "
            f"{probability_mass.max():.4f}"
        )

        print()
        print(
            "Worst deviations from 100%:"
        )

        deviation = (
            probability_mass
            - 1.0
        ).abs()

        worst_dates = (
            deviation
            .sort_values(
                ascending=False
            )
            .head(10)
            .index
        )

        for date in worst_dates:

            print(
                f"  {date}: "
                f"{probability_mass.loc[date]:.4f}"
            )

    # --------------------------------------------------------
    # 2. Edge calibration
    # --------------------------------------------------------

    print()
    print(
        "EDGE CALIBRATION"
    )
    print("-" * 100)

    for threshold in EDGE_THRESHOLDS:

        trades = evaluations[
            evaluations[
                "net_edge"
            ] >= threshold
        ]

        if trades.empty:
            continue

        average_probability = (
            trades[
                "model_probability"
            ].mean()
        )

        actual_win_rate = (
            trades[
                "won"
            ].mean()
        )

        average_cost = (
            trades[
                "total_cost"
            ].mean()
        )

        expected_pnl = (
            trades[
                "net_edge"
            ].sum()
        )

        actual_pnl = (
            trades[
                "net_pnl"
            ].sum()
        )

        print()
        print(
            f"Threshold: "
            f"{threshold:.2%}"
        )

        print(
            f"  Trades:                "
            f"{len(trades)}"
        )

        print(
            f"  Avg model probability: "
            f"{average_probability:.2%}"
        )

        print(
            f"  Actual win rate:       "
            f"{actual_win_rate:.2%}"
        )

        print(
            f"  Avg total cost:        "
            f"{average_cost:.2%}"
        )

        print(
            f"  Expected P&L:          "
            f"${expected_pnl:+.2f}"
        )

        print(
            f"  Actual P&L:            "
            f"${actual_pnl:+.2f}"
        )

    # --------------------------------------------------------
    # 3. YES vs NO
    # --------------------------------------------------------

    print()
    print(
        "SIDE BREAKDOWN — 3% NET EDGE"
    )
    print("-" * 100)

    trades = evaluations[
        evaluations[
            "net_edge"
        ] >= 0.03
    ]

    for side in (
        "YES",
        "NO"
    ):

        side_trades = trades[
            trades[
                "side"
            ] == side
        ]

        if side_trades.empty:
            continue

        print()
        print(
            f"{side}:"
        )

        print(
            f"  Trades:      "
            f"{len(side_trades)}"
        )

        print(
            f"  Win rate:    "
            f"{side_trades['won'].mean():.2%}"
        )

        print(
            f"  Avg edge:    "
            f"{side_trades['net_edge'].mean():.2%}"
        )

        print(
            f"  Net P&L:     "
            f"${side_trades['net_pnl'].sum():+.2f}"
        )

    # --------------------------------------------------------
    # 4. Edge buckets
    # --------------------------------------------------------

    data = evaluations.copy()

    data[
        "edge_bucket"
    ] = pd.cut(
        data[
            "net_edge"
        ],
        bins=[
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            1.00
        ],
        right=False
    )

    bucket_summary = (
        data.dropna(
            subset=[
                "edge_bucket"
            ]
        )
        .groupby(
            "edge_bucket",
            observed=True
        )
        .agg(
            trades=(
                "won",
                "size"
            ),

            average_model_probability=(
                "model_probability",
                "mean"
            ),

            win_rate=(
                "won",
                "mean"
            ),

            average_edge=(
                "net_edge",
                "mean"
            ),

            net_pnl=(
                "net_pnl",
                "sum"
            )
        )
    )

    print()
    print(
        "EDGE BUCKETS"
    )
    print("-" * 100)

    print(
        bucket_summary
    )

def print_settlement_alignment(
    scored,
    markets
):

    print()
    print("=" * 100)
    print(
        "SETTLEMENT SOURCE ALIGNMENT"
    )
    print("=" * 100)

    scored_lookup = (
        scored
        .set_index(
            "target_date"
        )
        .to_dict(
            orient="index"
        )
    )

    rows = []

    for (
        target_date,
        group
    ) in markets.groupby(
        "target_date"
    ):

        probability_row = (
            scored_lookup.get(
                target_date
            )
        )

        if probability_row is None:
            continue

        actual_temperature = int(
            probability_row[
                "actual_temperature"
            ]
        )

        official_yes = []

        nws_yes = []

        for _, market in group.iterrows():

            ticker = market[
                "ticker"
            ]

            result = str(
                market.get(
                    "result",
                    ""
                )
            ).lower()

            # -----------------------------------------------
            # Official Kalshi winner
            # -----------------------------------------------

            if result == "yes":

                official_yes.append(
                    ticker
                )

            # -----------------------------------------------
            # Winner implied by NWS actual temperature
            # -----------------------------------------------

            strike_type = (
                market[
                    "strike_type"
                ]
            )

            floor_strike = (
                market[
                    "floor_strike"
                ]
            )

            cap_strike = (
                market[
                    "cap_strike"
                ]
            )

            wins = False

            if strike_type == "less":

                wins = (
                    actual_temperature
                    < float(
                        cap_strike
                    )
                )

            elif strike_type == "greater":

                wins = (
                    actual_temperature
                    > float(
                        floor_strike
                    )
                )

            elif strike_type == "between":

                wins = (
                    actual_temperature
                    >= float(
                        floor_strike
                    )
                    and
                    actual_temperature
                    <= float(
                        cap_strike
                    )
                )

            if wins:

                nws_yes.append(
                    ticker
                )

        if (
            len(official_yes) != 1
            or
            len(nws_yes) != 1
        ):

            continue

        rows.append({

            "target_date":
                target_date,

            "actual_temperature":
                actual_temperature,

            "official_winner":
                official_yes[0],

            "nws_winner":
                nws_yes[0],

            "same_winner":
                (
                    official_yes[0]
                    ==
                    nws_yes[0]
                )
        })

    alignment = pd.DataFrame(
        rows
    )

    if alignment.empty:

        print(
            "No comparable settlement "
            "dates found."
        )

        return

    transition_date = (
        "2026-08-14"
    )

    before = alignment[
        alignment[
            "target_date"
        ] < transition_date
    ]

    after = alignment[
        alignment[
            "target_date"
        ] >= transition_date
    ]

    print()
    print(
        f"Total comparable dates: "
        f"{len(alignment)}"
    )

    print(
        f"Overall agreement:      "
        f"{alignment['same_winner'].mean():.2%}"
    )

    print()

    print(
        "BEFORE AUGUST 14"
    )

    print(
        f"Dates:     "
        f"{len(before)}"
    )

    if not before.empty:

        print(
            f"Agreement: "
            f"{before['same_winner'].mean():.2%}"
        )

    print()

    print(
        "AUGUST 14 AND LATER"
    )

    print(
        f"Dates:     "
        f"{len(after)}"
    )

    if not after.empty:

        print(
            f"Agreement: "
            f"{after['same_winner'].mean():.2%}"
        )

    disagreements = alignment[
        ~alignment[
            "same_winner"
        ]
    ]

    print()
    print(
        "DISAGREEMENT DATES"
    )

    print("-" * 100)

    if disagreements.empty:

        print(
            "None."
        )

    else:

        print(
            disagreements[
                [
                    "target_date",
                    "actual_temperature",
                    "nws_winner",
                    "official_winner"
                ]
            ].to_string(
                index=False
            )
        )

def print_model_vs_market(
    evaluations,
    markets
):

    print()
    print("=" * 100)
    print(
        "MODEL VS KALSHI MARKET"
    )
    print("=" * 100)

    market_columns = markets[
        [
            "target_date",
            "ticker",
            "yes_bid",
            "yes_ask",
            "result"
        ]
    ].copy()

    data = evaluations.merge(
        market_columns,
        on=[
            "target_date",
            "ticker"
        ],
        how="left"
    )

    data["market_mid"] = (
        (
            data["yes_bid"]
            +
            data["yes_ask"]
        )
        / 2.0
    )

    daily_results = []

    for target_date, group in data.groupby(
        "target_date"
    ):

        # Only compare complete six-contract days.
        if len(group) != 6:
            continue

        group = group.copy()

        # -----------------------------------------------
        # Normalize market midpoints.
        #
        # Bid/ask spreads mean raw midpoints may not
        # add to exactly 1.
        # -----------------------------------------------

        market_total = (
            group[
                "market_mid"
            ].sum()
        )

        if market_total <= 0:
            continue

        group[
            "market_probability"
        ] = (
            group[
                "market_mid"
            ]
            / market_total
        )

        # -----------------------------------------------
        # Actual outcome
        # -----------------------------------------------

        group[
            "actual"
        ] = (
            group[
                "result"
            ].astype(str).str.lower()
            == "yes"
        ).astype(float)

        if group[
            "actual"
        ].sum() != 1:

            continue

        # -----------------------------------------------
        # Multiclass Brier scores
        # -----------------------------------------------

        model_brier = (
            (
                group[
                    "model_yes"
                ]
                -
                group[
                    "actual"
                ]
            ) ** 2
        ).sum()

        market_brier = (
            (
                group[
                    "market_probability"
                ]
                -
                group[
                    "actual"
                ]
            ) ** 2
        ).sum()

        # -----------------------------------------------
        # Probability assigned to actual winner
        # -----------------------------------------------

        winner = group[
            group[
                "actual"
            ] == 1
        ].iloc[0]

        model_winner_probability = float(
            winner[
                "model_yes"
            ]
        )

        market_winner_probability = float(
            winner[
                "market_probability"
            ]
        )

        # Protect log loss from log(0).
        epsilon = 1e-12

        model_log_loss = (
            -np.log(
                max(
                    model_winner_probability,
                    epsilon
                )
            )
        )

        market_log_loss = (
            -np.log(
                max(
                    market_winner_probability,
                    epsilon
                )
            )
        )

        daily_results.append({

            "target_date":
                target_date,

            "model_brier":
                model_brier,

            "market_brier":
                market_brier,

            "model_log_loss":
                model_log_loss,

            "market_log_loss":
                market_log_loss,

            "model_winner_probability":
                model_winner_probability,

            "market_winner_probability":
                market_winner_probability
        })

    comparison = pd.DataFrame(
        daily_results
    )

    if comparison.empty:

        print(
            "No complete dates available."
        )

        return

    print()

    print(
        f"Dates compared: "
        f"{len(comparison)}"
    )

    print()

    print(
        "MULTICLASS BRIER SCORE"
    )

    print(
        f"  Model:  "
        f"{comparison['model_brier'].mean():.4f}"
    )

    print(
        f"  Kalshi: "
        f"{comparison['market_brier'].mean():.4f}"
    )

    print()

    print(
        "LOG LOSS"
    )

    print(
        f"  Model:  "
        f"{comparison['model_log_loss'].mean():.4f}"
    )

    print(
        f"  Kalshi: "
        f"{comparison['market_log_loss'].mean():.4f}"
    )

    print()

    print(
        "AVG PROBABILITY GIVEN TO ACTUAL WINNER"
    )

    print(
        f"  Model:  "
        f"{comparison['model_winner_probability'].mean():.2%}"
    )

    print(
        f"  Kalshi: "
        f"{comparison['market_winner_probability'].mean():.2%}"
    )

    print()

    model_better = (
        comparison[
            "model_brier"
        ]
        <
        comparison[
            "market_brier"
        ]
    ).mean()

    print(
        f"Model beats Kalshi on Brier: "
        f"{model_better:.2%} of days"
    )

def print_model_market_blend(
    evaluations,
    markets
):

    print()
    print("=" * 100)
    print(
        "MODEL + KALSHI BLEND"
    )
    print("=" * 100)

    market_columns = markets[
        [
            "target_date",
            "ticker",
            "yes_bid",
            "yes_ask",
            "result"
        ]
    ].copy()

    data = evaluations.merge(
        market_columns,
        on=[
            "target_date",
            "ticker"
        ],
        how="left"
    )

    data["market_mid"] = (
        (
            data["yes_bid"]
            +
            data["yes_ask"]
        )
        / 2.0
    )

    weights = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ]

    results = []

    for model_weight in weights:

        daily_brier = []
        daily_log_loss = []
        daily_winner_probability = []

        for target_date, group in data.groupby(
            "target_date"
        ):

            if len(group) != 6:
                continue

            group = group.copy()

            # --------------------------------------------
            # Normalize Kalshi probabilities
            # --------------------------------------------

            market_total = (
                group[
                    "market_mid"
                ].sum()
            )

            if market_total <= 0:
                continue

            group[
                "market_probability"
            ] = (
                group[
                    "market_mid"
                ]
                / market_total
            )

            # --------------------------------------------
            # Actual winner
            # --------------------------------------------

            group[
                "actual"
            ] = (
                group[
                    "result"
                ]
                .astype(str)
                .str.lower()
                == "yes"
            ).astype(float)

            if group[
                "actual"
            ].sum() != 1:

                continue

            # --------------------------------------------
            # Blend
            #
            # weight 0 = pure Kalshi
            # weight 1 = pure weather model
            # --------------------------------------------

            group[
                "blended_probability"
            ] = (
                model_weight
                * group[
                    "model_yes"
                ]
                +
                (
                    1.0
                    - model_weight
                )
                * group[
                    "market_probability"
                ]
            )

            # --------------------------------------------
            # Brier
            # --------------------------------------------

            brier = (
                (
                    group[
                        "blended_probability"
                    ]
                    -
                    group[
                        "actual"
                    ]
                ) ** 2
            ).sum()

            daily_brier.append(
                brier
            )

            # --------------------------------------------
            # Winner probability / log loss
            # --------------------------------------------

            winner = group[
                group[
                    "actual"
                ] == 1
            ].iloc[0]

            winner_probability = float(
                winner[
                    "blended_probability"
                ]
            )

            epsilon = 1e-12

            log_loss = (
                -np.log(
                    max(
                        winner_probability,
                        epsilon
                    )
                )
            )

            daily_log_loss.append(
                log_loss
            )

            daily_winner_probability.append(
                winner_probability
            )

        if not daily_brier:
            continue

        results.append({

            "model_weight":
                model_weight,

            "market_weight":
                1.0
                - model_weight,

            "brier":
                np.mean(
                    daily_brier
                ),

            "log_loss":
                np.mean(
                    daily_log_loss
                ),

            "winner_probability":
                np.mean(
                    daily_winner_probability
                )
        })

    results = pd.DataFrame(
        results
    )

    print()

    print(
        f"{'Model':<12}"
        f"{'Kalshi':<12}"
        f"{'Brier':<14}"
        f"{'Log Loss':<14}"
        f"{'Winner Prob':<14}"
    )

    print("-" * 66)

    for _, row in results.iterrows():

        print(
            f"{row['model_weight']:<12.0%}"
            f"{row['market_weight']:<12.0%}"
            f"{row['brier']:<14.4f}"
            f"{row['log_loss']:<14.4f}"
            f"{row['winner_probability']:<14.2%}"
        )

    best_brier = results.loc[
        results[
            "brier"
        ].idxmin()
    ]

    best_log = results.loc[
        results[
            "log_loss"
        ].idxmin()
    ]

    print()
    print(
        "Best Brier blend:"
    )

    print(
        f"  Model: "
        f"{best_brier['model_weight']:.0%}"
    )

    print(
        f"  Kalshi: "
        f"{best_brier['market_weight']:.0%}"
    )

    print(
        f"  Brier: "
        f"{best_brier['brier']:.4f}"
    )

    print()

    print(
        "Best log-loss blend:"
    )

    print(
        f"  Model: "
        f"{best_log['model_weight']:.0%}"
    )

    print(
        f"  Kalshi: "
        f"{best_log['market_weight']:.0%}"
    )

    print(
        f"  Log loss: "
        f"{best_log['log_loss']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    (
        evaluations,
        scored,
        markets
    ) = build_contract_evaluations()

    print_backtest_summary(
        evaluations,
        scored,
        markets
    )

    print_diagnostics(
        evaluations
    )

    print_settlement_alignment(
        scored,
        markets
    )

    print_model_vs_market(
        evaluations,
        markets
    )

    print_model_market_blend(
        evaluations,
        markets
    )

    