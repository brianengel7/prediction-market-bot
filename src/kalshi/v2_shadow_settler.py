import argparse

import numpy as np
import pandas as pd

from src.database.db import (
    get_historical_market_entries,
    get_v2_shadow_trades,
    get_v2_market_only_shadow_decisions,
    update_v2_shadow_trade_settlement,
    update_v2_market_only_shadow_settlement
)


# ============================================================
# SETTLEMENT RESULT
# ============================================================

def parse_market_result(
    row
):
    """
    Return:

        "YES"
        "NO"
        None

    None means we do not yet have a valid settlement.
    """

    result = row.get(
        "result"
    )

    if result is not None:

        result_text = (
            str(
                result
            )
            .strip()
            .lower()
        )

        if result_text == "yes":
            return "YES"

        if result_text == "no":
            return "NO"

    settlement_value = row.get(
        "settlement_value"
    )

    if (
        settlement_value is not None
        and
        not pd.isna(
            settlement_value
        )
    ):

        settlement_value = float(
            settlement_value
        )

        if np.isclose(
            settlement_value,
            1.0
        ):
            return "YES"

        if np.isclose(
            settlement_value,
            0.0
        ):
            return "NO"

    return None


# ============================================================
# FIND SETTLEMENT FOR TICKER
# ============================================================

def get_contract_settlement(
    target_date,
    ticker
):

    history = (
        get_historical_market_entries(
            target_date=
                target_date
        )
    )

    if history.empty:

        return None

    matches = (
        history[
            history[
                "ticker"
            ]
            ==
            ticker
        ]
        .copy()
    )

    if matches.empty:

        return None

    # There may be multiple historical decision-time rows
    # for the same ticker. Settlement should be identical
    # across them, so take the newest valid result.

    matches = (
        matches.sort_values(
            "entry_time",
            ascending=False
        )
    )

    for _, row in (
        matches.iterrows()
    ):

        market_result = (
            parse_market_result(
                row
            )
        )

        if market_result is not None:

            return {
                "result":
                    market_result,

                "market_status":
                    row.get(
                        "market_status"
                    ),

                "settlement_value":
                    row.get(
                        "settlement_value"
                    )
            }

    return None


# ============================================================
# GRADE ONE TRADE
# ============================================================

def grade_trade(
    side,
    total_cost,
    market_result
):

    side = (
        str(
            side
        )
        .strip()
        .upper()
    )

    market_result = (
        str(
            market_result
        )
        .strip()
        .upper()
    )

    if side not in (
        "YES",
        "NO"
    ):

        raise ValueError(
            f"Invalid trade side: {side}"
        )

    if market_result not in (
        "YES",
        "NO"
    ):

        raise ValueError(
            f"Invalid settlement result: "
            f"{market_result}"
        )

    won = (
        side
        ==
        market_result
    )

    payout = (
        1.0
        if won
        else 0.0
    )

    net_pnl = (
        payout
        -
        float(
            total_cost
        )
    )

    return {
        "won":
            won,

        "payout":
            payout,

        "net_pnl":
            net_pnl
    }


# ============================================================
# SETTLE NBM TRADES
# ============================================================

def settle_nbm_trades(
    save=False
):

    trades = (
        get_v2_shadow_trades()
        .copy()
    )

    if trades.empty:

        return []

    unsettled = (
        trades[
            trades[
                "settled"
            ]
            ==
            0
        ]
        .copy()
    )

    results = []

    for _, trade in (
        unsettled.iterrows()
    ):

        settlement = (
            get_contract_settlement(
                target_date=
                    trade[
                        "target_date"
                    ],

                ticker=
                    trade[
                        "ticker"
                    ]
            )
        )

        if settlement is None:

            results.append({

                "strategy":
                    "NBM",

                "target_date":
                    trade[
                        "target_date"
                    ],

                "ticker":
                    trade[
                        "ticker"
                    ],

                "side":
                    trade[
                        "side"
                    ],

                "status":
                    "PENDING"
            })

            continue

        grade = (
            grade_trade(
                side=
                    trade[
                        "side"
                    ],

                total_cost=
                    trade[
                        "total_cost"
                    ],

                market_result=
                    settlement[
                        "result"
                    ]
            )
        )

        if save:

            update_v2_shadow_trade_settlement(
                trade_id=
                    trade[
                        "id"
                    ],

                won=
                    grade[
                        "won"
                    ],

                payout=
                    grade[
                        "payout"
                    ],

                net_pnl=
                    grade[
                        "net_pnl"
                    ]
            )

        results.append({

            "strategy":
                "NBM",

            "target_date":
                trade[
                    "target_date"
                ],

            "ticker":
                trade[
                    "ticker"
                ],

            "side":
                trade[
                    "side"
                ],

            "market_result":
                settlement[
                    "result"
                ],

            "won":
                grade[
                    "won"
                ],

            "total_cost":
                float(
                    trade[
                        "total_cost"
                    ]
                ),

            "payout":
                grade[
                    "payout"
                ],

            "net_pnl":
                grade[
                    "net_pnl"
                ],

            "status":
                "SETTLED"
        })

    return results


# ============================================================
# SETTLE MARKET-ONLY TRADES
# ============================================================

def settle_market_only_trades(
    save=False
):

    decisions = (
        get_v2_market_only_shadow_decisions()
        .copy()
    )

    if decisions.empty:

        return []

    # PASS decisions were stored as trade_taken=0.
    # They have no contract exposure and should not
    # be graded as wins or losses.

    unsettled = (
        decisions[
            (
                decisions[
                    "trade_taken"
                ]
                ==
                1
            )
            &
            (
                decisions[
                    "settled"
                ]
                ==
                0
            )
        ]
        .copy()
    )

    results = []

    for _, trade in (
        unsettled.iterrows()
    ):

        settlement = (
            get_contract_settlement(
                target_date=
                    trade[
                        "target_date"
                    ],

                ticker=
                    trade[
                        "ticker"
                    ]
            )
        )

        if settlement is None:

            results.append({

                "strategy":
                    "MARKET_ONLY",

                "target_date":
                    trade[
                        "target_date"
                    ],

                "ticker":
                    trade[
                        "ticker"
                    ],

                "side":
                    trade[
                        "side"
                    ],

                "status":
                    "PENDING"
            })

            continue

        grade = (
            grade_trade(
                side=
                    trade[
                        "side"
                    ],

                total_cost=
                    trade[
                        "total_cost"
                    ],

                market_result=
                    settlement[
                        "result"
                    ]
            )
        )

        if save:

            update_v2_market_only_shadow_settlement(
                decision_id=
                    trade[
                        "id"
                    ],

                won=
                    grade[
                        "won"
                    ],

                payout=
                    grade[
                        "payout"
                    ],

                net_pnl=
                    grade[
                        "net_pnl"
                    ]
            )

        results.append({

            "strategy":
                "MARKET_ONLY",

            "target_date":
                trade[
                    "target_date"
                ],

            "ticker":
                trade[
                    "ticker"
                ],

            "side":
                trade[
                    "side"
                ],

            "market_result":
                settlement[
                    "result"
                ],

            "won":
                grade[
                    "won"
                ],

            "total_cost":
                float(
                    trade[
                        "total_cost"
                    ]
                ),

            "payout":
                grade[
                    "payout"
                ],

            "net_pnl":
                grade[
                    "net_pnl"
                ],

            "status":
                "SETTLED"
        })

    return results


# ============================================================
# PRINT SETTLEMENT ATTEMPTS
# ============================================================

def print_results(
    results
):

    print()
    print("=" * 90)
    print(
        "SHADOW TRADE SETTLEMENT CHECK"
    )
    print("=" * 90)

    if not results:

        print(
            "No unsettled shadow trades."
        )

        return

    for result in results:

        if (
            result[
                "status"
            ]
            ==
            "PENDING"
        ):

            print(
                f"{result['strategy']:<12} | "
                f"{result['target_date']} | "
                f"{result['ticker']} | "
                f"{result['side']} | "
                f"PENDING"
            )

            continue

        outcome = (
            "WIN"
            if result[
                "won"
            ]
            else
            "LOSS"
        )

        print(
            f"{result['strategy']:<12} | "
            f"{result['target_date']} | "
            f"{result['ticker']} | "
            f"{result['side']} | "
            f"Market {result['market_result']} | "
            f"{outcome} | "
            f"P&L ${result['net_pnl']:+.2f}"
        )


# ============================================================
# CUMULATIVE FORWARD PERFORMANCE
# ============================================================

def summarize_strategy(
    dataframe,
    strategy_name,
    trade_filter=None
):

    if dataframe.empty:

        print(
            f"{strategy_name}: no decisions yet."
        )

        return

    data = (
        dataframe.copy()
    )

    if trade_filter is not None:

        data = (
            data[
                trade_filter(
                    data
                )
            ]
            .copy()
        )

    settled = (
        data[
            data[
                "settled"
            ]
            ==
            1
        ]
        .copy()
    )

    # Market-only PASS rows are marked settled but
    # should have been removed by trade_filter above.

    if settled.empty:

        print(
            f"{strategy_name}: "
            f"no settled trades yet."
        )

        return

    wins = int(
        settled[
            "won"
        ].fillna(
            0
        ).sum()
    )

    trades = len(
        settled
    )

    losses = (
        trades
        -
        wins
    )

    capital = float(
        settled[
            "total_cost"
        ].sum()
    )

    pnl = float(
        settled[
            "net_pnl"
        ].fillna(
            0
        ).sum()
    )

    roi = (
        pnl
        /
        capital
        if capital > 0
        else 0.0
    )

    print(
        f"{strategy_name}"
    )

    print(
        f"  Settled trades: "
        f"{trades}"
    )

    print(
        f"  Wins:           "
        f"{wins}"
    )

    print(
        f"  Losses:         "
        f"{losses}"
    )

    print(
        f"  Win rate:       "
        f"{wins / trades:.2%}"
    )

    print(
        f"  Capital:        "
        f"${capital:.2f}"
    )

    print(
        f"  Realized P&L:   "
        f"${pnl:+.2f}"
    )

    print(
        f"  Realized ROI:   "
        f"{roi:+.2%}"
    )


def print_forward_summary():

    print()
    print("=" * 90)
    print(
        "CUMULATIVE FORWARD PERFORMANCE"
    )
    print("=" * 90)

    nbm = (
        get_v2_shadow_trades()
    )

    market_only = (
        get_v2_market_only_shadow_decisions()
    )

    summarize_strategy(
        dataframe=
            nbm,

        strategy_name=
            "NBM"
    )

    print()

    summarize_strategy(
        dataframe=
            market_only,

        strategy_name=
            "MARKET-ONLY",

        trade_filter=
            lambda dataframe:
                dataframe[
                    "trade_taken"
                ]
                ==
                1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Write settlement results "
            "to the database."
        )
    )

    args = (
        parser.parse_args()
    )

    results = []

    results.extend(
        settle_nbm_trades(
            save=
                args.save
        )
    )

    results.extend(
        settle_market_only_trades(
            save=
                args.save
        )
    )

    print_results(
        results
    )

    if not args.save:

        print()
        print(
            "DRY RUN ONLY - database was NOT updated."
        )

    print_forward_summary()


if __name__ == "__main__":

    main()