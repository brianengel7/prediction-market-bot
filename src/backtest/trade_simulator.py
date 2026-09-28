from src.database.db import (
    get_market_snapshots
)

from src.weather.model_versions import (
    WEATHER_MODEL_VERSION
)

from src.kalshi.settlements import (
    get_market_settlement
)

from src.kalshi.fees import (
    calculate_taker_fee
)


# ============================================================
# BUILD SIMULATED TRADES
# ============================================================

def simulate_trades(
    target_date,
    minimum_edge=0.05
):
    """
    Simulate one trade per contract.

    Entry:
        First snapshot where executable model edge
        is at least minimum_edge.

    Exit:
        Official Kalshi settlement payout.

    Calculates both gross and net P&L,
    including Kalshi taker fees.
    """

    snapshots = get_market_snapshots(
        target_date=target_date,
        model_version=
            WEATHER_MODEL_VERSION
    )

    if snapshots.empty:

        print(
            f"No market snapshots "
            f"found for {target_date}."
        )

        return []

    trades = []

    # --------------------------------------------------------
    # Evaluate each Kalshi contract separately
    # --------------------------------------------------------

    for ticker, group in snapshots.groupby(
        "ticker"
    ):

        group = group.sort_values(
            "snapshot_time"
        )

        # ----------------------------------------------------
        # Find first net tradeable edge
        # ----------------------------------------------------

        entry = None
        side = None
        net_model_edge = None

        for _, row in group.iterrows():

            # -----------------------------------------------
            # YES side
            # -----------------------------------------------

            yes_ask = row[
                "yes_ask"
            ]

            if yes_ask is not None:

                yes_price = float(
                    yes_ask
                )

                yes_fee = (
                    calculate_taker_fee(
                        price=yes_price,
                        contracts=1
                    )
                )

                yes_total_cost = (
                    yes_price
                    + yes_fee
                )

                yes_net_edge = (
                    float(
                        row["model_yes"]
                    )
                    - yes_total_cost
                )

            else:

                yes_net_edge = None

            # -----------------------------------------------
            # NO side
            # -----------------------------------------------

            no_ask = row[
                "no_ask"
            ]

            if no_ask is not None:

                no_price = float(
                    no_ask
                )

                no_fee = (
                    calculate_taker_fee(
                        price=no_price,
                        contracts=1
                    )
                )

                no_total_cost = (
                    no_price
                    + no_fee
                )

                no_net_edge = (
                    float(
                        row["model_no"]
                    )
                    - no_total_cost
                )

            else:

                no_net_edge = None

            # -----------------------------------------------
            # Choose best NET side
            # -----------------------------------------------

            available_edges = []

            if yes_net_edge is not None:

                available_edges.append(
                    (
                        "YES",
                        yes_net_edge
                    )
                )

            if no_net_edge is not None:

                available_edges.append(
                    (
                        "NO",
                        no_net_edge
                    )
                )

            if not available_edges:
                continue

            row_side, row_net_edge = max(
                available_edges,
                key=lambda item:
                    item[1]
            )

            # -----------------------------------------------
            # First snapshot meeting required NET edge
            # -----------------------------------------------

            if row_net_edge >= minimum_edge:

                entry = row

                side = row_side

                net_model_edge = (
                    row_net_edge
                )

                break


        if entry is None:
            continue

        # ----------------------------------------------------
        # Executable entry price
        # ----------------------------------------------------

        if side == "YES":

            entry_price = float(
                entry["yes_ask"]
            )

        elif side == "NO":

            entry_price = float(
                entry["no_ask"]
            )

        else:
            continue

        if side == "YES":

            model_probability = float(
                entry["model_yes"]
            )

        else:

            model_probability = float(
                entry["model_no"]
            )


        gross_model_edge = (
            model_probability
            - entry_price
        )

        entry_fee = (
            calculate_taker_fee(
                price=entry_price,
                contracts=1
            )
        )

        total_cost = (
            entry_price
            + entry_fee
        )

        # ----------------------------------------------------
        # Get official Kalshi settlement
        # ----------------------------------------------------

        settlement = (
            get_market_settlement(
                ticker
            )
        )

        # ----------------------------------------------------
        # Market has not settled yet
        # ----------------------------------------------------

        if not settlement[
            "is_settled"
        ]:

            trade = {

                "ticker":
                    ticker,

                "title":
                    entry["title"],

                "entry_time":
                    entry[
                        "snapshot_time"
                    ],

                "side":
                    side,

                "model_edge":
                    gross_model_edge,

                "entry_price":
                    entry_price,

                "entry_fee":
                    entry_fee,

                "total_cost":
                    total_cost,

                "net_model_edge":
                    net_model_edge,

                "status":
                    "PENDING",

                "payout":
                    None,

                "gross_pnl":
                    None,

                "gross_return":
                    None
            }

            trades.append(
                trade
            )

            continue

        # ----------------------------------------------------
        # Official payout
        # ----------------------------------------------------

        if side == "YES":

            payout = float(
                settlement[
                    "yes_payout"
                ]
            )

        else:

            payout = float(
                settlement[
                    "no_payout"
                ]
            )

        # ----------------------------------------------------
        # Gross P&L per contract
        # ----------------------------------------------------

        gross_pnl = (
            payout
            - entry_price
        )

        gross_return = (
            gross_pnl
            / entry_price
        )

        net_pnl = (
            payout
            - total_cost
        )

        net_return = (
            net_pnl
            / total_cost
        )

        if payout > 0:

            status = "WIN"

        else:

            status = "LOSS"

        # ----------------------------------------------------
        # Save simulated trade
        # ----------------------------------------------------

        trade = {

            "ticker":
                ticker,

            "title":
                entry["title"],

            "entry_time":
                entry[
                    "snapshot_time"
                ],

            "side":
                side,

            "model_edge":
                float(
                    entry[
                        "best_edge"
                    ]
                ),

            "entry_price":
                entry_price,

            "entry_fee":
                entry_fee,

            "total_cost":
                total_cost,

            "net_model_edge":
                net_model_edge,

            "status":
                status,

            "payout":
                payout,

            "gross_pnl":
                gross_pnl,

            "gross_return":
                gross_return,

            "net_pnl":
                net_pnl,

            "net_return":
                net_return
        }

        trades.append(
            trade
        )

    return trades


# ============================================================
# DISPLAY
# ============================================================

def print_trade_results(
    trades
):

    print()
    print("=" * 90)
    print(
        "SIMULATED TRADES"
    )
    print("=" * 90)

    if not trades:

        print(
            "No trades met the "
            "minimum edge threshold."
        )

        return

    settled_trades = []

    for trade in trades:

        print()
        print(
            trade["title"]
        )

        print(
            f"  Ticker:     "
            f"{trade['ticker']}"
        )

        print(
            f"  Entry time: "
            f"{trade['entry_time']}"
        )

        print(
            f"  Side:       "
            f"{trade['side']}"
        )

        print(
            f"  Entry:      "
            f"${trade['entry_price']:.2f}"
        )

        print(
            f"  Fee:        "
            f"${trade['entry_fee']:.2f}"
        )

        print(
            f"  Total cost: "
            f"${trade['total_cost']:.2f}"
        )

        print(
            f"  Gross edge: "
            f"{trade['model_edge']:+.2%}"
        )

        print(
            f"  Net edge:   "
            f"{trade['net_model_edge']:+.2%}"
        )

        print(
            f"  Status:     "
            f"{trade['status']}"
        )

        if trade[
            "status"
        ] != "PENDING":

            print(
                f"  Payout:     "
                f"${trade['payout']:.2f}"
            )

            print(
                f"  Gross P&L:  "
                f"${trade['gross_pnl']:+.2f}"
            )

            print(
                f"  Gross return:"
                f" {trade['gross_return']:+.2%}"
            )

            print(
                f"  Net P&L:    "
                f"${trade['net_pnl']:+.2f}"
            )

            print(
                f"  Net return: "
                f"{trade['net_return']:+.2%}"
            )

            settled_trades.append(
                trade
            )

        print("-" * 90)

    # --------------------------------------------------------
    # Portfolio summary
    # --------------------------------------------------------

    print()

    if not settled_trades:

        print(
            "No simulated trades "
            "have settled yet."
        )

        return

    total_cost = sum(
        trade["total_cost"]
        for trade in settled_trades
    )

    gross_pnl = sum(
        trade["gross_pnl"]
        for trade in settled_trades
    )

    net_pnl = sum(
        trade["net_pnl"]
        for trade in settled_trades
    )

    wins = sum(
        trade["status"] == "WIN"
        for trade in settled_trades
    )

    losses = sum(
        trade["status"] == "LOSS"
        for trade in settled_trades
    )

    net_return = (
        net_pnl
        / total_cost
    )

    print("=" * 90)
    print(
        "SUMMARY"
    )
    print("=" * 90)

    print(
        f"Settled trades: "
        f"{len(settled_trades)}"
    )

    print(
        f"Wins:           "
        f"{wins}"
    )

    print(
        f"Losses:         "
        f"{losses}"
    )

    print(
        f"Total cost:     "
        f"${total_cost:.2f}"
    )

    print(
        f"Gross P&L:      "
        f"${gross_pnl:+.2f}"
    )

    print(
        f"Net P&L:        "
        f"${net_pnl:+.2f}"
    )

    print(
        f"Net return:     "
        f"{net_return:+.2%}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    trades = simulate_trades(
        target_date=
            "2026-09-28",

        minimum_edge=
            0.03
    )

    print_trade_results(
        trades
    )