import argparse

from src.database.db import (
    get_live_orders,
    update_live_order_settlement
)

from src.kalshi.trading_client import (
    get_market_settlement
)


# ============================================================
# SETTLE LIVE ORDERS
# ============================================================

def settle_live_orders(
    save=False
):

    orders = (
        get_live_orders()
    )

    if orders.empty:

        print(
            "No live orders found."
        )

        return

    unsettled = (
        orders[
            orders[
                "settled"
            ]
            ==
            0
        ]
        .copy()
    )

    if unsettled.empty:

        print(
            "No unsettled live orders."
        )

        return

    print()
    print("=" * 80)
    print(
        "LIVE ORDER SETTLEMENT CHECK"
    )
    print("=" * 80)

    settled_count = 0
    pending_count = 0

    for _, order in unsettled.iterrows():

        # ----------------------------------------------------
        # Only actual fills have financial exposure.
        # ----------------------------------------------------

        fill_count = order.get(
            "fill_count"
        )

        if (
            fill_count is None
            or
            float(fill_count) <= 0
        ):

            print(
                f"{order['ticker']}: "
                f"no filled contracts - skipping."
            )

            continue

        # ----------------------------------------------------
        # Require reconciled execution economics.
        # ----------------------------------------------------

        average_fill_price = (
            order.get(
                "average_fill_price"
            )
        )

        average_fee_paid = (
            order.get(
                "average_fee_paid"
            )
        )

        if (
            average_fill_price is None
            or
            average_fee_paid is None
        ):

            print(
                f"{order['ticker']}: "
                f"fill exists but execution "
                f"economics are unreconciled."
            )

            continue

        # ----------------------------------------------------
        # Query Kalshi directly for settlement.
        # No candlestick/backfill dependency.
        # ----------------------------------------------------

        settlement = (
            get_market_settlement(
                ticker=
                    order[
                        "ticker"
                    ]
            )
        )

        market_result = (
            settlement.get(
                "result"
            )
        )

        if market_result not in (
            "YES",
            "NO"
        ):

            print(
                f"{order['ticker']}: "
                f"PENDING "
                f"(status="
                f"{settlement.get('status')})"
            )

            pending_count += 1

            continue

        side = (
            str(
                order[
                    "side"
                ]
            )
            .strip()
            .upper()
        )

        won = (
            side
            ==
            market_result
        )

        filled_contracts = float(
            fill_count
        )

        execution_price = float(
            average_fill_price
        )

        fee_per_contract = float(
            average_fee_paid
        )

        # ----------------------------------------------------
        # Actual capital spent
        # ----------------------------------------------------

        total_entry_cost = (
            filled_contracts
            *
            execution_price
        )

        total_fees = (
            filled_contracts
            *
            fee_per_contract
        )

        total_cost = (
            total_entry_cost
            +
            total_fees
        )

        # ----------------------------------------------------
        # Binary contract pays $1 if correct.
        # ----------------------------------------------------

        payout = (
            filled_contracts
            if won
            else 0.0
        )

        net_pnl = (
            payout
            -
            total_cost
        )

        print()
        print(
            f"{order['strategy']} | "
            f"{order['ticker']}"
        )

        print(
            f"Side:         "
            f"{side}"
        )

        print(
            f"Result:       "
            f"{market_result}"
        )

        print(
            f"Contracts:    "
            f"{filled_contracts:g}"
        )

        print(
            f"Fill price:   "
            f"${execution_price:.4f}"
        )

        print(
            f"Fees:         "
            f"${total_fees:.4f}"
        )

        print(
            f"Total cost:   "
            f"${total_cost:.4f}"
        )

        print(
            f"Outcome:      "
            f"{'WIN' if won else 'LOSS'}"
        )

        print(
            f"Payout:       "
            f"${payout:.4f}"
        )

        print(
            f"Net P&L:      "
            f"${net_pnl:+.4f}"
        )

        if save:

            update_live_order_settlement(
                order_id=
                    order[
                        "id"
                    ],

                won=
                    won,

                payout=
                    payout,

                net_pnl=
                    net_pnl
            )

            print(
                "Settlement saved."
            )

        else:

            print(
                "DRY RUN - settlement "
                "not saved."
            )

        settled_count += 1

    print()
    print("=" * 80)

    print(
        f"Settled: {settled_count}"
    )

    print(
        f"Pending: {pending_count}"
    )

    if not save:

        print(
            "DRY RUN ONLY - database "
            "was NOT updated."
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Save confirmed settlements "
            "and realized P&L."
        )
    )

    args = (
        parser.parse_args()
    )

    settle_live_orders(
        save=
            args.save
    )


if __name__ == "__main__":

    main()