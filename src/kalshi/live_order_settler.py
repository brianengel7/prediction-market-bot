import argparse
import math
from src.database.db import (
    get_live_orders,
    update_live_order_reconciliation,
    update_live_order_settlement
)

from src.kalshi.trading_client import (
    get_order_fills,
    summarize_order_fills,
    get_market_settlement
)
import pandas as pd


# ============================================================
# SETTLE LIVE ORDERS
# ============================================================
def is_missing_number(
    value
):
    """
    Treat None, NaN, and infinite values
    as missing/invalid numeric data.
    """

    if value is None:
        return True

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return True

    return not math.isfinite(
        value
    )

def retry_fill_reconciliation(
    order,
    save=False
):
    """
    Retry fill reconciliation for an order whose
    execution economics were not captured successfully.
    """

    status = (
        str(
            order.get(
                "order_status",
                ""
            )
        )
        .strip()
        .upper()
    )

    fill_count = (
        order.get(
            "fill_count"
        )
    )

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

    # --------------------------------------------------------
    # Determine whether this row actually needs reconciliation.
    # --------------------------------------------------------

    known_fill = (
        not is_missing_number(
            fill_count
        )
        and
        float(fill_count) > 0
    )

    missing_economics = (
        is_missing_number(
            average_fill_price
        )
        or
        is_missing_number(
            average_fee_paid
        )
    )

    explicitly_unreconciled = (
        "UNRECONCILED"
        in
        status
    )

    if not (
        explicitly_unreconciled
        or
        (
            known_fill
            and
            missing_economics
        )
    ):

        return None

    # --------------------------------------------------------
    # We need the Kalshi order ID to query fills.
    # --------------------------------------------------------

    kalshi_order_id = (
        order.get(
            "kalshi_order_id"
        )
    )

    if (
        kalshi_order_id is None
        or
        str(
            kalshi_order_id
        )
        .strip()
        .lower()
        in (
            "",
            "nan",
            "none"
        )
    ):

        print(
            f"{order['ticker']}: "
            f"cannot retry reconciliation - "
            f"Kalshi order ID is missing."
        )

        return None

    print(
        f"{order['ticker']}: "
        f"retrying fill reconciliation..."
    )

    try:

        fills = (
            get_order_fills(
                order_id=
                    kalshi_order_id
            )
        )

        summary = (
            summarize_order_fills(
                fills=
                    fills,

                outcome_side=
                    order[
                        "side"
                    ]
            )
        )

    except Exception as error:

        print(
            f"{order['ticker']}: "
            f"reconciliation retry failed - "
            f"{type(error).__name__}: "
            f"{error}"
        )

        return None

    reconciled_count = (
        summary[
            "fill_count"
        ]
    )

    if (
        reconciled_count
        <=
        0
    ):

        print(
            f"{order['ticker']}: "
            f"Kalshi returned no fills yet."
        )

        return None

    reconciled_price = (
        summary[
            "average_fill_price"
        ]
    )

    reconciled_fee = (
        summary[
            "average_fee_paid"
        ]
    )

    if (
        is_missing_number(
            reconciled_price
        )
        or
        is_missing_number(
            reconciled_fee
        )
    ):

        print(
            f"{order['ticker']}: "
            f"fill data is still incomplete."
        )

        return None

    contracts = float(
        order[
            "contracts"
        ]
    )

    if (
        reconciled_count
        >=
        contracts
    ):

        repaired_status = (
            "FILLED"
        )

    else:

        repaired_status = (
            "PARTIAL"
        )

    print(
        f"{order['ticker']}: "
        f"reconciliation recovered "
        f"{reconciled_count:g} fill(s) "
        f"at ${reconciled_price:.4f} "
        f"with ${reconciled_fee:.4f} "
        f"fee/contract."
    )

    if save:

        update_live_order_reconciliation(
            order_id=
                order[
                    "id"
                ],

            fill_count=
                reconciled_count,

            average_fill_price=
                reconciled_price,

            average_fee_paid=
                reconciled_fee,

            order_status=
                repaired_status
        )

        print(
            f"{order['ticker']}: "
            f"reconciliation saved."
        )

    return {
        "fill_count":
            float(
                reconciled_count
            ),

        "average_fill_price":
            float(
                reconciled_price
            ),

        "average_fee_paid":
            float(
                reconciled_fee
            ),

        "order_status":
            repaired_status
    }


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

        reconciliation = (
            retry_fill_reconciliation(
                order=
                    order,

                save=
                    save
            )
        )

        if reconciliation is not None:

            fill_count = (
                reconciliation[
                    "fill_count"
                ]
            )

            average_fill_price = (
                reconciliation[
                    "average_fill_price"
                ]
            )

            average_fee_paid = (
                reconciliation[
                    "average_fee_paid"
                ]
            )

        else:

            fill_count = (
                order.get(
                    "fill_count"
                )
            )

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
        # ----------------------------------------------------
        # Only actual fills have financial exposure.
        # ----------------------------------------------------

        if pd.isna(fill_count) or float(fill_count) <= 0:
            print(
                f"{order['ticker']}: "
                f"no filled contracts - skipping."
            )
            pending_count += 1
            continue

        if pd.isna(average_fill_price) or pd.isna(average_fee_paid):
            print(
                f"{order['ticker']}: "
                f"fill exists but execution "
                f"economics are unreconciled."
            )
            pending_count += 1
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

        market_status = str(settlement.get("status") or "").strip().lower()
        market_result = str(settlement.get("result") or "").strip().upper()

        if market_status != "finalized" or market_result not in ("YES", "NO"):
            print(
                f"{order['ticker']}: PENDING "
                f"(status={market_status}, result={market_result})"
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