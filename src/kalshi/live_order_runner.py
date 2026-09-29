import argparse

from src.database.db import (
    get_live_orders,
    get_v2_shadow_trades,
    get_v2_market_only_shadow_decisions,
    save_live_order
)

from src.kalshi.live_executor import (
    build_client_order_id,
    execute_live_trade
)


# ============================================================
# LOAD SAVED STRATEGY DECISION
# ============================================================

def load_saved_trade(
    strategy,
    target_date
):

    strategy = (
        str(strategy)
        .strip()
        .upper()
    )

    target_date = str(
        target_date
    )

    if strategy == "NBM":

        trades = (
            get_v2_shadow_trades()
            .copy()
        )

        if trades.empty:

            raise RuntimeError(
                "No NBM shadow trades exist."
            )

        match = (
            trades[
                trades[
                    "target_date"
                ].astype(str)
                ==
                target_date
            ]
            .copy()
        )

        if match.empty:

            raise RuntimeError(
                f"No saved NBM trade "
                f"for {target_date}."
            )

        trade = (
            match.iloc[-1]
            .to_dict()
        )

        return trade

    if strategy == "MARKET_ONLY":

        decisions = (
            get_v2_market_only_shadow_decisions()
            .copy()
        )

        if decisions.empty:

            raise RuntimeError(
                "No market-only decisions exist."
            )

        match = (
            decisions[
                decisions[
                    "target_date"
                ].astype(str)
                ==
                target_date
            ]
            .copy()
        )

        if match.empty:

            raise RuntimeError(
                f"No saved market-only decision "
                f"for {target_date}."
            )

        decision = (
            match.iloc[-1]
            .to_dict()
        )

        if int(
            decision[
                "trade_taken"
            ]
        ) != 1:

            raise RuntimeError(
                f"Market-only decision for "
                f"{target_date} was PASS. "
                f"No live order allowed."
            )

        return decision

    raise RuntimeError(
        f"Unknown strategy: "
        f"{strategy}"
    )


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

def ensure_not_already_submitted(
    strategy,
    trade
):

    client_order_id = (
        build_client_order_id(
            strategy=
                strategy,

            target_date=
                trade[
                    "target_date"
                ],

            ticker=
                trade[
                    "ticker"
                ],

            side=
                trade[
                    "side"
                ]
        )
    )

    orders = (
        get_live_orders()
    )

    if not orders.empty:

        duplicate = (
            orders[
                orders[
                    "client_order_id"
                ]
                ==
                client_order_id
            ]
        )

        if not duplicate.empty:

            existing = (
                duplicate.iloc[-1]
            )

            raise RuntimeError(
                "LIVE ORDER BLOCKED: "
                "this strategy decision already "
                "has a live-order record.\n"
                f"Client order ID: "
                f"{client_order_id}\n"
                f"Existing status: "
                f"{existing['order_status']}"
            )

    return client_order_id


# ============================================================
# AUDIT RECORD
# ============================================================

def write_live_order_record(
    strategy,
    trade,
    client_order_id,
    order_status,
    contracts=1,
    result=None,
    error_message=None
):

    if result is None:
        result = {}

    save_live_order({

        "strategy":
            strategy,

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

        "decision_time":
            trade[
                "decision_time"
            ],

        "entry_price":
            trade[
                "entry_price"
            ],

        "net_edge":
            trade[
                "net_edge"
            ],

        "contracts":
            contracts,

        "client_order_id":
            client_order_id,

        "kalshi_order_id":
            result.get(
                "order_id"
            ),

        "order_status":
            order_status,

        "fill_count":
            result.get(
                "fill_count"
            ),

        "remaining_count":
            result.get(
                "remaining_count"
            ),

        "average_fill_price":
            result.get(
                "average_fill_price"
            ),

        "average_fee_paid":
            result.get(
                "average_fee_paid"
            ),

        "error_message":
            error_message
    })


# ============================================================
# EXECUTE SAVED DECISION
# ============================================================

def execute_saved_trade(
    strategy,
    target_date,
    contracts=1
):

    strategy = (
        str(strategy)
        .strip()
        .upper()
    )

    trade = (
        load_saved_trade(
            strategy=
                strategy,

            target_date=
                target_date
        )
    )

    client_order_id = (
        ensure_not_already_submitted(
            strategy=
                strategy,

            trade=
                trade
        )
    )

    print()
    print("=" * 80)
    print(
        "LIVE ORDER INTENT"
    )
    print("=" * 80)

    print(
        f"Strategy:      "
        f"{strategy}"
    )

    print(
        f"Target date:   "
        f"{trade['target_date']}"
    )

    print(
        f"Ticker:        "
        f"{trade['ticker']}"
    )

    print(
        f"Side:          "
        f"{trade['side']}"
    )

    print(
        f"Entry price:   "
        f"{float(trade['entry_price']):.2%}"
    )

    print(
        f"Net edge:      "
        f"{float(trade['net_edge']):+.2%}"
    )

    print(
        f"Contracts:     "
        f"{contracts}"
    )

    print(
        f"Client ID:     "
        f"{client_order_id}"
    )

    # --------------------------------------------------------
    # Record BEFORE network submission.
    #
    # If Python crashes or the network response disappears,
    # the database will still show that an order submission
    # was attempted. This prevents blind duplicate retries.
    # --------------------------------------------------------

    write_live_order_record(
        strategy=
            strategy,

        trade=
            trade,

        client_order_id=
            client_order_id,

        order_status=
            "SUBMITTING",

        contracts=
            contracts
    )

    try:

        result = (
            execute_live_trade(
                trade=
                    trade,

                strategy=
                    strategy,

                contracts=
                    contracts
            )
        )

        fill_count = float(
            result.get(
                "fill_count"
            )
            or 0
        )

        remaining_count = float(
            result.get(
                "remaining_count"
            )
            or 0
        )

        if (
            fill_count
            >=
            float(contracts)
            and
            remaining_count
            == 0
        ):

            status = "FILLED"

        elif fill_count > 0:

            status = "PARTIAL"

        else:

            status = "UNFILLED"

        write_live_order_record(
            strategy=
                strategy,

            trade=
                trade,

            client_order_id=
                client_order_id,

            order_status=
                status,

            contracts=
                contracts,

            result=
                result
        )

        print()
        print("=" * 80)
        print(
            "LIVE ORDER RESULT"
        )
        print("=" * 80)

        print(
            f"Status:        "
            f"{status}"
        )

        print(
            f"Kalshi ID:     "
            f"{result.get('order_id')}"
        )

        print(
            f"Filled:        "
            f"{fill_count}"
        )

        print(
            f"Remaining:     "
            f"{remaining_count}"
        )

        return result

    except Exception as error:

        write_live_order_record(
            strategy=
                strategy,

            trade=
                trade,

            client_order_id=
                client_order_id,

            order_status=
                "ERROR",

            contracts=
                contracts,

            error_message=
                f"{type(error).__name__}: "
                f"{error}"
        )

        raise


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "NBM",
            "MARKET_ONLY"
        ]
    )

    parser.add_argument(
        "--target-date",
        required=True,
        help=(
            "Target market date, "
            "for example 2026-09-30."
        )
    )

    parser.add_argument(
        "--contracts",
        type=int,
        default=1
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually submit the real Kalshi order."
        )
    )

    args = (
        parser.parse_args()
    )

    if not args.execute:

        trade = (
            load_saved_trade(
                strategy=
                    args.strategy,

                target_date=
                    args.target_date
            )
        )

        print()
        print("=" * 80)
        print(
            "REAL ORDER READY - NOT SUBMITTED"
        )
        print("=" * 80)

        print(
            f"Strategy:    "
            f"{args.strategy}"
        )

        print(
            f"Target date: "
            f"{trade['target_date']}"
        )

        print(
            f"Ticker:      "
            f"{trade['ticker']}"
        )

        print(
            f"Side:        "
            f"{trade['side']}"
        )

        print(
            f"Entry price: "
            f"{float(trade['entry_price']):.2%}"
        )

        print(
            f"Net edge:    "
            f"{float(trade['net_edge']):+.2%}"
        )

        print()
        print(
            "No order submitted. "
            "Use --execute to permit submission."
        )

        return

    execute_saved_trade(
        strategy=
            args.strategy,

        target_date=
            args.target_date,

        contracts=
            args.contracts
    )


if __name__ == "__main__":

    main()