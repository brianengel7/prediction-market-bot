import os
import uuid

import pandas as pd

from src.kalshi.trading_client import (
    place_order,
    get_order_fills,
    summarize_order_fills
)

from src.database.db import (
    get_live_orders,
    save_live_order
)


MINIMUM_NET_EDGE = 0.025
MAX_CONTRACTS = 1
MAX_DECISION_AGE_MINUTES = 15


def live_trading_enabled():

    value = (
        os.getenv(
            "KALSHI_LIVE_TRADING_ENABLED",
            ""
        )
        .strip()
        .lower()
    )

    return value in (
        "1",
        "true",
        "yes",
        "on"
    )


def build_client_order_id(
    strategy,
    target_date,
    ticker,
    side
):
    """
    Deterministic order ID.

    If the same strategy/date/ticker/side is accidentally
    submitted twice, Kalshi sees the same client_order_id
    instead of creating a second independent order.
    """

    key = (
        f"prediction-market-bot:"
        f"{strategy}:"
        f"{target_date}:"
        f"{ticker}:"
        f"{side}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            key
        )
    )

def ensure_not_already_submitted(
    client_order_id
):

    orders = get_live_orders()

    if orders.empty:
        return

    duplicate = orders[
        orders["client_order_id"]
        ==
        client_order_id
    ]

    if not duplicate.empty:

        existing = duplicate.iloc[-1]

        raise RuntimeError(
            "LIVE ORDER BLOCKED: "
            "this strategy decision already "
            "has a live-order record.\n"
            f"Client order ID: "
            f"{client_order_id}\n"
            f"Existing status: "
            f"{existing['order_status']}"
        )

def record_live_order(
    strategy,
    trade,
    contracts,
    client_order_id,
    order_status,
    result=None,
    error_message=None
):

    if result is None:
        result = {}

    save_live_order({

        "strategy":
            strategy,

        "target_date":
            trade["target_date"],

        "ticker":
            trade["ticker"],

        "side":
            trade["side"],

        "decision_time":
            trade["decision_time"],

        "entry_price":
            trade["entry_price"],

        "net_edge":
            trade["net_edge"],

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

        # Populated later from /portfolio/fills
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


def validate_live_trade(
    trade,
    contracts=1
):

    required_fields = (
        "target_date",
        "ticker",
        "side",
        "decision_time",
        "entry_price",
        "net_edge"
    )

    for field in required_fields:

        if field not in trade:

            raise RuntimeError(
                f"Live trade rejected: "
                f"missing {field}."
            )

    side = (
        str(
            trade["side"]
        )
        .strip()
        .upper()
    )

    if side not in (
        "YES",
        "NO"
    ):

        raise RuntimeError(
            f"Live trade rejected: "
            f"invalid side {side}."
        )

    contracts = float(
        contracts
    )

    if contracts <= 0:

        raise RuntimeError(
            "Live trade rejected: "
            "contracts must be positive."
        )

    if contracts > MAX_CONTRACTS:

        raise RuntimeError(
            f"Live trade rejected: "
            f"{contracts} contracts exceeds "
            f"maximum of {MAX_CONTRACTS}."
        )

    entry_price = float(
        trade[
            "entry_price"
        ]
    )

    if not (
        0.0
        <
        entry_price
        <
        1.0
    ):

        raise RuntimeError(
            f"Live trade rejected: "
            f"invalid entry price "
            f"{entry_price:.4f}."
        )

    net_edge = float(
        trade[
            "net_edge"
        ]
    )

    minimum_edge = float(
        trade.get(
            "minimum_edge",
            MINIMUM_NET_EDGE
        )
    )

    required_edge = max(
        MINIMUM_NET_EDGE,
        minimum_edge
    )

    if (
        net_edge
        <
        required_edge
    ):

        raise RuntimeError(
            f"Live trade rejected: "
            f"net edge {net_edge:.2%} "
            f"is below required "
            f"{required_edge:.2%}."
        )

    decision_time = pd.Timestamp(
        trade[
            "decision_time"
        ]
    )

    if decision_time.tzinfo is None:

        decision_time = (
            decision_time.tz_localize(
                "UTC"
            )
        )

    else:

        decision_time = (
            decision_time.tz_convert(
                "UTC"
            )
        )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    decision_age_minutes = (
        (
            now
            -
            decision_time
        )
        .total_seconds()
        /
        60.0
    )

    if decision_age_minutes < -1:

        raise RuntimeError(
            "Live trade rejected: "
            "decision timestamp is "
            "in the future."
        )

    if (
        decision_age_minutes
        >
        MAX_DECISION_AGE_MINUTES
    ):

        raise RuntimeError(
            f"Live trade rejected: "
            f"decision is "
            f"{decision_age_minutes:.1f} "
            f"minutes old."
        )

    return {
        "target_date":
            str(
                trade[
                    "target_date"
                ]
            ),

        "ticker":
            trade[
                "ticker"
            ],

        "side":
            side,

        "entry_price":
            entry_price,

        "net_edge":
            net_edge,

        "contracts":
            contracts
    }

def reconcile_order(
    order_result,
    outcome_side
):
    """
    Retrieve individual Kalshi fills for an accepted order
    and calculate actual outcome-side execution economics.
    """

    order_id = (
        order_result.get(
            "order_id"
        )
    )

    if not order_id:

        raise RuntimeError(
            "Cannot reconcile order: "
            "Kalshi order_id is missing."
        )

    fills = (
        get_order_fills(
            order_id=
                order_id
        )
    )

    summary = (
        summarize_order_fills(
            fills=
                fills,

            outcome_side=
                outcome_side
        )
    )

    return {
        "order_id":
            order_id,

        "client_order_id":
            order_result.get(
                "client_order_id"
            ),

        "fill_count":
            summary[
                "fill_count"
            ],

        "remaining_count":
            order_result.get(
                "remaining_count"
            ),

        "average_fill_price":
            summary[
                "average_fill_price"
            ],

        "average_fee_paid":
            summary[
                "average_fee_paid"
            ],

        "total_fee":
            summary[
                "total_fee"
            ],

        "fills":
            fills
    }


def execute_live_trade(
    trade,
    strategy,
    contracts=1
):
    """
    Submit a REAL Kalshi order.

    Requires:
        1. --live from the strategy runner
        2. KALSHI_LIVE_TRADING_ENABLED=true
    """

    # --------------------------------------------------------
    # 1. Global live-money lock
    # --------------------------------------------------------

    if not live_trading_enabled():

        raise RuntimeError(
            "LIVE TRADING DISABLED. "
            "Set KALSHI_LIVE_TRADING_ENABLED=true "
            "to permit real orders."
        )

    # --------------------------------------------------------
    # 2. Validate strategy decision
    # --------------------------------------------------------

    validated = validate_live_trade(
        trade=trade,
        contracts=contracts
    )

    # --------------------------------------------------------
    # 3. Deterministic order ID
    # --------------------------------------------------------

    client_order_id = (
        build_client_order_id(
            strategy=strategy,

            target_date=
                validated["target_date"],

            ticker=
                validated["ticker"],

            side=
                validated["side"]
        )
    )

    # --------------------------------------------------------
    # 4. Local duplicate protection
    # --------------------------------------------------------

    ensure_not_already_submitted(
        client_order_id
    )

    print()
    print("=" * 72)
    print(
        "SUBMITTING REAL KALSHI ORDER"
    )
    print("=" * 72)

    print(
        f"Strategy:    {strategy}"
    )

    print(
        f"Target date: "
        f"{validated['target_date']}"
    )

    print(
        f"Ticker:      "
        f"{validated['ticker']}"
    )

    print(
        f"Side:        "
        f"{validated['side']}"
    )

    print(
        f"Contracts:   "
        f"{validated['contracts']}"
    )

    print(
        f"Limit price: "
        f"{validated['entry_price']:.2%}"
    )

    print(
        f"Net edge:    "
        f"{validated['net_edge']:+.2%}"
    )

    print(
        f"Client ID:   "
        f"{client_order_id}"
    )

    # --------------------------------------------------------
    # 5. Record intent BEFORE network submission
    # --------------------------------------------------------

    record_live_order(
        strategy=
            strategy,

        trade=
            trade,

        contracts=
            contracts,

        client_order_id=
            client_order_id,

        order_status=
            "SUBMITTING"
    )

    # --------------------------------------------------------
    # 6. Real API submission
    # --------------------------------------------------------

    try:

        result = place_order(
            ticker=
                validated["ticker"],

            side=
                validated["side"],

            entry_price=
                validated[
                    "entry_price"
                ],

            contracts=
                validated[
                    "contracts"
                ],

            client_order_id=
                client_order_id
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

        # ----------------------------------------------------
        # Determine immediate order status
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Reconcile actual fills
        # ----------------------------------------------------

        reconciled_result = result.copy()

        if fill_count > 0:

            try:

                reconciliation = (
                    reconcile_order(
                        order_result=
                            result,

                        outcome_side=
                            validated[
                                "side"
                            ]
                    )
                )

                if (
                    reconciliation[
                        "fill_count"
                    ]
                    >
                    0
                ):

                    reconciled_result[
                        "fill_count"
                    ] = (
                        reconciliation[
                            "fill_count"
                        ]
                    )

                    reconciled_result[
                        "average_fill_price"
                    ] = (
                        reconciliation[
                            "average_fill_price"
                        ]
                    )

                    reconciled_result[
                        "average_fee_paid"
                    ] = (
                        reconciliation[
                            "average_fee_paid"
                        ]
                    )

                    print()
                    print(
                        "FILL RECONCILED"
                    )

                    print(
                        f"Actual fill price: "
                        f"{reconciliation['average_fill_price']:.2%}"
                    )

                    print(
                        f"Average fee/contract: "
                        f"${reconciliation['average_fee_paid']:.4f}"
                    )

                    print(
                        f"Total fee: "
                        f"${reconciliation['total_fee']:.4f}"
                    )

                else:

                    status = (
                        f"{status}_"
                        f"UNRECONCILED"
                    )

                    print()
                    print(
                        "WARNING: Kalshi reported "
                        "a fill but /portfolio/fills "
                        "returned no matching fill yet."
                    )

            except Exception as fill_error:

                status = (
                    f"{status}_"
                    f"UNRECONCILED"
                )

                print()
                print(
                    "WARNING: fill reconciliation "
                    "failed."
                )

                print(
                    f"{type(fill_error).__name__}: "
                    f"{fill_error}"
                )

        # ----------------------------------------------------
        # Save final execution state
        # ----------------------------------------------------

        record_live_order(
            strategy=
                strategy,

            trade=
                trade,

            contracts=
                contracts,

            client_order_id=
                client_order_id,

            order_status=
                status,

            result=
                reconciled_result
        )

        print()
        print("=" * 72)
        print(
            "LIVE ORDER RESULT"
        )
        print("=" * 72)

        print(
            f"Status:      "
            f"{status}"
        )

        print(
            f"Order ID:    "
            f"{result.get('order_id')}"
        )

        print(
            f"Filled:      "
            f"{fill_count}"
        )

        print(
            f"Remaining:   "
            f"{remaining_count}"
        )

        return (
            reconciled_result
        )
    except Exception as error:

        record_live_order(
            strategy=
                strategy,

            trade=
                trade,

            contracts=
                contracts,

            client_order_id=
                client_order_id,

            order_status=
                "ERROR",

            error_message=
                f"{type(error).__name__}: "
                f"{error}"
        )

        raise