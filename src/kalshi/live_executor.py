import os
import uuid
import math

import pandas as pd

from src.database.db import (
    get_live_orders,
    save_live_order
)
from decimal import Decimal, ROUND_CEILING

from src.kalshi.client import get_event, get_series

from src.kalshi.trading_client import (
    place_order,
    get_order_fills,
    summarize_order_fills,
    find_order_by_client_order_id,
    OrderSubmissionUnknownError
)


MAX_ORDER_LIFETIME_MINUTES = 30
MINIMUM_NET_EDGE = 0.025
MAX_CONTRACTS = 1
MAX_LIVE_CONTRACTS_PER_TARGET_DATE = 1
MAX_DECISION_AGE_MINUTES = 15
MINIMUM_EXPECTED_RETURN_ON_RISK = 0.10


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

        existing = (
            duplicate.iloc[-1]
        )

        existing_status = (
            str(
                existing[
                    "order_status"
                ]
            )
            .strip()
            .upper()
        )

        raise RuntimeError(
            "LIVE ORDER BLOCKED: "
            "this strategy decision already "
            "has a live-order record.\n"
            f"Client order ID: "
            f"{client_order_id}\n"
            f"Existing status: "
            f"{existing_status}\n"
            "The existing order must be "
            "resolved before another "
            "submission is permitted."
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
        "adjusted_probability",
        "fee",
        "total_cost",
        "net_edge",
        "expected_return_on_risk",
        "trade_taken",
    )

    if not bool(
        trade[
            "trade_taken"
        ]
    ):

        raise RuntimeError(
            "Live trade rejected: "
            "strategy decision is PASS."
        )

    for field in required_fields:

        if field not in trade:

            raise RuntimeError(
                f"Live trade rejected: "
                f"missing {field}."
            )

    if not bool(
        trade[
            "trade_taken"
        ]
    ):

        raise RuntimeError(
            "Live trade rejected: "
            "strategy decision is PASS."
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

    adjusted_probability = float(
        trade[
            "adjusted_probability"
        ]
    )

    fee = float(
        trade[
            "fee"
        ]
    )

    total_cost = float(
        trade[
            "total_cost"
        ]
    )

    expected_return_on_risk = float(
        trade[
            "expected_return_on_risk"
        ]
    )

    net_edge = float(
        trade[
            "net_edge"
        ]
    )


    values = (
        adjusted_probability,
        fee,
        total_cost,
        net_edge,
        expected_return_on_risk,
    )

    if not all(
        math.isfinite(value)
        for value in values
    ):

        raise RuntimeError(
            "Live trade rejected: "
            "non-finite trade economics."
        )

    calculated_total_cost = (
        entry_price
        +
        fee
    )

    if not math.isclose(
        total_cost,
        calculated_total_cost,
        rel_tol=0.0,
        abs_tol=1e-9
    ):

        raise RuntimeError(
            "Live trade rejected: "
            "total cost does not equal "
            "entry price + fee."
        )


    calculated_net_edge = (
        adjusted_probability
        -
        total_cost
    )

    if not math.isclose(
        net_edge,
        calculated_net_edge,
        rel_tol=0.0,
        abs_tol=1e-9
    ):

        raise RuntimeError(
            "Live trade rejected: "
            "net edge is inconsistent "
            "with probability and cost."
        )


    calculated_expected_roi = (
        net_edge
        /
        total_cost
    )

    if not math.isclose(
        expected_return_on_risk,
        calculated_expected_roi,
        rel_tol=0.0,
        abs_tol=1e-9
    ):

        raise RuntimeError(
            "Live trade rejected: "
            "expected return/risk is inconsistent."
        )

    minimum_expected_roi = float(
        trade.get(
            "minimum_expected_return_on_risk",
            MINIMUM_EXPECTED_RETURN_ON_RISK
        )
    )

    required_expected_roi = max(
        MINIMUM_EXPECTED_RETURN_ON_RISK,
        minimum_expected_roi
    )

    if (
        expected_return_on_risk
        <
        required_expected_roi
    ):

        raise RuntimeError(
            f"Live trade rejected: "
            f"expected return/risk "
            f"{expected_return_on_risk:.2%} "
            f"is below required "
            f"{required_expected_roi:.2%}."
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

        "decision_time":
            decision_time,

        "adjusted_probability":
            adjusted_probability,

        "fee":
            fee,

        "total_cost":
            total_cost,

        "expected_return_on_risk":
            expected_return_on_risk,

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

def recover_unknown_submission(
    trade,
    strategy,
    contracts,
    client_order_id,
    save=True
):
    """
    Check Kalshi for an order whose POST response
    was lost or otherwise ambiguous.

    This function NEVER submits another order.
    """

    print()
    print(
        "CHECKING KALSHI FOR "
        "UNKNOWN SUBMISSION..."
    )

    server_order = (
        find_order_by_client_order_id(
            client_order_id=
                client_order_id,

            ticker=
                trade[
                    "ticker"
                ]
        )
    )

    if server_order is None:

        print(
            "No matching Kalshi order "
            "found yet."
        )

        return None

    kalshi_order_id = (
        server_order.get(
            "order_id"
        )
    )

    if not kalshi_order_id:

        raise RuntimeError(
            "Recovered Kalshi order is "
            "missing order_id."
        )

    print(
        f"Recovered Kalshi order: "
        f"{kalshi_order_id}"
    )

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
                trade[
                    "side"
                ]
        )
    )

    reconciled_count = float(summary["fill_count"])
    reported_count = server_order.get("fill_count_fp")
    previous_count = trade.get("fill_count")

    try:
        previous_count = float(previous_count)
    except (TypeError, ValueError):
        previous_count = 0.0

    if not math.isfinite(previous_count) or previous_count < 0:
        previous_count = 0.0

    fill_count = max(
        reconciled_count,
        float(reported_count) if reported_count is not None else 0.0,
        previous_count
    )

    fills_complete = (
        fill_count > 0
        and reconciled_count + 1e-9 >= fill_count
    )

    contracts = float(contracts)

    server_remaining = server_order.get("remaining_count_fp")
    remaining_count = (
        float(server_remaining)
        if server_remaining is not None
        else max(contracts - fill_count, 0.0)
    )

    if (
        fill_count
        >=
        contracts
    ):

        recovered_status = (
            "FILLED" if fills_complete else "FILLED_UNRECONCILED"
        )

    elif fill_count > 0:

        recovered_status = (
            "PARTIAL" if fills_complete else "PARTIAL_UNRECONCILED"
        )

    else:

        server_status = (
            str(
                server_order.get(
                    "status",
                    ""
                )
            )
            .strip()
            .lower()
        )

        if server_status == "canceled" and reported_count is not None:

            recovered_status = (
                "UNFILLED"
            )

        else:

            recovered_status = (
                "RECOVERED_"
                +
                server_status.upper()
            )

    recovered_result = {

        "order_id":
            kalshi_order_id,

        "client_order_id":
            client_order_id,

        "fill_count":
            fill_count,

        "remaining_count":
            remaining_count,

        "average_fill_price":
            summary["average_fill_price"] if fills_complete else None,

        "average_fee_paid":
            summary["average_fee_paid"] if fills_complete else None,

        "order_status":
            recovered_status
    }

    if save:
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
                recovered_status,

            result=
                recovered_result
        )

    print(
        f"Unknown submission recovered "
        f"as {recovered_status}."
    )

    return recovered_result

def enforce_global_pilot_risk_cap(
    target_date,
    requested_contracts
):
    """
    Prevent total live exposure for one target date
    from exceeding the pilot risk limit.

    Definitively UNFILLED or ERROR orders consume
    no exposure.

    Ambiguous/in-flight orders reserve their full
    requested contract count.
    """

    orders = get_live_orders()

    if orders.empty:
        return

    same_date = orders[
        orders["target_date"].astype(str)
        ==
        str(target_date)
    ]

    if same_date.empty:
        return

    reserved_contracts = 0.0

    for _, order in same_date.iterrows():

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

        remaining_count = order.get("remaining_count")
        pending_remainder = (
            not pd.isna(remaining_count)
            and float(remaining_count) > 0
        )

        # An older UNFILLED row may still have pending contracts.
        if status == "ERROR" or (
            status == "UNFILLED" and not pending_remainder
        ):
            continue

        fill_count = order.get(
            "fill_count"
        )

        if pending_remainder:
            existing_contracts = order.get("contracts")
            reserved_contracts += max(
                float(fill_count) if not pd.isna(fill_count) else 0.0,
                float(existing_contracts)
                if not pd.isna(existing_contracts)
                else MAX_CONTRACTS,
            )
            continue

        # Known fills count as actual exposure.
        if (
            not pd.isna(
                fill_count
            )
            and
            float(fill_count) > 0
        ):

            reserved_contracts += float(
                fill_count
            )

            continue

        # Anything else may still represent exposure:
        # SUBMITTING, SUBMISSION_UNKNOWN,
        # unreconciled orders, recovered orders, etc.
        existing_contracts = order.get(
            "contracts"
        )

        if pd.isna(
            existing_contracts
        ):

            existing_contracts = (
                MAX_CONTRACTS
            )

        reserved_contracts += float(
            existing_contracts
        )

    projected_contracts = (
        reserved_contracts
        +
        float(
            requested_contracts
        )
    )

    if (
        projected_contracts
        >
        MAX_LIVE_CONTRACTS_PER_TARGET_DATE
    ):

        raise RuntimeError(
            "LIVE TRADE BLOCKED BY GLOBAL "
            "PILOT RISK CAP.\n"
            f"Target date: {target_date}\n"
            f"Existing/reserved contracts: "
            f"{reserved_contracts:g}\n"
            f"Requested contracts: "
            f"{float(requested_contracts):g}\n"
            f"Maximum allowed: "
            f"{MAX_LIVE_CONTRACTS_PER_TARGET_DATE}"
        )


def enforce_trade_budget(validated):
    budget = Decimal(
        os.getenv("KALSHI_MAX_TRADE_DOLLARS", "1.00")
    )
    if not budget.is_finite() or budget <= 0:
        raise RuntimeError(
            "KALSHI_MAX_TRADE_DOLLARS must be positive and finite."
        )

    if Decimal(str(validated["contracts"])) != Decimal("1"):
        raise RuntimeError(
            "This dollar-cap check supports the one-contract pilot."
        )

    series_ticker, event_date, _ = str(
        validated["ticker"]
    ).split("-", 2)

    series_info = get_series(series_ticker)
    event_info = get_event(f"{series_ticker}-{event_date}")

    fee_type = (
        event_info.get("fee_type_override")
        or series_info.get("fee_type")
    )
    override = event_info.get("fee_multiplier_override")
    multiplier = Decimal(str(
        override
        if override is not None
        else series_info.get("fee_multiplier")
    ))

    if (
        fee_type != "quadratic"
        or not multiplier.is_finite()
        or not Decimal("0") <= multiplier <= Decimal("1")
    ):
        raise RuntimeError(
            "Live trade blocked: fee schedule needs review."
        )

    price_limit = Decimal(
        str(validated["entry_price"])
    ).quantize(
        Decimal("0.0001"), rounding=ROUND_CEILING
    )

    # Conservative allowance for fees and rounding.
    reserved_cost = (
        price_limit + Decimal("0.03")
    ).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )

    if reserved_cost > budget:
        raise RuntimeError(
            f"Live trade blocked: reserved cost ${reserved_cost:.2f} "
            f"exceeds trade cap ${budget:.2f}."
        )

    print(
        f"Trade cap: ${budget:.2f}; "
        f"reserved cost: ${reserved_cost:.2f}"
    )

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
    enforce_trade_budget(validated)

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

    enforce_global_pilot_risk_cap(
        target_date=
            validated[
                "target_date"
            ],

        requested_contracts=
            validated[
                "contracts"
            ]
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
    result = None
    try:

        expiration_time = int(
            (
                validated[
                    "decision_time"
                ]
                +
                pd.Timedelta(
                    minutes=
                        MAX_ORDER_LIFETIME_MINUTES
                )
            )
            .timestamp()
        )

        now_timestamp = int(
            pd.Timestamp.now(
                tz="UTC"
            ).timestamp()
        )

        if (
            expiration_time
            <=
            now_timestamp
        ):

            raise RuntimeError(
                "Live trade rejected: "
                "order expiration time "
                "has already passed."
            )

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
                client_order_id,

            time_in_force=
                "good_till_canceled",

            expiration_time=
                expiration_time
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

        expiration_utc = (
            pd.Timestamp(
                expiration_time,
                unit="s",
                tz="UTC"
            )
        )

        print(
            f"Order expires: "
            f"{expiration_utc}"
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

            status = "ACCEPTED_PENDING" if remaining_count > 0 else "UNFILLED"

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
        result = reconciled_result
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

    except OrderSubmissionUnknownError as error:

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
                "SUBMISSION_UNKNOWN",

            error_message=
                f"{type(error).__name__}: "
                f"{error}"
        )

        print()
        print("=" * 72)
        print(
            "ORDER SUBMISSION STATE UNKNOWN"
        )
        print("=" * 72)

        print(
            "The bot will NOT assume "
            "the order failed."
        )

        recovered = (
            recover_unknown_submission(
                trade=
                    trade,

                strategy=
                    strategy,

                contracts=
                    contracts,

                client_order_id=
                    client_order_id
            )
        )

        if recovered is None:

            raise RuntimeError(
                "Live order remains "
                "SUBMISSION_UNKNOWN. "
                "Do NOT submit another order "
                "for this decision until "
                "reconciliation succeeds."
            ) from error

        return recovered
        
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

            order_status=(
                "ERROR" if result is None else "ACCEPTED_UNRECONCILED"
            ),
            result=result,

            error_message=
                f"{type(error).__name__}: "
                f"{error}"
        )

        raise