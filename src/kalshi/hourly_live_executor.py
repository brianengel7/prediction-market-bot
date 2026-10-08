import fcntl
import math

from contextlib import contextmanager
from pathlib import Path
from decimal import Decimal

from src.database.db import get_live_orders

from src.kalshi.fees import calculate_taker_fee

from src.kalshi.live_executor import (
    calculate_position_size,
    get_live_fee_multiplier,
    live_trading_enabled,
    build_client_order_id,
    record_live_order,
    ensure_not_already_submitted,
    reconcile_order,
)

from src.kalshi.trading_client import (
    place_order,
    OrderSubmissionUnknownError,
)


STRATEGIES = {
    45: "HOURLY_INDEX_45M",
    5: "HOURLY_INDEX_5M",
}


@contextmanager
def execution_lock():
    """
    Serialize both traders within one Codespace.
    """
    path = Path(
        "/tmp/prediction_bot_hourly_execution.lock"
    )

    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)

        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def existing_event_order(event):
    """
    Prevent a second live order on the same
    hourly event, including another strike.
    """
    orders = get_live_orders()

    if orders is None:
        raise RuntimeError(
            "Cannot read live order ledger"
        )

    if orders.empty:
        return False

    if "ticker" not in orders.columns:
        raise RuntimeError(
            "Live-order ledger missing ticker"
        )

    # Fail closed even for unresolved orders.
    return (
        orders["ticker"]
        .astype(str)
        .str.startswith(event + "-")
        .any()
    )


def quote_economics(
    ticker,
    price,
    fair,
    quantity,
):
    multiplier = float(
        get_live_fee_multiplier(ticker)
    )

    fee = float(
        calculate_taker_fee(
            price=price,
            contracts=quantity,
            multiplier=multiplier,
        )
    )

    unit_fee = fee / quantity
    cost = price + unit_fee

    edge_after_buffer = (
        fair - cost - 0.01
    )

    roi_after_buffer = (
        edge_after_buffer / (cost + 0.01)
    )

    return (
        fee,
        cost,
        edge_after_buffer,
        roi_after_buffer,
    )


def size_and_check(
    trade,
    displayed_quantity,
):
    """
    Use the SAME position sizing as the
    existing daily trader.
    """
    sizing = calculate_position_size(trade)

    count = int(
        sizing["contracts"]
    )

    if (
        count < 1
        or displayed_quantity + 1e-9 < count
    ):
        raise RuntimeError(
            "Insufficient best-ask depth: "
            f"{displayed_quantity} < {count}"
        )

    fee, cost, edge, roi = quote_economics(
        trade["ticker"],
        trade["entry_price"],
        trade["adjusted_probability"],
        count,
    )

    if (
        edge < 0.025 - 1e-9
        or roi < 0.05 - 1e-9
    ):
        raise RuntimeError(
            "Position fails buffered economics: "
            f"edge={edge:.2%}, ROI={roi:.2%}"
        )

    total_cost = Decimal(
        str(
            count * trade["entry_price"]
            + fee
        )
    )

    if total_cost > sizing["trade_budget"]:
        raise RuntimeError(
            "Position exceeds environment-based "
            "cash budget"
        )

    return count, fee, edge, roi, sizing


def execute_hourly_trade(
    trade,
    horizon,
    displayed_quantity,
):
    """
    Submit a real hourly order.

    - Global live trading lock
    - Independent 5% ROI requirement
    - Existing cash-based sizing
    - Event duplicate protection
    - Immediate-or-cancel orders
    - Durable execution intent
    - Fill reconciliation
    """
    if horizon not in STRATEGIES:
        raise ValueError(
            "Unsupported hourly horizon"
        )

    if not live_trading_enabled():
        raise RuntimeError(
            "KALSHI_LIVE_TRADING_ENABLED "
            "is not enabled"
        )

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    close = datetime.fromisoformat(
        trade["market_close_time"]
    )

    decision = datetime.fromisoformat(
        trade["decision_time"]
    )

    if (
        close.tzinfo is None
        or decision.tzinfo is None
    ):
        raise RuntimeError(
            "Hourly times must have UTC offsets"
        )

    age = (
        now - decision
    ).total_seconds()

    if (
        age < 30
        or age > 190
        or (close - now).total_seconds() <= 15
    ):
        raise RuntimeError(
            "Hourly decision too old/new "
            "or too close to settlement"
        )

    if (
        trade["side"] not in ("YES", "NO")
        or not trade["ticker"].startswith(
            "KXTEMPNYCHS-"
        )
        or not trade["ticker"].startswith(
            trade["event"] + "-"
        )
    ):
        raise RuntimeError(
            "Invalid hourly event, ticker, or side"
        )

    price = float(
        trade["entry_price"]
    )

    fair = float(
        trade["adjusted_probability"]
    )

    if not (
        math.isfinite(price)
        and math.isfinite(fair)
        and 0.05 <= price <= 0.95
        and 0 <= fair <= 1
    ):
        raise RuntimeError(
            "Invalid hourly price or "
            "model probability"
        )

    with execution_lock():

        if existing_event_order(
            trade["event"]
        ):
            raise RuntimeError(
                "Another live order exists "
                "for this hourly event"
            )

        count, fee, edge, roi, sizing = (
            size_and_check(
                trade,
                displayed_quantity,
            )
        )

        strategy = STRATEGIES[horizon]

        client_id = build_client_order_id(
            strategy=strategy,
            target_date=trade["target_date"],
            decision_time=trade["decision_time"],
            ticker=trade["ticker"],
            side=trade["side"],
        )

        ensure_not_already_submitted(
            client_id
        )

        # Save intent before sending order.
        record_live_order(
            strategy=strategy,
            trade=trade,
            contracts=count,
            client_order_id=client_id,
            order_status="SUBMITTING",
        )

        try:
            result = place_order(
                ticker=trade["ticker"],
                side=trade["side"],
                entry_price=trade["entry_price"],
                contracts=count,
                client_order_id=client_id,
                time_in_force="immediate_or_cancel",
                expiration_time=None,
            )

        except OrderSubmissionUnknownError as exc:

            record_live_order(
                strategy=strategy,
                trade=trade,
                contracts=count,
                client_order_id=client_id,
                order_status="SUBMISSION_UNKNOWN",
                error_message=str(exc),
            )

            raise RuntimeError(
                f"UNKNOWN ORDER STATUS {client_id}. "
                "HALT and reconcile; do not retry."
            ) from exc

        except Exception as exc:

            record_live_order(
                strategy=strategy,
                trade=trade,
                contracts=count,
                client_order_id=client_id,
                order_status="REJECTED_OR_ERROR",
                error_message=str(exc),
            )

            raise

        filled = float(
            result.get("fill_count") or 0
        )

        remaining = float(
            result.get("remaining_count") or 0
        )

        status = (
            "FILLED"
            if filled >= count and remaining == 0
            else "PARTIAL"
            if filled > 0
            else "UNFILLED"
        )

        if filled:
            try:
                reconciled = reconcile_order(
                    result,
                    trade["side"],
                )

                if (
                    float(reconciled["fill_count"])
                    + 1e-9
                    < filled
                ):
                    status += "_UNRECONCILED"

                else:
                    result = {
                        **result,
                        **{
                            k: reconciled[k]
                            for k in (
                                "fill_count",
                                "average_fill_price",
                                "average_fee_paid",
                            )
                        }
                    }

            except Exception as exc:
                status += "_UNRECONCILED"

                print(
                    "FILL RECONCILIATION WARNING:",
                    exc,
                    flush=True,
                )

        record_live_order(
            strategy=strategy,
            trade=trade,
            contracts=count,
            client_order_id=client_id,
            order_status=status,
            result=result,
        )

        print(
            f"LIVE {horizon}m "
            f"{trade['ticker']} "
            f"{trade['side']} "
            f"contracts={count} "
            f"budget=${sizing['trade_budget']} "
            f"order={result.get('order_id')} "
            f"status={status}",
            flush=True,
        )

        if "UNRECONCILED" in status:
            raise RuntimeError(
                "Fill reconciliation incomplete; "
                "inspect ledger before continuing"
            )

        return result
