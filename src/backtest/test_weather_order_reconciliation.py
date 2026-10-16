import asyncio
import unittest

from decimal import Decimal as D

from src.backtest.weather_order_reconciliation import (
    Execution,
    ReconciliationError,
    reconcile_order,
    second_leg_decision,
)


def order(filled, status="executed"):
    return {
        "order_id": "o1",
        "client_order_id": "cid1",
        "ticker": "WX-1",
        "book_side": "bid",
        "fill_count_fp": f"{filled}.00",
        "remaining_count_fp": "0.00",
        "status": status,
    }


def fills(filled):
    return {
        "fills": [
            {
                "fill_id": "f1",
                "order_id": "o1",
                "ticker": "WX-1",
                "book_side": "bid",
                "count_fp": f"{filled}.00",
            }
        ] if filled else [],
        "cursor": "",
    }


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeApi:
    def __init__(self, order_data, fill_data):
        self.base = "https://kalshi.invalid/trade-api/v2"
        self.order_data = order_data
        self.fill_data = fill_data
        self.client = self

    def headers(self, method, route):
        return {}

    async def get(self, route, **kwargs):
        route = route.removeprefix(self.base)

        if route == "/portfolio/orders":
            return FakeResp({
                "orders": self.order_data,
                "cursor": "",
            })

        if route == "/portfolio/fills":
            return FakeResp(self.fill_data)

        if route == "/portfolio/orders/o1":
            return {"order": self.order_data[0]}

        raise AssertionError(route)


class Tests(unittest.TestCase):

    def test_full_fill(self):
        result = asyncio.run(
            reconcile_order(
                FakeApi([order(2)], fills(2)),
                ticker="WX-1",
                client_id="cid1",
                requested=2,
            )
        )

        self.assertTrue(result.full)

    def test_confirmed_unfilled(self):
        result = asyncio.run(
            reconcile_order(
                FakeApi(
                    [order(0, "canceled")],
                    fills(0),
                ),
                ticker="WX-1",
                client_id="cid1",
                requested=2,
            )
        )

        self.assertTrue(result.unfilled)

    def test_missing_order_is_unknown(self):
        with self.assertRaises(ReconciliationError):
            asyncio.run(
                reconcile_order(
                    FakeApi([], fills(0)),
                    ticker="WX-1",
                    client_id="cid1",
                    requested=2,
                )
            )

    def test_order_fill_mismatch_blocks(self):
        with self.assertRaises(ReconciliationError):
            asyncio.run(
                reconcile_order(
                    FakeApi(
                        [order(2)],
                        fills(1),
                    ),
                    ticker="WX-1",
                    client_id="cid1",
                    requested=2,
                )
            )

    def test_partial_fok_blocks(self):
        with self.assertRaises(ReconciliationError):
            asyncio.run(
                reconcile_order(
                    FakeApi(
                        [order(1)],
                        fills(1),
                    ),
                    ticker="WX-1",
                    client_id="cid1",
                    requested=2,
                )
            )

    def test_retry_price_bounds(self):
        first = Execution(
            "A", "a", "1",
            1, 1, 0, "executed",
        )

        second = Execution(
            "B", "b", "2",
            1, 0, 0, "canceled",
        )

        base = {
            "second_attempts": 1,
            "second_ask": D(".47"),
            "first_price": D(".45"),
            "first_fee": D(".01"),
            "fee_second": D(".01"),
            "available_depth": 1,
            "qty": 1,
        }

        self.assertEqual(
            second_leg_decision(
                first, second, **base
            ).action,
            "RETRY",
        )

        self.assertEqual(
            second_leg_decision(
                first, None, **base
            ).action,
            "HALT",
        )

        self.assertEqual(
            second_leg_decision(
                first,
                second,
                **{
                    **base,
                    "second_attempts": 2,
                },
            ).action,
            "UNWIND",
        )

        self.assertEqual(
            second_leg_decision(
                first,
                second,
                **{
                    **base,
                    "second_ask": D(".53"),
                },
            ).action,
            "UNWIND",
        )


if __name__ == "__main__":
    unittest.main()
