"""Verify delayed Kalshi order visibility and fail-closed behavior."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.backtest import weather_order_reconciliation as rec


class DelayedVisibilityTests(unittest.TestCase):

    def test_order_appears_on_seventh_attempt(self):
        result = rec.Execution(
            "WX-1", "client-1", "order-1",
            25, 25, 0, "executed",
        )

        responses = [
            rec.ReconciliationError("Order not visible")
            for _ in range(6)
        ] + [result]

        with (
            patch.object(
                rec, "reconcile_once",
                new_callable=AsyncMock,
                side_effect=responses,
            ) as lookup,
            patch.object(
                rec.asyncio, "sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            verified = asyncio.run(
                rec.reconcile_order(
                    object(),
                    ticker="WX-1",
                    client_id="client-1",
                    requested=25,
                )
            )

        self.assertTrue(verified.full)
        self.assertEqual(lookup.await_count, 7)
        self.assertEqual(sleep.await_count, 6)

    def test_twelve_failures_still_block(self):
        with (
            patch.object(
                rec, "reconcile_once",
                new_callable=AsyncMock,
                side_effect=rec.ReconciliationError(
                    "Still not visible"
                ),
            ) as lookup,
            patch.object(
                rec.asyncio, "sleep",
                new_callable=AsyncMock,
            ) as sleep,
        ):
            with self.assertRaisesRegex(
                rec.ReconciliationError,
                "after 12 attempts",
            ):
                asyncio.run(
                    rec.reconcile_order(
                        object(),
                        ticker="WX-1",
                        client_id="client-1",
                        requested=25,
                    )
                )

        self.assertEqual(lookup.await_count, 12)
        self.assertEqual(sleep.await_count, 11)


if __name__ == "__main__":
    unittest.main()
