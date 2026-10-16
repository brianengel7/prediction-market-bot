"""Additional offline tests for live weather recovery."""
import asyncio
import unittest
from unittest.mock import patch

from src.backtest import (
    test_weather_recovery_integration as base,
)
from src.backtest import weather_multicity_live as live


class RecoveryEdgeCases(unittest.TestCase):

    def fixture(self):
        fx = base.FlowTests("test_both_fill")
        fx.setUp()
        self.addCleanup(fx.doCleanups)
        return fx

    def test_first_rejection_stops_basket(self):
        fx = self.fixture()
        result, api = fx.run_flow(["reject"])

        self.assertEqual(result, (True, False))
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(
            fx.journal.data["attempts"][-1]["status"],
            "NOT_SUBMITTED",
        )

    def test_second_rejection_one_retry(self):
        fx = self.fixture()
        result, api = fx.run_flow([
            "fill", "reject", "fill",
        ])

        self.assertEqual(result, (True, False))
        self.assertEqual(len(api.calls), 3)
        self.assertEqual(
            fx.journal.data["attempts"][-1]["status"],
            "COMPLETE",
        )

    def test_failed_unwind_halts(self):
        fx = self.fixture()
        result, api = fx.run_flow([
            "fill", "unfilled", "unfilled", "reject",
        ])

        self.assertEqual(result, (True, True))
        self.assertEqual(len(api.calls), 4)
        self.assertEqual(
            fx.journal.data["attempts"][-1]["status"],
            "EXPOSED_UNWIND_UNFILLED",
        )

    def test_uncertain_unwind_no_repeat(self):
        fx = self.fixture()
        result, api = fx.run_flow([
            "fill", "unfilled", "unfilled", "absent",
        ])

        self.assertEqual(result, (True, True))
        self.assertEqual(len(api.calls), 4)
        self.assertEqual(
            fx.journal.data["attempts"][-1]["status"],
            "UNRESOLVED_NEEDS_RECONCILIATION",
        )

    def test_uncertain_first_verified(self):
        fx = self.fixture()
        result, api = fx.run_flow([
            "unknown-fill", "fill",
        ])

        self.assertEqual(result, (True, False))
        self.assertEqual(len(api.calls), 2)
        self.assertEqual(
            fx.journal.data["attempts"][-1]["status"],
            "COMPLETE",
        )

    def test_position_mismatch_halts(self):
        fx = self.fixture()
        api = base.FakeExchange(["fill", "fill"])
        original_get = api.get

        async def broken_get(path, **kwargs):
            if (
                "/portfolio/positions" in path
                and api.orders
            ):
                return base.Response({
                    "market_positions": [],
                    "cursor": "",
                })
            return await original_get(path, **kwargs)

        api.get = broken_get

        with patch.object(
            live, "evaluate", return_value=fx.op
        ):
            result = asyncio.run(live.execute(
                api,
                fx.journal,
                fx.books,
                fx.seen,
                fx.op,
                ("A", "B"),
                fx.args,
            ))

        self.assertEqual(result, (True, True))
        self.assertEqual(len(api.calls), 2)
        self.assertEqual(
            fx.journal.data["attempts"][-1]["status"],
            "UNRESOLVED_NEEDS_RECONCILIATION",
        )


if __name__ == "__main__":
    unittest.main()
