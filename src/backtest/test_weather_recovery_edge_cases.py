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

    def test_first_rejected_second_filled_gtc_unwind(self):
        fx = self.fixture()
        result, api = fx.run_flow(
            ["reject", "fill"], ["fill"]
        )
        self.assertEqual(result, (True, False))
        self.assertEqual(len(api.calls), 3)

        rec = fx.journal.data["attempts"][-1]
        self.assertEqual(rec["status"], "UNWOUND")
        self.assertEqual(rec["unwind_ticker"], "B")
        self.assertTrue(rec["first_rejected"])
        self.assertEqual(api.positions["B"], 0)

    def test_second_rejected_gtc_rests(self):
        fx = self.fixture()
        result, api = fx.run_flow(
            ["fill", "reject"], ["resting"]
        )
        self.assertEqual(result, (True, True))
        self.assertEqual(len(api.calls), 3)

        rec = fx.journal.data["attempts"][-1]
        self.assertEqual(rec["status"], "UNWIND_RESTING")
        self.assertEqual(rec["unwind_ticker"], "A")
        self.assertTrue(rec["second_rejected"])

    def test_failed_gtc_unwind_halts(self):
        fx = self.fixture()
        result, api = fx.run_flow(
            ["fill", "unfilled"], ["reject"]
        )
        self.assertEqual(result, (True, True))
        self.assertEqual(len(api.calls), 3)

        rec = fx.journal.data["attempts"][-1]
        self.assertEqual(
            rec["status"], "EXPOSED_UNWIND_REJECTED"
        )
        self.assertEqual(api.positions["A"], 2)

    def test_uncertain_gtc_unwind_no_repeat(self):
        fx = self.fixture()
        result, api = fx.run_flow(
            ["fill", "unfilled"], ["unknown"]
        )
        self.assertEqual(result, (True, True))
        self.assertEqual(len(api.calls), 3)

        rec = fx.journal.data["attempts"][-1]
        self.assertEqual(
            rec["status"], "EXPOSED_UNWIND_UNKNOWN"
        )
        self.assertTrue(rec["unwind_client_id"])
        self.assertEqual(
            len([c for c in api.calls if c[1] == "ask"]),
            1,
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
