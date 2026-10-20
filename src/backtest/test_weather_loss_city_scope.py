"""Offline regression tests for city-scoped protective exits."""
import inspect
import unittest

from decimal import Decimal as D
from types import SimpleNamespace

from src.backtest import weather_multicity_live as live


class CityScopedExitTests(unittest.TestCase):

    def setUp(self):
        self.journal = object.__new__(live.Journal)
        self.journal.data = {
            "single_attempt_claimed": False,
            "attempts": [],
            "loss_exits": {
                "DEN:DEN-EVENT": {
                    "state": "ACTIVE",
                    "legs": ["A", "B"],
                },
            },
        }
        self.journal.write = lambda: None

        self.args = SimpleNamespace(
            one_trade=False,
            max_daily_attempts=50,
            max_city_attempts=50,
            max_city_event_pairs=100,
            city_cooldown_minutes=D(0),
        )

    def op(self, city, event, legs=("A", "B")):
        return live.Opportunity(
            city, event, legs,
            (D(".40"), D(".40")),
            D(".85"), 1, D(".81"),
        )

    def test_denver_blocked(self):
        reason = self.journal.entry_block_reason(
            self.op("DEN", "DEN-EVENT"), self.args,
        )
        self.assertIn("protective exit unfinished", reason)

    def test_nyc_remains_eligible(self):
        self.assertIsNone(
            self.journal.entry_block_reason(
                self.op("NYC", "NYC-EVENT"), self.args,
            )
        )

    def test_nyc_can_reserve_while_denver_exits(self):
        record = self.journal.reserve(
            self.op("NYC", "NYC-EVENT"),
            one_trade=False,
            args=self.args,
        )
        self.assertEqual(record["status"], "RESERVED")
        self.assertEqual(record["key"], "NYC:NYC-EVENT")

    def test_denver_other_pair_still_blocked(self):
        reason = self.journal.entry_block_reason(
            self.op("DEN", "DEN-EVENT", ("B", "C")),
            self.args,
        )
        self.assertIn("protective exit unfinished", reason)

    def test_verified_denver_exit_allows_reentry(self):
        entry = self.journal.data["loss_exits"]["DEN:DEN-EVENT"]
        entry["state"] = "DONE"
        entry["closure_verified"] = True

        self.assertIsNone(
            self.journal.entry_block_reason(
                self.op("DEN", "DEN-EVENT"), self.args,
            )
        )

    def test_unverified_denver_exit_blocks(self):
        entry = self.journal.data["loss_exits"]["DEN:DEN-EVENT"]
        entry["state"] = "DONE"

        self.assertIsNotNone(
            self.journal.entry_block_reason(
                self.op("DEN", "DEN-EVENT"), self.args,
            )
        )

    def test_websocket_no_global_exit_pause(self):
        source = inspect.getsource(live.ws_session)

        self.assertNotIn(
            "if execution is not None or exit_task is not None:",
            source,
        )
        self.assertIn(
            "event_key in outstanding_exits(journal)",
            source,
        )
        self.assertIn(
            "event_key in loss_manager.below_since",
            source,
        )
        self.assertIn(
            "if code not in entry_active:",
            source,
        )

    def test_global_uncertainty_halt_retained(self):
        source = inspect.getsource(live.main)
        self.assertIn("unsafe_exits = [", source)


if __name__ == "__main__":
    unittest.main()
