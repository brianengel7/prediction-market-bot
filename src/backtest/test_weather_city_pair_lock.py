"""Offline tests for persistent city/event pair-lock safety."""

import asyncio
import tempfile
import unittest

from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.backtest import weather_multicity_live as live


class CityPairLockTests(unittest.TestCase):

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)

        state_patch = patch.object(
            live, "STATE_FILE",
            Path(folder.name) / "journal.json"
        )
        state_patch.start()
        self.addCleanup(state_patch.stop)

        self.journal = live.Journal()

        self.args = SimpleNamespace(
            one_trade=False,
            max_pairs=100,
            max_city_event_pairs=100,
            max_cost=D("30"),
            max_daily_attempts=50,
            max_city_attempts=50,
            city_cooldown_minutes=0,
        )

    def op(self, legs=("A", "B"), event="E1"):
        return live.Opportunity(
            "NYC", event, legs,
            (D(".80"), D(".10")),
            D(".85"), 2, D("1.90")
        )

    def record(self, status="COMPLETE"):
        rec = self.journal.reserve(
            self.op(),
            one_trade=False,
            args=self.args,
        )
        self.journal.update(rec, status)

    def test_same_pair_allowed(self):
        self.record()
        self.assertIsNone(
            self.journal.entry_block_reason(
                self.op(("B", "A")), self.args
            )
        )

    def test_third_contract_blocked(self):
        self.record()
        self.assertIn(
            "pair locked",
            self.journal.entry_block_reason(
                self.op(("A", "C")), self.args
            )
        )

    def test_reserve_rejects_third_contract(self):
        self.record()
        with self.assertRaisesRegex(
            RuntimeError, "pair locked"
        ):
            self.journal.reserve(
                self.op(("A", "C")),
                one_trade=False,
                args=self.args,
            )
        self.assertEqual(
            len(self.journal.data["attempts"]), 1
        )

    def test_lock_persists_after_restart(self):
        self.record()
        reopened = live.Journal()
        self.assertIn(
            "pair locked",
            reopened.entry_block_reason(
                self.op(("A", "C")), self.args
            )
        )

    def test_new_event_unlocked(self):
        self.record()
        self.assertIsNone(
            self.journal.entry_block_reason(
                self.op(("A", "C"), "E2"),
                self.args,
            )
        )

    def test_unwound_attempt_not_locked(self):
        self.record("UNWOUND")
        self.assertIsNone(
            self.journal.entry_block_reason(
                self.op(("A", "C")), self.args
            )
        )

    def test_existing_three_market_history_blocks(self):
        self.record()
        self.journal.data["attempts"].append({
            "key": "NYC:E1",
            "status": "COMPLETE",
            "legs": ["A", "C"],
            "qty": 2,
        })
        self.assertIn(
            "multiple completed pairs",
            self.journal.entry_block_reason(
                self.op(), self.args
            )
        )

    def test_preflight_blocks_before_api(self):
        self.record()

        op = self.op(("A", "C"))
        api = SimpleNamespace(cached_balance=D("100"))

        with patch.object(
            live, "evaluate", return_value=op
        ):
            result = asyncio.run(
                live.execute(
                    api, self.journal, {}, {},
                    op, ("A", "B", "C"),
                    self.args,
                )
            )

        self.assertEqual(result, (False, False))

    def test_existing_third_position_blocks(self):
        op = self.op()
        api = SimpleNamespace(cached_balance=D("100"))

        positions = AsyncMock(return_value={
            "A": D(2),
            "B": D(2),
            "C": D(1),
        })

        with (
            patch.object(live, "evaluate", return_value=op),
            patch.object(
                live, "positions_for_event", positions
            ),
        ):
            result = asyncio.run(
                live.execute(
                    api, self.journal, {},
                    {
                        "A": live.time.monotonic(),
                        "B": live.time.monotonic(),
                    },
                    op, ("A", "B", "C"),
                    self.args,
                )
            )

        self.assertEqual(result, (False, False))
        self.assertEqual(
            len(self.journal.data["attempts"]), 0
        )
        positions.assert_awaited_once_with(
            api, "E1", ("A", "B", "C")
        )


if __name__ == "__main__":
    unittest.main()
