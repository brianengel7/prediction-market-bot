"""Offline-only protective exit tests. No credentials or network."""
import asyncio
import unittest
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import patch

from src.backtest.weather_loss_management import (
    LossMonitor, held_cities, outstanding_exits, recover_loss_exits,
)


class FakeJournal:
    def __init__(self):
        self.data = {"attempts": [{
            "key": "DEN:EVENT",
            "legs": ["A", "B"],
            "qty": 10,
            "status": "COMPLETE",
        }]}
        self.writes = 0

    def write(self):
        self.writes += 1

    def update(self, record, status, **extras):
        record.update(status=status, **extras)
        self.write()


class Tests(unittest.TestCase):
    def setUp(self):
        self.journal = FakeJournal()
        self.api = SimpleNamespace(
            positions={"A": D(10), "B": D(10)}
        )
        self.cities = {
            "DEN": ("EVENT", ("A", "B"), None)
        }
        self.bids = {
            "A": D(".36"), "B": D(".21")
        }
        self.depths = {
            "A": D(10), "B": D(10)
        }
        self.books = {"A": "A", "B": "B"}
        self.last_seen = {"A": 0.0, "B": 0.0}
        self.calls = []
        self.logs = []
        self.reject_second = False
        self.raise_unknown = False

    def quotes(self, ticker):
        return (
            self.bids[ticker], D(".9"),
            self.depths[ticker], D(10),
        )

    async def positions(self, _api, _event, _legs):
        return dict(self.api.positions)

    async def submit(
        self, _api, journal, entry, prefix,
        ticker, side, price, qty, *, reduce_only,
    ):
        assert side == "ask"
        assert reduce_only is True

        self.calls.append((ticker, price, qty))
        entry[prefix + "_client_id"] = "CID" + str(len(self.calls))
        journal.write()

        if self.raise_unknown:
            raise RuntimeError("Uncertain order response")

        if self.reject_second and ticker == "B":
            entry[prefix + "_rejected"] = True
            journal.write()
            return None

        self.api.positions[ticker] -= D(qty)
        entry[prefix + "_filled"] = qty
        journal.write()

        return SimpleNamespace(
            full=True,
            unfilled=False,
            filled=qty,
        )

    def manager(self, enabled):
        return LossMonitor(
            self.journal,
            enabled=enabled,
            quotes=self.quotes,
            positions=self.positions,
            submit=self.submit,
            log=self.logs.append,
        )

    def tick(self, manager, seconds):
        self.last_seen = {
            "A": seconds,
            "B": seconds,
        }
        with patch(
            "src.backtest.weather_loss_management.time.monotonic",
            return_value=seconds,
        ):
            asyncio.run(
                manager.tick(
                    self.api,
                    self.cities,
                    self.books,
                    self.last_seen,
                )
            )

    def test_above_threshold_never_triggers(self):
        self.bids["A"] = D(".45")
        self.bids["B"] = D(".30")
        manager = self.manager(True)

        for t in (0, 1, 2, 4):
            self.tick(manager, t)

        self.assertEqual(self.calls, [])
        self.assertNotIn("loss_exits", self.journal.data)

    def test_two_second_shadow_never_submits(self):
        manager = self.manager(False)

        for t in (0, 1, 2, 3):
            self.tick(manager, t)

        self.assertEqual(self.calls, [])
        self.assertEqual(self.journal.writes, 0)
        self.assertEqual(len(self.logs), 1)

    def test_stale_quotes_reset_trigger(self):
        manager = self.manager(True)
        self.tick(manager, 0)

        with patch(
            "src.backtest.weather_loss_management.time.monotonic",
            return_value=10,
        ):
            asyncio.run(
                manager.tick(
                    self.api,
                    self.cities,
                    self.books,
                    {"A": 0, "B": 0},
                )
            )

        self.tick(manager, 11)
        self.tick(manager, 12)

        self.assertEqual(self.calls, [])

    def test_full_liquidation_but_no_reentry(self):
        manager = self.manager(True)

        for t in (0, 1, 2, 8):
            self.tick(manager, t)

        self.assertEqual(
            [x[0] for x in self.calls],
            ["A", "B"],
        )
        self.assertEqual(
            self.api.positions,
            {"A": 0, "B": 0},
        )
        self.assertEqual(
            self.journal.data["loss_exits"]["DEN:EVENT"]["state"],
            "DONE",
        )

    def test_partial_depth_and_one_leg_rejected(self):
        self.depths["A"] = D(3)
        self.reject_second = True
        manager = self.manager(True)

        for t in (0, 1, 2):
            self.tick(manager, t)

        self.assertEqual(self.api.positions["A"], D(7))
        self.assertEqual(self.api.positions["B"], D(10))
        self.assertEqual(self.calls[0][2], 3)

        self.assertEqual(
            self.journal.data["loss_exits"]["DEN:EVENT"]["state"],
            "ACTIVE",
        )
        self.assertTrue(outstanding_exits(self.journal))

    def test_unknown_exit_halts_and_persists(self):
        self.raise_unknown = True
        manager = self.manager(True)

        self.tick(manager, 0)
        self.tick(manager, 1)

        self.tick(manager, 2)

        entry = self.journal.data["loss_exits"]["DEN:EVENT"]
        self.assertEqual(entry["state"], "RECONCILING")
        self.assertTrue(entry["loss_sell_0_client_id"])

    def test_restart_reconciles_without_resubmission(self):
        entry = {
            "state": "HALTED",
            "loss_sell_0_client_id": "CID1",
            "loss_sell_0_ticker": "A",
            "loss_sell_0_qty": 3,
        }
        self.journal.data["loss_exits"] = {
            "DEN:EVENT": entry
        }

        async def reconcile(*args, **kwargs):
            self.assertEqual(kwargs["side"], "ask")
            return SimpleNamespace(
                order_id="OID1",
                filled=3,
                terminal=True,
            )

        asyncio.run(
            recover_loss_exits(
                self.api, self.journal, reconcile,
            )
        )

        self.assertEqual(entry["state"], "ACTIVE")
        self.assertEqual(D(str(entry["loss_sell_0_filled"])), D(3))
        self.assertEqual(self.calls, [])

    def test_position_mismatch_blocks_selling(self):
        self.api.positions["A"] = D(9)
        manager = self.manager(True)

        self.tick(manager, 0)
        self.tick(manager, 1)

        self.tick(manager, 2)

        self.assertEqual(self.calls, [])
        self.assertEqual(
            self.journal.data["loss_exits"]["DEN:EVENT"]["state"],
            "HALTED",
        )

    def test_held_city_discovery(self):
        self.assertEqual(
            held_cities(self.journal, ["DEN", "NYC"]),
            {"DEN"},
        )

    def test_stopped_pair_remains_blocked(self):
        from src.backtest.weather_multicity_live import Journal

        journal = object.__new__(Journal)
        journal.data = {
            "attempts": [],
            "loss_exits": {
                "DEN:EVENT": {"state": "DONE"}
            },
        }

        opportunity = SimpleNamespace(
            key="DEN:EVENT",
            city="DEN",
            event="EVENT",
            legs=("A", "B"),
        )

        reason = journal.entry_block_reason(
            opportunity,
            SimpleNamespace(),
        )
        self.assertIn("loss exit", reason)


if __name__ == "__main__":
    unittest.main()
