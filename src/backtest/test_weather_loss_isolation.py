"""Offline tests: unresolved protective sells are city-scoped."""
import asyncio
import time
import unittest
from decimal import Decimal as D
from types import SimpleNamespace

from src.backtest.weather_loss_management import (
    LossMonitor,
    recover_loss_exits,
)


class FakeJournal:
    def __init__(self):
        self.data = {
            "attempts": [
                {
                    "key": "DEN:DEN-EVENT",
                    "legs": ["A", "B"],
                    "qty": 10,
                    "status": "COMPLETE",
                },
                {
                    "key": "NYC:NYC-EVENT",
                    "legs": ["C", "D"],
                    "qty": 10,
                    "status": "COMPLETE",
                },
            ],
            "loss_exits": {
                "DEN:DEN-EVENT": {
                    "state": "RECONCILING",
                    "legs": ["A", "B"],
                    "original_qty_each": 10,
                    "loss_sell_0_client_id": "CID-A",
                    "loss_sell_0_ticker": "A",
                    "loss_sell_0_qty": "10",
                },
            },
        }

    def write(self):
        pass

    def update(self, record, status, **fields):
        record.update(status=status, **fields)


class SellIsolationTests(unittest.TestCase):

    def test_failed_reconciliation_is_nonfatal(self):
        journal = FakeJournal()
        queries = []

        async def unavailable(api, **kwargs):
            queries.append(kwargs["client_id"])
            raise RuntimeError("Exchange temporarily unavailable")

        asyncio.run(
            recover_loss_exits(None, journal, unavailable)
        )

        entry = journal.data["loss_exits"]["DEN:DEN-EVENT"]
        self.assertEqual(entry["state"], "RECONCILING")
        self.assertEqual(queries, ["CID-A"])
        self.assertNotIn("loss_sell_0_filled", entry)

        async def terminal(api, **kwargs):
            queries.append(kwargs["client_id"])
            return SimpleNamespace(
                order_id="ORDER-A",
                filled=D("4.00"),
                terminal=True,
            )

        asyncio.run(
            recover_loss_exits(None, journal, terminal)
        )

        self.assertEqual(queries, ["CID-A", "CID-A"])
        self.assertEqual(entry["state"], "ACTIVE")
        self.assertEqual(D(entry["loss_sell_0_filled"]), D(4))

    def test_denver_pending_nyc_still_scans(self):
        journal = FakeJournal()
        seen = []
        queries = []

        async def unavailable(api, **kwargs):
            queries.append(kwargs["client_id"])
            raise RuntimeError("Temporary failure")

        manager = LossMonitor(
            journal,
            enabled=True,
            quotes=lambda ticker: (
                D(".25"), D(".26"), D(10), D(10),
            ),
            positions=None,
            submit=None,
            log=lambda msg: None,
            reconcile=unavailable,
        )

        async def fake_sell(api, key, *args):
            seen.append(key)

        manager._sell_once = fake_sell

        now = time.monotonic()
        manager.basket_identity["NYC:NYC-EVENT"] = (
            ("C", "D"), 10
        )
        manager.below_since["NYC:NYC-EVENT"] = now - 5

        cities = {
            "DEN": ("DEN-EVENT", ("A", "B"), None),
            "NYC": ("NYC-EVENT", ("C", "D"), None),
        }
        books = {t: t for t in ("A", "B", "C", "D")}
        last_seen = {t: now for t in books}

        asyncio.run(
            manager.tick(None, cities, books, last_seen)
        )

        self.assertEqual(queries, ["CID-A"])
        self.assertEqual(seen, ["NYC:NYC-EVENT"])
        self.assertEqual(
            journal.data["loss_exits"]["DEN:DEN-EVENT"]["state"],
            "RECONCILING",
        )


if __name__ == "__main__":
    unittest.main()
