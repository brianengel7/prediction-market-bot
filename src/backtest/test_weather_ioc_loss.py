"""Offline IOC protective-sell tests. No network or orders."""
import json
import unittest
from decimal import Decimal as D
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.backtest import weather_multicity_live as live
from src.backtest import weather_order_reconciliation as recon
from src.backtest.weather_loss_management import LossMonitor


class FakeJournal:
    def __init__(self):
        self.data = {"attempts": [{
            "key": "DEN:E",
            "status": "COMPLETE",
            "qty": 3,
            "legs": ["A", "B"],
        }]}
        self.writes = 0

    def write(self):
        json.dumps(self.data)
        self.writes += 1

    def update(self, record, status, **extras):
        record.update(status=status, **extras)
        self.write()


class Tests(unittest.IsolatedAsyncioTestCase):

    async def test_ioc_request(self):
        api = object.__new__(live.Kalshi)
        api.base = "https://example.com/trade-api/v2"
        api.headers = lambda *_: {}
        api.client = SimpleNamespace(post=AsyncMock(
            return_value=SimpleNamespace(
                status_code=201,
                json=lambda: {"order_id": "OID"},
                text="",
            )
        ))

        _, order_id = await api.ioc_sell(
            "A", D(".21"), D("1.25"), "CID"
        )

        body = api.client.post.await_args.kwargs["json"]

        self.assertEqual(order_id, "OID")
        self.assertEqual(body["side"], "ask")
        self.assertEqual(
            body["time_in_force"], "immediate_or_cancel"
        )
        self.assertIs(body["reduce_only"], True)
        self.assertEqual(body["count"], "1.25")
        self.assertEqual(body["price"], "0.2100")

    async def test_partial_order_reconciliation(self):
        order = {
            "order_id": "OID",
            "client_order_id": "CID",
            "ticker": "A",
            "book_side": "ask",
            "action": "sell",
            "fill_count_fp": "1.25",
            "remaining_count_fp": "0.00",
            "status": "canceled",
        }

        fills = [
            {
                "fill_id": f"F{i}",
                "order_id": "OID",
                "ticker": "A",
                "action": "sell",
                "book_side": "ask",
                "side": "no",
                "count_fp": amount,
            }
            for i, amount in enumerate(("0.50", "0.75"))
        ]

        api = SimpleNamespace(
            get=AsyncMock(return_value={"order": order})
        )

        with patch.object(
            recon, "paged",
            new=AsyncMock(return_value=fills),
        ):
            result = await recon.reconcile_once(
                api,
                ticker="A",
                client_id="CID",
                requested=D(3),
                order_id="OID",
                side="ask",
                allow_partial=True,
            )

            self.assertEqual(result.filled, D("1.25"))
            self.assertTrue(result.terminal)
            self.assertFalse(result.full)

            with self.assertRaises(recon.ReconciliationError):
                await recon.reconcile_once(
                    api,
                    ticker="A",
                    client_id="CID",
                    requested=3,
                    order_id="OID",
                    side="ask",
                )

    async def test_uncertain_ack_no_duplicate_post(self):
        journal = FakeJournal()
        entry = {"state": "ACTIVE"}

        api = SimpleNamespace(
            ioc_sell=AsyncMock(
                side_effect=live.OrderUnknown("timeout")
            )
        )

        async def prove(*args, **kwargs):
            self.assertIsNone(kwargs["order_id"])
            self.assertTrue(kwargs["allow_partial"])
            return recon.Execution(
                "A", kwargs["client_id"], "OID",
                D(2), D("1.25"), D(0), "canceled",
            )

        with patch.object(live, "reconcile_order", prove):
            result = await live.submit_verified(
                api, journal, entry, "loss_sell_0",
                "A", "ask", D(".30"), D(2),
                reduce_only=True,
            )

        self.assertEqual(result.filled, D("1.25"))
        api.ioc_sell.assert_awaited_once()
        self.assertEqual(
            entry["loss_sell_0_filled"], "1.25"
        )

    async def test_partial_then_complete_liquidation(self):
        journal = FakeJournal()
        holdings = {"A": D(3), "B": D(3)}
        counts = {"A": 0, "B": 0}

        api = SimpleNamespace(
            balance=AsyncMock(return_value=D(50))
        )

        async def positions(_api, event, legs):
            return dict(holdings)

        async def submit(
            api, journal, entry, prefix,
            ticker, side, price, qty, *, reduce_only,
        ):
            self.assertEqual(side, "ask")
            self.assertTrue(reduce_only)

            counts[ticker] += 1

            filled = (
                D("1.25")
                if ticker == "A" and counts["A"] == 1
                else D(str(qty))
            )

            holdings[ticker] -= filled

            entry[prefix + "_client_id"] = prefix
            entry[prefix + "_filled"] = str(filled)
            journal.write()

            return SimpleNamespace(
                full=filled == qty,
                unfilled=filled == 0,
                terminal=True,
                filled=filled,
            )

        monitor = LossMonitor(
            journal,
            enabled=True,
            quotes=lambda _: (
                D(".30"), D(".35"), D(5), D(5)
            ),
            positions=positions,
            submit=submit,
            log=lambda _: None,
        )

        entry = {
            "state": "ACTIVE",
            "legs": ["A", "B"],
            "original_qty_each": 3,
        }
        journal.data["loss_exits"] = {"DEN:E": entry}

        books = {"A": "A", "B": "B"}

        async def cycle():
            with patch(
                "src.backtest.weather_loss_management.time.monotonic",
                return_value=0.0,
            ):
                await monitor._sell_once(
                    api, "DEN:E", "E",
                    ("A", "B"), 3, entry,
                    books, {"A": 0.0, "B": 0.0},
                )

        await cycle()

        self.assertEqual(holdings["A"], D("1.75"))
        self.assertEqual(holdings["B"], 0)

        await cycle()

        self.assertEqual(holdings["A"], 0)
        self.assertEqual(holdings["B"], 0)

        await cycle()

        self.assertEqual(entry["state"], "DONE")
        self.assertTrue(entry["closure_verified"])
        self.assertTrue(
            journal.data["attempts"][0][
                "closed_exposure_verified"
            ]
        )
        self.assertEqual(api.balance.await_count, 1)


if __name__ == "__main__":
    unittest.main()
