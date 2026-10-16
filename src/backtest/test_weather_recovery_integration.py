"""Fully offline integration tests. No live API or orders."""
import asyncio
import tempfile
import time
import unittest
import sys
import types

from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


scanner = types.ModuleType(
    "src.backtest.weather_yes_basket_scanner"
)

class Book:
    pass

scanner.Book = Book
scanner.WS_URL = "wss://fake.invalid"
scanner.scan = lambda *a, **kw: None
sys.modules[scanner.__name__] = scanner

capacity = types.ModuleType(
    "src.backtest.weather_multicity_capacity_scanner"
)
capacity.CITY = {
    "NYC": (
        "New York",
        "KXHIGHNY",
        "America/New_York",
    )
}
sys.modules[capacity.__name__] = capacity

fees = types.ModuleType("src.kalshi.fees")
fees.calculate_taker_fee = (
    lambda **kw: D("0.01")
)
sys.modules[fees.__name__] = fees

from src.backtest import weather_multicity_live as live


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class FakeExchange:
    base = "https://fake.invalid"
    cached_balance = D("50")

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.orders = []
        self.fills = []
        self.positions = {
            "A": D(0),
            "B": D(0),
        }
        self.client = self
        self.calls = []

    def headers(self, *args):
        return {}

    async def balance(self):
        return self.cached_balance

    async def fok(
        self, ticker, side, price, qty,
        client_id, *, reduce_only=False,
    ):
        outcome = self.outcomes.pop(0)

        self.calls.append((
            ticker, side, qty,
            reduce_only, outcome,
        ))

        if outcome == "reject":
            raise live.OrderRejected(
                "HTTP 400 definite rejection"
            )

        if outcome != "absent":
            oid = f"o{len(self.orders)}"
            filled = (
                qty
                if outcome in ("fill", "unknown-fill")
                else 0
            )
            status = (
                "executed" if filled else "canceled"
            )

            self.orders.append({
                "order_id": oid,
                "ticker": ticker,
                "client_order_id": client_id,
                "book_side": side,
                "fill_count_fp": f"{filled}.00",
                "remaining_count_fp": "0.00",
                "status": status,
            })

            if filled:
                self.fills.append({
                    "order_id": oid,
                    "fill_id": f"f{len(self.fills)}",
                    "ticker": ticker,
                    "side": "yes",
                    "action": (
                        "buy" if side == "bid" else "sell"
                    ),
                    "count_fp": f"{filled}.00",
                })

                direction = (
                    1 if side == "bid" else -1
                )

                self.positions[ticker] += (
                    D(filled) * direction
                )

        if (
            outcome.startswith("unknown")
            or outcome == "absent"
        ):
            raise live.OrderUnknown(
                "network uncertain"
            )

        return outcome == "fill", oid

    async def get(self, path, **kwargs):
        if path.startswith(self.base):
            path = path.removeprefix(self.base)
            params = kwargs.get("params") or {}

            if path == "/portfolio/orders":
                rows = [
                    row for row in self.orders
                    if row["ticker"] == params["ticker"]
                ]
                return Response({
                    "orders": rows,
                    "cursor": "",
                })

            if path == "/portfolio/fills":
                rows = [
                    row for row in self.fills
                    if row["order_id"]
                    == params["order_id"]
                ]
                return Response({
                    "fills": rows,
                    "cursor": "",
                })

            if path == "/portfolio/positions":
                rows = [
                    {
                        "ticker": ticker,
                        "position_fp": str(value),
                    }
                    for ticker, value
                    in self.positions.items()
                    if value
                ]
                return Response({
                    "market_positions": rows,
                    "cursor": "",
                })

            raise RuntimeError(path)

        if path.startswith("/portfolio/orders/"):
            oid = path.rsplit("/", 1)[1]
            matches = [
                row for row in self.orders
                if row["order_id"] == oid
            ]
            return {
                "order": (
                    matches[0] if matches else None
                )
            }

        raise RuntimeError(path)


class FlowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)

        state_patch = patch.object(
            live,
            "STATE_FILE",
            Path(directory.name) / "state.json",
        )
        state_patch.start()
        self.addCleanup(state_patch.stop)

        self.journal = live.Journal()

        self.op = live.Opportunity(
            "NYC",
            "EVENT",
            ("A", "B"),
            (D(".45"), D(".47")),
            D(".9"),
            2,
            D("1.88"),
        )

        self.books = {
            ticker: SimpleNamespace(
                ready=True,
                levels={
                    "yes": {D(".40"): D(20)},
                    "no": {D(".53"): D(20)},
                },
            )
            for ticker in ("A", "B")
        }

        self.seen = {
            ticker: time.monotonic()
            for ticker in ("A", "B")
        }

        self.args = SimpleNamespace(
            one_trade=False,
            max_pairs=10,
            max_city_event_pairs=20,
            max_cost=D("10"),
            max_daily_attempts=50,
            max_city_attempts=50,
            city_cooldown_minutes=D("0"),
        )

    def run_flow(self, outcomes):
        exchange = FakeExchange(outcomes)

        with patch.object(
            live,
            "evaluate",
            return_value=self.op,
        ):
            result = asyncio.run(
                live.execute(
                    exchange,
                    self.journal,
                    self.books,
                    self.seen,
                    self.op,
                    ("A", "B"),
                    self.args,
                )
            )

        return result, exchange

    def test_both_fill(self):
        result, api = self.run_flow([
            "fill", "fill",
        ])
        self.assertEqual(result, (True, False))
        self.assertEqual(
            self.journal.data["attempts"][-1]["status"],
            "COMPLETE",
        )
        self.assertEqual(len(api.calls), 2)

    def test_second_unfilled_then_retry_fill(self):
        result, api = self.run_flow([
            "fill", "unfilled", "fill",
        ])
        self.assertEqual(result, (True, False))
        self.assertEqual(
            self.journal.data["attempts"][-1]["status"],
            "COMPLETE",
        )
        self.assertEqual(len(api.calls), 3)

    def test_retry_unfilled_then_unwind(self):
        result, api = self.run_flow([
            "fill", "unfilled",
            "unfilled", "fill",
        ])
        self.assertEqual(result, (True, False))
        self.assertEqual(
            self.journal.data["attempts"][-1]["status"],
            "UNWOUND",
        )
        self.assertEqual(
            api.calls[-1][:2], ("A", "ask")
        )
        self.assertTrue(api.calls[-1][3])
        self.assertEqual(api.positions["A"], 0)

    def test_unknown_second_filled(self):
        result, api = self.run_flow([
            "fill", "unknown-fill",
        ])
        self.assertEqual(result, (True, False))
        self.assertEqual(
            self.journal.data["attempts"][-1]["status"],
            "COMPLETE",
        )
        self.assertEqual(len(api.calls), 2)

    def test_unknown_second_not_found_halts(self):
        result, api = self.run_flow([
            "fill", "absent",
        ])
        self.assertEqual(result, (True, True))
        self.assertEqual(
            self.journal.data["attempts"][-1]["status"],
            "UNRESOLVED_NEEDS_RECONCILIATION",
        )
        self.assertEqual(len(api.calls), 2)

    def test_restart_recovers_two_fills(self):
        result, api = self.run_flow([
            "fill", "fill",
        ])

        record = self.journal.data["attempts"][-1]
        self.journal.update(
            record,
            "UNRESOLVED_NEEDS_RECONCILIATION",
        )

        asyncio.run(
            live.recover_startup(
                api, self.journal
            )
        )

        self.assertEqual(
            record["status"],
            "COMPLETE",
        )


if __name__ == "__main__":
    unittest.main()
