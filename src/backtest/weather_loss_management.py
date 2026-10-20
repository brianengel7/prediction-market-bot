"""Conservative, opt-in loss management for multicity YES baskets."""
from __future__ import annotations

import time
from datetime import datetime
from decimal import Decimal as D

STOP_BID = D("0.60")
CONFIRM_SECONDS = 2.0
QUOTE_MAX_AGE = 3.0
RETRY_SECONDS = 5.0


def held_cities(journal, allowed):
    allowed = set(allowed)
    return {
        record.get("key", "").split(":", 1)[0]
        for record in journal.data["attempts"]
        if record.get("status") == "COMPLETE"
        and record.get("closed_exposure_verified") is not True
        and ":" in record.get("key", "")
    } & allowed


def outstanding_exits(journal):
    return {
        key: entry
        for key, entry in journal.data.get("loss_exits", {}).items()
        if entry.get("state") != "DONE"
    }


def _pending_order_fields(entry):
    for key, cid in entry.items():
        if not key.startswith("loss_sell_") or not key.endswith("_client_id"):
            continue
        prefix = key[:-len("_client_id")]
        if not entry.get(prefix + "_rejected") and prefix + "_filled" not in entry:
            yield prefix, cid


async def recover_loss_exits(api, journal, reconcile_order):
    """Recover saved sell orders without submitting replacements."""
    for key, entry in outstanding_exits(journal).items():
        pending = list(_pending_order_fields(entry))
        if entry.get("state") == "HALTED" and not pending:
            raise RuntimeError(f"Loss exit {key} requires manual review")

        for prefix, cid in pending:
            result = await reconcile_order(
                api,
                ticker=entry[prefix + "_ticker"],
                client_id=cid,
                requested=D(str(entry[prefix + "_qty"])),
                order_id=entry.get(prefix + "_order_id"),
                side="ask",
                allow_partial=True,
            )
            journal.update(
                entry, "LOSS_SELL_RECONCILED",
                **{
                    prefix + "_order_id": result.order_id,
                    prefix + "_filled": (
                        str(result.filled)
                        if result.filled != int(result.filled)
                        else int(result.filled)
                    ),
                },
            )

        entry["state"] = "ACTIVE"
        journal.write()



def finish_loss_exit(journal, key, entry, legs, original):
    """Called only after Kalshi positions are verified as zero."""
    if entry.get("state") != "ACTIVE":
        raise RuntimeError("Exit must be active to finalize")

    if tuple(sorted(entry.get("legs", []))) != tuple(sorted(legs)):
        raise RuntimeError("Exit pair identity mismatch")

    if int(entry.get("original_qty_each", -1)) != int(original):
        raise RuntimeError("Original exit quantity mismatch")

    if list(_pending_order_fields(entry)):
        raise RuntimeError("Outstanding unverified sell order identity")

    verified_sold = {ticker: D(0) for ticker in legs}

    for name, value in entry.items():
        if name.startswith("loss_sell_") and name.endswith("_filled"):
            prefix = name[:-len("_filled")]
            ticker = entry.get(prefix + "_ticker")

            if ticker not in verified_sold or not entry.get(
                prefix + "_client_id"
            ):
                raise RuntimeError("Missing verified sell identity")

            if entry.get(prefix + "_rejected"):
                raise RuntimeError(
                    "Rejected sell cannot have verified fills"
                )

            amount = D(str(value))

            if amount < 0 or amount > D(str(entry[prefix + "_qty"])):
                raise RuntimeError("Invalid verified sell quantity")

            verified_sold[ticker] += amount

    if any(v != D(original) for v in verified_sold.values()):
        raise RuntimeError(
            "Cannot close until both legs are fully sold"
        )

    records = [
        a for a in journal.data["attempts"]
        if a.get("key") == key
        and a.get("status") == "COMPLETE"
        and a.get("closed_exposure_verified") is not True
    ]

    if not records or any(
        tuple(sorted(r.get("legs", []))) != tuple(sorted(legs))
        for r in records
    ):
        raise RuntimeError(
            "Completed basket history inconsistent with exit"
        )

    if sum(int(r["qty"]) for r in records) != original:
        raise RuntimeError(
            "Basket quantity changed while exiting"
        )

    closed_at = datetime.now().astimezone().isoformat()

    for record in records:
        record["closed_exposure_verified"] = True
        record["closed_reason"] = "combined_bid_below_60c"
        record["loss_exit_closed_at"] = closed_at

    entry["state"] = "DONE"
    entry["closed_at"] = closed_at
    entry["closure_verified"] = True

    # One durable update. No historical attempt is deleted.
    journal.write()



class LossMonitor:
    def __init__(self, journal, *, enabled, quotes, positions, submit, log):
        self.journal = journal
        self.enabled = enabled
        self.quotes = quotes
        self.positions = positions
        self.submit = submit
        self.log = log
        self.below_since = {}
        self.next_attempt = {}
        self.shadow_fired = set()
        self.basket_identity = {}

    def _baskets(self, cities):
        result = {}

        for city, (event, _markets, _date) in cities.items():
            key = f"{city}:{event}"
            previous = self.journal.data.get("loss_exits", {}).get(key)
            if previous is not None and previous.get("state") == "DONE":
                if previous.get("closure_verified") is not True:
                    raise RuntimeError(f"Unverified loss closure: {key}")
                continue  # One stop already used for this event.
            records = [
                a for a in self.journal.data["attempts"]
                if a.get("key") == key
                and a.get("status") == "COMPLETE"
                and a.get("closed_exposure_verified") is not True
            ]
            if not records:
                continue

            pairs = {tuple(sorted(a["legs"])) for a in records}
            if len(pairs) != 1:
                raise RuntimeError(f"Ambiguous owned pair: {key}")

            legs = next(iter(pairs))
            result[key] = (
                city, event, legs,
                sum(int(a["qty"]) for a in records),
            )

        return result

    def _expected(self, key, legs, original):
        result = {ticker: D(original) for ticker in legs}
        entry = self.journal.data.get("loss_exits", {}).get(key, {})

        for name, value in entry.items():
            if name.startswith("loss_sell_") and name.endswith("_filled"):
                prefix = name[:-len("_filled")]
                ticker = entry[prefix + "_ticker"]
                if ticker not in result:
                    raise RuntimeError("Unrecognized recorded exit ticker")
                result[ticker] -= D(str(value))

        if any(v < 0 for v in result.values()):
            raise RuntimeError("Recorded exit fills exceed purchased exposure")

        return result

    def _fresh(self, legs, books, last_seen, now):
        if any(
            ticker not in books
            or now - last_seen.get(ticker, -1e12) > QUOTE_MAX_AGE
            or last_seen.get(ticker, -1e12) > now
            for ticker in legs
        ):
            return None

        q = [self.quotes(books[ticker]) for ticker in legs]
        if any(row is None for row in q):
            return None

        return q

    async def tick(self, api, cities, books, last_seen):
        now = time.monotonic()

        for key, (city, event, legs, original) in self._baskets(cities).items():
            identity = (legs, original)
            if self.basket_identity.get(key) != identity:
                self.basket_identity[key] = identity
                self.below_since.pop(key, None)
                self.next_attempt.pop(key, None)
                self.shadow_fired.discard(key)
            entry = self.journal.data.get("loss_exits", {}).get(key)

            if entry is not None and entry.get("state") == "DONE":
                continue

            if entry is not None and entry.get("state") != "ACTIVE":
                raise RuntimeError(f"Unresolved loss exit {key}")

            q = self._fresh(legs, books, last_seen, now)
            if q is None and entry is None:
                self.below_since.pop(key, None)
                continue

            if entry is None:
                bid_sum = q[0][0] + q[1][0]
                if bid_sum >= STOP_BID:
                    self.below_since.pop(key, None)
                    self.shadow_fired.discard(key)
                    continue

                first = self.below_since.setdefault(key, now)
                if now - first < CONFIRM_SECONDS:
                    continue

                if not self.enabled:
                    if key not in self.shadow_fired:
                        self.log(
                            f"SHADOW LOSS EXIT {key} {legs} "
                            f"bids={q[0][0]}+{q[1][0]}={bid_sum}; "
                            "NO ORDERS"
                        )
                        self.shadow_fired.add(key)
                    continue

                entry = {
                    "state": "ACTIVE",
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "legs": list(legs),
                    "original_qty_each": original,
                    "trigger_bid_sum": str(bid_sum),
                }

                self.journal.data.setdefault("loss_exits", {})[key] = entry
                self.journal.write()

                self.log(f"LOSS EXIT TRIGGERED {key}; blocking new buys")

            if now < self.next_attempt.get(key, 0):
                continue

            self.next_attempt[key] = now + RETRY_SECONDS

            await self._sell_once(
                api, key, event, legs, original,
                entry, books, last_seen,
            )

    async def _sell_once(
        self, api, key, event, legs, original,
        entry, books, last_seen,
    ):
        expected = self._expected(key, legs, original)
        actual = await self.positions(api, event, legs)

        if actual != expected:
            entry["state"] = "HALTED"
            entry["error"] = (
                f"Position mismatch expected={expected} actual={actual}"
            )
            self.journal.write()
            raise RuntimeError(f"LOSS EXIT HALTED {key}: {entry['error']}")

        if all(value == 0 for value in actual.values()):
            finish_loss_exit(
                self.journal, key, entry, legs, original
            )
            # Refresh cash before allowing immediate re-entry.
            if hasattr(api, "balance"):
                await api.balance()
            self.log(
                f"LOSS EXIT VERIFIED CLOSED {key}; "
                "pair lock released; re-entry rules still apply"
            )
            return

        for ticker in legs:
            amount = actual[ticker]

            if amount == 0:
                continue

            if amount < 0 or amount != amount.quantize(D("0.01")):
                entry["state"] = "HALTED"
                entry["error"] = (
                    f"Invalid holding precision: {ticker}={amount}"
                )
                self.journal.write()
                raise RuntimeError(entry["error"])

            fresh = self._fresh(
                legs, books, last_seen, time.monotonic()
            )
            if fresh is None:
                return

            quote = fresh[legs.index(ticker)]
            bid, depth = quote[0], quote[2]
            qty = min(
                amount, D(str(depth))
            ).quantize(
                D("0.01"), rounding="ROUND_DOWN"
            )

            if qty < D("0.01"):
                continue

            seq = int(entry.get("next_seq", 0))
            entry["next_seq"] = seq + 1
            prefix = f"loss_sell_{seq}"

            entry[prefix + "_ticker"] = ticker
            entry[prefix + "_qty"] = str(qty)
            entry[prefix + "_limit"] = str(bid)
            self.journal.write()

            try:
                verified = await self.submit(
                    api, self.journal, entry, prefix,
                    ticker, "ask", bid, qty,
                    reduce_only=True,
                )
            except Exception as exc:
                entry["state"] = "HALTED"
                entry["error"] = repr(exc)
                self.journal.write()
                raise

            if verified is None:
                self.log(
                    f"LOSS SELL REJECTED {key} {ticker}; "
                    "will recheck"
                )
                continue

            if not (
                verified.full
                or verified.unfilled
                or getattr(verified, "terminal", False)
            ):
                entry["state"] = "HALTED"
                entry["error"] = "Nonterminal/partial sell result"
                self.journal.write()
                raise RuntimeError(entry["error"])

            self.log(
                f"LOSS SELL VERIFIED {key} {ticker} "
                f"filled={verified.filled}/{qty} limit={bid}"
            )
            actual[ticker] -= D(verified.filled)

        # The next tick independently verifies remaining Kalshi positions.
