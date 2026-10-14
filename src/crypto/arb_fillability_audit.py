"""Read-only, event-driven two-venue top-of-book audit.

No orders are submitted. Top-of-book persistence is not
proof of an exchange-side fill.
"""
import json
import time
from collections import Counter
from decimal import Decimal


class PassiveFillabilityAudit:
    HORIZONS_MS = (25, 50, 100, 250)
    LATE_MS = 25
    CENSOR_MS = 5000
    MAX_QUOTE_AGE_MS = 200

    def __init__(self, engine):
        self.engine = engine
        self.active = {}
        self.counts = {h: Counter() for h in self.HORIZONS_MS}
        self.completed = 0
        self.fresh_starts = 0
        self.best_net_cents = None
        self.next_id = 0

    def _ages(self):
        s = self.engine.state
        now = time.monotonic()
        k = s.books.get(s.kalshi_btc) if s.kalshi_btc else None
        p = s.poly if s.poly_slug else None

        def value(book):
            if book is None or not getattr(book, "ready", False):
                return None
            seen = getattr(book, "seen", None)
            if seen is None:
                return None
            return round(max(0, (now - seen) * 1000), 3)

        return {
            "kalshi_ms": value(k),
            "polymarket_us_ms": value(p),
        }

    @classmethod
    def _fresh(cls, ages):
        return all(
            x is not None and x <= cls.MAX_QUOTE_AGE_MS
            for x in ages.values()
        )

    def _sample(self, episode, legs, row, now_ns, ages):
        elapsed = (now_ns - episode["started_ns"]) / 1e6
        initial = episode["initial_legs"]
        qty = Decimal(episode["quantity"])
        qty_now = (
            Decimal(row["evaluated_pairs"])
            if row is not None else Decimal(0)
        )
        valid_legs = len(legs) == 2 and all(
            x is not None for x in legs
        )

        leg_checks = []

        if valid_legs:
            for current, anchor in zip(legs, initial):
                same = (
                    (
                        current.venue,
                        current.ticker,
                        current.side,
                    )
                    == (
                        anchor["venue"],
                        anchor["market"],
                        anchor["side"],
                    )
                )

                price_ok = (
                    same
                    and current.price <= Decimal(anchor["ask"])
                )
                top_depth_ok = (
                    same and current.size >= qty
                )

                leg_checks.append({
                    "venue": current.venue,
                    "side": current.side,
                    "ask": str(current.price),
                    "top_qty": str(current.size),
                    "at_or_better_than_entry_ask": bool(price_ok),
                    "top_depth_at_least_entry_qty": bool(
                        top_depth_ok
                    ),
                    "ask_inferred_from_opposite_book": bool(
                        current.inferred
                    ),
                })

        original_top_held = bool(
            valid_legs
            and len(leg_checks) == 2
            and all(
                x["at_or_better_than_entry_ask"]
                and x["top_depth_at_least_entry_qty"]
                for x in leg_checks
            )
        )

        positive_new = bool(
            row is not None
            and qty_now >= qty
            and Decimal(
                row["buffered_cents_per_pair"]
            ) > 0
        )

        return {
            "elapsed_ms": round(elapsed, 3),
            "original_top_held": original_top_held,
            "positive_repriced_net": positive_new,
            "fresh_local_book_proxy": self._fresh(ages),
            "book_ages_ms": ages,
            "reprice_gross_cents": (
                row["gross_cents_per_pair"] if row else None
            ),
            "reprice_net_est_cents": (
                row["buffered_cents_per_pair"] if row else None
            ),
            "reprice_quantity": (
                row["evaluated_pairs"] if row else None
            ),
            "legs": leg_checks,
            "no_actual_fills_tested": True,
        }

    def _start(self, key, row, now_ns, ages):
        self.next_id += 1
        fresh = self._fresh(ages)
        self.fresh_starts += int(fresh)

        episode = {
            "id": self.next_id,
            "direction": key,
            "started_ns": now_ns,
            "started_at": row["at"],
            "quantity": row["evaluated_pairs"],
            "initial_legs": [
                dict(x) for x in row["legs"]
            ],
            "start_record": dict(row),
            "initial_net_est_cents": row[
                "buffered_cents_per_pair"
            ],
            "max_net_est_cents": row[
                "buffered_cents_per_pair"
            ],
            "start_ages_ms": ages,
            "start_fresh_local_proxy": fresh,
            "poly_no_is_inferred": any(
                x["venue"] == "polymarket_us"
                and x["side"] == "no"
                and x["opposite_bid_inferred"]
                for x in row["legs"]
            ),
            "checkpoints": {},
            "last_positive_ms": 0.0,
            "max_observation_gap_ms": 0.0,
            "last_ns": now_ns,
            "observations": 1,
            "first_original_top_loss_ms": None,
        }

        self.active[key] = episode

        print(
            f"FILL_START id={episode['id']} "
            f"direction={key} "
            f"net={episode['initial_net_est_cents']}c "
            f"qty={episode['quantity']} "
            f"K_age={ages['kalshi_ms']}ms "
            f"P_age={ages['polymarket_us_ms']}ms "
            f"poly_NO_inferred={episode['poly_no_is_inferred']}",
            flush=True,
        )

    def observe(self, key, kind, legs, row):
        if kind != "CROSS_EXCHANGE":
            return

        now_ns = time.monotonic_ns()
        ages = self._ages()

        positive = (
            row is not None
            and Decimal(
                row["buffered_cents_per_pair"]
            ) > 0
        )

        episode = self.active.get(key)

        if episode is None:
            if (
                positive
                and len(legs) == 2
                and all(x is not None for x in legs)
            ):
                self._start(key, row, now_ns, ages)
            return

        episode["observations"] += 1

        gap_ms = (
            now_ns - episode["last_ns"]
        ) / 1e6

        episode["last_ns"] = now_ns
        episode["max_observation_gap_ms"] = max(
            episode["max_observation_gap_ms"],
            gap_ms,
        )

        sample = self._sample(
            episode, legs, row, now_ns, ages
        )
        elapsed = sample["elapsed_ms"]

        if sample["positive_repriced_net"]:
            episode["last_positive_ms"] = elapsed

            if Decimal(
                row["buffered_cents_per_pair"]
            ) > Decimal(
                episode["max_net_est_cents"]
            ):
                episode["max_net_est_cents"] = row[
                    "buffered_cents_per_pair"
                ]

        if (
            not sample["original_top_held"]
            and episode["first_original_top_loss_ms"] is None
        ):
            episode["first_original_top_loss_ms"] = elapsed

        for target in self.HORIZONS_MS:
            if (
                str(target) not in episode["checkpoints"]
                and elapsed >= target
            ):
                snap = dict(sample)
                snap["target_ms"] = target
                snap["lateness_ms"] = round(
                    elapsed - target, 3
                )
                episode["checkpoints"][str(target)] = snap

        if not sample["positive_repriced_net"]:
            self._finish(
                episode, elapsed, "NET_OR_DEPTH_LOST"
            )

        elif elapsed >= self.CENSOR_MS:
            self._finish(
                episode, elapsed, "CENSORED_AT_5S"
            )

    def _finish(self, episode, end_ms, reason):
        if (
            self.active.get(episode["direction"])
            is not episode
        ):
            return

        del self.active[episode["direction"]]

        results = {}

        for target in self.HORIZONS_MS:
            snap = episode["checkpoints"].get(
                str(target)
            )

            if snap is None:
                result = (
                    "LOST_BEFORE"
                    if reason != "CENSORED_AT_5S"
                    and end_ms < target
                    else "INDETERMINATE"
                )

            elif snap["lateness_ms"] > self.LATE_MS:
                result = "LATE_OBSERVATION"

            elif (
                episode["first_original_top_loss_ms"]
                is not None
                and episode[
                    "first_original_top_loss_ms"
                ] < target
            ):
                result = "ORIGINAL_TOP_LOST_BEFORE"

            elif not snap["original_top_held"]:
                result = "ORIGINAL_TOP_NOT_AVAILABLE"

            elif not snap["positive_repriced_net"]:
                result = "NET_NOT_POSITIVE"

            elif not snap["fresh_local_book_proxy"]:
                result = "TOP_HELD_OLD_BOOK_PROXY"

            else:
                result = "TOP_HELD_FRESH_BOOK_PROXY"

            results[str(target)] = result
            self.counts[target][result] += 1

        self.completed += 1

        peak = Decimal(
            episode["max_net_est_cents"]
        )

        if (
            self.best_net_cents is None
            or peak > self.best_net_cents
        ):
            self.best_net_cents = peak

        record = {
            "episode_id": episode["id"],
            "direction": episode["direction"],
            "started_at": episode["started_at"],
            "duration_until_exit_or_censor_ms": round(
                end_ms, 3
            ),
            "end_reason": reason,
            "initial_quantity": episode["quantity"],
            "initial_legs": episode["initial_legs"],
            "initial_net_est_cents": episode[
                "initial_net_est_cents"
            ],
            "max_net_est_cents": episode[
                "max_net_est_cents"
            ],
            "start_ages_ms": episode[
                "start_ages_ms"
            ],
            "start_fresh_local_proxy": episode[
                "start_fresh_local_proxy"
            ],
            "poly_no_ask_inferred_not_verified_executable": (
                episode["poly_no_is_inferred"]
            ),
            "first_original_top_loss_ms": episode[
                "first_original_top_loss_ms"
            ],
            "last_positive_observation_ms": round(
                episode["last_positive_ms"], 3
            ),
            "max_local_check_gap_ms": round(
                episode["max_observation_gap_ms"], 3
            ),
            "observations": episode["observations"],
            "horizon_results": results,
            "horizon_snapshots": episode["checkpoints"],
            "settlement_equivalence_verified": False,
            "polymarket_execution_semantics_verified": False,
            "actual_order_fill_tested": False,
            "orders_submitted": 0,
            "top_level_only_conservative_test": True,
        }

        print(
            f"FILL_END id={episode['id']} "
            f"{episode['direction']} "
            f"dur~{end_ms:.2f}ms "
            f"net_peak={peak:+.3f}c "
            f"25={results['25']} "
            f"50={results['50']} "
            f"100={results['100']} "
            f"250={results['250']} "
            f"P_NO_inferred={episode['poly_no_is_inferred']}",
            flush=True,
        )

        print(
            "FILL_JSON "
            + json.dumps(
                record, separators=(",", ":")
            ),
            flush=True,
        )

        # Reuse the existing asynchronous database worker.
        # Verify whether it preserves the nested audit field.
        try:
            if getattr(
                self.engine.args, "log_db", False
            ):
                db = dict(episode["start_record"])
                db["kind"] = "PASSIVE_FILLABILITY"
                db["readiness"] = record
                self.engine._queue(db)

        except Exception as exc:
            print(
                "FILL_DB_QUEUE_ERROR "
                + str(exc)[:120],
                flush=True,
            )

    def print_summary(self):
        chunks = [
            f"FILL_SUMMARY completed={self.completed}",
            f"active={len(self.active)}",
            f"fresh_starts={self.fresh_starts}",
        ]

        for target in self.HORIZONS_MS:
            c = self.counts[target]

            lost = (
                c["LOST_BEFORE"]
                + c["ORIGINAL_TOP_LOST_BEFORE"]
                + c["ORIGINAL_TOP_NOT_AVAILABLE"]
                + c["NET_NOT_POSITIVE"]
            )

            unknown = (
                c["INDETERMINATE"]
                + c["LATE_OBSERVATION"]
            )

            chunks.append(
                f"{target}ms_fresh="
                f"{c['TOP_HELD_FRESH_BOOK_PROXY']}"
                f"_old={c['TOP_HELD_OLD_BOOK_PROXY']}"
                f"_lost={lost}"
                f"_unknown={unknown}"
            )

        chunks.append(
            f"best_net_est={self.best_net_cents}c"
        )

        print(" ".join(chunks), flush=True)
