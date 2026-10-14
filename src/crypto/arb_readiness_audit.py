"""Event-driven, read-only cross-exchange quote-survival audit."""

import json
import time
from datetime import datetime, timezone
from decimal import Decimal


class ExecutionReadinessAudit:
    TARGETS_MS = (50, 100, 250)
    ON_TIME_WINDOW_MS = 25
    FRESH_PROXY_MAX_MS = 200
    CENSOR_MS = 5000

    def __init__(self, engine, price_fn):
        self.engine = engine
        self.price_fn = price_fn
        self.active = {}
        self.suppressed = set()
        self.top_since = {}
        self.next_id = 0
        self.completed = 0
        self.fresh_starts = 0
        self.best_net = None
        self.counts = {
            t: {
                "ON_TIME_POSITIVE": 0,
                "LATE_POSITIVE": 0,
                "LOST_BEFORE": 0,
                "INDETERMINATE": 0,
            }
            for t in self.TARGETS_MS
        }

    def _ages(self):
        state = self.engine.state
        now = time.monotonic()

        kb = (
            state.books.get(state.kalshi_btc)
            if state.kalshi_btc else None
        )
        pb = state.poly if state.poly_slug else None

        def age(book):
            if book is None or not getattr(book, "ready", False):
                return None

            seen = getattr(book, "seen", None)
            if seen is None:
                return None

            return round(
                max(0, now - seen) * 1000, 2
            )

        return {
            "kalshi_ms": age(kb),
            "poly_ms": age(pb),
        }

    def _top_age(self, legs, now_ns):
        result = {}

        for leg in legs or []:
            if leg is None:
                continue

            key = (leg.venue, leg.ticker, leg.side)
            signature = (
                str(leg.price),
                str(leg.size),
            )

            previous = self.top_since.get(key)

            if previous is None or previous[0] != signature:
                previous = (signature, now_ns)
                self.top_since[key] = previous

            result[f"{leg.venue}_{leg.side}"] = round(
                (now_ns - previous[1]) / 1e6, 2
            )

        return result

    def _start(self, key, row, ages, tops, now_ns):
        self.next_id += 1

        fresh = all(
            age is not None
            and age <= self.FRESH_PROXY_MAX_MS
            for age in ages.values()
        )

        self.fresh_starts += int(fresh)

        episode = {
            "id": self.next_id,
            "direction": key,
            "started_at": row["at"],
            "started_ns": now_ns,
            "start_record": dict(row),
            "start_book_age_ms": ages,
            "start_top_age_ms": tops,
            "fresh_start_proxy": fresh,
            "initial_quantity": row["evaluated_pairs"],
            "start_legs": row["legs"],
            "best_net_est_cents": row[
                "buffered_cents_per_pair"
            ],
            "best_legs": row["legs"],
            "last_positive_elapsed_ms": 0.0,
            "positive_observations": 1,
            "max_observation_gap_ms": 0.0,
            "last_observed_ns": now_ns,
            "checkpoints": {},
        }

        self.active[key] = episode

        print(
            f"READINESS_START id={episode['id']} "
            f"{key} "
            f"net={episode['best_net_est_cents']}c "
            f"qty={episode['initial_quantity']} "
            f"K_age={ages['kalshi_ms']}ms "
            f"P_age={ages['poly_ms']}ms "
            f"fresh_proxy={fresh}",
            flush=True,
        )

    def _checkpoint(
        self, episode, target, row,
        ages, tops, elapsed_ms
    ):
        lateness = round(elapsed_ms - target, 2)

        result = (
            "ON_TIME_POSITIVE"
            if lateness <= self.ON_TIME_WINDOW_MS
            else "LATE_POSITIVE"
        )

        episode["checkpoints"][str(target)] = {
            "result": result,
            "target_ms": target,
            "observed_at_ms": round(elapsed_ms, 2),
            "lateness_ms": lateness,
            "book_age_proxy_ms": ages,
            "fresh_quote_proxy": all(
                age is not None
                and age <= self.FRESH_PROXY_MAX_MS
                for age in ages.values()
            ),
            "gross_cents": row[
                "gross_cents_per_pair"
            ],
            "net_est_cents": row[
                "buffered_cents_per_pair"
            ],
            "evaluated_pairs": row[
                "evaluated_pairs"
            ],
            "legs": row["legs"],
            "top_age_lower_bound_ms": tops,
        }

    def observe(self, key, kind, legs, row):
        if kind != "CROSS_EXCHANGE":
            return

        now_ns = time.monotonic_ns()
        ages = self._ages()
        tops = self._top_age(legs, now_ns)

        positive = (
            row is not None
            and Decimal(
                row["buffered_cents_per_pair"]
            ) > 0
        )

        episode = self.active.get(key)

        if episode is None:
            if not positive:
                self.suppressed.discard(key)
                return

            if (
                key not in self.suppressed
                and all(
                    v is not None
                    for v in ages.values()
                )
            ):
                self._start(
                    key, row, ages, tops, now_ns
                )

            return

        elapsed_ms = (
            now_ns - episode["started_ns"]
        ) / 1e6

        gap_ms = (
            now_ns - episode["last_observed_ns"]
        ) / 1e6

        episode["max_observation_gap_ms"] = max(
            episode["max_observation_gap_ms"],
            round(gap_ms, 2),
        )

        episode["last_observed_ns"] = now_ns

        if not positive:
            self._end(
                episode,
                now_ns,
                "NONPOSITIVE_OR_NO_FRESH_PAIR",
            )
            self.suppressed.discard(key)
            return

        if (
            Decimal(row["evaluated_pairs"])
            < Decimal(episode["initial_quantity"])
        ):
            self._end(
                episode,
                now_ns,
                "ORIGINAL_DEPTH_LOST",
            )
            self._start(
                key, row, ages, tops, now_ns
            )
            return

        if elapsed_ms >= self.CENSOR_MS:
            self._end(
                episode,
                now_ns,
                "CENSORED_AT_5_SECONDS",
            )
            self.suppressed.add(key)
            return

        episode["positive_observations"] += 1
        episode["last_positive_elapsed_ms"] = round(
            elapsed_ms, 2
        )

        net = Decimal(
            row["buffered_cents_per_pair"]
        )

        if net > Decimal(
            episode["best_net_est_cents"]
        ):
            episode["best_net_est_cents"] = (
                row["buffered_cents_per_pair"]
            )
            episode["best_legs"] = row["legs"]

        for target in self.TARGETS_MS:
            if (
                elapsed_ms >= target
                and str(target)
                not in episode["checkpoints"]
            ):
                self._checkpoint(
                    episode,
                    target,
                    row,
                    ages,
                    tops,
                    elapsed_ms,
                )

    def _end(self, episode, now_ns, reason):
        key = episode["direction"]

        if self.active.get(key) is not episode:
            return

        self.active.pop(key, None)

        end_ms = round(
            (
                now_ns - episode["started_ns"]
            ) / 1e6,
            2,
        )

        results = {}

        for target in self.TARGETS_MS:
            sample = episode[
                "checkpoints"
            ].get(str(target))

            if sample:
                status = sample["result"]

            elif reason.startswith("CENSORED"):
                status = "INDETERMINATE"

            elif end_ms < target:
                status = "LOST_BEFORE"

            else:
                status = "INDETERMINATE"

            self.counts[target][status] += 1
            results[str(target)] = status

        record = {
            "episode_id": episode["id"],
            "direction": key,
            "started_at": episode["started_at"],
            "ended_at": datetime.now(
                timezone.utc
            ).isoformat(timespec="milliseconds"),
            "first_not_positive_observed_ms": end_ms,
            "last_positive_observed_ms": episode[
                "last_positive_elapsed_ms"
            ],
            "end_reason": reason,
            "start_book_age_proxy_ms": episode[
                "start_book_age_ms"
            ],
            "start_top_age_lower_bound_ms": episode[
                "start_top_age_ms"
            ],
            "start_fresh_quote_proxy": episode[
                "fresh_start_proxy"
            ],
            "initial_quantity": episode[
                "initial_quantity"
            ],
            "start_prices": episode[
                "start_legs"
            ],
            "best_net_est_cents": episode[
                "best_net_est_cents"
            ],
            "best_prices": episode[
                "best_legs"
            ],
            "positive_observations": episode[
                "positive_observations"
            ],
            "max_gap_between_local_checks_ms": episode[
                "max_observation_gap_ms"
            ],
            "horizon_results": results,
            "horizon_observations": episode[
                "checkpoints"
            ],
            "times_are_local_receipt_proxies": True,
            "execution_verified": False,
            "settlement_equivalence_verified": False,
            "orders_submitted": 0,
        }

        self.completed += 1

        best = Decimal(
            episode["best_net_est_cents"]
        )

        if self.best_net is None or best > self.best_net:
            self.best_net = best

        print(
            f"READINESS_END id={episode['id']} "
            f"{key} "
            f"positive_until~"
            f"{episode['last_positive_elapsed_ms']:.1f}ms "
            f"first_fail~{end_ms:.1f}ms "
            f"best_net={best:+.3f}c "
            f"50ms={results['50']} "
            f"100ms={results['100']} "
            f"250ms={results['250']} "
            f"max_check_gap="
            f"{episode['max_observation_gap_ms']:.1f}ms "
            f"start_fresh={episode['fresh_start_proxy']}",
            flush=True,
        )

        print(
            "READINESS_JSON "
            + json.dumps(
                record,
                separators=(",", ":"),
            ),
            flush=True,
        )

        db_record = dict(
            episode["start_record"]
        )
        db_record["kind"] = "EXECUTION_READINESS"
        db_record["readiness"] = record

        try:
            self.engine._queue(db_record)
        except Exception as exc:
            print(
                "READINESS_DB_QUEUE_ERROR "
                + str(exc)[:160],
                flush=True,
            )

    def print_summary(self):
        parts = [
            f"READINESS_SUMMARY_V2 "
            f"completed={self.completed}",
            f"active={len(self.active)}",
            f"fresh_start_le_200ms={self.fresh_starts}",
        ]

        for target in self.TARGETS_MS:
            counts = self.counts[target]

            parts.append(
                f"at_{target}ms="
                f"on_time:"
                f"{counts['ON_TIME_POSITIVE']},"
                f"late:"
                f"{counts['LATE_POSITIVE']},"
                f"lost_before:"
                f"{counts['LOST_BEFORE']},"
                f"unknown:"
                f"{counts['INDETERMINATE']}"
            )

        parts.append(
            f"best_net_est={self.best_net}c"
        )

        print(" ".join(parts), flush=True)
