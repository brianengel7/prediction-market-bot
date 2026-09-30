import math
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from src.kalshi import client


def _retry_after_seconds(value):
    if not value:
        return 0.0

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = (
                deadline - datetime.now(timezone.utc)
            ).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0

    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


class _PacedRequests:
    def __init__(self, original, interval_seconds, max_attempts):
        self.original = original
        self.interval_seconds = interval_seconds
        self.max_attempts = max_attempts
        self.next_request_at = 0.0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get(self, *args, **kwargs):
        for attempt in range(self.max_attempts):
            wait = self.next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)

            self.next_request_at = (
                time.monotonic() + self.interval_seconds
            )
            response = self.original.get(*args, **kwargs)

            if response.status_code != 429:
                return response

            retry_after = _retry_after_seconds(
                response.headers.get("Retry-After")
            )
            response.close()

            if attempt + 1 == self.max_attempts:
                raise SystemExit(
                    "Kalshi kept returning HTTP 429 after "
                    f"{self.max_attempts} attempts. Backfill stopped; "
                    "rerun the same dates later."
                )

            wait = max(
                self.interval_seconds,
                min(60.0, 2.0 ** (attempt + 1)),
                retry_after,
            ) + random.uniform(0.0, 1.0)

            self.next_request_at = time.monotonic() + wait
            print(
                f"  HTTP 429: waiting {wait:.1f}s, then retrying "
                f"the same request ({attempt + 2}/{self.max_attempts})...",
                flush=True,
            )


@contextmanager
def hourly_backfill_requests(interval_seconds=0.25, max_attempts=8):
    original = client.requests
    client.requests = _PacedRequests(
        original, interval_seconds, max_attempts
    )
    try:
        yield
    finally:
        client.requests = original