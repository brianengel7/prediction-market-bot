import time

import pandas as pd
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.kalshi.historical_market import (
    get_event_entry_quotes
)

from src.database.db import (
    save_historical_market_entries
)

NYC_TIMEZONE = ZoneInfo("America/New_York")


def backfill_market_history(
    start_date, end_date, decision_hour=20, decision_minute=5,
    calibration_quotes=False,
):
    saved_dates = skipped_dates = failed_dates = total_saved = 0

    for date in pd.date_range(start_date, end_date, freq="D"):
        target_date = date.strftime("%Y-%m-%d")
        print(f"Collecting {target_date}...", flush=True)

        try:
            entries = get_event_entry_quotes(
                target_date,
                decision_hour=decision_hour,
                decision_minute=decision_minute,
                calibration_quotes=calibration_quotes,
            )

            if not entries or (
                calibration_quotes and (
                    len(entries) != 6
                    or len({entry["ticker"] for entry in entries}) != 6
                )
            ):
                skipped_dates += 1
                print(f"  SKIPPED: {len(entries)} usable contract quotes.")
            else:
                saved = save_historical_market_entries(
                    entries, calibration_quotes=calibration_quotes,
                )
                saved_dates += 1
                total_saved += saved
                print(f"  Saved {saved} contracts.")

        except Exception as error:
            failed_dates += 1
            print(f"  FAILED: {type(error).__name__}: {error}")

        time.sleep(0.25)

    print(f"Dates saved: {saved_dates}")
    print(f"Dates skipped for quote coverage: {skipped_dates}")
    print(f"Dates failed: {failed_dates}")
    print(f"Contracts saved: {total_saved}")


def get_default_refresh_range(
    lookback_days=3
):
    """
    Refresh the most recent completed calendar days.

    We intentionally stop at yesterday because today's
    daily-high market may not be settled yet.
    """

    today = (
        datetime.now(
            NYC_TIMEZONE
        ).date()
    )

    end_date = (
        today
        -
        timedelta(
            days=1
        )
    )

    start_date = (
        end_date
        -
        timedelta(
            days=
                lookback_days
                - 1
        )
    )

    return (
        start_date.strftime(
            "%Y-%m-%d"
        ),
        end_date.strftime(
            "%Y-%m-%d"
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        type=str,
        default=None
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=3
    )

    args = (
        parser.parse_args()
    )

    # ----------------------------------------
    # Explicit date range if supplied
    # ----------------------------------------

    if (
        args.start_date is not None
        or
        args.end_date is not None
    ):

        if (
            args.start_date is None
            or
            args.end_date is None
        ):

            raise ValueError(
                "Provide both --start-date "
                "and --end-date."
            )

        start_date = (
            args.start_date
        )

        end_date = (
            args.end_date
        )

    # ----------------------------------------
    # Otherwise automatically refresh recent
    # completed dates
    # ----------------------------------------

    else:

        (
            start_date,
            end_date
        ) = (
            get_default_refresh_range(
                lookback_days=
                    args.lookback_days
            )
        )

    backfill_market_history(
        start_date=
            start_date,

        end_date=
            end_date,

        decision_hour=
            20,

        decision_minute=
            5
    )


if __name__ == "__main__":

    main()