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


def backfill_market_history(
    start_date,
    end_date,
    decision_hour=20,
    decision_minute=5
):

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    total_saved = 0
    successful_dates = 0
    failed_dates = 0

    print()
    print("=" * 80)
    print(
        "KALSHI HISTORICAL MARKET BACKFILL"
    )
    print("=" * 80)

    print(
        f"Start: {start_date}"
    )

    print(
        f"End:   {end_date}"
    )

    print(
        f"Days:  {len(dates)}"
    )

    print()

    for index, date in enumerate(
        dates,
        start=1
    ):

        target_date = (
            date.strftime(
                "%Y-%m-%d"
            )
        )

        print(
            f"[{index}/{len(dates)}] "
            f"{target_date}"
        )

        try:

            entries = (
                get_event_entry_quotes(
                    target_date,
                    decision_hour=
                        decision_hour,
                    decision_minute=
                        decision_minute
                )
            )

            saved = (
                save_historical_market_entries(
                    entries
                )
            )

            total_saved += saved

            successful_dates += 1

            print(
                f"  Saved "
                f"{saved} contracts."
            )

        except Exception as error:

            failed_dates += 1

            print(
                f"  FAILED: "
                f"{type(error).__name__}: "
                f"{error}"
            )

        # Small pause so we don't hammer
        # Kalshi during the bulk download.
        time.sleep(
            0.25
        )

    print()
    print("=" * 80)
    print(
        "BACKFILL COMPLETE"
    )
    print("=" * 80)

    print(
        f"Successful dates: "
        f"{successful_dates}"
    )

    print(
        f"Failed dates:     "
        f"{failed_dates}"
    )

    print(
        f"Contracts saved:  "
        f"{total_saved}"
    )


NYC_TIMEZONE = ZoneInfo(
    "America/New_York"
)


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