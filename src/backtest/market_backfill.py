import time

import pandas as pd

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


if __name__ == "__main__":

    backfill_market_history(
        start_date=
            "2026-09-27",

        end_date=
            "2026-09-28",

        decision_hour=
            20,

        decision_minute=
            5
    )