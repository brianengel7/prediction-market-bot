from datetime import (
    datetime,
    time,
    timedelta,
    timezone
)

import pandas as pd
from zoneinfo import ZoneInfo


NYC_TIMEZONE = ZoneInfo(
    "America/New_York"
)

# NWS climate reports use local STANDARD time.
NYC_STANDARD_TIME = timezone(
    timedelta(hours=-5)
)


def get_kxhighny_settlement_window(
    target_date
):
    """
    Return the exact KXHIGHNY climate-day window.

    NWS climate reports use local standard time.

    During EDT this becomes:
        1:00 AM -> 1:00 AM next day

    During EST this becomes:
        12:00 AM -> 12:00 AM next day

    End is exclusive.
    """

    target_date = pd.Timestamp(
        target_date
    ).date()

    next_date = (
        target_date
        + timedelta(days=1)
    )

    start_standard = pd.Timestamp(
        datetime.combine(
            target_date,
            time.min
        ),
        tz=NYC_STANDARD_TIME
    )

    end_standard = pd.Timestamp(
        datetime.combine(
            next_date,
            time.min
        ),
        tz=NYC_STANDARD_TIME
    )

    start_local = (
        start_standard.tz_convert(
            NYC_TIMEZONE
        )
    )

    end_local = (
        end_standard.tz_convert(
            NYC_TIMEZONE
        )
    )

    return start_local, end_local