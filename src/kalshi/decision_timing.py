import pandas as pd


DECISION_HOUR_UTC = 20
DECISION_MINUTE_UTC = 5

# For the frozen 20:05 UTC strategy, quotes must be
# captured no later than 20:07 UTC.
MAX_QUOTE_DELAY_MINUTES = 2


def validate_decision_quote_time(
    quote_time
):
    """
    Require the Kalshi quote snapshot used for a saved/live
    decision to have been captured within the frozen
    20:05 UTC decision window.
    """

    quote_time = pd.Timestamp(
        quote_time
    )

    if quote_time.tzinfo is None:

        quote_time = (
            quote_time.tz_localize(
                "UTC"
            )
        )

    else:

        quote_time = (
            quote_time.tz_convert(
                "UTC"
            )
        )

    decision_start = (
        quote_time.normalize()
        +
        pd.Timedelta(
            hours=
                DECISION_HOUR_UTC,

            minutes=
                DECISION_MINUTE_UTC
        )
    )

    decision_end = (
        decision_start
        +
        pd.Timedelta(
            minutes=
                MAX_QUOTE_DELAY_MINUTES
        )
    )

    if not (
        decision_start
        <=
        quote_time
        <=
        decision_end
    ):

        raise RuntimeError(
            "Decision rejected: Kalshi quotes "
            "were not captured inside the "
            "frozen decision window.\n"
            f"Quote time: {quote_time}\n"
            f"Required window: "
            f"{decision_start} -> "
            f"{decision_end}"
        )

    return quote_time