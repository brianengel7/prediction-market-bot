import re
from datetime import datetime


def get_weather_target_date(event):
    """
    Extract the measured weather date from a
    KXHIGHNY event title.

    Example:
    Highest temperature in New York City on Sep 16, 2026?
    """

    title = event["title"]

    pattern = (
        r"\b("
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|"
        r"Sep|Oct|Nov|Dec"
        r")\s+"
        r"(\d{1,2}),\s+"
        r"(\d{4})"
    )

    match = re.search(
        pattern,
        title
    )

    if match is None:
        raise ValueError(
            f"Could not determine weather date "
            f"from event title: {title}"
        )

    date_string = match.group(0)

    target_date = datetime.strptime(
        date_string,
        "%b %d, %Y"
    ).date()

    return target_date.isoformat()