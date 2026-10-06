import os
from dataclasses import dataclass

import requests


BASE_URL = (
    "https://api.sportsgameodds.com/v2"
)

HOME_MONEYLINE_ODD_ID = (
    "points-home-game-ml-home"
)

AWAY_MONEYLINE_ODD_ID = (
    "points-away-game-ml-away"
)


@dataclass
class BookMoneyline:
    sportsbook: str
    home_odds: float
    away_odds: float


@dataclass
class MoneylineEvent:
    event_id: str
    league: str
    commence_time: str | None
    home_team: str
    away_team: str
    books: list[BookMoneyline]


def get_api_key() -> str:

    api_key = (
        os.getenv(
            "SPORTSGAMEODDS_API_KEY"
        )
    )

    if not api_key:

        raise RuntimeError(
            "SPORTSGAMEODDS_API_KEY "
            "is not set."
        )

    return api_key


def _parse_american_odds(
    value,
) -> float | None:

    if value is None:
        return None

    try:

        return float(
            str(value)
            .replace(
                "+",
                ""
            )
            .strip()
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def fetch_events(
    league: str = "NFL",
    starts_after: str | None = None,
    starts_before: str | None = None,
):

    params = {
        "leagueID":
            league,

        "oddsAvailable":
            "true",

        "type":
            "match",

        "oddID":
            (
                f"{HOME_MONEYLINE_ODD_ID},"
                f"{AWAY_MONEYLINE_ODD_ID}"
            ),
    }

    if starts_after is not None:

        params[
            "startsAfter"
        ] = starts_after

    if starts_before is not None:

        params[
            "startsBefore"
        ] = starts_before

    response = requests.get(

        f"{BASE_URL}/events",

        headers={
            "x-api-key":
                get_api_key(),
        },

        params=
            params,

        timeout=30,
    )

    response.raise_for_status()

    payload = (
        response.json()
    )

    if not payload.get(
        "success",
        False,
    ):

        raise RuntimeError(
            f"SportsGameOdds API error: "
            f"{payload.get('error')}"
        )

    return (
        payload.get(
            "data",
            []
        )
    )


def _team_name(
    event,
    side,
) -> str:

    team = (
        event.get(
            "teams",
            {}
        )
        .get(
            side,
            {}
        )
    )

    names = (
        team.get(
            "names",
            {}
        )
    )

    return (
        names.get(
            "long"
        )
        or
        names.get(
            "medium"
        )
        or
        names.get(
            "short"
        )
        or
        team.get(
            "teamID"
        )
        or
        side
    )


def parse_moneyline_event(
    event,
) -> MoneylineEvent | None:

    odds = (
        event.get(
            "odds",
            {}
        )
    )

    home_market = (
        odds.get(
            HOME_MONEYLINE_ODD_ID
        )
    )

    away_market = (
        odds.get(
            AWAY_MONEYLINE_ODD_ID
        )
    )

    if (
        not home_market
        or
        not away_market
    ):

        return None

    home_books = (
        home_market.get(
            "byBookmaker",
            {}
        )
    )

    away_books = (
        away_market.get(
            "byBookmaker",
            {}
        )
    )

    sportsbooks = sorted(
        set(
            home_books
        )
        &
        set(
            away_books
        )
    )

    books = []

    for sportsbook in sportsbooks:

        home_quote = (
            home_books[
                sportsbook
            ]
        )

        away_quote = (
            away_books[
                sportsbook
            ]
        )

        if not home_quote.get(
            "available",
            True,
        ):

            continue

        if not away_quote.get(
            "available",
            True,
        ):

            continue

        home_odds = (
            _parse_american_odds(
                home_quote.get(
                    "odds"
                )
            )
        )

        away_odds = (
            _parse_american_odds(
                away_quote.get(
                    "odds"
                )
            )
        )

        if (
            home_odds is None
            or
            away_odds is None
        ):

            continue

        books.append(
            BookMoneyline(

                sportsbook=
                    sportsbook,

                home_odds=
                    home_odds,

                away_odds=
                    away_odds,
            )
        )

    if not books:

        return None

    return MoneylineEvent(

        event_id=
            str(
                event.get(
                    "eventID",
                    ""
                )
            ),

        league=
            str(
                event.get(
                    "leagueID",
                    ""
                )
            ),

        commence_time=(
            event.get(
                "startTime"
            )
            or
            event.get(
                "status",
                {}
            ).get(
                "startsAt"
            )
        ),

        home_team=
            _team_name(
                event,
                "home",
            ),

        away_team=
            _team_name(
                event,
                "away",
            ),

        books=
            books,
    )


def get_moneyline_events(
    league: str = "NFL",
    starts_after: str | None = None,
    starts_before: str | None = None,
) -> list[MoneylineEvent]:

    raw_events = (
        fetch_events(
            league=
                league,

            starts_after=
                starts_after,

            starts_before=
                starts_before,
        )
    )

    parsed = []

    for event in raw_events:

        moneyline_event = (
            parse_moneyline_event(
                event
            )
        )

        if moneyline_event:

            parsed.append(
                moneyline_event
            )

    return parsed


def main():

    events = (
        get_moneyline_events(
            league="NFL"
        )
    )

    print(
        f"Moneyline events: "
        f"{len(events)}"
    )

    print()

    for event in events:

        print(
            f"{event.away_team} "
            f"@ "
            f"{event.home_team}"
        )

        print(
            f"Event ID: "
            f"{event.event_id}"
        )

        for book in (
            event.books
        ):

            print(
                f"  "
                f"{book.sportsbook:15s} "
                f"away "
                f"{book.away_odds:+.0f} | "
                f"home "
                f"{book.home_odds:+.0f}"
            )

        print()


if __name__ == "__main__":

    main()