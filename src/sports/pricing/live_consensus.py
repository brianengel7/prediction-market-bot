from src.sports.data.sportsbook_odds import (
    get_moneyline_events,
)

from src.sports.pricing.consensus import (
    SportsbookQuote,
    build_moneyline_consensus,
)


MINIMUM_SPORTSBOOKS = 3


def build_event_consensus(
    event,
):
    """
    Convert sportsbook moneylines for one event into
    a no-vig consensus probability.

    Home team = team A
    Away team = team B
    """

    quotes = []

    for book in event.books:

        quotes.append(
            SportsbookQuote(
                sportsbook=
                    book.sportsbook,

                team_a_odds=
                    book.home_odds,

                team_b_odds=
                    book.away_odds,
            )
        )

    if (
        len(quotes)
        <
        MINIMUM_SPORTSBOOKS
    ):

        return None

    consensus = (
        build_moneyline_consensus(
            quotes
        )
    )

    return consensus


def main():

    events = (
        get_moneyline_events(
            league="NFL"
        )
    )

    print()
    print("=" * 90)
    print(
        "NFL SPORTSBOOK MONEYLINE CONSENSUS"
    )
    print("=" * 90)

    print(
        f"Events found:        "
        f"{len(events)}"
    )

    print(
        f"Minimum books:       "
        f"{MINIMUM_SPORTSBOOKS}"
    )

    print()

    for event in events:

        consensus = (
            build_event_consensus(
                event
            )
        )

        if consensus is None:

            continue

        print("-" * 90)

        print(
            f"{event.away_team} "
            f"@ "
            f"{event.home_team}"
        )

        print(
            f"Event ID: "
            f"{event.event_id}"
        )

        print(
            f"Sportsbooks:         "
            f"{consensus.sportsbook_count}"
        )

        print()

        print(
            f"{event.home_team}: "
            f"{consensus.team_a_probability:.2%}"
        )

        print(
            f"{event.away_team}: "
            f"{consensus.team_b_probability:.2%}"
        )

        print()

        print(
            f"Home probability range: "
            f"{consensus.team_a_min_probability:.2%}"
            f" → "
            f"{consensus.team_a_max_probability:.2%}"
        )

        print(
            f"Book disagreement:      "
            f"{consensus.team_a_probability_range:.2%}"
        )

    print("-" * 90)


if __name__ == "__main__":

    main()