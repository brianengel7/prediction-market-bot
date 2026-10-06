from src.kalshi.client import (
    get_markets,
)

from src.kalshi.markets import (
    parse_markets,
)


SERIES_TICKER = "KXNFLGAME"


def get_open_nfl_moneylines():
    """
    Retrieve all currently open Kalshi NFL
    game-winner contracts.

    Uses the existing shared Kalshi infrastructure.
    """

    raw = (
        get_markets(
            series_ticker=
                SERIES_TICKER,

            status=
                "open",
        )
    )

    markets = (
        parse_markets(
            raw
        )
    )

    # Fail loudly if the Kalshi endpoint ever
    # returns something outside the requested series.
    invalid = [
        market
        for market in markets
        if not str(
            market.get(
                "ticker",
                "",
            )
        ).startswith(
            SERIES_TICKER
            +
            "-"
        )
    ]

    if invalid:

        raise RuntimeError(
            "Kalshi returned markets outside "
            f"{SERIES_TICKER}."
        )

    return sorted(
        markets,
        key=lambda market: (
            str(
                market.get(
                    "event_ticker",
                    "",
                )
            ),
            str(
                market.get(
                    "ticker",
                    "",
                )
            ),
        ),
    )


def format_price(
    value,
):
    if value is None:
        return "-"

    try:
        return (
            f"{float(value):.3f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )


def main():

    markets = (
        get_open_nfl_moneylines()
    )

    event_tickers = sorted({
        market.get(
            "event_ticker"
        )
        for market in markets
    })

    print()
    print("=" * 100)
    print(
        "OPEN KALSHI NFL MONEYLINE MARKETS"
    )
    print("=" * 100)

    print(
        f"Series:        "
        f"{SERIES_TICKER}"
    )

    print(
        f"Events:        "
        f"{len(event_tickers)}"
    )

    print(
        f"Contracts:     "
        f"{len(markets)}"
    )

    print()

    current_event = None

    for market in markets:

        event_ticker = (
            market.get(
                "event_ticker"
            )
        )

        if (
            event_ticker
            !=
            current_event
        ):

            if current_event is not None:

                print()

            print("-" * 100)

            print(
                f"EVENT: "
                f"{event_ticker}"
            )

            current_event = (
                event_ticker
            )

        ticker = (
            market.get(
                "ticker"
            )
        )

        title = (
            market.get(
                "title"
            )
            or
            market.get(
                "subtitle"
            )
            or
            ""
        )

        yes_bid = (
            market.get(
                "yes_bid"
            )
        )

        yes_ask = (
            market.get(
                "yes_ask"
            )
        )

        no_bid = (
            market.get(
                "no_bid"
            )
        )

        no_ask = (
            market.get(
                "no_ask"
            )
        )

        print(
            f"{ticker}"
        )

        print(
            f"  title:   "
            f"{title}"
        )

        print(
            f"  YES:     "
            f"{format_price(yes_bid)}"
            f" / "
            f"{format_price(yes_ask)}"
        )

        print(
            f"  NO:      "
            f"{format_price(no_bid)}"
            f" / "
            f"{format_price(no_ask)}"
        )

    print()
    print("-" * 100)


if __name__ == "__main__":

    main()