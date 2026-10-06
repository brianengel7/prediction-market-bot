import re
from dataclasses import dataclass


# ============================================================
# NFL TEAM ABBREVIATIONS
# ============================================================

NFL_TEAM_CODES = {

    "Arizona Cardinals":
        "ARI",

    "Atlanta Falcons":
        "ATL",

    "Baltimore Ravens":
        "BAL",

    "Buffalo Bills":
        "BUF",

    "Carolina Panthers":
        "CAR",

    "Chicago Bears":
        "CHI",

    "Cincinnati Bengals":
        "CIN",

    "Cleveland Browns":
        "CLE",

    "Dallas Cowboys":
        "DAL",

    "Denver Broncos":
        "DEN",

    "Detroit Lions":
        "DET",

    "Green Bay Packers":
        "GB",

    "Houston Texans":
        "HOU",

    "Indianapolis Colts":
        "IND",

    "Jacksonville Jaguars":
        "JAC",

    "Kansas City Chiefs":
        "KC",

    "Las Vegas Raiders":
        "LV",

    "Los Angeles Chargers":
        "LAC",

    "Los Angeles Rams":
        "LAR",

    "Miami Dolphins":
        "MIA",

    "Minnesota Vikings":
        "MIN",

    "New England Patriots":
        "NE",

    "New Orleans Saints":
        "NO",

    "New York Giants":
        "NYG",

    "New York Jets":
        "NYJ",

    "Philadelphia Eagles":
        "PHI",

    "Pittsburgh Steelers":
        "PIT",

    "San Francisco 49ers":
        "SF",

    "Seattle Seahawks":
        "SEA",

    "Tampa Bay Buccaneers":
        "TB",

    "Tennessee Titans":
        "TEN",

    "Washington Commanders":
        "WAS",
}


# ============================================================
# MATCH RESULT
# ============================================================

@dataclass
class MatchedMoneyline:
    sportsbook_event_id: str

    kalshi_event_ticker: str

    away_team: str
    home_team: str

    away_code: str
    home_code: str

    away_market: dict
    home_market: dict


# ============================================================
# TEAM NORMALIZATION
# ============================================================

def team_code(
    team_name: str,
) -> str:

    team_name = (
        str(
            team_name
        )
        .strip()
    )

    if (
        team_name
        not in
        NFL_TEAM_CODES
    ):

        raise ValueError(
            f"Unknown NFL team: "
            f"{team_name!r}"
        )

    return (
        NFL_TEAM_CODES[
            team_name
        ]
    )


# ============================================================
# PARSE KALSHI EVENT
# ============================================================

def parse_kalshi_nfl_event_ticker(
    event_ticker: str,
):
    """
    Example:

        KXNFLGAME-26OCT11CHIGB

    returns:

        date_code = 26OCT11
        team_blob = CHIGB

    We do not try to infer where one team code ends
    and the next begins here. Matching is done using
    the known home/away team codes from SportsGameOdds.
    """

    match = re.fullmatch(
        r"KXNFLGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]+)",
        str(
            event_ticker
        ),
    )

    if not match:

        raise ValueError(
            f"Unexpected Kalshi NFL "
            f"event ticker: "
            f"{event_ticker}"
        )

    return (
        match.group(1),
        match.group(2),
    )


# ============================================================
# MATCH ONE SPORTSBOOK EVENT
# ============================================================

def match_moneyline_event(
    sportsbook_event,
    kalshi_markets,
):
    """
    Match one SportsGameOdds NFL event to its Kalshi event.

    Matching currently uses the two team abbreviations.

    We accept either code ordering because Kalshi event
    tickers encode the matchup independently from the
    home/away representation used by the sportsbook feed.
    """

    away_code = (
        team_code(
            sportsbook_event.away_team
        )
    )

    home_code = (
        team_code(
            sportsbook_event.home_team
        )
    )

    by_event = {}

    for market in kalshi_markets:

        event_ticker = (
            market.get(
                "event_ticker"
            )
        )

        if not event_ticker:

            continue

        by_event.setdefault(
            event_ticker,
            [],
        ).append(
            market
        )

    matches = []

    for (
        event_ticker,
        markets,
    ) in by_event.items():

        try:

            _date_code, team_blob = (
                parse_kalshi_nfl_event_ticker(
                    event_ticker
                )
            )

        except ValueError:

            continue

        valid_blobs = {
            away_code
            +
            home_code,

            home_code
            +
            away_code,
        }

        if (
            team_blob
            not in
            valid_blobs
        ):

            continue

        matches.append(
            (
                event_ticker,
                markets,
            )
        )

    if not matches:

        return None

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple Kalshi events matched "
            f"{sportsbook_event.away_team} @ "
            f"{sportsbook_event.home_team}: "
            f"{[row[0] for row in matches]}"
        )

    (
        event_ticker,
        event_markets,
    ) = matches[0]

    away_market = None
    home_market = None

    for market in event_markets:

        ticker = str(
            market.get(
                "ticker",
                ""
            )
        )

        if ticker.endswith(
            "-"
            +
            away_code
        ):

            away_market = (
                market
            )

        elif ticker.endswith(
            "-"
            +
            home_code
        ):

            home_market = (
                market
            )

    if (
        away_market is None
        or
        home_market is None
    ):

        raise RuntimeError(
            f"Matched event "
            f"{event_ticker}, but could not "
            f"find both team contracts."
        )

    return MatchedMoneyline(

        sportsbook_event_id=
            sportsbook_event.event_id,

        kalshi_event_ticker=
            event_ticker,

        away_team=
            sportsbook_event.away_team,

        home_team=
            sportsbook_event.home_team,

        away_code=
            away_code,

        home_code=
            home_code,

        away_market=
            away_market,

        home_market=
            home_market,
    )


# ============================================================
# MATCH ALL EVENTS
# ============================================================

def match_moneyline_events(
    sportsbook_events,
    kalshi_markets,
):
    matched = []

    unmatched = []

    for event in sportsbook_events:

        result = (
            match_moneyline_event(
                sportsbook_event=
                    event,

                kalshi_markets=
                    kalshi_markets,
            )
        )

        if result is None:

            unmatched.append(
                event
            )

        else:

            matched.append(
                result
            )

    return (
        matched,
        unmatched,
    )