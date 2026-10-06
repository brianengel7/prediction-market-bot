import pytest

from src.sports.matching.market_matcher import (
    parse_kalshi_nfl_event_ticker,
    team_code,
)


def test_team_code():

    assert (
        team_code(
            "Dallas Cowboys"
        )
        ==
        "DAL"
    )

    assert (
        team_code(
            "Tampa Bay Buccaneers"
        )
        ==
        "TB"
    )


def test_unknown_team():

    with pytest.raises(
        ValueError
    ):

        team_code(
            "Fake NFL Team"
        )


def test_parse_event_ticker():

    date_code, team_blob = (
        parse_kalshi_nfl_event_ticker(
            "KXNFLGAME-26OCT11CHIGB"
        )
    )

    assert (
        date_code
        ==
        "26OCT11"
    )

    assert (
        team_blob
        ==
        "CHIGB"
    )


def test_parse_bad_event_ticker():

    with pytest.raises(
        ValueError
    ):

        parse_kalshi_nfl_event_ticker(
            "NOT-A-KALSHI-NFL-EVENT"
        )