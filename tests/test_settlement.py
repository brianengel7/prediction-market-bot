from src.weather.settlement import (
    get_kxhighny_settlement_window
)


def test_summer_window():

    start, end = (
        get_kxhighny_settlement_window(
            "2026-07-15"
        )
    )

    assert start.hour == 1
    assert end.hour == 1


def test_winter_window():

    start, end = (
        get_kxhighny_settlement_window(
            "2026-01-15"
        )
    )

    assert start.hour == 0
    assert end.hour == 0