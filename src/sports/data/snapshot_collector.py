import argparse

import pandas as pd

from src.kalshi.client import (
    get_event,
    get_series,
)

from src.sports.data.kalshi_moneylines import (
    SERIES_TICKER,
    get_open_nfl_moneylines,
)

from src.sports.data.sportsbook_odds import (
    get_moneyline_events,
)

from src.sports.data.sports_db import (
    get_sports_snapshot_counts,
    save_sports_snapshot,
)

from src.sports.matching.market_matcher import (
    match_moneyline_events,
)

from src.sports.pricing.live_consensus import (
    build_event_consensus,
)

from src.sports.pricing.odds_math import (
    remove_two_way_vig,
    sportsbook_hold,
)


# ============================================================
# TIME
# ============================================================

def normalize_time(
    value,
):

    if value in (
        None,
        "",
    ):

        return None

    timestamp = (
        pd.Timestamp(
            value
        )
    )

    if timestamp.tzinfo is None:

        timestamp = (
            timestamp.tz_localize(
                "UTC"
            )
        )

    else:

        timestamp = (
            timestamp.tz_convert(
                "UTC"
            )
        )

    return (
        timestamp.isoformat()
    )


# ============================================================
# FEE METADATA
# ============================================================

def get_fee_metadata(
    event_ticker,
    series_info,
):

    event_info = (
        get_event(
            event_ticker
        )
    )

    fee_type = (
        event_info.get(
            "fee_type_override"
        )
        or
        series_info.get(
            "fee_type"
        )
    )

    override = (
        event_info.get(
            "fee_multiplier_override"
        )
    )

    if override is None:

        multiplier = (
            series_info.get(
                "fee_multiplier"
            )
        )

    else:

        multiplier = (
            override
        )

    multiplier = (
        None
        if multiplier is None
        else float(
            multiplier
        )
    )

    return (
        fee_type,
        multiplier,
    )


# ============================================================
# COLLECT
# ============================================================

def collect_snapshot(
    window_hours=4,
    all_open=False,
):

    now = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

    # One logical timestamp for both feeds.
    # Re-running within the same minute updates rather
    # than creates a duplicate snapshot.
    snapshot_time = (
        now.floor(
            "min"
        ).isoformat()
    )

    collected_at = (
        now.isoformat()
    )

    # --------------------------------------------------------
    # Limit SportsGameOdds requests to games near kickoff.
    # --------------------------------------------------------

    if all_open:

        starts_after = None
        starts_before = None

    else:

        starts_after = (
            now.isoformat()
        )

        starts_before = (
            now
            +
            pd.Timedelta(
                hours=
                    window_hours
            )
        ).isoformat()

    sportsbook_events = (
        get_moneyline_events(

            league=
                "NFL",

            starts_after=
                starts_after,

            starts_before=
                starts_before,
        )
    )

    if not sportsbook_events:

        print(
            "No NFL moneyline events "
            "inside collection window."
        )

        return {
            "saved_events":
                0,

            "saved_quotes":
                0,
        }

    # --------------------------------------------------------
    # Kalshi board
    # --------------------------------------------------------

    kalshi_markets = (
        get_open_nfl_moneylines()
    )

    (
        matches,
        unmatched,
    ) = (
        match_moneyline_events(

            sportsbook_events=
                sportsbook_events,

            kalshi_markets=
                kalshi_markets,
        )
    )

    sportsbook_by_id = {
        event.event_id:
            event
        for event
        in sportsbook_events
    }

    series_info = (
        get_series(
            SERIES_TICKER
        )
    )

    saved_events = 0
    saved_quotes = 0

    print()
    print("=" * 100)
    print(
        "NFL SPORTS SNAPSHOT COLLECTION"
    )
    print("=" * 100)

    print(
        f"Snapshot time:       "
        f"{snapshot_time}"
    )

    print(
        f"Sportsbook events:   "
        f"{len(sportsbook_events)}"
    )

    print(
        f"Matched events:      "
        f"{len(matches)}"
    )

    print(
        f"Unmatched events:    "
        f"{len(unmatched)}"
    )

    print()

    # --------------------------------------------------------
    # Save each matched event
    # --------------------------------------------------------

    for match in matches:

        event = (
            sportsbook_by_id[
                match.sportsbook_event_id
            ]
        )

        consensus = (
            build_event_consensus(
                event
            )
        )

        if consensus is None:

            continue

        (
            fee_type,
            fee_multiplier,
        ) = get_fee_metadata(

            event_ticker=
                match.kalshi_event_ticker,

            series_info=
                series_info,
        )

        event_row = {

            "snapshot_time":
                snapshot_time,

            "collected_at":
                collected_at,

            "league":
                "NFL",

            "sportsbook_event_id":
                event.event_id,

            "kalshi_event_ticker":
                match.kalshi_event_ticker,

            "kickoff_time":
                normalize_time(
                    event.commence_time
                ),

            "away_team":
                match.away_team,

            "home_team":
                match.home_team,

            "away_code":
                match.away_code,

            "home_code":
                match.home_code,

            "sportsbook_count":
                consensus.sportsbook_count,

            "away_consensus_probability":
                consensus.team_b_probability,

            "home_consensus_probability":
                consensus.team_a_probability,

            "away_probability_min":
                (
                    1.0
                    -
                    consensus.team_a_max_probability
                ),

            "away_probability_max":
                (
                    1.0
                    -
                    consensus.team_a_min_probability
                ),

            "home_probability_min":
                consensus.team_a_min_probability,

            "home_probability_max":
                consensus.team_a_max_probability,

            "probability_range":
                consensus.team_a_probability_range,

            "fee_type":
                fee_type,

            "fee_multiplier":
                fee_multiplier,

            "away_market_ticker":
                match.away_market[
                    "ticker"
                ],

            "away_yes_bid":
                match.away_market.get(
                    "yes_bid"
                ),

            "away_yes_ask":
                match.away_market.get(
                    "yes_ask"
                ),

            "away_no_bid":
                match.away_market.get(
                    "no_bid"
                ),

            "away_no_ask":
                match.away_market.get(
                    "no_ask"
                ),

            "home_market_ticker":
                match.home_market[
                    "ticker"
                ],

            "home_yes_bid":
                match.home_market.get(
                    "yes_bid"
                ),

            "home_yes_ask":
                match.home_market.get(
                    "yes_ask"
                ),

            "home_no_bid":
                match.home_market.get(
                    "no_bid"
                ),

            "home_no_ask":
                match.home_market.get(
                    "no_ask"
                ),
        }

        sportsbook_rows = []

        for book in event.books:

            (
                home_fair,
                away_fair,
            ) = remove_two_way_vig(

                book.home_odds,
                book.away_odds,
            )

            hold = (
                sportsbook_hold(
                    book.home_odds,
                    book.away_odds,
                )
            )

            sportsbook_rows.append({

                "snapshot_time":
                    snapshot_time,

                "collected_at":
                    collected_at,

                "sportsbook_event_id":
                    event.event_id,

                "sportsbook":
                    book.sportsbook,

                "away_odds":
                    float(
                        book.away_odds
                    ),

                "home_odds":
                    float(
                        book.home_odds
                    ),

                "away_fair_probability":
                    away_fair,

                "home_fair_probability":
                    home_fair,

                "sportsbook_hold":
                    hold,
            })

        save_sports_snapshot(

            event_row=
                event_row,

            sportsbook_rows=
                sportsbook_rows,
        )

        saved_events += 1

        saved_quotes += (
            len(
                sportsbook_rows
            )
        )

        print(
            f"SAVED "
            f"{match.away_team} @ "
            f"{match.home_team}"
        )

        print(
            f"  kickoff: "
            f"{event_row['kickoff_time']}"
        )

        print(
            f"  books:   "
            f"{len(sportsbook_rows)}"
        )

        print(
            f"  away:    "
            f"{consensus.team_b_probability:.2%}"
        )

        print(
            f"  home:    "
            f"{consensus.team_a_probability:.2%}"
        )

        print()

    # --------------------------------------------------------
    # Read-back
    # --------------------------------------------------------

    counts = (
        get_sports_snapshot_counts()
    )

    print("=" * 100)

    print(
        f"Saved event snapshots this run: "
        f"{saved_events}"
    )

    print(
        f"Saved sportsbook quotes this run: "
        f"{saved_quotes}"
    )

    print(
        f"Database event snapshots: "
        f"{counts['event_snapshots']}"
    )

    print(
        f"Database sportsbook quotes: "
        f"{counts['sportsbook_quotes']}"
    )

    if unmatched:

        print()
        print(
            "UNMATCHED:"
        )

        for event in unmatched:

            print(
                f"  "
                f"{event.away_team} @ "
                f"{event.home_team}"
            )

    return {
        "saved_events":
            saved_events,

        "saved_quotes":
            saved_quotes,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=4.0,
        help=(
            "Collect games starting within this many hours. "
            "Default: 4."
        ),
    )

    parser.add_argument(
        "--all-open",
        action="store_true",
        help=(
            "Diagnostic mode: collect every open NFL event "
            "returned by SportsGameOdds."
        ),
    )

    args = (
        parser.parse_args()
    )

    if (
        args.window_hours
        <=
        0
    ):

        parser.error(
            "--window-hours must be positive."
        )

    collect_snapshot(

        window_hours=
            args.window_hours,

        all_open=
            args.all_open,
    )


if __name__ == "__main__":

    main()