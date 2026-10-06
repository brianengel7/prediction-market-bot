import argparse

from src.kalshi.client import (
    get_event,
    get_series,
)

from src.kalshi.fees import (
    calculate_taker_fee,
)

from src.sports.data.sportsbook_odds import (
    get_moneyline_events,
)

from src.sports.data.kalshi_moneylines import (
    get_open_nfl_moneylines,
    SERIES_TICKER,
)

from src.sports.matching.market_matcher import (
    match_moneyline_events,
)

from src.sports.pricing.live_consensus import (
    build_event_consensus,
)


# ============================================================
# FAIR VALUE
# ============================================================

def kalshi_fair_value(
    no_tie_win_probability,
    tie_probability,
):
    """
    Sportsbook no-vig moneylines are effectively probabilities
    conditional on there not being a tie.

    Kalshi KXNFLGAME pays:
        $1.00 if team wins
        $0.50 if game ties
        $0.00 if team loses

    Therefore:

        fair value
            =
        P(win) + 0.5 * P(tie)

    where:

        P(win)
            =
        (1 - P(tie)) * sportsbook conditional win probability
    """

    probability = float(
        no_tie_win_probability
    )

    tie_probability = float(
        tie_probability
    )

    if not (
        0.0
        <=
        probability
        <=
        1.0
    ):

        raise ValueError(
            "Win probability must be between 0 and 1."
        )

    if not (
        0.0
        <=
        tie_probability
        <=
        1.0
    ):

        raise ValueError(
            "Tie probability must be between 0 and 1."
        )

    return (
        (
            1.0
            -
            tie_probability
        )
        *
        probability
        +
        0.5
        *
        tie_probability
    )


# ============================================================
# FEE SCHEDULE
# ============================================================

def get_event_fee_multiplier(
    event_ticker,
    series_info,
):
    """
    Read the actual Kalshi fee configuration rather than
    assuming the weather-market multiplier applies to sports.
    """

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

    event_multiplier = (
        event_info.get(
            "fee_multiplier_override"
        )
    )

    if (
        event_multiplier
        is not None
    ):

        multiplier = float(
            event_multiplier
        )

    else:

        multiplier = float(
            series_info.get(
                "fee_multiplier"
            )
        )

    supported_fee_types = {
        "quadratic",
        "quadratic_with_maker_fees",
    }

    if (
        fee_type
        not in
        supported_fee_types
    ):

        raise RuntimeError(
            f"Unsupported Kalshi fee type "
            f"for {event_ticker}: "
            f"{fee_type!r}"
        )

    if not (
        0.0
        <=
        multiplier
        <=
        1.0
    ):

        raise RuntimeError(
            f"Invalid fee multiplier "
            f"for {event_ticker}: "
            f"{multiplier}"
        )

    return multiplier


# ============================================================
# PRICE HELPERS
# ============================================================

def valid_price(
    value,
):
    if value is None:
        return False

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return False

    return (
        0.0
        <
        value
        <
        1.0
    )


def build_execution_candidate(
    team_name,
    ticker,
    side,
    ask,
    fair_probability,
    fee_multiplier,
):
    """
    Calculate executable Kalshi edge after taker fees.
    """

    if not valid_price(
        ask
    ):

        return None

    ask = float(
        ask
    )

    fee = float(
        calculate_taker_fee(
            price=
                ask,

            contracts=
                1,

            multiplier=
                fee_multiplier,
        )
    )

    total_cost = (
        ask
        +
        fee
    )

    net_edge = (
        fair_probability
        -
        total_cost
    )

    expected_roi = (
        net_edge
        /
        total_cost
        if total_cost > 0
        else float("-inf")
    )

    return {

        "team":
            team_name,

        "ticker":
            ticker,

        "side":
            side,

        "ask":
            ask,

        "fee":
            fee,

        "total_cost":
            total_cost,

        "fair_probability":
            fair_probability,

        "net_edge":
            net_edge,

        "expected_roi":
            expected_roi,
    }


# ============================================================
# BEST EXECUTION FOR ONE TEAM
# ============================================================

def build_team_candidates(
    team_name,
    team_market,
    opponent_market,
    fair_probability,
    fee_multiplier,
):
    """
    There are two economically equivalent ways to get exposure
    to one team's payoff:

        YES on that team's contract
        NO on the opponent's contract

    Both pay:
        1.00 if this team wins
        0.50 if game ties
        0.00 if this team loses

    We evaluate both and keep the cheaper/better execution.
    """

    candidates = []

    yes_candidate = (
        build_execution_candidate(

            team_name=
                team_name,

            ticker=
                team_market[
                    "ticker"
                ],

            side=
                "YES",

            ask=
                team_market.get(
                    "yes_ask"
                ),

            fair_probability=
                fair_probability,

            fee_multiplier=
                fee_multiplier,
        )
    )

    if yes_candidate:

        candidates.append(
            yes_candidate
        )

    no_candidate = (
        build_execution_candidate(

            team_name=
                team_name,

            ticker=
                opponent_market[
                    "ticker"
                ],

            side=
                "NO",

            ask=
                opponent_market.get(
                    "no_ask"
                ),

            fair_probability=
                fair_probability,

            fee_multiplier=
                fee_multiplier,
        )
    )

    if no_candidate:

        candidates.append(
            no_candidate
        )

    if not candidates:

        return None

    candidates.sort(
        key=lambda row:
            row[
                "net_edge"
            ],
        reverse=True,
    )

    return (
        candidates[
            0
        ]
    )


# ============================================================
# SCAN SPORTSBOOK VS KALSHI
# ============================================================

def scan_edges(
    tie_probability=0.0,
):
    sportsbook_events = (
        get_moneyline_events(
            league="NFL"
        )
    )

    kalshi_markets = (
        get_open_nfl_moneylines()
    )

    (
        matches,
        unmatched,
    ) = match_moneyline_events(

        sportsbook_events=
            sportsbook_events,

        kalshi_markets=
            kalshi_markets,
    )

    sportsbook_by_id = {
        event.event_id:
            event
        for event in sportsbook_events
    }

    series_info = (
        get_series(
            SERIES_TICKER
        )
    )

    results = []

    print()
    print("=" * 110)
    print(
        "NFL SPORTSBOOK VS KALSHI EDGE SCANNER"
    )
    print("=" * 110)

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

    print(
        f"Tie probability:     "
        f"{tie_probability:.3%}"
    )

    if (
        tie_probability
        ==
        0.0
    ):

        print(
            "Tie adjustment:      "
            "OFF (research diagnostic)"
        )

    else:

        print(
            "Tie adjustment:      "
            "ON"
        )

    print()

    for match in matches:

        sportsbook_event = (
            sportsbook_by_id[
                match.sportsbook_event_id
            ]
        )

        consensus = (
            build_event_consensus(
                sportsbook_event
            )
        )

        if consensus is None:

            continue

        home_fair = (
            kalshi_fair_value(
                no_tie_win_probability=
                    consensus.team_a_probability,

                tie_probability=
                    tie_probability,
            )
        )

        away_fair = (
            kalshi_fair_value(
                no_tie_win_probability=
                    consensus.team_b_probability,

                tie_probability=
                    tie_probability,
            )
        )

        fee_multiplier = (
            get_event_fee_multiplier(

                event_ticker=
                    match.kalshi_event_ticker,

                series_info=
                    series_info,
            )
        )

        home_candidate = (
            build_team_candidates(

                team_name=
                    match.home_team,

                team_market=
                    match.home_market,

                opponent_market=
                    match.away_market,

                fair_probability=
                    home_fair,

                fee_multiplier=
                    fee_multiplier,
            )
        )

        away_candidate = (
            build_team_candidates(

                team_name=
                    match.away_team,

                team_market=
                    match.away_market,

                opponent_market=
                    match.home_market,

                fair_probability=
                    away_fair,

                fee_multiplier=
                    fee_multiplier,
            )
        )

        print("-" * 110)

        print(
            f"{match.away_team} "
            f"@ "
            f"{match.home_team}"
        )

        print(
            f"Kalshi event:       "
            f"{match.kalshi_event_ticker}"
        )

        print(
            f"Sportsbooks:        "
            f"{consensus.sportsbook_count}"
        )

        print(
            f"Book disagreement:  "
            f"{consensus.team_a_probability_range:.2%}"
        )

        print(
            f"Fee multiplier:     "
            f"{fee_multiplier:.3f}"
        )

        print()

        print(
            "SPORTSBOOK FAIR PROBABILITIES"
        )

        print(
            f"  {match.away_team}: "
            f"{consensus.team_b_probability:.2%}"
            f" -> Kalshi fair "
            f"{away_fair:.2%}"
        )

        print(
            f"  {match.home_team}: "
            f"{consensus.team_a_probability:.2%}"
            f" -> Kalshi fair "
            f"{home_fair:.2%}"
        )

        print()

        for candidate in [
            away_candidate,
            home_candidate,
        ]:

            if candidate is None:

                continue

            print(
                f"  {candidate['team']}"
            )

            print(
                f"    execution: "
                f"{candidate['side']} "
                f"{candidate['ticker']}"
            )

            print(
                f"    ask:       "
                f"{candidate['ask']:.3f}"
            )

            print(
                f"    fee:       "
                f"{candidate['fee']:.3f}"
            )

            print(
                f"    cost:      "
                f"{candidate['total_cost']:.3f}"
            )

            print(
                f"    fair:      "
                f"{candidate['fair_probability']:.2%}"
            )

            print(
                f"    edge:      "
                f"{candidate['net_edge']:+.2%}"
            )

            print(
                f"    exp ROI:   "
                f"{candidate['expected_roi']:+.2%}"
            )

            result = {
                **candidate,

                "event_ticker":
                    match.kalshi_event_ticker,

                "away_team":
                    match.away_team,

                "home_team":
                    match.home_team,

                "sportsbook_count":
                    consensus.sportsbook_count,

                "book_probability_range":
                    consensus.team_a_probability_range,

                "fee_multiplier":
                    fee_multiplier,
            }

            results.append(
                result
            )

    results.sort(
        key=lambda row:
            row[
                "net_edge"
            ],
        reverse=True,
    )

    return (
        results,
        unmatched,
    )


# ============================================================
# RANKED SUMMARY
# ============================================================

def print_ranked_summary(
    results,
):
    print()
    print("=" * 110)
    print(
        "RANKED EXECUTABLE EDGES"
    )
    print("=" * 110)

    if not results:

        print(
            "No valid candidates."
        )

        return

    print(
        f"{'TEAM':30s} "
        f"{'SIDE':4s} "
        f"{'COST':>7s} "
        f"{'FAIR':>8s} "
        f"{'EDGE':>8s} "
        f"{'EXP ROI':>9s} "
        f"{'BOOK RNG':>9s}"
    )

    print(
        "-" * 110
    )

    for row in results:

        print(
            f"{row['team'][:30]:30s} "
            f"{row['side']:4s} "
            f"{row['total_cost']:>7.3f} "
            f"{row['fair_probability']:>8.2%} "
            f"{row['net_edge']:>+8.2%} "
            f"{row['expected_roi']:>+9.2%} "
            f"{row['book_probability_range']:>9.2%}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--tie-probability",
        type=float,
        default=0.0,
        help=(
            "Assumed NFL tie probability as a decimal. "
            "Default 0.0 disables the tie adjustment."
        ),
    )

    args = (
        parser.parse_args()
    )

    if not (
        0.0
        <=
        args.tie_probability
        <=
        1.0
    ):

        parser.error(
            "--tie-probability must be between 0 and 1."
        )

    (
        results,
        unmatched,
    ) = scan_edges(
        tie_probability=
            args.tie_probability
    )

    print_ranked_summary(
        results
    )

    if unmatched:

        print()
        print("=" * 110)
        print(
            "UNMATCHED SPORTSBOOK EVENTS"
        )
        print("=" * 110)

        for event in unmatched:

            print(
                f"{event.away_team} "
                f"@ "
                f"{event.home_team}"
            )


if __name__ == "__main__":

    main()