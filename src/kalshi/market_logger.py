from datetime import datetime
from zoneinfo import ZoneInfo
import time
import argparse

import pandas as pd

from src.kalshi.client import (
    get_markets,
    get_event
)

from src.kalshi.markets import (
    parse_markets
)

from src.kalshi.events import (
    get_weather_target_date
)

from src.database.db import (
    get_weather_fair_values,
    save_market_snapshots
)

from src.weather.model_versions import (
    WEATHER_MODEL_VERSION
)

SERIES_TICKER = "KXHIGHNY"

NYC_TIMEZONE = ZoneInfo(
    "America/New_York"
)


# ============================================================
# FIND CURRENT WEATHER EVENT
# ============================================================

def get_current_weather_event():
    """
    Find the nearest future open KXHIGHNY event.

    This mirrors the event-selection logic used
    by main.py, but does NOT run any weather models.
    """

    markets_data = get_markets(
        series_ticker=SERIES_TICKER,
        status="open"
    )

    markets = parse_markets(
        markets_data
    )

    if not markets:
        raise RuntimeError(
            f"No open {SERIES_TICKER} markets found."
        )

    event_tickers = {
        market["event_ticker"]
        for market in markets
    }

    events = []

    for event_ticker in event_tickers:

        event = get_event(
            event_ticker
        )

        target_date = (
            get_weather_target_date(
                event
            )
        )

        events.append({
            "event_ticker":
                event_ticker,

            "event":
                event,

            "target_date":
                target_date
        })

    today = datetime.now(
        NYC_TIMEZONE
    ).date()

    future_events = [
        event_info
        for event_info in events
        if datetime.strptime(
            event_info[
                "target_date"
            ],
            "%Y-%m-%d"
        ).date() > today
    ]

    if not future_events:
        raise RuntimeError(
            "No future KXHIGHNY events found."
        )

    future_events.sort(
        key=lambda event_info:
            event_info[
                "target_date"
            ]
    )

    selected_event = (
        future_events[0]
    )

    event_ticker = (
        selected_event[
            "event_ticker"
        ]
    )

    target_date = (
        selected_event[
            "target_date"
        ]
    )

    event_markets = [
        market
        for market in markets
        if market[
            "event_ticker"
        ] == event_ticker
    ]

    return {
        "event_ticker":
            event_ticker,

        "target_date":
            target_date,

        "markets":
            event_markets
    }


# ============================================================
# EDGE CALCULATION
# ============================================================

def calculate_edge(
    model_probability,
    ask_price
):
    """
    Edge is calculated against the executable ask.

    Example:

    model probability = 0.40
    market ask        = 0.30

    edge = +0.10
    """

    if ask_price is None:
        return None

    return (
        float(model_probability)
        - float(ask_price)
    )


def choose_best_side(
    yes_edge,
    no_edge
):
    """
    Determine whether YES or NO currently offers
    positive executable model edge.
    """

    available_edges = []

    if yes_edge is not None:
        available_edges.append(
            (
                "YES",
                yes_edge
            )
        )

    if no_edge is not None:
        available_edges.append(
            (
                "NO",
                no_edge
            )
        )

    if not available_edges:
        return (
            "PASS",
            None
        )

    best_side, best_edge = max(
        available_edges,
        key=lambda item:
            item[1]
    )

    if best_edge <= 0:
        best_side = "PASS"

    return (
        best_side,
        best_edge
    )


# ============================================================
# MATCH LIVE MARKETS TO SAVED FAIR VALUES
# ============================================================

def build_edge_results(
    markets,
    fair_values
):
    """
    Join live Kalshi prices to saved MultiModel v1
    fair probabilities using ticker.
    """

    fair_value_lookup = {
        row["ticker"]:
            row
        for row in fair_values.to_dict(
            orient="records"
        )
    }

    results = []

    for market in markets:

        ticker = market[
            "ticker"
        ]

        fair_value = (
            fair_value_lookup.get(
                ticker
            )
        )

        if fair_value is None:

            print(
                f"Skipping {ticker}: "
                f"no saved fair value."
            )

            continue

        model_yes = float(
            fair_value[
                "model_yes"
            ]
        )

        model_no = float(
            fair_value[
                "model_no"
            ]
        )

        yes_bid = market.get(
            "yes_bid"
        )

        yes_ask = market.get(
            "yes_ask"
        )

        no_bid = market.get(
            "no_bid"
        )

        no_ask = market.get(
            "no_ask"
        )

        yes_edge = calculate_edge(
            model_probability=
                model_yes,

            ask_price=
                yes_ask
        )

        no_edge = calculate_edge(
            model_probability=
                model_no,

            ask_price=
                no_ask
        )

        (
            best_side,
            best_edge
        ) = choose_best_side(
            yes_edge=
                yes_edge,

            no_edge=
                no_edge
        )

        result = {

            # ----------------------------------------
            # Contract information
            # ----------------------------------------

            "ticker":
                ticker,

            "title":
                market[
                    "title"
                ],

            "strike_type":
                market[
                    "strike_type"
                ],

            "floor_strike":
                market.get(
                    "floor_strike"
                ),

            "cap_strike":
                market.get(
                    "cap_strike"
                ),

            # ----------------------------------------
            # Old GEFS fields.
            #
            # We intentionally DO NOT calculate
            # these in the lightweight logger.
            # ----------------------------------------

            "raw_gefs_yes":
                None,

            "bandwidth":
                None,

            # ----------------------------------------
            # Frozen MultiModel v1 fair values
            # ----------------------------------------

            "model_yes":
                model_yes,

            "model_no":
                model_no,

            # ----------------------------------------
            # Current Kalshi prices
            # ----------------------------------------

            "yes_bid":
                yes_bid,

            "yes_ask":
                yes_ask,

            "no_bid":
                no_bid,

            "no_ask":
                no_ask,

            # ----------------------------------------
            # Executable edge
            # ----------------------------------------

            "yes_edge":
                yes_edge,

            "no_edge":
                no_edge,

            "best_side":
                best_side,

            "best_edge":
                best_edge,

            # ----------------------------------------
            # Model version
            # ----------------------------------------

            "model_version":
                WEATHER_MODEL_VERSION
        }

        results.append(
            result
        )

    results.sort(
        key=lambda result:
            (
                result[
                    "best_edge"
                ]
                if result[
                    "best_edge"
                ] is not None
                else -999
            ),
        reverse=True
    )

    return results


# ============================================================
# DISPLAY
# ============================================================

def print_snapshot(
    event_ticker,
    target_date,
    snapshot_time,
    edge_results
):

    print()
    print("=" * 72)
    print(
        "KALSHI MARKET EDGE SNAPSHOT"
    )
    print("=" * 72)

    print(
        f"Event:       "
        f"{event_ticker}"
    )

    print(
        f"Target date: "
        f"{target_date}"
    )

    print(
        f"Model:       "
        f"{WEATHER_MODEL_VERSION}"
    )

    print(
        f"Snapshot:    "
        f"{snapshot_time}"
    )

    print(
        f"Markets:     "
        f"{len(edge_results)}"
    )

    print()

    for result in edge_results:

        print(
            result[
                "title"
            ]
        )

        print(
            f"  Model YES: "
            f"{result['model_yes']:.2%}"
        )

        print(
            f"  Model NO:  "
            f"{result['model_no']:.2%}"
        )

        print()

        print(
            f"  YES bid:   "
            f"{result['yes_bid']:.2%}"
        )

        print(
            f"  YES ask:   "
            f"{result['yes_ask']:.2%}"
        )

        print(
            f"  YES edge:  "
            f"{result['yes_edge']:+.2%}"
        )

        print()

        print(
            f"  NO bid:    "
            f"{result['no_bid']:.2%}"
        )

        print(
            f"  NO ask:    "
            f"{result['no_ask']:.2%}"
        )

        print(
            f"  NO edge:   "
            f"{result['no_edge']:+.2%}"
        )

        print()

        if (
            result[
                "best_side"
            ] == "PASS"
        ):

            print(
                f"  SIGNAL: PASS "
                f"(best available edge "
                f"{result['best_edge']:+.2%})"
            )

        else:

            print(
                f"  SIGNAL: "
                f"{result['best_side']} "
                f"{result['best_edge']:+.2%}"
            )

        print("-" * 72)


# ============================================================
# COLLECT SNAPSHOT
# ============================================================

def collect_market_snapshot():

    # --------------------------------------------------------
    # 1. Find current Kalshi event
    # --------------------------------------------------------

    event_info = (
        get_current_weather_event()
    )

    event_ticker = (
        event_info[
            "event_ticker"
        ]
    )

    target_date = (
        event_info[
            "target_date"
        ]
    )

    markets = (
        event_info[
            "markets"
        ]
    )

    # --------------------------------------------------------
    # 2. Load previously calculated MultiModel fair values
    #
    # NO GEFS / HRRR / NBM / MOS is run here.
    # --------------------------------------------------------

    fair_values = (
        get_weather_fair_values(
            target_date=
                target_date,

            model_version=
                WEATHER_MODEL_VERSION
        )
    )

    if fair_values.empty:

        raise RuntimeError(
            f"No saved fair values found "
            f"for {target_date} using "
            f"{WEATHER_MODEL_VERSION}.\n"
            f"Run main.py first to create "
            f"the day's model probabilities."
        )

    # --------------------------------------------------------
    # 3. Match fair values to live Kalshi prices
    # --------------------------------------------------------

    edge_results = (
        build_edge_results(
            markets=
                markets,

            fair_values=
                fair_values
        )
    )

    if not edge_results:

        raise RuntimeError(
            "No live Kalshi contracts matched "
            "the saved fair-value tickers."
        )

    # --------------------------------------------------------
    # 4. Timestamp snapshot
    # --------------------------------------------------------

    snapshot_time = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    # --------------------------------------------------------
    # 5. Save snapshot
    # --------------------------------------------------------

    save_market_snapshots(
        snapshot_time=
            snapshot_time,

        target_date=
            target_date,

        edge_results=
            edge_results
    )

    # --------------------------------------------------------
    # 6. Display
    # --------------------------------------------------------

    print_snapshot(
        event_ticker=
            event_ticker,

        target_date=
            target_date,

        snapshot_time=
            snapshot_time,

        edge_results=
            edge_results
    )

    print()
    print(
        f"Saved {len(edge_results)} "
        f"market snapshots."
    )

    return edge_results

# ============================================================
# CONTINUOUS LOGGER
# ============================================================

def run_market_logger(
    interval_seconds=300
):
    """
    Continuously collect Kalshi market snapshots.

    Weather fair values remain frozen from main.py.
    Only live market prices are refreshed.
    """

    print()
    print("=" * 72)
    print("KALSHI MARKET LOGGER STARTED")
    print("=" * 72)

    print(
        f"Logging every "
        f"{interval_seconds} seconds."
    )

    print(
        "Press Ctrl+C to stop."
    )

    while True:

        try:

            collect_market_snapshot()

        except KeyboardInterrupt:

            print()
            print(
                "Market logger stopped."
            )

            break

        except Exception as error:

            print()
            print(
                "Market snapshot failed:"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "Will retry at the next "
                "scheduled interval."
            )

        try:

            time.sleep(
                interval_seconds
            )

        except KeyboardInterrupt:

            print()
            print(
                "Market logger stopped."
            )

            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
            "Collect Kalshi weather market snapshots."
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=
            "Collect one snapshot and exit."
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help=
            "Seconds between snapshots "
            "(default: 300)."
    )

    args = parser.parse_args()

    if args.once:

        collect_market_snapshot()

    else:

        run_market_logger(
            interval_seconds=
                args.interval
        )