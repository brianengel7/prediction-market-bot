from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from src.kalshi.client import get_markets, get_event
from src.kalshi.markets import parse_markets
from src.kalshi.events import get_weather_target_date
from src.weather.hrrr import compare_hrrr_runs, get_recent_hrrr_runs
from src.weather.gefs import get_latest_gefs_run, get_gefs_ensemble
from src.database.db import save_model_snapshot, save_market_snapshots
from src.weather.observations import get_observed_high
from src.weather.client import get_nws_high_forecast
from src.strategy.weather_edge import analyze_weather_markets, PROBABILITY_MODEL_VERSION
from src.weather.live_bundle import (
    get_live_forecast_bundle
)


def main():

    NYC_TIMEZONE = ZoneInfo(
        "America/New_York"
    )

    markets_data = get_markets(
        series_ticker="KXHIGHNY",
        status="open"
    )

    markets = parse_markets(
        markets_data
    )

    if not markets:
        raise RuntimeError(
            "No open KXHIGHNY markets found."
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

        event_target_date = (
            get_weather_target_date(
                event
            )
        )

        events.append({
            "event_ticker": event_ticker,
            "event": event,
            "target_date": event_target_date
        })


    today = datetime.now(
        NYC_TIMEZONE
    ).date()


    future_events = [
        event_info
        for event_info in events
        if datetime.strptime(
            event_info["target_date"],
            "%Y-%m-%d"
        ).date() > today
    ]


    if not future_events:
        raise RuntimeError(
            "No future KXHIGHNY events found."
        )


    future_events.sort(
        key=lambda event_info:
        event_info["target_date"]
    )


    selected_event = future_events[0]


    event_ticker = selected_event[
        "event_ticker"
    ]

    event = selected_event[
        "event"
    ]

    target_date = selected_event[
        "target_date"
    ]

    nws_result = get_nws_high_forecast(
        target_date
    )

    markets = [
        market
        for market in markets
        if market["event_ticker"]
        == event_ticker
    ]
    # HRRR

    run_times = get_recent_hrrr_runs(
        target_date=target_date,
        count=3
    )

    print()
    print("HRRR RUNS")

    for run_time in run_times:
        print(run_time)

    hrrr_results = compare_hrrr_runs(
        run_times,
        target_date
    )

    # Observations

    observation_results = (
        get_observed_high(
            today
        )
    )

    # RAW EDGE CALCULATION

    bundle = get_live_forecast_bundle(
        target_date
    )

    calibration = bundle[
        "calibration"
    ]

    gefs_results = bundle[
        "gefs"
    ]

    edge_results = analyze_weather_markets(
        markets=markets,
        gefs_results=gefs_results,
        calibration=calibration
    )

    snapshot_time = save_model_snapshot(
        target_date=target_date,
        gefs_results=gefs_results,
        hrrr_results=hrrr_results,
        nws_result=nws_result,
        model_version=PROBABILITY_MODEL_VERSION
    )

    save_market_snapshots(
        snapshot_time=snapshot_time,
        target_date=target_date,
        edge_results=edge_results
    )

    # HRRR SUMMARY

    print()
    print("HRRR SUMMARY")

    for result in hrrr_results:

        if result["change"] is None:
            change = "N/A"
        else:
            change = (
                f"{result['change']:+.2f}°F"
            )

        print(
            f"{result['run_time'].strftime('%m/%d %HZ')} | "
            f"Max: "
            f"{result['max_temperature']:.2f}°F | "
            f"Change: {change}"
        )

    # GEFS SUMMARY

    print()
    print("LATEST GEFS SUMMARY")

    print(
        f"Run: "
        f"{gefs_results['run_time'].strftime('%m/%d %HZ')}"
    )

    print(
        f"Mean high: "
        f"{gefs_results['mean_high']:.2f}°F"
    )

    print(
        f"Median high: "
        f"{gefs_results['median_high']:.2f}°F"
    )

    print(
        f"Std deviation: "
        f"{gefs_results['std_high']:.2f}°F"
    )

    print(
        f"Lowest member: "
        f"{gefs_results['min_high']:.2f}°F"
    )

    print(
        f"Highest member: "
        f"{gefs_results['max_high']:.2f}°F"
    )

    # MODEL VS KALSHI

    print()
    print("CALIBRATED MULTI-MODEL VS KALSHI")

    for result in edge_results:

        print()
        print(result["title"])

        print(
            f"Raw GEFS:       "
            f"{result['raw_gefs_yes'] * 100:.2f}% "
            f"({result['yes_count']}/"
            f"{result['total_members']})"
        )

        print(
            f"GEFS KDE:       "
            f"{result['gefs_kde_yes'] * 100:.2f}%"
        )

        print(
            f"Calibrated YES: "
            f"{result['model_yes'] * 100:.2f}%"
        )

        print(
            f"Calibrated NO:  "
            f"{result['model_no'] * 100:.2f}%"
        )

        print(
            f"Fair high:      "
            f"{result['fair_point_forecast']:.2f}°F"
        )

        print()

        print(
            f"YES bid:    "
            f"{result['yes_bid']:6.2%}"
        )

        print(
            f"YES ask:    "
            f"{result['yes_ask']:6.2%}"
        )

        print(
            f"YES edge:   "
            f"{result['yes_edge']:+6.2%}"
        )

        print()

        print(
            f"NO bid:     "
            f"{result['no_bid']:6.2%}"
        )

        print(
            f"NO ask:     "
            f"{result['no_ask']:6.2%}"
        )

        print(
            f"NO edge:    "
            f"{result['no_edge']:+6.2%}"
        )

        print()

        if result["best_edge"] > 0:

            print(
                f"CALIBRATED EDGE: "
                f"{result['best_side']} "
                f"{result['best_edge']:+.2%}"
            )

        else:

            print("CALIBRATED EDGE:   PASS")

    print()
    print("CENTRAL PARK OBSERVATIONS")

    if observation_results is None:

        print(
            "No observations found."
        )

    else:

        print(
            f"Latest: "
            f"{observation_results['latest_temperature']:.2f}°F "
            f"at "
            f"{observation_results['latest_time']:%I:%M %p}"
        )

        print(
            f"High so far: "
            f"{observation_results['observed_high']:.2f}°F "
            f"at "
            f"{observation_results['high_time']:%I:%M %p}"
        )

        print(
            f"Observations: "
            f"{observation_results['observation_count']}"
        )

    print()
    print("FORECAST COMPARISON")

    latest_hrrr = hrrr_results[-1]

    print(
        f"HRRR latest: "
        f"{latest_hrrr['max_temperature']:.2f}°F"
    )

    print(
        f"GEFS mean:   "
        f"{gefs_results['mean_high']:.2f}°F"
    )

    print(
        f"GEFS median: "
        f"{gefs_results['median_high']:.2f}°F"
    )

    if nws_result is not None:

        print(
            f"NWS high:    "
            f"{nws_result['temperature']:.2f}°F"
        )

        print(
            f"NWS period:  "
            f"{nws_result['name']}"
        )

        print(
            f"NWS forecast: "
            f"{nws_result['short_forecast']}"
        )

    else:

        print(
            "NWS high:    Not found"
        )

    print()
    print("CALIBRATED MODEL INPUTS")

    for source, value in (
        calibration[
            "raw_forecasts"
        ].items()
    ):

        print(
            f"{source:10s}: "
            f"{value:.2f}°F"
        )

    print(
        f"Fair high:  "
        f"{calibration['point_forecast']:.2f}°F"
    )

    print(
        f"Uncertainty: "
        f"{calibration['residual_std']:.2f}°F"
    )

if __name__ == "__main__":
    main()