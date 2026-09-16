from src.kalshi.nws import get_markets
from src.kalshi.markets import parse_markets

from src.weather.hrrr import (
    compare_hrrr_runs,
    get_recent_hrrr_runs
)

from src.weather.gefs import (
    get_latest_gefs_run,
    get_gefs_ensemble
)

from src.strategy.weather_edge import (
    analyze_weather_markets
)


target_date = "2026-09-16"

# HRRR

run_times = get_recent_hrrr_runs(
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

# KALSHI

markets_data = get_markets(
    series_ticker="KXHIGHNY",
    status="open"
)

markets = parse_markets(
    markets_data
)

# GEFS

gefs_run = get_latest_gefs_run()

members = range(0, 31)

gefs_results = get_gefs_ensemble(
    run_time=gefs_run,
    target_date=target_date,
    members=members
)

# RAW EDGE CALCULATION

edge_results = analyze_weather_markets(
    markets=markets,
    gefs_results=gefs_results
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
print("GEFS SUMMARY")

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
print("RAW GEFS MODEL VS KALSHI")

for result in edge_results:

    print()
    print(result["title"])

    print(
        f"GEFS YES:   "
        f"{result['model_yes']:6.2%} "
        f"({result['yes_count']}/"
        f"{result['total_members']})"
    )

    print(
        f"GEFS NO:    "
        f"{result['model_no']:6.2%}"
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
            f"RAW EDGE:   "
            f"{result['best_side']} "
            f"{result['best_edge']:+.2%}"
        )

    else:

        print("RAW EDGE:   PASS")