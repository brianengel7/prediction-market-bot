from src.database.db import (
    get_market_snapshots
)

from src.weather.model_versions import (
    WEATHER_MODEL_VERSION
)


def analyze_market_history(
    target_date
):

    snapshots = (
        get_market_snapshots(
            target_date=target_date,
            model_version=
                WEATHER_MODEL_VERSION
        )
    )

    if snapshots.empty:

        print(
            f"No market snapshots "
            f"found for {target_date}."
        )

        return

    print()
    print("=" * 90)
    print(
        "KALSHI MARKET HISTORY"
    )
    print("=" * 90)

    print(
        f"Target date: "
        f"{target_date}"
    )

    print(
        f"Model: "
        f"{WEATHER_MODEL_VERSION}"
    )

    print(
        f"Total rows: "
        f"{len(snapshots)}"
    )

    print(
        f"Snapshots: "
        f"{snapshots['snapshot_time'].nunique()}"
    )

    print()

    for ticker, group in snapshots.groupby(
        "ticker"
    ):

        group = group.sort_values(
            "snapshot_time"
        )

        first = group.iloc[0]
        latest = group.iloc[-1]

        print(
            first["title"]
        )

        print(
            f"  Ticker: "
            f"{ticker}"
        )

        print(
            f"  Observations: "
            f"{len(group)}"
        )

        print()

        print(
            f"  Model YES: "
            f"{latest['model_yes']:.2%}"
        )

        print()

        print(
            f"  First YES ask: "
            f"{first['yes_ask']:.2%}"
        )

        print(
            f"  Latest YES ask: "
            f"{latest['yes_ask']:.2%}"
        )

        print(
            f"  Min YES ask: "
            f"{group['yes_ask'].min():.2%}"
        )

        print(
            f"  Max YES ask: "
            f"{group['yes_ask'].max():.2%}"
        )

        print()

        print(
            f"  First YES edge: "
            f"{first['yes_edge']:+.2%}"
        )

        print(
            f"  Latest YES edge: "
            f"{latest['yes_edge']:+.2%}"
        )

        print(
            f"  Max YES edge: "
            f"{group['yes_edge'].max():+.2%}"
        )

        print()

        print(
            f"  First NO edge: "
            f"{first['no_edge']:+.2%}"
        )

        print(
            f"  Latest NO edge: "
            f"{latest['no_edge']:+.2%}"
        )

        print(
            f"  Max NO edge: "
            f"{group['no_edge'].max():+.2%}"
        )

        print("-" * 90)


if __name__ == "__main__":

    analyze_market_history(
        "2026-09-28"
    )