import numpy as np
import pandas as pd

from src.backtest.probability_calibration import (
    run_probability_backtest
)

from src.weather.fair_distribution import (
    probability_for_market
)

from src.database.db import (
    get_historical_market_entries
)


START_DATE = "2026-07-19"
END_DATE = "2026-09-10"

DECISION_TIMES = [
    "18:00",
    "19:00",
    "20:05",
    "21:00",
    "22:00"
]


def build_timing_data():

    scored, _ = (
        run_probability_backtest()
    )

    scored = scored.copy()

    scored["target_date"] = (
        pd.to_datetime(
            scored["target_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    markets = (
        get_historical_market_entries(
            start_date=
                START_DATE,
            end_date=
                END_DATE
        )
    )

    markets = markets.copy()

    markets["target_date"] = (
        pd.to_datetime(
            markets["target_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    markets["decision_time"] = (
        pd.to_datetime(
            markets["decision_time"],
            utc=True
        )
    )

    markets["decision_label"] = (
        markets[
            "decision_time"
        ]
        .dt.strftime(
            "%H:%M"
        )
    )

    markets = markets[
        markets[
            "decision_label"
        ].isin(
            DECISION_TIMES
        )
    ].copy()

    calibration_lookup = (
        scored
        .set_index(
            "target_date"
        )
        .to_dict(
            orient="index"
        )
    )

    results = []

    for _, market in markets.iterrows():

        target_date = (
            market[
                "target_date"
            ]
        )

        calibration = (
            calibration_lookup.get(
                target_date
            )
        )

        if calibration is None:
            continue

        model_yes = (
            probability_for_market(
                strike_type=
                    market[
                        "strike_type"
                    ],

                floor_strike=
                    market[
                        "floor_strike"
                    ],

                cap_strike=
                    market[
                        "cap_strike"
                    ],

                point_forecast=
                    float(
                        calibration[
                            "point_forecast"
                        ]
                    ),

                residual_mean=
                    float(
                        calibration[
                            "residual_mean"
                        ]
                    ),

                residual_std=
                    float(
                        calibration[
                            "residual_std"
                        ]
                    )
            )
        )

        result = str(
            market.get(
                "result",
                ""
            )
        ).lower()

        if result not in (
            "yes",
            "no"
        ):
            continue

        yes_bid = market.get(
            "yes_bid"
        )

        yes_ask = market.get(
            "yes_ask"
        )

        if (
            yes_bid is None
            or
            yes_ask is None
            or
            pd.isna(
                yes_bid
            )
            or
            pd.isna(
                yes_ask
            )
        ):
            continue

        results.append({

            "target_date":
                target_date,

            "decision_label":
                market[
                    "decision_label"
                ],

            "ticker":
                market[
                    "ticker"
                ],

            "model_yes":
                model_yes,

            "market_mid":
                (
                    float(
                        yes_bid
                    )
                    +
                    float(
                        yes_ask
                    )
                )
                / 2.0,

            "actual":
                1.0
                if result == "yes"
                else 0.0
        })

    return pd.DataFrame(
        results
    )


def get_complete_dates(
    data
):

    complete_date_sets = []

    for decision_time in DECISION_TIMES:

        subset = data[
            data[
                "decision_label"
            ] == decision_time
        ]

        counts = (
            subset
            .groupby(
                "target_date"
            )
            .size()
        )

        complete_dates = set(
            counts[
                counts == 6
            ].index
        )

        complete_date_sets.append(
            complete_dates
        )

    if not complete_date_sets:
        return set()

    return set.intersection(
        *complete_date_sets
    )


def calculate_metrics(
    data,
    decision_time,
    common_dates
):

    subset = data[
        (
            data[
                "decision_label"
            ] == decision_time
        )
        &
        (
            data[
                "target_date"
            ].isin(
                common_dates
            )
        )
    ].copy()

    daily_results = []

    for (
        target_date,
        group
    ) in subset.groupby(
        "target_date"
    ):

        if len(group) != 6:
            continue

        group = group.copy()

        market_total = (
            group[
                "market_mid"
            ].sum()
        )

        if market_total <= 0:
            continue

        group[
            "market_probability"
        ] = (
            group[
                "market_mid"
            ]
            /
            market_total
        )

        if (
            group[
                "actual"
            ].sum()
            != 1
        ):
            continue

        model_brier = (
            (
                group[
                    "model_yes"
                ]
                -
                group[
                    "actual"
                ]
            ) ** 2
        ).sum()

        market_brier = (
            (
                group[
                    "market_probability"
                ]
                -
                group[
                    "actual"
                ]
            ) ** 2
        ).sum()

        winner = group[
            group[
                "actual"
            ] == 1
        ].iloc[0]

        model_winner_probability = (
            float(
                winner[
                    "model_yes"
                ]
            )
        )

        market_winner_probability = (
            float(
                winner[
                    "market_probability"
                ]
            )
        )

        epsilon = 1e-12

        model_log_loss = (
            -np.log(
                max(
                    model_winner_probability,
                    epsilon
                )
            )
        )

        market_log_loss = (
            -np.log(
                max(
                    market_winner_probability,
                    epsilon
                )
            )
        )

        daily_results.append({

            "model_brier":
                model_brier,

            "market_brier":
                market_brier,

            "model_log_loss":
                model_log_loss,

            "market_log_loss":
                market_log_loss,

            "model_winner_probability":
                model_winner_probability,

            "market_winner_probability":
                market_winner_probability,

            "model_better":
                model_brier
                <
                market_brier
        })

    daily = pd.DataFrame(
        daily_results
    )

    if daily.empty:
        return None

    return {

        "time":
            decision_time,

        "dates":
            len(daily),

        "model_brier":
            daily[
                "model_brier"
            ].mean(),

        "market_brier":
            daily[
                "market_brier"
            ].mean(),

        "model_log_loss":
            daily[
                "model_log_loss"
            ].mean(),

        "market_log_loss":
            daily[
                "market_log_loss"
            ].mean(),

        "model_winner_probability":
            daily[
                "model_winner_probability"
            ].mean(),

        "market_winner_probability":
            daily[
                "market_winner_probability"
            ].mean(),

        "model_better":
            daily[
                "model_better"
            ].mean()
    }


def run_timing_analysis():

    data = (
        build_timing_data()
    )

    common_dates = (
        get_complete_dates(
            data
        )
    )

    print()
    print("=" * 115)
    print(
        "KALSHI MARKET TIMING ANALYSIS"
    )
    print("=" * 115)

    print(
        f"Dates complete at ALL times: "
        f"{len(common_dates)}"
    )

    print()

    print(
        f"{'Time':<10}"
        f"{'Dates':<8}"
        f"{'Model Brier':<15}"
        f"{'Kalshi Brier':<15}"
        f"{'Model Log':<14}"
        f"{'Kalshi Log':<14}"
        f"{'Model Win P':<14}"
        f"{'Kalshi Win P':<14}"
        f"{'Model Better':<14}"
    )

    print("-" * 115)

    for decision_time in DECISION_TIMES:

        metrics = (
            calculate_metrics(
                data=
                    data,

                decision_time=
                    decision_time,

                common_dates=
                    common_dates
            )
        )

        if metrics is None:
            continue

        print(
            f"{metrics['time']:<10}"
            f"{metrics['dates']:<8}"
            f"{metrics['model_brier']:<15.4f}"
            f"{metrics['market_brier']:<15.4f}"
            f"{metrics['model_log_loss']:<14.4f}"
            f"{metrics['market_log_loss']:<14.4f}"
            f"{metrics['model_winner_probability']:<14.2%}"
            f"{metrics['market_winner_probability']:<14.2%}"
            f"{metrics['model_better']:<14.2%}"
        )


if __name__ == "__main__":

    run_timing_analysis()