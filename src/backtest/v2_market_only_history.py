import os

import pandas as pd

from src.database.db import (
    get_historical_market_entries
)

from src.kalshi.decision_timing import (
    SYNC_CALIBRATION_PREFIX,
    historical_calibration_age_limit,
)

from src.kalshi.historical_market import (
    get_decision_time,
    build_event_ticker,
)


def load_synchronized_market_history(
    minimum_dates=20,
    series_ticker="KXHIGHNY",
):

    if not os.environ.get(
        "PREDICTION_DATABASE_URL",
        ""
    ).strip():

        raise RuntimeError(
            "PREDICTION_DATABASE_URL is missing. "
            "Configure the Supabase connection first."
        )

    series_ticker = (
        str(series_ticker)
        .strip()
        .upper()
    )

    history = (
        get_historical_market_entries(
            series_ticker=
                series_ticker,
        )
        .copy()
    )

    required = {
        "target_date",
        "ticker",
        "decision_time",
        "market_status",
        "result",
        "settlement_value",
        "calibration_quote_time",
        "calibration_yes_bid",
        "calibration_yes_ask",
        "calibration_candle_source",
    }

    missing = (
        required
        -
        set(history.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing synchronized calibration columns: "
            +
            ", ".join(
                sorted(missing)
            )
        )

    for column in (
        "decision_time",
        "calibration_quote_time",
    ):

        history[column] = pd.to_datetime(
            history[column],
            utc=True,
            errors="coerce",
            format="mixed",
        )

    history[
        "target_date"
    ] = pd.to_datetime(
        history[
            "target_date"
        ],
        errors="coerce",
    ).dt.strftime(
        "%Y-%m-%d"
    )

    # ------------------------------------------------------------
    # Keep only synchronized decision-time calibration snapshots
    # BEFORE grouping by target date.
    # ------------------------------------------------------------

    history = (
        history[
            history[
                "calibration_candle_source"
            ]
            .fillna("")
            .astype(str)
            .str.startswith(
                SYNC_CALIBRATION_PREFIX
            )
        ]
        .copy()
    )

    rows = []

    for (
        target_date,
        group
    ) in history.groupby(
        "target_date"
    ):

        group = (
            group
            .sort_values(
                "ticker"
            )
            .copy()
        )

        expected_decision = (
            get_decision_time(
                target_date
            )
        )

        event_ticker = (
            build_event_ticker(
                target_date,
                series_ticker=
                    series_ticker,
            )
        )

        expected_event = (
            event_ticker
            +
            "-"
        )

        if (
            len(group) != 6
            or
            group[
                "ticker"
            ].nunique() != 6
            or
            not group[
                "ticker"
            ].astype(str)
            .str.startswith(
                expected_event
            ).all()
            or
            not group[
                "decision_time"
            ].eq(
                expected_decision
            ).all()
            or
            not group[
                "market_status"
            ].astype(str)
            .str.strip()
            .str.lower()
            .eq(
                "finalized"
            ).all()
            or
            not group[
                "calibration_candle_source"
            ].astype(str)
            .str.startswith(
                SYNC_CALIBRATION_PREFIX
            ).all()
            or
            not group[
                "event_ticker"
            ].astype(str).eq(
                event_ticker
            ).all()
        ):

            continue

        timestamps = (
            group[
                "calibration_quote_time"
            ]
        )

        try:

            age_limit = (
                historical_calibration_age_limit(
                    group[
                        "calibration_candle_source"
                    ],
                    timestamps,
                )
            )

        except ValueError:

            continue

        ages = (
            expected_decision
            -
            timestamps
        ).dt.total_seconds() / 60.0

        yes_bid = pd.to_numeric(
            group[
                "calibration_yes_bid"
            ],
            errors="coerce",
        )

        yes_ask = pd.to_numeric(
            group[
                "calibration_yes_ask"
            ],
            errors="coerce",
        )

        if not (
            ages.between(
                0,
                age_limit
            )
            &
            yes_bid.between(
                0,
                1
            )
            &
            yes_ask.between(
                0,
                1
            )
            &
            yes_bid.le(
                yes_ask
            )
        ).all():

            continue

        outcomes = (
            group[
                "result"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1.0,
                "no": 0.0,
            })
        )

        settlement = pd.to_numeric(
            group[
                "settlement_value"
            ],
            errors="coerce",
        )

        if (
            settlement.notna()
            &
            ~settlement.isin([
                0,
                1
            ])
        ).any():

            continue

        if (
            outcomes.notna()
            &
            settlement.notna()
            &
            outcomes.ne(
                settlement
            )
        ).any():

            continue

        outcomes = (
            outcomes.fillna(
                settlement
            )
        )

        if (
            not outcomes.isin([
                0,
                1
            ]).all()
            or
            outcomes.sum() != 1
        ):

            continue

        market_mid = (
            yes_bid
            +
            yes_ask
        ) / 2.0

        midpoint_sum = float(
            market_mid.sum()
        )

        if midpoint_sum <= 0:

            continue

        group[
            "market_probability"
        ] = (
            market_mid
            /
            midpoint_sum
        )

        group[
            "yes_bid"
        ] = yes_bid

        group[
            "yes_ask"
        ] = yes_ask

        # Binary Kalshi complement:
        # buying NO at its ask corresponds
        # to the complement of the YES bid.
        group[
            "no_ask"
        ] = (
            1.0
            -
            yes_bid
        )

        group[
            "outcome"
        ] = outcomes.astype(int)

        group[
            "quote_time_utc"
        ] = timestamps

        group[
            "quote_age_minutes"
        ] = ages

        rows.extend(
            group[[
                "target_date",
                "ticker",
                "market_probability",
                "yes_bid",
                "yes_ask",
                "no_ask",
                "outcome",
                "quote_time_utc",
                "quote_age_minutes",
            ]]
            .to_dict(
                "records"
            )
        )

    dataframe = pd.DataFrame(
        rows
    )

    date_count = (
        dataframe[
            "target_date"
        ].nunique()
        if not dataframe.empty
        else 0
    )

    if (
        date_count
        <
        minimum_dates
    ):

        raise RuntimeError(
            f"Only {date_count} complete "
            f"synchronized dates found; "
            f"{minimum_dates} required."
        )

    return dataframe