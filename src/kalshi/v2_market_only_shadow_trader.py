import argparse

import numpy as np
import pandas as pd

from src.backtest.v2_source_signal_test import (
    load_dataset
)

from src.backtest.v2_market_only_control import (
    fit_market_alpha,
    market_only_probabilities
)

from src.backtest.v2_market_only_walkforward import (
    get_training_dates
)

from src.database.db import (
    get_historical_market_entries,
    save_v2_market_only_shadow_decision
)

from src.kalshi.market_logger import (
    get_current_weather_event
)

from src.kalshi.fees import (
    calculate_taker_fee
)

from src.kalshi.live_executor import (
    execute_live_trade
)


# ============================================================
# FROZEN FORWARD STRATEGY
# ============================================================

LOOKBACK_DAYS = 20

ALPHA_VALUES = np.arange(
    0.50,
    5.0001,
    0.025
)

MINIMUM_EDGE = 0.025

DECISION_HOUR_UTC = 20
DECISION_MINUTE_UTC = 5

SAVE_WINDOW_MINUTES = 15

def load_market_only_calibration_history():

    history = (
        get_historical_market_entries()
        .copy()
    )

    if history.empty:

        raise RuntimeError(
            "No historical Kalshi entries found."
        )

    # --------------------------------------------------------
    # We only want the exact strategy decision time:
    # 20:05 UTC.
    # --------------------------------------------------------

    history = history[
        (
            history[
                "decision_time"
            ].dt.hour
            ==
            DECISION_HOUR_UTC
        )
        &
        (
            history[
                "decision_time"
            ].dt.minute
            ==
            DECISION_MINUTE_UTC
        )
    ].copy()

    if history.empty:

        raise RuntimeError(
            "No 20:05 UTC historical entries found."
        )

    history[
        "target_date"
    ] = pd.to_datetime(
        history[
            "target_date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    calibration_rows = []

    # --------------------------------------------------------
    # Build one normalized six-contract distribution per day.
    # --------------------------------------------------------

    for (
        target_date,
        group
    ) in history.groupby(
        "target_date"
    ):

        group = (
            group.copy()
            .reset_index(
                drop=True
            )
        )

        if len(
            group
        ) != 6:
            continue

        # ----------------------------------------------------
        # Require complete midpoint quotes.
        # ----------------------------------------------------

        if (
            group[
                "yes_bid"
            ].isna().any()
            or
            group[
                "yes_ask"
            ].isna().any()
        ):
            continue

        group[
            "market_mid"
        ] = (
            group[
                "yes_bid"
            ].astype(
                float
            )
            +
            group[
                "yes_ask"
            ].astype(
                float
            )
        ) / 2.0

        raw_mid_sum = float(
            group[
                "market_mid"
            ].sum()
        )

        if raw_mid_sum <= 0:
            continue

        group[
            "market_probability"
        ] = (
            group[
                "market_mid"
            ]
            /
            raw_mid_sum
        )

        # ----------------------------------------------------
        # Convert settlement into contract outcome.
        # ----------------------------------------------------

        outcomes = []

        for _, row in (
            group.iterrows()
        ):

            result = row.get(
                "result"
            )

            result_text = (
                str(
                    result
                )
                .strip()
                .lower()
            )

            if result_text == "yes":

                outcome = 1.0

            elif result_text == "no":

                outcome = 0.0

            else:

                settlement_value = row.get(
                    "settlement_value"
                )

                if pd.isna(
                    settlement_value
                ):

                    outcome = np.nan

                else:

                    settlement_value = float(
                        settlement_value
                    )

                    if np.isclose(
                        settlement_value,
                        1.0
                    ):

                        outcome = 1.0

                    elif np.isclose(
                        settlement_value,
                        0.0
                    ):

                        outcome = 0.0

                    else:

                        outcome = np.nan

            outcomes.append(
                outcome
            )

        group[
            "outcome"
        ] = outcomes

        if group[
            "outcome"
        ].isna().any():

            continue

        # Exactly one bucket should have settled YES.
        if not np.isclose(
            group[
                "outcome"
            ].sum(),
            1.0
        ):

            continue

        for _, row in (
            group.iterrows()
        ):

            calibration_rows.append({

                "target_date":
                    target_date,

                "ticker":
                    row[
                        "ticker"
                    ],

                "market_probability":
                    float(
                        row[
                            "market_probability"
                        ]
                    ),

                "outcome":
                    float(
                        row[
                            "outcome"
                        ]
                    )
            })

    dataframe = pd.DataFrame(
        calibration_rows
    )

    if dataframe.empty:

        raise RuntimeError(
            "No complete settled calibration dates "
            "could be constructed."
        )

    return dataframe
# ============================================================
# HISTORICAL CALIBRATION
# ============================================================

def calculate_live_alpha(
    target_date
):

    dataframe = (
        load_market_only_calibration_history()
    )

    all_dates = sorted(
        dataframe[
            "target_date"
        ].unique()
    )

    training_dates = (
        get_training_dates(
            all_dates=
                all_dates,

            target_date=
                target_date
        )
    )

    if len(
        training_dates
    ) != LOOKBACK_DAYS:

        raise RuntimeError(
            f"Expected {LOOKBACK_DAYS} "
            f"training dates, found "
            f"{len(training_dates)}."
        )

    training_data = (
        dataframe[
            dataframe[
                "target_date"
            ].isin(
                training_dates
            )
        ]
        .copy()
    )

    (
        alpha,
        results
    ) = fit_market_alpha(
        training_data=
            training_data,

        alpha_values=
            ALPHA_VALUES
    )

    best_row = (
        results.loc[
            np.isclose(
                results[
                    "alpha"
                ],
                alpha
            )
        ]
        .iloc[
            0
        ]
    )

    latest_allowed_date = (
        pd.Timestamp(
            target_date
        )
        -
        pd.Timedelta(
            days=2
        )
    ).strftime(
        "%Y-%m-%d"
    )

    return {

        "alpha":
            float(
                alpha
            ),

        "training_start":
            training_dates[
                0
            ],

        "training_end":
            training_dates[
                -1
            ],

        "latest_allowed_date":
            latest_allowed_date,

        "training_log_loss":
            float(
                best_row[
                    "log_loss"
                ]
            ),

        "training_brier":
            float(
                best_row[
                    "brier"
                ]
            )
    }


# ============================================================
# LIVE KALSHI DISTRIBUTION
# ============================================================

def build_live_market_dataframe(
    markets
):

    if len(
        markets
    ) != 6:

        raise RuntimeError(
            f"Expected 6 live contracts, "
            f"found {len(markets)}."
        )

    rows = []

    for market in markets:

        yes_bid = market.get(
            "yes_bid"
        )

        yes_ask = market.get(
            "yes_ask"
        )

        no_ask = market.get(
            "no_ask"
        )

        if (
            yes_bid is None
            or
            yes_ask is None
            or
            no_ask is None
        ):

            raise RuntimeError(
                f"Incomplete quote for "
                f"{market['ticker']}."
            )

        yes_bid = float(
            yes_bid
        )

        yes_ask = float(
            yes_ask
        )

        no_ask = float(
            no_ask
        )

        midpoint = (
            yes_bid
            +
            yes_ask
        ) / 2.0

        rows.append({

            "ticker":
                market[
                    "ticker"
                ],

            "market_mid":
                midpoint,

            "yes_bid":
                yes_bid,

            "yes_ask":
                yes_ask,

            "no_ask":
                no_ask
        })

    dataframe = pd.DataFrame(
        rows
    )

    raw_mid_sum = float(
        dataframe[
            "market_mid"
        ].sum()
    )

    if raw_mid_sum <= 0:

        raise RuntimeError(
            "Invalid live midpoint sum."
        )

    dataframe[
        "market_probability"
    ] = (
        dataframe[
            "market_mid"
        ]
        /
        raw_mid_sum
    )

    return (
        dataframe,
        raw_mid_sum
    )


# ============================================================
# TRADE CANDIDATES
# ============================================================

def build_trade_candidates(
    market_data,
    alpha
):

    adjusted_yes = (
        market_only_probabilities(
            market_data,
            alpha
        )
    )

    candidates = []

    for index, row in (
        market_data.iterrows()
    ):

        market_yes = float(
            row[
                "market_probability"
            ]
        )

        adjusted_yes_probability = float(
            adjusted_yes[
                index
            ]
        )

        market_no = (
            1.0
            -
            market_yes
        )

        adjusted_no_probability = (
            1.0
            -
            adjusted_yes_probability
        )

        # ----------------------------------------------------
        # YES
        # ----------------------------------------------------

        yes_ask = float(
            row[
                "yes_ask"
            ]
        )

        yes_fee = float(
            calculate_taker_fee(
                price=
                    yes_ask,

                contracts=
                    1,

                multiplier=
                    1
            )
        )

        yes_cost = (
            yes_ask
            +
            yes_fee
        )

        candidates.append({

            "ticker":
                row[
                    "ticker"
                ],

            "side":
                "YES",

            "kalshi_probability":
                market_yes,

            "adjusted_probability":
                adjusted_yes_probability,

            "entry_price":
                yes_ask,

            "fee":
                yes_fee,

            "total_cost":
                yes_cost,

            "net_edge":
                (
                    adjusted_yes_probability
                    -
                    yes_cost
                )
        })

        # ----------------------------------------------------
        # NO
        # ----------------------------------------------------

        no_ask = float(
            row[
                "no_ask"
            ]
        )

        no_fee = float(
            calculate_taker_fee(
                price=
                    no_ask,

                contracts=
                    1,

                multiplier=
                    1
            )
        )

        no_cost = (
            no_ask
            +
            no_fee
        )

        candidates.append({

            "ticker":
                row[
                    "ticker"
                ],

            "side":
                "NO",

            "kalshi_probability":
                market_no,

            "adjusted_probability":
                adjusted_no_probability,

            "entry_price":
                no_ask,

            "fee":
                no_fee,

            "total_cost":
                no_cost,

            "net_edge":
                (
                    adjusted_no_probability
                    -
                    no_cost
                )
        })

    candidates.sort(
        key=lambda row:
            row[
                "net_edge"
            ],

        reverse=True
    )

    return candidates


# ============================================================
# TIMING SAFEGUARD
# ============================================================

def inside_save_window():

    now = pd.Timestamp.now(
        tz="UTC"
    )

    start = (
        now.normalize()
        +
        pd.Timedelta(
            hours=
                DECISION_HOUR_UTC,

            minutes=
                DECISION_MINUTE_UTC
        )
    )

    end = (
        start
        +
        pd.Timedelta(
            minutes=
                SAVE_WINDOW_MINUTES
        )
    )

    return (
        start
        <=
        now
        <=
        end
    )


# ============================================================
# SHADOW TRADER
# ============================================================

def run_shadow_trader(
    save=False,
    live=False
):
    if live and not save:

        raise RuntimeError(
            "--live requires --save. "
            "A real order cannot be submitted "
            "without saving the shadow decision."
        )

    event_info = (
        get_current_weather_event()
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

    decision_time = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

    # --------------------------------------------------------
    # Fit alpha using historical outcomes only.
    # --------------------------------------------------------

    calibration = (
        calculate_live_alpha(
            target_date
        )
    )

    alpha = (
        calibration[
            "alpha"
        ]
    )

    # --------------------------------------------------------
    # Build current Kalshi distribution.
    # --------------------------------------------------------

    (
        market_data,
        raw_mid_sum
    ) = (
        build_live_market_dataframe(
            markets
        )
    )

    candidates = (
        build_trade_candidates(
            market_data=
                market_data,

            alpha=
                alpha
        )
    )

    if not candidates:

        raise RuntimeError(
            "No trade candidates generated."
        )

    best = (
        candidates[
            0
        ]
    )

    trade_taken = (
        best[
            "net_edge"
        ]
        >=
        MINIMUM_EDGE
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "MARKET-ONLY V1 SHADOW TRADER"
    )
    print("=" * 80)

    print(
        f"Target date:       "
        f"{target_date}"
    )

    print(
        f"Decision time:     "
        f"{decision_time}"
    )

    print()

    print(
        f"Training window:   "
        f"{calibration['training_start']} "
        f"-> "
        f"{calibration['training_end']}"
    )

    print(
        f"Latest eligible:   "
        f"{calibration['latest_allowed_date']}"
    )

    print(
        f"Fitted alpha:      "
        f"{alpha:.3f}"
    )

    print(
        f"Training log loss: "
        f"{calibration['training_log_loss']:.4f}"
    )

    print(
        f"Training Brier:    "
        f"{calibration['training_brier']:.4f}"
    )

    print()

    print(
        f"Raw midpoint sum:  "
        f"{raw_mid_sum:.4f}"
    )

    print(
        f"Minimum edge:      "
        f"{MINIMUM_EDGE:.2%}"
    )

    # --------------------------------------------------------
    # Staleness warning
    # --------------------------------------------------------

    if (
        calibration[
            "training_end"
        ]
        <
        calibration[
            "latest_allowed_date"
        ]
    ):

        print()

        print(
            "WARNING: historical calibration "
            "dataset is not current through the "
            "latest eligible settled date."
        )

    # --------------------------------------------------------
    # Top candidates
    # --------------------------------------------------------

    print()
    print(
        "TOP CANDIDATES"
    )
    print("-" * 80)

    for candidate in (
        candidates[
            :6
        ]
    ):

        print(
            f"{candidate['ticker']:<30} "
            f"{candidate['side']:<3} "
            f"Kalshi: "
            f"{candidate['kalshi_probability']:.2%}  "
            f"Adjusted: "
            f"{candidate['adjusted_probability']:.2%}  "
            f"Cost: "
            f"{candidate['total_cost']:.2%}  "
            f"Edge: "
            f"{candidate['net_edge']:+.2%}"
        )

    print()

    if trade_taken:

        print(
            f"DECISION: "
            f"{best['side']} "
            f"{best['ticker']}"
        )

    else:

        print(
            "DECISION: PASS"
        )

        print(
            f"Best available edge: "
            f"{best['net_edge']:+.2%}"
        )

    print()

    print(
        f"Adjusted probability: "
        f"{best['adjusted_probability']:.2%}"
    )

    print(
        f"Entry ask:            "
        f"{best['entry_price']:.2%}"
    )

    print(
        f"Fee:                  "
        f"{best['fee']:.2%}"
    )

    print(
        f"Total cost:           "
        f"{best['total_cost']:.2%}"
    )

    print(
        f"Net edge:             "
        f"{best['net_edge']:+.2%}"
    )

    # --------------------------------------------------------
    # Build saved decision
    # --------------------------------------------------------

    decision = {

        "target_date":
            target_date,

        "decision_time":
            decision_time,

        "training_start":
            calibration[
                "training_start"
            ],

        "training_end":
            calibration[
                "training_end"
            ],

        "lookback_days":
            LOOKBACK_DAYS,

        "alpha":
            alpha,

        "raw_mid_sum":
            raw_mid_sum,

        "trade_taken":
            trade_taken,

        "ticker":
            best[
                "ticker"
            ],

        "side":
            best[
                "side"
            ],

        "kalshi_probability":
            best[
                "kalshi_probability"
            ],

        "adjusted_probability":
            best[
                "adjusted_probability"
            ],

        "entry_price":
            best[
                "entry_price"
            ],

        "fee":
            best[
                "fee"
            ],

        "total_cost":
            best[
                "total_cost"
            ],

        "net_edge":
            best[
                "net_edge"
            ],

        "minimum_edge":
            MINIMUM_EDGE
    }

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if not save:

        print()
        print(
            "DRY RUN ONLY - decision was NOT saved."
        )

        return decision

    # --------------------------------------------------------
    # Enforce decision timing
    # --------------------------------------------------------

    if not inside_save_window():

        raise RuntimeError(
            "Outside the allowed "
            "20:05-20:20 UTC save window."
        )

    save_v2_market_only_shadow_decision(
        decision
    )

    if (
        live
        and
        int(
            decision["trade_taken"]
        ) == 1
    ):

        print()
        print("=" * 80)
        print(
            "LIVE EXECUTION REQUESTED"
        )
        print("=" * 80)

        live_result = (
            execute_live_trade(
                trade=decision,
                strategy="MARKET_ONLY",
                contracts=1
            )
        )

        print()
        print(
            "REAL ORDER SUBMITTED."
        )

        print(
            f"Kalshi order ID: "
            f"{live_result.get('order_id')}"
        )

    print()

    if trade_taken:

        print(
            "MARKET-ONLY SHADOW TRADE SAVED."
        )

    else:

        print(
            "MARKET-ONLY PASS DECISION SAVED."
        )

    return decision


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        action="store_true"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Submit a qualifying market-only "
            "decision as a real Kalshi order."
        )
    )

    args = (
        parser.parse_args()
    )

    run_shadow_trader(
        save=args.save,
        live=args.live
    )


if __name__ == "__main__":

    main()