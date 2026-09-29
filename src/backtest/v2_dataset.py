from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.analysis import (
    load_backtest_data
)

from src.backtest.probability_calibration import (
    run_probability_backtest
)

from src.weather.calibrated_ensemble import (
    load_historical_data,
    build_training_matrix,
    calculate_calibration_parameters,
    calibrate_forecasts
)

from src.weather.fair_distribution import (
    probability_for_market
)

from src.database.db import (
    get_historical_market_entries
)


# ============================================================
# CONFIG
# ============================================================

START_DATE = "2026-07-19"
END_DATE = "2026-09-26"

DECISION_TIME = "20:05"

SOURCES = [
    "GEFS",
    "GFS_MOS",
    "HRRR",
    "NBM"
]

OUTPUT_PATH = (
    Path(__file__)
    .resolve()
    .parents[2]
    / "data"
    / "research"
    / "weather_market_v2.csv"
)


# ============================================================
# WEATHER FEATURES
# ============================================================

def build_weather_features():

    scored, _ = (
        run_probability_backtest()
    )

    scored = scored.copy()

    scored["target_date"] = (
        pd.to_datetime(
            scored["target_date"]
        )
    )

    # --------------------------------------------------------
    # Historical raw forecasts
    # --------------------------------------------------------

    historical = (
        load_historical_data()
    )

    training_matrix = (
        build_training_matrix(
            historical
        )
    )

    # --------------------------------------------------------
    # Extra forecast uncertainty fields
    # --------------------------------------------------------

    raw_backtests = (
        load_backtest_data()
    )

    raw_backtests[
        "target_date"
    ] = pd.to_datetime(
        raw_backtests[
            "target_date"
        ]
    )

    forecast_std = (
        raw_backtests
        .pivot_table(
            index="target_date",
            columns="source",
            values="forecast_std",
            aggfunc="first"
        )
    )

    rows = []

    for _, scored_row in scored.iterrows():

        target_date = (
            scored_row[
                "target_date"
            ]
        )

        if (
            target_date
            not in training_matrix.index
        ):
            continue

        # IMPORTANT:
        # only use dates BEFORE target date
        prior_training = (
            training_matrix[
                training_matrix.index
                <
                target_date
            ]
        )

        current = (
            training_matrix.loc[
                target_date
            ]
        )

        parameters = (
            calculate_calibration_parameters(
                prior_training
            )
        )

        raw_forecasts = {
            source:
                float(
                    current[
                        source
                    ]
                )

            for source in SOURCES
        }

        corrected_forecasts = (
            calibrate_forecasts(
                raw_forecasts,
                parameters
            )
        )

        raw_values = np.array(
            list(
                raw_forecasts.values()
            ),
            dtype=float
        )

        corrected_values = np.array(
            list(
                corrected_forecasts.values()
            ),
            dtype=float
        )

        row = {

            "target_date":
                target_date,

            # --------------------------------------------
            # Existing walk-forward v1 distribution
            # --------------------------------------------

            "point_forecast":
                float(
                    scored_row[
                        "point_forecast"
                    ]
                ),

            "residual_mean":
                float(
                    scored_row[
                        "residual_mean"
                    ]
                ),

            "residual_std":
                float(
                    scored_row[
                        "residual_std"
                    ]
                ),

            "distribution_center":
                float(
                    scored_row[
                        "distribution_center"
                    ]
                ),

            # --------------------------------------------
            # Model consensus
            # --------------------------------------------

            "raw_consensus_mean":
                float(
                    raw_values.mean()
                ),

            "raw_consensus_std":
                float(
                    raw_values.std(
                        ddof=0
                    )
                ),

            "raw_consensus_range":
                float(
                    raw_values.max()
                    -
                    raw_values.min()
                ),

            "corrected_consensus_mean":
                float(
                    corrected_values.mean()
                ),

            "corrected_consensus_std":
                float(
                    corrected_values.std(
                        ddof=0
                    )
                )
        }

        # --------------------------------------------
        # Individual weather models
        # --------------------------------------------

        for source in SOURCES:

            row[
                f"{source}_raw"
            ] = (
                raw_forecasts[
                    source
                ]
            )

            row[
                f"{source}_corrected"
            ] = (
                corrected_forecasts[
                    source
                ]
            )

            row[
                f"{source}_bias"
            ] = float(
                parameters[
                    "biases"
                ][source]
            )

            row[
                f"{source}_weight"
            ] = float(
                parameters[
                    "weights"
                ][source]
            )

        # --------------------------------------------
        # Native ensemble dispersion
        # --------------------------------------------

        row[
            "GEFS_ensemble_std"
        ] = np.nan

        row[
            "NBM_forecast_std"
        ] = np.nan

        if (
            target_date
            in forecast_std.index
        ):

            if "GEFS" in forecast_std.columns:

                value = forecast_std.loc[
                    target_date,
                    "GEFS"
                ]

                if not pd.isna(
                    value
                ):
                    row[
                        "GEFS_ensemble_std"
                    ] = float(
                        value
                    )

            if "NBM" in forecast_std.columns:

                value = forecast_std.loc[
                    target_date,
                    "NBM"
                ]

                if not pd.isna(
                    value
                ):
                    row[
                        "NBM_forecast_std"
                    ] = float(
                        value
                    )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MARKET FEATURES
# ============================================================

def build_market_features():

    markets = (
        get_historical_market_entries(
            start_date=
                START_DATE,

            end_date=
                END_DATE
        )
    )

    markets = markets.copy()

    markets[
        "target_date"
    ] = pd.to_datetime(
        markets[
            "target_date"
        ]
    )

    markets[
        "decision_time"
    ] = pd.to_datetime(
        markets[
            "decision_time"
        ],
        utc=True
    )

    markets[
        "entry_time"
    ] = pd.to_datetime(
        markets[
            "entry_time"
        ],
        utc=True
    )

    markets[
        "decision_label"
    ] = (
        markets[
            "decision_time"
        ]
        .dt.strftime(
            "%H:%M"
        )
    )

    # --------------------------------------------------------
    # Use one fixed market time initially
    # --------------------------------------------------------

    markets = markets[
        markets[
            "decision_label"
        ]
        ==
        DECISION_TIME
    ].copy()

    # --------------------------------------------------------
    # Only complete six-contract days
    # --------------------------------------------------------

    counts = (
        markets
        .groupby(
            "target_date"
        )
        .size()
    )

    complete_dates = (
        counts[
            counts == 6
        ]
        .index
    )

    markets = markets[
        markets[
            "target_date"
        ].isin(
            complete_dates
        )
    ].copy()

    # --------------------------------------------------------
    # Market probability
    # --------------------------------------------------------

    markets[
        "market_mid"
    ] = (
        (
            markets[
                "yes_bid"
            ].astype(float)
            +
            markets[
                "yes_ask"
            ].astype(float)
        )
        / 2.0
    )

    markets[
        "market_spread"
    ] = (
        markets[
            "yes_ask"
        ].astype(float)
        -
        markets[
            "yes_bid"
        ].astype(float)
    )

    # Normalize the six contract mids so
    # daily probability mass equals 1.
    markets[
        "market_probability"
    ] = (
        markets[
            "market_mid"
        ]
        /
        markets
        .groupby(
            "target_date"
        )[
            "market_mid"
        ]
        .transform(
            "sum"
        )
    )

    markets[
        "quote_delay_minutes"
    ] = (
        (
            markets[
                "entry_time"
            ]
            -
            markets[
                "decision_time"
            ]
        )
        .dt.total_seconds()
        / 60.0
    )

    markets[
        "outcome"
    ] = (
        markets[
            "result"
        ]
        .astype(str)
        .str.lower()
        .eq(
            "yes"
        )
        .astype(int)
    )

    return markets


# ============================================================
# CONTRACT GEOMETRY
# ============================================================

def get_contract_reference(
    row
):

    strike_type = (
        row[
            "strike_type"
        ]
    )

    floor_strike = (
        row[
            "floor_strike"
        ]
    )

    cap_strike = (
        row[
            "cap_strike"
        ]
    )

    if strike_type == "between":

        return (
            (
                float(
                    floor_strike
                )
                +
                float(
                    cap_strike
                )
            )
            / 2.0
        )

    if strike_type == "less":

        return (
            float(
                cap_strike
            )
            - 0.5
        )

    if strike_type == "greater":

        return (
            float(
                floor_strike
            )
            + 0.5
        )

    return np.nan


# ============================================================
# BUILD V2 DATASET
# ============================================================

def build_v2_dataset():

    weather = (
        build_weather_features()
    )

    markets = (
        build_market_features()
    )

    data = markets.merge(
        weather,
        on="target_date",
        how="inner"
    )

    # --------------------------------------------------------
    # Contract location
    # --------------------------------------------------------

    data[
        "contract_reference"
    ] = (
        data.apply(
            get_contract_reference,
            axis=1
        )
    )

    # --------------------------------------------------------
    # V1 model probability
    # --------------------------------------------------------

    data[
        "v1_model_yes"
    ] = data.apply(

        lambda row:
            probability_for_market(

                strike_type=
                    row[
                        "strike_type"
                    ],

                floor_strike=
                    row[
                        "floor_strike"
                    ],

                cap_strike=
                    row[
                        "cap_strike"
                    ],

                point_forecast=
                    row[
                        "point_forecast"
                    ],

                residual_mean=
                    row[
                        "residual_mean"
                    ],

                residual_std=
                    row[
                        "residual_std"
                    ]
            ),

        axis=1
    )

    # --------------------------------------------------------
    # Model vs market disagreement
    # --------------------------------------------------------

    data[
        "v1_minus_market"
    ] = (
        data[
            "v1_model_yes"
        ]
        -
        data[
            "market_probability"
        ]
    )

    # --------------------------------------------------------
    # Log-odds
    # --------------------------------------------------------

    epsilon = 0.001

    market_probability = (
        data[
            "market_probability"
        ]
        .clip(
            epsilon,
            1.0 - epsilon
        )
    )

    data[
        "market_logit"
    ] = np.log(
        market_probability
        /
        (
            1.0
            -
            market_probability
        )
    )

    v1_probability = (
        data[
            "v1_model_yes"
        ]
        .clip(
            epsilon,
            1.0 - epsilon
        )
    )

    data[
        "v1_logit"
    ] = np.log(
        v1_probability
        /
        (
            1.0
            -
            v1_probability
        )
    )

    # --------------------------------------------------------
    # Weather-model distance from contract
    # --------------------------------------------------------

    for source in SOURCES:

        data[
            f"{source}_raw_distance"
        ] = (
            data[
                f"{source}_raw"
            ]
            -
            data[
                "contract_reference"
            ]
        )

        data[
            f"{source}_corrected_distance"
        ] = (
            data[
                f"{source}_corrected"
            ]
            -
            data[
                "contract_reference"
            ]
        )

    data[
        "consensus_distance"
    ] = (
        data[
            "distribution_center"
        ]
        -
        data[
            "contract_reference"
        ]
    )

    # --------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "target_date",
            "ticker"
        ]
    ).reset_index(
        drop=True
    )

    return data


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    data
):

    print()
    print("=" * 100)
    print(
        "WEATHER MARKET V2 DATASET"
    )
    print("=" * 100)

    print(
        f"Decision time:   "
        f"{DECISION_TIME} UTC"
    )

    print(
        f"Rows:            "
        f"{len(data)}"
    )

    print(
        f"Dates:           "
        f"{data['target_date'].nunique()}"
    )

    print(
        f"Contracts/date:  "
        f"{len(data) / data['target_date'].nunique():.2f}"
    )

    print(
        f"YES outcome rate:"
        f" {data['outcome'].mean():.2%}"
    )

    print()

    market_mass = (
        data
        .groupby(
            "target_date"
        )[
            "market_probability"
        ]
        .sum()
    )

    model_mass = (
        data
        .groupby(
            "target_date"
        )[
            "v1_model_yes"
        ]
        .sum()
    )

    print(
        "PROBABILITY MASS"
    )

    print(
        f"Market min/max: "
        f"{market_mass.min():.6f} / "
        f"{market_mass.max():.6f}"
    )

    print(
        f"V1 min/max:     "
        f"{model_mass.min():.6f} / "
        f"{model_mass.max():.6f}"
    )

    print()

    preview_columns = [
        "target_date",
        "ticker",
        "strike_type",
        "market_probability",
        "v1_model_yes",
        "v1_minus_market",
        "GEFS_raw",
        "GFS_MOS_raw",
        "HRRR_raw",
        "NBM_raw",
        "raw_consensus_std",
        "GEFS_ensemble_std",
        "outcome"
    ]

    print(
        data[
            preview_columns
        ]
        .head(12)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = (
        build_v2_dataset()
    )

    validate_dataset(
        dataset
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        f"Saved dataset to:"
    )

    print(
        OUTPUT_PATH
    )