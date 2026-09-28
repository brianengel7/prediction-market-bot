import math

import numpy as np
import pandas as pd

from scipy.stats import norm

from src.weather.calibrated_ensemble import (
    load_historical_data,
    build_training_matrix,
    calculate_calibration_parameters,
    calibrate_forecasts,
    calculate_weighted_forecast
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TRAIN_DAYS = 30

# After the first calibrated forecasts are produced,
# wait until we have this many true out-of-sample
# residuals before trusting an uncertainty distribution.
MIN_RESIDUAL_DAYS = 15

SOURCES = [
    "GEFS",
    "GFS_MOS",
    "HRRR",
    "NBM"
]

# Wide enough to capture essentially all probability
# mass for NYC daily maximum temperatures.
MIN_TEMPERATURE = 0
MAX_TEMPERATURE = 130

EPSILON = 1e-12


# ============================================================
# SETTLEMENT HELPERS
# ============================================================

def settlement_temperature(
    actual_high
):
    """
    Convert the observed daily maximum into the integer
    temperature used by our settlement probability model.

    Example:
        65.2 -> 65
        65.6 -> 66
    """

    return int(
        math.floor(
            float(actual_high)
            + 0.5
        )
    )


def integer_temperature_probability(
    temperature,
    point_forecast,
    residual_mean,
    residual_std
):
    """
    Probability that the settled integer temperature
    equals `temperature`.

    A settlement at 66°F corresponds to the latent
    interval:

        [65.5, 66.5)
    """

    center = (
        float(point_forecast)
        + float(residual_mean)
    )

    lower = (
        float(temperature)
        - 0.5
    )

    upper = (
        float(temperature)
        + 0.5
    )

    probability = (
        norm.cdf(
            upper,
            loc=center,
            scale=residual_std
        )
        -
        norm.cdf(
            lower,
            loc=center,
            scale=residual_std
        )
    )

    return float(
        probability
    )


def build_integer_probabilities(
    point_forecast,
    residual_mean,
    residual_std
):
    """
    Build the full integer settlement distribution.
    """

    probabilities = {}

    for temperature in range(
        MIN_TEMPERATURE,
        MAX_TEMPERATURE + 1
    ):

        probabilities[
            temperature
        ] = (
            integer_temperature_probability(
                temperature=
                    temperature,

                point_forecast=
                    point_forecast,

                residual_mean=
                    residual_mean,

                residual_std=
                    residual_std
            )
        )

    total_probability = sum(
        probabilities.values()
    )

    # This should already be extremely close to 1.
    # Normalizing removes any tiny numerical tail loss.
    if total_probability > 0:

        probabilities = {
            temperature:
                probability
                / total_probability

            for (
                temperature,
                probability
            )
            in probabilities.items()
        }

    return probabilities


# ============================================================
# SCORING
# ============================================================

def calculate_multiclass_brier(
    probabilities,
    actual_temperature
):
    """
    Multiclass Brier score.

    Perfect forecast = 0.
    Lower is better.
    """

    score = 0.0

    for (
        temperature,
        probability
    ) in probabilities.items():

        if temperature == actual_temperature:
            outcome = 1.0
        else:
            outcome = 0.0

        score += (
            probability
            - outcome
        ) ** 2

    return float(
        score
    )


def calculate_actual_temperature_nll(
    probabilities,
    actual_temperature
):
    """
    Negative log likelihood of the actual
    settlement temperature.

    Lower is better.
    """

    probability = probabilities.get(
        actual_temperature,
        0.0
    )

    probability = max(
        probability,
        EPSILON
    )

    return float(
        -math.log(
            probability
        )
    )


def calculate_ranked_probability_score(
    probabilities,
    actual_temperature
):
    """
    Ranked Probability Score.

    This is useful because temperatures are ordered.

    Being wrong by 1°F should be treated differently
    from being wrong by 10°F.
    """

    temperatures = sorted(
        probabilities.keys()
    )

    cumulative_forecast = 0.0
    score = 0.0

    for temperature in temperatures:

        cumulative_forecast += (
            probabilities[
                temperature
            ]
        )

        if temperature >= actual_temperature:
            cumulative_actual = 1.0
        else:
            cumulative_actual = 0.0

        score += (
            cumulative_forecast
            - cumulative_actual
        ) ** 2

    return float(
        score
    )


# ============================================================
# INTERVAL COVERAGE
# ============================================================

def calculate_interval(
    point_forecast,
    residual_mean,
    residual_std,
    coverage
):
    """
    Return the central normal prediction interval.
    """

    center = (
        float(point_forecast)
        + float(residual_mean)
    )

    alpha = (
        1.0
        - coverage
    )

    lower_quantile = (
        alpha / 2.0
    )

    upper_quantile = (
        1.0
        - alpha / 2.0
    )

    lower = norm.ppf(
        lower_quantile,
        loc=center,
        scale=residual_std
    )

    upper = norm.ppf(
        upper_quantile,
        loc=center,
        scale=residual_std
    )

    return (
        float(lower),
        float(upper)
    )


# ============================================================
# WALK-FORWARD PROBABILITY BACKTEST
# ============================================================

def run_probability_backtest():
    """
    Run a true walk-forward probability backtest.

    For each historical target date:

    1. Train bias/weights using dates before the target.
    2. Generate the target day's calibrated point forecast.
    3. Use only PRIOR out-of-sample residuals to estimate
       residual mean/std.
    4. Generate today's probability distribution.
    5. Score it against the actual settlement.
    6. Only after scoring, add today's residual to history.

    This prevents look-ahead.
    """

    historical_data = (
        load_historical_data()
    )

    training = (
        build_training_matrix(
            historical_data
        )
    )

    if len(training) <= (
        MIN_TRAIN_DAYS
        + MIN_RESIDUAL_DAYS
    ):

        raise ValueError(
            "Not enough historical data "
            "for probability backtesting."
        )

    scored_rows = []

    reliability_rows = []

    past_residuals = []

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    for index in range(
        MIN_TRAIN_DAYS,
        len(training)
    ):

        # Everything BEFORE today's date.
        prior_training = training.iloc[
            :index
        ]

        current = training.iloc[
            index
        ]

        target_date = training.index[
            index
        ]

        # ----------------------------------------------------
        # CALIBRATION PARAMETERS
        # ----------------------------------------------------

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

        point_forecast = (
            calculate_weighted_forecast(
                corrected_forecasts,
                parameters
            )
        )

        actual_high = float(
            current[
                "actual_high"
            ]
        )

        actual_temperature = (
            settlement_temperature(
                actual_high
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Probability scoring happens BEFORE today's
        # residual is added to past_residuals.
        # ----------------------------------------------------

        if len(
            past_residuals
        ) >= MIN_RESIDUAL_DAYS:

            residual_mean = float(
                np.mean(
                    past_residuals
                )
            )

            residual_std = float(
                np.std(
                    past_residuals,
                    ddof=1
                )
            )

            if residual_std <= 0:

                raise ValueError(
                    "Residual standard deviation "
                    "must be positive."
                )

            probabilities = (
                build_integer_probabilities(
                    point_forecast=
                        point_forecast,

                    residual_mean=
                        residual_mean,

                    residual_std=
                        residual_std
                )
            )

            actual_probability = (
                probabilities.get(
                    actual_temperature,
                    0.0
                )
            )

            brier = (
                calculate_multiclass_brier(
                    probabilities,
                    actual_temperature
                )
            )

            nll = (
                calculate_actual_temperature_nll(
                    probabilities,
                    actual_temperature
                )
            )

            rps = (
                calculate_ranked_probability_score(
                    probabilities,
                    actual_temperature
                )
            )

            # ------------------------------------------------
            # PREDICTION INTERVALS
            # ------------------------------------------------

            interval_50 = (
                calculate_interval(
                    point_forecast,
                    residual_mean,
                    residual_std,
                    0.50
                )
            )

            interval_80 = (
                calculate_interval(
                    point_forecast,
                    residual_mean,
                    residual_std,
                    0.80
                )
            )

            interval_90 = (
                calculate_interval(
                    point_forecast,
                    residual_mean,
                    residual_std,
                    0.90
                )
            )

            inside_50 = (
                interval_50[0]
                <= actual_high
                <= interval_50[1]
            )

            inside_80 = (
                interval_80[0]
                <= actual_high
                <= interval_80[1]
            )

            inside_90 = (
                interval_90[0]
                <= actual_high
                <= interval_90[1]
            )

            scored_rows.append({
                "target_date":
                    target_date,

                "actual_high":
                    actual_high,

                "actual_temperature":
                    actual_temperature,

                "point_forecast":
                    point_forecast,

                "residual_mean":
                    residual_mean,

                "residual_std":
                    residual_std,

                "distribution_center":
                    point_forecast
                    + residual_mean,

                "actual_probability":
                    actual_probability,

                "brier":
                    brier,

                "nll":
                    nll,

                "rps":
                    rps,

                "inside_50":
                    inside_50,

                "inside_80":
                    inside_80,

                "inside_90":
                    inside_90,

                "residual_count":
                    len(
                        past_residuals
                    )
            })

            # ------------------------------------------------
            # RELIABILITY DATA
            # ------------------------------------------------

            for (
                temperature,
                probability
            ) in probabilities.items():

                if temperature == actual_temperature:
                    outcome = 1
                else:
                    outcome = 0

                # Ignore microscopic tail probabilities
                # unless this was the actual outcome.
                if (
                    probability >= 0.01
                    or outcome == 1
                ):

                    reliability_rows.append({
                        "target_date":
                            target_date,

                        "temperature":
                            temperature,

                        "probability":
                            probability,

                        "outcome":
                            outcome
                    })

        # ----------------------------------------------------
        # NOW today's outcome is allowed to enter history.
        # ----------------------------------------------------

        residual = (
            actual_high
            - point_forecast
        )

        past_residuals.append(
            residual
        )

    scored = pd.DataFrame(
        scored_rows
    )

    reliability = pd.DataFrame(
        reliability_rows
    )

    return (
        scored,
        reliability
    )


# ============================================================
# RELIABILITY TABLE
# ============================================================

def build_reliability_table(
    reliability
):
    if reliability.empty:
        return pd.DataFrame()

    bins = [
        0.00,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        1.00
    ]

    labels = [
        "0-2%",
        "2-5%",
        "5-10%",
        "10-15%",
        "15-20%",
        "20-30%",
        "30%+"
    ]

    data = reliability.copy()

    data[
        "probability_bin"
    ] = pd.cut(
        data[
            "probability"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    table = (
        data.groupby(
            "probability_bin",
            observed=True
        )
        .agg(
            forecasts=(
                "probability",
                "size"
            ),

            mean_forecast=(
                "probability",
                "mean"
            ),

            actual_frequency=(
                "outcome",
                "mean"
            )
        )
        .reset_index()
    )

    return table


# ============================================================
# SUMMARY
# ============================================================

def print_probability_summary(
    scored,
    reliability
):
    print()
    print("=" * 70)
    print(
        "PROBABILITY CALIBRATION BACKTEST"
    )
    print("=" * 70)

    print(
        f"Forecast dates: "
        f"{len(scored)}"
    )

    if scored.empty:

        print(
            "No probability forecasts "
            "were scored."
        )

        return

    print()

    print(
        "PROBABILITY SCORES"
    )

    print("-" * 70)

    print(
        f"Mean multiclass Brier: "
        f"{scored['brier'].mean():.4f}"
    )

    print(
        f"Mean actual-temp NLL:  "
        f"{scored['nll'].mean():.4f}"
    )

    print(
        f"Mean ranked score:     "
        f"{scored['rps'].mean():.4f}"
    )

    print(
        f"Mean probability given "
        f"to actual outcome: "
        f"{scored['actual_probability'].mean():.2%}"
    )

    print()

    print(
        "INTERVAL COVERAGE"
    )

    print("-" * 70)

    print(
        f"50% interval: "
        f"{scored['inside_50'].mean():.2%}"
    )

    print(
        f"80% interval: "
        f"{scored['inside_80'].mean():.2%}"
    )

    print(
        f"90% interval: "
        f"{scored['inside_90'].mean():.2%}"
    )

    reliability_table = (
        build_reliability_table(
            reliability
        )
    )

    print()
    print(
        "RELIABILITY"
    )

    print("-" * 70)

    if reliability_table.empty:

        print(
            "No reliability data."
        )

    else:

        display_table = (
            reliability_table.copy()
        )

        display_table[
            "mean_forecast"
        ] = (
            display_table[
                "mean_forecast"
            ]
            * 100
        )

        display_table[
            "actual_frequency"
        ] = (
            display_table[
                "actual_frequency"
            ]
            * 100
        )

        print(
            display_table.to_string(
                index=False,

                formatters={
                    "mean_forecast":
                        lambda x:
                        f"{x:.2f}%",

                    "actual_frequency":
                        lambda x:
                        f"{x:.2f}%"
                }
            )
        )

    print()
    print(
        "DAILY SCORES"
    )

    print("-" * 70)

    display_columns = [
        "target_date",
        "actual_high",
        "point_forecast",
        "distribution_center",
        "residual_std",
        "actual_probability",
        "brier",
        "nll",
        "rps"
    ]

    print(
        scored[
            display_columns
        ].to_string(
            index=False,

            formatters={
                "actual_high":
                    lambda x:
                    f"{x:.2f}",

                "point_forecast":
                    lambda x:
                    f"{x:.2f}",

                "distribution_center":
                    lambda x:
                    f"{x:.2f}",

                "residual_std":
                    lambda x:
                    f"{x:.2f}",

                "actual_probability":
                    lambda x:
                    f"{x:.2%}",

                "brier":
                    lambda x:
                    f"{x:.4f}",

                "nll":
                    lambda x:
                    f"{x:.4f}",

                "rps":
                    lambda x:
                    f"{x:.4f}"
            }
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    scored, reliability = (
        run_probability_backtest()
    )

    print_probability_summary(
        scored,
        reliability
    )