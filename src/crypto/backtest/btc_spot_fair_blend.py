import math

import numpy as np
import pandas as pd

from scipy.special import ndtr

from src.crypto.backtest.alpha_walkforward import (
    TRAINING_LOOKBACK_DAYS,
    MINIMUM_TRAINING_ROWS,
    apply_alpha,
    get_periods,
)

from src.crypto.backtest.btc_offset_walkforward import (
    load_feature_data,
    brier_score,
    log_loss,
    improvement,
    fit_alpha_for_training,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROBABILITY_EPSILON = 1e-6

BETA_GRID = np.arange(
    0.0,
    1.0001,
    0.025,
)


# ============================================================
# HELPERS
# ============================================================

def clip_probability(
    probabilities,
):

    return np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        PROBABILITY_EPSILON,
        1.0
        -
        PROBABILITY_EPSILON,
    )


def logit(
    probabilities,
):

    probabilities = (
        clip_probability(
            probabilities
        )
    )

    return np.log(
        probabilities
        /
        (
            1.0
            -
            probabilities
        )
    )


def sigmoid(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    result = np.empty_like(
        values
    )

    positive = (
        values >= 0
    )

    result[
        positive
    ] = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -values[
                    positive
                ]
            )
        )
    )

    exp_values = np.exp(
        values[
            ~positive
        ]
    )

    result[
        ~positive
    ] = (
        exp_values
        /
        (
            1.0
            +
            exp_values
        )
    )

    return result


# ============================================================
# EXTERNAL BTC FAIR VALUE
# ============================================================

def calculate_spot_fair(
    dataframe,
):

    return_since_open = (
        dataframe[
            "return_10m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rv_15 = (
        dataframe[
            "realized_vol_15m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rv_60 = (
        dataframe[
            "realized_vol_60m"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Convert realized volatility into approximate
    # one-minute variance.
    #
    # RV_N^2 ≈ N * variance_per_minute
    # --------------------------------------------------------

    variance_per_minute_15 = (
        rv_15 ** 2
        /
        15.0
    )

    variance_per_minute_60 = (
        rv_60 ** 2
        /
        60.0
    )

    # --------------------------------------------------------
    # Fixed 50/50 variance blend.
    #
    # This is deliberately NOT tuned on validation.
    # Short window captures current regime;
    # long window stabilizes it.
    # --------------------------------------------------------

    variance_per_minute = (
        0.5
        *
        variance_per_minute_15
        +
        0.5
        *
        variance_per_minute_60
    )

    variance_per_minute = np.maximum(
        variance_per_minute,
        1e-12,
    )

    # Five minutes remain.
    sigma_remaining = np.sqrt(
        5.0
        *
        variance_per_minute
    )

    z_score = (
        return_since_open
        /
        sigma_remaining
    )

    spot_probability = ndtr(
        z_score
    )

    return (
        clip_probability(
            spot_probability
        ),
        z_score,
        sigma_remaining,
    )


# ============================================================
# LOGIT BLEND
# ============================================================

def blend_probabilities(
    alpha_probability,
    spot_probability,
    beta,
):

    alpha_logit = logit(
        alpha_probability
    )

    spot_logit = logit(
        spot_probability
    )

    blended_logit = (
        (
            1.0
            -
            beta
        )
        *
        alpha_logit
        +
        beta
        *
        spot_logit
    )

    return (
        clip_probability(
            sigmoid(
                blended_logit
            )
        )
    )


# ============================================================
# BETA FIT
# ============================================================

def fit_beta(
    alpha_probability,
    spot_probability,
    outcomes,
):

    best_beta = None
    best_loss = None

    for beta in BETA_GRID:

        probability = (
            blend_probabilities(
                alpha_probability=
                    alpha_probability,

                spot_probability=
                    spot_probability,

                beta=
                    float(
                        beta
                    ),
            )
        )

        loss = log_loss(
            probability,
            outcomes,
        )

        if (
            best_loss is None
            or loss < best_loss
        ):

            best_loss = loss

            best_beta = float(
                beta
            )

    return (
        best_beta,
        best_loss,
    )


# ============================================================
# WALK FORWARD
# ============================================================

def run_walkforward(
    dataframe,
):

    (
        spot_probability,
        spot_z,
        sigma_remaining,
    ) = (
        calculate_spot_fair(
            dataframe
        )
    )

    dataframe = (
        dataframe
        .copy()
    )

    dataframe[
        "spot_probability"
    ] = (
        spot_probability
    )

    dataframe[
        "spot_z"
    ] = (
        spot_z
    )

    dataframe[
        "sigma_remaining"
    ] = (
        sigma_remaining
    )

    periods = get_periods(
        dataframe
    )

    validation_start = (
        periods[
            "validation_start"
        ]
    )

    validation_end = (
        periods[
            "validation_end"
        ]
    )

    validation_dates = (
        dataframe.loc[
            (
                dataframe[
                    "date"
                ]
                >=
                validation_start
            )
            &
            (
                dataframe[
                    "date"
                ]
                <=
                validation_end
            ),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    prediction_frames = []

    daily_scores = []

    for decision_date in validation_dates:

        training_start = (
            decision_date
            -
            pd.Timedelta(
                days=
                    TRAINING_LOOKBACK_DAYS
            )
        )

        training_end = (
            decision_date
            -
            pd.Timedelta(
                days=1
            )
        )

        training_data = (
            dataframe[
                (
                    dataframe[
                        "date"
                    ]
                    >=
                    training_start
                )
                &
                (
                    dataframe[
                        "date"
                    ]
                    <=
                    training_end
                )
            ]
            .copy()
        )

        evaluation_data = (
            dataframe[
                dataframe[
                    "date"
                ]
                ==
                decision_date
            ]
            .copy()
        )

        if (
            len(
                training_data
            )
            <
            MINIMUM_TRAINING_ROWS
        ):

            continue

        if evaluation_data.empty:

            continue

        # ----------------------------------------------------
        # Rolling alpha
        # ----------------------------------------------------

        alpha = (
            fit_alpha_for_training(
                training_data
            )
        )

        training_alpha = (
            apply_alpha(
                training_data[
                    "kalshi_midpoint"
                ]
                .to_numpy(
                    dtype=float
                ),
                alpha,
            )
        )

        evaluation_alpha = (
            apply_alpha(
                evaluation_data[
                    "kalshi_midpoint"
                ]
                .to_numpy(
                    dtype=float
                ),
                alpha,
            )
        )

        # ----------------------------------------------------
        # Fit one BTC-weight parameter using training only
        # ----------------------------------------------------

        beta, _ = (
            fit_beta(
                alpha_probability=
                    training_alpha,

                spot_probability=
                    training_data[
                        "spot_probability"
                    ]
                    .to_numpy(
                        dtype=float
                    ),

                outcomes=
                    training_data[
                        "outcome"
                    ]
                    .to_numpy(
                        dtype=float
                    ),
            )
        )

        blended_probability = (
            blend_probabilities(
                alpha_probability=
                    evaluation_alpha,

                spot_probability=
                    evaluation_data[
                        "spot_probability"
                    ]
                    .to_numpy(
                        dtype=float
                    ),

                beta=
                    beta,
            )
        )

        raw_probability = (
            evaluation_data[
                "kalshi_midpoint"
            ]
            .to_numpy(
                dtype=float
            )
        )

        evaluation_spot = (
            evaluation_data[
                "spot_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        outcomes = (
            evaluation_data[
                "outcome"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        prediction_frame = (
            evaluation_data[
                [
                    "ticker",
                    "date",
                    "quote_time",
                    "outcome",

                    "kalshi_yes_ask",
                    "kalshi_no_ask",

                    "kalshi_midpoint",
                    "spot_probability",
                    "spot_z",
                    "sigma_remaining",
                ]
            ]
            .copy()
        )

        prediction_frame[
            "alpha_probability"
        ] = (
            evaluation_alpha
        )

        prediction_frame[
            "blend_probability"
        ] = (
            blended_probability
        )

        prediction_frame[
            "alpha"
        ] = alpha

        prediction_frame[
            "beta"
        ] = beta

        prediction_frames.append(
            prediction_frame
        )

        # ----------------------------------------------------
        # Daily scores
        # ----------------------------------------------------

        daily_scores.append(
            {
                "date":
                    decision_date,

                "rows":
                    len(
                        evaluation_data
                    ),

                "training_rows":
                    len(
                        training_data
                    ),

                "alpha":
                    alpha,

                "beta":
                    beta,

                "raw_brier":
                    brier_score(
                        raw_probability,
                        outcomes,
                    ),

                "alpha_brier":
                    brier_score(
                        evaluation_alpha,
                        outcomes,
                    ),

                "spot_brier":
                    brier_score(
                        evaluation_spot,
                        outcomes,
                    ),

                "blend_brier":
                    brier_score(
                        blended_probability,
                        outcomes,
                    ),

                "raw_log_loss":
                    log_loss(
                        raw_probability,
                        outcomes,
                    ),

                "alpha_log_loss":
                    log_loss(
                        evaluation_alpha,
                        outcomes,
                    ),

                "spot_log_loss":
                    log_loss(
                        evaluation_spot,
                        outcomes,
                    ),

                "blend_log_loss":
                    log_loss(
                        blended_probability,
                        outcomes,
                    ),
            }
        )

    if not prediction_frames:

        raise RuntimeError(
            "No validation predictions generated."
        )

    return (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),

        pd.DataFrame(
            daily_scores
        ),

        periods,
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    predictions,
    daily_scores,
):

    outcomes = (
        predictions[
            "outcome"
        ]
        .to_numpy(
            dtype=float
        )
    )

    model_definitions = [
        (
            "RAW KALSHI",
            predictions[
                "kalshi_midpoint"
            ],
        ),

        (
            "ALPHA",
            predictions[
                "alpha_probability"
            ],
        ),

        (
            "BTC SPOT ONLY",
            predictions[
                "spot_probability"
            ],
        ),

        (
            "ALPHA + BTC BLEND",
            predictions[
                "blend_probability"
            ],
        ),
    ]

    rows = []

    for (
        label,
        probabilities,
    ) in model_definitions:

        probabilities = (
            np.asarray(
                probabilities,
                dtype=float,
            )
        )

        rows.append(
            {
                "model":
                    label,

                "rows":
                    len(
                        outcomes
                    ),

                "brier":
                    brier_score(
                        probabilities,
                        outcomes,
                    ),

                "log_loss":
                    log_loss(
                        probabilities,
                        outcomes,
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    print()
    print("=" * 100)
    print(
        "VALIDATION MODEL COMPARISON"
    )
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            formatters={
                "brier":
                    lambda value:
                        f"{value:.6f}",

                "log_loss":
                    lambda value:
                        f"{value:.6f}",
            },
        )
    )

    alpha_row = (
        summary.loc[
            summary[
                "model"
            ]
            ==
            "ALPHA"
        ]
        .iloc[
            0
        ]
    )

    blend_row = (
        summary.loc[
            summary[
                "model"
            ]
            ==
            "ALPHA + BTC BLEND"
        ]
        .iloc[
            0
        ]
    )

    alpha_brier_improvement = (
        improvement(
            alpha_row[
                "brier"
            ],
            blend_row[
                "brier"
            ],
        )
    )

    alpha_log_improvement = (
        improvement(
            alpha_row[
                "log_loss"
            ],
            blend_row[
                "log_loss"
            ],
        )
    )

    print()
    print(
        "BLEND VS ALPHA"
    )

    print(
        f"Brier improvement:        "
        f"{alpha_brier_improvement:+.3%}"
    )

    print(
        f"Log-loss improvement:     "
        f"{alpha_log_improvement:+.3%}"
    )

    # --------------------------------------------------------
    # Beta
    # --------------------------------------------------------

    beta = (
        daily_scores[
            "beta"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print()
    print(
        "ROLLING BTC WEIGHT"
    )

    print(
        f"Mean beta:                "
        f"{np.mean(beta):.4f}"
    )

    print(
        f"Median beta:              "
        f"{np.median(beta):.4f}"
    )

    print(
        f"Minimum beta:             "
        f"{np.min(beta):.4f}"
    )

    print(
        f"Maximum beta:             "
        f"{np.max(beta):.4f}"
    )

    print(
        f"Days beta = 0:            "
        f"{int(np.sum(np.isclose(beta, 0.0)))}"
        f"/{len(beta)}"
    )

    print(
        f"Days beta >= 0.10:        "
        f"{int(np.sum(beta >= 0.10))}"
        f"/{len(beta)}"
    )

    # --------------------------------------------------------
    # Day-level comparison
    # --------------------------------------------------------

    beats_alpha = (
        (
            daily_scores[
                "blend_brier"
            ]
            <
            daily_scores[
                "alpha_brier"
            ]
        )
        &
        (
            daily_scores[
                "blend_log_loss"
            ]
            <
            daily_scores[
                "alpha_log_loss"
            ]
        )
    )

    print()
    print(
        f"Blend beats alpha both:   "
        f"{int(beats_alpha.sum())}"
        f"/{len(daily_scores)} days"
    )

    # --------------------------------------------------------
    # External probability diagnostic
    # --------------------------------------------------------

    print()
    print(
        "BTC SPOT FAIR DIAGNOSTICS"
    )

    print(
        f"Mean spot probability:    "
        f"{predictions['spot_probability'].mean():.4f}"
    )

    print(
        f"Mean |spot - alpha|:      "
        f"{np.mean(
            np.abs(
                predictions[
                    'spot_probability'
                ]
                -
                predictions[
                    'alpha_probability'
                ]
            )
        ):.4f}"
    )

    print(
        f"Mean spot z-score:        "
        f"{predictions['spot_z'].mean():+.4f}"
    )

    print(
        f"Spot z-score std:         "
        f"{predictions['spot_z'].std():.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = (
        load_feature_data()
    )

    required = [
        "return_10m",
        "realized_vol_15m",
        "realized_vol_60m",
    ]

    missing = [
        column
        for column in required
        if column
        not in dataframe.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: "
            f"{missing}"
        )

    (
        predictions,
        daily_scores,
        periods,
    ) = (
        run_walkforward(
            dataframe
        )
    )

    print()
    print("=" * 100)
    print(
        "BTC SPOT-FAIR BLEND WALK-FORWARD"
    )
    print("=" * 100)

    print(
        f"Validation:               "
        f"{periods['validation_start'].date()} "
        f"-> "
        f"{periods['validation_end'].date()}"
    )

    print(
        f"RESERVED TEST:            "
        f"{periods['test_start'].date()} "
        f"-> "
        f"{periods['test_end'].date()}"
    )

    print(
        f"Training lookback:        "
        f"{TRAINING_LOOKBACK_DAYS} days"
    )

    print(
        f"Validation rows:          "
        f"{len(predictions)}"
    )

    print(
        f"Validation days:          "
        f"{daily_scores['date'].nunique()}"
    )

    print()
    print(
        "External fair:"
    )

    print(
        "P(YES) = Phi("
        "BTC return since market open "
        "/ expected remaining 5m sigma)"
    )

    print()

    print(
        "Blend:"
    )

    print(
        "logit(P) = "
        "(1-beta)*logit(alpha Kalshi) "
        "+ beta*logit(BTC fair)"
    )

    print()

    print(
        "Beta is fitted using PRIOR 60-day "
        "training data only."
    )

    print(
        "Reserved test is NOT evaluated."
    )

    summarize(
        predictions=
            predictions,

        daily_scores=
            daily_scores,
    )


if __name__ == "__main__":

    main()