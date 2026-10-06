import re

import numpy as np
import pandas as pd

from src.backtest.v2_market_only_control import (
    ALPHA_VALUES,
    kalshi_probabilities,
    normalize_probabilities,
    score_model,
)


# ============================================================
# SHIFT SEARCH GRID
# ============================================================

# Positive = shift probability toward hotter temperatures.
# Negative = shift probability toward cooler temperatures.
#
# Start reasonably conservative. If an optimum repeatedly
# hits +/- 3 F, that is a diagnostic rather than a reason
# to silently widen the grid.
SHIFT_VALUES = np.arange(
    -3.0,
    3.0001,
    0.25,
)


# ============================================================
# MARKET BUCKET GEOMETRY
# ============================================================

def _ticker_suffix(
    ticker
):
    return (
        str(ticker)
        .strip()
        .upper()
        .rsplit("-", 1)[-1]
    )


def _parse_between_center(
    ticker
):
    """
    Interior Kalshi daily-high buckets use Bxx.x tickers.

    Example:
        B69.5 -> center 69.5 F
        B71.5 -> center 71.5 F
    """

    suffix = _ticker_suffix(
        ticker
    )

    match = re.fullmatch(
        r"B(-?\d+(?:\.\d+)?)",
        suffix,
    )

    if not match:

        return None

    return float(
        match.group(1)
    )


def _parse_tail_threshold(
    ticker
):
    """
    Tail buckets use Txx tickers.

    The numeric threshold lets us determine whether the
    contract is the cold tail or hot tail relative to
    the interior buckets.
    """

    suffix = _ticker_suffix(
        ticker
    )

    match = re.fullmatch(
        r"T(-?\d+(?:\.\d+)?)",
        suffix,
    )

    if not match:

        return None

    return float(
        match.group(1)
    )


def market_bucket_centers(
    group
):
    """
    Return one representative temperature center for each
    contract, in the SAME row order as `group`.

    Interior bucket centers come directly from B tickers.

    The two open-ended tail buckets are represented one
    normal bucket-width beyond the nearest interior center.

    Example:

        T69
        B69.5
        B71.5
        B73.5
        B75.5
        T76

    becomes approximately:

        67.5
        69.5
        71.5
        73.5
        75.5
        77.5

    This gives us an ordered temperature axis on which to
    move probability mass without pretending the tails have
    known finite bounds.
    """

    if len(group) != 6:

        raise ValueError(
            f"Expected 6 contracts, "
            f"found {len(group)}."
        )

    tickers = (
        group[
            "ticker"
        ]
        .astype(str)
        .tolist()
    )

    interior = []

    tails = []

    for index, ticker in enumerate(
        tickers
    ):

        center = (
            _parse_between_center(
                ticker
            )
        )

        if center is not None:

            interior.append(
                (
                    index,
                    center,
                )
            )

            continue

        threshold = (
            _parse_tail_threshold(
                ticker
            )
        )

        if threshold is not None:

            tails.append(
                (
                    index,
                    threshold,
                )
            )

            continue

        raise ValueError(
            f"Unsupported weather ticker: "
            f"{ticker}"
        )

    if len(interior) < 2:

        raise ValueError(
            "Need at least two interior "
            "temperature buckets."
        )

    if len(tails) != 2:

        raise ValueError(
            f"Expected 2 tail contracts, "
            f"found {len(tails)}."
        )

    interior_values = sorted(
        center
        for _, center in interior
    )

    spacings = np.diff(
        interior_values
    )

    bucket_width = float(
        np.median(
            spacings
        )
    )

    if (
        not np.isfinite(
            bucket_width
        )
        or
        bucket_width <= 0
    ):

        raise ValueError(
            "Could not infer a valid "
            "temperature bucket width."
        )

    # Require regular interior buckets.
    if not np.allclose(
        spacings,
        bucket_width,
        atol=1e-6,
    ):

        raise ValueError(
            f"Irregular bucket spacing: "
            f"{spacings}"
        )

    min_center = min(
        interior_values
    )

    max_center = max(
        interior_values
    )

    low_tail = min(
        tails,
        key=lambda item:
            item[1],
    )

    high_tail = max(
        tails,
        key=lambda item:
            item[1],
    )

    centers = np.full(
        len(group),
        np.nan,
        dtype=float,
    )

    for index, center in interior:

        centers[
            index
        ] = center

    centers[
        low_tail[0]
    ] = (
        min_center
        -
        bucket_width
    )

    centers[
        high_tail[0]
    ] = (
        max_center
        +
        bucket_width
    )

    if not np.isfinite(
        centers
    ).all():

        raise ValueError(
            "Could not construct all "
            "temperature bucket centers."
        )

    if len(
        np.unique(
            centers
        )
    ) != len(
        centers
    ):

        raise ValueError(
            "Temperature bucket centers "
            "are not unique."
        )

    return centers


# ============================================================
# SHIFT PROBABILITY MASS
# ============================================================

def shift_probability_mass(
    probabilities,
    centers,
    shift_f
):
    """
    Shift a discrete probability distribution by `shift_f`
    degrees Fahrenheit.

    Positive:
        move probability toward hotter buckets.

    Negative:
        move probability toward cooler buckets.

    Fractional shifts are split linearly between neighboring
    temperature buckets.

    Probability pushed beyond an extreme bucket remains in
    that open-ended tail bucket.
    """

    probabilities = (
        normalize_probabilities(
            probabilities
        )
    )

    centers = np.asarray(
        centers,
        dtype=float,
    )

    if (
        len(probabilities)
        !=
        len(centers)
    ):

        raise ValueError(
            "Probabilities and centers "
            "must have equal lengths."
        )

    order = np.argsort(
        centers
    )

    sorted_centers = (
        centers[
            order
        ]
    )

    sorted_probabilities = (
        probabilities[
            order
        ]
    )

    shifted = np.zeros(
        len(
            sorted_probabilities
        ),
        dtype=float,
    )

    shift_f = float(
        shift_f
    )

    for (
        source_center,
        probability,
    ) in zip(
        sorted_centers,
        sorted_probabilities,
    ):

        destination = (
            source_center
            +
            shift_f
        )

        # Cold-tail overflow.
        if (
            destination
            <=
            sorted_centers[0]
        ):

            shifted[0] += (
                probability
            )

            continue

        # Hot-tail overflow.
        if (
            destination
            >=
            sorted_centers[-1]
        ):

            shifted[-1] += (
                probability
            )

            continue

        right = int(
            np.searchsorted(
                sorted_centers,
                destination,
                side="right",
            )
        )

        left = (
            right
            -
            1
        )

        left_center = (
            sorted_centers[
                left
            ]
        )

        right_center = (
            sorted_centers[
                right
            ]
        )

        span = (
            right_center
            -
            left_center
        )

        if span <= 0:

            raise ValueError(
                "Temperature centers "
                "must be increasing."
            )

        right_weight = (
            (
                destination
                -
                left_center
            )
            /
            span
        )

        left_weight = (
            1.0
            -
            right_weight
        )

        shifted[
            left
        ] += (
            probability
            *
            left_weight
        )

        shifted[
            right
        ] += (
            probability
            *
            right_weight
        )

    # Restore original dataframe row order.
    restored = np.empty(
        len(
            shifted
        ),
        dtype=float,
    )

    restored[
        order
    ] = shifted

    return normalize_probabilities(
        restored
    )


# ============================================================
# MODELS
# ============================================================

def market_shift_probabilities(
    group,
    shift_f
):
    market = (
        kalshi_probabilities(
            group
        )
    )

    centers = (
        market_bucket_centers(
            group
        )
    )

    return (
        shift_probability_mass(
            probabilities=
                market,

            centers=
                centers,

            shift_f=
                shift_f,
        )
    )


def market_alpha_shift_probabilities(
    group,
    alpha,
    shift_f
):
    """
    Model order:

        raw market
            ->
        temperature shift
            ->
        alpha sharpening / flattening
    """

    shifted = (
        market_shift_probabilities(
            group=
                group,

            shift_f=
                shift_f,
        )
    )

    sharpened = (
        shifted
        **
        float(
            alpha
        )
    )

    return normalize_probabilities(
        sharpened
    )


# ============================================================
# FIT SHIFT ONLY
# ============================================================

def fit_market_shift(
    training_data,
    shift_values=None
):
    if shift_values is None:

        shift_values = (
            SHIFT_VALUES
        )

    results = []

    for shift_f in shift_values:

        scores = (
            score_model(
                training_data,

                lambda group:
                    market_shift_probabilities(
                        group=
                            group,

                        shift_f=
                            shift_f,
                    )
            )
        )

        if scores.empty:

            continue

        results.append({
            "shift_f":
                float(
                    shift_f
                ),

            "log_loss":
                float(
                    scores[
                        "log_loss"
                    ].mean()
                ),

            "brier":
                float(
                    scores[
                        "brier"
                    ].mean()
                ),
        })

    results = pd.DataFrame(
        results
    )

    if results.empty:

        raise RuntimeError(
            "No valid shift results."
        )

    best = (
        results
        .sort_values(
            [
                "log_loss",
                "brier",
                "shift_f",
            ]
        )
        .iloc[0]
    )

    return (
        float(
            best[
                "shift_f"
            ]
        ),
        results,
    )


# ============================================================
# FIT ALPHA + SHIFT
# ============================================================

def fit_market_alpha_shift(
    training_data,
    alpha_values=None,
    shift_values=None
):
    if alpha_values is None:

        alpha_values = (
            ALPHA_VALUES
        )

    if shift_values is None:

        shift_values = (
            SHIFT_VALUES
        )

    results = []

    for shift_f in shift_values:

        for alpha in alpha_values:

            scores = (
                score_model(
                    training_data,

                    lambda group,
                    alpha=alpha,
                    shift_f=shift_f:
                        market_alpha_shift_probabilities(
                            group=
                                group,

                            alpha=
                                alpha,

                            shift_f=
                                shift_f,
                        )
                )
            )

            if scores.empty:

                continue

            results.append({
                "alpha":
                    float(
                        alpha
                    ),

                "shift_f":
                    float(
                        shift_f
                    ),

                "log_loss":
                    float(
                        scores[
                            "log_loss"
                        ].mean()
                    ),

                "brier":
                    float(
                        scores[
                            "brier"
                        ].mean()
                    ),
            })

    results = pd.DataFrame(
        results
    )

    if results.empty:

        raise RuntimeError(
            "No valid alpha-shift "
            "results."
        )

    best = (
        results
        .sort_values(
            [
                "log_loss",
                "brier",
                "alpha",
                "shift_f",
            ]
        )
        .iloc[0]
    )

    return (
        float(
            best[
                "alpha"
            ]
        ),

        float(
            best[
                "shift_f"
            ]
        ),

        results,
    )