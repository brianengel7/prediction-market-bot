import math
import numpy as np

def round_temperature(temp_f):
    return int(
        np.floor(temp_f + 0.5)
    )

def get_rounded_member_highs(
    gefs_results
):
    return np.array([
        round_temperature(
            member["max_temperature"]
        )
        for member in gefs_results["members"]
    ])

def get_member_highs(gefs_results):

    return np.array([
        member["max_temperature"]
        for member in gefs_results["members"]
    ], dtype=float)


def calculate_kde_bandwidth(member_highs):

    member_highs = np.asarray(
        member_highs,
        dtype=float
    )

    n = len(member_highs)

    if n < 2:
        return 1.0

    std = np.std(
        member_highs,
        ddof=1
    )

    q25, q75 = np.percentile(
        member_highs,
        [25, 75]
    )

    iqr = q75 - q25

    robust_sigma = iqr / 1.34

    candidates = [
        value
        for value in [std, robust_sigma]
        if value > 0
    ]

    if not candidates:
        return 1.0

    sigma = min(candidates)

    bandwidth = (
        0.9
        * sigma
        * n ** (-1 / 5)
    )

    # Avoid an unrealistically tiny kernel.
    return max(
        float(bandwidth),
        0.25
    )


def normal_cdf(x, mean, std):

    z = (
        (x - mean)
        /
        (std * math.sqrt(2))
    )

    return 0.5 * (
        1 + math.erf(z)
    )


def ensemble_cdf(
    temperature,
    member_highs,
    bandwidth=None
):

    member_highs = np.asarray(
        member_highs,
        dtype=float
    )

    if bandwidth is None:
        bandwidth = calculate_kde_bandwidth(
            member_highs
        )

    probabilities = [
        normal_cdf(
            temperature,
            member_temperature,
            bandwidth
        )
        for member_temperature in member_highs
    ]

    return float(
        np.mean(probabilities)
    )


def calculate_smoothed_market_probability(
    market,
    gefs_results
):

    member_highs = get_member_highs(
        gefs_results
    )

    bandwidth = calculate_kde_bandwidth(
        member_highs
    )

    strike_type = market[
        "strike_type"
    ].lower()

    floor = market.get(
        "floor_strike"
    )

    cap = market.get(
        "cap_strike"
    )

    if strike_type == "less":

        if cap is None:
            raise ValueError(
                "Less market is missing cap_strike."
            )

        cap = float(cap)

        # round(T) < cap
        upper_boundary = cap - 0.5

        probability = ensemble_cdf(
            upper_boundary,
            member_highs,
            bandwidth
        )

    elif strike_type == "greater":

        if floor is None:
            raise ValueError(
                "Greater market is missing floor_strike."
            )

        floor = float(floor)

        # round(T) > floor
        lower_boundary = floor + 0.5

        probability = (
            1.0
            - ensemble_cdf(
                lower_boundary,
                member_highs,
                bandwidth
            )
        )

    elif strike_type == "between":

        if floor is None or cap is None:
            raise ValueError(
                "Between market is missing a strike."
            )

        floor = float(floor)
        cap = float(cap)

        lower_boundary = floor - 0.5
        upper_boundary = cap + 0.5

        probability = (
            ensemble_cdf(
                upper_boundary,
                member_highs,
                bandwidth
            )
            -
            ensemble_cdf(
                lower_boundary,
                member_highs,
                bandwidth
            )
        )

    else:
        raise ValueError(
            f"Unsupported strike type: "
            f"{strike_type}"
        )

    probability = max(
        0.0,
        min(1.0, probability)
    )

    return {
        "probability": probability,
        "bandwidth": bandwidth
    }