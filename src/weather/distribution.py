import numpy as np


def round_temperature(temp_f):

    return int(np.floor(temp_f + 0.5))


def get_rounded_member_highs(gefs_results):

    return np.array([
        round_temperature(
            member["max_temperature"]
        )
        for member in gefs_results["members"]
    ])


def calculate_bucket_probability(
    member_highs,
    lower=None,
    upper=None
):

    member_highs = np.asarray(member_highs)

    mask = np.ones(
        len(member_highs),
        dtype=bool
    )

    if lower is not None:
        mask &= member_highs >= lower

    if upper is not None:
        mask &= member_highs <= upper

    probability = np.mean(mask)

    return float(probability)


def build_bucket_probabilities(
    gefs_results,
    buckets
):

    member_highs = (
        get_rounded_member_highs(
            gefs_results
        )
    )

    results = []

    for bucket in buckets:

        probability = (
            calculate_bucket_probability(
                member_highs,
                lower=bucket.get("lower"),
                upper=bucket.get("upper")
            )
        )

        results.append({
            "label": bucket["label"],
            "lower": bucket.get("lower"),
            "upper": bucket.get("upper"),
            "probability": probability,
            "member_count": int(
                probability
                * len(member_highs)
            )
        })

    return results