import math

from scipy.stats import norm


def get_distribution_parameters(
    point_forecast,
    residual_mean,
    residual_std
):
    """
    Convert the calibrated ensemble point
    forecast into the center and spread of
    the settlement distribution.

    Residual convention:
        residual = actual - forecast
    """

    mean_temperature = (
        float(point_forecast)
        + float(residual_mean)
    )

    std_temperature = float(
        residual_std
    )

    return (
        mean_temperature,
        std_temperature
    )


def probability_integer_temperature(
    temperature,
    mean_temperature,
    std_temperature
):
    """
    Probability that the official rounded
    daily high settles at one integer °F.

    Example:
        settlement = 85°F

    corresponds approximately to latent
    temperature:

        84.5 <= T < 85.5
    """

    lower = (
        temperature
        - 0.5
    )

    upper = (
        temperature
        + 0.5
    )

    return (
        norm.cdf(
            upper,
            loc=mean_temperature,
            scale=std_temperature
        )
        -
        norm.cdf(
            lower,
            loc=mean_temperature,
            scale=std_temperature
        )
    )


def build_integer_distribution(
    point_forecast,
    residual_mean,
    residual_std,
    num_std=5
):
    """
    Build settlement probabilities for
    integer Fahrenheit temperatures.
    """

    (
        mean_temperature,
        std_temperature
    ) = get_distribution_parameters(
        point_forecast,
        residual_mean,
        residual_std
    )

    minimum = math.floor(
        mean_temperature
        - num_std
        * std_temperature
    )

    maximum = math.ceil(
        mean_temperature
        + num_std
        * std_temperature
    )

    probabilities = {}

    for temperature in range(
        minimum,
        maximum + 1
    ):

        probability = (
            probability_integer_temperature(
                temperature,
                mean_temperature,
                std_temperature
            )
        )

        probabilities[
            temperature
        ] = probability

    total = sum(
        probabilities.values()
    )

    # Normalize tiny missing tail mass.
    if total > 0:

        probabilities = {
            temperature:
                probability / total

            for (
                temperature,
                probability
            ) in probabilities.items()
        }

    return probabilities


def probability_between(
    floor_temperature,
    cap_temperature,
    mean_temperature,
    std_temperature
):
    """
    Inclusive settlement interval.

    Example:
        84-85°F means official high
        settles at either 84°F or 85°F.

    Continuous boundaries therefore are:
        83.5 <= T < 85.5
    """

    lower = (
        floor_temperature
        - 0.5
    )

    upper = (
        cap_temperature
        + 0.5
    )

    return (
        norm.cdf(
            upper,
            loc=mean_temperature,
            scale=std_temperature
        )
        -
        norm.cdf(
            lower,
            loc=mean_temperature,
            scale=std_temperature
        )
    )


def probability_less_than(
    cap_temperature,
    mean_temperature,
    std_temperature
):
    """
    Example Kalshi contract:

        '< 84°F'

    Winning settlements:
        ... 81, 82, 83

    Continuous cutoff:
        83.5°F
    """

    cutoff = (
        cap_temperature
        - 0.5
    )

    return norm.cdf(
        cutoff,
        loc=mean_temperature,
        scale=std_temperature
    )


def probability_greater_than(
    floor_temperature,
    mean_temperature,
    std_temperature
):
    """
    Example Kalshi contract:

        '> 87°F'

    Winning settlements:
        88, 89, ...

    Continuous cutoff:
        87.5°F
    """

    cutoff = (
        floor_temperature
        + 0.5
    )

    return (
        1.0
        - norm.cdf(
            cutoff,
            loc=mean_temperature,
            scale=std_temperature
        )
    )


def probability_for_market(
    strike_type,
    floor_strike,
    cap_strike,
    point_forecast,
    residual_mean,
    residual_std
):
    """
    Calculate calibrated YES probability
    for a Kalshi temperature contract.
    """

    (
        mean_temperature,
        std_temperature
    ) = get_distribution_parameters(
        point_forecast,
        residual_mean,
        residual_std
    )

    if strike_type == "between":

        return probability_between(
            floor_strike,
            cap_strike,
            mean_temperature,
            std_temperature
        )

    if strike_type == "less":

        return probability_less_than(
            cap_strike,
            mean_temperature,
            std_temperature
        )

    if strike_type == "greater":

        return probability_greater_than(
            floor_strike,
            mean_temperature,
            std_temperature
        )

    raise ValueError(
        f"Unknown strike type: "
        f"{strike_type}"
    )

if __name__ == "__main__":

    point_forecast = 85.0

    residual_mean = 0.147
    residual_std = 2.256

    distribution = (
        build_integer_distribution(
            point_forecast,
            residual_mean,
            residual_std
        )
    )

    print()
    print(
        "CALIBRATED TEMPERATURE DISTRIBUTION"
    )

    print("-" * 50)

    for temperature, probability in (
        distribution.items()
    ):

        if probability >= 0.001:

            print(
                f"{temperature}°F: "
                f"{probability * 100:6.2f}%"
            )

    print()

    probability = probability_for_market(
        strike_type="between",
        floor_strike=84,
        cap_strike=85,
        point_forecast=point_forecast,
        residual_mean=residual_mean,
        residual_std=residual_std
    )

    print(
        f"84-85°F YES probability: "
        f"{probability * 100:.2f}%"
    )