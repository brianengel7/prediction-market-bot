from decimal import (
    Decimal,
    ROUND_CEILING
)


TAKER_FEE_RATE = Decimal(
    "0.07"
)


def round_fee_up(
    fee
):
    """
    Round a Kalshi fee upward
    to the next cent.
    """

    fee = Decimal(
        str(fee)
    )

    return fee.quantize(
        Decimal("0.01"),
        rounding=ROUND_CEILING
    )


def calculate_taker_fee(
    price,
    contracts=1,
    multiplier=1
):
    """
    Calculate Kalshi taker fee.

    fee =
        0.07 * M * C * P * (1 - P)
    """

    price = Decimal(
        str(price)
    )

    contracts = Decimal(
        str(contracts)
    )

    multiplier = Decimal(
        str(multiplier)
    )

    raw_fee = (
        TAKER_FEE_RATE
        * multiplier
        * contracts
        * price
        * (
            Decimal("1")
            - price
        )
    )

    return float(
        round_fee_up(
            raw_fee
        )
    )


if __name__ == "__main__":

    print(
        calculate_taker_fee(
            0.79
        )
    )

    print(
        calculate_taker_fee(
            0.76
        )
    )