"""Logical compatibility for hourly temperature threshold contracts."""
import re

PATTERN = re.compile(
    r"^(KXTEMPNYCHS-\d{2}[A-Z]{3}\d{4})-T(\d+(?:\.\d+)?)$"
)


def parse_leg(leg):
    match = PATTERN.fullmatch(str(leg["ticker"]))
    if not match:
        raise ValueError(
            f"Invalid hourly threshold ticker: {leg['ticker']}"
        )

    side = str(leg["side"]).upper()
    if side not in ("YES", "NO"):
        raise ValueError(f"Invalid side: {side}")

    return match.group(1), float(match.group(2)), side


def compatible(existing, candidate):
    """Require a nonempty region where all positions could win."""
    items = [*existing, candidate]
    parsed = [parse_leg(x) for x in items]

    if len({x[0] for x in parsed}) != 1:
        return False

    if len({x["ticker"] for x in items}) != len(items):
        return False

    lower = max(
        (x[1] for x in parsed if x[2] == "YES"),
        default=float("-inf"),
    )
    upper = min(
        (x[1] for x in parsed if x[2] == "NO"),
        default=float("inf"),
    )

    return lower < upper


if __name__ == "__main__":
    event = "KXTEMPNYCHS-26OCT0812"

    def leg(threshold, side):
        return {
            "ticker": f"{event}-T{threshold}.99",
            "side": side,
        }

    assert compatible([leg(65, "YES")], leg(70, "NO"))
    assert compatible([leg(65, "YES")], leg(68, "YES"))
    assert not compatible([leg(70, "YES")], leg(65, "NO"))
    assert not compatible([leg(65, "YES")], leg(65, "NO"))
    assert not compatible([leg(65, "YES")], leg(65, "YES"))

    print("PASS: compatibility and duplicate protection")
