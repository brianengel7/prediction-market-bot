import base64
import os
import time
import uuid

from decimal import (
    Decimal,
    ROUND_HALF_UP
)

from urllib.parse import (
    urlparse
)

import requests

from cryptography.hazmat.primitives import (
    hashes,
    serialization
)

from cryptography.hazmat.primitives.asymmetric import (
    padding
)

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BASE_URL = (
    "https://external-api.kalshi.com"
    "/trade-api/v2"
)

ORDER_PATH = (
    "/portfolio/events/orders"
)

FILLS_PATH = (
    "/portfolio/fills"
)

ORDERS_PATH = (
    "/portfolio/orders"
)

class OrderSubmissionUnknownError(
    RuntimeError
):
    """
    The order may or may not have reached Kalshi.

    Never assume the order failed.
    """


# ============================================================
# CREDENTIALS
# ============================================================

def get_api_key_id():

    api_key_id = os.getenv(
        "KALSHI_API_KEY_ID"
    )

    if not api_key_id:

        raise RuntimeError(
            "KALSHI_API_KEY_ID "
            "environment variable is not set."
        )

    return api_key_id


def get_private_key_path():

    private_key_path = os.getenv(
        "KALSHI_PRIVATE_KEY_PATH"
    )

    if not private_key_path:

        raise RuntimeError(
            "KALSHI_PRIVATE_KEY_PATH "
            "environment variable is not set."
        )

    if not os.path.exists(
        private_key_path
    ):

        raise RuntimeError(
            f"Kalshi private key file "
            f"does not exist: "
            f"{private_key_path}"
        )

    return private_key_path


def load_private_key():

    private_key_path = (
        get_private_key_path()
    )

    with open(
        private_key_path,
        "rb"
    ) as key_file:

        private_key = (
            serialization
            .load_pem_private_key(
                key_file.read(),
                password=None
            )
        )

    return private_key


# ============================================================
# SIGNING
# ============================================================

def sign_text(
    private_key,
    text
):

    message = (
        text.encode(
            "utf-8"
        )
    )

    if isinstance(
        private_key,
        Ed25519PrivateKey
    ):

        signature = (
            private_key.sign(
                message
            )
        )

    else:

        signature = (
            private_key.sign(
                message,

                padding.PSS(
                    mgf=
                        padding.MGF1(
                            hashes.SHA256()
                        ),

                    salt_length=
                        padding.PSS
                        .DIGEST_LENGTH
                ),

                hashes.SHA256()
            )
        )

    return (
        base64
        .b64encode(
            signature
        )
        .decode(
            "utf-8"
        )
    )


def build_auth_headers(
    method,
    url
):

    timestamp = str(
        int(
            time.time()
            * 1000
        )
    )

    parsed_url = (
        urlparse(
            url
        )
    )

    # Kalshi requires the URL path
    # WITHOUT query parameters.
    path = (
        parsed_url.path
    )

    message = (
        timestamp
        +
        method.upper()
        +
        path
    )

    private_key = (
        load_private_key()
    )

    signature = (
        sign_text(
            private_key=
                private_key,

            text=
                message
        )
    )

    return {
        "KALSHI-ACCESS-KEY":
            get_api_key_id(),

        "KALSHI-ACCESS-TIMESTAMP":
            timestamp,

        "KALSHI-ACCESS-SIGNATURE":
            signature,

        "Content-Type":
            "application/json"
    }


# ============================================================
# PRICE HELPERS
# ============================================================

def normalize_price(
    price
):
    """
    Convert a probability / dollar price into
    Kalshi fixed-point dollar format.

    Example:
        0.47 -> "0.4700"
    """

    price = Decimal(
        str(
            price
        )
    )

    if (
        price <= Decimal("0")
        or
        price >= Decimal("1")
    ):

        raise ValueError(
            f"Invalid Kalshi price: "
            f"{price}"
        )

    price = (
        price.quantize(
            Decimal("0.0001"),
            rounding=
                ROUND_HALF_UP
        )
    )

    return format(
        price,
        ".4f"
    )


def normalize_count(
    contracts
):

    count = Decimal(
        str(
            contracts
        )
    )

    if count <= 0:

        raise ValueError(
            "Contract count "
            "must be positive."
        )

    count = (
        count.quantize(
            Decimal("0.01")
        )
    )

    return format(
        count,
        ".2f"
    )


# ============================================================
# CONVERT OUR YES / NO TRADE INTO KALSHI V2 ORDER
# ============================================================

def convert_trade_to_v2_order(
    ticker,
    side,
    entry_price,
    contracts=1,
    client_order_id=None
):

    side = (
        str(
            side
        )
        .strip()
        .upper()
    )

    outcome_price = Decimal(
        str(
            entry_price
        )
    )

    if side == "YES":

        # Buy YES.
        book_side = "bid"

        yes_price = (
            outcome_price
        )

    elif side == "NO":

        # Buying NO at X is economically
        # equivalent to selling YES at 1-X.
        book_side = "ask"

        yes_price = (
            Decimal("1")
            -
            outcome_price
        )

    else:

        raise ValueError(
            f"Invalid trade side: "
            f"{side}"
        )

    if client_order_id is None:

        client_order_id = str(
            uuid.uuid4()
        )

    return {

        "ticker":
            ticker,

        "side":
            book_side,

        "count":
            normalize_count(
                contracts
            ),

        "price":
            normalize_price(
                yes_price
            ),

        "time_in_force":
            "fill_or_kill",

        "self_trade_prevention_type":
            "taker_at_cross",

        "client_order_id":
            client_order_id
    }


# ============================================================
# PLACE REAL ORDER
# ============================================================

def place_order(
    ticker,
    side,
    entry_price,
    contracts=1,
    client_order_id=None,
    timeout=15
):
    """
    PLACE A REAL KALSHI ORDER.

    This function performs a production POST
    against Kalshi's trading API.
    """

    base_url = os.getenv(
        "KALSHI_BASE_URL",
        DEFAULT_BASE_URL
    )

    url = (
        base_url
        +
        ORDER_PATH
    )

    order = (
        convert_trade_to_v2_order(
            ticker=
                ticker,

            side=
                side,

            entry_price=
                entry_price,

            contracts=
                contracts,

            client_order_id=
                client_order_id
        )
    )

    headers = (
        build_auth_headers(
            method=
                "POST",

            url=
                url
        )
    )

    try:
        response = (
            requests.post(
                url,
                headers=
                    headers,
                json=
                    order,
                timeout=
                    timeout
            )
        )

    except requests.exceptions.RequestException as error:

        raise OrderSubmissionUnknownError(
            "Kalshi order submission state "
            "is UNKNOWN because the HTTP "
            "request failed after submission "
            "may have begun.\n"
            f"{type(error).__name__}: "
            f"{error}"
        ) from error


    # ------------------------------------------------------------
    # These responses are ambiguous enough that we do NOT
    # assume the order failed.
    # ------------------------------------------------------------

    if (
        response.status_code
        in (
            408,
            409
        )
        or
        response.status_code >= 500
    ):

        raise OrderSubmissionUnknownError(
            "Kalshi order submission state "
            "is UNKNOWN.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )


    # ------------------------------------------------------------
    # Normal definitive rejection
    # ------------------------------------------------------------

    if response.status_code != 201:

        raise RuntimeError(
            f"Kalshi order failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    try:
        result = response.json()
        if (
            not isinstance(result, dict)
            or not result.get("order_id")
            or result.get("fill_count") is None
            or result.get("remaining_count") is None
        ):
            raise ValueError("HTTP 201 response is missing order fields")
    except (ValueError, TypeError) as error:
        raise OrderSubmissionUnknownError(
            "Kalshi returned HTTP 201, but its order response "
            "could not be verified. Check client_order_id before retrying."
        ) from error

    return {
        "request":
            order,

        "response":
            result,

        "order_id":
            result.get(
                "order_id"
            ),

        "client_order_id":
            order[
                "client_order_id"
            ],

        "ticker":
            ticker,

        "outcome_side":
            side.upper(),

        "requested_entry_price":
            float(
                entry_price
            ),

        "contracts":
            float(
                contracts
            ),

        "fill_count":
            result.get(
                "fill_count"
            ),

        "remaining_count":
            result.get(
                "remaining_count"
            ),

        "average_fill_price":
            result.get(
                "average_fill_price"
            ),

        "average_fee_paid":
            result.get(
                "average_fee_paid"
            )
    }

def get_orders(
    ticker=None,
    status=None,
    limit=1000,
    timeout=15
):
    """
    Retrieve authenticated Kalshi orders.
    """

    base_url = os.getenv(
        "KALSHI_BASE_URL",
        DEFAULT_BASE_URL
    )

    url = (
        base_url
        +
        ORDERS_PATH
    )

    all_orders = []
    cursor = None
    seen_cursors = set()

    while True:
        params = {"limit": limit}

        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            url,
            headers=build_auth_headers(method="GET", url=url),
            params=params,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Kalshi order lookup failed.\n"
                f"HTTP {response.status_code}\n{response.text}"
            )

        data = response.json()
        all_orders.extend(data.get("orders", []))

        next_cursor = data.get("cursor")
        if not next_cursor:
            return all_orders

        if next_cursor in seen_cursors:
            raise RuntimeError(
                "Kalshi order lookup repeated a pagination cursor."
            )

        seen_cursors.add(next_cursor)
        cursor = next_cursor


def find_order_by_client_order_id(
    client_order_id,
    ticker=None
):
    """
    Search recent Kalshi orders for our
    deterministic client_order_id.
    """

    if not client_order_id:

        raise ValueError(
            "client_order_id is required."
        )

    statuses = (
        "resting",
        "executed",
        "canceled"
    )

    for status in statuses:

        orders = (
            get_orders(
                ticker=
                    ticker,

                status=
                    status
            )
        )

        for order in orders:

            if (
                str(
                    order.get(
                        "client_order_id"
                    )
                )
                ==
                str(
                    client_order_id
                )
            ):

                return order

    return None

def get_order_fills(
    order_id,
    timeout=15
):
    """
    Retrieve actual Kalshi fills for one order.
    """

    if not order_id:

        raise ValueError(
            "order_id is required."
        )

    base_url = os.getenv(
        "KALSHI_BASE_URL",
        DEFAULT_BASE_URL
    )

    url = (
        base_url
        +
        FILLS_PATH
    )

    headers = build_auth_headers(
        method="GET",
        url=url
    )

    response = requests.get(
        url,
        headers=headers,
        params={
            "order_id":
                order_id,

            "limit":
                100
        },
        timeout=timeout
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Kalshi fill lookup failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()

    return data.get(
        "fills",
        []
    )

def summarize_order_fills(
    fills,
    outcome_side
):
    """
    Aggregate all fills belonging to one order.

    Returns:
        - actual filled contracts
        - average outcome-side execution price
        - total fees
        - average fee per contract
    """

    if not fills:

        return {
            "fill_count":
                0.0,

            "average_fill_price":
                None,

            "total_fee":
                0.0,

            "average_fee_paid":
                None
        }

    outcome_side = (
        str(outcome_side)
        .strip()
        .upper()
    )

    if outcome_side not in (
        "YES",
        "NO"
    ):

        raise ValueError(
            f"Invalid outcome side: "
            f"{outcome_side}"
        )

    total_count = 0.0
    total_cost = 0.0
    total_fee = 0.0

    for fill in fills:

        count = float(
            fill[
                "count_fp"
            ]
        )

        if outcome_side == "YES":

            price = float(
                fill[
                    "yes_price_dollars"
                ]
            )

        else:

            price = float(
                fill[
                    "no_price_dollars"
                ]
            )

        fee = float(
            fill.get(
                "fee_cost",
                0.0
            )
        )

        total_count += count

        total_cost += (
            count
            *
            price
        )

        total_fee += fee

    if total_count <= 0:

        return {
            "fill_count":
                0.0,

            "average_fill_price":
                None,

            "total_fee":
                total_fee,

            "average_fee_paid":
                None
        }

    average_fill_price = (
        total_cost
        /
        total_count
    )

    average_fee_paid = (
        total_fee
        /
        total_count
    )

    return {
        "fill_count":
            total_count,

        "average_fill_price":
            average_fill_price,

        "total_fee":
            total_fee,

        "average_fee_paid":
            average_fee_paid
    }

def get_market_settlement(
    ticker,
    timeout=15
):
    """
    Retrieve the current settlement state
    for one Kalshi market.
    """

    if not ticker:

        raise ValueError(
            "ticker is required."
        )

    base_url = os.getenv(
        "KALSHI_BASE_URL",
        DEFAULT_BASE_URL
    )

    url = (
        base_url
        +
        f"/markets/{ticker}"
    )

    response = requests.get(
        url,
        timeout=timeout
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Kalshi market lookup failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()

    market = data.get(
        "market"
    )

    if not market:

        raise RuntimeError(
            f"Kalshi returned no market "
            f"for {ticker}."
        )

    result = (
        market.get(
            "result"
        )
    )

    if result is not None:

        result = (
            str(result)
            .strip()
            .upper()
        )

    settlement_value = (
        market.get(
            "settlement_value_dollars"
        )
    )

    if settlement_value not in (
        None,
        ""
    ):

        settlement_value = float(
            settlement_value
        )

    return {
        "ticker":
            ticker,

        "status":
            market.get(
                "status"
            ),

        "result":
            result,

        "settlement_value":
            settlement_value,

        "settlement_time":
            market.get(
                "settlement_ts"
            )
    }