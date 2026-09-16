import re

def parse_markets(markets_data):
    parsed_markets = []

    markets = markets_data["markets"]

    for market in markets:
        ticker = market["ticker"]
        title = market["title"]

        strike_type = market["strike_type"]
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")

        yes_bid = float(market["yes_bid_dollars"])
        yes_ask = float(market["yes_ask_dollars"])
        no_bid = float(market["no_bid_dollars"])
        no_ask = float(market["no_ask_dollars"])
        yes_spread = yes_ask - yes_bid
        no_spread = no_ask - no_bid
        yes_midpoint = (yes_bid + yes_ask) / 2
        no_midpoint = (no_bid + no_ask) /2

        parsed_market = {
            "ticker": ticker,
            "title": title,

            "strike_type": strike_type,
            "floor_strike": floor_strike,
            "cap_strike": cap_strike,

            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_spread": yes_spread,
            "yes_midpoint": yes_midpoint,

            "no_bid": no_bid,
            "no_ask": no_ask,
            "no_spread": no_spread,
            "no_midpoint": no_midpoint,


        }

        parsed_markets.append(parsed_market)

    return parsed_markets