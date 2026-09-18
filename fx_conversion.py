"""
fx_conversion.py

Converts position-level Greeks - which Garman-Kohlhagen naturally
denominates in a mix of different currencies depending on the pair -
into one common reporting currency (USD), so portfolio-level totals
are actually valid to add together.

This exists because of a real issue engineered for the project (DEVLOG.md,
Problem 2): summing Cash Gamma across the book was adding
USD figures to JPY figures as if they were the same unit.
This happened cause the pair entered is denominated as USD/JPY

Two different conversions are needed, not one:
  - Theta, Vega, and Cash Gamma are P&L figures denominated in each
    pair's DOMESTIC (quote) currency.
  - Delta represents notional exposure denominated in each pair's
    FOREIGN (base) currency - a structurally different quantity.
"""


def domestic_currency(pair: str) -> str:
  #The quote currency - what Theta/Vega/Cash Gamma are denominated in."""
    _, quote = pair.split("/") # _ is the base currency, which we don't need here
    return quote


def usd_rate(currency: str, live_rates: dict) -> float:
    """
    Multiplier that converts one unit of currency into USD.

    This project only ever needs two cases - USD and JPY - because of
    how the five pairs happen to be quoted (four of five already have
    USD as one side)
    """
    if currency == "USD":
        return 1.0
    if currency == "JPY":
        # live_rates["USD/JPY"] is quoted as JPY per 1 USD,
        # so 1 JPY = 1 / spot USD.
        return 1.0 / live_rates["USD/JPY"]
    raise ValueError(
        f"No USD conversion rate defined for {currency}. "
        "Add a case here if a pair introducing a new currency is added."
    )


def domestic_amount_to_usd(amount: float, pair: str, live_rates: dict) -> float:
    """
    Converts an amount already denominated in pair's domestic (quote)
    currency - e.g. Theta, Vega, or Cash Gamma - into USD.
    """
    return amount * usd_rate(domestic_currency(pair), live_rates)


def foreign_amount_to_usd(amount: float, pair: str, live_rates: dict) -> float:
    """
    Converts an amount denominated in pair's foreign (base) currency
    - i.e. Delta, which represents notional exposure in the base
    currency - into USD.

    Two steps: base currency -> domestic currency (multiply by spot,
    since spot itself IS the base-to-domestic conversion rate), then
    domestic -> USD (same conversion domestic_amount_to_usd() does).
    """
    spot = live_rates[pair]
    domestic_amount = amount * spot
    return domestic_amount_to_usd(domestic_amount, pair, live_rates)
