"""
portfolio.py

Defines the structure of a single FX position and builds the simulated
portfolio: a mix of futures and options across the five pairs in
data_layer.py, with defined notionals, strikes, and expiries.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from data_layer import FX_PAIRS


@dataclass
class Position:
    """
    One line item in the portfolio - either a futures contract or an
    options contract on a single FX pair.
    """

    pair: str                      # must match a key in FX_PAIRS, e.g. "EUR/USD"
    instrument: str                # "Futures" or "Option"
    direction: str                 # "Long" or "Short"
    notional: float                # position size, in units of the base currency
    expiry: date                   # contract/option expiration date

    # The next three only apply to certain instruments. Giving them
    # defaults of None means a futures position simply leaves strike and
    # option_type unset, rather than us needing two totally separate
    # classes for futures vs. options.
    
    entry_price: Optional[float] = None     # futures: the rate it was dealt at.
    strike: Optional[float] = None          # options only
    option_type: Optional[str] = None       # "Call" or "Put" - options only

    def __post_init__(self):
        # Runs automatically right after a Position is created.
        # Catches mistakes immediately (e.g. a typo'd pair name)
        if self.pair not in FX_PAIRS:
            raise ValueError(f"Unknown pair '{self.pair}'. Must be one of {list(FX_PAIRS.keys())}")
        if self.instrument not in ("Futures", "Option"):
            raise ValueError(f"instrument must be 'Futures' or 'Option'")
        if self.direction not in ("Long", "Short"):
            raise ValueError("direction must be 'Long' or 'Short'")
        if self.instrument == "Option" and self.option_type not in ("Call", "Put"):
            raise ValueError("Options must have option_type 'Call' or 'Put'")

    def days_to_expiry(self, as_of: Optional[date] = None) -> int:
        """
        Days remaining until this position expires.

        The Garman-Kohlhagen formula takes time-to-expiry (in years) as an
        input.
        """
        as_of = as_of or date.today()
        return (self.expiry - as_of).days


def build_portfolio() -> list[Position]:
    """
    Constructs the simulated portfolio: one futures position and one
    options position per currency pair, ten positions total
    """
    portfolio = [
        # --- EUR/USD ---
        Position(
            pair="EUR/USD", instrument="Futures", direction="Long",
            notional=10_000_000, entry_price=1.1650,
            expiry=date(2026, 9, 19),
        ),
        Position(
            pair="EUR/USD", instrument="Option", direction="Long",
            notional=5_000_000, strike=1.1800, option_type="Call",
            expiry=date(2026, 11, 27),
        ),

        # --- GBP/USD ---
        Position(
            pair="GBP/USD", instrument="Futures", direction="Short",
            notional=8_000_000, entry_price=1.3600,
            expiry=date(2026, 9, 19),
        ),
        Position(
            pair="GBP/USD", instrument="Option", direction="Long",
            notional=5_000_000, strike=1.3400, option_type="Put",
            expiry=date(2026, 10, 27),
        ),

        # --- USD/JPY ---
        Position(
            pair="USD/JPY", instrument="Futures", direction="Long",
            notional=12_000_000, entry_price=159.00,
            expiry=date(2026, 9, 19),
        ),
        Position(
            pair="USD/JPY", instrument="Option", direction="Short",
            notional=6_000_000, strike=162.00, option_type="Call",
            expiry=date(2026, 11, 27),
        ),

        # --- SGD/USD ---
        Position(
            pair="SGD/USD", instrument="Futures", direction="Short",
            notional=7_000_000, entry_price=0.7870,
            expiry=date(2026, 9, 19),
        ),
        Position(
            pair="SGD/USD", instrument="Option", direction="Long",
            notional=4_000_000, strike=0.7800, option_type="Put",
            expiry=date(2027, 1, 27),
        ),

        # INR/USD
        # Note: INR is a managed-float currency
        # and isn't as liquid or freely traded as the others
        Position(
            pair="INR/USD", instrument="Futures", direction="Long",
            notional=15_000_000, entry_price=0.0105,
            expiry=date(2026, 9, 19),
        ),
        Position(
            pair="INR/USD", instrument="Option", direction="Long",
            notional=5_000_000, strike=0.0107, option_type="Call",
            expiry=date(2026, 12, 27),
        ),
    ]
    return portfolio


def portfolio_to_dataframe(portfolio: list[Position]) -> pd.DataFrame:
    """
    Converts the list of Position objects into a pandas DataFrame -
    the format Streamlit's st.dataframe() expects for a table
    """
    # Empty string renders as a genuinely blank cell in Streamlit.
    # Streamlit prints "None" for both None and NaN,
    # so "" is the one that actually works here.
    rows = []
    for pos in portfolio:
        rows.append({
            "Pair": pos.pair,
            "Instrument": pos.instrument,
            "Direction": pos.direction,
            "Notional": pos.notional,
            "Entry Price": pos.entry_price if pos.entry_price is not None else "",
            "Strike": pos.strike if pos.strike is not None else "",
            "Option Type": pos.option_type if pos.option_type is not None else "",
            "Expiry": pos.expiry,
            "Days to Expiry": pos.days_to_expiry(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    book = build_portfolio()
    df = portfolio_to_dataframe(book)
    print(f"Built portfolio with {len(book)} positions:\n")
    print(df.to_string(index=False))
