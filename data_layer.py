"""
data_layer.py

Handles all communication with Yahoo Finance via yfinance.
 This file's objective is: given a currency pair, get its price data.
"""

import yfinance as yf
import pandas as pd

# Yahoo Finance identifies FX spot pairs with a "=X" suffix on the ticker.
# EUR/USD becomes "EURUSD=X", GBP/USD becomes "GBPUSD=X", and so on.
# This dict is the single source of truth mapping our readable pair names
# to the tickers yfinance actually understands.
FX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "SGD/USD": "SGDUSD=X",
    "INR/USD": "INRUSD=X",
}


def get_live_rates() -> pd.DataFrame:
    """
    Fetch the latest available spot rate for each FX pair.

    Returns a DataFrame with one row per pair:
        Pair | Ticker | Spot Rate
    """
    rows = []
    for pair_name, ticker in FX_PAIRS.items():
        # .history(period="1d") pulls the most recent trading day's data.
        # yfinance returns a DataFrame with columns like Open/High/Low/Close.
        data = yf.Ticker(ticker).history(period="1d")

        if data.empty:
            # If Yahoo returns nothing (market closed, ticker issue, no
            # internet), we don't want the whole thing to crash - we record
            # it as missing and keep going.
            spot = None
        else:
            # .iloc[-1] grabs the last row's Close value - the most
            # recent price in the returned window.
            spot = round(data["Close"].iloc[-1], 4)

        rows.append({"Pair": pair_name, "Ticker": ticker, "Spot Rate": spot})

    return pd.DataFrame(rows)


def get_historical_data(period: str = "6mo") -> dict[str, pd.DataFrame]:
    """
    Fetch historical daily closing prices for each FX pair.

    Returns a dict keyed by pair name, e.g. {"EUR/USD": <DataFrame>}.
    Each DataFrame is indexed by Date and has a single "Close" column.
    """
    history = {}
    for pair_name, ticker in FX_PAIRS.items():
        data = yf.Ticker(ticker).history(period=period)
        history[pair_name] = data[["Close"]]
    return history


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    print("Fetching live FX spot rates...\n")
    live_rates = get_live_rates()
    print(live_rates)

    print("\nFetching 6 months of historical data for EUR/USD...\n")
    hist = get_historical_data()
    print(hist["EUR/USD"].tail())
