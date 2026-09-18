"""
pricing.py

Implements the Garman-Kohlhagen model - the FX-specific variant of
Black-Scholes-Merton - and uses it to compute Delta, Gamma, Vega, and
Theta for every options position in the portfolio, plus a portfolio-
level aggregation.

Why Garman-Kohlhagen and not plain BSM: standard Black-Scholes prices
an option on a stock, which pays no interest of its own (or a dividend
yield, in the extended version). An FX option is different - it's a
claim on holding ANOTHER CURRENCY, and that currency earns its own
risk-free rate the whole time you hold it. Garman-Kohlhagen handles
this by treating the foreign interest rate exactly the way the
dividend-adjusted BSM model treats a dividend yield: as a continuous
carry cost that reduces the option's value relative to a no-yield
asset. Two interest rates in, not one - domestic and foreign.
"""

import numpy as np
from scipy.stats import norm

from data_layer import get_historical_data
from fx_conversion import domestic_amount_to_usd, foreign_amount_to_usd
from portfolio import Position, build_portfolio

# Domestic/foreign risk-free rates by currency. These are real policy
# rates as of August 2026 (Fed funds midpoint, ECB deposit rate, BoE
# Bank Rate, BoJ policy rate, SORA, RBI repo rate) - manually sourced
# as a snapshot, not pulled live. A production system would pull these, 
# from a rates API (e.g. FRED) rather than hardcoding a point-in-time snapshot
# and ideally use each pair's own currency-specific curve point matching
# each option's tenor, not one flat rate per currency.
RATES = {
    "USD": 0.0363,  # Fed funds effective range midpoint (3.50-3.75%)
    "EUR": 0.0225,  # ECB deposit facility rate
    "GBP": 0.0375,  # BoE Bank Rate
    "JPY": 0.0100,  # BoJ policy rate
    "SGD": 0.0144,  # SORA (Singapore Overnight Rate Average)
    "INR": 0.0525,  # RBI repo rate
}


def _domestic_foreign(pair: str) -> tuple[str, str]:
    """
    Splits "EUR/USD" into (domestic, foreign) currencies.

    Convention: for a pair BASE/QUOTE, the spot rate is quoted as
    "QUOTE per 1 BASE" (e.g. EUR/USD = 1.166 means 1 EUR = 1.166 USD).
    Garman-Kohlhagen calls the currency the price is quoted IN the
    domestic currency, and the currency being priced the foreign
    currency. So QUOTE = domestic, BASE = foreign.
    """
    base, quote = pair.split("/")
    return quote, base


def historical_volatility(pair: str, period: str = "6mo") -> float:
    """
    Annualized volatility of daily log returns, estimated from the same
    historical data data_layer.py already pulls. This feeds sigma into
    Garman-Kohlhagen - it's estimated from public data, not assumed.
    """
    history = get_historical_data(period=period)
    closes = history[pair]["Close"]
    log_returns = np.log(closes / closes.shift(1)).dropna()
    daily_vol = log_returns.std()
    return daily_vol * np.sqrt(252)  # 252 trading days/year


def garman_kohlhagen(spot, strike, T, r_d, r_f, sigma, option_type):
    """
    Core pricing formula. Returns price and Greeks for one option.

    spot, strike   - exchange rates (domestic per unit of foreign)
    T              - time to expiry, in years
    r_d, r_f       - domestic and foreign risk-free rates
    sigma          - annualized volatility
    option_type    - "Call" or "Put"
    """
    d1 = (np.log(spot / strike) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    df_for = np.exp(-r_f * T)  # foreign discount factor - this is the "dividend yield" analog
    df_dom = np.exp(-r_d * T)   # domestic discount factor

    if option_type == "Call":
        price = spot * df_for * norm.cdf(d1) - strike * df_dom * norm.cdf(d2)
        delta = df_for * norm.cdf(d1)
        theta = (
            -(spot * sigma * df_for * norm.pdf(d1)) / (2 * np.sqrt(T))
            + r_f * spot * df_for * norm.cdf(d1)
            - r_d * strike * df_dom * norm.cdf(d2)
        )
    else:  # Put
        price = strike * df_dom * norm.cdf(-d2) - spot * df_for * norm.cdf(-d1)
        delta = -df_for * norm.cdf(-d1)
        theta = (
            -(spot * sigma * df_for * norm.pdf(d1)) / (2 * np.sqrt(T))
            - r_f * spot * df_for * norm.cdf(-d1)
            + r_d * strike * df_dom * norm.cdf(-d2)
        )

    # Gamma and Vega have the same formula for calls and puts.
    gamma = (df_for * norm.pdf(d1)) / (spot * sigma * np.sqrt(T))
    vega = spot * df_for * norm.pdf(d1) * np.sqrt(T)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,     # convention: value per 1 percentage
                                # point move in volatility, not per
                                # 1.00 (100%) move
        "theta": theta / 365,   # convention: value per calendar day,
                                # not per year
    }


def price_position(position: Position, spot: float) -> dict:
    """
    Prices one options Position and returns Greeks scaled by notional
    and direction. Returns None for futures - separately in
    build_greeks_table(), since a future isn't priced with an options
    model at all.
    """
    if position.instrument != "Option":
        return None

    domestic, foreign = _domestic_foreign(position.pair)
    T = position.days_to_expiry() / 365
    sigma = historical_volatility(position.pair)

    result = garman_kohlhagen(
        spot=spot,
        strike=position.strike,
        T=T,
        r_d=RATES[domestic],
        r_f=RATES[foreign],
        sigma=sigma,
        option_type=position.option_type,
    )

    # Being short flips the sign of every Greek: short a call behaves,
    # directionally, like being long a put.
    sign = 1 if position.direction == "Long" else -1
    scaled = {k: v * position.notional * sign for k, v in result.items()}
    scaled["sigma"] = sigma
    return scaled


def build_greeks_table(portfolio: list[Position], live_rates: dict) -> "pd.DataFrame":
    """
    Runs every position in the portfolio through the pricer and returns
    one row per position with its Greeks.

    Futures get a simplified treatment: a future has no optionality, so
    gamma/vega/theta are zero, and its delta is just its notional
    (scaled by direction) - a 1-for-1 exposure to the underlying
    """
    import pandas as pd

    rows = []
    for pos in portfolio:
        spot = live_rates[pos.pair]

        if pos.instrument == "Option":
            greeks = price_position(pos, spot)
        else:  # Futures
            sign = 1 if pos.direction == "Long" else -1
            greeks = {
                "price": None, "delta": pos.notional * sign,
                "gamma": 0, "vega": 0, "theta": 0, "sigma": None,
            }

        # Raw Gamma is in units of 1/spot, so a pair quoted at a tiny
        # spot level (INR/USD at ~0.0105) produces a huge raw number
        # purely from the quoting convention
        # Cash Gamma fixes this: it's the P&L impact of a 1% move in
        # the pair, which is comparable in scale across pairs.
        cash_gamma = 0.5 * greeks["gamma"] * (0.01 * spot) ** 2

        # DEVLOG Problem 2 fix: Cash Gamma, Vega, and Theta are P&L
        # figures in each pair's own domestic currency (USD for four
        # of the five pairs, JPY for USD/JPY). Delta is a different
        # kind of quantity - notional exposure in the FOREIGN currency.
        # Both get converted to USD here so portfolio totals are
        # actually valid to sum.
        delta_usd = foreign_amount_to_usd(greeks["delta"], pos.pair, live_rates)
        cash_gamma_usd = domestic_amount_to_usd(cash_gamma, pos.pair, live_rates)
        vega_usd = domestic_amount_to_usd(greeks["vega"], pos.pair, live_rates)
        theta_usd = domestic_amount_to_usd(greeks["theta"], pos.pair, live_rates)

        rows.append({
            "Pair": pos.pair,
            "Instrument": pos.instrument,
            "Direction": pos.direction,
            "Delta": greeks["delta"],
            "Delta (USD)": delta_usd,
            "Gamma": greeks["gamma"],
            "Cash Gamma (1% move)": cash_gamma,
            "Cash Gamma (USD)": cash_gamma_usd,
            "Vega": greeks["vega"],
            "Vega (USD)": vega_usd,
            "Theta": greeks["theta"],
            "Theta (USD)": theta_usd,
        })

    return pd.DataFrame(rows)


def aggregate_greeks(greeks_table: "pd.DataFrame") -> dict:
    """
    Sums each Greek across the whole book, in USD - the portfolio-level
    exposure a risk desk actually watches. Summing the raw (non-USD)
    columns would add mixed currencies together and produce a
    meaningless number (see DEVLOG.md, Problem 2) - the USD columns
    are what make this aggregation valid.
    """
    return {
        "Total Delta (USD)": greeks_table["Delta (USD)"].sum(),
        "Total Cash Gamma (USD, 1% move)": greeks_table["Cash Gamma (USD)"].sum(),
        "Total Vega (USD)": greeks_table["Vega (USD)"].sum(),
        "Total Theta (USD)": greeks_table["Theta (USD)"].sum(),
    }


if __name__ == "__main__":
    from data_layer import get_live_rates

    print("Fetching live rates...")
    rates_df = get_live_rates()
    live_rates = dict(zip(rates_df["Pair"], rates_df["Spot Rate"]))

    print("\nBuilding portfolio...")
    book = build_portfolio()

    print("\nPricing every position...\n")
    table = build_greeks_table(book, live_rates)
    print(table.to_string(index=False))

    print("\nPortfolio-level aggregated Greeks:\n")
    for k, v in aggregate_greeks(table).items():
        print(f"{k}: {v:,.2f}")
