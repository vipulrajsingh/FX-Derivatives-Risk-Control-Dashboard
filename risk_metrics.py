"""
risk_metrics.py

Historical simulation VaR and CVaR, rolling drawdown, and margin-at-
risk for the portfolio.

Methodology - historical simulation, full revaluation:
For each day in a historical lookback window, take that day's actual
percentage return for each pair and apply it as a shock to TODAY's
spot rate (not that day's spot rate - basically, we're asking "what would today's
portfolio have done if it experienced that historical day's move?").
Futures P&L is computed directly from the shocked spot. Options are
fully re-priced through Garman-Kohlhagen at the shocked spot - 
this is more accurate than a delta- normal approximation, 
especially for a book with real convexity (Gamma) in it

This produces one simulated portfolio P&L figure per historical day.
VaR, CVaR, and drawdown are all read off that same distribution

Important Assumption: volatility (sigma) is held at
today's estimated level while spot is shocked. A fuller model would
also let implied vol move with each historical scenario. 
This is a known limitation, and something that could 
be improved in a future iteration
"""

import numpy as np
import pandas as pd

from data_layer import FX_PAIRS, get_historical_data
from fx_conversion import domestic_amount_to_usd
from portfolio import Position, build_portfolio
from pricing import RATES, _domestic_foreign, garman_kohlhagen, historical_volatility


def historical_pnl_series(portfolio: list[Position], live_rates: dict, period: str = "1y") -> pd.Series:
    """
    Builds the core simulation: one portfolio P&L figure (in USD) per
    historical trading day, from walking today's book through each
    day's actual historical return.

    period="1y" is a deliberate choice:
    too short a window under-samples tail events (a 99% VaR needs
    enough observations in the bottom 1% to mean anything), too long a
    window risks blending in a stale market regime. One year is a
    reasonable middle ground for this project; scaling to 2y+ would be
    a legitimate next iteration if more tail robustness is wanted.
    """
    history = get_historical_data(period=period)
    returns = {pair: df["Close"].pct_change().dropna() for pair, df in history.items()}

    # Some pairs may have slightly different trading calendars/holidays -
    # only keep dates common to all five, so every day in the simulation
    # has a return for every position.
    common_dates = sorted(set.intersection(*(set(r.index) for r in returns.values())))

    vols = {pair: historical_volatility(pair, period=period) for pair in FX_PAIRS}

    # Precompute each position's static inputs and (for options) its
    # baseline price today - once, not once per historical day.
    position_info = []
    for pos in portfolio:
        info = {"position": pos}
        if pos.instrument == "Option":
            domestic, foreign = _domestic_foreign(pos.pair)
            info["T"] = pos.days_to_expiry() / 365
            info["r_d"] = RATES[domestic]
            info["r_f"] = RATES[foreign]
            info["sigma"] = vols[pos.pair]
            info["base_price"] = garman_kohlhagen(
                spot=live_rates[pos.pair], strike=pos.strike, T=info["T"],
                r_d=info["r_d"], r_f=info["r_f"], sigma=info["sigma"],
                option_type=pos.option_type,
            )["price"]
        position_info.append(info)

    daily_pnls = []
    for d in common_dates:
        total_usd = 0.0
        for info in position_info:
            pos = info["position"]
            current_spot = live_rates[pos.pair]
            shocked_spot = current_spot * (1 + returns[pos.pair].loc[d])
            sign = 1 if pos.direction == "Long" else -1

            if pos.instrument == "Futures":
                pnl_domestic = pos.notional * sign * (shocked_spot - current_spot)
            else:
                shocked_price = garman_kohlhagen(
                    spot=shocked_spot, strike=pos.strike, T=info["T"],
                    r_d=info["r_d"], r_f=info["r_f"], sigma=info["sigma"],
                    option_type=pos.option_type,
                )["price"]
                pnl_domestic = (shocked_price - info["base_price"]) * pos.notional * sign

            total_usd += domestic_amount_to_usd(pnl_domestic, pos.pair, live_rates)

        daily_pnls.append(total_usd)

    return pd.Series(daily_pnls, index=common_dates, name="Portfolio P&L (USD)")


def historical_var(pnl_series: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss level such that `confidence`% of
    historical scenarios were better than it. Returned as a positive
    number (the SIZE of the potential loss), which is the market
    convention even though it's actually the left tail of the P&L
    distribution. Another way to think of it would be: What's the min
    annual loss the portfolio would have experienced in 5% of the 
    sample dataset days
    """
    percentile = (1 - confidence) * 100
    return -np.percentile(pnl_series, percentile)


def historical_cvar(pnl_series: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR / Expected Shortfall: the AVERAGE loss across the
    scenarios worse than the VaR cutoff - not just the cutoff itself.

    This is the better tail-risk number
    VaR only tells you the threshold a bad day crosses, not how
    bad it gets once it does. Two portfolios can have identical VaR but
    very different CVaR if one has a much fatter tail beyond that
    cutoff - CVaR captures that difference.
    """
    percentile = (1 - confidence) * 100
    threshold = np.percentile(pnl_series, percentile)
    tail = pnl_series[pnl_series <= threshold]
    return -tail.mean()


def rolling_drawdown(pnl_series: pd.Series) -> pd.Series:
    """
    Runs the historical P&L series as a cumulative equity curve and
    computes drawdown (decline from the running peak) at every point.

    Important framing: this ISN'T a backtest of realized historical
    returns - this specific portfolio didn't exist a year ago. It's a
    scenario replay: "if I'd held exactly today's book through the
    last year's actual market moves, what would the drawdown path have
    looked like?"
    """
    cumulative = pnl_series.cumsum()
    running_peak = cumulative.cummax()
    return cumulative - running_peak


def max_drawdown(pnl_series: pd.Series) -> float:
    """
    The worst peak-to-trough decline observed in the replay - 
    a single number summary of rolling_drawdown()
    """
    return rolling_drawdown(pnl_series).min()


def margin_at_risk(pnl_series: pd.Series, confidence: float = 0.99, margin_period_days: int = 2) -> float:
    """
    Approximates an initial margin requirement: scale a 
    single-day VaR (at a high confidence level, 99% here) 
    up to a multi-day "margin period of risk" using
    the square-root-of-time rule.

    This assumes daily P&L moves are independent and identically
    distributed, which is a simplified assumption 
    """
    var_99 = historical_var(pnl_series, confidence=confidence)
    return var_99 * np.sqrt(margin_period_days)


if __name__ == "__main__":
    from data_layer import get_live_rates

    print("Fetching live rates...")
    rates_df = get_live_rates()
    live_rates = dict(zip(rates_df["Pair"], rates_df["Spot Rate"]))

    print("Building portfolio and running historical simulation (this takes a bit longer)...\n")
    book = build_portfolio()
    pnl = historical_pnl_series(book, live_rates)

    print(f"Simulated {len(pnl)} historical trading days.\n")
    print(f"95% VaR:  ${historical_var(pnl, 0.95):>14,.2f}")
    print(f"99% VaR:  ${historical_var(pnl, 0.99):>14,.2f}")
    print(f"95% CVaR: ${historical_cvar(pnl, 0.95):>14,.2f}")
    print(f"99% CVaR: ${historical_cvar(pnl, 0.99):>14,.2f}")
    print(f"Max Drawdown (1y replay): ${max_drawdown(pnl):>14,.2f}")
    print(f"Margin-at-Risk (99%, 2-day MPOR): ${margin_at_risk(pnl):>14,.2f}")
