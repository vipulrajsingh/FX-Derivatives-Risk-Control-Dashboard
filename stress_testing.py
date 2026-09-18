"""
stress_testing.py

Three historical stress scenarios, replayed against TODAY's portfolio:
the March 2020 COVID vol spike, the 2022 USD strength cycle, and the
2015 SNB floor removal. This pulls fixed historical date 
ranges anchored to specific real events.

Methodology: for each scenario, two things get shocked simultaneously -
  1. SPOT - the cumulative % move each pair actually made during that
     historical window, applied to today's spot rate.
  2. VOLATILITY - the REALIZED volatility that actually prevailed
     during that historical window (not today's estimate),
     used to reprice every option.

Shocking spot alone would understate what a real stress event does -
the whole point of a "vol spike" scenario is that volatility itself
moves, not just price. An option repriced with today's benign vol
under a stressed spot would materially understate the real historical
P&L impact, especially for the portfolio with Gamma in it.

Note on the SNB scenario specifically: this project holds no CHF
exposure, and the SNB shock's epicenter was EUR/CHF, not any pair in
this book. It's included anyway using a legitimate and standard stress-
testing method: replay what the SAME broader market dislocation did
to the pairs actually held, not just the instrument at the shock's
center
"""

import numpy as np
import pandas as pd
import yfinance as yf

from data_layer import FX_PAIRS
from fx_conversion import domestic_amount_to_usd
from portfolio import Position, build_portfolio
from pricing import RATES, _domestic_foreign, garman_kohlhagen, historical_volatility

STRESS_SCENARIOS = {
    "2020 COVID Vol Spike": {
        "start": "2020-02-19",
        "end": "2020-03-27",
        "description": (
            "Global risk-off shock as COVID-19 escalated into a pandemic. "
            "Equities fell roughly a third in five weeks; the US Dollar "
            "surged as a flight-to-safety asset despite the Fed cutting "
            "rates to zero - a global dollar funding squeeze temporarily "
            "overrode the usual 'lower rates -> weaker dollar' relationship."
        ),
    },
    "2022 USD Strength Cycle": {
        "start": "2022-01-03",
        "end": "2022-09-30",
        "description": (
            "The Fed's most aggressive hiking cycle in decades drove "
            "sustained broad-based USD strength. EUR/USD fell to parity "
            "for the first time in 20 years; GBP/USD hit an all-time low "
            "following the UK's September 2022 mini-budget crisis."
        ),
    },
    "2015 SNB Floor Removal": {
        "start": "2015-01-14",
        "end": "2015-01-20",
        "description": (
            "The Swiss National Bank abruptly abandoned its EUR/CHF 1.20 "
            "floor with no warning, sending CHF up over 30% intraday. "
            "This portfolio holds no CHF - included to test how a broader "
            "market-wide dislocation, not just the instrument at its "
            "epicenter, would have hit the pairs actually held."
        ),
    },
}


def fetch_stress_window(start: str, end: str) -> dict:
    """
    Pulls actual historical daily closes for all five pairs over a
    fixed historical date window
    """
    data = {}
    for pair, ticker in FX_PAIRS.items():
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if not hist.empty:
            data[pair] = hist["Close"]
    return data


def scenario_shock(start: str, end: str) -> dict:
    """
    For a historical window, computes per pair:
      - spot_shock: cumulative start-to-end % move
      - stress_vol: realized volatility DURING that window, annualized
    """
    window_data = fetch_stress_window(start, end)
    shocks = {}
    for pair, closes in window_data.items():
        if len(closes) < 2:
            continue
        spot_shock = closes.iloc[-1] / closes.iloc[0] - 1
        daily_returns = closes.pct_change().dropna()
        stress_vol = daily_returns.std() * np.sqrt(252)
        shocks[pair] = {"spot_shock": spot_shock, "stress_vol": stress_vol}
    return shocks


def apply_scenario(portfolio: list[Position], live_rates: dict, shocks: dict, today_vols: dict) -> tuple[float, pd.DataFrame]:
    """
    Revalues the current portfolio under one historical stress
    scenario. Returns (total P&L in USD, per-position breakdown).

    Futures: linear P&L from the shocked spot alone.
    Options: fully repriced twice through Garman-Kohlhagen - once at
    today's spot/vol (the baseline), once at the shocked spot AND the
    scenario's stress-period volatility (the stressed case) - so the
    P&L captures both the price move and the vol regime shift at once.
    """
    total_usd = 0.0
    rows = []

    for pos in portfolio:
        if pos.pair not in shocks:
            continue  # this pair had no data in this historical window

        current_spot = live_rates[pos.pair]
        shock = shocks[pos.pair]
        shocked_spot = current_spot * (1 + shock["spot_shock"])
        sign = 1 if pos.direction == "Long" else -1

        if pos.instrument == "Futures":
            pnl_domestic = pos.notional * sign * (shocked_spot - current_spot)
        else:
            domestic, foreign = _domestic_foreign(pos.pair)
            T = pos.days_to_expiry() / 365
            base_price = garman_kohlhagen(
                spot=current_spot, strike=pos.strike, T=T,
                r_d=RATES[domestic], r_f=RATES[foreign],
                sigma=today_vols[pos.pair], option_type=pos.option_type,
            )["price"]
            stressed_price = garman_kohlhagen(
                spot=shocked_spot, strike=pos.strike, T=T,
                r_d=RATES[domestic], r_f=RATES[foreign],
                sigma=shock["stress_vol"], option_type=pos.option_type,
            )["price"]
            pnl_domestic = (stressed_price - base_price) * pos.notional * sign

        pnl_usd = domestic_amount_to_usd(pnl_domestic, pos.pair, live_rates)
        total_usd += pnl_usd
        rows.append({
            "Pair": pos.pair,
            "Instrument": pos.instrument,
            "Direction": pos.direction,
            "Spot Shock": f"{shock['spot_shock']:+.2%}",
            "P&L (USD)": pnl_usd,
        })

    return total_usd, pd.DataFrame(rows)


def scenario_path(portfolio: list[Position], live_rates: dict, today_vols: dict, start: str, end: str) -> pd.Series:
    """
    Computes the CURRENT portfolio's P&L on every trading day within a
    historical stress window - not just the start-to-end total that
    apply_scenario() returns, but the actual day-by-day trajectory the
    book would have taken through the crisis

    Methodology: for each day in the window, the cumulative return
    from the window's FIRST day to THAT day is applied as the spot
    shock - so the path starts near zero and its final value matches
    apply_scenario()'s total P&L exactly (same underlying shock, just
    read at every intermediate day instead of only the endpoint)

    Two Assumptions here to note:
      - Volatility is held at the window's OVERALL realized level
        throughout the path, not recomputed day by day. Some of these
        windows are short (the SNB scenario is ~6 trading days) - too
        few points for a meaningfully separate rolling volatility
        estimate at each day
      - Time-to-expiry (T) is held fixed at today's actual distance to
        each option's expiry throughout the path, not decremented day
        by day. This matches the existing apply_scenario() framing:
        "what would happen to TODAY's portfolio if it experienced this
        historical trajectory starting now" - not a literal calendar
        walk-forward with decaying time value layered on top
    """
    window_data = fetch_stress_window(start, end)
    common_dates = sorted(set.intersection(*(set(c.index) for c in window_data.values())))
    if not common_dates:
        return pd.Series(dtype=float)

    stress_vols = {}
    for pair, closes in window_data.items():
        daily_returns = closes.pct_change().dropna()
        stress_vols[pair] = daily_returns.std() * np.sqrt(252)

    position_info = []
    for pos in portfolio:
        info = {"position": pos}
        if pos.instrument == "Option" and pos.pair in stress_vols:
            domestic, foreign = _domestic_foreign(pos.pair)
            info["T"] = pos.days_to_expiry() / 365
            info["r_d"] = RATES[domestic]
            info["r_f"] = RATES[foreign]
            info["stress_vol"] = stress_vols[pos.pair]
            info["base_price"] = garman_kohlhagen(
                spot=live_rates[pos.pair], strike=pos.strike, T=info["T"],
                r_d=info["r_d"], r_f=info["r_f"], sigma=today_vols[pos.pair],
                option_type=pos.option_type,
            )["price"]
        position_info.append(info)

    window_start_price = {pair: closes.loc[common_dates[0]] for pair, closes in window_data.items()}

    daily_pnls = []
    for d in common_dates:
        total_usd = 0.0
        for info in position_info:
            pos = info["position"]
            if pos.pair not in window_data:
                continue
            current_spot = live_rates[pos.pair]
            window_return = window_data[pos.pair].loc[d] / window_start_price[pos.pair] - 1
            shocked_spot = current_spot * (1 + window_return)
            sign = 1 if pos.direction == "Long" else -1

            if pos.instrument == "Futures":
                pnl_domestic = pos.notional * sign * (shocked_spot - current_spot)
            else:
                shocked_price = garman_kohlhagen(
                    spot=shocked_spot, strike=pos.strike, T=info["T"],
                    r_d=info["r_d"], r_f=info["r_f"], sigma=info["stress_vol"],
                    option_type=pos.option_type,
                )["price"]
                pnl_domestic = (shocked_price - info["base_price"]) * pos.notional * sign

            total_usd += domestic_amount_to_usd(pnl_domestic, pos.pair, live_rates)

        daily_pnls.append(total_usd)

    return pd.Series(daily_pnls, index=common_dates, name="Portfolio P&L (USD)")


if __name__ == "__main__":
    from data_layer import get_live_rates

    print("Fetching live rates...")
    rates_df = get_live_rates()
    live_rates = dict(zip(rates_df["Pair"], rates_df["Spot Rate"]))

    print("Building portfolio and today's volatility baseline...\n")
    book = build_portfolio()
    today_vols = {pair: historical_volatility(pair) for pair in FX_PAIRS}

    for name, info in STRESS_SCENARIOS.items():
        print("=" * 70)
        print(name)
        print(info["description"])
        print()

        shocks = scenario_shock(info["start"], info["end"])
        total_usd, breakdown = apply_scenario(book, live_rates, shocks, today_vols)

        print(breakdown.to_string(index=False))
        print(f"\nTotal scenario P&L: ${total_usd:,.2f}\n")
