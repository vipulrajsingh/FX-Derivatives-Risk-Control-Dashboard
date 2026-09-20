# FX Derivatives Risk Control Dashboard

An interactive risk management dashboard for a simulated FX derivatives
book - live pricing, Greeks, historical simulation VaR/CVaR, drawdown,
margin-at-risk, historical stress testing, a four-tier alert system, 
and a validated mean-reversion signal, built from scratch in Python
and Streamlit.

This project exists to demonstrate a specific combination: hands-on
engineering ability (full stack, including the pricing model,is 
self-built, not a wrapper around an existing library) paired with
applied portfolio/derivatives risk knowledge from CFA Level III study
and live personal trading experience across crypto derivatives and
DeFi. The goal was never to build something novel - Garman-Kohlhagen
and historical VaR are standard methods - but to build them correctly, 
test them rigorously

## Live Dashboard

```
streamlit run app.py
```

Four tabs: **Portfolio & Greeks** (with interactive position entry),
**Risk Metrics & Alerts**, **Stress Testing**, **Signal Creation**.

## Architecture

```
data_layer.py        Live/historical FX rates via yfinance (5 pairs)
portfolio.py         Position dataclass + validation; simulated 10-position book
pricing.py           Garman-Kohlhagen pricing + Greeks (Delta/Gamma/Vega/Theta)
fx_conversion.py     Converts Greeks into a single reporting currency (USD)
risk_metrics.py      Historical simulation VaR/CVaR, drawdown, margin-at-risk
stress_testing.py    Three historical crisis scenarios replayed on today's book
alerts.py            Four-tier risk state classification (OK/Warning/Critical/Escalate)
mean_reversion.py    Cross-pair spread signal, ADF stationarity test, backtest
app.py               Streamlit dashboard tying everything together
```

Each module is independently runnable as a script (eg: `python pricing.py`, 
etc.) and has no dependency on Streamlit.

## Methodology Summary

**Pricing:** Garman-Kohlhagen (the FX-specific variant of
Black-Scholes) - foreign interest rate treated as a continuous carry
cost, the same role a dividend yield plays in equity option pricing.
Validated against textbook Black-Scholes values by setting the foreign
rate to zero.

**Risk metrics:** Historical simulation with full revaluation, not a
delta-normal approximation - every option is fully re-priced under
each historical day's actual return, not linearly approximated,
because the book carries real Gamma. VaR and CVaR are read directly
off that simulated P&L distribution; drawdown treats it as a
cumulative equity curve.

**Stress testing:** Three historical windows (2020 COVID vol spike,
2022 USD strength cycle, 2015 SNB floor removal), each replayed by
shocking *both* spot (the historical cumulative move) and volatility
(the historical realized vol during that window) - not spot alone,
since a "vol spike" scenario that doesn't actually shock volatility
would understate its own premise.

**Alerts:** Four tiers - an `ESCALATE` state triggers when three or 
more risk metrics are simultaneously elevated even without any single 
metric breaching a hard limit, mirroring how correlated stress across 
multiple risk factors is treated as more serious than one isolated breach.

**Mean reversion:** A generalized cross-pair spread (correctly handles
FX quoting-convention chaining - dividing vs. multiplying logs
depending on which currency two pairs share), filtered through an
Augmented Dickey-Fuller stationarity test before any signal is trusted,
then backtested against a frozen entry-time baseline (a naive version
of this backtest produced a misleading 100% hit rate across every
pair - see `DEVLOG.md`, for how that was caught and fixed).

## Key Findings & Limitations

- **The mean-reversion signal does not hold up across lookback
  windows.** Pairs that appeared stationary over 1 year of history
  failed the same test over 5 years, and vice versa - strong evidence
  that FX cross-rate relationships, unlike classic equity pairs
  trades, lack a persistent structural anchor (they're driven by
  diverging central bank cycles, not a stable economic link).
  Documented as a finding
- **`RISK_CAPITAL = $10M`** (in `alerts.py`) is a stated assumption,
  not derived from anything in the data. It matters quite a bit -
  doubling it moves the same real portfolio from an `ESCALATE` alert
  state to fully clean `OK` with zero change in actual risk. Any real
  deployment would need this sourced from an actual capital mandate,
  not an assumed round number.
- **Live data is Yahoo Finance via `yfinance`** - free and
  reproducible, but not institutional-grade
- **Volatility in the risk simulation is held at today's estimated
  level** while spot is shocked across historical scenarios (Garman-
  Kohlhagen inputs); a fuller model would let implied vol move
  scenario-by-scenario for VaR too, the way stress testing already
  does.

Every problem actually hit while building this - including several bugs, 
like a module accidentally shadowing Python's own `signal` standard library, 
and the backtest's frozen-baseline fix - is logged in `DEVLOG.md` with 
root cause and fix

## CFA Curriculum Connections

| Project component | CFA Level |
|---|---|
| Garman-Kohlhagen option pricing | Level II - Derivatives |
| Historical simulation VaR/CVaR | Level III - Risk Management |
| Portfolio Greeks & delta-hedging concepts | Level II/III - Derivatives, Risk Management |
| Cointegration / stationarity testing | Level II - Quantitative Methods (Time-Series Analysis) |
| Historical scenario analysis / stress testing | Level III - Risk Management |
| Currency risk management (Garman-Kohlhagen rate treatment) | Level III - Economics, Currency Management |

## Tech Stack

Python, Streamlit, yfinance, pandas, numpy, scipy, statsmodels

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Further Reading (in this repo)

- `MEAN_REVERSION_METHODOLOGY.md` - the mean-reversion signal explained from first principles, no code
- `DEVLOG.md` - every real problem hit during the build, root cause, and fix

*Built by Vipul Raj Singh as a portfolio project during a CFA Level III
candidacy, combining a background in industrial engineering/automation
with applied derivatives and risk management.*
