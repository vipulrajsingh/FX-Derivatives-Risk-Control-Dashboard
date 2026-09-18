"""
alerts.py

Takes the Greeks (pricing.py) and risk metrics (risk_metrics.py) 
built in this project and classifies each one into a 
risk state - OK / WARNING / CRITICAL - against explicit, stated limits,
then rolls those up into one overall portfolio verdict, 
including a fourth state: ESCALATE.

Limits are set as a percentage of an assumed RISK_CAPITAL base. For eg:
a limit like "$300K VaR" means nothing without knowing how much
capital that risk is being taken against. RISK_CAPITAL here is a
stated assumption, not derived from the portfolio's own notional, 
where capital allocated is typically much smaller than notional traded. 
A $10M capital base supporting this book's $50M+ of futures notional is a realistic.
"""

import pandas as pd

RISK_CAPITAL = 10_000_000  # default assumption - overridable in the dashboard


""" 
metric name -> (warning %, critical %, one_sided)
Percentages, not dollar amounts, are the actual source of truth
one_sided = True means only the stated direction is a risk 
(currently only Theta - large POSITIVE theta means the book 
is earning from time decay, which is not something to alert on).
"""

RISK_LIMIT_PERCENTAGES = {
    "VaR 95%":          (0.0150, 0.0300, False),  # 1.5% / 3.0% of capital
    "VaR 99%":          (0.0250, 0.0500, False),  # 2.5% / 5.0%
    "CVaR 99%":         (0.0300, 0.0600, False),  # 3.0% / 6.0%
    "Max Drawdown":     (0.0500, 0.1000, False),  # 5.0% / 10.0%
    "Margin-at-Risk":   (0.0300, 0.0600, False),  # 3.0% / 6.0%
    "Net Delta (USD)":  (0.4000, 0.7000, False),  # 40% / 70% - net directional FX exposure
    "Cash Gamma (USD)": (0.0015, 0.0035, False),  # 0.15% / 0.35% - P&L swing per 1% spot move
    "Vega (USD)":       (0.0020, 0.0040, False),  # 0.20% / 0.40% - P&L swing per 1pt vol move
    "Theta (USD/day)":  (-0.0010, -0.0025, True),  # one-sided: only bleed matters
}


def build_risk_limits(risk_capital: float) -> dict:
    """
    Builds the actual dollar RISK_LIMITS dict for a given risk capital
    base. Kept as a function, not a fixed constant, specifically so
    the dashboard can let risk capital be an adjustable input
    """
    return {
        name: (warn_pct * risk_capital, crit_pct * risk_capital, one_sided)
        for name, (warn_pct, crit_pct, one_sided) in RISK_LIMIT_PERCENTAGES.items()
    }


# Default limits at the default capital - used by classify_metric()'s
# standalone script usage below and anywhere a specific capital isn't
# supplied.
RISK_LIMITS = build_risk_limits(RISK_CAPITAL)


def classify_metric(value: float, warning: float, critical: float, one_sided: bool = False) -> str:
    """
    Classifies one risk metric against its warning/critical limits.

    Two-sided metrics (most of them) compare against the ABSOLUTE
    value - a large negative Net Delta is exactly as much a
    directional risk as a large positive one; a portfolio can be
    dangerously short just as easily as dangerously long. One-sided
    metrics (Theta) only flag on the stated "bad" direction.
    """
    check_value = value if one_sided else abs(value)
    if one_sided:
        if check_value <= critical:
            return "CRITICAL"
        elif check_value <= warning:
            return "WARNING"
        return "OK"
    else:
        if check_value >= critical:
            return "CRITICAL"
        elif check_value >= warning:
            return "WARNING"
        return "OK"


def evaluate_risk_state(metrics: dict, risk_limits: dict = None) -> pd.DataFrame:
    """
    Classifies a dict of {metric_name: value} against risk_limits (a
    dict shaped like RISK_LIMITS/build_risk_limits() output) and
    returns a one-row-per-metric summary table. Defaults to the
    standard RISK_LIMITS if none is supplied, so existing standalone
    usage of this function is unaffected. Unrecognized metric names
    are silently skipped rather than raising.
    """
    risk_limits = risk_limits if risk_limits is not None else RISK_LIMITS
    rows = []
    for name, value in metrics.items():
        if name not in risk_limits:
            continue
        warning, critical, one_sided = risk_limits[name]
        status = classify_metric(value, warning, critical, one_sided)
        rows.append({
            "Metric": name,
            "Value": value,
            "Warning Limit": warning,
            "Critical Limit": critical,
            "Status": status,
        })
    return pd.DataFrame(rows)


def overall_status(status_table: pd.DataFrame) -> str:
    """
    Rolls up individual metric statuses into one portfolio-level
    verdict, with four tiers, not three:

      CRITICAL - any single metric has breached its critical limit.
      ESCALATE - no single metric is critical, but 3 or more are
        simultaneously in WARNING. The reasoning: several risk
        dimensions elevated at once is a materially different (and
        often worse) situation than one metric alone being stretched,
        even with no individual critical breach - think about correlated 
        stress across multiple factors at once, not just isolated breaches.
      WARNING - 1-2 metrics in WARNING, none CRITICAL.
      OK - everything inside its warning limit.
    """
    critical_count = (status_table["Status"] == "CRITICAL").sum()
    warning_count = (status_table["Status"] == "WARNING").sum()

    if critical_count > 0:
        return "CRITICAL"
    if warning_count >= 3:
        return "ESCALATE"
    if warning_count > 0:
        return "WARNING"
    return "OK"


if __name__ == "__main__":
    from data_layer import get_live_rates
    from portfolio import build_portfolio
    from pricing import build_greeks_table, aggregate_greeks
    from risk_metrics import (
        historical_pnl_series, historical_var, historical_cvar,
        max_drawdown, margin_at_risk,
)

    print("Fetching live rates and building portfolio...")
    rates_df = get_live_rates()
    live_rates = dict(zip(rates_df["Pair"], rates_df["Spot Rate"]))
    book = build_portfolio()

    print("Computing Greeks...")
    greeks_table = build_greeks_table(book, live_rates)
    greeks_agg = aggregate_greeks(greeks_table)

    print("Running historical simulation for VaR/CVaR/Drawdown/Margin...")
    pnl = historical_pnl_series(book, live_rates)

    metrics = {
        "Net Delta (USD)": greeks_agg["Total Delta (USD)"],
        "Cash Gamma (USD)": greeks_agg["Total Cash Gamma (USD, 1% move)"],
        "Vega (USD)": greeks_agg["Total Vega (USD)"],
        "Theta (USD/day)": greeks_agg["Total Theta (USD)"],
        "VaR 95%": historical_var(pnl, 0.95),
        "VaR 99%": historical_var(pnl, 0.99),
        "CVaR 99%": historical_cvar(pnl, 0.99),
        "Max Drawdown": max_drawdown(pnl),
        "Margin-at-Risk": margin_at_risk(pnl),
    }

    status_table = evaluate_risk_state(metrics)
    pd.set_option("display.width", 150)
    print("\n" + "=" * 72)
    print("RISK CONTROL ROOM - CURRENT STATE")
    print("=" * 72)
    print(status_table.to_string(index=False))

    verdict = overall_status(status_table)
    print(f"\nOVERALL PORTFOLIO STATUS: {verdict}")
