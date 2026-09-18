"""
app.py

The Streamlit dashboard - refined per user feedback after the first
full run-through. Notable fixes and additions over the first version:

  - REAL BUG FIXED: the Mean Reversion tab's title claimed "5-year
    history" but the underlying scan_all_pairs()/backtest_all_pairs()
    calls had no period argument, silently defaulting to 1 year. Now
    explicitly passes period="5y" to match what the label claims.
  - Positions can now be EDITED in place (not just added/removed),
    via st.data_editor, reconstructing validated Position objects from
    the edited table using Position's own __post_init__ validation -
    no duplicated validation logic.
  - Risk capital is now a real adjustable input, not a hardcoded
    constant - see alerts.build_risk_limits().
  - Tables display with a 1-based index and center-aligned text via a
    shared display_df() helper.
  - Charts added to the Risk Metrics tab (headline metric comparison,
    cumulative P&L / drawdown path, daily P&L distribution).
  - Signal Creation (renamed from Mean Reversion) moved to the last
    tab and gated behind a Run button, consistent with the other two
    slow-computation tabs.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from data_layer import FX_PAIRS, get_live_rates
from portfolio import Position, build_portfolio, portfolio_to_dataframe
from pricing import aggregate_greeks, build_greeks_table, historical_volatility
from risk_metrics import (
    historical_cvar, historical_pnl_series, historical_var,
    margin_at_risk, max_drawdown, rolling_drawdown,
)
from mean_reversion import backtest_all_pairs, scan_all_pairs
from stress_testing import STRESS_SCENARIOS, apply_scenario, scenario_path, scenario_shock
from alerts import build_risk_limits, evaluate_risk_state, overall_status

st.set_page_config(page_title="FX Derivatives Risk Control Dashboard", layout="wide")

MEAN_REVERSION_PERIOD = "5y"  # the actual fix for the mislabeled-period bug


# ---------------------------------------------------------------------------
# Display helper - 1-based index + center alignment, used for every table
# ---------------------------------------------------------------------------

def display_df(df: pd.DataFrame, **kwargs):
    """
    Consistent table display used everywhere in this dashboard:
      - Resets the index to start at 1, not 0.
      - Center-aligns headers and cell values via a pandas Styler.

    Honest caveat: Streamlit's dataframe grid renderer has changed its
    internals across versions, and isn't guaranteed to honor every
    Styler CSS property identically in every version - this is a
    best-effort styling pass, worth a visual check once actually run.
    """
    df = df.copy()
    df.index = range(1, len(df) + 1)
    styled = df.style.set_properties(**{"text-align": "center"}).set_table_styles(
        [{"selector": "th", "props": [("text-align", "center")]}]
    )
    st.dataframe(styled, use_container_width=True, **kwargs)


# ---------------------------------------------------------------------------
# Portfolio <-> editable DataFrame conversion
# ---------------------------------------------------------------------------

def portfolio_to_editable_df(portfolio: list[Position]) -> pd.DataFrame:
    """
    Like portfolio_to_dataframe() in portfolio.py, but shaped for
    st.data_editor specifically: blanks are NaN (proper numeric-column
    behavior for an editable grid), not empty strings (which is the
    right call for the read-only display in portfolio.py, but not for
    an editable one). Kept separate rather than changing
    portfolio_to_dataframe() itself, since that function's existing
    behavior is correct for its own read-only use case.
    """
    rows = []
    for pos in portfolio:
        rows.append({
            "Pair": pos.pair,
            "Instrument": pos.instrument,
            "Direction": pos.direction,
            "Notional": pos.notional,
            "Entry Price": pos.entry_price if pos.entry_price is not None else np.nan,
            "Strike": pos.strike if pos.strike is not None else np.nan,
            "Option Type": pos.option_type if pos.option_type is not None else "",
            "Expiry": pos.expiry,
        })
    return pd.DataFrame(rows)


def editable_df_to_portfolio(df: pd.DataFrame) -> list[Position]:
    """
    The inverse of portfolio_to_editable_df() - reconstructs validated
    Position objects from an edited DataFrame. Reuses Position's own
    __post_init__ validation directly; this function does no
    validation of its own beyond converting blank/NaN cells to None.
    """
    positions = []
    for _, row in df.iterrows():
        entry_price = row["Entry Price"]
        strike = row["Strike"]
        option_type = row["Option Type"]
        expiry = row["Expiry"]
        positions.append(Position(
            pair=row["Pair"],
            instrument=row["Instrument"],
            direction=row["Direction"],
            notional=float(row["Notional"]),
            expiry=expiry if isinstance(expiry, date) else pd.to_datetime(expiry).date(),
            entry_price=None if pd.isna(entry_price) else float(entry_price),
            strike=None if pd.isna(strike) else float(strike),
            option_type=None if (option_type == "" or pd.isna(option_type)) else option_type,
        ))
    return positions


# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def cached_live_rates():
    df = get_live_rates()
    return df, dict(zip(df["Pair"], df["Spot Rate"]))


@st.cache_data(ttl=3600)
def cached_mean_reversion_scan(period: str):
    return scan_all_pairs(period=period)


@st.cache_data(ttl=3600)
def cached_mean_reversion_backtest(period: str):
    return backtest_all_pairs(period=period)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "portfolio" not in st.session_state:
    st.session_state.portfolio = build_portfolio()
if "risk_capital" not in st.session_state:
    st.session_state.risk_capital = 10_000_000

st.title("FX Derivatives Risk Control Dashboard")

rates_df, live_rates = cached_live_rates()

tab_portfolio, tab_risk, tab_stress, tab_signal = st.tabs(
    ["Portfolio & Greeks", "Risk Metrics & Alerts", "Stress Testing", "Signal Creation"]
)

# ---------------------------------------------------------------------------
# TAB 1 - Portfolio & Greeks
# ---------------------------------------------------------------------------
with tab_portfolio:
    st.subheader("Live FX Rates")
    display_df(rates_df)

    st.subheader("Portfolio")

    col_add, col_manage = st.columns([2, 1])

    with col_add:
        with st.expander("Add a position", expanded=False):
            with st.form("add_position_form", clear_on_submit=True):
                pair = st.selectbox("Pair", list(FX_PAIRS.keys()))
                instrument = st.radio("Instrument", ["Futures", "Option"], horizontal=True)
                direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
                notional = st.number_input("Notional", min_value=0.0, value=1_000_000.0, step=100_000.0)

                strike, option_type = None, None
                if instrument == "Option":
                    strike = st.number_input("Strike", min_value=0.0, value=float(live_rates.get(pair, 1.0)))
                    option_type = st.selectbox("Option Type (Call / Put)", ["Call", "Put"])

                expiry = st.date_input("Expiry", value=date.today() + timedelta(days=90), min_value=date.today())
                submitted = st.form_submit_button("Add Position")

                if submitted:
                    try:
                        new_position = Position(
                            pair=pair, instrument=instrument, direction=direction,
                            notional=notional, expiry=expiry,
                            entry_price=live_rates.get(pair) if instrument == "Futures" else None,
                            strike=strike, option_type=option_type,
                        )
                        st.session_state.portfolio.append(new_position)
                        st.success(f"Added {direction} {instrument} on {pair}.")
                    except ValueError as e:
                        st.error(f"Invalid position: {e}")

    with col_manage:
        if st.button("Reset to default 10-position book"):
            st.session_state.portfolio = build_portfolio()
            st.rerun()
        if st.button("Clear all positions"):
            st.session_state.portfolio = []
            st.rerun()

    if st.session_state.portfolio:
        st.caption("Edit any cell directly in the table below to change an existing position.")
        editable_df = portfolio_to_editable_df(st.session_state.portfolio)
        editable_df.index = range(1, len(editable_df) + 1)  # 1-based index, same as display_df()

        edited_df = st.data_editor(
            editable_df,
            use_container_width=True,
            num_rows="fixed",  # add/remove happen via the form/selector above, not here
            column_config={
                "Pair": st.column_config.SelectboxColumn(options=list(FX_PAIRS.keys())),
                "Instrument": st.column_config.SelectboxColumn(options=["Futures", "Option"]),
                "Direction": st.column_config.SelectboxColumn(options=["Long", "Short"]),
                "Option Type": st.column_config.SelectboxColumn(options=["", "Call", "Put"]),
                # Entry Price / Strike deliberately have NO explicit NumberColumn
                # config - letting Streamlit auto-infer the column type from the
                # float64/NaN dtype, rather than forcing NumberColumn, since an
                # explicit NumberColumn appears to render blank cells as the text
                # "None" rather than empty. Auto-inference is the best available
                # attempt at a blank cell; it needs visual confirmation, since
                # this can't be verified without a live editable grid.
                "Notional": st.column_config.NumberColumn(min_value=0.0),
                "Expiry": st.column_config.DateColumn(min_value=date.today()),
            },
            key="portfolio_editor",
        )

        if not edited_df.equals(editable_df):
            try:
                st.session_state.portfolio = editable_df_to_portfolio(edited_df)
                st.rerun()
            except (ValueError, TypeError) as e:
                st.error(f"Couldn't apply edit: {e}")

        remove_idx = st.selectbox(
            "Remove a position (by row)",
            options=list(range(len(st.session_state.portfolio))),
            format_func=lambda i: f"{i + 1}: {st.session_state.portfolio[i].direction} "
                                   f"{st.session_state.portfolio[i].instrument} "
                                   f"{st.session_state.portfolio[i].pair}",
        )
        if st.button("Remove selected position"):
            st.session_state.portfolio.pop(remove_idx)
            st.rerun()
    else:
        st.info("No positions in the book. Add one above, or reset to the default 10-position book.")

    st.subheader("Greeks")
    if st.session_state.portfolio:
        greeks_table = build_greeks_table(st.session_state.portfolio, live_rates)
        display_df(greeks_table)

        greeks_agg = aggregate_greeks(greeks_table)
        cols = st.columns(4)
        cols[0].metric("Total Delta (USD)", f"${greeks_agg['Total Delta (USD)']:,.0f}")
        cols[1].metric("Total Cash Gamma (USD)", f"${greeks_agg['Total Cash Gamma (USD, 1% move)']:,.0f}")
        cols[2].metric("Total Vega (USD)", f"${greeks_agg['Total Vega (USD)']:,.0f}")
        cols[3].metric("Total Theta (USD/day)", f"${greeks_agg['Total Theta (USD)']:,.0f}")
    else:
        greeks_agg = None
        st.info("No positions to price.")


# ---------------------------------------------------------------------------
# TAB 2 - Risk Metrics & Alerts
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Assumed Risk Capital")
    st.caption(
        "Every warning/critical limit below is a percentage of this number - "
        "changing it rescales every limit at once. See DEVLOG.md for how "
        "much this single assumption can matter."
    )
    st.session_state.risk_capital = st.number_input(
        "Risk Capital (USD)", min_value=1_000_000, value=st.session_state.risk_capital,
        step=1_000_000, format="%d",
    )
    risk_limits = build_risk_limits(st.session_state.risk_capital)

    st.divider()
    st.subheader("Historical Simulation Risk Metrics")
    st.caption("Not auto-run - repricing the whole book across a year of history is the slow step in the project.")

    if st.button("Run Risk Analysis (VaR / CVaR / Drawdown / Margin-at-Risk)"):
        if not st.session_state.portfolio:
            st.warning("No positions in the book - nothing to analyze.")
        else:
            with st.spinner("Running historical simulation..."):
                pnl = historical_pnl_series(st.session_state.portfolio, live_rates)
                st.session_state.risk_pnl = pnl
                st.session_state.risk_results = {
                    "VaR 95%": historical_var(pnl, 0.95),
                    "VaR 99%": historical_var(pnl, 0.99),
                    "CVaR 95%": historical_cvar(pnl, 0.95),
                    "CVaR 99%": historical_cvar(pnl, 0.99),
                    "Max Drawdown": max_drawdown(pnl),
                    "Margin-at-Risk": margin_at_risk(pnl),
                }

    if "risk_results" in st.session_state:
        r = st.session_state.risk_results
        # All 5 headline numbers in ONE row of columns - the earlier
        # version split these across two separately-sized column rows
        # (3 then 2), which don't line up with each other. One row of
        # 5 fixes the alignment.
        cols = st.columns(5)
        cols[0].metric("95% VaR", f"${r['VaR 95%']:,.0f}")
        cols[1].metric("99% VaR", f"${r['VaR 99%']:,.0f}")
        cols[2].metric("99% CVaR", f"${r['CVaR 99%']:,.0f}")
        cols[3].metric("Max Drawdown", f"${r['Max Drawdown']:,.0f}")
        cols[4].metric("Margin-at-Risk", f"${r['Margin-at-Risk']:,.0f}")

        st.markdown("#### Risk Metric Comparison")
        chart_df = pd.DataFrame({
            "Metric": ["VaR 95%", "VaR 99%", "CVaR 99%", "Max Drawdown", "Margin-at-Risk"],
            "USD": [r["VaR 95%"], r["VaR 99%"], r["CVaR 99%"], abs(r["Max Drawdown"]), r["Margin-at-Risk"]],
        }).set_index("Metric")
        st.bar_chart(chart_df)

        pnl = st.session_state.risk_pnl
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Cumulative P&L / Drawdown Path (1y replay)")
            st.caption("Walks today's book through the last year's actual daily moves, in sequence.")
            dd = rolling_drawdown(pnl)
            cum_df = pd.DataFrame({
                "Cumulative P&L": pnl.cumsum(),
                "Drawdown from Peak": dd,
            })
            st.line_chart(cum_df)
        with col_b:
            st.markdown("#### Daily P&L Distribution (1y)")
            st.caption(f"The shape VaR/CVaR are read from. 95% VaR marks the left-tail cutoff at ${r['VaR 95%']:,.0f}.")
            counts, bin_edges = np.histogram(pnl, bins=25)
            hist_df = pd.DataFrame({"Daily P&L (USD)": bin_edges[:-1].round(0)}).set_index("Daily P&L (USD)")
            hist_df["Frequency"] = counts
            st.bar_chart(hist_df)

    st.divider()
    st.subheader("Risk Control Room - Alert Status")

    if greeks_agg is None:
        st.info("Add positions in the Portfolio tab to see alert status.")
    else:
        metrics = {
            "Net Delta (USD)": greeks_agg["Total Delta (USD)"],
            "Cash Gamma (USD)": greeks_agg["Total Cash Gamma (USD, 1% move)"],
            "Vega (USD)": greeks_agg["Total Vega (USD)"],
            "Theta (USD/day)": greeks_agg["Total Theta (USD)"],
        }
        if "risk_results" in st.session_state:
            r = st.session_state.risk_results
            metrics.update({
                "VaR 95%": r["VaR 95%"], "VaR 99%": r["VaR 99%"], "CVaR 99%": r["CVaR 99%"],
                "Max Drawdown": r["Max Drawdown"], "Margin-at-Risk": r["Margin-at-Risk"],
            })
        else:
            st.caption("VaR/CVaR/Drawdown/Margin-at-Risk not yet run - status below reflects Greeks only.")

        status_table = evaluate_risk_state(metrics, risk_limits)
        display_df(status_table)

        verdict = overall_status(status_table)
        if verdict == "OK":
            st.success(f"Overall status: {verdict}")
        elif verdict == "WARNING":
            st.warning(f"Overall status: {verdict}")
        elif verdict == "ESCALATE":
            st.error(f"🚨 Overall status: {verdict} — multiple risk dimensions elevated simultaneously")
        else:
            st.error(f"🔴 Overall status: {verdict}")


# ---------------------------------------------------------------------------
# TAB 3 - Stress Testing
# ---------------------------------------------------------------------------
with tab_stress:
    st.subheader("Historical Stress Scenarios")
    st.caption("Not auto-run - pulls three separate historical date windows plus today's volatility baseline.")

    if st.button("Run Stress Tests"):
        if not st.session_state.portfolio:
            st.warning("No positions in the book - nothing to stress.")
        else:
            with st.spinner("Running stress scenarios..."):
                today_vols = {pair: historical_volatility(pair) for pair in FX_PAIRS}
                results = {}
                for name, info in STRESS_SCENARIOS.items():
                    shocks = scenario_shock(info["start"], info["end"])
                    total, breakdown = apply_scenario(
                        st.session_state.portfolio, live_rates, shocks, today_vols
                    )
                    path = scenario_path(
                        st.session_state.portfolio, live_rates, today_vols,
                        info["start"], info["end"],
                    )
                    results[name] = {
                        "total": total, "breakdown": breakdown,
                        "description": info["description"], "path": path,
                    }
                st.session_state.stress_results = results

    if "stress_results" in st.session_state:
        results = st.session_state.stress_results

        st.markdown("#### Scenario Comparison")
        summary_df = pd.DataFrame({
            "Scenario": list(results.keys()),
            "Total P&L (USD)": [r["total"] for r in results.values()],
        }).set_index("Scenario")
        st.bar_chart(summary_df)

        st.divider()

        for name, result in results.items():
            st.markdown(f"**{name}**")
            st.caption(result["description"])

            if not result["path"].empty:
                st.markdown("*Portfolio P&L through the stress window*")
                st.caption(
                    "Day-by-day, not just start vs. end. Note: the path doesn't start at "
                    "exactly zero - volatility is modeled as already elevated from day one "
                    "of the window, while the spot move builds up gradually, so options "
                    "carry an immediate vega-driven jump before spot has moved at all."
                )
                st.line_chart(result["path"])

            display_df(result["breakdown"])
            st.metric("Total Scenario P&L", f"${result['total']:,.2f}")

            # Position-level breakdown chart - visualizes exactly which
            # position drove the scenario's P&L, not just the total.
            chart_data = result["breakdown"].copy()
            chart_data["Position"] = (
                chart_data["Pair"] + " " + chart_data["Instrument"] + " " + chart_data["Direction"]
            )
            chart_data = chart_data.set_index("Position")[["P&L (USD)"]]
            st.bar_chart(chart_data)

            st.divider()


# ---------------------------------------------------------------------------
# TAB 4 - Signal Creation (renamed from Mean Reversion, moved to last)
# ---------------------------------------------------------------------------
with tab_signal:
    st.subheader("Signal Creation - Mean Reversion Scanner")
    st.caption(
        "Methodology: builds a log-price spread between every pair of the project's 5 FX "
        "pairs (10 combinations), automatically chaining through whichever currency two "
        "pairs share. A rolling 20-day z-score flags dislocations beyond \u00b12 standard "
        "deviations. Every flagged spread is then run through an Augmented Dickey-Fuller "
        "stationarity test - only spreads that pass are treated as genuinely mean-reverting, "
        "not just currently stretched. Backtested against a frozen entry-time baseline "
        "(comparing exit price to the mean/std level AS THEY WERE at entry, not a "
        "constantly-recalculating one - see DEVLOG.md, Problem 4, for why that distinction "
        "matters) over 5 years of history."
    )

    if st.button("Run Signal Scan"):
        with st.spinner("Scanning all 10 pair combinations (5 years of history)..."):
            st.session_state.signal_scan = cached_mean_reversion_scan(MEAN_REVERSION_PERIOD)
            st.session_state.signal_backtest = cached_mean_reversion_backtest(MEAN_REVERSION_PERIOD)

    if "signal_scan" in st.session_state:
        st.markdown("#### Current Signal Scan")
        display_df(st.session_state.signal_scan)

        st.markdown("#### Backtest (5-year history, joined with stationarity result)")
        display_df(st.session_state.signal_backtest)
