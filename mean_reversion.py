"""
mean_reversion.py

A mean-reversion signal scanner across every combination of the
project's five FX pairs (10 combinations total). This is the 
"embedded pairs trade" piece of the project: relative-value 
thinking applied across the whole book, not just a single 
instrument in isolation.

The subtlety this module handles correctly: not every pair combination
can be spread the same way. EUR/USD, GBP/USD, SGD/USD, and INR/USD are
all quoted the same direction (USD per unit of the other currency), so
subtracting their logs cleanly cancels USD and leaves an implied cross
rate. USD/JPY is quoted the OPPOSITE direction (JPY per USD). Naively
subtracting logs for a USD/JPY combination would produce a number that
isn't a real cross rate at all - it needs the logs ADDED instead, to
correctly chain through the shared USD. Getting this wrong wouldn't
throw an error; it would just generate a meaningless "signal"
that happened to run without crashing.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from data_layer import FX_PAIRS, get_historical_data


def _pair_currencies(pair: str) -> tuple[str, str]:
    base, quote = pair.split("/")
    return base, quote


def compute_cross_spread(pair_a: str, pair_b: str, period: str = "1y") -> tuple[pd.Series, str]:
    """
    Computes the log spread between two FX pairs, automatically
    detecting which currency they share and combining them the correct
    way. Returns (spread series, name of the implied cross rate).

    Three cases, depending on which side the shared currency sits on:
      - Same quote currency (e.g. EUR/USD & GBP/USD both quote USD):
        SUBTRACT logs. Cancels the shared quote, leaves base_a/base_b.
      - pair_a's base = pair_b's quote (e.g. USD/JPY chained with
        EUR/USD): ADD logs. Chains through the shared currency.
      - pair_a's quote = pair_b's base: same chaining, ADD logs.
    """
    base_a, quote_a = _pair_currencies(pair_a)
    base_b, quote_b = _pair_currencies(pair_b)

    history = get_historical_data(period=period)
    a = history[pair_a]["Close"]
    b = history[pair_b]["Close"]
    df = pd.DataFrame({"a": a, "b": b}).dropna()

    if quote_a == quote_b:
        spread = np.log(df["a"]) - np.log(df["b"])
        implied = f"{base_a}/{base_b}"
    elif base_a == quote_b:
        spread = np.log(df["a"]) + np.log(df["b"])
        implied = f"{base_b}/{quote_a}"
    elif quote_a == base_b:
        spread = np.log(df["a"]) + np.log(df["b"])
        implied = f"{base_a}/{quote_b}"
    else:
        raise NotImplementedError(
            f"{pair_a} and {pair_b} don't share a currency in a way "
            "this project's five pairs require - not implemented."
        )

    return spread, implied


def rolling_stats(spread: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Returns the rolling mean and rolling std that rolling_zscore()
    combines into one number. Exposed separately because the backtest
    below needs to FREEZE these at the moment of entry, rather than
    let them keep recalculating through the life of the trade.
    """
    return spread.rolling(window).mean(), spread.rolling(window).std()


def rolling_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling z-score: how many standard deviations the spread currently
    sits from its own recent rolling average.
    """
    mean, std = rolling_stats(spread, window)
    return ((spread - mean) / std).dropna()


def generate_signal(z: float, entry_threshold: float = 2.0) -> str:
    """
    Classic mean-reversion entry rule. A z-score beyond +/-2 standard
    deviations is a statistically unusual dislocation - roughly a
    1-in-20 event for a single observation under a normal assumption -
    worth betting reverts back toward the recent average. Inside that
    band, nothing unusual is happening; no signal.
    """
    if z > entry_threshold:
        return "Short Spread"
    elif z < -entry_threshold:
        return "Long Spread"
    return "Neutral"


def stationarity_test(spread: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test: the formal statistical check for
    whether a spread is actually mean-reverting (stationary) or is
    really just trending / a random walk with no fixed level to revert
    to. An elevated "z-score breach rate" is a symptom worth noticing;
    this test is the actual diagnosis.

    Null hypothesis (H0): the series has a unit root - it is NOT
    stationary. We reject H0 (conclude it likely IS mean-reverting)
    when the p-value falls below a conventional threshold, 0.05 here.

    A pair that fails this test (p >= 0.05) has no statistical basis
    to be traded as mean-reverting, no matter how often its z-score
    crosses +/-2 - the entire premise of the signal doesn't hold for
    that specific spread, and any "signal" it generates is noise
    dressed up as a strategy.
    """
    result = adfuller(spread.dropna(), result_object=False)
    p_value = result[1]
    return {"adf_statistic": result[0], "p_value": p_value, "is_stationary": p_value < 0.05}


def _tradable_signal(raw_signal: str, is_stationary: bool) -> str:
    """
    Combines the raw z-score signal with the stationarity result into
    one honest verdict. A breach on a non-stationary spread isn't a
    real trade idea.
    """
    if raw_signal == "Neutral":
        return "Neutral"
    if not is_stationary:
        return f"{raw_signal} (ignore - not stationary)"
    return raw_signal


def scan_all_pairs(pairs: list = None, window: int = 20, period: str = "1y") -> pd.DataFrame:
    """
    Runs the mean-reversion signal across every combination of the
    project's FX pairs (10 combinations across the 5 pairs), including
    the ADF stationarity test for each one. Returns a summary table
    sorted by how extreme each combination's current z-score is.
    """
    pairs = pairs or list(FX_PAIRS.keys())

    rows = []
    for pair_a, pair_b in combinations(pairs, 2):
        spread, implied = compute_cross_spread(pair_a, pair_b, period=period)
        z = rolling_zscore(spread, window=window)
        if z.empty:
            continue

        latest_z = z.iloc[-1]
        raw_signal = generate_signal(latest_z)
        stat = stationarity_test(spread)

        rows.append({
            "Pair A": pair_a,
            "Pair B": pair_b,
            "Implied Cross": implied,
            "Latest Z-Score": latest_z,
            "Days |z|>2": int((z.abs() > 2.0).sum()),
            "ADF p-value": stat["p_value"],
            "Stationary?": "Yes" if stat["is_stationary"] else "No",
            "Tradable Signal": _tradable_signal(raw_signal, stat["is_stationary"]),
        })

    df = pd.DataFrame(rows)
    return df.reindex(df["Latest Z-Score"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def backtest_from_zscore(
    spread: pd.Series, z: pd.Series, entry_threshold: float = 2.0,
    holding_period: int = 10, window: int = 20,) -> pd.DataFrame:
    """
    The actual backtest logic - deliberately kept separate from
    compute_cross_spread() so it can be tested against a hand-built
    series with a known correct answer, not only real market data
    where the "right" answer isn't known in advance.

    CRITICAL METHODOLOGY POINT, found via a real bug: reversion is
    checked against the mean/std level that was in effect AT ENTRY,
    FROZEN at that point - not against a constantly-recalculating
    rolling mean. An earlier version of this function compared the
    z-score at entry to the z-score at exit directly, and it produced
    a 100% hit rate across every single pair tested, including ones
    that had already failed the ADF stationarity test. That result
    was too good to be true: a rolling mean naturally absorbs an 
    extreme reading into its own average as time passes, which 
    mechanically drags the z-score back toward zero even when the 
    spread itself never moves anywhere. Verified directly - a spread 
    engineered to jump once and then stay flat forever (zero genuine 
    reversion) still showed its z-score decay from 4.24 to 0.90 over 
    10 days, purely because the rolling window absorbed the jump into 
    its own "new normal." Comparing against a FROZEN entry-time 
    baseline instead tests whether the spread itself reverted, not 
    whether the moving average simply caught up to it.

    Rule: a NEW trade opens only when z CROSSES INTO the signal zone
    from inside it (not on every day it remains beyond threshold - a
    spread that stays stretched for two weeks is one dislocation, not
    fourteen). Each trade holds for `holding_period` trading days,
    then closes.
    """
    rolling_mean, rolling_std = rolling_stats(spread, window)

    z_values, z_dates = z.values, z.index
    trades = []
    in_position = False
    entry_index = None
    position_direction = None

    for i in range(1, len(z_values)):
        current_z, prev_z = z_values[i], z_values[i - 1]

        if not in_position:
            if prev_z <= entry_threshold < current_z:
                in_position, position_direction, entry_index = True, "Short Spread", i
            elif prev_z >= -entry_threshold > current_z:
                in_position, position_direction, entry_index = True, "Long Spread", i
        else:
            reached_holding_limit = (i - entry_index) >= holding_period
            data_ran_out = i == len(z_values) - 1
            if reached_holding_limit or data_ran_out:
                entry_date = z_dates[entry_index]
                exit_date = z_dates[i]

                entry_z = z_values[entry_index]
                # The frozen baseline: the mean/std AS THEY WERE at
                # entry, looked up once and never recalculated.
                
                frozen_mean = rolling_mean.loc[entry_date]
                frozen_std = rolling_std.loc[entry_date]
                exit_spread_level = spread.loc[exit_date]

                # Where does the spread sit today, measured against
                # the SAME baseline that flagged it as extreme in the
                # first place - not against a baseline that has since
                # moved to accommodate wherever the spread ended up.
                frozen_exit_z = (exit_spread_level - frozen_mean) / frozen_std

                if position_direction == "Short Spread":
                    reverted = frozen_exit_z < entry_z
                    z_move = entry_z - frozen_exit_z
                else:
                    reverted = frozen_exit_z > entry_z
                    z_move = frozen_exit_z - entry_z

                trades.append({
                    "Entry Date": entry_date,
                    "Exit Date": exit_date,
                    "Direction": position_direction,
                    "Entry Z": entry_z,
                    "Exit Z (vs frozen baseline)": frozen_exit_z,
                    "Reverted": reverted,
                    "Z Move (favorable=+)": z_move,
                })
                in_position = False

    return pd.DataFrame(trades)


def backtest_signal(
    pair_a: str, pair_b: str, window: int = 20, entry_threshold: float = 2.0,
    holding_period: int = 10, period: str = "1y",) -> tuple[pd.DataFrame, dict]:
    """
    Full backtest for one pair combination: builds the spread and
    z-score, then runs backtest_from_zscore() on it. Returns the
    individual trade log plus a summary (trade count, hit rate,
    average favorable z-move per trade).
    """
    spread, implied = compute_cross_spread(pair_a, pair_b, period=period)
    z = rolling_zscore(spread, window=window)
    trades = backtest_from_zscore(spread, z, entry_threshold=entry_threshold,
                                   holding_period=holding_period, window=window)

    if trades.empty:
        return trades, {"implied": implied, "num_trades": 0, "hit_rate": None, "avg_z_move": None}

    summary = {
        "implied": implied,
        "num_trades": len(trades),
        "hit_rate": trades["Reverted"].mean(),
        "avg_z_move": trades["Z Move (favorable=+)"].mean(),
    }
    return trades, summary


def backtest_all_pairs(
    pairs: list = None, window: int = 20, entry_threshold: float = 2.0,
    holding_period: int = 10, period: str = "1y",) -> pd.DataFrame:
    """
    Runs backtest_signal() across all 10 pair combinations and joins
    the result with each pair's stationarity verdict - the key check:
    pairs that PASSED the ADF test should show meaningfully better hit
    rates than pairs that FAILED it, if the stationarity filter is
    doing real work rather than being a theoretical add-on.
    """
    pairs = pairs or list(FX_PAIRS.keys())
    rows = []

    for pair_a, pair_b in combinations(pairs, 2):
        spread, implied = compute_cross_spread(pair_a, pair_b, period=period)
        z = rolling_zscore(spread, window=window)
        if z.empty:
            continue

        stat = stationarity_test(spread)
        trades = backtest_from_zscore(spread, z, entry_threshold=entry_threshold,
                                       holding_period=holding_period, window=window)

        rows.append({
            "Pair A": pair_a,
            "Pair B": pair_b,
            "Implied Cross": implied,
            "Stationary?": "Yes" if stat["is_stationary"] else "No",
            f"# Trades ({period})": len(trades),
            "Hit Rate": trades["Reverted"].mean() if not trades.empty else np.nan,
            "Avg Z Move": trades["Z Move (favorable=+)"].mean() if not trades.empty else np.nan,
        })

    return pd.DataFrame(rows).sort_values("Stationary?", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Extended to 5 years of history: with only 8-12 trades per
    # pair, the 1-year hit rates weren't statistically trustworthy on
    # their own (well within noise for that sample size). Five years
    # of history gives the ADF test a longer, more reliable series to
    # judge, and gives the backtest roughly 5x more trades per pair to
    # estimate a hit rate from - still not huge, but meaningfully
    # better grounded than 8-12 observations.
    LOOKBACK = "5y"

    print(f"Scanning all 10 pair combinations for mean-reversion signals ({LOOKBACK} history)...\n")
    results = scan_all_pairs(period=LOOKBACK)
    print(results.to_string(index=False))

    stationary_count = (results["Stationary?"] == "Yes").sum()
    active = results[~results["Tradable Signal"].isin(["Neutral"]) & ~results["Tradable Signal"].str.contains("ignore")]
    print(f"\n{stationary_count} of {len(results)} spreads pass the stationarity test.")
    print(f"{len(active)} of {len(results)} combinations currently show a tradable signal.")

    print(f"\nBacktesting all 10 combinations over {LOOKBACK} "
          "(10-day holding period per trade)...\n")
    backtest_results = backtest_all_pairs(period=LOOKBACK)
    print(backtest_results.to_string(index=False))
