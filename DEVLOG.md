# DEVLOG - Problems Encountered & Fixes

A running log of real engineering issues hit during the build, how each
was diagnosed, and how it was fixed (or, where still open, what the plan
is)

## Module 2 - Options Pricing and Greeks

### Problem 1: Raw Gamma isn't comparable across currency pairs

**What happened:** After aggregating Greeks across the ten-position
book, INR/USD's Gamma came out to ~4.16 billion - several orders of
magnitude larger than every other pair, which sat in the tens of
millions. It dominated the portfolio total to the point of making the
aggregate number meaningless.

**Root cause:** Gamma's formula has a `1/S` term (spot price in the
denominator). INR/USD trades at ~0.0105 - roughly three orders of
magnitude smaller than a pair like USD/JPY at ~159. That tiny
denominator mechanically inflates the raw Gamma number. It's an
artifact of the quoting convention, not a genuine difference in risk
between the two positions.

**Fix:** Added a **Cash Gamma** metric - the P&L impact of a 1% move in
the underlying, computed as:

Cash Gamma = 0.5 × Gamma × (0.01 × Spot)²

Squaring the spot term cancels out the `1/S` scaling problem, making
Gamma genuinely comparable across pairs regardless of their quote
level. After the fix, INR/USD's Cash Gamma came out in line with the 
other four pairs - confirming the fix actually worked rather than 
just moving the problem elsewhere.

**Status:** Fixed

### Problem 2: Cash Gamma is still denominated in different currencies per pair

**What happened:** Cash Gamma solved the *scale* problem but not a
second, subtler one underneath it: each pair's Cash Gamma is expressed
in that pair's own domestic currency. EUR/USD's Cash Gamma is in USD.
USD/JPY's is in JPY. `aggregate_greeks()` currently sums "Total Cash
Gamma" across the whole book, which silently adds USD figures to JPY
figures as if they were the same unit. They aren't.

**Root cause:** Garman-Kohlhagen naturally prices (and Greeks) in the
domestic currency of whichever pair it's applied to. Nothing in the
pipeline yet converts these back to one common reporting currency
before summing across positions.

**Fix:** Built a dedicated `fx_conversion.py` module and wired it into
`pricing.py`. Two separate conversions were needed, not one - Theta,
Vega, and Cash Gamma are P&L figures in each pair's *domestic* (quote)
currency, while Delta represents notional exposure in each pair's
*foreign* (base) currency, which needs an extra step (multiply by spot
to get into domestic terms first, then convert to USD). Tested both
directions explicitly: USD/JPY is the one pair in the book where 
foreign = USD and domestic = JPY, the reverse of the other four - 
confirmed its Delta round-trips back to its original USD value rather 
than getting double-converted, and its Theta/Vega/Cash Gamma now shrink 
correctly once converted out of JPY

**Status:** Fixed. `aggregate_greeks()` now sums the USD-converted
columns; the raw per-currency columns stay in the table too, for
reference.

## Module 5 - Signal and Stress Testing

### Problem 3: A filename silently broke pandas itself

**What happened:** Built the mean-reversion signal module and saved it
as `signal.py`. Testing the stress-testing module (built right after,
in the same session) failed with a error deep inside pandas:`AttributeError: 
partially initialized module 'pyarrow.lib' has no attribute 'KeyValueMetadata'

**Root cause:** `signal` is a Python **standard library module** (used
for OS-level signal handling). Naming
a project file `signal.py` in the working directory shadows it: any
`import signal` from that point on resolves to the wrong file instead.

**Fix:** Renamed `signal.py` to `mean_reversion.py`. Nothing else about
the module changed - purely a naming collision with Python's own
standard library.

**Status:** Fixed. General lesson worth keeping: avoid naming project
files after Python standard library modules (`signal`, `types`,
`token`, `email`, etc.) - the failure mode is exactly this kind of
confusing, seemingly unrelated error somewhere else entirely.

### Problem 4: A backtest that was too good to be true

**What happened:** After building `backtest_all_pairs()`, running it on
the real portfolio data showed a **100% hit rate on every single one
of the 10 pair combinations** - including the 8 that had already
*failed* the ADF stationarity test two steps earlier. That result
directly contradicts what the stationarity test was supposed to mean:
a spread that fails the test isn't supposed to reliably revert, yet
every single one appeared to.

**Root cause, found by deliberately trying to break it:** built a
synthetic spread engineered to jump once and then stay permanently at
its new level - zero genuine reversion, by construction. Its z-score
was 4.24 right at the jump, and had decayed to 0.90 just 10 trading
days later - while the spread's actual level had moved by less than
0.02 (essentially nothing). The original backtest compared the z-score
at entry to the z-score at exit directly - but the z-score's rolling
mean/std recalculate every day, so a rolling window naturally
*absorbs* an extreme reading into its own average as time passes. That
mechanically drags the z-score back toward zero regardless of whether
the underlying spread ever actually moved - making every trade look
like a "win" whether or not any real reversion happened.

**Fix:** Rewrote the backtest to freeze the mean and standard
deviation *at the moment of entry*, and measure the exit purely
against that frozen baseline - not a baseline that has since drifted
to accommodate wherever the spread ended up. Re-ran the exact
jump-and-never-revert scenario that exposed the bug: the actual jump
trade now correctly shows `Reverted: False` (entry z=4.24, frozen-
baseline exit z=4.32 - essentially unchanged, correctly read as no
reversion). Re-ran a genuinely mean-reverting synthetic series
afterward too, to confirm the fix didn't break the case that should
still show a high hit rate - it didn't (100% hit rate, unchanged).

**Status:** Fixed. General lesson worth keeping: **a suspiciously
perfect backtest result is a bug report, not a success** - the fix
came from refusing to accept a 100% hit rate at face value and
deliberately constructing a test case designed to break the logic,
rather than assuming the code was right because it ran without error.

### Finding: Stationarity results don't hold up across different lookback windows

**What happened:** Re-ran the full scan and backtest with 5 years of
history instead of 1. Neither pair that had passed the ADF test at 1
year (EUR/JPY, p=0.042; GBP/SGD, p=0.012) still passed at 5 years
(p=0.76 and p=0.092 respectively). A different pair entirely - EUR/SGD
- became the only pair to pass (p=0.045), despite showing no signs of
stationarity at 1 year (p=0.50). The 1-year test's hit-rate edge for
stationary pairs (~71% vs ~58%) also largely disappeared at 5-year
scale, with hit rates clustering around 50-62% across the board
regardless of stationarity status.

**Why, most likely:** unlike a classic equity pairs trade - where two
similar companies in the same industry often share a persistent
economic anchor keeping their prices linked over long periods - an FX
cross-rate has no equivalent structural anchor. It's driven by two
countries' central bank policy cycles, which diverge and converge on
their own multi-year timelines. A 5-year window very plausibly spans
multiple different policy regimes for any given pair, so "what looked
mean-reverting" is naturally regime-dependent rather than a fixed,
stable property of the pair.

**Why this matters:** a genuinely robust mean-reversion relationship 
should hold up whether tested over 1 year or 5 - the fact that it doesn't, 
here, is itself the real result. This isn't a bug to fix; it's evidence 
that this project's version of the signal, as currently built, does not 
have a persistent statistical edge that would be safe to trade on without 
much more rigorous testing (e.g. checking stability across several 
intermediate window lengths, not just two).

**Status:** Documented as a finding. No further action for this
project's scope - the honest limitation is worth observing
rather than cherry-picking whichever window happens to look best.

### Finding: The stress-test day-by-day P&L path doesn't start at zero

**What happened:** When stress testing was extended to show the
day-by-day P&L path through a historical window, not just the
start-to-end total, the path consistently showed an immediate jump on
day one rather than starting near zero.

**Why:** volatility is held at the scenario's overall realized level
for the *entire* path, while the spot shock builds up gradually from
day one. Options therefore carry an instant vega-driven P&L move
before spot has genuinely moved at all - a mechanical consequence of
holding volatility constant across the path, not a bug in the
day-by-day accumulation logic itself. Verified: the path's *final*
value still matches the independently-computed start-to-end total
exactly, confirming the underlying math is sound.

**Status:** Documented as a known, deliberate simplification. A more
granular model would let volatility ramp up across the window too,
rather than applying it as a single step-change from day one - a
reasonable next iteration, not a fix required for this project's
scope.

## Module 6 - Dashboard Assembly

### Problem 5: Dashboard silently ran the Mean Reversion tab on 1 year, not 5

**What happened:** The dashboard's Mean Reversion tab was labeled
"Backtest (5-year history...)" and displayed results that looked
plausible - but trade counts (8-12 per pair) matched what Module 5's
1-year backtest produced, not the 40-54 range confirmed for a genuine
5-year run. The label was correct about intent; the code silently
wasn't honouring it.

**Root cause:** `app.py` called `scan_all_pairs()` and
`backtest_all_pairs()` with no `period` argument. Both functions
default to `period="1y"` (a sensible default for standalone script use
- see mean_reversion.py). The dashboard never overrode that default,
so it silently ran on 1 year of data while the UI text claimed 5 -
two things drifting out of sync without either one throwing an error.

**Fix:** Added an explicit `MEAN_REVERSION_PERIOD = "5y"` constant in
`app.py` and passed it into both cached wrapper functions. Confirmed
via `inspect.signature()` that the underlying functions' defaults are
unchanged (still `1y`, correct for their own standalone use), and that
`app.py` now explicitly overrides that default rather than relying on
it silently.

**Status:** Fixed. General lesson: a UI label and the code producing
what it displays are two separate things that can drift apart
silently - worth spot-checking that a "5-year" claim is backed by an
actual `period="5y"` argument somewhere, not just assumed from a
function's name or a docstring.

*This file grows as the project does - new entries get added under each
week's heading as issues come up, fixed or not.*
