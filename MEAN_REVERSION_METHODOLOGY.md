# Mean Reversion Module — Methodology Explained

Everything here maps directly to `mean_reversion.py` 

## 1. The Big Picture — What Is This Module Actually Doing?

We hold five currency pairs. Some of them tend to move together,
because they're driven by similar things (both pairs involve a strong
or weak US Dollar, for instance). This module asks one question, over
and over, for every pair of currencies we hold:

> **"Are these two currencies drifting apart from their usual
> relationship right now — and if so, is that drift the kind that
> tends to snap back, or the kind that keeps going?"**

## 2. Concept 1: What Is a "Spread"?

Take two currency pairs we hold — say EUR/USD and GBP/USD. On their
own, each just tells we "how many US Dollars is 1 Euro/Pound worth."

But if we compare EUR/USD **against** GBP/USD instead of looking 
at either alone, the US Dollar part mostly cancels out, and what's 
left is essentially the exchange rate between the Euro and the Pound 
— even though we never directly downloaded that exchange rate.

Using Real numbers, from the data, an example of this (CFA Program covers this):
EUR/USD = 1.166   (1 Euro = 1.166 US Dollars)
GBP/USD = 1.359   (1 Pound = 1.359 US Dollars)
1.166 / 1.359 = 0.858


That 0.858 is (approximately) the real-world EUR/GBP rate — 1 Euro is
worth about 0.858 Pounds. And it is derived without ever pulling EUR/GBP
data directly. **This derived number is the "spread."** It's a single
number that tracks the *relative* value of Euro vs. Pound, stripped of
whatever the Dollar itself is doing that day.

In the code, this uses `log` (natural logarithm) instead of plain
division — `ln(1.166) - ln(1.359)` instead of `1.166 / 1.359` — for a
technical-but-simple reason: log differences behave more symmetrically.
A move from 1.0 to 1.1 and a move from 1.0 to 0.9 are "the same size"
in log terms (roughly +0.095 and -0.105), whereas in plain division
they look very different in scale. This symmetry matters for the
statistics.

## 3. Concept 2: Why We Can't Just Divide Every Pair — The Chaining Problem

Here's a subtle bug was avoided before it ever ran.

EUR/USD, GBP/USD, SGD/USD, and INR/USD are all quoted **the same
direction** — "how many US Dollars for 1 unit of the other currency."
Dividing any two of them cancels the Dollar cleanly, exactly like the
EUR/GBP example above.

**USD/JPY is quoted the opposite direction** — "how many Japanese Yen
for 1 US Dollar". If we blindly divided USD/JPY by EUR/USD the same way
the Dollar wouldn't cancel out cleanly — we'd get a number that *looks* 
like a valid spread (the code would run without any error) but isn't a real 
exchange rate at all. It'smathematically meaningless, and there's no warning.

**The fix: for USD/JPY combinations, we multiply instead of divide.**

USD/JPY = 159.32   (1 US Dollar = 159.32 Yen)
EUR/USD = 1.166    (1 Euro = 1.166 US Dollars)
159.32 × 1.166 = 185.77

That 185.77 is the real-world EUR/JPY rate — 1 Euro is worth about
185.77 Yen. This works because the US Dollar appears on *different
sides* in the two pairs (Yen-per-Dollar vs. Dollar-per-Euro), so
multiplying lets the Dollar "pass through", instead of cancelling out 
like it did in the division case.

**In the code**, `compute_cross_spread()` checks which currency two
pairs share and *automatically* decides divide vs. multiply. It was
verified against real-world approximate rates (0.858 for EUR/GBP,
185.8 for EUR/JPY) before being trusted, and further confirmed by
checking that reaching EUR/JPY two different ways (via USD/JPY then
EUR/USD, or the reverse order) gives the identical number.

## 4. Concept 3: The Z-Score — "How Unusual Is Today, Really?"

Once we have a spread (for eg: the EUR/GBP-equivalent number), we
track it every day for the past year. Some days it's a bit higher,
some days a bit lower — it wobbles around some rough average level.

**The z-score answers: "is today's wobble bigger than normal, or
totally ordinary?"**

z-score = (today's value − recent average) / (recent day-to-day variability)

- **z = 0** → today is exactly average. Nothing unusual.
- **z = +1** → today is a bit above average, but well within normal range.
- **z = +2 or beyond** → today is unusually far from average — the
  kind of day that only happens roughly 1 time in 20 if the spread
  behaves like a normal bell curve.

**In the code**, `rolling_zscore()` computes this using the last 20
trading days (about one calendar month) as the "recent average"
window — recalculated fresh every day, so it's always comparing today
to the *recent* past, not some fixed number from a year ago.

## 5. Concept 4: The Trading Rule

Once we have a z-score, the actual "signal" is a simple rule,
implemented in `generate_signal()`:

- **z > +2** → the spread is unusually *wide*. Bet that it narrows
  back toward normal -> labeled `"Short Spread"`.
- **z < -2** → the spread is unusually *narrow*. Bet that it widens
  back toward normal -> labeled `"Long Spread"`.
- **Anything in between** → nothing unusual → `"Neutral"`, do nothing.

This is the entire trading logic. The complexity in this module isn't
in this rule — it's in making sure the rule is only trusted where it
actually deserves to be trusted, which is the next section.

## 6. Concept 5: The Problem With Z-Scores Alone

Here's the problem, and it's the reason this module got a serious
upgrade partway through building it.

A z-score threshold assumes the spread behaves like a **rubber band**
— it can stretch further than usual sometimes, but it always wants to
snap back to roughly the same resting length. That's  **mean reversion**, 
and it's the entire premise the trading rule depends on.

But not every spread actually behaves like a rubber band. Some behave
like a **rope being pulled steadily in one direction** — drifting
further and further away with no pull back toward where it started.
That's **trending** (or, in the statistics term, **non-stationary**).

**Why this matters practically:** a z-score alone can't tell these two
apart. A steadily trending spread will *also* cross the ±2 threshold
regularly — and that does not mean it's about to snap back, but simply 
because it keeps moving further from wherever its "recent average" used 
to be. The signal looks identical either way. **Hence, We need a separate test 
to tell a rubber band from a rope.**

This is exactly what happened when this was first tested: several
pairs were crossing the ±2 threshold **20 to 34 times** out of roughly
230 trading days — 9-15% of the time. If these were true rubber bands,
that should happen only about 4.56% of the time (Assuming normality, Z > 2 
means that between -2 and 2 std of the mean, there's 95.44% data, while 
the tails only have about 4.56%. It is these tail values is what we 
are betting on for mean reversion) The gap between 4.56% and 9-15% was 
the tell that some of these spreads were ropes, not rubber bands, being 
mistaken for tradable signals. This is where I realised how I could use 
the Dickey-Fuller tests I learnt from the CFA curriculum. More on that below.

## 7. Concept 6: The ADF Test — Telling Rubber Bands From Ropes

The **Augmented Dickey-Fuller test** (ADF test) is a formal statistical
test that answers exactly this: *does this specific series actually
behave like a rubber band (mean-reverting) or a rope (trending)?*

We only need to interpret  the **p-value**.

- **Low p-value (below 0.05)** → strong evidence this spread IS a
  rubber band. Trustworthy to trade as mean-reverting.
- **High p-value (0.05 or above)** → not enough evidence it's a rubber
  band. Could easily be a rope. **Don't trust signals from this pair,
  even if the z-score crosses ±2.**

**This was tested against two series built specifically to have known
answers**, before trusting it on real data:

- A series deliberately built to be a rubber band (always pulled back
  toward zero) → p-value came back as `0.0000000000028` — extremely
  low, correctly identified as mean-reverting.
- A pure random walk, built to be a rope with no pull-back at all →
  p-value came back as `0.81` — correctly identified as NOT mean-
  reverting.

Only after confirming the test could tell known examples apart
correctly was it trusted on our real portfolio data.

**In the code**, `stationarity_test()` runs this and returns both the
raw p-value and a simple Yes/No verdict (`is_stationary`) using the
0.05 cutoff.

---

## 8. Putting It Together: What Actually Happens When We Run It

`scan_all_pairs()` is the function that ties everything above into one
report. Step by step, for **every possible pair of our 5 currency
pairs (10 combinations total)**:

1. Build the spread (dividing or multiplying correctly, per Section 3)
2. Compute today's z-score against the last 20 trading days
3. Check: does today's z-score cross ±2? → raw signal
4. Run the ADF test on the whole spread history → stationary or not
5. Combine steps 3 and 4 into one honest verdict — **if the z-score
   crosses the threshold but the ADF test fails, the signal gets
   explicitly labeled `"(ignore - not stationary)"`** rather than
   looking identical to a real signal
6. Repeat for all 10 combinations, and sort so the most extreme
   z-scores show up first

---

## 9. What We Actually Found When We Ran This

**10 of 10** combinations computed correctly | Including every USD/JPY combination — the chaining logic worked |
**Only 2 of 10** passed the stationarity test | EUR/JPY (p = 0.041) and GBP/SGD (p = 0.012) |
**0 of 10 showed an active tradable signal** | None currently sit beyond ±2 — separate from stationarity, this is just where the numbers happen to sit today |

**The story here:** I tested whether that specific pair actually 
deserved to be trusted as mean-reverting — and found 8 didn't, 
while two *other* combinations in the book did. 

## 10. Problems Hit Along the Way, and How They Were Fixed

### Problem: A file named `signal.py` broke pandas itself

The module was originally saved as `signal.py`. This is also the name
of a built-in Python library (used for handling things like Ctrl+C
keyboard interrupts) — and other libraries, including `pandas` and
`pyarrow`, quietly depend on that built-in internally.

The moment a personal file shared that exact name, Python got confused
about which "signal" was being asked for, and the failure showed up as
a strange, unrelated-looking crash **inside pandas**, nowhere near the
actual mistake. The fix was simply renaming the file to
`mean_reversion.py` — nothing about the logic itself was wrong.

**Lesson:** never name a project file after a Python standard library
module (`signal`, `types`, `email`, `token`, etc).

### Problem: Z-score threshold breaches were unusually frequent

Covered in detail in Section 6 above — the fix was adding the ADF
stationarity test rather than trusting the z-score threshold alone.

### Problem: A future-compatibility warning from the ADF test itself

The statistics library used for the ADF test (`statsmodels`) printed a
warning that its output format is scheduled to change in a future
release. Not an error — the numbers were correct either way — but left
unaddressed, a future library update could silently change what the
code receives back. Fixed by explicitly pinning the current output
format (`result_object=False`), so the code's behavior won't shift
underneath it later.

## 11. Quick Reference — Function Sheet

`_pair_currencies()` | Splits `"EUR/USD"` into `("EUR", "USD")` |
`compute_cross_spread()` | Builds the spread between two pairs, auto-detecting divide vs. multiply |
`rolling_zscore()` | How unusual today's spread is vs. its last 20 days |
`generate_signal()` | Turns a z-score into Short/Long/Neutral using the ±2 rule |
`stationarity_test()` | The ADF test — is this spread a rubber band or a rope? |
`_tradable_signal()` | Combines the z-score signal with the stationarity verdict into one honest label |
`scan_all_pairs()` | Runs everything above across all 10 pair combinations and returns one summary table |