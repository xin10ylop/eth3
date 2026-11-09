# ETH 5-minute reference-relative prediction

Continuous forecasts of how **ETH** trades relative to the **fixed opening price of the current
5-minute candle** (P0), at five horizons (10 s, 15 s, 30 s, 60 s and the 5-minute close):
`P(S >= P0)`, price quantiles, touch probabilities for `P0·(1±k)`, and up-first / down-first
probabilities, updated every second of the window.

The method was developed on BTC in [xin10ylop/btcmovement2](https://github.com/xin10ylop/btcmovement2).
ETH was **not** copied over blindly: it went through its own research phase (2025 data only),
every ETH-specific hypothesis was tested, and the one that held up was built into the model and
validated before the holdout was touched. The decision trail is in `artifacts/PREREGISTRATION.md`
(committed step by step, each amendment before the corresponding results were seen).

> RESULTS_PLACEHOLDER

---

## 1. What is different about ETH (research, 2025 data)

All comparisons use identical code on ETH and BTC (`research/r*.py`, outputs in `artifacts/research_*.txt`).

| 2025 | ETH | BTC |
|---|---|---|
| annualized volatility | 65% | 38% |
| tick (0.01) relative to 1 s volatility | 0.029 | 0.0014 |
| perp → spot lead (lag-1 corr) | 0.135 | 0.172 |
| own-signal direction R², next 30 s / 2 min / 5 min | 2.0 / 0.43 / 0.06% | 4.2 / 1.5 / 0.47% |
| magnitude (\|move\|) R², next 30 s / 5 min | 4.9 / 7.7% | 5.8 / 21.7% |
| P(price unchanged over 10 s) | 7.5% | 18.0% |
| 1-min return autocorrelation, variance ratio 5m/1m | **−0.013, 0.96** | −0.002, 1.02 |
| **reversal after a >1.5σ move over 60 / 150 / 300 / 600 s** | **+0.032 / +0.050 / +0.054 / +0.064** (t 28–57) | −0.007 / +0.008 / +0.013 / +0.023 |
| close pressure toward P0 in the last 10 s, Feb–Apr 2026 | +0.03..+0.05σ (t ≤ 2.2) | +0.14..+0.20σ (t up to 7.5) |

**Why ETH is "more predictable" - and in what sense only.** In the ordinary sense ETH is *less*
predictable than BTC: its spot tracks its own perpetual less tightly, its volatility is harder
to forecast, and its tails are less anticipatable. But ETH **overshoots**: large moves partially
reverse within one to two minutes, 3-5x more than in BTC, and as strongly from an arbitrary
anchor as from the 5-minute open (`research/r4_reversal_placebo.py`), i.e. it is a property of
how ETH absorbs liquidity shocks, not a reference-price effect. A plausible mechanism is that
ETH's order books are thinner relative to its volatility, so aggressive flow pushes the price
past fair value and liquidity providers pull it back. ETH's 1-minute returns are negatively
autocorrelated and its variance ratio is below one: a mean-reverting market at the minute scale.

**What did not hold up.** BTC does *not* lead ETH beyond what ETH's own perpetual already
shows: adding BTC features raises ETH's next-30 s R² by only +0.02 pp, linear or LightGBM
(`r2`, `r5`). The lead actually runs the other way: ETH's perpetual adds +0.85 pp (x1.17) to
BTC spot's next-10 s R², because the perpetuals lead both spots and BTC spot is the slowest of
the four markets. A cross-asset input was therefore not worth making the live ETH system
depend on a second market feed.

**What carried over.** The clock structure of volatility (1.6x at each 5-minute boundary,
0.74x at the end of the window, 3.7% of the window's variance left at e = 285) and the
regime drift of microstructure, so the clock-aware scale model and the regime-state features
are kept and re-estimated on ETH.

## 2. ETH-specific design

* Every component is re-estimated on ETH: intraday clock profile, scale model, no-move atom,
  drift, empirical residual law, calibration, touch and first-passage layers, regime state.
* **Reversal features** (`btcmodel/models.py: REV_FEATS`): recent moves and pull-backs from the
  window's high/low measured in *1-hour* volatility units, so a big move is not damped by the
  short-term volatility it creates itself. Adopted only after passing a rule fixed in advance on
  the validation set (better on every head; close-head ΔLL −0.00013, CI [−0.00022, −0.00003]).
* No cross-asset inputs (tested, rejected).
