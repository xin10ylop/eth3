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

**Headline (untouched 6-month holdout, May-Nov 2026, walk-forward model trained through April 2026):**
log-loss skill vs a naive Gaussian random walk of **11.3% / 9.2% / 6.4% / 4.3% / 2.8%** for the
10s / 15s / 30s / 60s / close heads, positive in 7 of 7 months for every head and significant in
every market regime, calibrated (ECE <= 0.006). In absolute terms the ETH 5-minute outcome is
**more predictable than BTC's** over the same period (final log-loss 0.462 vs 0.487) - section 3
explains why, and why ETH's *direction* is nonetheless less predictable than BTC's.

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

## 3. Is ETH more predictable than BTC? Yes and no - precisely

Same holdout period (2026-05-02 → 2026-11-02), same protocol, final models of both repositories:

| holdout log-loss of P(S >= P0) | ETH naive | **ETH final** | BTC naive | BTC final |
|---|---|---|---|---|
| 10 s | 0.216 | 0.192 | 0.217 | **0.177** |
| 30 s | 0.343 | 0.321 | 0.342 | **0.305** |
| 60 s | 0.441 | 0.422 | 0.442 | **0.410** |
| **5-minute close** | 0.475 | **0.462** | 0.529 | 0.487 |
| close, last 30 s of the window | 0.163 | **0.144** | 0.322 | 0.215 |

* **Short horizons (10-60 s): BTC is more predictable.** Its spot lags its perpetual more and
  its volatility is more forecastable (research R1/R6); ETH's directional AUC at 10 s is 0.630
  vs 0.662.
* **The 5-minute outcome: ETH is more predictable**, and increasingly so as the window unfolds.
  Not because ETH's direction is easier to call (at the open both are ~coin flips: AUC 0.531
  vs 0.540) but because ETH's path around its open stays *decided*. In 2025 both assets were
  equally "clean" (P(final side = side at e=285): 95.0% vs 95.6%); from 2026 BTC's endings got
  noisier (93.3% ETH vs 91.1% BTC, more late crossings of P0) because of the BTC-specific
  pull-to-the-open in the final seconds (research R3/R7). ETH barely shows it.
* **Overshoot-reversion is ETH's own edge** (3-5x BTC), now exploited by the reversal features.
* Consequence for the skill numbers: the naive baseline is already good on ETH's clean
  endings, so ETH's *skill over naive* is smaller even though its absolute forecast is better.
  The ETH advantage on the 5-minute outcome depends on the close-pressure regime staying
  BTC-specific; the regime-state features monitor it live.

## 4. Results on the untouched holdout (53,280 windows)

| head | naive Gaussian | **final** | skill | ΔLL [95% CI] | ECE | AUC at window open |
|---|---|---|---|---|---|---|
| 10 s | 0.2162 | **0.1918** | **11.3%** | −0.0245 [−0.0265, −0.0227] | 0.0013 | 0.620 |
| 15 s | 0.2581 | **0.2345** | **9.2%** | −0.0237 [−0.0256, −0.0219] | 0.0013 | 0.603 |
| 30 s | 0.3432 | **0.3211** | **6.4%** | −0.0221 [−0.0237, −0.0205] | 0.0020 | 0.578 |
| 60 s | 0.4413 | **0.4223** | **4.3%** | −0.0190 [−0.0203, −0.0178] | 0.0041 | 0.561 |
| 5 m close | 0.4748 | **0.4617** | **2.8%** | −0.0132 [−0.0146, −0.0117] | 0.0060 | 0.531 |

* **Consistency:** positive skill in **7 of 7 months for all five heads**; significant in all 50
  head x regime cells (vol terciles, weekday/weekend, US session, trend, rising vol, and the 5%
  most volatile days, where skill is still 1.0-4.3%). `artifacts/robustness_test_wf5.txt`,
  `artifacts/regime_test_wf5.txt`.
* **Close head:** skill grows from 0.2% at the open to 11.6% in the last 30 s; quantile coverage
  4.4/9.1/24.1/50.7/76.0/90.5/95.3% at nominal 5/10/25/50/75/90/95%; magnitude forecast 6.2%
  better than the naive scale; touch probabilities ΔLL −0.038 vs the Brownian formula.
  Full tables: `artifacts/tables_test_wf5.md`; figures: `artifacts/figures/`.
* **Regime-aware + reversal model vs the regime-blind 2025-only reference**, paired on identical
  rows (`artifacts/compare_test_frozen_vs_wf5.txt`): better on every head - 30 s −0.00095
  [−0.00119, −0.00073], 60 s −0.00106 [−0.00131, −0.00081], close −0.00046 [−0.00070, −0.00021].
* **Online recalibration: not adopted for ETH** - no head clears the pre-registered rule (all
  CIs include zero). Unlike BTC, ETH's calibration does not drift, consistent with its weak
  close-pressure regime.
* The live predictor reproduces the batch pipeline exactly on all five heads
  (`artifacts/predictor_consistency.txt`, max difference 0.0).

## 5. Usage

Two packaged model sets (same design; load either with `Predictor.load_dir`):

| directory | trained on | use |
|---|---|---|
| `models/wf5_20260430/` | 2025-01 → 2026-04-30 | **the holdout-evaluated models** - every number in section 4 |
| `models/prod_20261102/` | 2025-01 → 2026-11-02 (2025 thinned to every other window = recency tilt; early stopping on 2026-10-03 → 11-02) | **deployment** - same recipe on the newest data; no untouched data remains to score it, so its quality is inferred from the wf5 holdout |

```python
from btcmodel.predictor import Predictor
pred = Predictor.load_dir("models/prod_20261102")         # deployment (or models/wf5_20260430)
pred.ingest(arrays, t0_epoch)   # >= 9 days of ETHUSDT 1s spot klines + perp 1s bars, hour-aligned
out = pred.predict(now_epoch_second, p0=window_open_price)
out["heads"]["close"]["p_above_ref"]           # P(5-minute close >= P0)
out["heads"]["h10"]["quantiles_usd"]           # ETH price distribution 10 s ahead
out["heads"]["close"]["touch"]["+10bps"]       # P(trade at >= P0*1.001 before the close)
```

`python scripts/demo_live.py` replays a holdout window second by second through this interface.
Rebuild everything: `python -m btcmodel.data 2025-01-01 2026-11-02 6`, then
`bash scripts/run_asset_pipeline.sh timeline` (ETH is this repository's default asset).

## 6. Limitations

* Reference feed is Binance ETHUSDT spot; contracts settling on another feed need a refit on it.
* Probability forecasts, not a strategy: no fees, latency or impact.
* The 5-minute direction at the open is barely predictable (AUC 0.53).
* Regimes drift (BTC's close pressure appeared in Feb 2026; ETH's could too): refit monthly.

## 7. Repository map

| path | content |
|---|---|
| `research/r1..r7*.py` | the ETH research (outputs in `artifacts/research_*.txt`) |
| `btcmodel/` | data, timeline, features, regime state, models (incl. `REV_FEATS`), events, predictor |
| `scripts/` | pipeline (`run_asset_pipeline.sh`), training, frozen evaluation, walk-forward, analysis |
| `artifacts/` | pre-registration with amendments, results, tables, figures |
| `models/wf5_20260430/` | packaged holdout-evaluated ETH models |
| `models/prod_20261102/` | packaged deployment ETH models, refit through 2026-11-02 |
| `tests/` | lookahead tests, live-vs-batch consistency |
