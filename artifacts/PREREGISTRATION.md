# ETH replication - pre-registered before any ETH data was downloaded or inspected

(First committed in xin10ylop/btcmovement2 at commit 93fdb4d, before the ETH download started;
copied here as this repository's first commit, still before any ETH data was inspected.)

ETHUSDT is treated as an **independent replication** of the method developed on BTCUSDT.
Every design choice is inherited from BTC (xin10ylop/btcmovement2, artifacts/PREREGISTRATION.md) and frozen:

1. Same data sources (Binance spot ETHUSDT 1s klines = reference market; USD-M perpetual
   ETHUSDT aggTrades), same period 2025-01-01 .. 2026-11-02, same splits and embargo:
   train 2025-01-02..2025-12-31, valid 2026-01-02..2026-04-30, test 2026-05-02..2026-11-02.
2. Same features, regime-state features (incl. online close-pressure intensity), heads
   (10s/15s/30s/60s/close), architecture (scale, no-move atom, drift, empirical law,
   calibration from prior, touch and first-passage layers), hyper-parameters, cross-fitting,
   and the equal-weight logit blend with the direct GBM. **No ETH-specific tuning.** The ETH
   validation set is used only for early stopping, exactly as on BTC.
3. **Final ETH model** = walk-forward regime-aware refit through 2026-04-30 (early stopping
   on the last 30 days before that), scored once on the ETH holdout.
4. **Reference** = regime-blind model fitted on 2025 only (heads h30/h60/close, stacking
   h30 -> h60 -> close), scored on the same holdout rows.
5. Online recalibration adopted per head only if it improves the ETH holdout log-loss with a
   day-block bootstrap CI excluding zero (same rule as BTC).
6. ETH research scripts (2025 + validation only) are descriptive; they cannot change 1-5.
7. Deployment models: same recipe refit on all data through 2026-11-02 (2025 thinned 1:2).

## Amendment 1 - before any ETH data was inspected (user request)

The user asked for the full research process on ETH rather than a frozen copy of the BTC
design ("be logical, don't blindly copy-paste"). This amendment is written after the raw
download finished and before any ETH data was looked at. It replaces points 2-3 above:

* ETH gets its own research phase on **2025 data only** (validation only to confirm), with
  explicit hypotheses: (a) BTC leads ETH at the seconds scale, so cross-asset features can make
  ETH more predictable than BTC; (b) ETH's ~25x larger relative tick changes discreteness,
  the no-move atom and last-trade bounce; (c) perp-spot lead and clock structure may differ;
  (d) close pressure / regime drift may differ from BTC.
* Design changes motivated by that research are selected on the **validation set only**,
  against the BTC recipe applied to ETH as the baseline. Every adopted change and the final
  design are committed here before the holdout is scored.
* Unchanged: data sources, period, splits and embargo; the holdout (2026-05-02 .. 2026-11-02)
  is scored once, at the end, with the walk-forward protocol (fit through 2026-04-30); the
  regime-blind 2025-only reference; the online-recalibration adoption rule; the deployment refit.

## Amendment 2 - final ETH design, decided from 2025 research before any holdout scoring

Research results (artifacts/research_r*.txt, 2025 data; close pressure also Jan-Apr 2026):
* R1: ETH is *less* predictable than BTC on its own signals (next-10s OOS R2 3.6% vs 5.1%);
  weaker perp->spot lead (0.135 vs 0.172); smaller no-move atom despite a 25x larger relative
  tick (ETH moves more); clock structure essentially identical.
* R2: BTC does **not** lead ETH beyond ETH's own perpetual (+0.02 pp R2 at 10 s); the lead runs
  ETH -> BTC instead (+0.85 pp, x1.17 for BTC spot).
* R3: ETH close pressure from Feb 2026 is weak (+0.03..+0.05 sigma, t <= 2.2; BTC: up to t = 7.5).
* R4: ETH mean-reverts after large 2-3 min moves, generically (placebo anchor as strong as P0).
* R5 (LightGBM probe): BTC cross-asset features +0.02 pp (direction) / +0.04 pp (magnitude);
  1h-vol-scaled moves +0.00 pp. Neither clears the bar for adding a second live data feed.

Final ETH design: the BTC architecture with every component re-estimated on ETH data (clock
profile, scale, no-move atom, drift, residual law, calibration, touch/first-passage, regime
state incl. close pressure), no cross-asset inputs, BTC hyper-parameters and blend rule.
Holdout protocol unchanged (Amendment 1).

## Amendment 3 - ETH reversal features adopted (validation only; before any holdout scoring)

R6 (artifacts/research_r6_predictability_profile.txt) showed the one dimension in which ETH is
more predictable than BTC: it overshoots and mean-reverts after large moves (reversal spread
+0.03..+0.06 after >1.5-sigma moves over 60-600 s, t = 28-57; BTC -0.01..+0.02). Candidate
features: recent moves and window excursions in 1-hour-vol units (models.REV_FEATS). Rule fixed
before the test: adopt only if the close head improves on validation with a day-block CI
excluding zero and the 30 s / 60 s heads are not worse. Result (paired, identical rows):
30 s -0.00007 [-0.00012, -0.00001]; 60 s -0.00021 [-0.00028, -0.00014];
close -0.00013 [-0.00022, -0.00003]  ->  adopted (artifacts/compare_valid_frozen_vs_rev.txt).
Final ETH model = Amendment 2 design + REV_FEATS + regime-state features, walk-forward.
