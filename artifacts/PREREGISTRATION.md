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
