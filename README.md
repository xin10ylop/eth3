# ETH 5-minute reference-relative prediction

Continuous forecasts of how **ETH** trades relative to the **fixed opening price of the current
5-minute candle** (P0), at five horizons (10 s, 15 s, 30 s, 60 s, and the 5-minute close):
`P(S >= P0)`, price quantiles, touch probabilities for `P0·(1±k)`, and up-first / down-first
probabilities - updated every second of the window.

This repository is an **independent replication** on ETHUSDT of the method developed on BTCUSDT
in [xin10ylop/btcmovement2](https://github.com/xin10ylop/btcmovement2). Every design choice was
frozen on BTC and pre-registered before any ETH data was inspected
(`artifacts/PREREGISTRATION.md`): same data sources, splits, features, regime-state features,
architecture, hyper-parameters and decision rules - no ETH-specific tuning. The ETH holdout
(2026-05-02 → 2026-11-02) is therefore a clean out-of-sample test of whether the method
generalizes beyond BTC.

**Status:** pipeline running; results, figures and packaged models will be added here.
