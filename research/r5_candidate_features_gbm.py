"""R5: nonlinear test of candidate ETH-specific features before adopting any (2025 only).

LightGBM probes on the real decision-point features (train cache), fit Jan-Jun 2025, scored
Jul-Dec 2025 (1-day gap). Targets: standardized 30 s and window-close increments (direction)
and |increment| / naive scale (magnitude).  Feature sets:
  base       : the BTC-recipe features (what the model already has)
  +slowz     : recent moves scaled by 1-hour vol instead of 5-min vol (ETH reversion research)
  +btc       : BTC lead / gap / activity features at the same second (cross-asset research)
A candidate is only worth adopting if it improves out-of-sample R^2 materially.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from btcmodel.models import VOL_FEATS, DIR_FEATS  # noqa: E402
from btcmodel.dataset import CACHE  # noqa: E402

BTL = "/home/user/data/timeline"
ETL = "/home/user/data_eth/timeline"


def ffill(a):
    m = np.isnan(a)
    i = np.where(~m, np.arange(len(a)), 0)
    np.maximum.accumulate(i, out=i)
    return a[i]


cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])


def cross_feats(T0, e, v):
    """BTC features at ETH decision points (both timelines start 2025-01-01, same index)."""
    n = int((T0 + e).max()) + 2
    bs = np.log(np.load(os.path.join(BTL, "s_close.npy"), mmap_mode="r")[:n])
    bp = np.log(ffill(np.load(os.path.join(BTL, "p_close.npy"), mmap_mode="r")[:n].copy()))
    bad = ~np.isfinite(bp)
    bp[bad] = bs[bad]
    bcnt = np.load(os.path.join(BTL, "p_ntr.npy"), mmap_mode="r")[:n].astype(float)
    es = np.log(np.load(os.path.join(ETL, "s_close.npy"), mmap_mode="r")[:n])
    rb, re_ = np.diff(bp, prepend=bp[0]), np.diff(es, prepend=es[0])
    c = np.maximum(T0 + e - 1, 0)
    cb2, cxy, ccnt, cbas = cs(rb ** 2), cs(re_ * rb), cs(bcnt), cs(bp - bs)
    vb = np.sqrt(np.maximum((cb2[c + 1] - cb2[c - 299]) / 300, 1e-14))
    vb = np.maximum(vb, 2e-6)
    beta = (cxy[c + 1] - cxy[c - 3599]) / np.maximum(cb2[c + 1] - cb2[c - 3599], 1e-18)
    F = {"btc_beta_1h": beta}
    for L in (1, 2, 3, 5, 10, 30, 60, 300):
        F[f"btc_rp_{L}"] = (bp[c] - bp[c - L]) / (vb * np.sqrt(L))
        F[f"eth_gap_{L}"] = ((es[c] - es[c - L]) - beta * (bp[c] - bp[c - L])) / (v * np.sqrt(L))
    for L in (10, 60, 300):
        F[f"btc_bdev_{L}"] = ((bp[c] - bs[c]) - (cbas[c + 1] - cbas[c - L + 1]) / L) / vb
        F[f"btc_lcnt_{L}"] = np.log((ccnt[c + 1] - ccnt[c - L + 1]) / L + 1e-3)
        F[f"btc_lrv_{L}"] = np.log((cb2[c + 1] - cb2[c - L + 1]) / L + 1e-14)
    return pd.DataFrame({k: np.asarray(x, np.float32) for k, x in F.items()})


def slowz(d):
    v1h = np.sqrt(np.exp(d["lrv_3600"].values.astype(float)))
    v5 = np.sqrt(np.exp(d["lrv_300"].values.astype(float)))
    out = {}
    for L in (30, 60, 120, 300, 900):
        out[f"rz1h_{L}"] = d[f"r_{L}"].values * np.maximum(v5, 2e-6) / np.maximum(v1h, 2e-6)
    out["xz1h"] = d["x_v"].values * np.maximum(v5, 2e-6) / np.maximum(v1h, 2e-6)
    return pd.DataFrame({k: np.asarray(x, np.float32) for k, x in out.items()})


def r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.mean((y - y.mean()) ** 2)


if __name__ == "__main__":
    base = sorted(set(VOL_FEATS + DIR_FEATS) - {"abs_x_v", "wrange_v"})
    cols = base + ["T0", "day", "xcur", "v", "h30__inc", "h30__valid", "close__inc", "close__H"]
    d = pd.read_parquet(os.path.join(CACHE, "train.parquet"), columns=cols)
    d = d.iloc[::2].reset_index(drop=True)                     # every other row: speed
    e = d["e"].values.astype(np.int64)
    X = pd.concat([d[base].reset_index(drop=True), slowz(d), cross_feats(d["T0"].values, e, d["v"].values.astype(float))], axis=1)
    sets = {"base": base, "+slowz": base + list(slowz(d.iloc[:1]).columns),
            "+btc": base + [c for c in X.columns if c.startswith(("btc_", "eth_gap"))]}
    sets["+slowz+btc"] = sorted(set(sets["+slowz"]) | set(sets["+btc"]))
    mid = d["day"].quantile(0.5)
    tr = (d["day"] < mid - 1).values
    te = (d["day"] > mid + 1).values
    P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=3000,
             feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1,
             num_threads=4, seed=0)
    targets = {}
    v = d["v"].values.astype(float)
    m30 = d["h30__valid"].values
    targets["dir 30s"] = (np.clip(d["h30__inc"].values / (v * np.sqrt(30)), -8, 8), m30)
    H = d["close__H"].values.astype(float)
    targets["dir close"] = (np.clip(d["close__inc"].values / (v * np.sqrt(H)), -8, 8), np.ones(len(d), bool))
    targets["mag 30s"] = (np.clip(np.abs(d["h30__inc"].values) / (v * np.sqrt(30)), 0, 20), m30)
    for tn, (y, mask) in targets.items():
        line = []
        for sn, fs in sets.items():
            a, b = tr & mask, te & mask
            m = lgb.train(P, lgb.Dataset(X.loc[a, fs], y[a]), num_boost_round=300)
            line.append(f"{sn} {100 * r2(y[b], m.predict(X.loc[b, fs])):.3f}%")
        print(f"{tn:10s} OOS R2: " + " | ".join(line), flush=True)
