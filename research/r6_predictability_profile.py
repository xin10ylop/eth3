"""R6: in WHICH sense is ETH more (or less) predictable than BTC?  2025 only, identical method.

Dimensions (decision points every 7 s; fit Jan-Jun 2025, score Jul-Dec 2025):
  1. direction   : standardized next-H return, LightGBM on own signals          (R2)
  2. magnitude   : |next-H return| / naive scale, LightGBM on activity signals    (R2)
  3. tails       : P(|next-300s move| > 2 naive sigma), LightGBM                  (AUC)
  4. serial      : autocorrelation of 1-min / 5-min returns, variance ratios
  5. reversal    : P(up next 90 s | move over last L s beyond +-1.5 sigma), L = 60..600
  6. by regime   : direction R2 at 30 s split by US hours / weekend / vol tercile
"""
import os
from datetime import date

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

TL = {"ETH": "/home/user/data_eth/timeline", "BTC": "/home/user/data/timeline"}
N = (date(2026, 1, 1) - date(2025, 1, 1)).days * 86400
P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=3000, feature_fraction=0.8,
         bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=3, seed=0)
cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])


def ffill(a):
    m = np.isnan(a)
    i = np.where(~m, np.arange(len(a)), 0)
    np.maximum.accumulate(i, out=i)
    return a[i]


def r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.mean((y - y.mean()) ** 2)


def profile(asset):
    g = lambda k: np.asarray(np.load(os.path.join(TL[asset], k + ".npy"), mmap_mode="r")[:N])
    ls = np.log(g("s_close"))
    lp = np.log(ffill(g("p_close").copy()))
    bad = ~np.isfinite(lp)
    lp[bad] = ls[bad]
    hl = np.log(g("s_high")) - np.log(g("s_low"))
    cnt = g("s_count").astype(float) + g("p_ntr").astype(float)
    buy, sell = g("p_buy").astype(float), g("p_sell").astype(float)
    r = np.diff(ls, prepend=ls[0])
    c2, chl, cc, cb, cbu, cse = cs(r ** 2), cs(hl ** 2), cs(cnt), cs(lp - ls), cs(buy), cs(sell)
    t = np.arange(90000, N - 301, 7)
    S = lambda c_, L: (c_[t + 1] - c_[t + 1 - L])
    v = np.maximum(np.sqrt(S(c2, 300) / 300), 2e-6)
    vh = np.maximum(np.sqrt(S(c2, 3600) / 3600), 2e-6)
    b = lp - ls
    Xd = [(b[t] - S(cb, L) / L) / v for L in (10, 60, 300, 1800)]
    Xd += [(ls[t] - ls[t - L]) / (v * np.sqrt(L)) for L in (1, 2, 5, 15, 30, 60, 120, 300, 600)]
    Xd += [(lp[t] - lp[t - L]) / (v * np.sqrt(L)) for L in (1, 2, 5, 15, 60)]
    Xd += [(S(cbu, L) - S(cse, L)) / (S(cbu, L) + S(cse, L) + 1e-9) for L in (5, 30, 300)]
    Xd += [(ls[t] - ls[t - L]) / (vh * np.sqrt(L)) for L in (60, 150, 300, 600)]
    Xm = [np.log(S(c2, L) / L + 1e-14) - np.log(v ** 2) for L in (10, 30, 60, 900, 3600, 14400)]
    Xm += [np.log(S(chl, L) / L + 1e-14) - np.log(v ** 2) for L in (60, 300)]
    Xm += [np.log(S(cc, L) / L + 1e-3) for L in (10, 60, 300, 3600)]
    Xm += [(t % 3600).astype(float), ((t // 3600) % 168).astype(float), np.log(v)]
    Xd = np.clip(np.column_stack(Xd), -10, 10)
    Xm = np.column_stack(Xm)
    tr, te = t < N // 2 - 86400, t > N // 2 + 86400
    out = {}
    for H in (30, 120, 300):
        y = np.clip((ls[t + H] - ls[t]) / (v * np.sqrt(H)), -8, 8)
        m = lgb.train(P, lgb.Dataset(Xd[tr], y[tr]), 250)
        out[f"1 direction R2 {H}s %"] = 100 * r2(y[te], m.predict(Xd[te]))
        if H == 30:
            pred30, y30 = m.predict(Xd[te]), y[te]
    for H in (30, 300):
        y = np.clip(np.abs(ls[t + H] - ls[t]) / (v * np.sqrt(H)), 0, 20)
        m = lgb.train(P, lgb.Dataset(np.c_[Xm, Xd][tr], y[tr]), 250)
        out[f"2 magnitude R2 {H}s %"] = 100 * r2(y[te], m.predict(np.c_[Xm, Xd][te]))
    y = (np.abs(ls[t + 300] - ls[t]) > 2 * v * np.sqrt(300)).astype(float)
    m = lgb.train(dict(P, objective="binary"), lgb.Dataset(np.c_[Xm, Xd][tr], y[tr]), 250)
    out["3 tail event rate %"] = 100 * y[te].mean()
    out["3 tail AUC (|5m move| > 2 sigma)"] = roc_auc_score(y[te], m.predict(np.c_[Xm, Xd][te]))
    m1 = ls[::60]
    r1 = np.diff(m1)
    for k in (1, 2, 5):
        out[f"4 autocorr 1-min returns lag {k}"] = np.corrcoef(r1[k:], r1[:-k])[0, 1]
    r5 = np.diff(ls[::300])
    out["4 autocorr 5-min returns lag 1"] = np.corrcoef(r5[1:], r5[:-1])[0, 1]
    for k in (5, 15, 60):
        rk = np.diff(ls[:: 60 * k])
        out[f"4 variance ratio {k}-min vs 1-min"] = rk.var() / (k * r1.var())
    for L in (60, 150, 300, 600):
        z = (ls[t] - ls[t - L]) / (vh * np.sqrt(L))
        f = ls[t + 90] - ls[t]
        hi, lo = (z > 1.5) & (f != 0), (z < -1.5) & (f != 0)
        sp = (f[lo] > 0).mean() - (f[hi] > 0).mean()
        out[f"5 reversal spread after {L}s move >1.5sig"] = sp
        out[f"5   t-stat ({L}s)"] = sp / np.sqrt(0.25 / lo.sum() + 0.25 / hi.sum())
    how = ((t[te] // 3600) % 168)
    hod, dow = how % 24, how // 24
    us = (dow < 5) & (hod >= 13) & (hod < 21)
    wk = dow >= 5
    vq = np.quantile(v[te], [1 / 3, 2 / 3])
    for nm, m_ in [("US hours (13-21 UTC wkday)", us), ("weekend", wk), ("low vol", v[te] < vq[0]), ("high vol", v[te] > vq[1])]:
        out[f"6 direction R2 30s, {nm} %"] = 100 * r2(y30[m_], pred30[m_])
    return out


if __name__ == "__main__":
    res = {a: profile(a) for a in ("ETH", "BTC")}
    print(f"{'dimension (2025, OOS Jul-Dec)':48s} {'ETH':>10s} {'BTC':>10s}   more predictable")
    for k in res["ETH"]:
        e_, b_ = res["ETH"][k], res["BTC"][k]
        verdict = ""
        if any(s in k for s in ("R2", "AUC")):
            verdict = "ETH" if e_ > b_ else "BTC"
        print(f"{k:48s} {e_:10.4f} {b_:10.4f}   {verdict}")
