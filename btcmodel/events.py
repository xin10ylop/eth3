"""Path events relative to the fixed reference P0.

Touch:  P(price trades at/above P0*(1+k) (k>0) or at/below P0*(1+k) (k<0) at some point
        within the horizon), for levels not already crossed by the current price.
First passage: barriers P0*(1 +/- k); P(up barrier first), P(down first), P(neither) within
        the horizon, for current price strictly between the barriers.

Both use a structural Brownian prior (closed-form, with the head's own scale and drift)
and a LightGBM correction boosted *from* that prior (init_score).  The touch corrector
is monotone in the standardized distance to the level, so P(touch) is guaranteed to
decrease as the level moves away - coherent across thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit, expit

from .models import _fit
from .prob import p_touch, p_double_barrier

TOUCH_LEVELS = (-20, -10, -5, 5, 10, 20)       # bps relative to P0
FP_LEVELS = (5, 10, 20)

TOUCH_FEATS = ["d_std", "mu_to", "lsig_rel", "e", "lH", "is_down", "lrvsa_300", "jump_300", "som",
               "lev_std", "lp_struct", "ncross", "hod_et"]
FP_FEATS = ["a_std", "b_std", "mu", "lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "hod_et",
            "lpu", "lpd"]


def touch_rows(d: pd.DataFrame, head: str, model, levels=TOUCH_LEVELS):
    """Expand samples x levels into rows (only non-trivial: level not yet crossed)."""
    mu, s, H = model.bm_params(d)
    x = d["xcur"].values
    rows = []
    for k in levels:
        L = np.log1p(k * 1e-4)
        if k > 0:
            dist = L - x
            hit = d[f"{head}__mx"].values >= L
        else:
            dist = x - L
            hit = d[f"{head}__mn"].values <= L
        ok = dist > 0
        sgn = 1.0 if k > 0 else -1.0
        p = np.clip(p_touch(dist[ok], sgn * mu[ok], s[ok], H[ok]), 1e-6, 1 - 1e-6)
        r = pd.DataFrame({
            "row": np.flatnonzero(ok), "k": k, "y": hit[ok].astype(np.int8),
            "d_std": (dist[ok] / d["sig"].values[ok]).astype(np.float32),
            "mu_to": (sgn * d["mu"].values[ok]).astype(np.float32),
            "is_down": np.float32(k < 0),
            "lev_std": (abs(L) / d["sig"].values[ok]).astype(np.float32),
            "p_struct": p, "lp_struct": logit(p).astype(np.float32),
        })
        for c in ["lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "ncross", "hod_et", "day"]:
            r[c] = d[c].values[ok]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def naive_touch(d: pd.DataFrame, head: str, levels=TOUCH_LEVELS):
    """Driftless BM with naive trailing vol (baseline)."""
    H = d[f"{head}__H"].values.astype(float)
    s = d["v"].values.astype(float)
    x = d["xcur"].values
    out = []
    for k in levels:
        L = np.log1p(k * 1e-4)
        dist = (L - x) if k > 0 else (x - L)
        ok = dist > 0
        out.append(np.clip(p_touch(dist[ok], 0.0, s[ok], H[ok]), 1e-6, 1 - 1e-6))
    return np.concatenate(out)


class TouchModel:
    def fit(self, tr_rows, va_rows, seed=0, log=print):
        mono = [(-1 if f == "d_std" else 0) for f in TOUCH_FEATS]
        m = _fit(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=10000,
                      lambda_l2=50.0, seed=seed), tr_rows[TOUCH_FEATS], tr_rows["y"].values,
                 va_rows[TOUCH_FEATS], va_rows["y"].values,
                 init_tr=tr_rows["lp_struct"].values, init_va=va_rows["lp_struct"].values,
                 es=200, monotone=mono)
        self.m, self.rounds = m, m.best_iteration
        log(f"   touch corrector: {self.rounds} rounds")
        return self

    def predict(self, rows):
        if self.rounds <= 0:
            return rows["p_struct"].values
        return expit(rows["lp_struct"].values +
                     self.m.predict(rows[TOUCH_FEATS], num_iteration=self.rounds, raw_score=True))


def fp_rows(d: pd.DataFrame, head: str, model, levels=FP_LEVELS, naive=False):
    """Rows for first-passage events. Class: 0 neither, 1 up first, 2 down first."""
    if naive:
        H = d[f"{head}__H"].values.astype(float)
        s, mu = d["v"].values.astype(float), np.zeros(len(d))
    else:
        mu, s, H = model.bm_params(d)
    x = d["xcur"].values
    rows = []
    for k in levels:
        U, D = np.log1p(k * 1e-4), np.log1p(-k * 1e-4)
        fp = d[f"{head}__fp{k}"].values
        ok = (x < U) & (x > D) & (fp != 2)
        a, b = U - x[ok], x[ok] - D
        pu, pd_ = p_double_barrier(a, b, mu[ok], s[ok], H[ok])
        pu, pd_ = np.clip(pu, 1e-6, 1), np.clip(pd_, 1e-6, 1)
        pn = np.clip(1 - pu - pd_, 1e-6, 1)
        z = pu + pd_ + pn
        y = np.where(fp[ok] == 1, 1, np.where(fp[ok] == -1, 2, 0)).astype(np.int8)
        r = pd.DataFrame({"row": np.flatnonzero(ok), "k": k, "y": y,
                          "pn": pn / z, "pu": pu / z, "pd": pd_ / z})
        if not naive:
            sig = d["sig"].values[ok]
            r["a_std"] = (a / sig).astype(np.float32)
            r["b_std"] = (b / sig).astype(np.float32)
            r["lpu"] = np.log(r["pu"].values / r["pn"].values).astype(np.float32)
            r["lpd"] = np.log(r["pd"].values / r["pn"].values).astype(np.float32)
            for c in ["mu", "lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "hod_et", "day"]:
                r[c] = d[c].values[ok]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


class FirstPassageModel:
    def fit(self, tr, va, seed=0, log=print):
        init_tr = np.log(tr[["pn", "pu", "pd"]].values)
        init_va = np.log(va[["pn", "pu", "pd"]].values)
        m = _fit(dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=31,
                      min_data_in_leaf=10000, lambda_l2=50.0, seed=seed),
                 tr[FP_FEATS], tr["y"].values, va[FP_FEATS], va["y"].values,
                 init_tr=init_tr, init_va=init_va, es=200)
        self.m, self.rounds = m, m.best_iteration
        log(f"   first-passage corrector: {self.rounds} rounds")
        return self

    def predict(self, rows):
        base = np.log(rows[["pn", "pu", "pd"]].values)
        if self.rounds > 0:
            base = base + self.m.predict(rows[FP_FEATS], num_iteration=self.rounds, raw_score=True)
        P = np.exp(base - base.max(1, keepdims=True))
        return P / P.sum(1, keepdims=True)
