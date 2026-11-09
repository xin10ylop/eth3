"""Reference-relative predictive distribution models.

Architecture (one instance per head: 30s, 60s, window close)
------------------------------------------------------------
Let x = log(S_now / P0) (known), R = log(S_target / S_now) the future increment.
Then log(S_target / P0) = x + R, so EVERY reference-relative question is a question
about the conditional law of R, read at a reference-dependent point:

    P(S_target >= P0)        = P(R >= -x)
    P(S_target >= P0(1+k))   = P(R >= log(1+k) - x)
    quantiles of S_target/P0 = x + quantiles of R
    touch / first-passage    = law of the running max/min of the increment path

The fixed reference therefore never contaminates the model of market dynamics -
it only decides where the distribution is read.  The law of R is modelled as

    R = sigma_hat * (mu_hat + eps)

  * sigma_hat : scale model (expected |R|), seasonality-aware term structure
  * mu_hat    : standardized drift model (efficient-price gap, flow, lead-lag)
  * eps       : empirical, fat-tailed residual law (bucketed by horizon length)

and then a gradient-boosted *calibration-from-prior* layer (LightGBM init_score =
structural logit) corrects what the structure misses.  It can only move away from
the structural prior if that improves out-of-sample loss (early stopping).

Every stage is trained on out-of-fold (cross-fitted, time-blocked, purged)
predictions of the previous stage so that no stage learns from in-sample-optimistic
inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit

from .prob import p_touch, p_double_barrier

SQ2PI = np.sqrt(2 / np.pi)

VOL_FEATS = [
    "lrv_10", "lrv_30", "lrv_60", "lrv_120", "lrv_300", "lrv_900", "lrv_3600", "lrv_14400", "lrv_86400",
    "lrvsa_60", "lrvsa_300", "lrvsa_3600", "lrvp_30", "lrvp_300", "lpk_60", "lpk_300", "lpk_3600",
    "jump_300", "jump_3600", "lcnt_10", "lcnt_60", "lcnt_300", "lcnt_3600", "lvol_10", "lvol_60",
    "lvol_300", "lvol_3600", "lcnt_ratio", "lvol_ratio", "seas_now", "maxq_10",
    "e", "how", "hod_et", "dow_et", "moh", "som", "w_in_hour",
    "abs_x_v", "wrange_v", "ncross",
]
DIR_FEATS = [
    "r_1", "r_2", "r_3", "r_5", "r_10", "r_20", "r_30", "r_60", "r_120", "r_300", "r_900", "r_3600",
    "pls_1", "pls_2", "pls_5", "pls_10", "pls_30", "pls_60", "pls_300",
    "bdev_10", "bdev_60", "bdev_300", "bdev_1800",
    "simb_1", "pimb_1", "simb_5", "pimb_5", "simb_10", "pimb_10", "simb_30", "pimb_30", "simb_60",
    "pimb_60", "simb_300", "pimb_300", "simb_900", "pimb_900", "netflow_10", "netflow_60", "netflow_300",
    "rngpos_60", "rngpos_300", "e", "x_v", "wmax_v", "wmin_v", "ncross", "fup", "tsc",
    "prev_ret_v", "prev2_ret_v", "mtf_15m", "mtf_1h", "mtf_4h", "mtf_1d", "round_1000", "round_100",
    "lrvsa_300", "lcnt_ratio", "lvol_ratio", "jump_300", "hod_et", "som",
]

LGB_BASE = dict(
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.7,
    bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=127, verbose=-1, num_threads=4,
)


def add_derived(df: pd.DataFrame, head: str) -> pd.DataFrame:
    """Head-specific derived columns (all causal)."""
    H = df[f"{head}__H"].values.astype(np.float64)
    v = df["v"].values.astype(np.float64)
    out = {}
    out["lH"] = np.log(H)
    # naive scale: per-second trailing vol * sqrt(H) * E|N(0,1)|
    out["naive_scale"] = v * np.sqrt(H) * SQ2PI
    sint = df[f"{head}__sint"].values.astype(np.float64)
    # clock projection: seasonally-adjusted trailing variance x expected seasonal variance ahead
    out["lproj_300"] = df["lrvsa_300"].values + np.log(sint) - np.log(v ** 2)
    out["lproj_3600"] = df["lrvsa_3600"].values + np.log(sint) - np.log(v ** 2)
    out["lsint"] = np.log(sint)
    out["abs_x_v"] = np.abs(df["x_v"].values)
    out["wrange_v"] = df["wmax_v"].values - df["wmin_v"].values
    # distance to reference in units of the naive horizon scale
    out["z_naive"] = df["xcur"].values / (v * np.sqrt(H))
    for k, a in out.items():
        df[k] = a.astype(np.float32) if k not in ("naive_scale",) else a
    return df


def top_gain(m, names, k=8):
    g = m.feature_importance("gain")
    g = g / max(g.sum(), 1e-12)
    o = np.argsort(-g)[:k]
    return ", ".join(f"{names[i]}={g[i]:.2f}" for i in o)


def time_folds(day: np.ndarray, k: int = 4, purge_days: int = 1):
    """Contiguous time blocks with a purge gap; yields (train_idx, pred_idx)."""
    ud = np.unique(day)
    blocks = np.array_split(ud, k)
    for b in blocks:
        lo, hi = b.min(), b.max()
        pred = (day >= lo) & (day <= hi)
        train = (day < lo - purge_days) | (day > hi + purge_days)
        yield np.flatnonzero(train), np.flatnonzero(pred)


def _fit(params, Xtr, ytr, Xva=None, yva=None, rounds=3000, init_tr=None, init_va=None,
         weight=None, es=100, fixed_rounds=None, monotone=None):
    p = dict(LGB_BASE)
    p.update(params)
    if monotone is not None:
        p["monotone_constraints"] = monotone
        p["monotone_constraints_method"] = "advanced"
    dtr = lgb.Dataset(Xtr, ytr, init_score=init_tr, weight=weight, free_raw_data=True)
    if fixed_rounds is not None:
        return lgb.train(p, dtr, num_boost_round=max(fixed_rounds, 1))
    dva = lgb.Dataset(Xva, yva, init_score=init_va, reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(es, verbose=False)])
    return m


class EmpiricalDist:
    """Bucketed empirical law of standardized residuals (quantile table)."""

    LEVELS = np.linspace(0, 1, 4001)

    def __init__(self, edges: np.ndarray):
        self.edges = edges

    def bucket(self, H):
        return np.clip(np.searchsorted(self.edges, H, side="right") - 1, 0, len(self.edges) - 2)

    def fit(self, eps, H):
        b = self.bucket(H)
        self.q = np.stack([np.quantile(eps[b == j], self.LEVELS) for j in range(len(self.edges) - 1)])
        self.sorted = [np.sort(eps[b == j]) for j in range(len(self.edges) - 1)]
        return self

    # tail-refined probability grid for the compact representation
    GRID = np.unique(np.concatenate([[0.0], np.geomspace(1e-7, 1e-3, 120), np.linspace(1e-3, 1 - 1e-3, 4001),
                                     1 - np.geomspace(1e-3, 1e-7, 120), [1.0]]))

    def compact(self):
        """Replace the full sorted residual arrays by quantiles on a tail-refined grid
        (exact to ~1e-7 in probability in the tails; ~100x smaller)."""
        if getattr(self, "sorted", None) is not None:
            self.gq = np.stack([np.quantile(s, self.GRID, method="inverted_cdf") for s in self.sorted])
            self.sorted = None
        return self

    def sf(self, z, H):
        """P(eps >= z) (inclusive, so that ties count as 'up')."""
        b = self.bucket(H)
        out = np.empty(len(z))
        for j in range(len(self.edges) - 1):
            m = b == j
            if m.any():
                if getattr(self, "sorted", None) is not None:
                    s = self.sorted[j]
                    out[m] = 1.0 - np.searchsorted(s, z[m], side="left") / len(s)
                else:
                    q = self.gq[j]
                    # F(z-) via the left-continuous inverse of the quantile grid
                    k = np.searchsorted(q, z[m], side="left")
                    out[m] = 1.0 - self.GRID[np.clip(k - 1, 0, len(self.GRID) - 1)] * (k > 0)
        return out

    def ppf(self, a, H):
        b = self.bucket(H)
        return np.array([np.interp(a, self.LEVELS, self.q[j]) for j in b]) if np.ndim(a) == 0 else None

    def ppf_many(self, alphas, H):
        b = self.bucket(H)
        out = np.empty((len(H), len(alphas)))
        for j in range(len(self.edges) - 1):
            m = b == j
            out[m] = np.interp(alphas, self.LEVELS, self.q[j])[None, :]
        return out


class HeadModel:
    """Scale + drift + residual law + calibrated event layers for one horizon head."""

    def __init__(self, head: str, extra_dir_feats: list[str] | None = None, seed: int = 0,
                 regime_feats: list[str] | None = None):
        self.head = head
        self.extra = extra_dir_feats or []
        self.regime = regime_feats or []
        self.seed = seed
        edges = np.array([1, 11, 31, 61, 121, 201, 301]) if head == "close" else np.array([0, 10 ** 6])
        # (fixed-horizon heads h10/h15/h30/h60 use a single residual-law bucket)
        self.dist = EmpiricalDist(edges)

    # ------------------------------------------------------------------ helpers
    def _vf(self):
        return VOL_FEATS + ["lH", "lproj_300", "lproj_3600", "lsint"] + getattr(self, "regime", [])

    def _df(self):
        return DIR_FEATS + ["lH", "z_naive"] + self.extra + getattr(self, "regime", [])

    def _scale_target(self, df):
        return np.clip(np.abs(df[f"{self.head}__inc"].values) / df["naive_scale"].values, 0, 25)

    # ------------------------------------------------------------------ fitting
    def fit(self, tr: pd.DataFrame, va: pd.DataFrame, log=print):
        h = self.head
        P = {"seed": self.seed}
        # ---- scale (multiplicative correction of the naive scale) ----
        ys_tr, ys_va = self._scale_target(tr), self._scale_target(va)
        m = _fit(P, tr[self._vf()], ys_tr, va[self._vf()], ys_va)
        self.scale_rounds = m.best_iteration
        self.scale_model = m
        sc_va = m.predict(va[self._vf()], num_iteration=m.best_iteration)
        sc_oof = np.empty(len(tr))
        for itr, ipr in time_folds(tr["day"].values):
            mm = _fit(P, tr[self._vf()].iloc[itr], ys_tr[itr], fixed_rounds=self.scale_rounds)
            sc_oof[ipr] = mm.predict(tr[self._vf()].iloc[ipr])
        tr["sig"] = np.maximum(sc_oof, 0.05) * tr["naive_scale"].values
        va["sig"] = np.maximum(sc_va, 0.05) * va["naive_scale"].values
        log(f"[{h}] scale: {self.scale_rounds} rounds; valid MSE ratio: const={np.mean((ys_va - ys_tr.mean()) ** 2):.4f} "
            f"model={np.mean((ys_va - sc_va) ** 2):.4f}")
        log(f"[{h}] scale top features (gain share): {top_gain(m, self._vf())}")

        # ---- zero-move atom: pi0 = P(R == 0 | state) ----
        # ties resolve "up", and at short horizons the price frequently does not move at
        # all (48% with 1s left, 7.5% over 30s in 2025), so the atom is modelled explicitly.
        for d in (tr, va):
            d["lsig_rel"] = np.log(d["sig"].values / d["naive_scale"].values).astype(np.float32)
        zf = self._zf()
        z_tr = (tr[f"{h}__inc"].values == 0).astype(int)
        z_va = (va[f"{h}__inc"].values == 0).astype(int)
        m = _fit(dict(P, objective="binary", num_leaves=31, min_data_in_leaf=5000), tr[zf], z_tr, va[zf], z_va)
        self.zero_rounds = max(m.best_iteration, 1)
        self.zero_model = m
        pz_va = m.predict(va[zf], num_iteration=self.zero_rounds)
        pz_oof = np.empty(len(tr))
        for itr, ipr in time_folds(tr["day"].values):
            mm = _fit(dict(P, objective="binary", num_leaves=31, min_data_in_leaf=5000),
                      tr[zf].iloc[itr], z_tr[itr], fixed_rounds=self.zero_rounds)
            pz_oof[ipr] = mm.predict(tr[zf].iloc[ipr])
        tr["pi0"], va["pi0"] = np.clip(pz_oof, 0, 0.95), np.clip(pz_va, 0, 0.95)
        log(f"[{h}] zero-move atom: {self.zero_rounds} rounds; valid rate={z_va.mean():.4f}, "
            f"logloss const={-np.mean(z_va * np.log(z_tr.mean()) + (1 - z_va) * np.log(1 - z_tr.mean())):.4f} "
            f"model={-np.mean(z_va * np.log(np.clip(pz_va, 1e-6, 1)) + (1 - z_va) * np.log(np.clip(1 - pz_va, 1e-6, 1))):.4f}")

        # ---- drift (standardized) ----
        u_tr = np.clip(tr[f"{h}__inc"].values / tr["sig"].values, -8, 8)
        u_va = np.clip(va[f"{h}__inc"].values / va["sig"].values, -8, 8)
        dfeat = self._df() + ["lsig_rel"]
        m = _fit(dict(P, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5000), tr[dfeat], u_tr,
                 va[dfeat], u_va)
        self.drift_rounds = max(m.best_iteration, 1)
        self.drift_model = m
        mu_va = m.predict(va[dfeat], num_iteration=self.drift_rounds)
        mu_oof = np.empty(len(tr))
        for itr, ipr in time_folds(tr["day"].values):
            mm = _fit(dict(P, learning_rate=0.05, num_leaves=31, min_data_in_leaf=5000),
                      tr[dfeat].iloc[itr], u_tr[itr], fixed_rounds=self.drift_rounds)
            mu_oof[ipr] = mm.predict(tr[dfeat].iloc[ipr])
        tr["mu"], va["mu"] = mu_oof, mu_va
        r2 = 1 - np.mean((u_va - mu_va) ** 2) / np.mean((u_va - u_va.mean()) ** 2)
        log(f"[{h}] drift: {self.drift_rounds} rounds; valid R2(u)={r2 * 100:.3f}%  corr={np.corrcoef(mu_va, u_va)[0, 1]:.4f}")
        log(f"[{h}] drift top features (gain share): {top_gain(m, dfeat)}")
        self.dfeat = dfeat

        # ---- residual law from OOF residuals ----
        # identity: E[R / sig_c | R != 0] = E[R] / sig = mu  (with sig_c = sig / (1 - pi0))
        nz = tr[f"{h}__inc"].values != 0
        sig_c = tr["sig"].values / (1 - tr["pi0"].values)
        eps = np.clip(tr[f"{h}__inc"].values / sig_c - tr["mu"].values, -50, 50)
        self.dist.fit(eps[nz], tr[f"{h}__H"].values[nz])

        # ---- P(up) calibration from structural prior ----
        for d in (tr, va):
            d["p_struct"] = np.clip(self.struct_up(d), 1e-5, 1 - 1e-5)
            d["z_struct"] = self._z(d).astype(np.float32)
        cf = self.calib_feats()
        y_tr, y_va = (tr[f"{h}__y"].values >= 0).astype(int), (va[f"{h}__y"].values >= 0).astype(int)
        m = _fit(dict(P, objective="binary", learning_rate=0.04, num_leaves=15, min_data_in_leaf=20000,
                      lambda_l2=50.0), tr[cf], y_tr, va[cf], y_va,
                 init_tr=logit(tr["p_struct"].values), init_va=logit(va["p_struct"].values), es=200)
        self.calib_up = m
        self.calib_rounds = m.best_iteration
        log(f"[{h}] up-calibration: {self.calib_rounds} rounds; top features: {top_gain(m, cf)}")
        return self

    def calib_feats(self):
        base = ["z_struct", "mu", "lsig_rel", "e", "lH", "lrvsa_300", "lcnt_ratio", "jump_300",
                "som", "hod_et", "x_v", "ncross", "pi0"]
        reg = [f for f in ("reg_rv_1d_7d", "reg_catchup_1h", "reg_vr300_1d", "reg_volvol_1d")
               if f in getattr(self, "regime", [])]
        return base + reg

    # ------------------------------------------------------------------ predicting
    def prepare(self, d: pd.DataFrame) -> pd.DataFrame:
        sc = self.scale_model.predict(d[self._vf()], num_iteration=self.scale_rounds)
        d["sig"] = np.maximum(sc, 0.05) * d["naive_scale"].values
        d["lsig_rel"] = np.log(d["sig"].values / d["naive_scale"].values).astype(np.float32)
        d["pi0"] = np.clip(self.zero_model.predict(d[self._zf()], num_iteration=self.zero_rounds), 0, 0.95)
        d["mu"] = self.drift_model.predict(d[self.dfeat], num_iteration=self.drift_rounds)
        d["p_struct"] = np.clip(self.struct_up(d), 1e-5, 1 - 1e-5)
        d["z_struct"] = self._z(d).astype(np.float32)
        return d

    def _zf(self):
        return self._vf() + ["lsig_rel"]

    @staticmethod
    def _sig_c(d):
        return d["sig"].values / (1 - d["pi0"].values)

    def _z(self, d, level=0.0, use_drift=True):
        return (level - d["xcur"].values) / self._sig_c(d) - (d["mu"].values if use_drift else 0.0)

    def struct_up(self, d, use_drift=True):
        """P(S_T >= P0) = pi0 * 1{x >= 0} + (1 - pi0) * P_c(R >= -x)."""
        pc = self.dist.sf(self._z(d, 0.0, use_drift), d[f"{self.head}__H"].values)
        pi0 = d["pi0"].values
        return pi0 * (d["xcur"].values >= 0) + (1 - pi0) * pc

    def struct_terminal_sf(self, d, level_log):
        """P(log(S_T/P0) >= level) from the structural law."""
        pc = self.dist.sf(np.asarray(self._z(d, level_log), float), d[f"{self.head}__H"].values)
        pi0 = d["pi0"].values
        return pi0 * (d["xcur"].values >= level_log) + (1 - pi0) * pc

    def p_up(self, d):
        if self.calib_rounds <= 0:
            return d["p_struct"].values
        return expit(logit(d["p_struct"].values) +
                     self.calib_up.predict(d[self.calib_feats()], num_iteration=self.calib_rounds,
                                           raw_score=True))

    def quantiles(self, d, alphas):
        """Quantiles of log(S_target/P0) under the atom + continuous mixture.

        F(r) = pi0 * 1{r >= 0} + (1 - pi0) * F_c(r),  F_c the continuous law of R."""
        alphas = np.asarray(alphas, float)
        H = d[f"{self.head}__H"].values
        pi0 = d["pi0"].values[:, None]
        sc, mu, x = self._sig_c(d)[:, None], d["mu"].values[:, None], d["xcur"].values[:, None]
        b = self.dist.bucket(H)
        # F_c(0) for every row: P(eps <= -mu)
        fc0 = 1 - self.dist.sf(-d["mu"].values, H)[:, None]
        lo = (1 - pi0) * fc0                  # mass strictly below 0
        a = alphas[None, :]
        lev = np.where(a < lo, a / (1 - pi0), np.where(a > lo + pi0, (a - pi0) / (1 - pi0), np.nan))
        lev = np.clip(lev, 0, 1)
        q = np.zeros_like(lev)
        for j in range(len(self.dist.edges) - 1):
            m = b == j
            if m.any():
                q[m] = np.interp(lev[m], self.dist.LEVELS, self.dist.q[j])
        in_atom = (a >= lo) & (a <= lo + pi0)     # quantile falls on "no move"
        r = sc * (mu + q)
        r = np.where(a < lo, np.minimum(r, 0.0), np.maximum(r, 0.0))   # side of the atom
        r = np.where(in_atom, 0.0, r)
        r = np.maximum.accumulate(r, axis=1)      # interpolation noise must not break order
        return x + r

    # ---- BM-equivalent parameters for path events ----
    def bm_params(self, d):
        H = d[f"{self.head}__H"].values.astype(float)
        s = d["sig"].values / SQ2PI / np.sqrt(H)          # per-second diffusion
        mu = d["mu"].values * d["sig"].values / H          # per-second drift
        return mu, s, H
