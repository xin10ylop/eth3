"""Online regime-state features (all causal: bars <= c = T0 + e - 1 only).

A model fitted on one period is only as good as its ability to recognise that the
market has changed.  Volatility *level* is already handled by standardization; these
features describe the *state of the market's mechanics*, estimated online from the
recent past, so the models can condition on them instead of averaging over them:

  reg_ll_beta_{1h,6h}   current perp->spot lead-lag beta  (sum rs_t rp_{t-1} / sum rp_{t-1}^2)
  reg_catchup_{1h,6h}   realized catch-up of spot to the basis gap: regression slope of the
                        next-5s spot move on the basis deviation, using pairs fully realized
                        before the decision (how strong the main drift signal is *right now*)
  reg_ac1_1h            lag-1 autocorrelation of 1s spot returns (partial-adjustment speed)
  reg_vr60_1h, reg_vr300_1d  variance ratios: >1 trending, <1 mean-reverting microstructure
  reg_lrv_7d, reg_rv_1d_7d   7-day vol level and 1d/7d vol ratio (vol regime shift)
  reg_volvol_1d          dispersion of hourly log-RV over the last 24 completed hours
  reg_trend_{4h,1d}      vol-standardized trend
  reg_basis_{1h,7d}      perp premium level (bps): positioning / funding regime
  reg_lvolume_1d_7d      activity regime
  reg_cp{10,3}_{1d,7d}   close pressure: average move toward P0 in the final 10s/3s of recently
                         completed windows (in sigma units) - lets the model scale the
                         end-of-window "pull to the reference" by its current intensity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Roll, ffill_nan

DAY = 86400


def _lagged(a, k):
    out = np.empty_like(a)
    out[:k] = 0.0
    out[k:] = a[:-k]
    return out


def regime_features(tl, T0_abs: np.ndarray, e: np.ndarray, chunk_days: int = 31,
                    warm_days: int = 8) -> pd.DataFrame:
    """Compute regime features for decision points given in absolute timeline indices."""
    order = np.argsort(T0_abs, kind="stable")
    T0s, es = np.asarray(T0_abs, np.int64)[order], np.asarray(e, np.int64)[order]
    out = {}
    day = T0s // DAY
    d0, d1 = day.min(), day.max()
    parts = []
    for c0 in range(d0, d1 + 1, chunk_days):
        sel = (day >= c0) & (day < c0 + chunk_days)
        if not sel.any():
            continue
        a = max(0, (c0 - warm_days) * DAY)
        b = min(tl.n, (c0 + chunk_days) * DAY + 600)
        ls = np.log(np.asarray(tl["s_close"][a:b]))
        lp = np.log(ffill_nan(np.asarray(tl["p_close"][a:b]).copy()))
        bad = ~np.isfinite(lp)
        lp[bad] = ls[bad]
        rs = np.diff(ls, prepend=ls[0])
        rp = np.diff(lp, prepend=lp[0])
        basis = lp - ls
        vol = (np.asarray(tl["s_volume"][a:b], dtype=np.float64) + np.asarray(tl["p_buy"][a:b], dtype=np.float64)
               + np.asarray(tl["p_sell"][a:b], dtype=np.float64))
        n = len(ls)
        c = (T0s[sel] + es[sel] - 1) - a          # local index of last known bar
        avail = c + 1                              # number of bars available before decision
        F = {}
        R2 = Roll(rs ** 2)
        # lead-lag beta
        Rx = Roll(rs * _lagged(rp, 1))
        Rpp = Roll(_lagged(rp, 1) ** 2)
        for L, nm in [(3600, "1h"), (21600, "6h")]:
            F[f"reg_ll_beta_{nm}"] = Rx.sum(c, L) / (Rpp.sum(c, L) + 1e-18)
        # catch-up coefficient: pairs (t, t+5) with t + 5 <= c
        bdv = basis - Roll(basis).sum(np.arange(n), 300) / np.minimum(np.arange(n) + 1, 300)
        fwd = np.zeros(n)
        fwd[:-5] = ls[5:] - ls[:-5]
        q = _lagged(bdv * fwd, 5)                  # q_s = bdv_{s-5} * (ls_s - ls_{s-5}): known at s
        qq = _lagged(bdv * bdv, 5)
        Rq, Rqq = Roll(q), Roll(qq)
        for L, nm in [(3600, "1h"), (21600, "6h")]:
            F[f"reg_catchup_{nm}"] = Rq.sum(c, L) / (Rqq.sum(c, L) + 1e-18)
        # autocorrelation
        Rac = Roll(rs * _lagged(rs, 1))
        F["reg_ac1_1h"] = Rac.sum(c, 3600) / (R2.sum(c, 3600) + 1e-18)
        # variance ratios over aligned blocks (block-end bars only)
        for P, L, nm in [(60, 3600, "vr60_1h"), (300, DAY, "vr300_1d")]:
            idx = np.arange(n)
            w = np.zeros(n)
            ends = idx[((idx + a + 1) % P == 0) & (idx >= P)]
            w[ends] = (ls[ends] - ls[ends - P]) ** 2
            F[f"reg_{nm}"] = Roll(w).sum(c, L) / (R2.sum(c, L) + 1e-18)
        rv1d = R2.sum(c, DAY) / np.minimum(avail, DAY)
        rv7d = R2.sum(c, 7 * DAY) / np.minimum(avail, 7 * DAY)
        F["reg_lrv_7d"] = np.log(rv7d + 1e-14)
        F["reg_rv_1d_7d"] = np.log(rv1d + 1e-14) - np.log(rv7d + 1e-14)
        # vol-of-vol: hourly log RV over the last 24 completed hours
        nh = n // 3600
        hrv = np.log(R2.cs[(np.arange(1, nh + 1)) * 3600] - R2.cs[np.arange(nh) * 3600] + 1e-14)
        hc = (c + 1) // 3600                       # completed local hours
        hk = hc[:, None] - np.arange(1, 25)[None, :]
        okh = hk >= 0
        hv = np.where(okh, hrv[np.clip(hk, 0, nh - 1)], np.nan)
        F["reg_volvol_1d"] = np.nanstd(hv, axis=1)
        for L, nm in [(14400, "4h"), (DAY, "1d")]:
            F[f"reg_trend_{nm}"] = (ls[c] - ls[np.maximum(c - L, 0)]) / np.sqrt(R2.sum(c, L) + 1e-14)
        Rb = Roll(basis)
        F["reg_basis_1h"] = Rb.sum(c, 3600) / np.minimum(avail, 3600) * 1e4
        F["reg_basis_7d"] = Rb.sum(c, 7 * DAY) / np.minimum(avail, 7 * DAY) * 1e4
        Rv = Roll(vol)
        F["reg_lvolume_1d_7d"] = (np.log(Rv.sum(c, DAY) / np.minimum(avail, DAY) + 1e-9)
                                  - np.log(Rv.sum(c, 7 * DAY) / np.minimum(avail, 7 * DAY) + 1e-9))
        # close pressure: realized move TOWARD the window's own reference in the final k
        # seconds of COMPLETED windows (a regime that appeared in Feb 2026; see
        # research/eda5_close_pressure.py).  m_j = -sign(x) * R / (sigma*sqrt(k)) for windows
        # whose price was within 1.5 sigma_k of P0 when the final k seconds started.
        lo = np.log(np.asarray(tl["s_open"][a:b]))
        nW = n // 300
        Tj = np.arange(nW) * 300
        ok = Tj >= 3600
        volj = np.sqrt(np.maximum(R2.cs[Tj] - R2.cs[np.maximum(Tj - 3600, 0)], 0) / 3600)
        # fixed floor (a data-dependent floor over the chunk would peek at later windows)
        volj = np.maximum(volj, 1.5e-5)
        T0l = (T0s[sel] - a)
        J = T0l // 300                              # current window index (local); use j < J
        for k in (3, 10):
            ia, ib = Tj + 300 - k - 1, Tj + 299
            x = ls[ia] - lo[Tj]
            R = ls[ib] - ls[ia]
            sk = volj * np.sqrt(k)
            near = ok & (np.abs(x) > 0) & (np.abs(x) < 1.5 * sk)
            m = np.where(near, np.clip(-np.sign(x) * R / sk, -10, 10), 0.0)
            cm = np.concatenate([[0.0], np.cumsum(m)])
            cn = np.concatenate([[0.0], np.cumsum(near.astype(float))])
            for NW, nm in [(288, "1d"), (2016, "7d")]:
                if k == 3 and NW == 288:
                    continue
                lo_j = np.maximum(J - NW, 0)
                num, den = cm[J] - cm[lo_j], cn[J] - cn[lo_j]
                F[f"reg_cp{k}_{nm}"] = np.where(den >= 30, num / np.maximum(den, 1), np.nan)
        # insufficient history -> NaN (LightGBM routes missing values explicitly)
        short7 = avail < 7 * DAY
        for k in ["reg_lrv_7d", "reg_rv_1d_7d", "reg_basis_7d", "reg_lvolume_1d_7d"]:
            F[k] = np.where(short7, np.nan, F[k])
        df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})
        df["_pos"] = np.flatnonzero(sel)
        parts.append(df)
    res = pd.concat(parts, ignore_index=True).sort_values("_pos")
    res.index = order[res["_pos"].values]
    return res.drop(columns="_pos").sort_index()


REGIME_FEATS = ["reg_ll_beta_1h", "reg_ll_beta_6h", "reg_catchup_1h", "reg_catchup_6h", "reg_ac1_1h",
                "reg_vr60_1h", "reg_vr300_1d", "reg_lrv_7d", "reg_rv_1d_7d", "reg_volvol_1d",
                "reg_trend_4h", "reg_trend_1d", "reg_basis_1h", "reg_basis_7d", "reg_lvolume_1d_7d",
                "reg_cp10_1d", "reg_cp10_7d", "reg_cp3_7d"]
