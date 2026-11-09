"""Regime robustness of the final forecasts.

    python scripts/regime_eval.py test [tag]

1. Stratified skill: by volatility regime, trend regime, weekday/weekend, US session,
   and on the most extreme days - does the edge survive every kind of market?
2. Online recalibration (causal): every day d, a 2-parameter logistic map
   logit(p') = a + b*logit(p) (per elapsed-time bucket) is refit on the previous 14 days of
   *already resolved* windows and applied to day d.  If the market drifts away from the
   training regime this repairs calibration; if it doesn't, it should change nothing.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.special import logit, expit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.prob import logloss, block_bootstrap_diff, ece  # noqa: E402
from btcmodel.dataset import CACHE  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
from btcmodel.config import MODEL_DIR as MD, ART_DIR  # noqa: E402
T0_DATE = np.datetime64("2025-01-01")


def fit_platt(z, y, iters=25):
    """Logistic regression y ~ a + b z (Newton)."""
    w = np.array([0.0, 1.0])
    X = np.c_[np.ones_like(z), z]
    for _ in range(iters):
        p = expit(X @ w)
        g = X.T @ (y - p)
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(Hm, g)
    return w


def online_recal(z_hist, y_hist, d_hist, eb_hist, z, d, eb, window=14):
    out = np.empty_like(z)
    for day in np.unique(d):
        m = d == day
        h = (d_hist >= day - window) & (d_hist < day)          # strictly earlier days
        for b in np.unique(eb[m]):
            mb, hb = m & (eb == b), h & (eb_hist == b)
            if hb.sum() < 5000:
                out[mb] = z[mb]
                continue
            a, s = fit_platt(z_hist[hb], y_hist[hb])
            out[mb] = a + s * z[mb]
    return out


def main():
    lines = []
    reg = pd.read_parquet(os.path.join(CACHE, f"{SPLIT}.parquet"), columns=["T0", "e", "lrv_86400", "dow_et", "hod_et"])
    reg["e"] = reg["e"].astype(np.int64)
    try:
        r2 = pd.read_parquet(os.path.join(CACHE, f"{SPLIT}_regime.parquet"), columns=["reg_trend_1d", "reg_rv_1d_7d"])
        reg = pd.concat([reg, r2], axis=1)
    except Exception:
        pass
    trq = pd.read_parquet(os.path.join(CACHE, "train.parquet"), columns=["lrv_86400"])["lrv_86400"].quantile([1 / 3, 2 / 3]).values
    for head in ["h10", "h15", "h30", "h60", "close"]:
        f = os.path.join(MD, f"preds_{head}_{SPLIT}{TAG}.pkl")
        if not os.path.exists(f):
            continue
        P = pickle.load(open(f, "rb"))
        y, day, e = P["y"], P["day"], P["e"]
        pf = P["p"].get("E_blend", P["p"]["M_final"])
        p0 = P["p"]["B0_naive"]
        k = pd.DataFrame({"T0": P["T0"], "e": e.astype(np.int64)}).merge(reg, on=["T0", "e"], how="left")
        l0, lf = logloss(y, p0), logloss(y, pf)
        lines.append(f"\n##### {head} ({SPLIT}{TAG}) #####  overall skill {100 * (1 - lf.mean() / l0.mean()):.2f}%")
        strata = {
            "vol: low (train tercile)": k["lrv_86400"].values < trq[0],
            "vol: mid": (k["lrv_86400"].values >= trq[0]) & (k["lrv_86400"].values < trq[1]),
            "vol: high": k["lrv_86400"].values >= trq[1],
            "weekday": k["dow_et"].values < 5,
            "weekend": k["dow_et"].values >= 5,
            "US session (9-16 ET, weekdays)": (k["dow_et"].values < 5) & (k["hod_et"].values >= 9) & (k["hod_et"].values < 16),
        }
        if "reg_trend_1d" in k:
            a = np.abs(k["reg_trend_1d"].values)
            q = np.nanquantile(a, [1 / 3, 2 / 3])
            strata["trend 1d: weak"] = a < q[0]
            strata["trend 1d: strong"] = a >= q[1]
            v = k["reg_rv_1d_7d"].values
            strata["vol rising (1d >> 7d)"] = v > np.nanquantile(v, 0.9)
        # most extreme days by realized vol
        dv = pd.Series(k["lrv_86400"].values).groupby(day).mean()
        hot = dv.index[dv.values >= np.quantile(dv.values, 0.95)]
        strata["top-5% most volatile days"] = np.isin(day, hot)
        rows = []
        for nm, m in strata.items():
            m = np.asarray(m, bool)
            if m.sum() < 1000:
                continue
            dm, lo, hi, _ = block_bootstrap_diff(lf[m], l0[m], day[m], n_boot=500)
            rows.append({"stratum": nm, "share_%": 100 * m.mean(), "LL_naive": l0[m].mean(), "LL_model": lf[m].mean(),
                         "skill_%": 100 * (1 - lf[m].mean() / l0[m].mean()), "dLL_CI_hi": hi, "ECE": ece(y[m], pf[m])})
        lines.append(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))

        # online recalibration with burn-in from the validation tail
        fv = os.path.join(MD, f"preds_{head}_valid{TAG}.pkl")
        if SPLIT != "valid" and os.path.exists(fv):
            V = pickle.load(open(fv, "rb"))
            pv = V["p"].get("E_blend", V["p"]["M_final"])
            zb = lambda ee: np.digitize(ee, [1, 61, 181, 241, 271])
            zh = np.r_[logit(np.clip(pv, 1e-5, 1 - 1e-5)), logit(np.clip(pf, 1e-5, 1 - 1e-5))]
            yh = np.r_[V["y"], y]
            dh = np.r_[V["day"], day]
            ebh = np.r_[zb(V["e"]), zb(e)]
            zr = online_recal(zh, yh, dh, ebh, logit(np.clip(pf, 1e-5, 1 - 1e-5)), day, zb(e))
            pr = expit(zr)
            lr = logloss(y, pr)
            dm, lo, hi, _ = block_bootstrap_diff(lr, lf, day)
            lines.append(f"online recalibration (14-day causal window, per e-bucket): LL frozen={lf.mean():.5f} "
                         f"recal={lr.mean():.5f}  dLL={dm:.5f} CI[{lo:.5f},{hi:.5f}]  ECE frozen={ece(y, pf):.4f} recal={ece(y, pr):.4f}")
            month = (T0_DATE + day.astype("timedelta64[D]")).astype("datetime64[M]")
            mm = [f"{str(mo)}: {100 * (1 - lr[month == mo].mean() / lf[month == mo].mean()):+.2f}%" for mo in np.unique(month)]
            lines.append("   recal gain by month: " + ", ".join(mm))
    s = "\n".join(lines)
    print(s)
    open(os.path.join(ART_DIR, f"regime_{SPLIT}{TAG}.txt"), "w").write(s)


if __name__ == "__main__":
    main()
