"""Robustness analysis of saved predictions: consistency over time and where the edge lives.

    python scripts/robustness.py valid|test
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.prob import logloss, brier, ece, auc, block_bootstrap_diff  # noqa: E402
from btcmodel.config import MODEL_DIR, ART_DIR  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "valid"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
T0_DATE = np.datetime64("2025-01-01")


def main():
    out = []
    for head in ["h10", "h15", "h30", "h60", "close"]:
        f = os.path.join(MODEL_DIR, f"preds_{head}_{SPLIT}{TAG}.pkl")
        if not os.path.exists(f):
            continue
        P = pickle.load(open(f, "rb"))
        y, e, day = P["y"], P["e"], P["day"]
        p0, pd_ = P["p"]["B0_naive"], P["p"]["D_directGBM"]
        pm = P["p"].get("E_blend", P["p"]["M_final"])   # final forecast
        l0, lm, ld = logloss(y, p0), logloss(y, pm), logloss(y, pd_)
        out.append(f"\n##### {head} ({SPLIT}{TAG}) #####")
        # 1. month by month
        month = (T0_DATE + day.astype("timedelta64[D]")).astype("datetime64[M]")
        rows = []
        for m in np.unique(month):
            k = month == m
            rows.append({"month": str(m), "n": int(k.sum()), "LL_naive": l0[k].mean(), "LL_model": lm[k].mean(),
                         "skill_%": 100 * (1 - lm[k].mean() / l0[k].mean()),
                         "skill_direct_%": 100 * (1 - ld[k].mean() / l0[k].mean()),
                         "ECE_model": ece(y[k], pm[k])})
        t = pd.DataFrame(rows)
        out.append("month-by-month log-loss skill vs naive Gaussian:\n" + t.to_string(index=False, float_format="%.4f"))
        out.append(f"months with positive skill: {(t['skill_%'] > 0).sum()}/{len(t)}")
        # 2. uncertain zone: where the naive forecaster is between 0.2 and 0.8
        z = (p0 > 0.2) & (p0 < 0.8)
        m, lo, hi, _ = block_bootstrap_diff(lm[z], l0[z], day[z])
        out.append(f"uncertain zone (0.2<p_naive<0.8, {z.mean() * 100:.1f}% of rows): LL naive={l0[z].mean():.4f} "
                   f"model={lm[z].mean():.4f} skill={100 * (1 - lm[z].mean() / l0[z].mean()):.2f}%  "
                   f"dLL={m:.5f} CI[{lo:.5f},{hi:.5f}]")
        # 3. e = 0: pure 'which way' forecast (x = 0)
        k = e == 0
        if k.any():
            out.append(f"e=0 (start of window): AUC model={auc(y[k], pm[k]):.4f} direct={auc(y[k], pd_[k]):.4f}; "
                       f"LL naive={l0[k].mean():.5f} model={lm[k].mean():.5f} "
                       f"(skill {100 * (1 - lm[k].mean() / l0[k].mean()):.2f}%); Brier model={brier(y[k], pm[k]).mean():.5f}")
            m, lo, hi, _ = block_bootstrap_diff(lm[k], l0[k], day[k])
            out.append(f"    e=0 dLL model-naive={m:.5f} CI[{lo:.5f},{hi:.5f}]")
        # 4. tails: confident predictions
        for a, b in [(0.0, 0.02), (0.02, 0.1), (0.9, 0.98), (0.98, 1.0)]:
            k = (pm >= a) & (pm < b)
            if k.sum() > 100:
                out.append(f"model p in [{a},{b}): n={k.sum()}, mean p={pm[k].mean():.4f}, observed={y[k].mean():.4f}")
        # 5. directional (increment) skill
        inc, mu = P["inc"], P["mu"]
        nz = inc != 0
        out.append(f"directional AUC (sign of move from now): {auc((inc[nz] > 0).astype(int), mu[nz]):.4f}")
    s = "\n".join(out)
    print(s)
    with open(os.path.join(ART_DIR, f"robustness_{SPLIT}{TAG}.txt"), "w") as fh:
        fh.write(s)


if __name__ == "__main__":
    main()
