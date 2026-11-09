"""Evaluate FROZEN models (saved by `train_eval.py valid <head>`) on another split.

    python scripts/evaluate_saved.py test h30
    python scripts/evaluate_saved.py test h60
    python scripts/evaluate_saved.py test close

Nothing is refit: scale, drift, residual law, calibrators, touch/first-passage
correctors and the direct GBM are exactly the ones selected on the validation set.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr, logit, expit
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_eval import load_head, summarize, STACK_DIR, MODEL_DIR, OUT, HEAD_ORDER, TAG  # noqa: E402
from btcmodel.models import VOL_FEATS, DIR_FEATS, SQ2PI  # noqa: E402
from btcmodel.events import touch_rows, naive_touch, fp_rows  # noqa: E402
from btcmodel.prob import logloss, brier, ece, auc, block_bootstrap_diff, mlogloss  # noqa: E402

SPLIT = sys.argv[1]
HEAD = sys.argv[2]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def main():
    t0 = time.time()
    M = pickle.load(open(os.path.join(MODEL_DIR, f"models_{HEAD}_valid{TAG}.pkl"), "rb"))
    hm, tm, fm, dm = M["head"], M["touch"], M["fp"], M["direct"]
    ev = load_head(SPLIT, HEAD)
    log(f"=========== HEAD {HEAD} on {SPLIT} (frozen models) ===========  rows={len(ev)} windows={ev['T0'].nunique()}")
    for sh in hm.extra:
        short = sh.split("_")[1]
        s = pd.read_parquet(os.path.join(STACK_DIR, f"{short}_{SPLIT}_{SPLIT}{TAG}.parquet"))
        ev[sh] = pd.merge(ev[["T0", "e"]], s, on=["T0", "e"], how="left")["drift_px"].values
        ev[sh] = (ev[sh] / ev["naive_scale"]).astype(np.float32)
    hm.prepare(ev)
    pd.DataFrame({"T0": ev["T0"].values, "e": ev["e"].values,
                  "drift_px": ev["mu"].values * ev["sig"].values}).to_parquet(
        os.path.join(STACK_DIR, f"{HEAD}_{SPLIT}_{SPLIT}{TAG}.parquet"))

    y = (ev[f"{HEAD}__y"].values >= 0).astype(int)
    H = ev[f"{HEAD}__H"].values.astype(float)
    x = ev["xcur"].values
    g = ev["day"].values
    p = {}
    p["B0_naive"] = np.clip(ndtr(x / (ev["v"].values * np.sqrt(H))), 1e-5, 1 - 1e-5)
    p["B1_volmodel"] = np.clip(ndtr(x / (ev["sig"].values / SQ2PI)), 1e-5, 1 - 1e-5)
    p["B2_fattail"] = np.clip(hm.struct_up(ev, use_drift=False), 1e-5, 1 - 1e-5)
    p["B3_struct"] = ev["p_struct"].values
    p["M_final"] = np.clip(hm.p_up(ev), 1e-5, 1 - 1e-5)
    feats = sorted(set(VOL_FEATS + DIR_FEATS + ["lH", "z_naive", "lproj_300", "lsint"] + list(getattr(hm, "regime", []))))
    p["D_directGBM"] = np.clip(dm.predict(ev[feats], num_iteration=dm.best_iteration), 1e-5, 1 - 1e-5)
    p["E_blend"] = expit(0.5 * logit(p["M_final"]) + 0.5 * logit(p["D_directGBM"]))
    res = [summarize(k, y, pk, g, ref=p["B0_naive"] if k != "B0_naive" else None) for k, pk in p.items()]
    log(f"\n[{HEAD}] P(S_target >= P0) on {SPLIT}: n={len(y)}, base rate={y.mean():.4f}")
    log(pd.DataFrame(res).to_string(index=False))
    m_, lo, hi, _ = block_bootstrap_diff(logloss(y, p["E_blend"]), logloss(y, p["D_directGBM"]), g)
    log(f"[{HEAD}] blend vs direct GBM: dLL={m_:.5f} CI[{lo:.5f},{hi:.5f}]")
    m_, lo, hi, _ = block_bootstrap_diff(logloss(y, p["E_blend"]), logloss(y, p["M_final"]), g)
    log(f"[{HEAD}] blend vs structured: dLL={m_:.5f} CI[{lo:.5f},{hi:.5f}]")
    inc = ev[f"{HEAD}__inc"].values
    nz = inc != 0
    u = np.clip(inc / ev["sig"].values, -8, 8)
    r2 = 1 - np.mean((u - ev["mu"].values) ** 2) / np.mean((u - u.mean()) ** 2)
    log(f"[{HEAD}] drift: R2(u)={100 * r2:.3f}%  corr={np.corrcoef(ev['mu'].values, u)[0, 1]:.4f}  "
        f"directional AUC={auc((inc[nz] > 0).astype(int), ev['mu'].values[nz]):.4f}")
    # magnitude: predicted E|R| vs realized |R|
    absr = np.abs(inc)
    sg, nv = ev["sig"].values, ev["naive_scale"].values
    from scipy.stats import spearmanr
    mse_m, mse_n = np.mean((absr - sg) ** 2), np.mean((absr - nv) ** 2)
    log(f"[{HEAD}] magnitude E|R|: MSE skill vs naive scale={100 * (1 - mse_m / mse_n):.2f}%  "
        f"Spearman(pred, |R|) model={spearmanr(sg, absr)[0]:.4f} naive={spearmanr(nv, absr)[0]:.4f}  "
        f"mean |R| bps={absr.mean() * 1e4:.3f} pred={sg.mean() * 1e4:.3f}")
    eb = pd.cut(ev["e"].values, [-1, 0, 30, 60, 120, 180, 240, 270, 299])
    rows = []
    for b in eb.categories:
        mk = (eb == b)
        if mk.sum() == 0:
            continue
        rr = {"e_bucket": str(b), "n": int(mk.sum())}
        for k in ["B0_naive", "B1_volmodel", "B3_struct", "M_final", "D_directGBM", "E_blend"]:
            rr[k] = float(logloss(y[mk], p[k][mk]).mean())
        rr["skill_E_vs_B0_%"] = 100 * (1 - rr["E_blend"] / rr["B0_naive"])
        rows.append(rr)
    log(pd.DataFrame(rows).to_string(index=False))
    out = {"up_table": res, "by_e": rows}

    alphas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    Q = hm.quantiles(ev, alphas)
    yy = ev[f"{HEAD}__y"].values
    cover = [(yy <= Q[:, j]).mean() for j in range(len(alphas))]
    Qn = x[:, None] + ev["v"].values[:, None] * np.sqrt(H)[:, None] * norm.ppf(alphas)[None, :]
    pin = lambda Qm: np.mean([np.mean(np.maximum(a * (yy - Qm[:, j]), (a - 1) * (yy - Qm[:, j])))
                              for j, a in enumerate(alphas)])
    log(f"[{HEAD}] quantile coverage (target {alphas.tolist()}): {np.round(cover, 3).tolist()}")
    log(f"[{HEAD}] pinball (bps): model={pin(Q) * 1e4:.4f} naive={pin(Qn) * 1e4:.4f} skill={100 * (1 - pin(Q) / pin(Qn)):.2f}%")
    out["quantiles"] = {"coverage": cover, "pinball_model": pin(Q), "pinball_naive": pin(Qn)}

    rows_t = touch_rows(ev, HEAD, hm)
    pt, pn = tm.predict(rows_t), naive_touch(ev, HEAD)
    yt = rows_t["y"].values
    tr_ = []
    for k in sorted(rows_t["k"].unique()):
        mk = rows_t["k"].values == k
        tr_.append({"level_bps": int(k), "n": int(mk.sum()), "rate": float(yt[mk].mean()),
                    "LL_naiveBM": float(logloss(yt[mk], pn[mk]).mean()),
                    "LL_structBM": float(logloss(yt[mk], rows_t["p_struct"].values[mk]).mean()),
                    "LL_final": float(logloss(yt[mk], pt[mk]).mean()), "ECE_final": float(ece(yt[mk], pt[mk]))})
    tt = pd.DataFrame(tr_)
    tt["skill_vs_naive_%"] = 100 * (1 - tt["LL_final"] / tt["LL_naiveBM"])
    log(f"\n[{HEAD}] TOUCH within horizon (non-trivial levels):")
    log(tt.to_string(index=False))
    m_, lo, hi, _ = block_bootstrap_diff(logloss(yt, pt), logloss(yt, pn), rows_t["day"].values)
    log(f"[{HEAD}] touch dLL final-naive={m_:.5f} CI[{lo:.5f},{hi:.5f}]")
    out["touch"] = tr_
    del rows_t

    fev, fnv = fp_rows(ev, HEAD, hm), fp_rows(ev, HEAD, hm, naive=True)
    Pf = fm.predict(fev)
    fr = []
    for k in sorted(fev["k"].unique()):
        mk, mn = fev["k"].values == k, fnv["k"].values == k
        yk = fev["y"].values[mk]
        hit = yk > 0
        puc = Pf[mk][hit, 1] / (Pf[mk][hit, 1] + Pf[mk][hit, 2])
        fr.append({"k_bps": int(k), "n": int(mk.sum()), "P_up_first": float((yk == 1).mean()),
                   "P_dn_first": float((yk == 2).mean()),
                   "MLL_naive": float(mlogloss(fnv["y"].values[mn], fnv[["pn", "pu", "pd"]].values[mn]).mean()),
                   "MLL_struct": float(mlogloss(yk, fev[["pn", "pu", "pd"]].values[mk]).mean()),
                   "MLL_final": float(mlogloss(yk, Pf[mk]).mean()),
                   "AUC_upfirst|hit": float(auc((yk[hit] == 1).astype(int), puc))})
    ft = pd.DataFrame(fr)
    ft["skill_vs_naive_%"] = 100 * (1 - ft["MLL_final"] / ft["MLL_naive"])
    log(f"\n[{HEAD}] FIRST PASSAGE +/-k around P0 (neither / up-first / down-first):")
    log(ft.to_string(index=False))
    out["first_passage"] = fr

    tag = f"{HEAD}_{SPLIT}{TAG}"
    json.dump(out, open(os.path.join(OUT, f"results_{tag}.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, f"log_{tag}.txt"), "w").write("\n".join(LOG))
    pickle.dump({"y": y, "e": ev["e"].values, "day": g, "T0": ev["T0"].values, "p": p, "mu": ev["mu"].values,
                 "sig": ev["sig"].values, "inc": inc, "xcur": x},
                open(os.path.join(MODEL_DIR, f"preds_{tag}.pkl"), "wb"))
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
