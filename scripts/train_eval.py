"""Train all heads on TRAIN (early stopping on VALID) and evaluate on a chosen split.

Usage:  python scripts/train_eval.py valid      # development evaluation
        python scripts/train_eval.py test       # final holdout (run once)

Baseline ladder for P(S_target >= P0), every rung adds one idea:
  B0  naive      : Gaussian random walk, trailing-300s vol, sigma*sqrt(H)   (what most people do)
  B1  +vol model : Gaussian, learned scale (clock-aware term structure)
  B2  +fat tails : empirical residual law, no drift
  B3  +drift     : structural model (scale + drift + empirical law)
  M   final      : B3 + boosted calibration from the structural prior
  D   direct GBM : one big LightGBM classifier on all features (no structure) - the 'naive ML' rival
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from btcmodel.dataset import load_split  # noqa: E402
from btcmodel.models import HeadModel, add_derived, _fit, VOL_FEATS, DIR_FEATS, SQ2PI  # noqa: E402
from btcmodel.events import (touch_rows, naive_touch, TouchModel, fp_rows, FirstPassageModel)  # noqa: E402
from btcmodel.prob import logloss, brier, ece, auc, block_bootstrap_diff, mlogloss  # noqa: E402

from btcmodel.config import ART_DIR as OUT, MODEL_DIR, STACK_DIR  # noqa: E402
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
EVAL = sys.argv[1] if len(sys.argv) > 1 else "valid"
HEAD_ORDER = os.environ.get("BTC_HEADS", "h10,h15,h30,h60,close").split(",")  # shorter feed longer (stacking)
EXTRA_TARGET_HEADS = ("h10", "h15")                  # targets live in <split>_targets_extra.parquet
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
USE_REGIME = os.environ.get("BTC_REGIME", "0") == "1"     # opt-in regime-state features
TAG = os.environ.get("BTC_TAG", "")                       # version suffix for outputs
os.makedirs(STACK_DIR, exist_ok=True)
LOG = []
PREDS = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


META = ["T0", "e", "xcur", "v", "day"]


def load_head(split, head):
    """Load only the columns this head needs, rows valid for this head, float32 features."""
    from btcmodel.dataset import CACHE
    import pyarrow.parquet as pq
    allc = pq.read_schema(os.path.join(CACHE, f"{split}.parquet")).names
    feats = [c for c in allc if "__" not in c and c not in META]
    tcols = [c for c in allc if c.startswith(head + "__")]
    d = pd.read_parquet(os.path.join(CACHE, f"{split}.parquet"), columns=feats + tcols + META)
    if head in EXTRA_TARGET_HEADS:
        x = pd.read_parquet(os.path.join(CACHE, f"{split}_targets_extra.parquet"))
        x = x[[c for c in x.columns if c.startswith(head + "__")]]
        assert len(x) == len(d)
        d = pd.concat([d, x], axis=1)
    if USE_REGIME:
        r = pd.read_parquet(os.path.join(CACHE, f"{split}_regime.parquet"))
        assert len(r) == len(d)
        d = pd.concat([d, r], axis=1)
    d = d[d[f"{head}__valid"].values].reset_index(drop=True)
    for c in feats:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return add_derived(d, head)


def summarize(name, y, p, groups, ref=None):
    r = {"model": name, "logloss": float(logloss(y, p).mean()), "brier": float(brier(y, p).mean()),
         "ece": float(ece(y, p))}
    if ref is not None:
        m, lo, hi, _ = block_bootstrap_diff(logloss(y, p), logloss(y, ref), groups)
        r["dLL_vs_B0"] = m
        r["ci95"] = (lo, hi)
    return r


def main():
    t0 = time.time()
    results = {}
    models = {}
    for head in ([ONLY] if ONLY else HEAD_ORDER):
        log(f"\n=========== HEAD {head} ===========")
        tr, va = load_head("train", head), load_head("valid", head)
        ev = va if EVAL == "valid" else load_head(EVAL, head)
        log(f"loaded train {tr.shape}, valid {va.shape}, eval {ev.shape} ({time.time() - t0:.0f}s)")
        # cross-horizon stacking: short-horizon drift (higher SNR) as a feature for longer heads
        extra = []
        for sh in [h for h in HEAD_ORDER if HEAD_ORDER.index(h) < HEAD_ORDER.index(head)]:
            for d, key in ((tr, "train"), (va, "valid"), (ev, EVAL)):
                s = pd.read_parquet(os.path.join(STACK_DIR, f"{sh}_{key}_{EVAL}{TAG}.parquet"))
                # merge on (T0, e); rows of the short head are a subset of e-values
                d[f"mu_{sh}_px"] = pd.merge(d[["T0", "e"]], s, on=["T0", "e"], how="left")["drift_px"].values
                d[f"mu_{sh}_px"] = (d[f"mu_{sh}_px"] / d["naive_scale"]).astype(np.float32)
            extra.append(f"mu_{sh}_px")
        from btcmodel.regime import REGIME_FEATS
        hm = HeadModel(head, extra_dir_feats=extra,
                       regime_feats=REGIME_FEATS if USE_REGIME else None).fit(tr, va, log=log)
        if ev is not va:
            hm.prepare(ev)
        models[head] = hm
        # drift in price units (for stacking into longer heads)
        for d, key in ((tr, "train"), (va, "valid"), (ev, EVAL)):
            pd.DataFrame({"T0": d["T0"].values, "e": d["e"].values,
                          "drift_px": d["mu"].values * d["sig"].values}).to_parquet(
                os.path.join(STACK_DIR, f"{head}_{key}_{EVAL}{TAG}.parquet"))

        # ------------------------------------------------ P(up) ladder
        y = (ev[f"{head}__y"].values >= 0).astype(int)
        H = ev[f"{head}__H"].values.astype(float)
        x = ev["xcur"].values
        g = ev["day"].values
        p = {}
        p["B0_naive"] = np.clip(ndtr(x / (ev["v"].values * np.sqrt(H))), 1e-5, 1 - 1e-5)
        sd = ev["sig"].values / SQ2PI
        p["B1_volmodel"] = np.clip(ndtr(x / sd), 1e-5, 1 - 1e-5)
        p["B2_fattail"] = np.clip(hm.struct_up(ev, use_drift=False), 1e-5, 1 - 1e-5)
        p["B3_struct"] = ev["p_struct"].values
        p["M_final"] = hm.p_up(ev)
        # direct GBM rival
        feats = sorted(set(VOL_FEATS + DIR_FEATS + ["lH", "z_naive", "lproj_300", "lsint"]
                           + (REGIME_FEATS if USE_REGIME else [])))
        ytr = (tr[f"{head}__y"].values >= 0).astype(int)
        yva = (va[f"{head}__y"].values >= 0).astype(int)
        dm = _fit(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000),
                  tr[feats], ytr, va[feats], yva, es=100)
        p["D_directGBM"] = np.clip(dm.predict(ev[feats], num_iteration=dm.best_iteration), 1e-5, 1 - 1e-5)
        log(f"[{head}] direct GBM: {dm.best_iteration} rounds")
        # pre-registered final forecast (chosen on h30 validation, weight NOT tuned further):
        # equal-weight logit blend of the structured model and the direct GBM
        p["E_blend"] = expit(0.5 * logit(p["M_final"]) + 0.5 * logit(p["D_directGBM"]))

        res = []
        for k, pk in p.items():
            r = summarize(k, y, pk, g, ref=p["B0_naive"] if k != "B0_naive" else None)
            res.append(r)
        tab = pd.DataFrame(res)
        log(f"\n[{head}] P(S_target >= P0) on {EVAL}: n={len(y)}, windows={ev['T0'].nunique()}, base rate={y.mean():.4f}")
        log(tab.to_string(index=False))
        # pure direction skill (relative to *current* price) - the honest measure of edge
        inc = ev[f"{head}__inc"].values
        nz = inc != 0
        log(f"[{head}] directional AUC of drift (sign of increment, excl. zero): "
            f"{auc((inc[nz] > 0).astype(int), ev['mu'].values[nz]):.4f}; corr(mu, u)="
            f"{np.corrcoef(ev['mu'].values, np.clip(inc / ev['sig'].values, -8, 8))[0, 1]:.4f}")
        # by elapsed-time bucket
        eb = pd.cut(ev["e"].values, [-1, 0, 30, 60, 120, 180, 240, 270, 299])
        rows = []
        for b in eb.categories:
            m = (eb == b)
            if m.sum() == 0:
                continue
            rr = {"e_bucket": str(b), "n": int(m.sum())}
            for k in ["B0_naive", "B1_volmodel", "B3_struct", "M_final", "D_directGBM", "E_blend"]:
                rr[k] = float(logloss(y[m], p[k][m]).mean())
            rr["skill_M_vs_B0_%"] = 100 * (1 - rr["M_final"] / rr["B0_naive"])
            rr["skill_E_vs_B0_%"] = 100 * (1 - rr["E_blend"] / rr["B0_naive"])
            rows.append(rr)
        log(pd.DataFrame(rows).to_string(index=False))
        results[head] = {"up_table": res, "by_e": rows}
        PREDS[head] = {"y": y, "e": ev["e"].values, "day": g, "T0": ev["T0"].values, "p": p,
                       "mu": ev["mu"].values, "sig": ev["sig"].values, "inc": ev[f"{head}__inc"].values}

        # quantiles / CRPS proxy
        alphas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        Q = hm.quantiles(ev, alphas)
        yy = ev[f"{head}__y"].values
        cover = [(yy <= Q[:, j]).mean() for j in range(len(alphas))]
        from scipy.stats import norm
        Qn = x[:, None] + ev["v"].values[:, None] * np.sqrt(H)[:, None] * norm.ppf(alphas)[None, :]
        pin = lambda Qm: np.mean([np.mean(np.maximum(a * (yy - Qm[:, j]), (a - 1) * (yy - Qm[:, j])))
                                  for j, a in enumerate(alphas)])
        log(f"[{head}] quantile coverage (target {alphas.tolist()}): {np.round(cover, 3).tolist()}")
        log(f"[{head}] mean pinball loss (bps): model={pin(Q) * 1e4:.4f} naive-gauss={pin(Qn) * 1e4:.4f} "
            f"skill={100 * (1 - pin(Q) / pin(Qn)):.2f}%")
        results[head]["quantiles"] = {"coverage": cover, "pinball_model": pin(Q), "pinball_naive": pin(Qn)}

        # ------------------------------------------------ touch probabilities
        sub = lambda d, f: d.iloc[::f].reset_index(drop=True)
        trs = sub(tr, 3)
        tr_rows, va_rows = touch_rows(trs, head, hm), touch_rows(sub(va, 2), head, hm)
        tm = TouchModel().fit(tr_rows, va_rows, log=log)
        del tr_rows, va_rows
        ev_rows = touch_rows(ev, head, hm)
        pt = tm.predict(ev_rows)
        pn = naive_touch(ev, head)
        yt = ev_rows["y"].values
        trs_res = []
        for k in sorted(ev_rows["k"].unique()):
            m = ev_rows["k"].values == k
            trs_res.append({"level_bps": int(k), "n": int(m.sum()), "rate": float(yt[m].mean()),
                            "LL_naiveBM": float(logloss(yt[m], pn[m]).mean()),
                            "LL_structBM": float(logloss(yt[m], ev_rows["p_struct"].values[m]).mean()),
                            "LL_final": float(logloss(yt[m], pt[m]).mean()),
                            "ECE_final": float(ece(yt[m], pt[m]))})
        tt = pd.DataFrame(trs_res)
        tt["skill_vs_naive_%"] = 100 * (1 - tt["LL_final"] / tt["LL_naiveBM"])
        log(f"\n[{head}] TOUCH P(reach P0*(1+k) within horizon), non-trivial cases:")
        log(tt.to_string(index=False))
        m_all, lo, hi, _ = block_bootstrap_diff(logloss(yt, pt), logloss(yt, pn), ev_rows["day"].values)
        log(f"[{head}] touch: dLL final - naive = {m_all:.5f}  95% CI [{lo:.5f}, {hi:.5f}]")
        results[head]["touch"] = trs_res

        # ------------------------------------------------ first passage
        ftr, fva = fp_rows(trs, head, hm), fp_rows(sub(va, 2), head, hm)
        fm = FirstPassageModel().fit(ftr, fva, log=log)
        del ftr, fva
        fev = fp_rows(ev, head, hm)
        fnv = fp_rows(ev, head, hm, naive=True)
        Pf = fm.predict(fev)
        fres = []
        for k in sorted(fev["k"].unique()):
            m = fev["k"].values == k
            mn = fnv["k"].values == k
            yk = fev["y"].values[m]
            Pn = fnv[["pn", "pu", "pd"]].values[mn]
            Ps = fev[["pn", "pu", "pd"]].values[m]
            # among windows where one barrier was hit, P(up first) accuracy
            hitm = yk > 0
            pu_cond = Pf[m][hitm, 1] / (Pf[m][hitm, 1] + Pf[m][hitm, 2])
            fres.append({"k_bps": int(k), "n": int(m.sum()), "P_up_first": float((yk == 1).mean()),
                         "P_dn_first": float((yk == 2).mean()),
                         "MLL_naive": float(mlogloss(fnv["y"].values[mn], Pn).mean()),
                         "MLL_struct": float(mlogloss(yk, Ps).mean()),
                         "MLL_final": float(mlogloss(yk, Pf[m]).mean()),
                         "AUC_upfirst|hit": float(auc((yk[hitm] == 1).astype(int), pu_cond))})
        ft = pd.DataFrame(fres)
        ft["skill_vs_naive_%"] = 100 * (1 - ft["MLL_final"] / ft["MLL_naive"])
        log(f"\n[{head}] FIRST PASSAGE (+/-k around P0): classes neither/up-first/down-first")
        log(ft.to_string(index=False))
        results[head]["first_passage"] = fres
        models[head] = {"head": hm, "touch": tm, "fp": fm, "direct": dm}

        tag = f"{head}_{EVAL}{TAG}"
        with open(os.path.join(OUT, f"results_{tag}.json"), "w") as fh:
            json.dump(results[head], fh, indent=1, default=float)
        with open(os.path.join(OUT, f"log_{tag}.txt"), "w") as fh:
            fh.write("\n".join(LOG))
        with open(os.path.join(MODEL_DIR, f"models_{tag}.pkl"), "wb") as fh:
            pickle.dump(models[head], fh)
        with open(os.path.join(MODEL_DIR, f"preds_{tag}.pkl"), "wb") as fh:
            pickle.dump(PREDS[head], fh)
        del tr, va, ev, models[head], PREDS[head]
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
