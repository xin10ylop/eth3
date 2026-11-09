"""The live Predictor (raw buffer -> prediction) must reproduce the batch pipeline."""
import os
import pickle
import sys

import numpy as np

from btcmodel.dataset import load_season
from btcmodel.predictor import Predictor
from btcmodel.timeline import load_timeline, FIELDS


def run(split="valid", heads=("h30",), n=6, seed=0, tag="", model_split="valid", buffer_days=2):
    tl = load_timeline()
    from btcmodel.config import MODEL_DIR
    models = {h: pickle.load(open(os.path.join(MODEL_DIR, f"models_{h}_{model_split}{tag}.pkl"), "rb")) for h in heads}
    preds = {h: pickle.load(open(os.path.join(MODEL_DIR, f"preds_{h}_{split}{tag}.pkl"), "rb")) for h in heads}
    pr = Predictor(models, load_season())
    rng = np.random.default_rng(seed)
    P = preds[heads[0]]
    worst = 0.0
    for k in rng.choice(len(P["y"]), n, replace=False):
        T0, e = int(P["T0"][k]), int(P["e"][k])
        start = (T0 - buffer_days * 86400) // 3600 * 3600   # hour-aligned buffer start
        end = T0 + e                                       # only what is known at decision time
        arrays = {f: np.asarray(tl[f][start:end]) for f in FIELDS}
        pr.ingest(arrays, tl.t0_epoch + start)
        out = pr.predict(tl.t0_epoch + T0 + e, p0=float(np.asarray(tl["s_open"][T0])))
        for h in heads:
            Ph = preds[h]
            j = np.flatnonzero((Ph["T0"] == T0) & (Ph["e"] == e))
            if len(j) == 0 or h not in out["heads"]:
                continue
            batch = Ph["p"]["E_blend"][j[0]] if "E_blend" in Ph["p"] else None
            if batch is not None:
                worst = max(worst, abs(batch - out["heads"][h]["p_above_ref_blend"]))
            live = out["heads"][h]["p_above_ref_blend"]
            bm, lm = Ph["p"]["M_final"][j[0]], out["heads"][h]["p_above_ref_structured"]
            print(f"{h} T0={T0} e={e}: batch structured={bm:.6f} live structured={lm:.6f}"
                  + (f" | batch blend={batch:.6f} live blend={live:.6f}" if batch is not None else ""))
            worst = max(worst, abs(bm - lm))
    print("max |batch - live| =", worst)
    assert worst < 1e-4


if __name__ == "__main__":
    # python -m tests.test_predictor_consistency [tag] [split] [heads...]
    # e.g. python -m tests.test_predictor_consistency _wf5 test h10 h15 h30 h60 close
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    split = sys.argv[2] if len(sys.argv) > 2 else "valid"
    heads = tuple(sys.argv[3:]) or ("h30",)
    run(split=split, heads=heads, tag=tag, buffer_days=9 if tag else 2)
