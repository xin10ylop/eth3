"""Paired comparison of two model versions on identical holdout rows.

    python scripts/compare_versions.py test "" _wf5
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.prob import logloss, block_bootstrap_diff, ece, auc  # noqa: E402

SPLIT, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
from btcmodel.config import MODEL_DIR as MD, ART_DIR  # noqa: E402


def main():
    lines = [f"Paired comparison on {SPLIT}: A='{A or 'frozen-2025'}' vs B='{B}' (negative dLL = B better)"]
    for head in ["h10", "h15", "h30", "h60", "close"]:
        fa, fb = os.path.join(MD, f"preds_{head}_{SPLIT}{A}.pkl"), os.path.join(MD, f"preds_{head}_{SPLIT}{B}.pkl")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        Pa, Pb = pickle.load(open(fa, "rb")), pickle.load(open(fb, "rb"))
        assert np.array_equal(Pa["T0"], Pb["T0"]) and np.array_equal(Pa["e"], Pb["e"])
        y, day, e = Pb["y"], Pb["day"], Pb["e"]
        pa, pb, p0 = Pa["p"]["E_blend"], Pb["p"]["E_blend"], Pb["p"]["B0_naive"]
        la, lb, l0 = logloss(y, pa), logloss(y, pb), logloss(y, p0)
        rows = []
        for nm, m in [("all", np.ones(len(y), bool)), ("e = 0", e == 0), ("e < 240", e < 240),
                      ("last 60 s (e >= 240)", e >= 240), ("last 30 s (e >= 270)", e >= 270)]:
            if m.sum() < 1000:
                continue
            d, lo, hi, _ = block_bootstrap_diff(lb[m], la[m], day[m], n_boot=1000)
            rows.append({"rows": nm, "skill A %": 100 * (1 - la[m].mean() / l0[m].mean()),
                         "skill B %": 100 * (1 - lb[m].mean() / l0[m].mean()),
                         "dLL B-A": d, "CI lo": lo, "CI hi": hi, "ECE A": ece(y[m], pa[m]), "ECE B": ece(y[m], pb[m])})
        lines.append(f"\n## {head}\n" + pd.DataFrame(rows).to_string(index=False, float_format="%.5f"))
        inc = Pb["inc"]
        nz = inc != 0
        lines.append(f"directional AUC: A={auc((inc[nz] > 0).astype(int), Pa['mu'][nz]):.4f}  "
                     f"B={auc((inc[nz] > 0).astype(int), Pb['mu'][nz]):.4f}")
    s = "\n".join(lines)
    print(s)
    open(os.path.join(ART_DIR, f"compare_{SPLIT}{A or '_frozen'}_vs{B}.txt"), "w").write(s)


if __name__ == "__main__":
    main()
