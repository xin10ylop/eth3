"""Targets for additional short heads (10 s, 15 s) on the existing decision points.

Same conventions as features.FeatureEngine.targets: target price = close of bar
T0 + e + h - 1 (only while e + h <= 300), path extremes over bars T0+e .. T0+e+h-1."""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.dataset import CACHE, load_season  # noqa: E402
from btcmodel.features import _future_extremes, _first_passage, et_hour_of_week, W  # noqa: E402
from btcmodel.timeline import load_timeline  # noqa: E402

EXTRA = {"h10": 10, "h15": 15}

if __name__ == "__main__":
    tl = load_timeline()
    season = load_season()
    ls = np.log(np.asarray(tl["s_close"]))
    lhi = np.log(np.asarray(tl["s_high"]))
    llo = np.log(np.asarray(tl["s_low"]))
    lop = np.log(np.asarray(tl["s_open"]))
    how = et_hour_of_week(tl.t0_epoch, tl.n)
    for split in (sys.argv[1:] or ["train", "valid", "test"]):
        t = time.time()
        d = pd.read_parquet(os.path.join(CACHE, f"{split}.parquet"), columns=["T0", "e"])
        T0 = d["T0"].values.astype(np.int64)
        e = d["e"].values.astype(np.int64)
        p0 = lop[T0]
        cur = np.where(e == 0, p0, ls[np.maximum(T0 + e - 1, 0)])
        out = {}
        for hn, h in EXTRA.items():
            H = np.full(len(e), h, dtype=np.int64)
            valid = (e + H) <= W
            end = np.minimum(T0 + e + H - 1, tl.n - 1)
            y = ls[end] - p0
            start = T0 + e
            mx, mn = _future_extremes(lhi, llo, start, H)
            out[f"{hn}__valid"] = valid
            out[f"{hn}__y"] = y
            out[f"{hn}__inc"] = y - (cur - p0)
            out[f"{hn}__mx"] = mx - p0
            out[f"{hn}__mn"] = mn - p0
            out[f"{hn}__H"] = H.astype(np.int16)
            out[f"{hn}__sint"] = season.integrated(start, H, how).astype(np.float32)
            for kb in (5, 10, 20):
                up = p0 + np.log1p(kb * 1e-4)
                dn = p0 + np.log1p(-kb * 1e-4)
                out[f"{hn}__fp{kb}"] = _first_passage(lhi, llo, start, H, up, dn)
        pd.DataFrame(out).to_parquet(os.path.join(CACHE, f"{split}_targets_extra.parquet"))
        print(split, len(d), f"{time.time() - t:.0f}s", flush=True)
