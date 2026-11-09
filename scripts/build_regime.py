"""Compute regime-state features for every cached decision point (same row order)."""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.dataset import CACHE  # noqa: E402
from btcmodel.regime import regime_features  # noqa: E402
from btcmodel.timeline import load_timeline  # noqa: E402

if __name__ == "__main__":
    tl = load_timeline()
    for split in (sys.argv[1:] or ["train", "valid", "test"]):
        t = time.time()
        d = pd.read_parquet(os.path.join(CACHE, f"{split}.parquet"), columns=["T0", "e"])
        r = regime_features(tl, d["T0"].values.astype(np.int64), d["e"].values.astype(np.int64))
        assert len(r) == len(d)
        r.reset_index(drop=True).to_parquet(os.path.join(CACHE, f"{split}_regime.parquet"))
        print(split, r.shape, f"{time.time() - t:.0f}s")
        print(r.describe().T[["mean", "std", "min", "max"]].to_string())
