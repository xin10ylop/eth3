"""Build cached feature/target matrices for each split.

Split policy (chronological, with 1-day embargo between blocks):
  train  : 2025-01-02 .. 2025-12-31   (model fitting; seasonality estimated here only)
  valid  : 2026-01-02 .. 2026-04-30   (early stopping / model selection)
  test   : 2026-05-02 .. 2026-11-02   (final holdout, looked at once at the end)
"""
from __future__ import annotations

import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

from .data import DATA_DIR
from .features import Base, FeatureEngine, Seasonality, decision_points, et_hour_of_week
from .timeline import load_timeline

CACHE = os.path.join(DATA_DIR, "cache")

SPLITS = {
    "train": (date(2025, 1, 2), date(2025, 12, 31)),
    "valid": (date(2026, 1, 2), date(2026, 4, 30)),
    "test": (date(2026, 5, 2), date(2026, 11, 2)),
}
HEADS = {"h30": 30, "h60": 60, "close": None}


def split_range(tl, name):
    a, b = SPLITS[name]
    return tl.index_of(a), tl.index_of(b) + 86400


def fit_seasonality(tl) -> Seasonality:
    """Clock profile from the TRAIN split only."""
    i0, i1 = split_range(tl, "train")
    ls = np.log(np.asarray(tl["s_close"][i0 - 1:i1]))
    absr = np.abs(np.diff(ls))
    how = et_hour_of_week(tl.t0_epoch + i0, i1 - i0)
    return Seasonality.fit(absr, how, np.ones(len(absr), dtype=bool))


def build_all(stride_train=10, stride_eval=5, seed=0, chunk_days=30):
    os.makedirs(CACHE, exist_ok=True)
    tl = load_timeline()
    season = fit_seasonality(tl)
    with open(os.path.join(CACHE, "season.pkl"), "wb") as fh:
        pickle.dump(season, fh)
    rng = np.random.default_rng(seed)
    for name in SPLITS:
        i0, i1 = split_range(tl, name)
        stride = stride_train if name == "train" else stride_eval
        parts = []
        for c0 in range(i0, i1, chunk_days * 86400):
            c1 = min(c0 + chunk_days * 86400, i1)
            a = c0 - 86400                       # one day of warm-up history
            b = min(c1 + 86400, tl.n)            # room for targets past the chunk
            base = Base(tl, a, b)
            fe = FeatureEngine(base, season)
            T0, e = decision_points(base, c0 - a, c1 - a, stride,
                                    rng=rng if name == "train" else None)
            X, aux = fe.compute(T0, e)
            for hn, h in HEADS.items():
                t = fe.targets(aux, h)
                X[f"{hn}__valid"] = t["valid"]
                X[f"{hn}__y"] = t["y"].astype(np.float64)
                X[f"{hn}__inc"] = t["inc"].astype(np.float64)
                X[f"{hn}__mx"] = t["mx"].astype(np.float64)
                X[f"{hn}__mn"] = t["mn"].astype(np.float64)
                X[f"{hn}__H"] = t["H"].astype(np.int16)
                X[f"{hn}__sint"] = t["season_int"].astype(np.float32)
                for kb in (5, 10, 20):
                    X[f"{hn}__fp{kb}"] = fe.first_passage(aux, t["H"], kb)
            X["T0"] = aux["T0"] + a              # absolute timeline index
            X["xcur"] = (aux["cur"] - aux["p0"]).astype(np.float64)
            X["v"] = aux["v"].astype(np.float32)
            X["day"] = ((aux["T0"] + a) // 86400).astype(np.int32)
            parts.append(X)
            print(f"  {name} chunk day {c0 // 86400}: {len(np.unique(T0))} windows, {len(T0)} points", flush=True)
            del base, fe
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(os.path.join(CACHE, f"{name}.parquet"))
        print(f"saved {name}: {df.shape}, {df['T0'].nunique()} windows", flush=True)


def load_split(name) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(CACHE, f"{name}.parquet"))


def load_season() -> Seasonality:
    with open(os.path.join(CACHE, "season.pkl"), "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    build_all()
