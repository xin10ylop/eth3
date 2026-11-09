"""Live / on-demand inference.

    pred = Predictor.load("/path/to/models.pkl")
    pred.ingest(spot_1s_bars, perp_1s_bars)        # arrays covering >= 25h of history
    out = pred.predict(epoch_second)               # continuous, any second of a window

`predict` returns, for the 5-minute window that contains `epoch_second`:
  * the fixed reference P0 (open of the window) and the current price,
  * for each head (30s, 60s, close): P(S_target >= P0), quantiles of S_target in $,
    touch probabilities for levels P0*(1+k), first-passage probabilities, and the
    model's scale / drift diagnostics.

Inputs are the same fields as the archive (see data.py): spot 1s klines and perp
aggTrades aggregated to 1s.  The buffer must start on an hour boundary.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.special import logit

from .features import Base, FeatureEngine, W
from scipy.special import expit
from .models import add_derived, VOL_FEATS, DIR_FEATS

DIRECT_FEATS = sorted(set(VOL_FEATS + DIR_FEATS + ["lH", "z_naive", "lproj_300", "lsint"]))
from .events import TOUCH_LEVELS, FP_LEVELS, TOUCH_FEATS, FP_FEATS
from .prob import p_touch, p_double_barrier

HEAD_H = {"h10": 10, "h15": 15, "h30": 30, "h60": 60, "close": None}   # order matters (stacking)
QLEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class _Buffer:
    """Timeline-like source over an in-memory buffer (all seconds present)."""

    def __init__(self, arrays: dict, t0_epoch: int):
        self.d = arrays
        self.t0_epoch = t0_epoch
        self.n = len(next(iter(arrays.values())))
        self.present_sec = np.ones(self.n, dtype=bool)

    def __getitem__(self, k):
        return self.d[k]


class Predictor:
    def __init__(self, models: dict, season, recalibrator=None):
        self.models = models
        self.season = season
        self.recal = recalibrator          # optional btcmodel.recalibration.OnlineRecalibrator
        self.buf = None

    @staticmethod
    def load_dir(path: str, recalibrator=None):
        """Load packaged models (e.g. models/wf5_20260430/{h10,h15,h30,h60,close}.pkl.gz)."""
        import gzip
        import os
        models = {}
        for h in HEAD_H:
            f = os.path.join(path, f"{h}.pkl.gz")
            if os.path.exists(f):
                with gzip.open(f, "rb") as fh:
                    models[h] = pickle.load(fh)
        with open(os.path.join(path, "season.pkl"), "rb") as fh:
            season = pickle.load(fh)
        return Predictor(models, season, recalibrator)

    @staticmethod
    def load(path: str, season_path: str):
        with open(path, "rb") as fh:
            models = pickle.load(fh)
        with open(season_path, "rb") as fh:
            season = pickle.load(fh)
        return Predictor(models, season)

    def ingest(self, arrays: dict, t0_epoch: int):
        """arrays: dict of per-second arrays (fields as in timeline.FIELDS) starting at t0_epoch."""
        assert t0_epoch % 3600 == 0, "buffer must start on an hour boundary"
        self.buf = _Buffer(arrays, t0_epoch)

    def predict(self, epoch_second: int, p0: float | None = None) -> dict:
        """p0: the reference (first trade of the window). If None it is read from the buffer
        (open of the window's first 1s bar), which requires that bar to be present."""
        buf = self.buf
        i = epoch_second - buf.t0_epoch            # decision index (bars < i are known)
        T0 = i // W * W
        e = i - T0
        assert T0 >= 86400, "need >= 1 day of history before the window"
        assert i <= buf.n, "buffer does not reach the decision time"
        src = buf
        if buf.n < T0 + W:
            # pad the not-yet-happened part of the window (never read by any feature:
            # every feature uses bars < i only - see tests/test_no_lookahead.py)
            pad = T0 + W - buf.n
            arr = {}
            for k, a in buf.d.items():
                fill = a[-1] if np.asarray(a).dtype.kind == "f" and k[2:] in (
                    "open", "high", "low", "close", "vwap") else 0
                arr[k] = np.concatenate([np.asarray(a), np.full(pad, fill, dtype=np.asarray(a).dtype)])
            src = _Buffer(arr, buf.t0_epoch)
        if p0 is not None:
            arr = dict(src.d)
            so = np.array(arr["s_open"], dtype=np.float64)
            so[T0] = p0
            arr["s_open"] = so
            src = _Buffer(arr, buf.t0_epoch)
        base = Base(src, 0, T0 + W)
        fe = FeatureEngine(base, self.season)
        X, aux = fe.compute(np.array([T0]), np.array([e]))
        if any(getattr(m["head"], "regime", []) for m in self.models.values()):
            from .regime import regime_features
            R = regime_features(src, np.array([T0]), np.array([e]))
            for k in R.columns:
                X[k] = R[k].values
        X["xcur"] = aux["cur"] - aux["p0"]
        X["v"] = aux["v"].astype(np.float32)
        p0 = float(np.exp(aux["p0"][0]))
        out = {"window_start_epoch": int(buf.t0_epoch + T0), "elapsed_s": int(e), "P0": p0,
               "price_now": float(np.exp(aux["cur"][0])), "heads": {}}
        for head, h in HEAD_H.items():
            if head not in self.models:
                continue
            H = (W - e) if h is None else h
            if e + H > W:
                continue
            d = X.copy()
            d[f"{head}__H"] = np.int16(H)
            d[f"{head}__sint"] = self.season.integrated(np.array([T0 + e]), np.array([H]), base.how_hour).astype(np.float32)
            for sh in HEAD_H:
                if f"mu_{sh}_px" in self.models[head]["head"].extra:
                    sub = out["heads"].get(sh)
                    d[f"mu_{sh}_px"] = np.float32(np.nan if sub is None else sub["_drift_px"])
            d = add_derived(d, head)
            if any(c.startswith("mu_h") for c in self.models[head]["head"].extra):
                for c in self.models[head]["head"].extra:
                    d[c] = (d[c] / d["naive_scale"]).astype(np.float32)
            hm = self.models[head]["head"]
            hm.prepare(d)
            p_m = float(hm.p_up(d)[0])
            dm = self.models[head]["direct"]
            dfe = sorted(set(DIRECT_FEATS + list(getattr(hm, "regime", []))))
            p_d = float(dm.predict(d[dfe], num_iteration=dm.best_iteration)[0])
            p_m, p_d = min(max(p_m, 1e-5), 1 - 1e-5), min(max(p_d, 1e-5), 1 - 1e-5)
            p_blend = float(expit(0.5 * logit(p_m) + 0.5 * logit(p_d)))
            p_final = p_blend if self.recal is None else float(self.recal.apply(head, e, p_blend))
            r = {"horizon_s": int(H),
                 "p_above_ref": p_final,              # final forecast (recalibrated if enabled)
                 "p_above_ref_blend": p_blend,        # pre-registered blend, before recalibration
                 "p_above_ref_structured": p_m, "p_above_ref_direct": p_d}
            q = hm.quantiles(d, np.array(QLEVELS))[0]
            r["quantiles_usd"] = {str(a): float(p0 * np.exp(v)) for a, v in zip(QLEVELS, q)}
            r["scale_bps"] = float(d["sig"].values[0] * 1e4)
            r["drift_std"] = float(d["mu"].values[0])
            r["_drift_px"] = float(d["mu"].values[0] * d["sig"].values[0])
            r["touch"] = self._touch(d, hm, self.models[head]["touch"])
            r["first_passage"] = self._fp(d, hm, self.models[head]["fp"])
            out["heads"][head] = r
        for r in out["heads"].values():
            r.pop("_drift_px", None)
        return out

    @staticmethod
    def _touch(d, hm, tm):
        mu, s, H = hm.bm_params(d)
        x = d["xcur"].values[0]
        res = {}
        for k in TOUCH_LEVELS:
            L = np.log1p(k * 1e-4)
            dist = (L - x) if k > 0 else (x - L)
            if dist <= 0:
                res[f"{k:+d}bps"] = 1.0
                continue
            sgn = 1.0 if k > 0 else -1.0
            p = float(np.clip(p_touch(np.array([dist]), sgn * mu, s, H)[0], 1e-6, 1 - 1e-6))
            row = pd.DataFrame({"d_std": [dist / d["sig"].values[0]], "mu_to": [sgn * d["mu"].values[0]],
                                "is_down": [float(k < 0)], "lev_std": [abs(L) / d["sig"].values[0]],
                                "p_struct": [p], "lp_struct": [logit(p)]})
            for c in ["lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "ncross", "hod_et"]:
                row[c] = d[c].values
            row = row.astype({c: np.float32 for c in TOUCH_FEATS})
            res[f"{k:+d}bps"] = float(tm.predict(row)[0])
        return res

    @staticmethod
    def _fp(d, hm, fm):
        mu, s, H = hm.bm_params(d)
        x = d["xcur"].values[0]
        res = {}
        for k in FP_LEVELS:
            U, D = np.log1p(k * 1e-4), np.log1p(-k * 1e-4)
            if not (D < x < U):
                res[f"+/-{k}bps"] = {"up_first": float(x >= U), "down_first": float(x <= D), "neither": 0.0}
                continue
            a, b = np.array([U - x]), np.array([x - D])
            pu, pd_ = p_double_barrier(a, b, mu, s, H)
            pu, pd_ = max(pu[0], 1e-6), max(pd_[0], 1e-6)
            pn = max(1 - pu - pd_, 1e-6)
            z = pu + pd_ + pn
            row = pd.DataFrame({"pn": [pn / z], "pu": [pu / z], "pd": [pd_ / z],
                                "a_std": [a[0] / d["sig"].values[0]], "b_std": [b[0] / d["sig"].values[0]]})
            row["lpu"] = np.log(row["pu"] / row["pn"])
            row["lpd"] = np.log(row["pd"] / row["pn"])
            for c in ["mu", "lsig_rel", "e", "lH", "lrvsa_300", "jump_300", "som", "hod_et"]:
                row[c] = d[c].values
            row = row.astype({c: np.float32 for c in FP_FEATS})
            P = fm.predict(row)[0]
            res[f"+/-{k}bps"] = {"neither": float(P[0]), "up_first": float(P[1]), "down_first": float(P[2])}
        return res
