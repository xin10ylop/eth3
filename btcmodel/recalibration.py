"""Causal online recalibration (adopted per pre-registered rule 4 for 10s/15s/30s/close).

Every day, per head and elapsed-time bucket, a 2-parameter logistic map
    logit(p') = a + b * logit(p)
is refit on the previous `window_days` days of forecasts whose outcomes are already known,
then applied to new forecasts.  It repairs slow calibration drift between monthly refits
without touching the model.  Same procedure as scripts/regime_eval.py (holdout-validated).

    rc = OnlineRecalibrator()
    rc.add("close", e, p, y, day)       # log resolved forecasts (arrays or scalars)
    rc.refit(today)                     # once per day, before trading that day
    p_live = rc.apply("close", e, p)    # identity until enough history exists
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.special import expit, logit

EDGES = (1, 61, 181, 241, 271)                 # elapsed-second buckets (as evaluated)
ADOPTED = ("h10", "h15", "h30", "close")       # holdout CI excluded zero for these heads


def _bucket(e):
    return np.digitize(np.asarray(e), EDGES)


def fit_platt(z, y, iters=25):
    w = np.array([0.0, 1.0])
    X = np.c_[np.ones_like(z), z]
    for _ in range(iters):
        p = expit(X @ w)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return w


class OnlineRecalibrator:
    def __init__(self, window_days: int = 14, min_rows: int = 5000, heads=ADOPTED):
        self.window_days, self.min_rows, self.heads = window_days, min_rows, set(heads)
        self.hist = defaultdict(lambda: {"z": [], "y": [], "d": [], "b": []})
        self.params = {}

    def add(self, head, e, p, y, day):
        if head not in self.heads:
            return
        h = self.hist[head]
        h["z"].append(np.atleast_1d(logit(np.clip(p, 1e-5, 1 - 1e-5))))
        h["y"].append(np.atleast_1d(y).astype(float))
        h["d"].append(np.broadcast_to(np.atleast_1d(day), np.atleast_1d(p).shape).astype(np.int64))
        h["b"].append(_bucket(np.broadcast_to(np.atleast_1d(e), np.atleast_1d(p).shape)))

    def refit(self, today: int):
        """Fit on days in [today - window_days, today) only (strictly the past)."""
        for head, h in self.hist.items():
            if not h["z"]:
                continue
            z, y, d, b = (np.concatenate(h[k]) for k in ("z", "y", "d", "b"))
            keep = d >= today - self.window_days
            for k, v in (("z", z), ("y", y), ("d", d), ("b", b)):
                h[k] = [v[keep]]
            z, y, d, b = z[keep], y[keep], d[keep], b[keep]
            past = d < today
            for bk in range(len(EDGES) + 1):
                m = past & (b == bk)
                self.params[(head, bk)] = fit_platt(z[m], y[m]) if m.sum() >= self.min_rows else None

    def apply(self, head, e, p):
        if head not in self.heads:
            return p
        p = np.asarray(p, float)
        z = logit(np.clip(p, 1e-5, 1 - 1e-5))
        b = _bucket(np.broadcast_to(e, p.shape))
        out = z.copy()
        for bk in np.unique(b):
            w = self.params.get((head, int(bk)))
            if w is not None:
                m = b == bk
                out[m] = w[0] + w[1] * z[m]
        return expit(out)
