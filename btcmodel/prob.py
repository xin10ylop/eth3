"""Probability tools: Brownian first-passage formulas, scoring rules, bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, log_ndtr

EPS = 1e-6


# ----------------------------------------------------------------------------
# Brownian motion with drift: single and double barrier
# ----------------------------------------------------------------------------


def _F_hit(L, mu, s, T):
    """P(first passage of BM(mu, s) to level L>0 happens before T).  s = sigma (per unit time)."""
    sT = s * np.sqrt(T)
    a = ndtr((mu * T - L) / sT)
    # exp(2 mu L / s^2) * Phi(-(mu T + L)/sT), computed in log space for stability
    lb = 2 * mu * L / s ** 2 + log_ndtr((-mu * T - L) / sT)
    return a + np.exp(np.minimum(lb, 50.0))


def p_touch(a, mu, s, T):
    """P(max_{0..T} X >= a) for X = mu t + s W, a > 0 (a<=0 -> 1)."""
    a = np.asarray(a, dtype=float)
    out = np.where(a <= 0, 1.0, _F_hit(np.maximum(a, 1e-300), mu, s, T))
    return np.clip(out, 0.0, 1.0)


def p_double_barrier(a, b, mu, s, T, nterms: int = 6):
    """For X = mu t + s W from 0 with barriers +a (>0) and -b (<0), horizon T.

    Returns (P(up first within T), P(down first within T)).
    Method of images on the exit-at-upper-barrier density + Girsanov drift factor:
      P_up = e^{mu a/s^2} [ sum_{n>=0} e^{-mu L_n/s^2} F(L_n) - sum_{n>=1} e^{-mu L'_n/s^2} F(L'_n) ]
      L_n = a + 2 n w,  L'_n = 2 n w - a,  w = a + b,  F = first-passage CDF of BM(mu,s).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    mu = np.asarray(mu, float) + 0 * a; s = np.asarray(s, float) + 0 * a; T = np.asarray(T, float) + 0 * a

    def up(a, b, mu):
        w = a + b
        tot = np.zeros_like(a)
        for n in range(nterms):
            L = a + 2 * n * w
            tot += np.exp(np.clip(mu * (a - L) / s ** 2, -700, 50)) * _F_hit(L, mu, s, T)
            if n >= 1:
                L2 = 2 * n * w - a
                tot -= np.exp(np.clip(mu * (a - L2) / s ** 2, -700, 50)) * _F_hit(L2, mu, s, T)
        return tot

    pu = np.clip(up(a, b, mu), 0, 1)
    pd = np.clip(up(b, a, -mu), 0, 1)
    tot = pu + pd
    scale = np.where(tot > 1, 1 / np.maximum(tot, 1e-12), 1.0)
    return pu * scale, pd * scale


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y, p):
    return (p - y) ** 2


def mlogloss(y_idx, P):
    P = np.clip(P, EPS, 1)
    P = P / P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(y_idx)), y_idx])


def ece(y, p, nb=20):
    """Expected calibration error with equal-mass bins."""
    o = np.argsort(p)
    bins = np.array_split(o, nb)
    return sum(len(ix) * abs(y[ix].mean() - p[ix].mean()) for ix in bins) / len(p)


def reliability(y, p, nb=20):
    o = np.argsort(p)
    bins = np.array_split(o, nb)
    return np.array([[p[ix].mean(), y[ix].mean(), len(ix)] for ix in bins])


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def crps_from_quantiles(y, qs, levels):
    """CRPS approximated by integrated pinball loss over quantile levels."""
    tot = 0.0
    for j, a in enumerate(levels):
        d = y - qs[:, j]
        tot = tot + np.maximum(a * d, (a - 1) * d)
    return 2 * tot / len(levels)


def block_bootstrap_diff(loss_a, loss_b, groups, n_boot=2000, seed=0):
    """Mean loss difference (a - b) with a bootstrap CI that resamples whole groups
    (days), because all decision points of a day/window share outcomes and regime.

    Returns (mean diff, 2.5%, 97.5%, P(diff >= 0))."""
    rng = np.random.default_rng(seed)
    d = loss_a - loss_b
    ug, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=d)
    n = np.bincount(inv)
    G = len(ug)
    idx = rng.integers(0, G, size=(n_boot, G))
    bs = s[idx].sum(1) / n[idx].sum(1)
    m = d.mean()
    return m, np.quantile(bs, 0.025), np.quantile(bs, 0.975), (bs >= 0).mean()
