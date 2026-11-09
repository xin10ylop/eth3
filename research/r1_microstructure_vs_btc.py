"""R1: ETH vs BTC microstructure, side by side, 2025 only (training period).

What differs between the two markets, and therefore what must NOT be copied blindly:
tick size relative to volatility, last-trade bounce, perp->spot lead, basis-gap signal and how
its predictive power decays with horizon, the no-move atom, and the clock structure.
"""
import os
from datetime import date

import numpy as np

TL = {"ETH": "/home/user/data_eth/timeline", "BTC": "/home/user/data/timeline"}
N = (date(2026, 1, 1) - date(2025, 1, 1)).days * 86400


def ffill(a):
    m = np.isnan(a)
    i = np.where(~m, np.arange(len(a)), 0)
    np.maximum.accumulate(i, out=i)
    return a[i]


def load(asset):
    g = lambda k: np.load(os.path.join(TL[asset], k + ".npy"), mmap_mode="r")[:N]
    sc = np.asarray(g("s_close"))
    ls = np.log(sc)
    lp = np.log(ffill(np.asarray(g("p_close")).copy()))
    bad = ~np.isfinite(lp)
    lp[bad] = ls[bad]                     # before the first perp trade: fall back to spot
    print(asset, "non-finite spot:", int((~np.isfinite(ls)).sum()), " perp back-filled:", int(bad.sum()))
    return dict(ls=ls, price=sc, lp=lp,
                cnt=np.asarray(g("s_count")), lo=np.log(np.asarray(g("s_open"))))


def stats(asset):
    d = load(asset)
    ls, lp = d["ls"], d["lp"]
    rs, rp = np.diff(ls, prepend=ls[0]), np.diff(lp, prepend=lp[0])
    out = {}
    out["median price $"] = np.median(d["price"])
    out["tick (0.01) in bps"] = 0.01 / np.median(d["price"]) * 1e4
    out["1s vol bps"] = rs.std() * 1e4
    out["tick / 1s vol"] = out["tick (0.01) in bps"] / out["1s vol bps"]
    out["annualized vol"] = rs.std() * np.sqrt(365 * 86400)
    out["zero-trade seconds %"] = 100 * (d["cnt"] == 0).mean()
    out["zero 1s returns %"] = 100 * (rs == 0).mean()
    for k in (1, 2, 3):
        out[f"spot acf lag{k}"] = np.corrcoef(rs[k:], rs[:-k])[0, 1]
    for k in (1, 2, 5):
        out[f"perp->spot corr lag{k}"] = np.corrcoef(rs[k:], rp[:-k])[0, 1]
    out["spot<->perp corr 1s"] = np.corrcoef(rs, rp)[0, 1]
    # basis-gap signal: OOS linear R^2 for horizons (fit Jan-Jun, test Jul-Dec)
    cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    c2 = cs(rs ** 2)
    b = lp - ls
    cb = cs(b)
    t = np.arange(4000, N - 301, 11)
    v = np.sqrt(np.maximum((c2[t + 1] - c2[t - 299]) / 300, 1e-14))
    v = np.maximum(v, np.median(v) * 0.25)
    X = [(b[t] - (cb[t + 1] - cb[t - L + 1]) / L) / v for L in (10, 60, 300, 1800)]
    X += [(ls[t] - ls[t - L]) / v for L in (1, 2, 5, 15, 60, 300)]
    X += [(lp[t] - lp[t - L]) / v for L in (1, 2, 5, 15, 60)]
    X = np.clip(np.column_stack(X), -10, 10)
    tr, te = t < N // 2 - 86400, t > N // 2 + 86400
    for H in (10, 30, 60, 300):
        y = np.clip((ls[t + H] - ls[t]) / v, -30 * np.sqrt(H), 30 * np.sqrt(H))
        A = np.c_[np.ones(tr.sum()), X[tr]]
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        p = np.c_[np.ones(te.sum()), X[te]] @ beta
        out[f"own-signal OOS R2 {H}s %"] = 100 * (1 - np.mean((y[te] - p) ** 2) / np.mean((y[te] - y[tr].mean()) ** 2))
        out[f"P(no move over {H}s) %"] = 100 * np.mean(ls[t + H] == ls[t])
    # clock: second-of-window profile and remaining variance at e=285
    sec = np.arange(N) % 300
    a = np.abs(rs)
    prof = np.bincount(sec, weights=a, minlength=300) / np.bincount(sec, minlength=300)
    prof /= prof.mean()
    v2 = np.bincount(sec, weights=rs ** 2, minlength=300)
    v2 /= v2.sum()
    out["clock: |r| e=0 vs mean"] = prof[0]
    out["clock: |r| e=295-299 vs mean"] = prof[295:].mean()
    out["remaining var share at e=285 (uniform 0.05)"] = 1 - v2[:285].sum()
    # ties / atom at the window close
    T0 = np.arange(86400, N - 300, 300)
    out["P(close == open) %"] = 100 * np.mean(ls[T0 + 299] == d["lo"][T0])
    return out


if __name__ == "__main__":
    res = {a: stats(a) for a in ("ETH", "BTC")}
    print(f"{'statistic (2025)':45s} {'ETH':>12s} {'BTC':>12s}")
    for k in res["ETH"]:
        print(f"{k:45s} {res['ETH'][k]:12.4f} {res['BTC'][k]:12.4f}")
