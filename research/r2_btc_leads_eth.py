"""R2: does BTC lead ETH, and how much predictive power does it ADD to ETH's own signals?

2025 only. Linear probes (fit Jan-Jun, test Jul-Dec 2025, 1-day gaps), target = ETH spot
return over the next H seconds in units of ETH's trailing 1s vol.
  own   : ETH basis deviations + ETH spot/perp returns (what the BTC recipe already uses)
  +btc  : + BTC perp/spot returns and the "ETH lagging BTC" gap  (ETH ret - beta*BTC ret)
Reverse direction (ETH -> BTC) as a control: if BTC is the price-discovery leader, ETH
should add much less to BTC than BTC adds to ETH.
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
    ls = np.log(np.asarray(g("s_close")))
    lp = np.log(ffill(np.asarray(g("p_close")).copy()))
    bad = ~np.isfinite(lp)
    lp[bad] = ls[bad]
    return ls, lp


cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])


def probe(tgt, oth, name_t, name_o):
    ts, tp = tgt
    os_, op = oth
    r = np.diff(ts, prepend=ts[0])
    c2 = cs(r ** 2)
    t = np.arange(90000, N - 301, 7)
    v = np.sqrt(np.maximum((c2[t + 1] - c2[t - 299]) / 300, 1e-14))
    v = np.maximum(v, np.median(v) * 0.25)
    ro = np.diff(os_, prepend=os_[0])
    co2 = cs(ro ** 2)
    vo = np.sqrt(np.maximum((co2[t + 1] - co2[t - 299]) / 300, 1e-14))
    vo = np.maximum(vo, np.median(vo) * 0.25)
    # rolling 1h beta of target 1s returns on other-asset 1s returns
    cxy, cxx = cs(r * ro), cs(ro * ro)
    beta = (cxy[t + 1] - cxy[t - 3599]) / np.maximum(cxx[t + 1] - cxx[t - 3599], 1e-18)
    b = tp - ts
    cb = cs(b)
    own = [(b[t] - (cb[t + 1] - cb[t - L + 1]) / L) / v for L in (10, 60, 300, 1800)]
    own += [(ts[t] - ts[t - L]) / v for L in (1, 2, 5, 15, 60, 300)]
    own += [(tp[t] - tp[t - L]) / v for L in (1, 2, 5, 15, 60)]
    cross = []
    for L in (1, 2, 3, 5, 10, 30, 60, 300):
        cross.append((op[t] - op[t - L]) / vo)                                   # other perp move
        cross.append(((ts[t] - ts[t - L]) - beta * (op[t] - op[t - L])) / v)     # target lagging gap
    for L in (1, 5, 30):
        cross.append((os_[t] - os_[t - L]) / vo)                                 # other spot move
    Xo = np.clip(np.column_stack(own), -10, 10)
    Xa = np.clip(np.column_stack(own + cross), -10, 10)
    tr, te = t < N // 2 - 86400, t > N // 2 + 86400
    print(f"\n{name_o} -> {name_t}: median 1h beta of {name_t} on {name_o} = {np.median(beta):.3f}")
    for H in (5, 10, 15, 30, 60, 300):
        y = np.clip((ts[t + H] - ts[t]) / v, -30 * np.sqrt(H), 30 * np.sqrt(H))
        res = []
        for X in (Xo, Xa):
            A = np.c_[np.ones(tr.sum()), X[tr]]
            bb = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            p = np.c_[np.ones(te.sum()), X[te]] @ bb
            res.append(1 - np.mean((y[te] - p) ** 2) / np.mean((y[te] - y[tr].mean()) ** 2))
        print(f"  H={H:4d}s  OOS R2 own {100 * res[0]:6.3f}%   own+{name_o} {100 * res[1]:6.3f}%   "
              f"gain {100 * (res[1] - res[0]):+6.3f} pp  (x{res[1] / max(res[0], 1e-9):.2f})")


if __name__ == "__main__":
    eth, btc = load("ETH"), load("BTC")
    re, rb = np.diff(eth[0], prepend=eth[0][0]), np.diff(btc[0], prepend=btc[0][0])
    rpe, rpb = np.diff(eth[1], prepend=eth[1][0]), np.diff(btc[1], prepend=btc[1][0])
    print("contemporaneous 1s corr: ETH spot~BTC spot %.3f, ETH perp~BTC perp %.3f" % (
        np.corrcoef(re, rb)[0, 1], np.corrcoef(rpe, rpb)[0, 1]))
    for k in (1, 2, 3, 5):
        print(f"lag {k}s: corr(ETHspot[t+k], BTCperp[t]) {np.corrcoef(re[k:], rpb[:-k])[0, 1]:+.4f} | "
              f"corr(ETHspot[t+k], ETHperp[t]) {np.corrcoef(re[k:], rpe[:-k])[0, 1]:+.4f} | "
              f"corr(BTCspot[t+k], ETHperp[t]) {np.corrcoef(rb[k:], rpe[:-k])[0, 1]:+.4f}")
    probe(eth, btc, "ETH", "BTC")
    probe(btc, eth, "BTC", "ETH")
