"""Does BTC lead ETH at the 1-second scale?  2025 only (train period), descriptive.

Lead-lag correlations of 1s returns, and the out-of-sample (fit Jan-Jun, test Jul-Dec 2025)
incremental R^2 that BTC moves add to ETH's own lead-lag/basis signals for the next 30 s.
Not used by the frozen ETH replication; it measures whether a cross-asset input is worth it.
"""
import os

import numpy as np

from btcmodel.config import ASSET
from btcmodel.timeline import load_timeline

if ASSET == "BTC":
    raise SystemExit("run with BTC_ASSET=<alt>")
tl = load_timeline()
n = tl.index_of(__import__("datetime").date(2026, 1, 1))
BT = "/home/user/data/timeline"
if not os.path.exists(os.path.join(BT, "s_close.npy")):
    raise SystemExit("BTC timeline not available")


def ffill(a):
    m = np.isnan(a)
    i = np.where(~m, np.arange(len(a)), 0)
    np.maximum.accumulate(i, out=i)
    return a[i]


es = np.log(np.asarray(tl["s_close"][:n]))
ep = np.log(ffill(np.asarray(tl["p_close"][:n]).copy()))
bs = np.log(np.load(os.path.join(BT, "s_close.npy"), mmap_mode="r")[:n])
bp = np.log(ffill(np.load(os.path.join(BT, "p_close.npy"), mmap_mode="r")[:n].copy()))
r = {k: np.diff(v, prepend=v[0]) for k, v in dict(es=es, ep=ep, bs=bs, bp=bp).items()}
print(f"{ASSET} vs BTC, 2025, {n} seconds")
print("contemporaneous corr 1s: ETHspot~BTCspot %.3f  ETHperp~BTCperp %.3f" % (
    np.corrcoef(r["es"], r["bs"])[0, 1], np.corrcoef(r["ep"], r["bp"])[0, 1]))
for k in (1, 2, 3, 5):
    print(f"lag {k}s: corr(ETHspot[t+k], BTCperp[t]) {np.corrcoef(r['es'][k:], r['bp'][:-k])[0, 1]:+.4f}  "
          f"corr(ETHspot[t+k], ETHperp[t]) {np.corrcoef(r['es'][k:], r['ep'][:-k])[0, 1]:+.4f}  "
          f"corr(BTCspot[t+k], ETHperp[t]) {np.corrcoef(r['bs'][k:], r['ep'][:-k])[0, 1]:+.4f}")
# incremental OOS R^2 for the next 30 s of ETH spot
cs = lambda a: np.concatenate([[0.0], np.cumsum(a)])
c2 = cs(r["es"] ** 2)
t = np.arange(4000, n - 31, 7)
v = np.sqrt(np.maximum((c2[t + 1] - c2[t - 299]) / 300, 1e-14))
v = np.maximum(v, np.median(v) * 0.25)
bas = ep - es
cb = cs(bas)
own = [(bas[t] - (cb[t + 1] - cb[t - 59]) / 60) / v, (bas[t] - (cb[t + 1] - cb[t - 599]) / 600) / v]
for L in (1, 2, 5, 15, 60):
    own += [(es[t] - es[t - L]) / v, (ep[t] - ep[t - L]) / v]
cross = []
for L in (1, 2, 5, 15, 60):
    cross += [(bp[t] - bp[t - L]) / v, ((bp[t] - bp[t - L]) - (es[t] - es[t - L])) / v]
y = np.clip((es[t + 30] - es[t]) / v, -100, 100)
Xo = np.clip(np.column_stack(own), -10, 10)
Xc = np.clip(np.column_stack(own + cross), -10, 10)
tr, te = t < n // 2 - 86400, t > n // 2 + 86400


def oos(X):
    A = np.c_[np.ones(tr.sum()), X[tr]]
    b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    p = np.c_[np.ones(te.sum()), X[te]] @ b
    return 1 - np.mean((y[te] - p) ** 2) / np.mean((y[te] - y[tr].mean()) ** 2)


r_own, r_all = oos(Xo), oos(Xc)
print(f"next-30s OOS R^2 (Jul-Dec 2025, linear): own ETH signals {100 * r_own:.3f}%  "
      f"+ BTC signals {100 * r_all:.3f}%  (increment {100 * (r_all - r_own):+.3f} pp)")
