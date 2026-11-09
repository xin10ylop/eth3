"""R4: is ETH's reversion after large moves tied to the 5-minute open (a reference effect) or
generic short-term mean reversion?  2025 only, ETH and BTC side by side.

Reference version : z = move from P0 (e=0) to e=150, forward = next 90 s (to e=240).
Placebo version   : z = move from e=75 to e=225 (anchor is NOT a reference), forward = next 90 s.
Same lengths, same volatility normalization. If the reversion is equally strong in the placebo,
it is generic mean reversion (which return features can learn), not a pull toward P0.
"""
import os
from datetime import date

import numpy as np

TL = {"ETH": "/home/user/data_eth/timeline", "BTC": "/home/user/data/timeline"}
N = (date(2026, 1, 1) - date(2025, 1, 1)).days * 86400


def run(asset):
    ls = np.log(np.load(os.path.join(TL[asset], "s_close.npy"), mmap_mode="r")[:N])
    lo = np.log(np.load(os.path.join(TL[asset], "s_open.npy"), mmap_mode="r")[:N])
    r = np.diff(ls, prepend=ls[0])
    c2 = np.concatenate([[0.0], np.cumsum(r ** 2)])
    T0 = np.arange(86400, N - 700, 300)
    vol = np.sqrt(np.maximum((c2[T0] - c2[T0 - 3600]) / 3600, 1e-14))
    vol = np.maximum(vol, np.median(vol) * 0.2)
    out = {}
    for name, a, b in [("reference (P0 -> e150)", None, 150), ("placebo (e75 -> e225)", 75, 225)]:
        start = lo[T0] if a is None else ls[T0 + a - 1]
        z = (ls[T0 + b - 1] - start) / (vol * np.sqrt(150))
        f = ls[T0 + b - 1 + 90] - ls[T0 + b - 1]
        row = []
        for lo_, hi_ in [(-np.inf, -1.5), (-1.5, -0.5), (0.5, 1.5), (1.5, np.inf)]:
            m = (z > lo_) & (z <= hi_) & (f != 0)
            row.append(f"{(f[m] > 0).mean():.3f}({m.sum() // 1000}k)")
        m_hi, m_lo = (z > 1.5) & (f != 0), (z < -1.5) & (f != 0)
        spread = (f[m_lo] > 0).mean() - (f[m_hi] > 0).mean()
        se = np.sqrt(0.25 / m_lo.sum() + 0.25 / m_hi.sum())
        out[name] = (row, spread, spread / se)
    return out


if __name__ == "__main__":
    print("P(next 90s up | z bucket <-1.5, -1.5..-0.5, 0.5..1.5, >1.5);  reversion spread = P(up|z<-1.5) - P(up|z>1.5)")
    for asset in ("ETH", "BTC"):
        for name, (row, sp, t) in run(asset).items():
            print(f"{asset} {name:24s} {'  '.join(row)}   spread {sp:+.3f} (t={t:+.1f})")
