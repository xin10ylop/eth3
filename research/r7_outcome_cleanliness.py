"""R7: why is ETH's 5-minute OUTCOME more predictable than BTC's, although its direction is not?

Hypothesis: ETH's path relative to the window open is "cleaner" - once it has moved, the side
of P0 it is on tends to stay decided - while BTC hovers around P0 (tiny relative tick, large
no-move atom, and from Feb 2026 a pull toward P0 in the final seconds).
Train (2025) and validation (Jan-Apr 2026) periods only; identical code for both assets.
"""
import os
from datetime import date

import numpy as np

TL = {"ETH": "/home/user/data_eth/timeline", "BTC": "/home/user/data/timeline"}
D0 = date(2025, 1, 1)


def run(asset, a, b):
    i0, i1 = (a - D0).days * 86400, (b - D0).days * 86400
    ls = np.log(np.load(os.path.join(TL[asset], "s_close.npy"), mmap_mode="r")[i0 - 3600:i1])
    lo = np.log(np.load(os.path.join(TL[asset], "s_open.npy"), mmap_mode="r")[i0 - 3600:i1])
    r = np.diff(ls, prepend=ls[0])
    c2 = np.concatenate([[0.0], np.cumsum(r ** 2)])
    T0 = np.arange(3600, len(ls) - 300, 300)
    v = np.maximum(np.sqrt((c2[T0] - c2[T0 - 3600]) / 3600), 1e-6)
    p0 = lo[T0]
    fin = ls[T0 + 299] - p0
    out = {}
    for e in (60, 150, 240, 285):
        x = ls[T0 + e - 1] - p0
        z = x / (v * np.sqrt(300 - e))
        out[f"P(final side == side at e={e}) %"] = 100 * np.mean(np.sign(fin) == np.sign(x))
        out[f"share 'undecided' |z|<0.25 at e={e} %"] = 100 * np.mean(np.abs(z) < 0.25)
    seg = np.stack([ls[T0 + k] - p0 for k in range(150, 300)])
    out["P0 crossings in 2nd half of window (mean)"] = np.mean(np.sum(np.diff(np.sign(seg), axis=0) != 0, axis=0))
    z300 = fin / (v * np.sqrt(300))
    out["|final move| / (1h sigma*sqrt(300)) median"] = np.median(np.abs(z300))
    out["P(close == open) %"] = 100 * np.mean(fin == 0)
    return out


if __name__ == "__main__":
    for nm, a, b in [("2025 (train)", date(2025, 1, 2), date(2026, 1, 1)), ("Jan-Apr 2026 (valid)", date(2026, 1, 2), date(2026, 5, 1))]:
        E, B = run("ETH", a, b), run("BTC", a, b)
        print(f"\n{nm:45s} {'ETH':>9s} {'BTC':>9s}")
        for k in E:
            print(f"{k:45s} {E[k]:9.3f} {B[k]:9.3f}")
