"""Is there 'pressure toward the reference' in the final seconds of 5-minute windows?
Train + validation periods only (holdout untouched).

For each window and a horizon of k seconds ending at the window close, measure the move
TOWARD P0:  m = -sign(x_start) * R / sigma, where x_start = log(S/P0) at the start of the
k-second interval and R the k-second return.  Under a martingale E[m] = 0.
Placebo: identical statistic for k-second intervals ending mid-window (e = 200), measured
relative to the same P0 (the market should not care about P0 mid-window either)."""
import numpy as np
from datetime import date
from btcmodel.timeline import load_timeline

tl = load_timeline()
i1 = tl.index_of(date(2026, 5, 1))
ls = np.log(np.asarray(tl["s_close"][:i1]))
lo = np.log(np.asarray(tl["s_open"][:i1]))
rs = np.diff(ls, prepend=ls[0])
cs = np.cumsum(np.r_[0, rs ** 2])
T0 = np.arange(86400, i1 - 300, 300)
vol = np.sqrt((cs[T0] - cs[T0 - 3600]) / 3600)          # trailing 1h per-second vol, known at T0
vol = np.maximum(vol, np.median(vol) * 0.2)
p0 = lo[T0]
mon = (np.datetime64("2025-01-01") + (T0 // 86400).astype("timedelta64[D]")).astype("datetime64[M]")


def stat(end_e, k):
    a = T0 + end_e - k - 1          # bar index of price at interval start
    b = T0 + end_e - 1              # bar index of price at interval end
    x = ls[a] - p0
    R = ls[b] - ls[a]
    s = vol * np.sqrt(k)
    near = (np.abs(x) > 0) & (np.abs(x) < 1.5 * s)      # close enough that crossing is plausible
    m = -np.sign(x) * R / s
    cross = (np.sign(ls[b] - p0) != np.sign(x)) & (ls[b] != p0)
    return m, near, cross


print("E[move toward P0] in sigma units (t-stat), windows with |x| < 1.5 sigma_k at interval start")
for k in [2, 5, 10]:
    m_end, n_end, c_end = stat(300, k)      # interval ending at the window close
    m_mid, n_mid, c_mid = stat(200, k)      # placebo: ending mid-window
    print(f"\n-- last {k}s of the window vs placebo (interval ending at e=200) --")
    for mm in np.unique(mon):
        g = mon == mm
        a, b = m_end[g & n_end], m_mid[g & n_mid]
        ta = a.mean() / (a.std() / np.sqrt(len(a)))
        tb = b.mean() / (b.std() / np.sqrt(len(b)))
        print(f"{mm}: close {a.mean():+.4f} (t={ta:+5.1f}, cross {c_end[g & n_end].mean() * 100:4.1f}%) | "
              f"placebo {b.mean():+.4f} (t={tb:+5.1f}, cross {c_mid[g & n_mid].mean() * 100:4.1f}%)")
