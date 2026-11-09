"""EDA on the TRAINING period only (2025): data quality, stability of the clock
(seasonality) structure, tails of standardized increments, reference effects."""
import numpy as np
from datetime import date
from btcmodel.timeline import load_timeline
tl = load_timeline()
i1 = tl.index_of(date(2026, 1, 1))
ls = np.log(np.asarray(tl['s_close'][:i1])); rs = np.diff(ls, prepend=ls[0])
cnt = np.asarray(tl['s_count'][:i1])
pc = np.asarray(tl['p_close'][:i1]); lp = np.log(pc)
N = len(ls); t = np.arange(N)
print('2025 seconds', N, ' zero-trade spot seconds %.3f%%' % (100*(cnt == 0).mean()))
# data quality: longest run of zero-trade seconds, biggest 1s jumps, spot/perp divergence
z = (cnt == 0).astype(np.int8); d = np.diff(np.r_[0, z, 0]); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
L = en - st; o = np.argsort(-L)[:5]
print('longest zero-trade runs (s):', [(int(L[k]), str(np.datetime64('2025-01-01') + np.timedelta64(int(st[k]), 's'))) for k in o])
big = np.argsort(-np.abs(rs))[:5]
print('largest |1s ret| bps:', [(round(float(rs[k]*1e4), 1), str(np.datetime64('2025-01-01') + np.timedelta64(int(k), 's'))) for k in big])
b = (lp - ls) * 1e4
print('basis bps quantiles 0.001/0.5/0.999:', np.nanquantile(b, [0.001, 0.5, 0.999]))
# 1) clock structure: second-of-window profile, H1 vs H2 2025 (stability)
a = np.abs(rs)
half = N // 2
for nm, sl in [('H1', slice(0, half)), ('H2', slice(half, N))]:
    aa = a[sl]; tt = t[sl]
    sow = np.bincount(tt % 300, weights=aa, minlength=300) / np.bincount(tt % 300, minlength=300)
    sow /= sow.mean()
    print(f'{nm} second-of-window profile: e=0..4 {np.round(sow[:5],2)}, e=60..62 {np.round(sow[60:63],2)}, e=150 {sow[150]:.2f}, e=295..299 {np.round(sow[295:],2)}')
    # share of window variance still to come at elapsed e
    v2 = np.bincount(tt % 300, weights=aa**2, minlength=300); v2 /= v2.sum()
    rem = 1 - np.cumsum(np.r_[0, v2])[:-1]
    print(f'    remaining variance share at e=60: {rem[60]:.3f} (uniform 0.80), e=150: {rem[150]:.3f} (0.50), e=240: {rem[240]:.3f} (0.20), e=285: {rem[285]:.3f} (0.05)')
# 2) tails: increments standardized by trailing 300s vol
cs = np.cumsum(np.r_[0, rs**2]); v = np.sqrt(np.maximum((cs[300:] - cs[:-300]) / 300, 1e-14)); v = np.r_[np.full(299, np.nan), v]
for H in [1, 10, 30, 60, 300]:
    idx = np.arange(400, N - H, 37)
    u = (ls[idx + H] - ls[idx]) / (v[idx] * np.sqrt(H))
    u = u[np.isfinite(u)]
    k = np.mean((u - u.mean())**4) / u.var()**2
    print(f'H={H:3d}: sd(u)={u.std():.3f} excess kurtosis={k-3:6.2f}  P(|u|>3)={np.mean(np.abs(u)>3)*100:.2f}% (gauss 0.27%)  P(u==0)={np.mean(u==0)*100:.2f}%')
# 3) reference effect: does E[future increment] depend on x = distance from 5m open? (controls none)
T0 = np.arange(86400, N - 300, 300)
p0 = np.log(np.asarray(tl['s_open'][:i1]))[T0]
for e in [60, 150, 240]:
    c = T0 + e - 1
    x = (ls[c] - p0) / (v[c] * np.sqrt(300 - e))
    inc = (ls[T0 + 299] - ls[c]) / (v[c] * np.sqrt(300 - e))
    ok = np.isfinite(x) & np.isfinite(inc)
    x, inc = x[ok], inc[ok]
    qs = np.quantile(x, [0, .1, .3, .5, .7, .9, 1])
    bins = np.digitize(x, qs[1:-1])
    print(f'e={e}: mean standardized remaining increment by x-decile group:', np.round([inc[bins == j].mean() for j in range(6)], 3),
          ' corr(x, inc)=%.4f' % np.corrcoef(x, inc)[0, 1])
