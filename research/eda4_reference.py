"""Is the 5-minute open an attractor/repeller? Robust sign statistics, 2025 only."""
import numpy as np
from datetime import date
from btcmodel.timeline import load_timeline
tl = load_timeline()
i1 = tl.index_of(date(2026, 1, 1))
ls = np.log(np.asarray(tl['s_close'][:i1])); rs = np.diff(ls, prepend=ls[0]); N = len(ls)
lo = np.log(np.asarray(tl['s_open'][:i1]))
cs = np.cumsum(np.r_[0, rs**2]); v = np.r_[np.full(299, np.nan), np.sqrt((cs[300:] - cs[:-300]) / 300)]
v = np.maximum(v, np.nanmedian(v) * 0.25)
T0 = np.arange(86400, N - 300, 300)
p0 = lo[T0]
print('P(close==open) = %.4f%%' % (100 * np.mean(ls[T0 + 299] == p0)))
for e in [30, 60, 150, 240, 280]:
    c = T0 + e - 1
    x = (ls[c] - p0) / (v[c] * np.sqrt(300 - e))
    inc = ls[T0 + 299] - ls[c]
    edges = [-np.inf, -1.5, -0.5, -0.1, -1e-12, 1e-12, 0.1, 0.5, 1.5, np.inf]
    g = np.digitize(x, edges) - 1
    row = []
    for j in range(len(edges) - 1):
        m = (g == j) & (inc != 0)
        n = m.sum()
        pu = (inc[m] > 0).mean() if n else np.nan
        row.append(f'{pu:.3f}({n//1000}k)')
    print(f'e={e:3d}  P(remaining inc > 0 | x bucket [<-1.5,-0.5,-0.1,<0,=0,>0,0.1,0.5,>1.5]):', ' '.join(row))
# effect of x==0 exactly: final outcome
for e in [60, 150, 240]:
    c = T0 + e - 1
    m = ls[c] == p0
    print(f'e={e}: frac windows with price exactly at P0: {m.mean()*100:.2f}%  -> P(close>=P0)={np.mean(ls[T0[m]+299] >= p0[m]):.3f}, P(close==P0)={np.mean(ls[T0[m]+299] == p0[m]):.3f}')
