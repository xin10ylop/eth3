"""Regime features must not change when the future is replaced by garbage."""
import numpy as np
from btcmodel.timeline import load_timeline
from btcmodel.regime import regime_features
from tests.test_no_lookahead import FakeTL


def test():
    tl = load_timeline()
    rng = np.random.default_rng(5)
    for _ in range(4):
        T0 = (rng.integers(10 * 86400, tl.n - 2 * 86400) // 300) * 300
        e = int(rng.choice([0, 3, 200]))
        a = regime_features(tl, np.array([T0]), np.array([e]))
        sub = FakeTL.__new__(FakeTL)
        # perturb only a local slice around the decision to keep memory small
        lo, hi = (T0 - 9 * 86400) // 86400 * 86400, T0 + 2 * 86400
        class Slice:
            n = hi - lo
            def __init__(s):
                s.d = {}
                for k in ["s_close", "s_open", "p_close", "s_volume", "p_buy", "p_sell"]:
                    x = np.array(tl[k][lo:hi], dtype=np.float64)
                    cut = T0 + e - lo
                    x[cut:] = x[cut:] * np.exp(rng.normal(0, 0.05, len(x) - cut))
                    s.d[k] = x
            def __getitem__(s, k):
                return s.d[k]
        b = regime_features(Slice(), np.array([T0 - lo]), np.array([e]))
        rel = np.abs(a.values - b.values) / (np.abs(a.values) + 1e-9)
        diff = np.nanmax(rel)
        assert diff < 1e-4, (T0, e, dict(zip(a.columns, rel[0])))
        print(f"ok T0={T0} e={e} max rel diff {diff:.2e}")


if __name__ == "__main__":
    test()
