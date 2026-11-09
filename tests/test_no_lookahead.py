"""Lookahead test: features at decision (T0, e) must be identical when every bar
with index >= T0 + e is replaced by garbage (except P0 = open of bar T0, which is
known at e = 0 by definition).  Targets, by contrast, must change."""
import numpy as np
from btcmodel.timeline import load_timeline
from btcmodel.features import Base, FeatureEngine, Seasonality


class FakeTL:
    def __init__(self, tl, cut, keep_open_at=None, seed=0):
        self.n, self.present, self.t0_epoch = tl.n, tl.present, tl.t0_epoch
        rng = np.random.default_rng(seed)
        self.d = {}
        for k in ["s_open", "s_high", "s_low", "s_close", "s_volume", "s_qvol", "s_tb_base", "s_count",
                  "p_open", "p_high", "p_low", "p_close", "p_buy", "p_sell", "p_ntr", "p_maxq", "p_vwap"]:
            a = np.array(tl[k]).copy()
            p0 = a[keep_open_at] if keep_open_at is not None else None
            if a.dtype.kind == "f":
                a[cut:] = a[cut:] * np.exp(rng.normal(0, 0.05, len(a) - cut))
            else:
                a[cut:] = rng.integers(0, 1000, len(a) - cut)
            if k == "s_open" and keep_open_at is not None:
                a[keep_open_at] = p0
            self.d[k] = a

    def __getitem__(self, k):
        return self.d[k]


def test_no_lookahead():
    tl = load_timeline()
    base = Base(tl)
    season = Seasonality.fit(np.abs(base.rs), base.how_hour, base.present)
    fe = FeatureEngine(base, season)
    rng = np.random.default_rng(1)
    for trial in range(6):
        T0 = (rng.integers(2 * 86400, tl.n - 86400) // 300) * 300
        e = int(rng.choice([0, 1, 7, 150, 299]))
        X, aux = fe.compute(np.array([T0]), np.array([e]))
        tgt = fe.targets(aux, None)
        ftl = FakeTL(tl, T0 + e, keep_open_at=T0)
        fb = Base(ftl)
        fe2 = FeatureEngine(fb, season)
        X2, aux2 = fe2.compute(np.array([T0]), np.array([e]))
        tgt2 = fe2.targets(aux2, None)
        diff = (X - X2).abs().max()
        bad = diff[diff > 1e-5]
        assert len(bad) == 0, f"lookahead at T0={T0} e={e}: {bad.to_dict()}"
        assert abs(tgt["y"][0] - tgt2["y"][0]) > 0, "target should depend on the future"
        print(f"ok T0={T0} e={e}")


if __name__ == "__main__":
    test_no_lookahead()
