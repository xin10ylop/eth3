"""Causal feature engine and reference-relative targets.

Conventions (critical, read once):
  * A 5-minute window starts at T0 (index divisible by 300).  P0 = spot open of
    second T0 = the first trade at/after the boundary = Binance 5m candle open.
    P0 stays fixed for the whole window.
  * Decision at elapsed e in [0, 299] happens at wall time T0 + e (+ epsilon).
    Known information: every 1s bar with index < T0 + e, plus P0 itself.
    Current price S_e = P0 if e == 0 else close of bar T0 + e - 1.
  * Horizon h in {30, 60}: target price S_{e+h} = close of bar T0 + e + h - 1
    (valid only when e + h <= 300, i.e. it never leaves the window).
  * "Close" horizon: target S_300 = close of bar T0 + 299 = the 5m candle close,
    remaining time tau = 300 - e.
  * Path extremes over the horizon use 1s highs/lows of bars T0+e .. T0+e+h-1,
    i.e. any trade in the future part of the horizon.

All features below are functions of bars with index <= c = T0 + e - 1 (and P0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

W = 300  # window length (s)

# ----------------------------------------------------------------------------
# rolling helpers on a 1-D timeline
# ----------------------------------------------------------------------------


class Roll:
    """Trailing sums over (c-L, c] using one float64 cumsum per series."""

    def __init__(self, a: np.ndarray):
        self.cs = np.concatenate([[0.0], np.cumsum(np.asarray(a, dtype=np.float64))])

    def sum(self, c: np.ndarray, L: int) -> np.ndarray:
        lo = np.maximum(c + 1 - L, 0)
        return self.cs[c + 1] - self.cs[lo]


def ffill_nan(a):
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = np.where(~mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


# ----------------------------------------------------------------------------
# time / seasonality
# ----------------------------------------------------------------------------


def et_hour_of_week(t0_epoch: int, n: int) -> np.ndarray:
    """New-York-time hour-of-week (0..167, Monday 00h = 0) for every hour of the timeline.

    US macro releases and the equity open are fixed in Eastern time, so DST-aware
    clocks align them across the year (UTC clocks would smear them over 2 hours)."""
    nh = n // 3600 + 1
    idx = pd.to_datetime(t0_epoch + 3600 * np.arange(nh), unit="s", utc=True).tz_convert("America/New_York")
    return (idx.dayofweek.values * 24 + idx.hour.values).astype(np.int16)


class Seasonality:
    """Multiplicative intraday activity profile estimated on TRAINING data only.

    sigma(t) ~ how(hour-of-week, ET) * soh(second-of-hour)
    Estimated from mean absolute 1s spot returns (robust; 1 bin = >8k observations
    per year for soh)."""

    def __init__(self, how: np.ndarray, soh: np.ndarray):
        self.how = how.astype(np.float64)
        self.soh = soh.astype(np.float64)
        # cumulative soh over two hours so that integrals across the hour wrap work
        self._soh_cs = np.concatenate([[0.0], np.cumsum(np.r_[self.soh, self.soh])])

    @staticmethod
    def fit(abs_r: np.ndarray, how_idx_per_hour: np.ndarray, train_mask: np.ndarray) -> "Seasonality":
        n = len(abs_r)
        t = np.arange(n)
        m = train_mask & np.isfinite(abs_r)
        soh = np.bincount(t[m] % 3600, weights=abs_r[m], minlength=3600) / np.maximum(
            np.bincount(t[m] % 3600, minlength=3600), 1)
        soh /= soh.mean()
        hw = how_idx_per_hour[t[m] // 3600]
        how = np.bincount(hw, weights=abs_r[m] / soh[t[m] % 3600], minlength=168) / np.maximum(
            np.bincount(hw, minlength=168), 1)
        how /= how.mean()
        return Seasonality(how, soh)

    def point(self, i: np.ndarray, how_idx_per_hour: np.ndarray) -> np.ndarray:
        return self.how[how_idx_per_hour[i // 3600]] * self.soh[i % 3600]

    def integrated(self, start: np.ndarray, h: np.ndarray, how_idx_per_hour: np.ndarray) -> np.ndarray:
        """Mean *squared* seasonal factor over bars start .. start+h-1 (variance units).

        Uses soh**2 so that the result scales a per-second variance; hour-of-week is
        taken at the start (a 5-minute horizon never spans more than one hour change)."""
        s2 = self.soh ** 2
        cs = np.concatenate([[0.0], np.cumsum(np.r_[s2, s2])])
        a = start % 3600
        tot = cs[a + h] - cs[a]
        return self.how[how_idx_per_hour[start // 3600]] ** 2 * tot / np.maximum(h, 1)


# ----------------------------------------------------------------------------
# numba: within-window path statistics relative to P0 (causal)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _window_path_stats(ls, lhi, llo, lopen, n):
    """For every bar c: stats of the current window *up to and including* bar c.

    Returns arrays aligned with bar index c:
      wmax  = max log-high of window bars T0..c minus log P0
      wmin  = min log-low  ... minus log P0
      ncross= number of sign changes of (close - P0) over bars T0..c
      fup   = fraction of bars T0..c whose close >= P0
      tsc   = seconds since the last sign change (or since T0)
    """
    wmax = np.empty(n, np.float64)
    wmin = np.empty(n, np.float64)
    ncross = np.empty(n, np.float32)
    fup = np.empty(n, np.float32)
    tsc = np.empty(n, np.float32)
    for T0 in range(0, n - W + 1, W):
        p0 = lopen[T0]
        mx = -1e9
        mn = 1e9
        nc = 0
        nup = 0
        last = 0
        lastchg = T0
        for c in range(T0, T0 + W):
            d = ls[c] - p0
            if lhi[c] - p0 > mx:
                mx = lhi[c] - p0
            if llo[c] - p0 < mn:
                mn = llo[c] - p0
            s = 1 if d >= 0 else -1
            if c > T0 and s != last:
                nc += 1
                lastchg = c
            last = s
            if s > 0:
                nup += 1
            wmax[c] = mx
            wmin[c] = mn
            ncross[c] = nc
            fup[c] = nup / (c - T0 + 1)
            tsc[c] = c - lastchg
    return wmax, wmin, ncross, fup, tsc


@njit(cache=True)
def _future_extremes(lhi, llo, starts, h):
    """max log-high and min log-low over bars starts[k] .. starts[k]+h[k]-1."""
    m = len(starts)
    mx = np.empty(m)
    mn = np.empty(m)
    for k in range(m):
        a = starts[k]
        b = a + h[k]
        x = -1e18
        y = 1e18
        for j in range(a, b):
            if lhi[j] > x:
                x = lhi[j]
            if llo[j] < y:
                y = llo[j]
        mx[k] = x
        mn[k] = y
    return mx, mn


@njit(cache=True)
def _first_passage(lhi, llo, starts, h, up_lvl, dn_lvl):
    """Outcome over bars starts..starts+h-1: 1 = up level touched first, -1 = down first,
    0 = neither, 2 = both inside the same 1s bar (ambiguous)."""
    m = len(starts)
    out = np.zeros(m, np.int8)
    for k in range(m):
        for j in range(starts[k], starts[k] + h[k]):
            u = lhi[j] >= up_lvl[k]
            d = llo[j] <= dn_lvl[k]
            if u and d:
                out[k] = 2
                break
            if u:
                out[k] = 1
                break
            if d:
                out[k] = -1
                break
    return out


# ----------------------------------------------------------------------------
# base series
# ----------------------------------------------------------------------------


class Base:
    """Per-second base series derived from the timeline (all causal)."""

    def __init__(self, tl, a: int = 0, b: int | None = None):
        """Load slice [a, b) of a timeline-like source. Indices are local.

        The source needs __getitem__(field), n, t0_epoch and either `present` (per day) or
        `present_sec` (per second).  The slice start must be hour-aligned in epoch time so
        that local index % 3600 is the second-of-hour and windows start at multiples of 300."""
        b = tl.n if b is None else min(b, tl.n)
        assert (tl.t0_epoch + a) % 3600 == 0
        self.offset = a
        self.n = b - a
        g = lambda k: np.asarray(tl[k][a:b])
        self.ls = np.log(g("s_close"))
        self.lhi = np.log(g("s_high"))
        self.llo = np.log(g("s_low"))
        self.lopen = np.log(g("s_open"))
        lp = np.log(ffill_nan(g("p_close").copy()))
        # before the first perp trade fall back to spot (only matters at the very start)
        bad = ~np.isfinite(lp)
        lp[bad] = self.ls[bad]
        self.lp = lp
        self.rs = np.diff(self.ls, prepend=self.ls[0])
        self.rp = np.diff(self.lp, prepend=self.lp[0])
        self.basis = self.lp - self.ls
        vol = g("s_volume").astype(np.float32)
        tb = g("s_tb_base").astype(np.float32)
        self.s_buy, self.s_sell = tb, np.maximum(vol - tb, 0.0)
        self.p_buy = g("p_buy").astype(np.float32)
        self.p_sell = g("p_sell").astype(np.float32)
        self.s_cnt = g("s_count").astype(np.float32)
        self.p_cnt = g("p_ntr").astype(np.float32)
        self.p_maxq = g("p_maxq").astype(np.float32)
        self.hl = (self.lhi - self.llo).astype(np.float32)
        # data-quality mask: seconds belonging to days that are present
        if hasattr(tl, "present_sec"):
            self.present = np.asarray(tl.present_sec[a:b], dtype=bool)
        else:
            self.present = np.repeat(tl.present, 86400)[a:b]
        (self.wmax, self.wmin, self.ncross, self.fup, self.tsc) = _window_path_stats(
            self.ls, self.lhi, self.llo, self.lopen, self.n)
        self.t0_epoch = tl.t0_epoch + a
        self.how_hour = et_hour_of_week(self.t0_epoch, self.n)


# ----------------------------------------------------------------------------
# decision points
# ----------------------------------------------------------------------------


def decision_points(base: Base, i_from: int, i_to: int, stride: int, rng=None, include_e0=True):
    """Window starts T0 in [i_from, i_to) and elapsed seconds e.

    stride>1 subsamples elapsed seconds; with rng a random phase per window is used
    (so every e is represented across windows) and e = 0 is always included."""
    T0s = np.arange((i_from + W - 1) // W * W, i_to - W + 1, W)
    # need full window inside present days and one day of history
    ok = base.present[T0s] & base.present[T0s + W - 1] & (T0s >= 86400)
    T0s = T0s[ok]
    es = []
    tt = []
    for ph in range(stride):
        sel = T0s if rng is None else T0s[rng.integers(0, stride, len(T0s)) == ph]
        if rng is None and ph > 0:
            break
        e = np.arange(ph, W, stride)
        es.append(np.tile(e, len(sel)))
        tt.append(np.repeat(sel, len(e)))
    T0 = np.concatenate(tt)
    e = np.concatenate(es)
    if include_e0:
        T0 = np.concatenate([T0, T0s])
        e = np.concatenate([e, np.zeros(len(T0s), dtype=e.dtype)])
        key = T0 * W + e
        _, u = np.unique(key, return_index=True)
        T0, e = T0[u], e[u]
    o = np.lexsort((e, T0))
    return T0[o].astype(np.int64), e[o].astype(np.int64)


# ----------------------------------------------------------------------------
# feature computation
# ----------------------------------------------------------------------------


class FeatureEngine:
    def __init__(self, base: Base, season: Seasonality):
        self.b = base
        self.season = season
        b = base
        self.R = {
            "rs2": Roll(b.rs ** 2), "rp2": Roll(b.rp ** 2), "ars": Roll(np.abs(b.rs)),
            "hl2": Roll(b.hl ** 2), "bpv": Roll(np.abs(b.rs) * np.abs(np.r_[0.0, b.rs[:-1]])),
            "basis": Roll(b.basis), "sbuy": Roll(b.s_buy), "ssell": Roll(b.s_sell),
            "pbuy": Roll(b.p_buy), "psell": Roll(b.p_sell), "scnt": Roll(b.s_cnt),
            "pcnt": Roll(b.p_cnt), "rs": Roll(b.rs),
            "season2": Roll(season.point(np.arange(b.n), b.how_hour) ** 2),
        }

    def compute(self, T0: np.ndarray, e: np.ndarray) -> tuple[pd.DataFrame, dict]:
        b, R = self.b, self.R
        p0 = b.lopen[T0]
        c = T0 + e - 1                       # last fully known bar
        cur = np.where(e == 0, p0, b.ls[np.maximum(c, 0)])
        # at e == 0 every rolling stat uses bars <= T0-1 (c = T0-1): still causal
        F = {}
        f32 = lambda a: np.asarray(a, dtype=np.float32)

        # ---------------- volatility / activity ----------------
        rv = {}
        for L in [10, 30, 60, 120, 300, 900, 3600, 14400, 86400]:
            rv[L] = R["rs2"].sum(c, L) / L
            F[f"lrv_{L}"] = f32(np.log(rv[L] + 1e-14))
        # seasonally adjusted RV (realized / expected-by-clock)
        for L in [60, 300, 3600]:
            seas = R["season2"].sum(c, L) / L
            F[f"lrvsa_{L}"] = f32(np.log(rv[L] + 1e-14) - np.log(seas))
        for L in [30, 300]:
            F[f"lrvp_{L}"] = f32(np.log(R["rp2"].sum(c, L) / L + 1e-14))
        for L in [60, 300, 3600]:
            F[f"lpk_{L}"] = f32(np.log(R["hl2"].sum(c, L) / L + 1e-14))
        for L in [300, 3600]:
            bp = R["bpv"].sum(c, L) * (np.pi / 2) / L
            F[f"jump_{L}"] = f32(np.log(rv[L] + 1e-14) - np.log(bp + 1e-14))
        for L in [10, 60, 300, 3600]:
            F[f"lcnt_{L}"] = f32(np.log((R["scnt"].sum(c, L) + R["pcnt"].sum(c, L)) / L + 1e-3))
            F[f"lvol_{L}"] = f32(np.log((R["sbuy"].sum(c, L) + R["ssell"].sum(c, L) +
                                         R["pbuy"].sum(c, L) + R["psell"].sum(c, L)) / L + 1e-6))
        F["lcnt_ratio"] = F["lcnt_60"] - F["lcnt_3600"]
        F["lvol_ratio"] = F["lvol_10"] - F["lvol_3600"]
        F["seas_now"] = f32(self.season.point(np.maximum(c, 0), b.how_hour))
        F["maxq_10"] = f32(np.log1p(np.max(np.stack([b.p_maxq[np.maximum(c - k, 0)] for k in range(10)]), 0)))

        # vol unit for scaling directional features (per-second sd, floored)
        v = np.sqrt(np.maximum(rv[300], 0.25 * rv[3600]))
        v = np.maximum(v, 2e-6)

        # ---------------- direction / flow ----------------
        for L in [1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 900, 3600]:
            F[f"r_{L}"] = f32((cur - b.ls[np.maximum(c - L, 0)]) / (v * np.sqrt(L)))
        lpc = b.lp[np.maximum(c, 0)]
        for L in [1, 2, 5, 10, 30, 60, 300]:
            dp = lpc - b.lp[np.maximum(c - L, 0)]
            ds = b.ls[np.maximum(c, 0)] - b.ls[np.maximum(c - L, 0)]
            F[f"pls_{L}"] = f32((dp - ds) / v)           # perp move not yet in spot
        bas = b.basis[np.maximum(c, 0)]
        for L in [10, 60, 300, 1800]:
            F[f"bdev_{L}"] = f32((bas - R["basis"].sum(c, L) / L) / v)
        for L in [1, 5, 10, 30, 60, 300, 900]:
            sb, ss = R["sbuy"].sum(c, L), R["ssell"].sum(c, L)
            pb, ps = R["pbuy"].sum(c, L), R["psell"].sum(c, L)
            F[f"simb_{L}"] = f32((sb - ss) / (sb + ss + 1e-9))
            F[f"pimb_{L}"] = f32((pb - ps) / (pb + ps + 1e-9))
        avgv = (R["sbuy"].sum(c, 3600) + R["ssell"].sum(c, 3600) + R["pbuy"].sum(c, 3600) +
                R["psell"].sum(c, 3600)) / 3600 + 1e-9
        for L in [10, 60, 300]:
            net = (R["sbuy"].sum(c, L) - R["ssell"].sum(c, L) + R["pbuy"].sum(c, L) - R["psell"].sum(c, L))
            F[f"netflow_{L}"] = f32(net / (avgv * L))
        # realized autocorrelation proxy (lag-1) over 300s: partial-adjustment strength
        F["rskew_300"] = f32(R["rs"].sum(c, 300) / (np.sqrt(rv[300] * 300) + 1e-12))
        # position inside recent range
        for L in [60, 300]:
            hi = np.max(np.stack([b.lhi[np.maximum(c - k, 0)] for k in range(0, L, max(1, L // 30))]), 0)
            lo = np.min(np.stack([b.llo[np.maximum(c - k, 0)] for k in range(0, L, max(1, L // 30))]), 0)
            F[f"rngpos_{L}"] = f32((cur - lo) / np.maximum(hi - lo, 1e-9))

        # ---------------- window / reference ----------------
        x = cur - p0
        F["e"] = f32(e)
        F["x_bps"] = f32(x * 1e4)
        F["x_v"] = f32(x / (v * np.sqrt(np.maximum(W - e, 1))))
        cc = np.maximum(c, 0)
        has = e > 0
        F["wmax_v"] = f32(np.where(has, b.wmax[cc], 0.0) / v / np.sqrt(np.maximum(e, 1)))
        F["wmin_v"] = f32(np.where(has, b.wmin[cc], 0.0) / v / np.sqrt(np.maximum(e, 1)))
        F["ncross"] = f32(np.where(has, b.ncross[cc], 0))
        F["fup"] = f32(np.where(has, b.fup[cc], 0.5))
        F["tsc"] = f32(np.where(has, b.tsc[cc], 0))
        # previous windows
        F["prev_ret_v"] = f32((b.ls[T0 - 1] - b.lopen[T0 - W]) / (v * np.sqrt(W)))
        F["prev2_ret_v"] = f32((b.ls[T0 - W - 1] - b.lopen[T0 - 2 * W]) / (v * np.sqrt(W)))
        # multi-timeframe candle opens (15m, 1h, 4h, UTC day)
        for P, nm in [(900, "15m"), (3600, "1h"), (14400, "4h"), (86400, "1d")]:
            # candle boundaries in absolute (epoch) time, so any buffer alignment works
            st = (b.t0_epoch + T0) // P * P - b.t0_epoch
            el = np.maximum((T0 + e) - st, 1)
            F[f"mtf_{nm}"] = f32((cur - b.lopen[st]) / (v * np.sqrt(el)))
        # round numbers ($1000 / $100 grid), signed distance in vol units over the window
        price = np.exp(cur)
        for G in [1000.0, 100.0]:
            d = price - np.round(price / G) * G
            F[f"round_{int(G)}"] = f32(d / price / (v * np.sqrt(W)))
        # time
        sec_abs = T0 + e
        how = b.how_hour[sec_abs // 3600]
        F["how"] = how.astype(np.float32)
        F["hod_et"] = (how % 24).astype(np.float32)
        F["dow_et"] = (how // 24).astype(np.float32)
        F["moh"] = f32((sec_abs % 3600) // 60)
        F["som"] = f32(sec_abs % 60)
        F["w_in_hour"] = f32((T0 % 3600) // W)

        aux = {"T0": T0, "e": e, "c": c, "p0": p0, "cur": cur, "v": v}
        return pd.DataFrame(F), aux

    # ---------------- targets ----------------
    def targets(self, aux: dict, h: int | None) -> dict:
        """h = 30/60, or None for the window close."""
        b = self.b
        T0, e, cur, p0 = aux["T0"], aux["e"], aux["cur"], aux["p0"]
        H = (W - e) if h is None else np.full(len(e), h)
        valid = (e + H) <= W
        end = np.minimum(T0 + e + H - 1, len(b.ls) - 1)
        y = b.ls[end] - p0                       # log(S_target / P0)
        start = T0 + e
        mx, mn = _future_extremes(b.lhi, b.llo, start, H.astype(np.int64))
        return {
            "H": H, "valid": valid,
            "y": y,                              # reference-relative terminal log price
            "inc": y - (cur - p0),               # increment from now
            "up": (y >= 0).astype(np.int8),      # S_target >= P0  (ties -> up)
            "mx": mx - p0, "mn": mn - p0,        # future path extremes rel. P0
            "season_int": self.season.integrated(start, H.astype(np.int64), b.how_hour),
        }

    def first_passage(self, aux, H, k_bps):
        b = self.b
        lp0 = aux["p0"]
        up = lp0 + np.log1p(k_bps * 1e-4)
        dn = lp0 + np.log1p(-k_bps * 1e-4)
        start = aux["T0"] + aux["e"]
        return _first_passage(b.lhi, b.llo, start, H.astype(np.int64), up, dn)
