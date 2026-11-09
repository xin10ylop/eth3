"""Continuous per-second timeline built from the daily compact files.

All arrays share one index: i = seconds since `t0` (UTC midnight of the first day).
Prices are forward-filled *causally* (a second without trades carries the last
traded price), flows are zero-filled.  Nothing here looks at the future.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import date

import numpy as np

from .data import DATA_DIR, DAY_DIR, SEC_PER_DAY

TL_DIR = os.path.join(DATA_DIR, "timeline")

# (name in day file, dtype in timeline)
FIELDS = {
    "s_open": np.float64, "s_high": np.float64, "s_low": np.float64, "s_close": np.float64,
    "s_volume": np.float32, "s_qvol": np.float32, "s_tb_base": np.float32, "s_count": np.int32,
    "p_open": np.float64, "p_high": np.float64, "p_low": np.float64, "p_close": np.float64,
    "p_buy": np.float32, "p_sell": np.float32, "p_ntr": np.int32, "p_maxq": np.float32,
    "p_vwap": np.float64,
}


def _ffill(a: np.ndarray) -> np.ndarray:
    """Causal forward fill of NaNs."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = np.where(~mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    # leading NaNs stay NaN
    return out


def build_timeline(start: date | None = None, end: date | None = None) -> None:
    files = sorted(glob.glob(os.path.join(DAY_DIR, "*.npz")))
    days = [date.fromisoformat(os.path.basename(f)[:10]) for f in files]
    if start:
        files = [f for f, d in zip(files, days) if d >= start]
        days = [d for d in days if d >= start]
    if end:
        files = [f for f, d in zip(files, days) if d <= end]
        days = [d for d in days if d <= end]
    d0, d1 = days[0], days[-1]
    ndays = (d1 - d0).days + 1
    n = ndays * SEC_PER_DAY
    os.makedirs(TL_DIR, exist_ok=True)
    present = np.zeros(ndays, dtype=bool)
    arrs = {k: np.lib.format.open_memmap(os.path.join(TL_DIR, k + ".npy"), mode="w+",
                                          dtype=dt, shape=(n,)) for k, dt in FIELDS.items()}
    for k, a in arrs.items():
        a[:] = np.nan if np.dtype(FIELDS[k]).kind == "f" and k[2:] in (
            "open", "high", "low", "close", "vwap") else 0
    for f, d in zip(files, days):
        j = (d - d0).days
        z = np.load(f)
        sl = slice(j * SEC_PER_DAY, (j + 1) * SEC_PER_DAY)
        for k in FIELDS:
            if k in z.files:
                arrs[k][sl] = z[k]
        present[j] = True
    # spot zero-trade seconds: Binance repeats last close in OHLC; still ffill for gaps.
    for k in ["s_open", "s_high", "s_low", "s_close", "p_close", "p_vwap"]:
        arrs[k][:] = _ffill(np.asarray(arrs[k]))
    # perp seconds without trades: o/h/l = last close (flat bar)
    pc = np.asarray(arrs["p_close"])
    for k in ["p_open", "p_high", "p_low"]:
        a = np.asarray(arrs[k])
        m = np.isnan(a)
        a[m] = pc[m]
        arrs[k][:] = a
    for a in arrs.values():
        a.flush()
    np.save(os.path.join(TL_DIR, "meta_days_present.npy"), present)
    with open(os.path.join(TL_DIR, "meta.txt"), "w") as fh:
        fh.write(f"{d0.isoformat()}\n{d1.isoformat()}\n{n}\n")
    print(f"timeline {d0} .. {d1}: {n} seconds, {present.sum()}/{ndays} days present")


@dataclass
class Timeline:
    d0: date
    d1: date
    n: int
    present: np.ndarray

    def __getitem__(self, k: str) -> np.ndarray:
        return np.load(os.path.join(TL_DIR, k + ".npy"), mmap_mode="r")

    @property
    def t0_epoch(self) -> int:
        return int((np.datetime64(self.d0.isoformat()) - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"))

    def sec_to_date(self, i: np.ndarray) -> np.ndarray:
        return np.datetime64(self.d0.isoformat()) + (np.asarray(i) // SEC_PER_DAY).astype("timedelta64[D]")

    def index_of(self, d: date) -> int:
        return (d - self.d0).days * SEC_PER_DAY


def load_timeline() -> Timeline:
    with open(os.path.join(TL_DIR, "meta.txt")) as fh:
        a, b, n = fh.read().split()
    return Timeline(date.fromisoformat(a), date.fromisoformat(b), int(n),
                    np.load(os.path.join(TL_DIR, "meta_days_present.npy")))


if __name__ == "__main__":
    build_timeline()
