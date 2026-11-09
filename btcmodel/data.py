"""Data acquisition and compaction.

Sources (Binance public archive, data.binance.vision):
  * spot BTCUSDT 1-second klines  -> the *reference market*. The 5-minute reference
    price P0 is the open of the first 1s kline of each 5-minute window, which is
    exactly the open of the Binance 5m candle (first trade at/after the boundary).
  * USD-M perpetual BTCUSDT aggTrades -> aggregated to 1-second bars. The perp is
    where most BTC price discovery happens, so it is used as a leading signal.

Every day is compacted into one .npz with arrays of length 86400 indexed by
second-of-day (UTC).  Seconds without trades have count == 0 and NaN prices;
forward filling happens later, in a strictly causal way.
"""
from __future__ import annotations

import hashlib
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import requests

BASE = "https://data.binance.vision/data"
from .config import DATA_DIR, SYMBOL  # noqa: E402
DAY_DIR = os.path.join(DATA_DIR, "days")
SEC_PER_DAY = 86400


def _url_spot_1s(d: date, symbol: str = SYMBOL) -> str:
    return f"{BASE}/spot/daily/klines/{symbol}/1s/{symbol}-1s-{d.isoformat()}.zip"


def _url_perp_agg(d: date, symbol: str = SYMBOL) -> str:
    return f"{BASE}/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{d.isoformat()}.zip"


def _fetch(url: str, retries: int = 5) -> bytes:
    """Download a zip and verify it against Binance's published SHA256 checksum."""
    last_err = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            blob = r.content
            c = requests.get(url + ".CHECKSUM", timeout=60)
            if c.status_code == 200:
                want = c.text.strip().split()[0]
                got = hashlib.sha256(blob).hexdigest()
                if want != got:
                    raise IOError(f"checksum mismatch for {url}")
            return blob
        except FileNotFoundError:
            raise
        except Exception as e:  # network hiccup: exponential backoff
            last_err = e
            time.sleep(2 ** k)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def _read_csv_from_zip(blob: bytes, names: list[str]) -> pa.Table:
    zf = zipfile.ZipFile(io.BytesIO(blob))
    raw = zf.read(zf.namelist()[0])
    first = raw[:200].split(b"\n")[0]
    has_header = not first[:1].isdigit()
    ropts = pacsv.ReadOptions(column_names=names, skip_rows=1 if has_header else 0)
    return pacsv.read_csv(io.BytesIO(raw), read_options=ropts)


def _day_start_ms(d: date) -> int:
    return int((np.datetime64(d.isoformat()) - np.datetime64("1970-01-01")) / np.timedelta64(1, "ms"))


def compact_spot_1s(blob: bytes, d: date) -> dict:
    names = ["open_time", "open", "high", "low", "close", "volume", "close_time",
             "qvol", "count", "tb_base", "tb_quote", "ignore"]
    t = _read_csv_from_zip(blob, names)
    ot = t["open_time"].to_numpy().astype(np.int64)
    # Binance switched spot archives from ms to us timestamps in 2025.
    if ot.max() > 10 ** 14:
        ot = ot // 1000
    sec = (ot - _day_start_ms(d)) // 1000
    ok = (sec >= 0) & (sec < SEC_PER_DAY)
    sec = sec[ok]
    out = {}
    for k in ["open", "high", "low", "close"]:
        a = np.full(SEC_PER_DAY, np.nan)
        a[sec] = t[k].to_numpy()[ok]
        out["s_" + k] = a
    for k, dt in [("volume", np.float32), ("qvol", np.float32), ("tb_base", np.float32)]:
        a = np.zeros(SEC_PER_DAY, dtype=dt)
        a[sec] = t[k].to_numpy()[ok]
        out["s_" + k] = a
    a = np.zeros(SEC_PER_DAY, dtype=np.int32)
    a[sec] = t["count"].to_numpy()[ok]
    out["s_count"] = a
    return out


def compact_perp_agg(blob: bytes, d: date) -> dict:
    names = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
    t = _read_csv_from_zip(blob, names)
    ts = t["ts"].to_numpy().astype(np.int64)
    if ts.max() > 10 ** 14:
        ts = ts // 1000
    px = t["price"].to_numpy().astype(np.float64)
    q = t["qty"].to_numpy().astype(np.float64)
    ibm = t["is_buyer_maker"].to_numpy(zero_copy_only=False)
    if ibm.dtype != bool:
        ibm = np.array([str(x).lower() == "true" for x in ibm])
    ntr = (t["last_id"].to_numpy() - t["first_id"].to_numpy() + 1).astype(np.int64)
    order = np.argsort(ts, kind="stable")
    ts, px, q, ibm, ntr = ts[order], px[order], q[order], ibm[order], ntr[order]
    sec = (ts - _day_start_ms(d)) // 1000
    ok = (sec >= 0) & (sec < SEC_PER_DAY)
    ts, px, q, ibm, ntr, sec = ts[ok], px[ok], q[ok], ibm[ok], ntr[ok], sec[ok]

    out = {}
    o = np.full(SEC_PER_DAY, np.nan); h = o.copy(); l = o.copy(); c = o.copy()
    if len(sec):
        brk = np.flatnonzero(np.diff(sec)) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, len(sec)]
        us = sec[starts]
        o[us] = px[starts]
        c[us] = px[ends - 1]
        h[us] = np.maximum.reduceat(px, starts)
        l[us] = np.minimum.reduceat(px, starts)
    out["p_open"], out["p_high"], out["p_low"], out["p_close"] = o, h, l, c
    buy = np.bincount(sec, weights=np.where(~ibm, q, 0.0), minlength=SEC_PER_DAY)
    sell = np.bincount(sec, weights=np.where(ibm, q, 0.0), minlength=SEC_PER_DAY)
    out["p_buy"] = buy.astype(np.float32)
    out["p_sell"] = sell.astype(np.float32)
    out["p_ntr"] = np.bincount(sec, weights=ntr, minlength=SEC_PER_DAY).astype(np.int32)
    # largest single aggressive order (aggTrade) per second: a "whale" footprint
    mx = np.zeros(SEC_PER_DAY)
    np.maximum.at(mx, sec, q)
    out["p_maxq"] = mx.astype(np.float32)
    # notional-weighted price (vwap) per second
    pv = np.bincount(sec, weights=px * q, minlength=SEC_PER_DAY)
    vv = buy + sell
    with np.errstate(invalid="ignore", divide="ignore"):
        out["p_vwap"] = np.where(vv > 0, pv / np.maximum(vv, 1e-12), np.nan)
    return out


def build_day(d: date, force: bool = False) -> str:
    os.makedirs(DAY_DIR, exist_ok=True)
    path = os.path.join(DAY_DIR, f"{d.isoformat()}.npz")
    if os.path.exists(path) and not force:
        return path
    spot = compact_spot_1s(_fetch(_url_spot_1s(d)), d)
    try:
        perp = compact_perp_agg(_fetch(_url_perp_agg(d)), d)
    except FileNotFoundError:
        perp = {}
    np.savez_compressed(path + ".tmp.npz", **spot, **perp)
    os.replace(path + ".tmp.npz", path)
    return path


def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def build_range(a: date, b: date, workers: int = 6) -> list[str]:
    days = list(daterange(a, b))
    done, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(build_day, d): d for d in days}
        for i, f in enumerate(as_completed(futs)):
            d = futs[f]
            try:
                done.append(f.result())
            except Exception as e:
                failed.append((d, repr(e)))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(days)} days processed", flush=True)
    if failed:
        print("FAILED:", failed)
    return sorted(done)


if __name__ == "__main__":
    import sys
    a = date.fromisoformat(sys.argv[1])
    b = date.fromisoformat(sys.argv[2])
    build_range(a, b, workers=int(sys.argv[3]) if len(sys.argv) > 3 else 6)
