"""Live-inference demo with the packaged models: replay one window second by second.

    python scripts/demo_live.py [models/wf5_20260430] [YYYY-MM-DDTHH:MM]

Feeds the predictor ONLY data up to each decision second (exactly as live) and prints the
continuously updated forecasts relative to the window's fixed reference P0.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.predictor import Predictor  # noqa: E402
from btcmodel.timeline import load_timeline, FIELDS  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "wf5_20260430")
    when = np.datetime64(sys.argv[2] if len(sys.argv) > 2 else "2026-10-20T14:35")
    pred = Predictor.load_dir(path)
    tl = load_timeline()
    T0 = int((when - np.datetime64("2025-01-01")) / np.timedelta64(1, "s")) // 300 * 300
    start = (T0 - 9 * 86400) // 3600 * 3600
    p0 = float(tl["s_open"][T0])
    print(f"window {np.datetime64('2025-01-01') + np.timedelta64(T0, 's')} UTC   P0 = ${p0:,.2f}")
    print(" e    price      vs P0   P(10s>=P0) P(30s>=P0) P(close>=P0)  close 5%..95% ($)        P(touch +10bps) P(+/-10 up first)")
    for e in [0, 30, 60, 120, 180, 240, 270, 290]:
        arrays = {f: np.asarray(tl[f][start:T0 + e]) for f in FIELDS}
        pred.ingest(arrays, tl.t0_epoch + start)
        out = pred.predict(tl.t0_epoch + T0 + e, p0=p0)
        hd = out["heads"]
        c = hd["close"]
        q = c["quantiles_usd"]
        g = lambda h: f"{hd[h]['p_above_ref']:.3f}" if h in hd else "  -  "
        print(f"{e:3d} {out['price_now']:10,.2f} {1e4 * (out['price_now'] / p0 - 1):+7.2f}bp   {g('h10'):>6}     {g('h30'):>6}     "
              f"{c['p_above_ref']:.3f}     {q['0.05']:,.0f}..{q['0.95']:,.0f}      {c['touch']['+10bps']:.3f}          "
              f"{c['first_passage']['+/-10bps']['up_first']:.3f}")
    print(f"\nactual close ${float(tl['s_close'][T0 + 299]):,.2f} -> {'UP' if tl['s_close'][T0 + 299] >= p0 else 'DOWN'}")
    print("\nfull output at the last decision point:")
    print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    main()
