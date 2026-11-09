"""Refit every head for deployment on ALL available data.

The research protocol (train 2025 / valid Jan-Apr 2026 / test May-Nov 2026) is what the
reported numbers come from.  For live use, more recent data matters (regimes drift), so
this script refits with the same architecture and hyper-parameters on everything except
the most recent 30 days, which serve only as the early-stopping set.

    python scripts/train_production.py                       # all data, all heads
    python scripts/train_production.py --end 2026-04-30 --tag wf0430   # walk-forward refit

With --end, only decision points whose window lies on or before that date are used
(a walk-forward refit that only knows the past); the models are written as
models_<head>_valid<TAG>.pkl so that evaluate_saved.py (BTC_TAG=<TAG>) can score them.
Outputs /home/user/data/models/prod_<head>.pkl otherwise.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_eval import load_head, MODEL_DIR, STACK_DIR, HEAD_ORDER  # noqa: E402
from btcmodel.models import HeadModel, _fit, VOL_FEATS, DIR_FEATS  # noqa: E402
from btcmodel.events import touch_rows, TouchModel, fp_rows, FirstPassageModel  # noqa: E402

ES_DAYS = 30


def main(heads, end=None, tag=None, old_thin=1, resume=False):
    from btcmodel.regime import REGIME_FEATS
    from train_eval import USE_REGIME
    for head in heads:
        name = f"models_{head}_valid{tag}.pkl" if tag else f"prod_{head}.pkl"
        stack = [os.path.join(STACK_DIR, f"prod{tag or ''}_{head}_{k}.parquet") for k in ("tr", "va")]
        if resume and os.path.exists(os.path.join(MODEL_DIR, name)) and all(map(os.path.exists, stack)):
            print(f"resume: {name} already complete, skipping", flush=True)
            continue
        from btcmodel.dataset import SPLITS
        from datetime import date as _date
        splits = [s for s in ("train", "valid", "test")
                  if end is None or SPLITS[s][0] <= _date.fromisoformat(end)]
        # note: the valid/test caches are sampled every 5 s vs 10 s for train, so the most
        # recent regime gets ~2x the weight of 2025 - a deliberate recency tilt.
        parts = []
        for s in splits:
            x = load_head(s, head)
            if old_thin > 1:
                # thin the pre-2026 regime by whole windows (recency tilt), split by split so
                # that peak memory stays low; identical rows to thinning after concatenation
                old = x["day"].values < 365
                if old.any():
                    x = x[~old | ((x["T0"].values // 300) % old_thin == 0)].reset_index(drop=True)
            parts.append(x)
        d = pd.concat(parts, ignore_index=True)
        del parts, x
        if end is not None:
            end_day = int((np.datetime64(end) - np.datetime64("2025-01-01")) / np.timedelta64(1, "D"))
            d = d[d["day"].values <= end_day].reset_index(drop=True)
        last = d["day"].max()
        es = d["day"].values > last - ES_DAYS
        gap = (d["day"].values > last - ES_DAYS - 1) & ~es          # 1-day embargo
        tr, va = d[~es & ~gap].reset_index(drop=True), d[es].reset_index(drop=True)
        del d
        extra = []
        for sh in [h for h in HEAD_ORDER if HEAD_ORDER.index(h) < HEAD_ORDER.index(head)]:
            for x, key in ((tr, "tr"), (va, "va")):
                s = pd.read_parquet(os.path.join(STACK_DIR, f"prod{tag or ''}_{sh}_{key}.parquet"))
                x[f"mu_{sh}_px"] = pd.merge(x[["T0", "e"]], s, on=["T0", "e"], how="left")["drift_px"].values
                x[f"mu_{sh}_px"] = (x[f"mu_{sh}_px"] / x["naive_scale"]).astype(np.float32)
            extra.append(f"mu_{sh}_px")
        hm = HeadModel(head, extra_dir_feats=extra,
                       regime_feats=REGIME_FEATS if USE_REGIME else None).fit(tr, va)
        for x, key in ((tr, "tr"), (va, "va")):
            pd.DataFrame({"T0": x["T0"].values, "e": x["e"].values,
                          "drift_px": x["mu"].values * x["sig"].values}).to_parquet(
                os.path.join(STACK_DIR, f"prod{tag or ''}_{head}_{key}.parquet"))
        feats = sorted(set(VOL_FEATS + DIR_FEATS + ["lH", "z_naive", "lproj_300", "lsint"]
                           + (REGIME_FEATS if USE_REGIME else [])))
        ytr, yva = (tr[f"{head}__y"].values >= 0).astype(int), (va[f"{head}__y"].values >= 0).astype(int)
        dm = _fit(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000),
                  tr[feats], ytr, va[feats], yva, es=100)
        sub = lambda x, f: x.iloc[::f].reset_index(drop=True)
        tm = TouchModel().fit(touch_rows(sub(tr, 4), head, hm), touch_rows(sub(va, 2), head, hm))
        fm = FirstPassageModel().fit(fp_rows(sub(tr, 4), head, hm), fp_rows(sub(va, 2), head, hm))
        with open(os.path.join(MODEL_DIR, name), "wb") as fh:
            pickle.dump({"head": hm, "touch": tm, "fp": fm, "direct": dm}, fh)
        print(f"saved {name}", flush=True)
        del tr, va


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("heads", nargs="*", default=HEAD_ORDER)
    ap.add_argument("--end", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--old-thin", type=int, default=1, help="keep 1/N of 2025 windows")
    ap.add_argument("--resume", action="store_true", help="skip heads whose model + stack files exist")
    a = ap.parse_args()
    main(a.heads, a.end, a.tag, a.old_thin, a.resume)
