"""Package trained heads into a compact, loadable directory (Predictor.load_dir).

    python scripts/package_models.py prod models/prod_20261102          # prod_<head>.pkl
    python scripts/package_models.py valid_wf5 models/wf5_20260430      # models_<head>_valid_wf5.pkl
The residual law is compacted to a tail-refined quantile grid (verified: identical log-loss
to 5 decimals, max |dp| < 5e-4 vs the exact law)."""
import gzip
import os
import pickle
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.config import MODEL_DIR as MD, DATA_DIR  # noqa: E402


def src_name(kind, head):
    return f"prod_{head}.pkl" if kind == "prod" else f"models_{head}_{kind}.pkl"


if __name__ == "__main__":
    kind, out = sys.argv[1], sys.argv[2]
    heads = sys.argv[3:] or ["h10", "h15", "h30", "h60", "close"]
    os.makedirs(out, exist_ok=True)
    for h in heads:
        f = os.path.join(MD, src_name(kind, h))
        if not os.path.exists(f):
            print(f"missing {f}, skipped")
            continue
        M = pickle.load(open(f, "rb"))
        M["head"].dist.compact()
        with gzip.open(os.path.join(out, f"{h}.pkl.gz"), "wb", compresslevel=9) as fh:
            pickle.dump(M, fh, protocol=5)
        print(h, f"{os.path.getsize(os.path.join(out, f'{h}.pkl.gz')) / 1e6:.1f} MB")
    shutil.copy(os.path.join(DATA_DIR, "cache", "season.pkl"), os.path.join(out, "season.pkl"))
