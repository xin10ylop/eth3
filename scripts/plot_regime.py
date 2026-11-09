"""Figure: how BTC's microstructure mechanics drift across train / valid / test."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.dataset import CACHE
from btcmodel.config import FIG_DIR, ASSET  # noqa: E402

rows = []
for s in ["train", "valid", "test"]:
    d = pd.read_parquet(f"{CACHE}/{s}.parquet", columns=["day", "e"])
    r = pd.read_parquet(f"{CACHE}/{s}_regime.parquet", columns=["reg_ll_beta_6h", "reg_catchup_6h", "reg_ac1_1h", "reg_lrv_7d"])
    m = d["e"].values == 0
    r = r[m].copy()
    r["date"] = np.datetime64("2025-01-01") + d["day"].values[m].astype("timedelta64[D]")
    r["split"] = s
    rows.append(r)
R = pd.concat(rows)
W = R.groupby(R["date"].values.astype("datetime64[W]")).agg({"reg_ll_beta_6h": "mean", "reg_catchup_6h": "mean",
                                                               "reg_ac1_1h": "mean", "reg_lrv_7d": "mean"})
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axs[0].plot(W.index, W["reg_ll_beta_6h"], color="#3b6fb6"); axs[0].set_ylabel("perp->spot\nlead-lag beta")
axs[1].plot(W.index, W["reg_ac1_1h"], color="#d9480f"); axs[1].set_ylabel("spot 1s\nautocorrelation")
axs[2].plot(W.index, np.sqrt(np.exp(W["reg_lrv_7d"]) * 365 * 86400) * 100, color="#444"); axs[2].set_ylabel("annualized\nvol (%)")
for ax in axs:
    ax.axvspan(np.datetime64("2026-01-02"), np.datetime64("2026-04-30"), color="#7fa7d9", alpha=0.15)
    ax.axvspan(np.datetime64("2026-05-02"), np.datetime64("2026-11-02"), color="#d9480f", alpha=0.10)
axs[0].set_title(f"Regime drift in {ASSET} microstructure (weekly means). blue = validation, orange = test")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "regime_drift.png"), dpi=130)
print("saved")
