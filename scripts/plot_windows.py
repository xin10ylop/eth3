"""Anatomy of individual windows: price vs fixed P0 and the continuously updated forecasts."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.timeline import load_timeline  # noqa: E402

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
from btcmodel.config import FIG_DIR as FIG, MODEL_DIR  # noqa: E402


def main(n=4, seed=3):
    P = pickle.load(open(os.path.join(MODEL_DIR, f"preds_close_{SPLIT}{TAG}.pkl"), "rb"))
    tl = load_timeline()
    sc = np.asarray(tl["s_close"])
    rng = np.random.default_rng(seed)
    wins = np.unique(P["T0"])
    # pick windows that are actually uncertain (price crosses P0 at least twice)
    cand = []
    for T0 in rng.permutation(wins)[:3000]:
        p0 = tl["s_open"][T0]
        path = sc[T0:T0 + 300] - p0
        if np.sum(np.diff(np.sign(path)) != 0) >= 3:
            cand.append(T0)
        if len(cand) >= n:
            break
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    for ax, T0 in zip(axs, cand):
        p0 = float(tl["s_open"][T0])
        path = (sc[T0:T0 + 300] / p0 - 1) * 1e4
        m = P["T0"] == T0
        e = P["e"][m]
        o = np.argsort(e)
        ax.plot(np.arange(1, 301), path, color="#444", lw=1.0, label="spot vs P0 (bps)")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_ylabel("bps vs P0")
        ax2 = ax.twinx()
        ax2.plot(e[o], P["p"]["B0_naive"][m][o], color="#9aa0a6", lw=1.2, label="naive P(close>=P0)")
        ax2.plot(e[o], P["p"]["E_blend"][m][o], color="#d9480f", lw=1.6, label="model P(close>=P0)")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("probability")
        outcome = "UP" if sc[T0 + 299] >= p0 else "DOWN"
        ts = np.datetime64("2025-01-01") + np.timedelta64(int(T0), "s")
        ax.set_title(f"window {ts} UTC, P0=${p0:,.2f}, outcome {outcome}", fontsize=9)
        if ax is axs[0]:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    axs[-1].set_xlabel("elapsed seconds")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"windows_{SPLIT}{TAG}.png"), dpi=130)
    print("saved", os.path.join(FIG, f"windows_{SPLIT}{TAG}.png"))


if __name__ == "__main__":
    main()
