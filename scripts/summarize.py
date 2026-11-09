"""Markdown tables from saved result JSONs (no hand-copied numbers).

    python scripts/summarize.py test _v4      # split, tag
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from btcmodel.config import ART_DIR  # noqa: E402
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
NAMES = {"B0_naive": "naive Gaussian random walk (trailing vol)", "B1_volmodel": "+ clock-aware vol model",
         "B2_fattail": "+ no-move atom & fat-tailed law", "B3_struct": "+ drift (structural model)",
         "M_final": "+ calibration from prior", "D_directGBM": "direct GBM (rival)",
         "E_blend": "**final: blend**"}


def main():
    out = []
    for head, title in [("h10", "10 s"), ("h15", "15 s"), ("h30", "30 s"), ("h60", "60 s"), ("close", "5-minute close")]:
        f = os.path.join(ART_DIR, f"results_{head}_{SPLIT}{TAG}.json")
        if not os.path.exists(f):
            continue
        R = json.load(open(f))
        out.append(f"\n#### {title} head - P(S >= P0) on {SPLIT}\n")
        out.append("| model | log-loss | Brier | ECE | ΔLL vs naive [95% CI] |")
        out.append("|---|---|---|---|---|")
        for r in R["up_table"]:
            ci = r.get("ci95")
            d = "" if ci is None else f"{r['dLL_vs_B0']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            out.append(f"| {NAMES.get(r['model'], r['model'])} | {r['logloss']:.4f} | {r['brier']:.4f} | {r['ece']:.4f} | {d} |")
        be = R.get("by_e", [])
        if be:
            key = "skill_E_vs_B0_%" if "skill_E_vs_B0_%" in be[0] else "skill_M_vs_B0_%"
            out.append("\nlog-loss skill vs naive by elapsed time: " + ", ".join(
                f"e∈{r['e_bucket']}: {r[key]:.1f}%" for r in be))
        q = R.get("quantiles")
        if q:
            out.append(f"\nquantile coverage (nominal 5/10/25/50/75/90/95%): "
                       + "/".join(f"{100 * c:.1f}" for c in q["coverage"])
                       + f"; pinball skill vs naive Gaussian {100 * (1 - q['pinball_model'] / q['pinball_naive']):.2f}%")
        t = R.get("touch")
        if t:
            out.append("\n| touch level vs P0 | hit rate | LL naive BM | LL final | skill | ECE |")
            out.append("|---|---|---|---|---|---|")
            for r in t:
                out.append(f"| {r['level_bps']:+d} bps | {100 * r['rate']:.1f}% | {r['LL_naiveBM']:.4f} | {r['LL_final']:.4f} | "
                           f"{100 * (1 - r['LL_final'] / r['LL_naiveBM']):.1f}% | {r['ECE_final']:.4f} |")
        fp = R.get("first_passage")
        if fp:
            out.append("\n| first passage ±k | P(up first) | P(down first) | mlogloss naive | final | skill | AUC up-first given a hit |")
            out.append("|---|---|---|---|---|---|---|")
            for r in fp:
                out.append(f"| ±{r['k_bps']} bps | {100 * r['P_up_first']:.1f}% | {100 * r['P_dn_first']:.1f}% | {r['MLL_naive']:.4f} | "
                           f"{r['MLL_final']:.4f} | {100 * (1 - r['MLL_final'] / r['MLL_naive']):.1f}% | {r['AUC_upfirst|hit']:.3f} |")
    s = "\n".join(out)
    print(s)
    open(os.path.join(ART_DIR, f"tables_{SPLIT}{TAG}.md"), "w").write(s)


if __name__ == "__main__":
    main()
