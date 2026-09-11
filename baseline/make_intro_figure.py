#%%
"""
Compact "hook" figure for the Introduction, built from the same (R, L_uni)
data as the results-section synthesis scatter (analysis_representation/
synthesis_R_vs_Luni.csv), but restyled for a narrow single-column placement:
minimal text, direct-labeled clusters instead of a legend box, and short
axis-end cues so the plot reads at a glance even before the reader has seen
Theorem 1 / Corollary 1.

Does NOT replace the full results-section version (analysis_representation/
extra_synthesis_scatter.{pdf,png}), which keeps the per-dataset marker
legend and Pearson annotation for the rigorous reading.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/make_intro_figure.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

import analysis_representation as ar

OUT_DIR = "./analysis_representation"
CSV_PATH = f"{OUT_DIR}/synthesis_R_vs_Luni.csv"
DATASET_MARKER = {"micro_video": "o", "ml-1m": "s", "kuairand": "^"}


def load_records():
    records = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            records.append(dict(dataset=row["dataset"], backbone=row["backbone"],
                                 approach=row["approach"], seed=int(row["seed"]),
                                 R=float(row["R"]), L_uni=float(row["L_uni"])))
    return records


def make_intro_figure():
    ar.setup_style()
    records = load_records()
    all_R = np.array([r["R"] for r in records])
    all_L = np.array([r["L_uni"] for r in records])
    r_all, _ = pearsonr(all_R, all_L)

    fig, ax = plt.subplots(figsize=(3.35, 3.05))

    # faint global trend guide
    coeffs = np.polyfit(all_R, all_L, 1)
    xs = np.linspace(all_R.min(), all_R.max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="0.72", lw=1.1, ls=(0, (5, 3)), zorder=1)

    for approach, color in [("vanilla", ar.VANILLA_COLOR), ("ours", ar.OURS_COLOR)]:
        recs = [r for r in records if r["approach"] == approach]
        Rs = np.array([r["R"] for r in recs])
        Ls = np.array([r["L_uni"] for r in recs])
        ax.scatter(Rs, Ls, s=22, color=color, alpha=0.55, linewidths=0, zorder=3)

        groups = {}
        for r in recs:
            groups.setdefault((r["dataset"], r["backbone"]), []).append(r)
        means_R = [np.mean([g["R"] for g in gs]) for gs in groups.values()]
        means_L = [np.mean([g["L_uni"] for g in gs]) for gs in groups.values()]
        ax.scatter(means_R, means_L, s=60, color=color, alpha=0.98, edgecolors="white",
                   linewidths=0.8, zorder=4)

    # direct cluster labels -- no color legend box, matches the compact-figure
    # convention (color-coded text placed right next to what it names), each
    # placed in genuinely empty space just outside its own cluster's extent
    ax.text(0.985, -0.42, "Vanilla", color=ar.VANILLA_COLOR, fontsize=13, fontweight="bold",
            ha="right", va="top")
    ax.text(0.40, -3.30, "Ours", color=ar.OURS_COLOR, fontsize=13, fontweight="bold",
            ha="left", va="center")

    # small, unobtrusive correlation cue near the trend line, away from both
    # cluster labels
    ax.text(0.68, np.polyval(coeffs, 0.68) + 0.30, f"$r={r_all:.2f}$", color="0.5",
            fontsize=9, ha="center", va="bottom", style="italic")

    ax.set_xlim(-0.03, 1.05)
    ax.set_ylabel(r"uniformity $\mathcal{L}_{uni}$", fontsize=11)
    ax.set_xlabel(r"user-representation alignment $R$", fontsize=11, labelpad=20)

    # axis-end cues so the plot is legible before the reader has seen
    # Theorem 1 / Corollary 1: bold, dark, and directly under the 0/1 ticks
    ax.text(0.0, -0.155, "diverse", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9.5, color="0.25", fontweight="bold", style="italic")
    ax.text(1.0, -0.155, "collapsed", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=9.5, color="0.25", fontweight="bold", style="italic")

    ax.tick_params(labelsize=9)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/intro_teaser_R_vs_Luni.{ext}", dpi=400 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/intro_teaser_R_vs_Luni.{{pdf,png}}")


if __name__ == "__main__":
    make_intro_figure()
