"""Merged mechanism+effect figure (replaces the separate fig_mu_vs_frequency and
fig_tail_effect figures with one compact 3-panel figure) for the trimmed appendix.

(a) mu_v vs. item frequency percentile, all 3 datasets (mechanism)
(b) KuaiRand Tail Recall@10 vs. eta, all 6 backbones (effect)
(c) KuaiRand Tail-positive rank displacement, all 6 backbones (effect)

Reads only already-computed CSVs; no model is re-run.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from plot_mu_vs_frequency import bin_by_percentile

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11.5, "axes.titlesize": 11.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "lines.linewidth": 1.8, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def main():
    item_df = pd.read_csv(f"{C.RESULTS_DIR}/excitation_ratio_by_item.csv")
    eta_df = pd.read_csv(f"{C.RESULTS_DIR}/eta_tail_recall.csv")
    rank_df = pd.read_csv(f"{C.RESULTS_DIR}/tail_rank_shift.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    for ds in C.DATASETS:
        sub = item_df[item_df.dataset == ds]
        bins = bin_by_percentile(sub.train_freq.values.astype(np.float64), sub.mu.values.astype(np.float64))
        ax.plot(bins.pct_mid, bins.mu_mean, color=C.DATASET_COLOR[ds], marker=C.DATASET_MARKER[ds],
                markersize=3.5, label=C.DATASET_LABELS[ds])
    ax.set_yscale("log")
    ax.axvspan(90, 100, color="gray", alpha=0.08, zorder=0)
    ax.set_xlabel("Item frequency percentile")
    ax.set_ylabel(r"Mean $\mu_v$ (log)")
    ax.set_title("(a) Mechanism: $\\mu_v$ vs.\nfrequency percentile")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    ax = axes[1]
    kr = eta_df[eta_df.dataset == "kuairand"]
    for bb in C.BACKBONES:
        sub = kr[kr.backbone == bb].sort_values("eta")
        ax.plot(sub.eta, sub.recall_at_10, color=C.BACKBONE_COLOR[bb], marker="o", markersize=3,
                label=bb.upper() if bb != "grurec" else "GRU")
        default_row = sub[sub.eta_is_default]
        if len(default_row):
            ax.scatter(default_row.eta, default_row.recall_at_10, color=C.BACKBONE_COLOR[bb],
                       marker="*", s=110, zorder=5, edgecolors="black", linewidths=0.5)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Tail Recall@10")
    ax.set_title("(b) Effect: Tail Recall@10\nvs. $\\eta$ (6 backbones)")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=7)

    ax = axes[2]
    kr_rank = rank_df[rank_df.dataset == "kuairand"]
    order = C.BACKBONES
    data = [kr_rank[kr_rank.backbone == bb].delta_log_rank.values for bb in order]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, bb in zip(parts["bodies"], order):
        pc.set_facecolor(C.BACKBONE_COLOR[bb])
        pc.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([b.upper() if b != "grurec" else "GRU" for b in order], rotation=25, fontsize=8.5)
    ax.set_ylabel(r"$\Delta\log$-rank")
    ax.set_title("(c) Effect: Tail rank\ndisplacement (6 backbones)")

    fig.tight_layout()
    fig.savefig(f"{C.FIGURES_DIR}/fig_mechanism_effect.pdf")
    fig.savefig(f"{C.FIGURES_DIR}/fig_mechanism_effect.png", dpi=200)
    print("saved", f"{C.FIGURES_DIR}/fig_mechanism_effect.pdf/.png")


if __name__ == "__main__":
    main()
