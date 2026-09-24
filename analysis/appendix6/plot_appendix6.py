"""Publication-quality figures for Appendix A.6 follow-up analyses.

Style conventions (per task spec Sec. 2 / "GENERAL IMPLEMENTATION RULES"):
  - matplotlib only, vector PDF + PNG, restrained color palette
  - same dataset -> same color/marker in every figure (common.DATASET_COLOR/MARKER)
  - same backbone -> same color in every figure (common.BACKBONE_COLOR)
  - consistent font sizes / linewidths, minimal grid decoration

Reads only the CSV/NPZ files already written by analyze_*.py (Sec. "computational
shortcut" -- figures never re-run inference).
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

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(f"{C.FIGURES_DIR}/{name}.pdf")
    fig.savefig(f"{C.FIGURES_DIR}/{name}.png", dpi=200)
    print("saved", f"{C.FIGURES_DIR}/{name}.pdf/.png")


# ---------------------------------------------------------------------------
# Figure A: Temporal popularity regimes across datasets
# ---------------------------------------------------------------------------
def figure_a():
    summary = pd.read_csv(f"{C.RESULTS_DIR}/empirical_popularity_summary.csv")
    temporal = pd.read_csv(f"{C.RESULTS_DIR}/temporal_shift_summary.csv", header=[0, 1], index_col=0)
    curves = np.load(f"{C.RESULTS_DIR}/empirical_popularity_curves.npz")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    for ds in C.DATASETS:
        grid = curves[f"{ds}_grid"]
        mean = curves[f"{ds}_mean"]
        ax.plot(grid, mean, color=C.DATASET_COLOR[ds], marker=C.DATASET_MARKER[ds],
                markevery=20, markersize=4, label=C.DATASET_LABELS[ds])
    ax.plot([0, 100], [0, 1], color="gray", linestyle=":", linewidth=1, label="Uniform popularity")
    ax.set_xlabel("Item percentile (sorted by local popularity)")
    ax.set_ylabel("Cumulative interaction mass")
    ax.set_title("(a) Local 3-day popularity: cumulative mass")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    x = np.arange(len(C.DATASETS))
    means = [summary.loc[summary.dataset == ds, "H_norm_mean"].values[0] for ds in C.DATASETS]
    stds = [summary.loc[summary.dataset == ds, "H_norm_std"].values[0] for ds in C.DATASETS]
    colors = [C.DATASET_COLOR[ds] for ds in C.DATASETS]
    ax.bar(x, means, yerr=stds, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([C.DATASET_LABELS[d] for d in C.DATASETS])
    ax.set_ylabel(r"Normalized entropy $H_{norm}$ (local, 3-day)")
    ax.set_title("(b) Popularity concentration\n(higher = less concentrated)")
    ax.set_ylim(0, 1)

    ax = axes[2]
    jsd_mean = [temporal.loc[ds, ("jsd_bits", "mean")] for ds in C.DATASETS]
    jsd_std = [temporal.loc[ds, ("jsd_bits", "std")] for ds in C.DATASETS]
    ax.bar(x, jsd_mean, yerr=jsd_std, color=colors, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([C.DATASET_LABELS[d] for d in C.DATASETS])
    ax.set_ylabel("Adjacent-day JSD (bits)")
    ax.set_title("(c) Temporal instability\n(higher = faster-changing popularity)")

    fig.suptitle("Figure A: Temporal popularity regimes across datasets (data only, no trained model)", y=1.03)
    fig.tight_layout()
    save(fig, "fig_data_regime")


# ---------------------------------------------------------------------------
# Figure B: Empirical vs. Hawkes-estimated popularity concentration
# ---------------------------------------------------------------------------
def figure_b():
    curves = np.load(f"{C.RESULTS_DIR}/hawkes_vs_empirical_curves.npz")
    summary = pd.read_csv(f"{C.RESULTS_DIR}/hawkes_concentration_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ds in zip(axes, C.DATASETS):
        grid = curves[f"{ds}_grid"]
        if f"{ds}_emp_mean" in curves:
            ax.plot(grid, curves[f"{ds}_emp_mean"], color="black", linewidth=1.8,
                     label="Empirical (3-day)")
        h_mean = curves[f"{ds}_hawkes_mean"]
        h_std = curves[f"{ds}_hawkes_std"]
        ax.plot(grid, h_mean, color=C.DATASET_COLOR[ds], linewidth=1.8, label=r"Hawkes $\pi_\Phi$")
        ax.fill_between(grid, np.clip(h_mean - h_std, 0, 1), np.clip(h_mean + h_std, 0, 1),
                         color=C.DATASET_COLOR[ds], alpha=0.18, linewidth=0)
        ax.plot([0, 100], [0, 1], color="gray", linestyle=":", linewidth=1)
        ax.set_title(C.DATASET_LABELS[ds])
        ax.set_xlabel("Item percentile")
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("Cumulative popularity mass")

    fig.tight_layout()
    save(fig, "fig_empirical_vs_hawkes")

    print(summary[["dataset", "H_norm_emp_mean", "H_norm_hawkes_mean", "delta_H_norm_mean",
                    "top5pct_emp_mean", "top5pct_hawkes_mean", "delta_top5pct_mean"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Figure C: Effect of the popularity component on Tail ranking
# ---------------------------------------------------------------------------
def figure_c():
    eta_df = pd.read_csv(f"{C.RESULTS_DIR}/eta_tail_recall.csv")
    rank_df = pd.read_csv(f"{C.RESULTS_DIR}/tail_rank_shift.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes[0]
    kr = eta_df[eta_df.dataset == "kuairand"]
    for bb in C.BACKBONES:
        sub = kr[kr.backbone == bb].sort_values("eta")
        ax.plot(sub.eta, sub.recall_at_10, color=C.BACKBONE_COLOR[bb], marker="o",
                markersize=3.5, label=bb.upper() if bb != "grurec" else "GRU")
        default_row = sub[sub.eta_is_default]
        if len(default_row):
            ax.scatter(default_row.eta, default_row.recall_at_10, color=C.BACKBONE_COLOR[bb],
                       marker="*", s=140, zorder=5, edgecolors="black", linewidths=0.5)
    ax.set_xlabel(r"Inference-time mixing weight $\eta$")
    ax.set_ylabel("Tail Recall@10")
    ax.set_title("(a) KuaiRand: Tail Recall@10 vs. $\\eta$\n(all six backbones; $\\star$ = trained default $\\eta$)")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)

    ax = axes[1]
    kr_rank = rank_df[rank_df.dataset == "kuairand"]
    order = C.BACKBONES
    data = [kr_rank[kr_rank.backbone == bb].delta_log_rank.values for bb in order]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, bb in zip(parts["bodies"], order):
        pc.set_facecolor(C.BACKBONE_COLOR[bb])
        pc.set_alpha(0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([b.upper() if b != "grurec" else "GRU" for b in order], rotation=20)
    ax.set_ylabel(r"$\Delta\log$-rank = $\log(1{+}r_{comb}) - \log(1{+}r_{util})$")
    ax.set_title("(b) KuaiRand: Tail-positive rank displacement\n(> 0 = popularity term worsens rank)")

    fig.suptitle("Figure C: Effect of the learned popularity component on Tail ranking (KuaiRand)", y=1.03)
    fig.tight_layout()
    save(fig, "fig_tail_effect")


if __name__ == "__main__":
    figure_a()
    figure_b()
    figure_c()
