#%%
"""
Effect of the centering constraint (gamma_cen): representation geometry vs.
recommendation accuracy, side by side.

Reinterprets the gamma_cen sweep (assets/sen_gamma.xlsx, Recall@10) together
with the uniformity L_uni computed from the very same checkpoints
(figures/table_gamma_uniformity.csv, produced by
compute_gamma_uniformity.py) as one ablation path per backbone:

    Ours w/o centering (gamma_cen = 0) --> increasing centering --> Ours full (gamma_cen = gamma*)

and, to its left, a Vanilla reference point -- the backbone with NO
popularity/time decomposition at all (norm_backbone_* checkpoints), from
compute_vanilla_uniformity.py / compute_vanilla_recall.py. Vanilla is a
different model class from "Ours at gamma_cen=0" (it lacks the popularity
component entirely), so it is drawn as a separate marker connected by a
dashed segment, not folded into the gamma_cen axis as if it were just
another sweep point.

Top row: L_uni vs. gamma_cen (lower = more uniform / less collapsed).
Bottom row: Recall@10 vs. gamma_cen, same checkpoints, same x-axis.

One column per dataset, one shared legend.

Run from the repo root (after compute_gamma_uniformity.py,
compute_vanilla_uniformity.py and compute_vanilla_recall.py have produced
their CSVs):
    /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/analysis_gamma_centering.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_sensitivity import (
    DATASET_ORDER, DATASET_LABELS, MODEL_ORDER, MODEL_LABELS,
    MODEL_COLORS, MODEL_MARKERS, setup_style,
)

RECALL_PATH = "./assets/sen_gamma.xlsx"
UNIFORMITY_PATH = "./figures/table_gamma_uniformity.csv"
VANILLA_UNIFORMITY_PATH = "./figures/table_vanilla_uniformity.csv"
VANILLA_RECALL_PATH = "./figures/table_vanilla_recall.csv"
OUT_PATH = "./figures/fig_sensitivity_gamma.png"
X_LABEL = r"Centering coeff. $\gamma$"
N_XTICKS = 4


def load_recall(path):
    df = pd.read_excel(path)
    df = df[df["dataset"].isin(DATASET_ORDER) & df["model_name"].isin(MODEL_ORDER)].copy()
    for col in ["gamma", "test_recall_10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["gamma", "test_recall_10"])
    df = df.drop_duplicates(subset=["dataset", "model_name", "gamma"], keep="first")
    return df


def load_csv(path):
    return pd.read_csv(path)


def draw_row(axes, df, vanilla_df, metric_col, y_label):
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = df[df["dataset"] == dataset]
        van_sub = vanilla_df[vanilla_df["dataset"] == dataset]
        x_values = sorted(sub["gamma"].unique())
        # slot 0 is reserved for "Vanilla"; the gamma sweep starts at slot 1
        x_pos = {v: i + 1 for i, v in enumerate(x_values)}

        for model in MODEL_ORDER:
            m = sub[sub["model_name"] == model].sort_values("gamma")
            if m.empty:
                continue
            xs = [x_pos[v] for v in m["gamma"]]
            ys = list(m[metric_col])

            van_row = van_sub[van_sub["model_name"] == model]
            if not van_row.empty:
                van_y = van_row[metric_col].iloc[0]
                ax.plot(
                    [0, xs[0]], [van_y, ys[0]],
                    color=MODEL_COLORS[model], linewidth=1.4, linestyle="--",
                    alpha=0.6, zorder=2,
                )
                ax.plot(
                    [0], [van_y], marker=MODEL_MARKERS[model],
                    markersize=11, markerfacecolor="none",
                    markeredgecolor=MODEL_COLORS[model], markeredgewidth=2.0,
                    linestyle="none", zorder=3,
                )

            ax.plot(
                xs, ys,
                color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                markersize=11, markeredgewidth=1.1, markeredgecolor="white",
                linewidth=3.0, alpha=0.95, label=MODEL_LABELS[model], zorder=3,
            )

        ax.axvline(0.5, color="0.75", linestyle=":", linewidth=1.2, zorder=0)

        n_gamma = len(x_values)
        ax.set_xticks(range(n_gamma + 1), minor=True)
        n_lab = min(N_XTICKS, n_gamma)
        lab_idx = sorted(set(np.linspace(0, n_gamma - 1, n_lab).round().astype(int)))
        ax.set_xticks([0] + [i + 1 for i in lab_idx])
        ax.set_xticklabels(["Van."] + [f"{x_values[i]:g}" for i in lab_idx])
        ax.margins(x=0.06)
        ax.tick_params(width=1.1, length=6)
        ax.tick_params(which="minor", length=3)
    axes[0].set_ylabel(y_label, labelpad=10)


#%%
if __name__ == "__main__":
    setup_style()
    recall_df = load_recall(RECALL_PATH)
    uni_df = load_csv(UNIFORMITY_PATH)
    vanilla_uni_df = load_csv(VANILLA_UNIFORMITY_PATH)
    vanilla_recall_df = load_csv(VANILLA_RECALL_PATH)

    fig, axes = plt.subplots(2, len(DATASET_ORDER), figsize=(14.4, 8.4))

    draw_row(axes[0], uni_df, vanilla_uni_df, "l_uni", r"Uniformity $\mathcal{L}_{\mathrm{uni}}$")
    for ax, dataset in zip(axes[0], DATASET_ORDER):
        ax.set_title(DATASET_LABELS[dataset], pad=16)

    draw_row(axes[1], recall_df, vanilla_recall_df, "test_recall_10", "Recall@10")
    for ax in axes[1]:
        ax.set_xlabel(X_LABEL, labelpad=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(MODEL_ORDER), markerscale=1.5,
        bbox_to_anchor=(0.5, -0.015), frameon=False, columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.subplots_adjust(wspace=0.32, hspace=0.28)
    fig.savefig(OUT_PATH, dpi=400, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PATH} (+ .pdf)")
