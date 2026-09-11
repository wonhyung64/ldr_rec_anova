#%%
"""
Hyperparameter sensitivity plots for the Hawkes-ANOVA debiasing penalty.

Reads the eta (recency-decay mixing weight) and gamma (ANOVA-centering
penalty weight) sweeps logged in assets/sen_eta.xlsx and
assets/sen_gamma.xlsx, and renders one figure per hyperparameter -- 2
figures in total -- each with one Recall@10 panel per dataset and a
single shared legend across backbone models.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/analysis_sensitivity.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
FILES = {
    "eta": ("./assets/sen_eta.xlsx", r"$\eta$"),
    "gamma": ("./assets/sen_gamma.xlsx", r"$\gamma$"),
}
METRIC_COL, METRIC_LABEL = "test_recall_10", "Recall@10"
DATASET_ORDER = ["micro_video", "ml-1m", "kuairand"]
DATASET_LABELS = {"micro_video": "Micro Video", "ml-1m": "MovieLens", "kuairand": "KuaiRand"}
MODEL_ORDER = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
MODEL_LABELS = {
    "mf": "MF", "grurec": "GRU4Rec", "sasrec": "SASRec",
    "tisasrec": "TiSASRec", "fearec": "FEARec", "bsarec": "BSARec",
}
# muted, colorblind-safe palette in the spirit of recent ICLR camera-readies
MODEL_COLORS = {
    "mf": "#3B6FA0", "grurec": "#D98E4A", "sasrec": "#4E9F6E",
    "tisasrec": "#B24B4B", "fearec": "#7B679A", "bsarec": "#6B6B6B",
}
MODEL_MARKERS = {
    "mf": "o", "grurec": "s", "sasrec": "^",
    "tisasrec": "D", "fearec": "v", "bsarec": "P",
}
OUT_DIR = "./figures"


def setup_style():
    mpl.rcParams.update({
        "font.size": 20,
        "font.family": "sans-serif",
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 19,
        "axes.linewidth": 1.1,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_sweep(path, x_col):
    df = pd.read_excel(path)
    df = df[df["dataset"].isin(DATASET_ORDER) & df["model_name"].isin(MODEL_ORDER)].copy()
    for col in [x_col, METRIC_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[x_col, METRIC_COL])
    df = df.drop_duplicates(subset=["dataset", "model_name", x_col], keep="first")
    return df


def plot_sweep(df, x_col, x_label, out_path, n_xticks=4):
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(14.4, 4.6))
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = df[df["dataset"] == dataset]
        x_values = sorted(sub[x_col].unique())
        x_pos = {v: i for i, v in enumerate(x_values)}

        for model in MODEL_ORDER:
            m = sub[sub["model_name"] == model].sort_values(x_col)
            if m.empty:
                continue
            xs = [x_pos[v] for v in m[x_col]]
            ax.plot(
                xs, m[METRIC_COL],
                color=MODEL_COLORS[model], marker=MODEL_MARKERS[model],
                markersize=11, markeredgewidth=1.1, markeredgecolor="white",
                linewidth=3.0, alpha=0.95, label=MODEL_LABELS[model],
            )
        # label only a handful of ticks (evenly spaced by index) so text
        # never has to compete for space in a narrow, multi-panel figure
        ax.set_xticks(range(len(x_values)), minor=True)
        n_lab = min(n_xticks, len(x_values))
        lab_idx = sorted(set(np.linspace(0, len(x_values) - 1, n_lab).round().astype(int)))
        ax.set_xticks(lab_idx)
        ax.set_xticklabels([f"{x_values[i]:g}" for i in lab_idx])
        ax.set_xlabel(x_label, labelpad=8)
        ax.set_title(DATASET_LABELS[dataset], pad=16)
        ax.margins(x=0.08)
        ax.tick_params(width=1.1, length=6)
        ax.tick_params(which="minor", length=3)
    axes[0].set_ylabel(METRIC_LABEL, labelpad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(MODEL_ORDER), markerscale=1.5,
        bbox_to_anchor=(0.5, -0.05), frameon=False, columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.subplots_adjust(wspace=0.32)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path} (+ .pdf)")


#%%
if __name__ == "__main__":
    setup_style()
    for hp_name, (path, x_label) in FILES.items():
        df = load_sweep(path, hp_name)
        out_path = f"{OUT_DIR}/fig_sensitivity_{hp_name}.png"
        plot_sweep(df, hp_name, x_label, out_path)
