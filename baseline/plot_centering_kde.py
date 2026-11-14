#%%
"""
KDE of the centering value c(v, t) of Assumption 1 (should concentrate at 0):
one figure per backbone, 1 x 3 panels (Micro-Video, MovieLens-1M, KuaiRand), and
in every panel the distributions of Vanilla, Ours (gamma=0) and Ours (gamma=gamma*).

The KDE is a Gaussian smoothing of the exact histogram of c over (test time, item)
pairs (compute_centering_hist.py) with bandwidth BW_FRAC x std of that curve.
Colors follow the paper (Vanilla blue / Ours orange, seaborn "deep" family).

Run from the repo root (after compute_centering_hist.py):
    /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/plot_centering_kde.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from analysis_sensitivity import (
    DATASET_ORDER, DATASET_LABELS, setup_style,
)

HIST_PATH = "./figures/centering_hist.npz"
OUT_TMPL = "./figures/fig_centering_kde_{backbone}"
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
SETTINGS = ["vanilla", "gamma0", "tuned"]
COLORS = {"vanilla": "#4C72B0", "gamma0": "#55A868", "tuned": "#DD8452"}
LABELS = {"vanilla": "Vanilla", "gamma0": r"Ours ($\gamma=0$)", "tuned": r"Ours ($\gamma=\gamma^*$)"}
LSTYLE = {"vanilla": "-", "gamma0": "-", "tuned": "-"}
ZORDER = {"vanilla": 2, "gamma0": 3, "tuned": 4}
BW_FRAC = 0.1
X_COVER = 4.0   # x-range covers mean +- X_COVER std of every curve
YSCALE = "log"


def kde_from_hist(counts, edges, bw_frac=BW_FRAC):
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = counts / counts.sum()
    mean = (w * centers).sum()
    std = np.sqrt((w * (centers - mean) ** 2).sum())
    bin_w = edges[1] - edges[0]
    dens = gaussian_filter1d(w / bin_w, sigma=max(bw_frac * std, bin_w) / bin_w, mode="constant")
    return centers, dens, mean, std


def plot_backbone(hist, backbone, out_base, yscale=YSCALE):
    edges = hist["edges"]
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(14.4, 4.3))
    for ax, dataset in zip(axes, DATASET_ORDER):
        curves = {s: kde_from_hist(hist[f"{dataset}/{backbone}/{s}"], edges) for s in SETTINGS}
        lo = min(m - X_COVER * sd for (_, _, m, sd) in curves.values())
        hi = max(m + X_COVER * sd for (_, _, m, sd) in curves.values())
        lo, hi = min(lo, -0.1 * (hi - lo)), max(hi, 0.1 * (hi - lo))
        for s in SETTINGS:
            x, d, m, sd = curves[s]
            sel = (x >= lo) & (x <= hi)
            ax.fill_between(x[sel], d[sel], color=COLORS[s], alpha=0.18, linewidth=0, zorder=ZORDER[s] - 0.5)
            ax.plot(x[sel], d[sel], color=COLORS[s], linestyle=LSTYLE[s], linewidth=2.6, zorder=ZORDER[s])
        ax.axvline(0.0, color="0.45", linestyle=(0, (3, 2)), linewidth=1.2, zorder=1)
        ax.set_xlim(lo, hi)
        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-3)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel(r"$\bar{f}_{v,t}$")
        ax.tick_params(width=1.1, length=6)
    axes[0].set_ylabel("Density (log)" if yscale == "log" else "Density")
    handles = [Line2D([0], [0], color=COLORS[s], linewidth=3.0, label=LABELS[s]) for s in SETTINGS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


#%%
if __name__ == "__main__":
    setup_style()
    plt.rcParams["mathtext.fontset"] = "cm"  # LaTeX-like math font for the axis symbol / legend
    hist = np.load(HIST_PATH)
    for backbone in BACKBONES:
        base = OUT_TMPL.format(backbone=backbone)
        plot_backbone(hist, backbone, base)
        print(f"[saved] {base}.png/.pdf")
