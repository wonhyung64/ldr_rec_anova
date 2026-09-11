#%%
"""
Exploratory extra figures for the "Analysis of Representation" section,
drawing on visualization conventions from the representation-degeneration /
anisotropy literature and popularity-debiasing papers, as an alternative or
complement to representation_geometry_{full,compact}:

  A. Uncentered second-moment eigenvalue spectrum (cumulative energy vs.
     component index), Vanilla vs Ours. This generalizes the mean-resultant
     length R to all d directions at once and is the same family of
     diagnostic used to characterize "dimensional collapse" (Jing et al.,
     2022) and representation degeneration (Gao et al., 2019). We use the
     UNCENTERED second moment M = E[h h^T] (not the covariance) because
     centering would subtract out exactly the shared direction under study
     -- the same reason centered PCA showed no signal earlier in this
     project.

  B. Pairwise cosine-similarity distribution between random pairs of
     representations (the anisotropy-literature "self-similarity" plot,
     e.g. Ethayarajh, 2019) -- a more standard, model-agnostic relative of
     the L_uni computation already reported in the table.

  C. Popularity-colored scatter on the SAME jointly-fit PCA plane as
     representation_geometry_full's panel (a), following the visualization
     convention used throughout the popularity-debiasing literature
     (DICE/TIDE/PDA-style "before/after" embedding plots): color each point
     by the popularity of the item it is currently predicting, for Vanilla
     and Ours side by side, on shared axes. This is a candidate replacement
     for the (weak, ceiling-effect-limited) temporal-bucket correlation
     panel (c) of the main figure.

  D. A Wang & Isola (2020)-style synthesis scatter: x = mean resultant
     length R, y = uniformity loss L_uni, one point per (dataset, backbone,
     approach, seed), for ALL SIX backbones. This ties the new geometric
     diagnostic (R) directly back to the uniformity table already in the
     paper and shows the Vanilla/Ours separation holds architecture-wide.

This script is exploratory -- it draws all four candidates so they can be
compared before deciding which to keep. It reuses every model-building,
checkpoint-path, caching and metric function from analysis_representation.py
without duplicating them.

Usage (from the repo root):
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_representation_extra.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

import analysis_representation as ar

OUT_DIR = "./analysis_representation"
DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES_ALL = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
PRIMARY_BACKBONE = "sasrec"
SEEDS = [1, 2, 3, 4]
N_SAMPLE = 4000
N_TIME_BUCKETS = 8
DEVICE = "cpu"

POP_CMAP = "viridis"


def load_cached(dataset_name, backbone, approach, seed, cache_dir=OUT_DIR):
    path = f"{cache_dir}/representations_{dataset_name}_{backbone}_{approach}_{seed}.npz"
    data = np.load(path)
    return data["h"], data["t"], data["u"], data["v"]


def median_seed_from_csv(out_dir, dataset_name, backbone, seeds):
    """Recover the same median-performance-seed choice used in
    representation_geometry_full (by +Ours L_uni) from the cached CSV,
    so the extra figures use an identical, documented seed."""
    import csv
    vals = {}
    with open(f"{out_dir}/uniformity_summary.csv") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset_name and row["backbone"] == backbone and row["approach"] == "ours":
                vals[int(row["seed"])] = float(row["L_uni"])
    order = sorted(seeds, key=lambda s: vals[s])
    return order[len(order) // 2]


# ----------------------------------------------------------------------
# A. Uncentered second-moment eigenvalue spectrum
# ----------------------------------------------------------------------
def second_moment_spectrum(h_norm):
    M = (h_norm.T @ h_norm) / len(h_norm)
    eigvals = np.linalg.eigvalsh(M)[::-1]
    return np.clip(eigvals, 0, None)


def participation_ratio(eigvals):
    """Effective rank / participation ratio: (sum lambda)^2 / sum(lambda^2).
    1 = all energy in one direction (fully collapsed); d = perfectly isotropic."""
    s = eigvals.sum()
    return float(s ** 2 / np.sum(eigvals ** 2))


def plot_spectrum_panel(ax, dataset_name, backbone=PRIMARY_BACKBONE, seeds=SEEDS, max_components=25, show_legend=False):
    # IMPORTANT: the second-moment matrix is computed WITHIN each seed
    # separately, never on seeds pooled together. Vanilla's four independently
    # -trained runs each collapse onto their OWN arbitrary shared direction
    # (different random init); concatenating seeds before forming M = E[hh^T]
    # would make M see ~4 nearly-orthogonal "collapse axes" and report an
    # inflated participation ratio (~4) that reflects seed-to-seed disagreement,
    # not within-model anisotropy. Each seed's curve/PR is computed alone, then
    # averaged (mean line, min-max band across seeds).
    pr_text = []
    for approach, color, label in [("vanilla", ar.VANILLA_COLOR, ar.VANILLA_LABEL),
                                    ("ours", ar.OURS_COLOR, ar.OURS_LABEL)]:
        curves, prs = [], []
        for s in seeds:
            h_norm = ar.normalize_rows(load_cached(dataset_name, backbone, approach, s)[0])
            eigvals = second_moment_spectrum(h_norm)
            prs.append(participation_ratio(eigvals))
            k = min(max_components, len(eigvals))
            curves.append(np.cumsum(eigvals)[:k] / np.sum(eigvals))
        curves = np.array(curves)
        ks = np.arange(1, curves.shape[1] + 1)
        mean_curve = curves.mean(axis=0)
        ax.fill_between(ks, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(ks, mean_curve, color=color, lw=1.8, marker="o", ms=3.4,
                 markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1,
                 label=label, zorder=3, clip_on=False)
        pr_text.append(f"{label}: PR$\\,\\approx\\,${np.mean(prs):.1f}$\\pm${np.std(prs):.1f}")
    ax.set_xlabel("component index $k$")
    ax.set_ylabel("cumulative energy")
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(0.5, max_components + 0.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25][: max_components // 5 + 2])
    ax.text(0.97, 0.05, "\n".join(pr_text), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.6, color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    if show_legend:
        # sits in the empty gap between the flat Vanilla line (near the top)
        # and the rising Ours curve (well below it for all k in this range)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.82), frameon=False,
                  handlelength=1.4, borderaxespad=0.3)


# ----------------------------------------------------------------------
# B. Pairwise cosine-similarity distribution (anisotropy style)
# ----------------------------------------------------------------------
def plot_cosine_panel(ax, dataset_name, backbone=PRIMARY_BACKBONE, seeds=SEEDS, n_pairs=20000, show_legend=False, rng=None):
    # Vanilla's distribution is a near-delta spike (std ~0.03) while Ours spans
    # most of [-1, 1]; on a shared linear density scale the spike dwarfs Ours'
    # peak entirely. Each curve is normalized to its own max (peak height = 1)
    # so both SHAPES are readable together -- absolute density is not
    # comparable across curves, only within one, which is stated in the axis
    # label and figure caption.
    # Pairs are drawn WITHIN each seed only (never across seeds -- see the
    # same note in plot_spectrum_panel), then the 4 seeds' density curves are
    # averaged.
    rng = rng or np.random.default_rng(0)
    grid = np.linspace(-1, 1, 500)
    from scipy.stats import gaussian_kde
    mean_text = []
    for approach, color, label in [("vanilla", ar.VANILLA_COLOR, ar.VANILLA_LABEL),
                                    ("ours", ar.OURS_COLOR, ar.OURS_LABEL)]:
        dens_list, cos_means = [], []
        for s in seeds:
            h_norm = ar.normalize_rows(load_cached(dataset_name, backbone, approach, s)[0])
            n = len(h_norm)
            idx = rng.integers(0, n, size=(n_pairs, 2))
            idx = idx[idx[:, 0] != idx[:, 1]]
            cos = np.sum(h_norm[idx[:, 0]] * h_norm[idx[:, 1]], axis=1)
            cos_means.append(cos.mean())
            dens_list.append(gaussian_kde(cos, bw_method=0.06)(grid))
        dens_arr = np.array(dens_list)
        peak = dens_arr.mean(axis=0).max()
        mean_dens = dens_arr.mean(axis=0) / peak
        band_lo, band_hi = dens_arr.min(axis=0) / peak, dens_arr.max(axis=0) / peak
        ax.fill_between(grid, band_lo, band_hi, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(grid, mean_dens, color=color, lw=1.6, label=label, zorder=3)
        mean_text.append(f"{label}: mean$\\,=\\,${np.mean(cos_means):.2f}$\\pm${np.std(cos_means):.2f}")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"pairwise cosine similarity $\bar h_i^\top \bar h_j$ ($i\ne j$)")
    ax.set_ylabel("density (peak-normalized)")
    ax.set_yticks([])
    ax.text(0.03, 0.97, "\n".join(mean_text), transform=ax.transAxes, ha="left", va="top",
            fontsize=7.6, color="0.15",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    if show_legend:
        ax.legend(loc="upper right", frameon=False, handlelength=1.4, borderaxespad=0.3)


# ----------------------------------------------------------------------
# C. Popularity-colored scatter on the shared PCA plane
# ----------------------------------------------------------------------
def load_popularity_array(dataset_name):
    pop_dict = np.load(f"./data/{dataset_name}/item_popularity_recent_3d.npy", allow_pickle=True).item()
    m_item = max(pop_dict.keys()) + 1
    arr = np.zeros(m_item, dtype=np.float64)
    for item, cnt in pop_dict.items():
        arr[item] = cnt
    return arr


def plot_popularity_pca_row(axes_row, dataset_name, backbone=PRIMARY_BACKBONE, seeds=SEEDS, out_dir=OUT_DIR):
    from sklearn.decomposition import PCA
    seed = median_seed_from_csv(out_dir, dataset_name, backbone, seeds)
    h_van, t_van, u_van, v_van = load_cached(dataset_name, backbone, "vanilla", seed)
    h_ours, t_ours, u_ours, v_ours = load_cached(dataset_name, backbone, "ours", seed)
    h_van, h_ours = ar.normalize_rows(h_van), ar.normalize_rows(h_ours)

    n_common = min(len(h_van), len(h_ours))
    concat = np.concatenate([h_van[:n_common], h_ours[:n_common]], axis=0)
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(concat)
    proj_van, proj_ours = proj[:n_common], proj[n_common:]
    lim = np.abs(proj).max() * 1.1

    pop_arr = load_popularity_array(dataset_name)
    log_pop_van = np.log1p(pop_arr[v_van[:n_common]])
    log_pop_ours = np.log1p(pop_arr[v_ours[:n_common]])
    vmin = min(log_pop_van.min(), log_pop_ours.min())
    vmax = max(log_pop_van.max(), log_pop_ours.max())

    sca = None
    for ax, proj_xy, log_pop, title in [
        (axes_row[0], proj_van, log_pop_van, f"{ar.DISPLAY_DATASET[dataset_name]} -- Vanilla"),
        (axes_row[1], proj_ours, log_pop_ours, f"{ar.DISPLAY_DATASET[dataset_name]} -- Ours"),
    ]:
        sca = ax.scatter(proj_xy[:, 0], proj_xy[:, 1], c=log_pop, cmap=POP_CMAP, vmin=vmin, vmax=vmax,
                          s=5, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    return sca


# ----------------------------------------------------------------------
# D. Wang & Isola-style (R, L_uni) synthesis scatter, all 6 backbones
# ----------------------------------------------------------------------
def gather_all_backbones(datasets, backbones, seeds, cache_dir=OUT_DIR):
    rng = np.random.default_rng(0)
    records = []
    for dataset_name in datasets:
        for backbone in backbones:
            _, _, _, summ = ar.run_dataset(dataset_name, backbone, seeds, N_SAMPLE, N_TIME_BUCKETS,
                                            cache_dir, DEVICE, rng)
            for approach in ["vanilla", "ours"]:
                for seed in seeds:
                    records.append(dict(
                        dataset=dataset_name, backbone=backbone, approach=approach, seed=seed,
                        L_uni=summ["l_uni"][approach][seed],
                        R=summ["align"][approach][seed]["R"],
                    ))
    return records


DATASET_MARKER = {"micro_video": "o", "ml-1m": "s", "kuairand": "^"}


def plot_synthesis_scatter(ax, records):
    from scipy.stats import pearsonr

    all_R = np.array([r["R"] for r in records])
    all_L = np.array([r["L_uni"] for r in records])
    r_all, _ = pearsonr(all_R, all_L)

    # faint global trend guide across the full R range (both approaches pooled)
    coeffs = np.polyfit(all_R, all_L, 1)
    xs = np.linspace(all_R.min(), all_R.max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color="0.6", lw=1.0, ls=(0, (5, 3)), zorder=1)

    for approach, color, label in [("vanilla", ar.VANILLA_COLOR, ar.VANILLA_LABEL),
                                    ("ours", ar.OURS_COLOR, ar.OURS_LABEL)]:
        recs = [r for r in records if r["approach"] == approach]
        for dataset_name, marker in DATASET_MARKER.items():
            seed_recs = [r for r in recs if r["dataset"] == dataset_name]
            Rs = np.array([r["R"] for r in seed_recs])
            Ls = np.array([r["L_uni"] for r in seed_recs])
            ax.scatter(Rs, Ls, s=16, color=color, marker=marker, alpha=0.30, linewidths=0, zorder=2)

        # bold mean marker per (dataset, backbone)
        groups = {}
        for r in recs:
            groups.setdefault((r["dataset"], r["backbone"]), []).append(r)
        for (dataset_name, backbone), gs in groups.items():
            mR, mL = np.mean([g["R"] for g in gs]), np.mean([g["L_uni"] for g in gs])
            ax.scatter([mR], [mL], s=48, color=color, marker=DATASET_MARKER[dataset_name],
                       alpha=0.95, edgecolors="white", linewidths=0.7, zorder=5)

    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(r"mean resultant length $R$")
    ax.set_ylabel(r"uniformity loss $\mathcal{L}_{uni}$")

    # two-part legend (color = approach, marker shape = dataset) placed in the
    # large empty region between the two clusters -- never overlaps data
    color_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=ar.VANILLA_COLOR,
                             markeredgecolor="white", markersize=7, label=ar.VANILLA_LABEL),
                      Line2D([0], [0], marker="o", color="none", markerfacecolor=ar.OURS_COLOR,
                             markeredgecolor="white", markersize=7, label=ar.OURS_LABEL)]
    marker_handles = [Line2D([0], [0], marker=m, color="0.35", linestyle="none", markersize=6,
                              label=ar.DISPLAY_DATASET[d]) for d, m in DATASET_MARKER.items()]
    leg1 = ax.legend(handles=color_handles, loc="upper left", bbox_to_anchor=(0.30, 0.92),
                      frameon=False, handlelength=1.0, title="approach", title_fontsize=8, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="upper left", bbox_to_anchor=(0.30, 0.62),
              frameon=False, handlelength=1.0, title="dataset", title_fontsize=8, fontsize=8)
    ax.text(0.60, 0.08, f"Pearson $r={r_all:.2f}$\n(pooled, all points)", transform=ax.transAxes,
            fontsize=8, color="0.25", va="bottom", ha="left")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ar.setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- A + B: spectrum & cosine-similarity, 3 dataset rows x 2 cols ----
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(8.6, 2.9 * len(DATASETS)))
    fig.subplots_adjust(wspace=0.30, hspace=0.42)
    for row, dataset_name in enumerate(DATASETS):
        plot_spectrum_panel(axes[row, 0], dataset_name, show_legend=(row == 0))
        plot_cosine_panel(axes[row, 1], dataset_name, show_legend=(row == 0))
        axes[row, 0].set_ylabel(f"{ar.DISPLAY_DATASET[dataset_name]}\n" + axes[row, 0].get_ylabel())
    fig.text(0.27, 0.995, "(a) spectral collapse", ha="center", va="top", fontsize=11)
    fig.text(0.76, 0.995, "(b) pairwise cosine similarity", ha="center", va="top", fontsize=11)
    fig.suptitle(f"Anisotropy diagnostics ({ar.DISPLAY_BACKBONE[PRIMARY_BACKBONE]})", y=1.045, fontsize=12.5)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/extra_spectrum_cosine.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/extra_spectrum_cosine.{{pdf,png}}")

    # ---- C: popularity-colored PCA, 3 dataset rows x 2 cols (Vanilla | Ours) ----
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(6.6, 3.1 * len(DATASETS)))
    sca = None
    for row, dataset_name in enumerate(DATASETS):
        sca = plot_popularity_pca_row(axes[row], dataset_name)
    fig.suptitle(f"User-context representations colored by target-item popularity ({ar.DISPLAY_BACKBONE[PRIMARY_BACKBONE]})",
                 y=1.01)
    fig.tight_layout()
    cbar = fig.colorbar(sca, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(r"$\log(1 + \mathrm{popularity})$ of the target item")
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/extra_popularity_colored_pca.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/extra_popularity_colored_pca.{{pdf,png}}")

    # ---- D: (R, L_uni) synthesis scatter across all 6 backbones ----
    print("\n[gathering all 6 backbones for the synthesis scatter -- this loads 5 more backbones' checkpoints]")
    records = gather_all_backbones(DATASETS, BACKBONES_ALL, SEEDS)
    import csv
    with open(f"{OUT_DIR}/synthesis_R_vs_Luni.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "approach", "seed", "R", "L_uni"])
        for r in records:
            w.writerow([r["dataset"], r["backbone"], r["approach"], r["seed"], f"{r['R']:.6f}", f"{r['L_uni']:.6f}"])
    print(f"[saved] {OUT_DIR}/synthesis_R_vs_Luni.csv")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    plot_synthesis_scatter(ax, records)
    ax.set_title("Representation concentration vs. uniformity", fontsize=11.5)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/extra_synthesis_scatter.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/extra_synthesis_scatter.{{pdf,png}}")


if __name__ == "__main__":
    main()
