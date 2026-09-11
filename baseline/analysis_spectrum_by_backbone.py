#%%
"""
Spectral-collapse-only figure (drops the pairwise-cosine column, which
duplicates the message already carried by the uniformity table): for each
backbone, one wide 1x3 figure (columns = Micro-Video / MovieLens-1M /
KuaiRand, matching the table's column groups) meant to sit directly above
the uniformity table in the paper.

Generates one figure per backbone (all cached from analysis_representation_
extra.py's earlier full run -- no new model inference needed) and ranks them
to suggest the cleanest one to actually use.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_spectrum_by_backbone.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analysis_representation as ar
import analysis_representation_extra as arx

OUT_DIR = "./analysis_representation"
DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
SEEDS = [1, 2, 3, 4]
MAX_COMPONENTS = 25


def collect_pr(dataset_name, backbone, seeds=SEEDS):
    prs = {}
    for approach in ["vanilla", "ours"]:
        vals = []
        for s in seeds:
            h_norm = ar.normalize_rows(arx.load_cached(dataset_name, backbone, approach, s)[0])
            eigvals = arx.second_moment_spectrum(h_norm)
            vals.append(arx.participation_ratio(eigvals))
        prs[approach] = (float(np.mean(vals)), float(np.std(vals)))
    return prs


def make_figure(backbone):
    # One legend shared by all three panels, sitting in its own strip above
    # the column titles -- never competing with the data for space, and
    # avoiding the "floating box in whatever gap happens to be free" look.
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(10.8, 3.35))
    fig.subplots_adjust(wspace=0.28, top=0.80)
    for col, dataset_name in enumerate(DATASETS):
        arx.plot_spectrum_panel(axes[col], dataset_name, backbone=backbone, seeds=SEEDS,
                                 max_components=MAX_COMPONENTS, show_legend=False)
        axes[col].set_title(ar.DISPLAY_DATASET[dataset_name], fontsize=11, pad=8)
        if col > 0:
            axes[col].set_ylabel("")

    handles = [
        Line2D([0], [0], color=ar.VANILLA_COLOR, lw=1.8, marker="o", ms=5,
               markerfacecolor="white", markeredgecolor=ar.VANILLA_COLOR, markeredgewidth=1.1,
               label=ar.VANILLA_LABEL),
        Line2D([0], [0], color=ar.OURS_COLOR, lw=1.8, marker="o", ms=5,
               markerfacecolor="white", markeredgecolor=ar.OURS_COLOR, markeredgewidth=1.1,
               label=ar.OURS_LABEL),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, handlelength=2.4, columnspacing=2.0, fontsize=10.5)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/spectrum_only_{backbone}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_DIR}/spectrum_only_{backbone}.{{pdf,png}}")


def main():
    ar.setup_style()

    all_pr = {}
    for backbone in BACKBONES:
        make_figure(backbone)
        all_pr[backbone] = {d: collect_pr(d, backbone) for d in DATASETS}

    print("\n########## participation ratio summary (mean +- std over 4 seeds) ##########")
    print(f"{'backbone':>9s} | " + " | ".join(f"{ar.DISPLAY_DATASET[d]:^22s}" for d in DATASETS))
    for backbone in BACKBONES:
        cells = []
        for d in DATASETS:
            v_mean, v_std = all_pr[backbone][d]["vanilla"]
            o_mean, o_std = all_pr[backbone][d]["ours"]
            cells.append(f"V={v_mean:4.1f}  O={o_mean:5.1f}")
        print(f"{ar.DISPLAY_BACKBONE[backbone]:>9s} | " + " | ".join(f"{c:^22s}" for c in cells))

    # ranking: want the *worst-case* (min across datasets) Ours PR to be as
    # large as possible -- i.e. a consistently strong story on every dataset,
    # not one that only looks good on its best dataset.
    print("\n########## ranking (by worst-case Ours PR across the 3 datasets) ##########")
    scores = []
    for backbone in BACKBONES:
        ours_prs = [all_pr[backbone][d]["ours"][0] for d in DATASETS]
        vanilla_prs = [all_pr[backbone][d]["vanilla"][0] for d in DATASETS]
        worst_ours = min(ours_prs)
        worst_ratio = min(o / v for o, v in zip(ours_prs, vanilla_prs))
        scores.append((backbone, worst_ours, worst_ratio, ours_prs))
        print(f"  {ar.DISPLAY_BACKBONE[backbone]:>9s}: worst-case Ours PR={worst_ours:6.1f}  "
              f"worst-case Ours/Vanilla ratio={worst_ratio:7.1f}x  per-dataset Ours PR={['%.1f' % p for p in ours_prs]}")
    best = max(scores, key=lambda x: x[1])
    print(f"\n  ==> suggested backbone: {ar.DISPLAY_BACKBONE[best[0]]} "
          f"(worst-case Ours PR={best[1]:.1f}, i.e. even on its weakest dataset Ours keeps "
          f"{best[1]:.1f}x the effective rank of Vanilla's ~1)")


if __name__ == "__main__":
    main()
