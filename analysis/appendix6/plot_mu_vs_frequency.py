"""Figure D (new, for the finalized/trimmed appendix narrative): learned Hawkes
background rate mu_v against item training-frequency percentile, for all three datasets,
on the same axes -- the direct visual evidence for the "discontinuous ~600x jump in the
top frequency decile, only on KuaiRand" claim (Sec. 7.3 of the original task spec /
Analysis 3).

Uses equal-*count* percentile bins (not qcut on raw frequency, which collapses on
Micro-Video's massive tie count at freq=1) so every dataset gets the same number of
bins regardless of duplicate frequency values, with extra resolution in the top decile
(90-100%) to show exactly where the jump happens.

Reads only results/appendix6/excitation_ratio_by_item.csv (already computed by
analyze_hawkes_components.py; no model is re-run).

Outputs:
    figures/appendix6/fig_mu_vs_frequency.pdf / .png
    results/appendix6/mu_vs_frequency_bins.csv
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
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "lines.linewidth": 2.0, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})

# Coarse bins for the bulk (0-90th percentile, 10 bins), fine bins for the top decile
# (90-100th percentile, 10 more bins) where the KuaiRand discontinuity lives.
BULK_EDGES = np.linspace(0, 90, 10)          # 0,10,...,80,90 (9 bins over 0-90)
TOP_EDGES = np.linspace(90, 100, 11)         # 90,91,...,100 (10 bins over 90-100)
PCT_EDGES = np.unique(np.concatenate([BULK_EDGES, TOP_EDGES]))


def bin_by_percentile(freq, mu):
    order = np.argsort(freq)  # ascending: low-frequency first, percentile 0 = rarest
    freq_sorted = freq[order]
    mu_sorted = mu[order]
    n = len(freq_sorted)
    ranks_pct = (np.arange(1, n + 1) / n) * 100.0

    rows = []
    for lo, hi in zip(PCT_EDGES[:-1], PCT_EDGES[1:]):
        sel = (ranks_pct > lo) & (ranks_pct <= hi)
        if sel.sum() == 0:
            continue
        rows.append(dict(pct_lo=lo, pct_hi=hi, pct_mid=(lo + hi) / 2, n_items=int(sel.sum()),
                          freq_mean=float(freq_sorted[sel].mean()), mu_mean=float(mu_sorted[sel].mean()),
                          mu_median=float(np.median(mu_sorted[sel]))))
    return pd.DataFrame(rows)


def main():
    item_df = pd.read_csv(f"{C.RESULTS_DIR}/excitation_ratio_by_item.csv")

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    all_bins = []
    for ds in C.DATASETS:
        sub = item_df[item_df.dataset == ds]
        freq = sub.train_freq.values.astype(np.float64)
        mu = sub.mu.values.astype(np.float64)
        bins = bin_by_percentile(freq, mu)
        bins["dataset"] = ds
        all_bins.append(bins)
        ax.plot(bins.pct_mid, bins.mu_mean, color=C.DATASET_COLOR[ds], marker=C.DATASET_MARKER[ds],
                markersize=4.5, label=C.DATASET_LABELS[ds])

    ax.set_yscale("log")
    ax.set_xlabel("Item training-frequency percentile\n(0 = rarest, 100 = most frequent)")
    ax.set_ylabel(r"Mean learned background rate $\mu_v$ (log scale)")
    ax.set_title("Learned $\\mu_v$ vs. item frequency percentile\n(fine bins in the top decile)")
    ax.axvspan(90, 100, color="gray", alpha=0.08, zorder=0)
    ax.text(95, ax.get_ylim()[0] * 2, "top\n10%", ha="center", fontsize=8, color="gray")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    fig.savefig(f"{C.FIGURES_DIR}/fig_mu_vs_frequency.pdf")
    fig.savefig(f"{C.FIGURES_DIR}/fig_mu_vs_frequency.png", dpi=200)
    print("saved", f"{C.FIGURES_DIR}/fig_mu_vs_frequency.pdf/.png")

    out = pd.concat(all_bins, ignore_index=True)
    out.to_csv(f"{C.RESULTS_DIR}/mu_vs_frequency_bins.csv", index=False)

    print("\n=== Top-decile mu_mean jump (last two fine bins vs. 80-90th pct bin) ===")
    for ds in C.DATASETS:
        sub = out[out.dataset == ds].sort_values("pct_mid")
        b_80_90 = sub[(sub.pct_lo == 80) & (sub.pct_hi == 90)]
        b_99_100 = sub[(sub.pct_lo == 99) & (sub.pct_hi == 100)]
        if len(b_80_90) and len(b_99_100):
            v0 = b_80_90.mu_mean.values[0]
            v1 = b_99_100.mu_mean.values[0]
            print(f"{ds:12s} 80-90th pct mu_mean={v0:.5f}  99-100th pct mu_mean={v1:.5f}  ratio={v1/v0:.1f}x")
    print(f"\nsaved {C.RESULTS_DIR}/mu_vs_frequency_bins.csv")


if __name__ == "__main__":
    main()
