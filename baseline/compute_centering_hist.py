#%%
"""
Histogram of the centering value c(v, t) over (test time, item) pairs, for the
KDE figure (plot_centering_kde.py).

c(v, t_i) = <ubar(t_i), v_hat> / tau is stored in factorized form by
save_centering_values.py (figures/centering_c/*.npz). The full matrix has up to
4e8 entries per checkpoint, so the pairs are binned chunk by chunk on a fixed grid
[-10, 10] (|c| <= 1/tau <= 10 always) with bin width BIN_W. c changes very little
over the test period (std over all pairs vs over items of the time-averaged c
differ by < 4%), so N_TIMES test times (random, the same for all checkpoints of a
dataset) x ALL items are used per checkpoint.

Output: figures/centering_hist.npz with, for every key "{dataset}/{backbone}/{setting}",
the integer counts per bin, plus the shared bin edges ("edges").

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/compute_centering_hist.py
"""
import numpy as np

DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
SETTINGS = ["vanilla", "gamma0", "tuned"]
IN_DIR = "./figures/centering_c"
OUT_PATH = "./figures/centering_hist.npz"
LIM = 10.0
BIN_W = 0.0002
N_TIMES = 2000
CHUNK = 250

rng = np.random.default_rng(0)

#%%
if __name__ == "__main__":
    n_bins = int(round(2 * LIM / BIN_W))
    out = {"edges": np.linspace(-LIM, LIM, n_bins + 1)}
    for dataset in DATASETS:
        n_ctx = np.load(f"{IN_DIR}/contexts_{dataset}.npz")["time"].shape[0]
        rows = np.sort(rng.choice(n_ctx, size=min(N_TIMES, n_ctx), replace=False))
        for backbone in BACKBONES:
            for setting in SETTINGS:
                z = np.load(f"{IN_DIR}/{dataset}_{backbone}_{setting}.npz")
                A, V, tau = z["ubar"][rows], z["V"], float(z["tau"])
                counts = np.zeros(n_bins, dtype=np.int64)
                for s in range(0, len(A), CHUNK):
                    c = A[s:s + CHUNK] @ V.T / tau
                    idx = np.clip(((c + LIM) / BIN_W).astype(np.int64), 0, n_bins - 1)
                    counts += np.bincount(idx.ravel(), minlength=n_bins)
                out[f"{dataset}/{backbone}/{setting}"] = counts
                print(f"[{dataset}|{backbone}|{setting}] pairs={counts.sum():,}", flush=True)
    np.savez_compressed(OUT_PATH, **out)
    print(f"\n[saved] {OUT_PATH}\n[done]")
