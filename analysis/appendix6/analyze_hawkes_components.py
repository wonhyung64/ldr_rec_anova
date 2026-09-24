"""Analysis 3 (task spec Sec. 7): why did the Hawkes model learn this way?

Decomposes lambda_v(t) = mu_v + Exc_v(t), Exc_v(t) = alpha_v * sum_{t_j<t} exp(-beta(t-t_j)),
at the same daily snapshot grid used by Analyses 1/2, for SASRec+Ours (seed=1) on all
three datasets.

3.1 Excitation ratio R_exc(v,t) = Exc_v(t) / lambda_v(t): global mean/median, Head vs Tail
    item mean (using the repository's own static Head/Tail item labels,
    common.load_item_head_tail_labels), and top-1%-by-training-frequency items vs the rest.
3.2 Learned decay timescale: beta and half_life = log(2)/beta, in days and hours.
3.3 mu_v, mean Exc_v(t), mean lambda_v(t), and excitation ratio, binned by item training-
    frequency decile (10 bins).

Outputs:
    results/appendix6/excitation_metrics.csv           (per dataset x snapshot x item-group summary)
    results/appendix6/excitation_ratio_by_item.csv      (per dataset x item, time-averaged)
    results/appendix6/decay_timescale_summary.csv       (per dataset: beta, half-life)
    results/appendix6/excitation_by_frequency_decile.csv (per dataset x decile)
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from analyze_empirical_popularity import snapshot_grid

MODEL_NAME = "sasrec"
SEED = 1
EPS = 1e-12


def main():
    decay_rows = []
    freq_bin_rows = []
    item_avg_rows = []

    for dataset_name in C.DATASETS:
        t0 = time.time()
        ds = C.load_dataset(dataset_name, rescale_item_time_array=True)
        grid, tmin, tmax = snapshot_grid(ds)
        model, eta_default = C.load_ours_ckpt(dataset_name, MODEL_NAME, SEED, ds)

        # mu/alpha/beta once; excitation term recomputed per snapshot (batched)
        with torch.no_grad():
            mu, alpha, beta = model.prior_parameters_from_embeddings()
        mu = mu.numpy(); alpha = alpha.numpy(); beta = float(beta.numpy())
        half_life_days = np.log(2) / beta
        decay_rows.append(dict(dataset=dataset_name, beta_per_day=beta,
                                half_life_days=half_life_days, half_life_hours=half_life_days * 24))

        item_time_array = ds.item_time_array.astype(np.float32)
        m_item = ds.m_item
        exc_sum = np.zeros(m_item, dtype=np.float64)
        lam_sum = np.zeros(m_item, dtype=np.float64)
        ratio_sum = np.zeros(m_item, dtype=np.float64)
        n_valid = np.zeros(m_item, dtype=np.int64)  # snapshots where lambda>0 (always true, mu>0)

        chunk = 64
        for s in range(0, len(grid), chunk):
            e = min(s + chunk, len(grid))
            t_batch = grid[s:e].astype(np.float32).reshape(-1, 1, 1)
            mask = item_time_array[None, :, :] < t_batch
            delta = np.clip(t_batch - item_time_array[None, :, :], 0.0, None)
            h = (np.exp(-beta * delta) * mask).sum(-1)          # (chunk, m_item)
            exc = alpha[None, :] * h
            lam = mu[None, :] + exc
            ratio = exc / np.maximum(lam, EPS)
            exc_sum += exc.sum(0)
            lam_sum += lam.sum(0)
            ratio_sum += ratio.sum(0)
            n_valid += (e - s)

        mean_exc = exc_sum / n_valid
        mean_lam = lam_sum / n_valid
        mean_ratio = ratio_sum / n_valid  # time-average of the ratio (not ratio of means)

        # training frequency per item
        train_dict = np.load(f"{C.REPO_ROOT}/data/{dataset_name}/training_dict.npy", allow_pickle=True).item()
        cnt = Counter()
        for u, items in train_dict.items():
            cnt.update(items)
        freq = np.array([cnt.get(i, 0) for i in range(m_item)], dtype=np.float64)

        head_items, tail_items = C.load_item_head_tail_labels(dataset_name, kind="overall")
        head_mask = np.array([i in head_items for i in range(m_item)])
        tail_mask = np.array([i in tail_items for i in range(m_item)])
        top1pct_thresh = np.percentile(freq[freq > 0], 99) if (freq > 0).any() else np.inf
        top1pct_mask = freq >= top1pct_thresh

        print(f"[{dataset_name}] beta={beta:.4f} half_life={half_life_days:.4f}d "
              f"global_mean_ratio={mean_ratio.mean():.4f} median_ratio={np.median(mean_ratio):.4f} "
              f"head_mean={mean_ratio[head_mask].mean() if head_mask.any() else np.nan:.4f} "
              f"tail_mean={mean_ratio[tail_mask].mean() if tail_mask.any() else np.nan:.4f} "
              f"top1pct_freq_mean={mean_ratio[top1pct_mask].mean() if top1pct_mask.any() else np.nan:.4f} "
              f"rest_mean={mean_ratio[~top1pct_mask].mean():.4f} "
              f"({time.time()-t0:.1f}s)")

        decay_rows[-1].update(
            global_mean_ratio=float(mean_ratio.mean()), global_median_ratio=float(np.median(mean_ratio)),
            head_mean_ratio=float(mean_ratio[head_mask].mean()) if head_mask.any() else np.nan,
            tail_mean_ratio=float(mean_ratio[tail_mask].mean()) if tail_mask.any() else np.nan,
            top1pct_freq_mean_ratio=float(mean_ratio[top1pct_mask].mean()) if top1pct_mask.any() else np.nan,
            rest_mean_ratio=float(mean_ratio[~top1pct_mask].mean()),
        )

        for i in range(m_item):
            item_avg_rows.append(dict(dataset=dataset_name, item=i, train_freq=freq[i],
                                       mu=float(mu[i]), mean_exc=float(mean_exc[i]),
                                       mean_lambda=float(mean_lam[i]), mean_ratio=float(mean_ratio[i]),
                                       is_head=bool(head_mask[i]), is_tail=bool(tail_mask[i])))

        # frequency decile bins (only over items with >=1 training interaction)
        has_freq = freq > 0
        deciles = pd.qcut(freq[has_freq], 10, labels=False, duplicates="drop")
        idx_has = np.where(has_freq)[0]
        for d in sorted(np.unique(deciles)):
            sel = idx_has[deciles == d]
            freq_bin_rows.append(dict(
                dataset=dataset_name, decile=int(d), n_items=len(sel),
                freq_min=float(freq[sel].min()), freq_max=float(freq[sel].max()), freq_mean=float(freq[sel].mean()),
                mu_mean=float(mu[sel].mean()), mean_exc_mean=float(mean_exc[sel].mean()),
                mean_lambda_mean=float(mean_lam[sel].mean()), mean_ratio_mean=float(mean_ratio[sel].mean()),
            ))

    pd.DataFrame(decay_rows).to_csv(f"{C.RESULTS_DIR}/decay_timescale_summary.csv", index=False)
    pd.DataFrame(item_avg_rows).to_csv(f"{C.RESULTS_DIR}/excitation_ratio_by_item.csv", index=False)
    pd.DataFrame(freq_bin_rows).to_csv(f"{C.RESULTS_DIR}/excitation_by_frequency_decile.csv", index=False)

    print("\n=== Decay timescale & excitation ratio summary ===")
    print(pd.DataFrame(decay_rows).to_string(index=False))
    print(f"\nsaved to {C.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
