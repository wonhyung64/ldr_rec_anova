"""Save Tail Recall@10 / NDCG@10 at eta=0 (utility-only ranking, no retraining) for all
six KuaiRand backbones, mean +/- std over seeds 1-4, as CSV.

Skips the Hawkes/logprob computation entirely (not needed when eta=0), so this is much
faster than the full sweep scripts.

Usage: python analysis_kuairand_eta0_csv.py
Outputs:
  analysis_representation/tail_diag/eta0_tail_per_seed.csv    (24 rows: backbone x seed)
  analysis_representation/tail_diag/eta0_tail_summary.csv     (6 rows: backbone mean+/-std)
"""
import os
import sys
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_kuairand_tail_diagnosis import (
    load_dataset, load_ours_ckpt, build_eval_batch, additional_feat, OUT_DIR,
)

MODELS = ["mf", "grurec", "sasrec", "fearec", "bsarec", "tisasrec"]
SEEDS = [1, 2, 3, 4]


def run_utility_only_scores(model, ds, users, hist_item_np, hist_time_np, model_name, batch_size=256, device="cpu"):
    """f_Psi only (eta=0): cosine(u,v)/tau, i.e. exactly what eta=0 * logprob + 1 * resid reduces to."""
    N, m_item = len(users), ds.m_item
    out = np.zeros((N, m_item), dtype=np.float32)
    with torch.no_grad():
        v_all = F.normalize(model.get_item_repr(torch.arange(m_item, device=device)), dim=-1, eps=1e-8)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            hist_t = torch.tensor(hist_item_np[s:e], dtype=torch.long, device=device)
            feat_t = additional_feat(model_name, users[s:e], hist_item_np[s:e], hist_time_np[s:e], device)
            u = F.normalize(model.encode_user(hist_t, feat_t), dim=-1, eps=1e-8)
            out[s:e] = (torch.matmul(u, v_all.T) / model.tau).cpu().numpy()
    for i, u in enumerate(users):
        out[i, list(ds._allPos[u])] = -9999.0
    return out


def recall_ndcg_at_k(scores, items, k=10):
    true_score = scores[np.arange(len(items)), items]
    rank = (scores > true_score[:, None]).sum(axis=1) + 1  # 1-indexed
    hit = rank <= k
    recall = hit.mean()
    ndcg = np.where(hit, 1.0 / np.log2(rank + 1), 0.0).mean()
    return float(recall), float(ndcg)


def main():
    ds = load_dataset()
    users, items, times, hist_item_np, hist_time_np = build_eval_batch(ds, ds.test_tail_recent_3d_dict)
    print(f"tail_recent_3d: N={len(users)}\n")

    per_seed_rows = []
    for model_name in MODELS:
        for seed in SEEDS:
            t0 = time.time()
            model, eta_default = load_ours_ckpt(model_name, seed, ds)
            scores = run_utility_only_scores(model, ds, users, hist_item_np, hist_time_np, model_name)
            recall10, ndcg10 = recall_ndcg_at_k(scores, items, k=10)
            per_seed_rows.append(dict(backbone=model_name, seed=seed, eta=0.0,
                                       recall_at_10=recall10, ndcg_at_10=ndcg10))
            print(f"{model_name:10s} seed={seed} eta=0  Recall@10={recall10:.4f}  NDCG@10={ndcg10:.4f}  ({time.time()-t0:.1f}s)")

    per_seed_path = f"{OUT_DIR}/eta0_tail_per_seed.csv"
    with open(per_seed_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "seed", "eta", "recall_at_10", "ndcg_at_10"])
        w.writeheader()
        w.writerows(per_seed_rows)
    print("\nsaved", per_seed_path)

    summary_rows = []
    for model_name in MODELS:
        rs = [r["recall_at_10"] for r in per_seed_rows if r["backbone"] == model_name]
        ns = [r["ndcg_at_10"] for r in per_seed_rows if r["backbone"] == model_name]
        summary_rows.append(dict(
            backbone=model_name, eta=0.0, n_seeds=len(rs),
            recall_at_10_mean=float(np.mean(rs)), recall_at_10_std=float(np.std(rs)),
            ndcg_at_10_mean=float(np.mean(ns)), ndcg_at_10_std=float(np.std(ns)),
        ))
    summary_path = f"{OUT_DIR}/eta0_tail_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backbone", "eta", "n_seeds",
                                           "recall_at_10_mean", "recall_at_10_std",
                                           "ndcg_at_10_mean", "ndcg_at_10_std"])
        w.writeheader()
        w.writerows(summary_rows)
    print("saved", summary_path)

    print("\n=== Summary (mean +/- std over 4 seeds), Tail, eta=0 ===")
    for r in summary_rows:
        print(f"{r['backbone']:10s} Recall@10={r['recall_at_10_mean']:.4f}+/-{r['recall_at_10_std']:.4f}  "
              f"NDCG@10={r['ndcg_at_10_mean']:.4f}+/-{r['ndcg_at_10_std']:.4f}")


if __name__ == "__main__":
    main()
