"""Analysis 6 (task spec Sec. 11): sanity check -- is the degradation specific to the
Hawkes specification, or would any popularity signal combined at inference cause it?

No new recommendation model is trained. The already-trained utility checkpoint f_Psi
(SASRec+Ours, KuaiRand, seed=1) is reused unchanged; only the inference-time *popularity*
term is swapped:

    A. Utility only:              score = f_Psi
    B. Learned Hawkes popularity:  score = eta*log(pi_Hawkes(v|t)) + (1-eta)*f_Psi
    C. 3-day sliding-window:       score = eta*log(pi_3d(v|t))     + (1-eta)*f_Psi
                                    pi_3d(v|t) propto count(v, [t-3d, t)) + eps
    D. Static training popularity: score = eta*log(pi_static(v))  + (1-eta)*f_Psi
                                    pi_static(v) propto training_count(v) + eps

All four use the *same* eta (the trained default for SASRec/KuaiRand, eta=0.5) and the
same full-ranking protocol / seen-item masking as Table 1/4. eps is a fixed numerical-
stability constant (NOT tuned), set once and reported.

Outputs:
    results/appendix6/popularity_substitution.csv   (dataset x split x popularity_source x Recall/NDCG@10)
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from analyze_empirical_popularity import build_item_time_dict

MODEL_NAME = "sasrec"
SEED = 1
EPS = 1e-6  # fixed numerical-stability additive smoothing, not tuned
WINDOW_DAYS = 3.0


def local_3d_counts_at_times(item_times, query_times, m_item):
    n = len(query_times)
    out = np.zeros((n, m_item), dtype=np.float64)
    lo = np.asarray(query_times) - WINDOW_DAYS
    hi = np.asarray(query_times)
    query = np.concatenate([lo, hi])
    for it, times in item_times.items():
        if it >= m_item:
            continue
        idx = np.searchsorted(times, query)
        out[:, it] = idx[n:] - idx[:n]
    return out


def evaluate_split(name, split_dict, ds, model, model_name, item_times, static_pop, eta, dataset_name):
    users, items, times, hist_item_np, hist_time_np = C.build_eval_batch(ds, split_dict)
    util_scores = C.run_utility_scores(model, ds, users, hist_item_np, hist_time_np, model_name)

    results = {}
    r_util = C.rank_of_true_item(util_scores, items)
    results["A_utility_only"] = r_util

    lam_all, *_ = C.hawkes_lambda_at_times(model, ds, times)
    logprob_hawkes = np.log(lam_all + 1e-12) - np.log(lam_all.sum(-1, keepdims=True) + 1e-12)
    comb = C.mask_seen(eta * logprob_hawkes + (1 - eta) * util_scores, ds, users)
    results["B_hawkes"] = C.rank_of_true_item(comb, items)

    c3d = local_3d_counts_at_times(item_times, times, ds.m_item) + EPS
    logprob_3d = np.log(c3d) - np.log(c3d.sum(-1, keepdims=True))
    comb = C.mask_seen(eta * logprob_3d + (1 - eta) * util_scores, ds, users)
    results["C_sliding_window_3d"] = C.rank_of_true_item(comb, items)

    static_p = static_pop + EPS
    logprob_static = np.log(static_p) - np.log(static_p.sum())
    comb = C.mask_seen(eta * logprob_static[None, :] + (1 - eta) * util_scores, ds, users)
    results["D_static_training_freq"] = C.rank_of_true_item(comb, items)

    rows = []
    for source, r in results.items():
        rows.append(dict(dataset=dataset_name, split=name, popularity_source=source, eta=eta,
                          n=len(users), recall_at_10=float((r <= 10).mean()),
                          ndcg_at_10=float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0).mean())))
    return rows


def main():
    dataset_name = "kuairand"
    ds = C.load_dataset(dataset_name, rescale_item_time_array=True)
    model, eta = C.load_ours_ckpt(dataset_name, MODEL_NAME, SEED, ds)
    print(f"using eta (trained default) = {eta}, eps = {EPS}")

    item_times = build_item_time_dict(ds, dataset_name)
    item_times = {k: np.asarray(v, dtype=np.float64) for k, v in item_times.items()}

    train_dict = np.load(f"{C.REPO_ROOT}/data/{dataset_name}/training_dict.npy", allow_pickle=True).item()
    cnt = Counter()
    for u, items_ in train_dict.items():
        cnt.update(items_)
    static_pop = np.array([cnt.get(i, 0) for i in range(ds.m_item)], dtype=np.float64)

    splits = {
        "all": ds.test_dict,
        "head": ds.test_head_recent_3d_dict,
        "tail": ds.test_tail_recent_3d_dict,
    }

    all_rows = []
    for split_name, split_dict in splits.items():
        t0 = time.time()
        rows = evaluate_split(split_name, split_dict, ds, model, MODEL_NAME, item_times, static_pop, eta, dataset_name)
        all_rows.extend(rows)
        print(f"[{split_name}] done in {time.time()-t0:.1f}s")
        for r in rows:
            print(f"    {r['popularity_source']:25s} Recall@10={r['recall_at_10']:.4f} NDCG@10={r['ndcg_at_10']:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{C.RESULTS_DIR}/popularity_substitution.csv", index=False)
    print("\n=== Full table ===")
    print(df.pivot_table(index="popularity_source", columns="split", values="recall_at_10").to_string())
    print(f"\nsaved to {C.RESULTS_DIR}/popularity_substitution.csv")


if __name__ == "__main__":
    main()
