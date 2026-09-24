"""Analysis 5 (task spec Sec. 10): Tail Recall@10 as a function of eta, full grid, on the
repository's own Tail evaluation set (ds.test_tail_recent_3d_dict), same full-ranking
protocol and seen-item masking as Table 1/4.

Primary: KuaiRand, all six backbones, seed=1 (full eta grid). Supplementary: Micro-Video
and MovieLens, SASRec only (same grid), for cross-dataset context.

No retraining: reuses the same trained Ours checkpoints as Table 1/4; only the inference-
time eta is varied.

Outputs:
    results/appendix6/eta_tail_recall.csv   (dataset x backbone x eta x Recall@10)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEED = 1
ETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUPPLEMENTARY = {"micro_video": ["sasrec"], "ml-1m": ["sasrec"]}


def main():
    rows = []
    for dataset_name in C.DATASETS:
        backbones = C.BACKBONES if dataset_name == "kuairand" else SUPPLEMENTARY[dataset_name]
        ds = C.load_dataset(dataset_name, rescale_item_time_array=True)
        users, items, times, hist_item_np, hist_time_np = C.build_eval_batch(ds, ds.test_tail_recent_3d_dict)
        print(f"[{dataset_name}] tail_recent_3d N={len(users)}")

        for model_name in backbones:
            t0 = time.time()
            model, eta_default = C.load_ours_ckpt(dataset_name, model_name, SEED, ds)
            util_scores = C.run_utility_scores(model, ds, users, hist_item_np, hist_time_np, model_name)
            lam_all, mu, alpha, beta = C.hawkes_lambda_at_times(model, ds, times)
            logprob = np.log(lam_all + 1e-12) - np.log(lam_all.sum(-1, keepdims=True) + 1e-12)

            for eta in ETA_GRID:
                combined = eta * logprob + (1 - eta) * util_scores
                combined = C.mask_seen(combined, ds, users)
                r = C.rank_of_true_item(combined, items)
                recall10 = float((r <= 10).mean())
                ndcg10 = float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0).mean())
                rows.append(dict(dataset=dataset_name, backbone=model_name, eta=eta,
                                  eta_is_default=bool(np.isclose(eta, eta_default)),
                                  recall_at_10=recall10, ndcg_at_10=ndcg10, n=len(users)))
            print(f"  {model_name:10s} eta_default={eta_default} done in {time.time()-t0:.1f}s")

            pd.DataFrame(rows).to_csv(f"{C.RESULTS_DIR}/eta_tail_recall.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== KuaiRand: Tail Recall@10 vs eta, all backbones ===")
    piv = df[df.dataset == "kuairand"].pivot(index="eta", columns="backbone", values="recall_at_10")
    print(piv.to_string())

    # explicitly check monotonicity per the task spec's caution ("monotonic하지 않으면 monotonic하다고 쓰지 마라")
    print("\n=== Monotonicity check (Tail Recall@10 strictly non-increasing in eta?) ===")
    for (dataset_name, model_name), sub in df.groupby(["dataset", "backbone"]):
        sub = sub.sort_values("eta")
        vals = sub["recall_at_10"].values
        is_monotonic_nonincreasing = np.all(np.diff(vals) <= 1e-9)
        onset_idx = np.argmax(np.diff(vals) < -1e-9) if (np.diff(vals) < -1e-9).any() else None
        onset_eta = sub["eta"].values[onset_idx + 1] if onset_idx is not None else None
        print(f"{dataset_name:12s} {model_name:10s} monotonic_nonincreasing={is_monotonic_nonincreasing} "
              f"first_decrease_at_eta={onset_eta}")

    print(f"\nsaved to {C.RESULTS_DIR}/eta_tail_recall.csv")


if __name__ == "__main__":
    main()
