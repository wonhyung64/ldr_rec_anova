"""Analysis 4 (task spec Sec. 8): direct effect of the popularity component on Tail
positive-item rank.

For every Tail test event (u_i, v_i, t_i) -- using the repository's own, already-
validated Tail evaluation set ds.test_tail_recent_3d_dict, exactly as Table 1/4 use it,
with the same full-ranking protocol and seen-item masking (common.mask_seen /
common._allPos) -- we compute:

    r_util(i)  = rank of v_i under utility-only score f_Psi(x_u(t), v)
    r_comb(i)  = rank of v_i under the trained-default-eta combined score
                 eta*log(pi_Phi) + (1-eta)*f_Psi
    delta_rank(i) = r_comb(i) - r_util(i)

Rank 1 = best. delta_rank > 0 means adding the popularity term makes the positive Tail
item's rank *worse*.

Also reports the Recall@10-crossing probability
    P(r_util <= 10 and r_comb > 10)
which directly quantifies how much of the observed Tail Recall@10 degradation is
attributable to items that *were* being ranked correctly by utility alone but get pushed
out once popularity is reintroduced.

Primary target: KuaiRand, all six backbones, seed=1 (matches the task spec's explicit
ask for backbone-level results on KuaiRand). Micro-Video and MovieLens are additionally
computed for SASRec only, as supplementary cross-dataset context.

Outputs:
    results/appendix6/tail_rank_shift.csv           (per dataset x backbone x test event)
    results/appendix6/tail_rank_shift_summary.csv    (per dataset x backbone, aggregate stats)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEED = 1
KUAIRAND_BACKBONES = C.BACKBONES  # all six, per spec
SUPPLEMENTARY = {"micro_video": ["sasrec"], "ml-1m": ["sasrec"]}


def compute_for(dataset_name, model_name, ds, users, items, times, hist_item_np, hist_time_np):
    model, eta = C.load_ours_ckpt(dataset_name, model_name, SEED, ds)

    # utility-only score f_Psi (eta=0 component)
    util_scores = C.run_utility_scores(model, ds, users, hist_item_np, hist_time_np, model_name)
    r_util = C.rank_of_true_item(util_scores, items)

    # combined score at the trained default eta
    lam_all, mu, alpha, beta = C.hawkes_lambda_at_times(model, ds, times)
    logprob = np.log(lam_all + 1e-12) - np.log(lam_all.sum(-1, keepdims=True) + 1e-12)
    combined = eta * logprob + (1 - eta) * util_scores  # util_scores already seen-masked (-9999)
    # re-apply mask defensively (logprob term is finite everywhere, util_scores already -9999 at seen items)
    combined = C.mask_seen(combined, ds, users)
    r_comb = C.rank_of_true_item(combined, items)

    return r_util, r_comb, eta


def main():
    rows = []
    summary_rows = []

    for dataset_name in C.DATASETS:
        backbones = KUAIRAND_BACKBONES if dataset_name == "kuairand" else SUPPLEMENTARY[dataset_name]
        ds = C.load_dataset(dataset_name, rescale_item_time_array=True)
        users, items, times, hist_item_np, hist_time_np = C.build_eval_batch(ds, ds.test_tail_recent_3d_dict)
        print(f"[{dataset_name}] tail_recent_3d N={len(users)}")

        for model_name in backbones:
            t0 = time.time()
            r_util, r_comb, eta = compute_for(dataset_name, model_name, ds, users, items, times,
                                               hist_item_np, hist_time_np)
            delta = r_comb.astype(np.int64) - r_util.astype(np.int64)
            delta_log = np.log1p(r_comb) - np.log1p(r_util)

            crossing = ((r_util <= 10) & (r_comb > 10))
            improved_into_10 = ((r_util > 10) & (r_comb <= 10))

            for i in range(len(users)):
                rows.append(dict(dataset=dataset_name, backbone=model_name, seed=SEED,
                                  user=int(users[i]), item=int(items[i]), eta=eta,
                                  r_util=int(r_util[i]), r_comb=int(r_comb[i]),
                                  delta_rank=int(delta[i]), delta_log_rank=float(delta_log[i])))

            summary_rows.append(dict(
                dataset=dataset_name, backbone=model_name, eta=eta, n=len(users),
                mean_delta_rank=float(delta.mean()), median_delta_rank=float(np.median(delta)),
                mean_delta_log_rank=float(delta_log.mean()), median_delta_log_rank=float(np.median(delta_log)),
                frac_delta_positive=float((delta > 0).mean()), frac_delta_negative=float((delta < 0).mean()),
                frac_delta_zero=float((delta == 0).mean()),
                q10_delta=float(np.percentile(delta, 10)), q90_delta=float(np.percentile(delta, 90)),
                p_crossing_out_of_top10=float(crossing.mean()),
                p_crossing_into_top10=float(improved_into_10.mean()),
                recall10_util=float((r_util <= 10).mean()), recall10_comb=float((r_comb <= 10).mean()),
            ))
            print(f"  {model_name:10s} eta={eta} mean_delta_rank={delta.mean():.1f} "
                  f"P(cross out of top10)={crossing.mean():.4f} "
                  f"R@10 util={( r_util<=10).mean():.4f} -> comb={(r_comb<=10).mean():.4f} "
                  f"({time.time()-t0:.1f}s)")

    pd.DataFrame(rows).to_csv(f"{C.RESULTS_DIR}/tail_rank_shift.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{C.RESULTS_DIR}/tail_rank_shift_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nsaved to {C.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
