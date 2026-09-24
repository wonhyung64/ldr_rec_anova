"""4-seed mean +/- std version of the eta=0-vs-backbone verification, formatted like
the paper's Table 1 (mean +/- std over seeds 1-4), for all six KuaiRand backbones.

Companion to analysis_kuairand_verify_eta0_all_backbones.py (which only used seed=1).
See analysis_representation/kuairand_tail_diagnosis_summary.md Sec. 2b.

Usage: python analysis_kuairand_eta0_4seed_table.py
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_kuairand_tail_diagnosis import (
    load_dataset, load_backbone_ckpt, load_ours_ckpt, build_eval_batch,
    run_model_scores, run_ours_scores, rank_of_true_item, mask_pos, OUT_DIR,
)

MODELS = ["mf", "grurec", "sasrec", "fearec", "bsarec", "tisasrec"]
SEEDS = [1, 2, 3, 4]


def main():
    ds = load_dataset()
    users, items, times, hist_item_np, hist_time_np = build_eval_batch(ds, ds.test_tail_recent_3d_dict)
    print(f"tail_recent_3d: N={len(users)}\n")

    per_seed = {m: {"backbone": [], "default": [], "eta0": []} for m in MODELS}
    eta_default_by_model = {}

    for model_name in MODELS:
        for seed in SEEDS:
            t0 = time.time()
            backbone = load_backbone_ckpt(model_name, seed, ds)
            b_scores = run_model_scores(backbone, ds, users, hist_item_np, hist_time_np, model_name=model_name)
            b_r10 = float((rank_of_true_item(b_scores, items) <= 10).mean())

            model, eta_default = load_ours_ckpt(model_name, seed, ds)
            eta_default_by_model[model_name] = eta_default
            resid_all, logprob_all, *_ = run_ours_scores(
                model, ds, users, times, hist_item_np, hist_time_np, model_name=model_name)

            combined_default = mask_pos(eta_default * logprob_all + (1 - eta_default) * resid_all, ds, users)
            r10_default = float((rank_of_true_item(combined_default, items) <= 10).mean())

            combined_eta0 = mask_pos(0.0 * logprob_all + 1.0 * resid_all, ds, users)
            r10_eta0 = float((rank_of_true_item(combined_eta0, items) <= 10).mean())

            per_seed[model_name]["backbone"].append(b_r10)
            per_seed[model_name]["default"].append(r10_default)
            per_seed[model_name]["eta0"].append(r10_eta0)

            print(f"{model_name:10s} seed={seed} backbone={b_r10:.4f} "
                  f"ours@default(eta={eta_default})={r10_default:.4f} ours@eta0={r10_eta0:.4f} "
                  f"({time.time()-t0:.1f}s)")

            # save incrementally after every seed so partial progress is never lost
            summary = {}
            for m in MODELS:
                if len(per_seed[m]["backbone"]) == 0:
                    continue
                summary[m] = {
                    "eta_default": eta_default_by_model.get(m),
                    "n_seeds": len(per_seed[m]["backbone"]),
                    "backbone_mean": float(np.mean(per_seed[m]["backbone"])),
                    "backbone_std": float(np.std(per_seed[m]["backbone"])),
                    "default_mean": float(np.mean(per_seed[m]["default"])),
                    "default_std": float(np.std(per_seed[m]["default"])),
                    "eta0_mean": float(np.mean(per_seed[m]["eta0"])),
                    "eta0_std": float(np.std(per_seed[m]["eta0"])),
                    "per_seed_backbone": per_seed[m]["backbone"],
                    "per_seed_default": per_seed[m]["default"],
                    "per_seed_eta0": per_seed[m]["eta0"],
                }
            json.dump(summary, open(f"{OUT_DIR}/eta0_4seed_table.json", "w"), indent=2)

    print("\n=== Final table (mean +/- std over seeds 1-4) ===")
    print(f"{'Backbone':10s} {'Backbone R@10':>16s} {'Ours@default':>16s} {'Ours@eta0':>16s}")
    for m in MODELS:
        b = per_seed[m]["backbone"]; d = per_seed[m]["default"]; e = per_seed[m]["eta0"]
        print(f"{m:10s} {np.mean(b):.4f}+/-{np.std(b):.4f}   "
              f"{np.mean(d):.4f}+/-{np.std(d):.4f}   {np.mean(e):.4f}+/-{np.std(e):.4f}")

    print("\nsaved", f"{OUT_DIR}/eta0_4seed_table.json")


if __name__ == "__main__":
    main()
