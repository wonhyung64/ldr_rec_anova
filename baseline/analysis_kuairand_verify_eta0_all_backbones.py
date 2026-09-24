"""Directly verify, for every KuaiRand backbone, whether the utility-only ranking
(eta=0, i.e. the popularity term pi_Phi turned off entirely) beats the independently
trained backbone on the tail_recent_3d split -- and whether the trained default eta does.

Companion to analysis_kuairand_tail_diagnosis.py; see
analysis_representation/kuairand_tail_diagnosis_summary.md Sec. 2b for the result and
discussion (5 of 6 backbones confirm eta=0 beats backbone; TiSASRec is a narrow
exception).

Usage: python analysis_kuairand_verify_eta0_all_backbones.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_kuairand_tail_diagnosis import (
    load_dataset, load_backbone_ckpt, load_ours_ckpt, build_eval_batch,
    run_model_scores, run_ours_scores, rank_of_true_item, mask_pos, OUT_DIR,
)

if __name__ == "__main__":
    ds = load_dataset()
    users, items, times, hist_item_np, hist_time_np = build_eval_batch(ds, ds.test_tail_recent_3d_dict)
    print(f"tail_recent_3d: N={len(users)}\n")

    results = {}
    for model_name in ["grurec", "fearec", "bsarec", "mf", "tisasrec", "sasrec"]:
        t0 = time.time()
        backbone = load_backbone_ckpt(model_name, 1, ds)
        b_scores = run_model_scores(backbone, ds, users, hist_item_np, hist_time_np, model_name=model_name)
        b_r10 = float((rank_of_true_item(b_scores, items) <= 10).mean())

        model, eta_default = load_ours_ckpt(model_name, 1, ds)
        resid_all, logprob_all, *_ = run_ours_scores(
            model, ds, users, times, hist_item_np, hist_time_np, model_name=model_name)

        combined_default = mask_pos(eta_default * logprob_all + (1 - eta_default) * resid_all, ds, users)
        r10_default = float((rank_of_true_item(combined_default, items) <= 10).mean())

        combined_eta0 = mask_pos(0.0 * logprob_all + 1.0 * resid_all, ds, users)
        r10_eta0 = float((rank_of_true_item(combined_eta0, items) <= 10).mean())

        beats_at_eta0 = r10_eta0 > b_r10
        beats_at_default = r10_default > b_r10
        results[model_name] = dict(backbone_r10=b_r10, eta_default=eta_default,
                                    r10_at_default_eta=r10_default, r10_at_eta0=r10_eta0,
                                    beats_backbone_at_eta0=beats_at_eta0,
                                    beats_backbone_at_default_eta=beats_at_default)
        print(f"{model_name:10s} backbone={b_r10:.4f}  ours@default(eta={eta_default})={r10_default:.4f} "
              f"[{'BEATS' if beats_at_default else 'loses to'} backbone]  "
              f"ours@eta=0={r10_eta0:.4f} [{'BEATS' if beats_at_eta0 else 'loses to'} backbone]  "
              f"({time.time()-t0:.1f}s)")

        json.dump(results, open(f"{OUT_DIR}/eta0_verification_all_backbones.json", "w"), indent=2)

    print("\nsaved", f"{OUT_DIR}/eta0_verification_all_backbones.json")
