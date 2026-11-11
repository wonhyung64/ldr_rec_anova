"""Shared pieces of the sampled-softmax loss-control baselines.

Reviewer request: train the *vanilla* backbones with the same multi-negative
sampled-softmax objective as "+Ours" (importance-corrected l_choice, M=15
uniform negatives) but WITHOUT the temporal-popularity prior (lambda_v == 1)
and WITHOUT the ANOVA-centering penalty (gamma == 0), so that the gain from
temporal popularity modelling can be separated from the gain of simply
switching the training loss.

Everything that differs from the original vanilla scripts (cf.py, seq_rec.py,
seq_rec_tisasrec.py) lives here:
  * sampled_softmax_loss  -- l_choice with log(lambda_v) = 0
  * evaluation            -- batched, scores through model.score_all_items
                             (so the cosine/tau scoring used in training is
                             also used at test time)
  * metric json dump      -- every number needed for Table 1 / Table 2 / Fig. 2
"""
import os
import re
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import score_pair, score_all
from .procedure import computeTopNAccuracy


SPLITS = [
    "head_overall", "head_recent_3d", "head_recent_7d",
    "tail_overall", "tail_recent_3d", "tail_recent_7d",
]
METRIC_NAMES = ["precision", "recall", "ndcg", "mrr"]


def get_epoch(path):
    match = re.search(r"_e(\d+)_", path.name)
    return int(match.group(1)) if match else -1


# --------------------------------------------------------------------------- #
# model / loss
# --------------------------------------------------------------------------- #
def init_item_embedding(model, std):
    """Same N(0, 0.02) item-embedding init that the Hawkes/ANOVA models apply in
    HawkesDebias.reset_parameters(), so that the only difference to +Ours is
    the missing prior / centering term."""
    if std is not None and std > 0:
        nn.init.normal_(model.item_embedding.weight, std=std)


def sample_uniform_negatives(pos_item, num_items, k):
    """k negatives per positive, uniform over V \\ {pos} (rows may repeat)."""
    neg = np.random.randint(0, num_items - 1, size=(len(pos_item), k), dtype=np.int64)
    neg += (neg >= pos_item[:, None])
    return neg


def sampled_softmax_loss(model, candidate_items, hist_items, additional_feat,
                         num_items, num_negatives, correction=True):
    """l_choice with a uniform popularity prior (lambda_v == 1).

    candidate_items[:, 0] is the observed item, [:, 1:] are M uniform negatives.
    This is hawkes_choice.importance_corrected_choice_loss with log(lambda)=0:
    the uniform proposal collapses the importance weight to the constant
    log(|V|-1) - log(M) added to every negative logit.
    """
    util = score_pair(model, candidate_items, hist_items, additional_feat)
    if correction:
        log_corr = math.log(max(num_items - 1, 1)) - math.log(num_negatives)
        logits = torch.cat([util[:, :1], util[:, 1:] + log_corr], dim=1)
    else:
        logits = util
    return -F.log_softmax(logits, dim=-1)[:, 0].mean()


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def ckpt_pattern(args, epoch="???"):
    return (f"_{args.model_name}_ss_tau{args.tau}_scorenorm{args.score_norm}"
            f"_corr{args.ss_correction}_e{epoch}_seed{args.seed}.pt")


def resume_if_possible(model, optimizer, args):
    matched = sorted(Path(args.save_path).glob(ckpt_pattern(args)))
    if len(matched) == 0:
        return 0
    recent = max(matched, key=get_epoch)
    ckpt = torch.load(recent, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"MODEL LOADED! ({recent.name})")
    return ckpt["epoch"]


def save_checkpoint(model, optimizer, args, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, f"{args.save_path}/{ckpt_pattern(args, epoch)}")


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_pairs(model, dataset, pair_dict, args, mode, batch_size=512):
    """Full-ranking evaluation of {(user, item): time} pairs.

    mode:
      "user"      -- additional_feat = user id (GRU/SASRec/FEARec/BSARec, MF)
      "user_nohist" -- same, but skip building histories (MF ignores them)
      "time"      -- additional_feat = history timestamps in seconds (TiSASRec)
    Items the user interacted with in the training set are excluded, exactly as
    in the original vanilla scripts.
    """
    model.eval()
    pairs = list(pair_dict.items())
    topk = max(args.topks)
    pred_list, gt_list = [], []

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        users = [u for (u, _), _ in chunk]
        gts = [i for (_, i), _ in chunk]
        times = [t for _, t in chunk]
        user_t = torch.tensor(users, dtype=torch.long, device=args.device)

        if mode == "user_nohist":
            hist_t = torch.full((len(users), args.max_seq_len), dataset.m_item,
                                dtype=torch.long, device=args.device)
            feat = user_t
        else:
            hist_np, hist_time_np = dataset.build_histories(
                [(u, 0, t) for u, t in zip(users, times)], args.max_seq_len)
            hist_t = torch.tensor(hist_np, dtype=torch.long, device=args.device)
            if mode == "time":
                feat = torch.tensor(hist_time_np, dtype=torch.long, device=args.device) * 24 * 60 * 60
            else:
                feat = user_t

        scores = score_all(model, hist_t, feat)
        for row, u in enumerate(users):
            scores[row, list(dataset._allPos[u])] = float("-inf")
        top = torch.topk(scores, k=topk, dim=1).indices.cpu().tolist()

        pred_list.extend(top)
        gt_list.extend([[i] for i in gts])

    return computeTopNAccuracy(gt_list, pred_list, args.topks)


def flatten_metrics(prefix, results, topks):
    out = {}
    for name, per_k in zip(METRIC_NAMES, results):
        for k, v in zip(topks, per_k):
            out[f"{prefix}_{name}_{k}"] = float(v)
    return out


def evaluate_valid(model, dataset, args, mode):
    res = evaluate_pairs(model, dataset, dataset.valid_user_item_time, args, mode)
    return flatten_metrics("valid", res, args.topks)


def evaluate_test_all(model, dataset, args, mode):
    """Overall test metrics (Table 1) + head/tail x {overall, recent_3d, recent_7d}
    test metrics (Table 2 / Fig. 2)."""
    metrics = {}
    res = evaluate_pairs(model, dataset, dataset.test_user_item_time, args, mode)
    metrics.update(flatten_metrics("test", res, args.topks))

    split_dicts = {
        "head_overall": dataset.test_head_overall_dict,
        "head_recent_3d": dataset.test_head_recent_3d_dict,
        "head_recent_7d": dataset.test_head_recent_7d_dict,
        "tail_overall": dataset.test_tail_overall_dict,
        "tail_recent_3d": dataset.test_tail_recent_3d_dict,
        "tail_recent_7d": dataset.test_tail_recent_7d_dict,
    }
    for split_name in SPLITS:
        pairs = dataset.set_to_pair(split_dicts[split_name], dataset.time_dict, dataset.time_unit)
        res = evaluate_pairs(model, dataset, pairs, args, mode)
        metrics.update(flatten_metrics(f"test_{split_name}", res, args.topks))
    return metrics


def with_epoch_suffix(metrics, epoch):
    """wandb key convention of the existing runs: both plain and *_{epoch}."""
    out = dict(metrics)
    out.update({f"{k}_{epoch}": v for k, v in metrics.items()})
    return out


def dump_result_json(args, epoch, metrics, valid_metrics=None):
    os.makedirs(args.result_dir, exist_ok=True)
    path = os.path.join(args.result_dir, f"{args.dataset}_{args.model_name}_seed{args.seed}.json")
    payload = {
        "dataset": args.dataset,
        "model": args.model_name,
        "seed": args.seed,
        "epoch": epoch,
        "loss": "sampled_softmax",
        "num_negatives": args.contrast_size - 1,
        "ss_correction": args.ss_correction,
        "tau": args.tau,
        "score_norm": args.score_norm,
        "topks": list(args.topks),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list))},
        "metrics": metrics,
    }
    if valid_metrics:
        payload["valid_metrics"] = valid_metrics
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {path}")
    return path
