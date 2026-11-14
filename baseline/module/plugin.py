"""Helpers for the "separately estimated popularity + plug-in utility learning" experiment
(ICLR-27 rebuttal, W5: unregularized joint estimation of the Hawkes intensity).

Protocol
    1. Phi_hat (Hawkes parameters: log_beta, p_item_embedding, base_net, excitation_net) is taken
       from a checkpoint trained by the *popularity-aware sampling* paper (ldr_rec; Phi fitted with
       the sampled conditional mark likelihood l_pop) and then FROZEN.
    2. Only the utility parameters Psi (backbone) are trained, with the original objective
       l_choice(Phi_hat, Psi) + gamma * l_cen(Psi).
    3. Inference is the original eq. (3); additionally the whole eta grid is evaluated so that the
       "utility-only" (eta = 0) row can be read from the very same run.

Time-unit caveat (important)
    The popularity paper's code base (ldr_rec) does not rescale KuaiRand's millisecond timestamps,
    so its Hawkes decay rate beta for KuaiRand lives in 1/(ms / 86400) = 1/(86.4 s).  This code base
    (module/dataset.py) converts KuaiRand to seconds first, so its "day" is a real day (1000x
    longer).  beta therefore has to be multiplied by 1000 to describe the same intensity
    (beta*dt is invariant under this rescaling).  Micro-Video / MovieLens are unaffected.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .procedure import computeTopNAccuracy

HAWKES_PREFIXES = ("log_beta", "p_item_embedding.", "base_net.", "excitation_net.")

# KuaiRand: ldr_rec stored raw ms timestamps, this repo stores seconds -> 1000x.
DEFAULT_BETA_SCALE = {"kuairand": 1000.0}


def resolve_beta_scale(dataset_name, user_value):
    """user_value < 0 means "auto"."""
    if user_value is not None and user_value >= 0:
        return float(user_value)
    return DEFAULT_BETA_SCALE.get(dataset_name, 1.0)


def plugin_ckpt_path(plugin_root, dataset_name, plugin_model, seed):
    return os.path.join(plugin_root, dataset_name, f"{plugin_model}_seed{seed}.pt")


def _inverse_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32)
    return y + torch.log1p(-torch.exp(-y))


def load_frozen_hawkes(model, ckpt_path, beta_scale=1.0, device="cpu"):
    """Copy the Hawkes intensity parameters from `ckpt_path` into `model` and freeze them.

    Only keys in HAWKES_PREFIXES are read; the backbone weights stored in the checkpoint are ignored,
    so the checkpoint's backbone type is irrelevant.  Returns a dict of what was loaded.
    """
    if not hasattr(model, "p_item_embedding"):
        raise ValueError(
            "plug-in Hawkes parameters require the unshared Hawkes prior (p_item_embedding); "
            "run with --ablation=shared."
        )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"plug-in checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    hawkes_state = {k: v for k, v in state.items() if k.startswith(HAWKES_PREFIXES)}

    own_state = model.state_dict()
    expected = {k for k in own_state if k.startswith(HAWKES_PREFIXES)}
    missing = expected - set(hawkes_state)
    unexpected = set(hawkes_state) - expected
    if missing or unexpected:
        raise RuntimeError(f"Hawkes parameter mismatch. missing={sorted(missing)} unexpected={sorted(unexpected)}")
    for k, v in hawkes_state.items():
        if tuple(own_state[k].shape) != tuple(v.shape):
            raise RuntimeError(
                f"shape mismatch for {k}: model {tuple(own_state[k].shape)} vs checkpoint {tuple(v.shape)} "
                f"(different item catalog / data split?)"
            )

    beta_ckpt = float(torch.nn.functional.softplus(hawkes_state["log_beta"]) + 1e-6)
    if beta_scale != 1.0:
        # model.current_beta() = softplus(log_beta) + 1e-6  ->  pick log_beta so that it equals scale * beta_ckpt
        target = beta_scale * beta_ckpt - 1e-6
        hawkes_state["log_beta"] = _inverse_softplus(target).reshape(hawkes_state["log_beta"].shape)

    model.load_state_dict(hawkes_state, strict=False)
    model.to(device)

    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith(HAWKES_PREFIXES):
            p.requires_grad_(False)
            n_frozen += p.numel()

    return {
        "ckpt_path": ckpt_path,
        "ckpt_epoch": int(ckpt["epoch"]) if "epoch" in ckpt else None,
        "beta_ckpt": beta_ckpt,
        "beta_scale": float(beta_scale),
        "beta_used": float(model.current_beta()),
        "n_frozen_params": n_frozen,
    }


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def hawkes_state_fingerprint(model):
    """Cheap checksum to prove Phi did not move during training."""
    s = 0.0
    for name, p in model.named_parameters():
        if name.startswith(HAWKES_PREFIXES):
            s += float(p.detach().cpu().double().sum())
    return s


# --------------------------------------------------------------------------------------
# Popularity concentration (same statistics as analysis/appendix6: Table 5 of the paper)
# --------------------------------------------------------------------------------------
def _normalized_entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    nz = p > eps
    h = -(p[nz] * np.log(p[nz])).sum()
    return float(h / np.log(len(p))) if len(p) > 1 else float("nan")


def _top_q_mass(p, q=0.05):
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p_sorted = np.sort(p)[::-1] / total
    k = max(1, int(round(q * len(p_sorted))))
    return float(np.cumsum(p_sorted)[k - 1])


@torch.no_grad()
def hawkes_intensity_all_items(model, item_time_array_t, t_days):
    """lambda_v(t; Phi) for every item at one query time (days). item_time_array_t is (m_item, L) in days."""
    mu, alpha, beta = model.prior_parameters_from_embeddings()
    t = torch.tensor(float(t_days), dtype=torch.float32, device=item_time_array_t.device)
    mask = item_time_array_t < t
    delta = (t - item_time_array_t).clamp(min=0.0)
    h = (torch.exp(-beta * delta) * mask).sum(-1)
    return mu + alpha * h


@torch.no_grad()
def hawkes_concentration(model, dataset, device, window_days=3.0, step_days=1.0, top_q=0.05):
    """Normalized entropy and top-q mass of pi_Phi(v|t) = lambda_v(t)/sum_k lambda_k(t), averaged over
    daily snapshots (grid identical to analysis/appendix6/analyze_empirical_popularity.snapshot_grid)."""
    model.eval()
    all_times = np.array([t for items in dataset.time_dict.values() for t in items.values()], dtype=np.float64) / 86400.0
    grid = np.arange(all_times.min() + window_days, all_times.max(), step_days)
    item_time_t = torch.tensor(dataset.item_time_array, dtype=torch.float32, device=device)

    h_list, top_list = [], []
    for t in grid:
        lam = hawkes_intensity_all_items(model, item_time_t, t).cpu().double().numpy()
        h_list.append(_normalized_entropy(lam))
        top_list.append(_top_q_mass(lam, top_q))
    h_arr, top_arr = np.array(h_list), np.array(top_list)

    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()
    mu_np = mu.cpu().double().numpy()
    return {
        "n_snapshots": int(len(grid)),
        "H_norm_mean": float(np.nanmean(h_arr)), "H_norm_std": float(np.nanstd(h_arr)),
        f"top{int(round(top_q * 100))}pct_mass_mean": float(np.nanmean(top_arr)),
        f"top{int(round(top_q * 100))}pct_mass_std": float(np.nanstd(top_arr)),
        "mu_top10_items_share": float(np.sort(mu_np)[::-1][:10].sum() / mu_np.sum()),
        "mu_top5pct_share": _top_q_mass(mu_np, top_q),
        "beta": float(beta),
    }


# --------------------------------------------------------------------------------------
# Multi-eta evaluation (one pass over users, all etas at once)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_eta_grid(model, dataset, args, split_pairs, resid_fn, etas):
    """split_pairs: list of (split_name, {(user, item): time_in_days}).
    resid_fn(user, hist_item_np, hist_time_np) -> 1-D utility score tensor over all items (on args.device).

    Ranking is exactly the original eq.(3) implementation:  eta * log pi_Phi(v|t) + (1-eta) * f_Psi,
    previously-consumed items masked, top-k taken.  Returns {split: {str(eta): {metric@k: value}}}.
    """
    model.eval()
    mu, alpha, beta = model.prior_parameters_from_embeddings()
    item_time_t = torch.tensor(dataset.item_time_array, dtype=torch.float32, device=args.device)
    max_k = max(args.topks)
    results = {}

    for split_name, pair_dict in split_pairs:
        preds = {eta: [] for eta in etas}
        gt_list = []
        for (user, item), pos_time_val in pair_dict.items():
            hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
            resid = resid_fn(user, hist_item_np, hist_time_np)

            t = torch.tensor(float(pos_time_val), dtype=torch.float32, device=args.device)
            mask = item_time_t < t
            delta = (t - item_time_t).clamp(min=0.0)
            h = (torch.exp(-beta * delta) * mask).sum(-1)
            logits = mu + alpha * h
            item_log_prob = torch.log(logits + 1e-12) - torch.log(logits.sum() + 1e-12)

            exclude_items = list(dataset._allPos[user])
            for eta in etas:
                pred = (item_log_prob * eta + resid * (1 - eta)).cpu()
                pred[exclude_items] = -9999
                _, pred_k = torch.topk(pred, k=max_k)
                preds[eta].append(pred_k.cpu())
            gt_list.append([item])

        results[split_name] = {}
        for eta in etas:
            precision, recall, ndcg, mrr = computeTopNAccuracy(gt_list, preds[eta], args.topks)
            entry = {"n_events": len(gt_list)}
            for name, vals in (("precision", precision), ("recall", recall), ("ndcg", ndcg), ("mrr", mrr)):
                for k, v in zip(args.topks, vals):
                    entry[f"{name}@{k}"] = float(v)
            results[split_name][str(eta)] = entry
    return results


def parse_eta_grid(text):
    return [float(x) for x in str(text).split(",") if x.strip() != ""]


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[plugin] wrote {path}")
