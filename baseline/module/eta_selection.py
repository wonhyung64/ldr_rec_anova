"""Inference-time eta selection on a validation tail slice (ICLR-27 rebuttal, W5 follow-up).

The review points out that the validation-selected eta (chosen on *overall* Recall@10, which favours Head items)
hurts Tail Recall@10.  This module provides the pieces to select eta on the validation analogue of the test
"Tail (recent 3 days)" slice instead:

  * recent_head_tail_items(): the head/tail item split used by split_test_head_tail.py, generalised so that it can
    be built for the validation period with the same rule and without peeking at the period itself
        - popularity = item frequency among interactions in the `window_days` days *before the target period starts*
          (test: train+valid interactions;  valid: train interactions only)
        - items ranked by (-count, item id); top `head_ratio` of the item universe = head, the rest = tail
  * evaluate_eta_grid_batched(): same ranking as eq.(3) / plugin.evaluate_eta_grid, but scores a batch of events at once.
"""
from collections import Counter

import numpy as np
import torch

from .procedure import computeTopNAccuracy

SECONDS_PER_DAY = 60 * 60 * 24


def _interactions(set_dict, time_dict):
    return [(u, i, time_dict[u][i]) for u, items in set_dict.items() for i in items]


def recent_head_tail_items(dataset, target="valid", window_days=3, head_ratio=0.2):
    """Returns (head_items, tail_items, info) for the recent-`window_days` popularity before `target` starts.

    target="test"  reproduces data/{dataset}/testing_dict_{head,tail}_recent_{N}d.npy (train+valid history window).
    target="valid" is its validation analogue (train-only history window).
    dataset.time_dict must be in seconds (module/dataset.py already converts KuaiRand ms -> s).
    """
    td = dataset.time_dict
    train = _interactions(dataset.train_dict, td)
    valid = _interactions(dataset.valid_dict, td)
    test = _interactions(dataset.test_dict, td)

    universe = {i for _, i, _ in train + valid + test}
    if target == "valid":
        pre, target_inter = train, valid
    elif target == "test":
        pre, target_inter = train + valid, test
    else:
        raise ValueError(target)

    start = min(t for _, _, t in target_inter)
    cutoff = start - window_days * SECONDS_PER_DAY
    counts = Counter(i for _, i, t in pre if t >= cutoff)
    popularity = {i: counts.get(i, 0) for i in universe}

    ranked = sorted(popularity.items(), key=lambda kv: (-kv[1], kv[0]))
    n_head = max(1, round(len(ranked) * head_ratio))
    head = {i for i, _ in ranked[:n_head]}
    tail = {i for i, _ in ranked[n_head:]}
    info = {
        "target": target, "window_days": window_days, "head_ratio": head_ratio,
        "n_items": len(universe), "n_head_items": len(head), "n_tail_items": len(tail),
        "n_window_interactions": int(sum(counts.values())),
    }
    return head, tail, info


def filter_dict_by_items(set_dict, item_set):
    return {u: [i for i in items if i in item_set] for u, items in set_dict.items()}


@torch.no_grad()
def evaluate_eta_grid_batched(model, dataset, args, split_pairs, resid_fn, etas, batch_size=128):
    """split_pairs: list of (split_name, {(user, item): time_in_days}).
    resid_fn(users, hist_item_np, hist_time_np) -> (B, m_item) utility scores on args.device.
    Returns {split: {str(eta): {"n_events", "recall@k", "ndcg@k", ...}}}.
    """
    model.eval()
    device = args.device
    mu, alpha, beta = model.prior_parameters_from_embeddings()
    item_time_t = torch.tensor(dataset.item_time_array, dtype=torch.float32, device=device)  # (m_item, L) days
    m_item, L = item_time_t.shape
    hawkes_chunk = max(1, int(2e7 // (m_item * L)))
    max_k = max(args.topks)
    results = {}

    for split_name, pair_dict in split_pairs:
        keys = list(pair_dict.keys())
        times = [pair_dict[k] for k in keys]
        preds = {eta: [] for eta in etas}
        gt_list = []

        for s in range(0, len(keys), batch_size):
            ks, ts = keys[s:s + batch_size], times[s:s + batch_size]
            users = [k[0] for k in ks]
            B = len(ks)
            hist_item_np, hist_time_np = dataset.build_histories(zip(users, [0] * B, ts), args.max_seq_len)
            resid = resid_fn(users, hist_item_np, hist_time_np)                     # (B, m_item)

            t = torch.tensor(ts, dtype=torch.float32, device=device)
            lam = torch.empty(B, m_item, dtype=torch.float32, device=device)
            for hs in range(0, B, hawkes_chunk):
                tq = t[hs:hs + hawkes_chunk].view(-1, 1, 1)
                mask = item_time_t[None] < tq
                delta = (tq - item_time_t[None]).clamp(min=0.0)
                h = (torch.exp(-beta * delta) * mask).sum(-1)
                lam[hs:hs + hawkes_chunk] = mu[None] + alpha[None] * h
            item_log_prob = torch.log(lam + 1e-12) - torch.log(lam.sum(-1, keepdim=True) + 1e-12)

            # previously consumed items are masked out
            excl = [np.asarray(dataset._allPos[u], dtype=np.int64) for u in users]
            rows = torch.from_numpy(np.repeat(np.arange(B), [len(e) for e in excl])).to(device)
            cols = torch.from_numpy(np.concatenate(excl)).to(device)
            seen = torch.zeros(B, m_item, dtype=torch.bool, device=device)
            seen[rows, cols] = True

            for eta in etas:
                pred = (item_log_prob * eta + resid * (1 - eta)).masked_fill(seen, -9999)
                _, pred_k = torch.topk(pred, k=max_k, dim=-1)
                preds[eta].extend(list(pred_k.cpu()))
            gt_list.extend([[k[1]] for k in ks])

        results[split_name] = {}
        for eta in etas:
            precision, recall, ndcg, mrr = computeTopNAccuracy(gt_list, preds[eta], args.topks)
            entry = {"n_events": len(gt_list)}
            for name, vals in (("precision", precision), ("recall", recall), ("ndcg", ndcg), ("mrr", mrr)):
                for k, v in zip(args.topks, vals):
                    entry[f"{name}@{k}"] = float(v)
            results[split_name][str(eta)] = entry
    return results


def select_eta(results, split_name, metric="recall@10"):
    """argmax over eta of `metric` on `split_name`; ties -> smallest eta (the most conservative popularity weight)."""
    per_eta = results[split_name]
    best_eta, best_val = None, -1.0
    for eta_str in sorted(per_eta, key=float):
        v = per_eta[eta_str][metric]
        if v > best_val + 1e-12:
            best_eta, best_val = eta_str, v
    return best_eta, best_val
