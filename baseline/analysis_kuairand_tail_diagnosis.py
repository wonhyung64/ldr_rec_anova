"""KuaiRand Tail-item degradation diagnosis (Table 1 appendix material).

Diagnoses why several KuaiRand sequential backbones (SASRec, GRU, FEARec, BSARec,
TiSASRec) score *worse* on the Tail evaluation slice under the proposed framework than
under the backbone alone, in contrast to Micro-Video/MovieLens where Tail also improves.

Root cause (see analysis_representation/kuairand_tail_diagnosis_summary.md for the full
write-up): the unshared Hawkes prior's background rate mu_v = softplus(MLP(z_v)) is
unregularized in scale and, on KuaiRand specifically, collapses onto a tiny "celebrity"
clique of items (e.g. for SASRec, 10 of 7076 items hold 87.3% of all mu mass, vs 7.1% of
44503 items on Micro-Video with the same architecture). Because those items occupy the
top-10-by-popularity ranking at effectively every test time regardless of user, mixing in
the popularity term at the trained eta=alpha1 mechanically crowds "tail_recent_3d" items
(which are by construction *not* part of that clique) out of the top-10.

Usage:
    python analysis_kuairand_tail_diagnosis.py --model sasrec --seed 1 --mode sweep
    python analysis_kuairand_tail_diagnosis.py --model sasrec --seed 1 --mode mucap
    python analysis_kuairand_tail_diagnosis.py --model sasrec --seed 1 --mode concentration
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_BACKBONE = os.path.join(REPO_ROOT, "weights", "kuairand")
WEIGHTS_OURS = os.path.join(REPO_ROOT, "weights_hawkes_anova_generalize", "kuairand")
OUT_DIR = os.path.join(REPO_ROOT, "analysis_representation", "tail_diag")
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparameters actually used to produce Table 1 for KuaiRand (report_0831-4.sh).
GAMMA_FOR = dict(mf=0.1, grurec=10.0, sasrec=0.5, tisasrec=0.1, fearec=0.1, bsarec=0.1)
ALPHA1_FOR = dict(mf=0.3, grurec=0.5, sasrec=0.5, tisasrec=0.5, fearec=0.5, bsarec=0.5)
BSAREC_ALPHA = 0.9
ALPHA1_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_dataset():
    t0 = time.time()
    ds = UserItemTime(os.path.join(REPO_ROOT, "data"), "kuairand", "d", 50, 50)
    # Match debiased_seq_rec_hawkes_anova.py line 48: item_time_array is built from
    # self.time_dict (seconds), but pos_time_t used at inference (from set_to_pair) is
    # in DAYS. Without this rescale, the Hawkes mask (item_time_array < t) is ~always
    # False and the popularity term degenerates to the base rate mu only.
    ds.item_time_array = ds.item_time_array / 86400.0
    print(f"[load_dataset] built in {time.time()-t0:.1f}s (item_time_array rescaled to days)")
    return ds


def default_kwargs(model_name, ds, device, tau):
    kwargs = dict(
        num_users=ds.n_user, num_items=ds.m_item, embedding_k=128,
        device=device, tau=tau, depth=0, max_seq_len=50, n_heads=1, dropout=0.2,
    )
    if model_name == "bsarec":
        kwargs["c"] = 3
        kwargs["alpha"] = BSAREC_ALPHA
    if model_name == "tisasrec":
        # kuairand -> 512 (ml-1m uses 2048); see seq_rec_tisasrec.py / debiased_seq_rec_tisasrec_hawkes_anova.py
        kwargs["time_span"] = 512
    return kwargs


def additional_feat(model_name, users, hist_item_np, hist_time_np, device):
    """TiSASRec's encode_user takes history TIMESTAMPS (seconds) instead of a user id."""
    if model_name == "tisasrec":
        return torch.tensor(hist_time_np, dtype=torch.long, device=device) * 24 * 60 * 60
    return torch.tensor(users, dtype=torch.long, device=device)


def load_backbone_ckpt(model_name, seed, ds, device="cpu"):
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**default_kwargs(model_name, ds, device, tau=0.5)).to(device)
    path = f"{WEIGHTS_BACKBONE}/_backbone_{model_name}_e500_seed{seed}.pt"
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_ours_ckpt(model_name, seed, ds, device="cpu"):
    model_cls = MODEL_REGISTRY[model_name]
    debiased_cls = build_unshared_debias_model(model_cls)  # ablation="shared" CLI value
    kwargs = default_kwargs(model_name, ds, device, tau=0.1)
    kwargs["score_norm"] = "normalized"
    model = debiased_cls(**kwargs).to(device)
    lam = GAMMA_FOR[model_name]
    suffix = "cfhawkesanova" if model_name == "mf" else "hawkesanova"
    path = f"{WEIGHTS_OURS}/_{model_name}_lambdacen{lam}_tau0.1_scorenormnormalized_e500_seed{seed}_ablationshared_{suffix}.pt"
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ALPHA1_FOR[model_name]


def build_eval_batch(ds, data_split, max_seq_len=50):
    pairs = ds.set_to_pair(data_split, ds.time_dict, ds.time_unit)
    users, items, times = zip(*[(u, v, t) for (u, v), t in pairs.items()])
    users, items, times = np.array(users, np.int64), np.array(items, np.int64), np.array(times, np.float64)
    events = list(zip(users.tolist(), [0] * len(users), times.tolist()))
    hist_item_np, hist_time_np = ds.build_histories(events, max_seq_len)
    return users, items, times, hist_item_np, hist_time_np


def run_model_scores(model, ds, users, hist_item_np, hist_time_np=None, model_name=None, batch_size=256, device="cpu"):
    N, m_item = len(users), ds.m_item
    out = np.zeros((N, m_item), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            hist_t = torch.tensor(hist_item_np[s:e], dtype=torch.long, device=device)
            feat_t = additional_feat(model_name, users[s:e], hist_item_np[s:e],
                                      None if hist_time_np is None else hist_time_np[s:e], device)
            u = model.encode_user(hist_t, feat_t)
            v_all = model.get_item_repr(torch.arange(m_item, device=device))
            out[s:e] = torch.matmul(u, v_all.T).cpu().numpy()
    for i, u in enumerate(users):
        out[i, list(ds._allPos[u])] = -9999.0
    return out


def run_ours_scores(model, ds, users, times, hist_item_np, hist_time_np=None, model_name=None, batch_size=256, device="cpu"):
    """Returns (resid_all, logprob_all, mu, alpha, beta) -- combine post hoc via
    eta * logprob_all + (1-eta) * resid_all for any eta without recomputation."""
    N, m_item = len(users), ds.m_item
    resid_all = np.zeros((N, m_item), dtype=np.float32)
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()
        mu, alpha, beta = mu.cpu().numpy(), alpha.cpu().numpy(), float(beta.cpu().numpy())
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            hist_t = torch.tensor(hist_item_np[s:e], dtype=torch.long, device=device)
            feat_t = additional_feat(model_name, users[s:e], hist_item_np[s:e],
                                      None if hist_time_np is None else hist_time_np[s:e], device)
            u = F.normalize(model.encode_user(hist_t, feat_t), dim=-1, eps=1e-8)
            v_all = F.normalize(model.get_item_repr(torch.arange(m_item, device=device)), dim=-1, eps=1e-8)
            resid_all[s:e] = (torch.matmul(u, v_all.T) / model.tau).cpu().numpy()

    item_time_array = ds.item_time_array.astype(np.float32)
    logprob_all = np.zeros((N, m_item), dtype=np.float32)
    for s in range(0, N, 64):
        e = min(s + 64, N)
        t_batch = times[s:e].astype(np.float32).reshape(-1, 1, 1)
        mask = item_time_array[None, :, :] < t_batch
        delta = np.clip(t_batch - item_time_array[None, :, :], 0.0, None)
        h = (np.exp(-beta * delta) * mask).sum(-1)
        logits = mu[None, :] + alpha[None, :] * h
        logprob_all[s:e] = np.log(logits + 1e-12) - np.log(logits.sum(-1, keepdims=True) + 1e-12)
    return resid_all, logprob_all, mu, alpha, beta


def rank_of_true_item(scores, items):
    true_score = scores[np.arange(len(items)), items]
    return (scores > true_score[:, None]).sum(axis=1) + 1


def mask_pos(scores, ds, users):
    for i, u in enumerate(users):
        scores[i, list(ds._allPos[u])] = -9999.0
    return scores


def mode_sweep(args, ds):
    """Sweep eta post hoc on tail/head/all splits (no retraining)."""
    splits = {
        "tail_recent_3d": ds.test_tail_recent_3d_dict,
        "head_recent_3d": ds.test_head_recent_3d_dict,
        "all": ds.test_dict,
    }
    model, eta_default = load_ours_ckpt(args.model, args.seed, ds)
    backbone = load_backbone_ckpt(args.model, args.seed, ds)
    results = {}
    for split_name, split_dict in splits.items():
        users, items, times, hist_item_np, hist_time_np = build_eval_batch(ds, split_dict)
        if len(users) > args.max_n:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(users), size=args.max_n, replace=False)
            users, items, times, hist_item_np, hist_time_np = (
                users[idx], items[idx], times[idx], hist_item_np[idx], hist_time_np[idx])

        b_scores = run_model_scores(backbone, ds, users, hist_item_np, hist_time_np, model_name=args.model)
        b_r10 = float((rank_of_true_item(b_scores, items) <= 10).mean())

        resid_all, logprob_all, *_ = run_ours_scores(model, ds, users, times, hist_item_np, hist_time_np, model_name=args.model)
        row = {"n": len(users), "backbone_r10": b_r10, "eta_default": eta_default}
        for eta in ALPHA1_GRID:
            combined = mask_pos(eta * logprob_all + (1 - eta) * resid_all, ds, users)
            row[f"eta_{eta:.1f}"] = float((rank_of_true_item(combined, items) <= 10).mean())
        results[split_name] = row
        print(f"[{split_name}] backbone={b_r10:.4f} default(eta={eta_default})={row[f'eta_{eta_default:.1f}']:.4f}")

    out_path = f"{OUT_DIR}/{args.model}_eta_sweep.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print("saved", out_path)


def mode_concentration(args, ds):
    """Report mu concentration stats for the learned Hawkes prior."""
    model, eta = load_ours_ckpt(args.model, args.seed, ds)
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()
    mu = mu.numpy()
    top10 = np.sort(mu)[-10:]
    print(f"{args.model}: eta={eta} median(mu)={np.median(mu):.6g} max(mu)={mu.max():.2f} "
          f"ratio={mu.max()/np.median(mu):.0f} top10_share={top10.sum()/mu.sum()*100:.1f}% "
          f"(of {len(mu)} items)")


def mode_mucap(args, ds):
    """Cap mu at a percentile at inference (no retraining); recompute Tail Recall@10 at
    the *original* trained eta."""
    users, items, times, hist_item_np, hist_time_np = build_eval_batch(ds, ds.test_tail_recent_3d_dict)
    model, eta = load_ours_ckpt(args.model, args.seed, ds)
    resid_all, logprob_all_orig, mu, alpha, beta = run_ours_scores(
        model, ds, users, times, hist_item_np, hist_time_np, model_name=args.model)

    combined0 = mask_pos(eta * logprob_all_orig + (1 - eta) * resid_all, ds, users)
    r10_default = float((rank_of_true_item(combined0, items) <= 10).mean())
    print(f"{args.model}: eta={eta} uncapped tail R@10={r10_default:.4f}")

    item_time_array = ds.item_time_array.astype(np.float32)
    results = {"eta": eta, "r10_uncapped": r10_default}
    for pct in [90, 95, 99, 99.9]:
        cap = np.percentile(mu, pct)
        mu_capped = np.minimum(mu, cap)
        N = len(users)
        logprob_capped = np.zeros((N, ds.m_item), dtype=np.float32)
        for s in range(0, N, 64):
            e = min(s + 64, N)
            t_batch = times[s:e].astype(np.float32).reshape(-1, 1, 1)
            mask = item_time_array[None, :, :] < t_batch
            delta = np.clip(t_batch - item_time_array[None, :, :], 0.0, None)
            h = (np.exp(-beta * delta) * mask).sum(-1)
            logits = mu_capped[None, :] + alpha[None, :] * h
            logprob_capped[s:e] = np.log(logits + 1e-12) - np.log(logits.sum(-1, keepdims=True) + 1e-12)
        combined = mask_pos(eta * logprob_capped + (1 - eta) * resid_all, ds, users)
        r10 = float((rank_of_true_item(combined, items) <= 10).mean())
        results[f"r10_cap_p{pct}"] = r10
        print(f"  cap@p{pct} (mu<={cap:.4f}, orig max={mu.max():.1f}): tail R@10={r10:.4f}")

    out_path = f"{OUT_DIR}/{args.model}_mucap.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sasrec", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mode", default="sweep", choices=["sweep", "concentration", "mucap"])
    p.add_argument("--max-n", type=int, default=6000, help="subsample cap for large splits (sweep mode)")
    args = p.parse_args()

    ds = load_dataset()
    {"sweep": mode_sweep, "concentration": mode_concentration, "mucap": mode_mucap}[args.mode](args, ds)
