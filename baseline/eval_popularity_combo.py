#%%
"""Utility + popularity ranking with the popularity signal computed from ALL interactions observable at the query time,
including the test events that happened before it (ICLR-27 rebuttal, W5 follow-up).  Nothing is trained.

    score(u, v, t) = eta * log pi(v | t) + (1 - eta) * f_Psi(u, v, t)          (eq. (3) with a swapped pi)

pi signals (all evaluated on the test events, using test-period interactions):
    global_all      item count over ALL data (train+valid+test)  + eps                  [non-causal: includes the event]
    global_all_loo  same, with the query event itself removed                           [non-causal, no self leak]
    slide_3d        count in [t - 3d, t) + eps                                           [causal, test events before t]
    hawkes          lambda_v(t) of the frozen plug-in Phi (stored in the checkpoint), history = what item_time_array holds
                    (the last 50 events of every item, test events included)             [causal]
f_Psi is the trained plug-in utility model (checkpoint of `--setting plugin`).  eta = 0 is the utility-only reference.
Evaluation protocol = Table 1/4 (full ranking, training items masked, one ground truth per event);
splits: test, test head / tail (recent 3d).
"""
import os
import sys
import json
import time

import numpy as np
import torch

from module.utils import parse_args, set_seed, set_device
from module.dataset import UserItemTime
from module.model import score_all, MODEL_REGISTRY
from module.debias import build_unshared_debias_model
from module.plugin import parse_eta_grid, save_json

from module.popularity_stream import EventStream, HawkesState, WindowCounter

args = parse_args()
set_seed(args.seed)
args.device = set_device(args.device)
t_start = time.time()

#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
dataset.item_time_array = dataset.item_time_array / 86400.0
m = dataset.m_item

model_name = args.model_name.lower()
model_class = MODEL_REGISTRY[model_name]
is_tisasrec = model_name == "tisasrec"
is_cf = model_name in ("mf", "ncf")
model_kwargs = dict(
    num_users=dataset.n_user, num_items=m, embedding_k=args.recdim, device=args.device, tau=args.tau,
    depth=args.depth, max_seq_len=args.max_seq_len, n_heads=args.n_heads, dropout=args.dropout,
    score_norm=args.score_norm,
)
if model_name == "bsarec":
    model_kwargs.update(c=args.c, alpha=args.alpha)
if is_tisasrec:
    model_kwargs["time_span"] = 2048 if args.dataset == "ml-1m" else 512
model = build_unshared_debias_model(model_class)(**model_kwargs).to(args.device)

suffix = ("_cfhawkesanova" if is_cf else "_hawkesanova") + "_plugin"
ckpt_path = (f"{args.weights_path}/{args.dataset}/_{args.model_name}_lambdacen{args.lambda_cen}_tau{args.tau}"
             f"_scorenorm{args.score_norm}_e{args.ckpt_epoch}_seed{args.seed}_ablation{args.ablation}{suffix}.pt")
ckpt = torch.load(ckpt_path, map_location=args.device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
with torch.no_grad():
    mu_t, alpha_t, beta_t = model.prior_parameters_from_embeddings()
mu, alpha, beta = mu_t.double().cpu().numpy(), alpha_t.double().cpu().numpy(), float(beta_t)
print(f"[pop-combo] loaded {ckpt_path}; frozen plug-in Phi beta={beta:.4f}/day")


@torch.no_grad()
def resid_fn(users, hist_item_np, hist_time_np):
    hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
    if is_tisasrec:
        feat = torch.tensor(hist_time_np, dtype=torch.long, device=args.device) * 24 * 60 * 60
    else:
        feat = torch.tensor(users, dtype=torch.long, device=args.device)
    if model_name == "bsarec":   # per-event scoring, as in eval_eta_selection.py / the training scripts
        return torch.cat([score_all(model, hist_item_t[i:i + 1], feat[i:i + 1]) for i in range(len(users))], dim=0)
    return score_all(model, hist_item_t, feat)


#%% events (chronological) and head / tail membership
pairs = sorted(dataset.test_user_item_time.items(), key=lambda kv: kv[1])
if args.pop_max_events:
    pairs = pairs[: args.pop_max_events]
keys = [k for k, _ in pairs]
times = np.array([t for _, t in pairs], dtype=np.float64)
users_all = np.array([k[0] for k in keys])
gt = np.array([k[1] for k in keys])
pair_set = lambda d: set(dataset.set_to_pair(d, dataset.time_dict, dataset.time_unit).keys())
head_keys, tail_keys = pair_set(dataset.test_head_recent_3d_dict), pair_set(dataset.test_tail_recent_3d_dict)
is_head = np.array([k in head_keys for k in keys])
is_tail = np.array([k in tail_keys for k in keys])
splits = {"test": np.ones(len(keys), bool), "test_head_recent_3d": is_head, "test_tail_recent_3d": is_tail}
print(f"[pop-combo] events: " + ", ".join(f"{n}={int(v.sum())}" for n, v in splits.items()))

#%% popularity signals
stream = EventStream(dataset)
EPS = args.pop_eps
c_all = np.zeros(m)
for _, d in dataset.time_dict.items():
    for i in d:
        c_all[i] += 1
hawkes = HawkesState(mu, alpha, beta, stream, stream.keep_deployed)
slide = WindowCounter(stream, m, -args.pop_window_days, 0.0)
states = [hawkes, slide]


def lognorm(w):
    w = np.asarray(w, dtype=np.float64)
    return np.log(w) - np.log(w.sum())


def pop_logpi(t, v):
    """dict signal -> log pi(. | t) for the query event (item v)."""
    lo = c_all + EPS
    loo = c_all.copy()
    loo[v] -= 1.0
    return {
        "global_all": lognorm(lo),
        "global_all_loo": lognorm(loo + EPS),
        "slide_3d": lognorm(slide.cnt + EPS),
        "hawkes": lognorm(hawkes.weights(t) + 1e-12),
    }


SIGNALS = ["global_all", "global_all_loo", "slide_3d", "hawkes"]
etas = parse_eta_grid(args.pop_etas)
combos = [("utility_only", 0.0)] + [(s, e) for s in SIGNALS for e in etas if e > 0]
max_k = max(args.topks)
preds = {c: np.zeros((len(keys), max_k), dtype=np.int64) for c in combos}

#%% ranking
B = args.eval_batch_size
tl = time.time()
for s in range(0, len(keys), B):
    e_ = min(s + B, len(keys))
    users = users_all[s:e_].tolist()
    ts = times[s:e_].tolist()
    nB = e_ - s
    logpi = {sig: np.zeros((nB, m), dtype=np.float32) for sig in SIGNALS}
    for j in range(nB):
        for st in states:
            st.advance(ts[j])
        for sig, vec in pop_logpi(ts[j], int(gt[s + j])).items():
            logpi[sig][j] = vec
    hist_item_np, hist_time_np = dataset.build_histories(zip(users, [0] * nB, ts), args.max_seq_len)
    resid = resid_fn(users, hist_item_np, hist_time_np).float()
    seen = torch.zeros(nB, m, dtype=torch.bool, device=args.device)
    for j, u in enumerate(users):
        seen[j, torch.as_tensor(np.asarray(dataset._allPos[u], dtype=np.int64), device=args.device)] = True
    for (sig, eta) in combos:
        if sig == "utility_only":
            pred = resid
        else:
            pred = torch.from_numpy(logpi[sig]).to(args.device) * eta + resid * (1 - eta)
        top = torch.topk(pred.masked_fill(seen, -9999), k=max_k, dim=-1)[1]
        preds[(sig, eta)][s:e_] = top.cpu().numpy()
    if (s // B) % 50 == 0:
        print(f"  {e_}/{len(keys)} events ({time.time() - tl:.0f}s)")


#%% metrics (single ground truth per event -> closed forms of computeTopNAccuracy)
def metrics(top, mask):
    hit_pos = (top[mask] == gt[mask][:, None])
    found = hit_pos.any(1)
    pos = np.where(found, hit_pos.argmax(1), 10 ** 9)          # 0-indexed rank
    out = {"n_events": int(mask.sum())}
    for k in args.topks:
        h = pos < k
        out[f"precision@{k}"] = float(h.mean() / k)
        out[f"recall@{k}"] = float(h.mean())
        out[f"ndcg@{k}"] = float(np.where(h, 1.0 / np.log2(pos + 2.0), 0.0).mean())
        out[f"mrr@{k}"] = float(np.where(h, 1.0 / (pos + 1.0), 0.0).mean())
    return out


results = {}
for (sig, eta), top in preds.items():
    results[f"{sig}@eta{eta:g}"] = {sp: metrics(top, mk) for sp, mk in splits.items()}

print("[pop-combo] test Recall@10  (overall / head / tail)   [NDCG@10 overall]")
for name, r in results.items():
    print(f"   {name:<24s} {r['test']['recall@10']:.4f} / {r['test_head_recent_3d']['recall@10']:.4f} / "
          f"{r['test_tail_recent_3d']['recall@10']:.4f}    {r['test']['ndcg@10']:.4f}")

out = dict(dataset=args.dataset, model=args.model_name, seed=args.seed, gamma=args.lambda_cen, tau=args.tau,
           ckpt_path=ckpt_path, ckpt_epoch=int(ckpt["epoch"]), beta=beta, window_days=args.pop_window_days, eps=EPS,
           etas=etas, n_events=len(keys), results=results, elapsed_sec=round(time.time() - t_start, 1))
save_json(f"{args.pop_result_dir}/{args.dataset}_{args.model_name}_gamma{args.lambda_cen}_seed{args.seed}.json", out)
