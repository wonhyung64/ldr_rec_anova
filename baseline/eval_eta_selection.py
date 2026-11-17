#%%
"""Select the inference-time eta on the validation "Tail, recent 3 days" slice (ICLR-27 rebuttal, W5 follow-up).

Loads an already trained checkpoint (no training) and, for every eta on a grid, computes Recall/NDCG on
    valid                    all validation events                  (the original selection criterion)
    valid_head/tail_recent_3d  validation events whose item is head/tail under the recent-3d popularity
                             (head = top 20% items by frequency in the 3 days before the validation period,
                              built exactly like the test slices but from train-only history -> no leakage)
    test, test_head/tail_recent_3d  the usual test slices (Table 1)
and reports, for each selection rule, the chosen eta and the test metrics obtained with it:
    valid_overall_recall@10        argmax_eta  valid Recall@10                   (what the paper did)
    valid_tail_recent_3d_recall@10 argmax_eta  valid-Tail(recent 3d) Recall@10   (this experiment)
    default_alpha1                 the eta used in Table 1 (--alpha1)

--setting joint  : checkpoints of the original jointly trained model   (*_hawkesanova.pt / *_cfhawkesanova.pt)
--setting plugin : checkpoints of the frozen-Phi plug-in model         (*_hawkesanova_plugin.pt / *_cfhawkesanova_plugin.pt)
The model is built exactly like in the training scripts, so pass the same --model-name/--dataset/--seed/--lambda-cen/
--tau/--score-norm/--ablation (and --alpha/--c for bsarec) as in training.
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
from module.debias import build_debias_model, build_unshared_debias_model, build_linear_debias_model, build_unshared_linear_debias_model
from module.plugin import parse_eta_grid, save_json
from module.eta_selection import (
    recent_head_tail_items, filter_dict_by_items, evaluate_eta_grid_batched, select_eta,
)

args = parse_args()
set_seed(args.seed)
args.device = set_device(args.device)
t_start = time.time()

#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
dataset.item_time_array = dataset.item_time_array / 86400.0   # same convention as the training scripts (days)

model_name = args.model_name.lower()
if model_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
model_class = MODEL_REGISTRY[model_name]
is_cf = model_name in ("mf", "ncf")
is_tisasrec = model_name == "tisasrec"

debiased_class = {
    "shared": build_unshared_debias_model, "linear": build_linear_debias_model,
    "none": build_debias_model, "both": build_unshared_linear_debias_model,
}[args.ablation](model_class)

model_kwargs = dict(
    num_users=dataset.n_user, num_items=dataset.m_item, embedding_k=args.recdim, device=args.device,
    tau=args.tau, depth=args.depth, max_seq_len=args.max_seq_len, n_heads=args.n_heads, dropout=args.dropout,
    score_norm=args.score_norm,
)
if model_name == "bsarec":
    model_kwargs.update(c=args.c, alpha=args.alpha)
if is_tisasrec:
    model_kwargs["time_span"] = 2048 if args.dataset == "ml-1m" else 512
model = debiased_class(**model_kwargs).to(args.device)

#%% checkpoint (file name pattern identical to the training scripts')
suffix = ("_cfhawkesanova" if is_cf else "_hawkesanova") + ("_plugin" if args.setting == "plugin" else "")
ckpt_path = (f"{args.weights_path}/{args.dataset}/_{args.model_name}_lambdacen{args.lambda_cen}_tau{args.tau}"
             f"_scorenorm{args.score_norm}_e{args.ckpt_epoch}_seed{args.seed}_ablation{args.ablation}{suffix}.pt")
if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f"trained checkpoint not found: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=args.device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
with torch.no_grad():
    _, _, beta_loaded = model.prior_parameters_from_embeddings()
print(f"[eta-select] loaded {ckpt_path} (epoch {ckpt['epoch']}), beta={float(beta_loaded):.6g}")


#%% utility scores for a batch of events (identical to the per-user evaluation of the training scripts)
@torch.no_grad()
def resid_fn(users, hist_item_np, hist_time_np):
    hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
    if is_tisasrec:
        feat = torch.tensor(hist_time_np, dtype=torch.long, device=args.device) * 24 * 60 * 60
    else:
        feat = torch.tensor(users, dtype=torch.long, device=args.device)
    if model_name == "bsarec":
        # BSARec's nn.MultiheadAttention has no batch_first, so with (B, L, E) inputs attention runs across the
        # batch axis and the users of a batch influence each other (max |diff| ~0.5 on scores of scale ~9).
        # The training scripts evaluate one user at a time, so score one event at a time here too.
        return torch.cat([score_all(model, hist_item_t[i:i + 1], feat[i:i + 1]) for i in range(len(users))], dim=0)
    return score_all(model, hist_item_t, feat)


#%% evaluation slices
days = args.select_days
head_v, tail_v, info_v = recent_head_tail_items(dataset, "valid", days)
valid_head = filter_dict_by_items(dataset.valid_dict, head_v)
valid_tail = filter_dict_by_items(dataset.valid_dict, tail_v)

# sanity: the same rule applied to the test period must reproduce the stored test slices
head_t, tail_t, info_t = recent_head_tail_items(dataset, "test", days)
stored_tail = getattr(dataset, f"test_tail_recent_{days}d_dict")
reproduced = {u: sorted(v) for u, v in filter_dict_by_items(dataset.test_dict, tail_t).items()} == \
             {u: sorted(v) for u, v in stored_tail.items()}
print(f"[eta-select] test tail-recent-{days}d split reproduced by the validation-split rule: {reproduced}")
if not reproduced:
    print("[eta-select] WARNING: re-derived test tail slice differs from data/*/testing_dict_tail_recent_*.npy")

pair = lambda d: dataset.set_to_pair(d, dataset.time_dict, dataset.time_unit)
split_pairs = [
    ("valid", dataset.valid_user_item_time),
    (f"valid_head_recent_{days}d", pair(valid_head)),
    (f"valid_tail_recent_{days}d", pair(valid_tail)),
    ("test", dataset.test_user_item_time),
    (f"test_head_recent_{days}d", pair(getattr(dataset, f"test_head_recent_{days}d_dict"))),
    (f"test_tail_recent_{days}d", pair(stored_tail)),
]
print("[eta-select] events: " + ", ".join(f"{n}={len(p)}" for n, p in split_pairs))

etas = sorted(set(parse_eta_grid(args.eta_select_grid) + [float(args.alpha1)]))
results = evaluate_eta_grid_batched(model, dataset, args, split_pairs, resid_fn, etas, batch_size=args.eval_batch_size)

#%% selection
vt, tt, ht = f"valid_tail_recent_{days}d", f"test_tail_recent_{days}d", f"test_head_recent_{days}d"
metric = "recall@10"


def test_at(eta_str):
    return {
        "eta": float(eta_str),
        "test_overall": results["test"][eta_str],
        "test_head": results[ht][eta_str],
        "test_tail": results[tt][eta_str],
    }


sel = {}
e, v = select_eta(results, "valid", metric)
sel["valid_overall_recall@10"] = {"selected_on": "valid", "valid_value": v, **test_at(e)}
e, v = select_eta(results, vt, metric)
sel[f"valid_tail_recent_{days}d_recall@10"] = {"selected_on": vt, "valid_value": v, **test_at(e)}
sel["default_alpha1"] = {"selected_on": None, **test_at(str(float(args.alpha1)))}
sel["utility_only_eta0"] = {"selected_on": None, **test_at("0.0")}

print("[eta-select] selection summary (test Recall@10: overall / head / tail)")
for name, s in sel.items():
    print(f"   {name:<34s} eta={s['eta']:.1f}  "
          f"{s['test_overall'][metric]:.4f} / {s['test_head'][metric]:.4f} / {s['test_tail'][metric]:.4f}")
print(f"   valid-tail {metric} per eta: " + " ".join(f"{e}:{results[vt][e][metric]:.4f}" for e in results[vt]))

out = {
    "setting": args.setting, "dataset": args.dataset, "model": args.model_name, "seed": args.seed,
    "gamma": args.lambda_cen, "tau": args.tau, "score_norm": args.score_norm, "eta_default": args.alpha1,
    "ckpt_path": ckpt_path, "ckpt_epoch": int(ckpt["epoch"]), "beta": float(beta_loaded),
    "recent_window_days": days, "valid_split_info": info_v, "test_split_info": info_t,
    "test_split_reproduced": bool(reproduced),
    "eta_grid": etas, "eval": results, "selection": sel,
    "elapsed_sec": round(time.time() - t_start, 1),
}
save_json(f"{args.eta_result_dir}/{args.setting}/{args.dataset}_{args.model_name}_gamma{args.lambda_cen}_seed{args.seed}.json", out)
