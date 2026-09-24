#%%
"""
Uniformity L_uni(P_t) of the learned user-context representations, swept
over the centering coefficient gamma_cen, for the same checkpoints used in
the Recall@10-vs-gamma sensitivity sweep (assets/sen_gamma.xlsx).

This reuses the exact L_uni definition and checkpoint-loading logic from
analysis_luni_table.py (Table 2/3's "Ours" column is WINNING_LAMBDA's
special case of this same sweep) so the two figures are directly
comparable. Only checkpoint architecture matters for encode_user(), so the
per-model "eta" (test-time recency/score-mixing weight) used elsewhere
does not need to be reproduced here.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/compute_gamma_uniformity.py
"""
import csv
import torch
import torch.nn.functional as F
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
GAMMA_GRID = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0]
SEED = 1
RECDIM = 128
DROPOUT = 0.2
MAX_SEQ_LEN = 50
N_HEADS = 1
DEPTH = 0
HAWKES_TAU = 0.1
SCORE_NORM = "normalized"
ABLATION = "shared"
N_USERS_SAMPLE = 4000
BATCH_SIZE = 512
N_PAIRS = 20000

# the lambda_cen actually used in the paper's main-table checkpoints
# (Table 2/3 "+Ours") -- the point highlighted as "Full" in the figure
WINNING_LAMBDA = {
    "micro_video": {"mf": 1.0, "grurec": 1.0, "sasrec": 3.0, "tisasrec": 0.5, "fearec": 0.1, "bsarec": 0.1},
    "ml-1m":       {"mf": 3.0, "grurec": 10.0, "sasrec": 0.5, "tisasrec": 1.0, "fearec": 0.1, "bsarec": 0.1},
    "kuairand":    {"mf": 0.1, "grurec": 10.0, "sasrec": 0.5, "tisasrec": 0.1, "fearec": 0.1, "bsarec": 0.1},
}
BSAREC_ALPHA_FOR = {"micro_video": 0.7, "ml-1m": 0.7, "kuairand": 0.9}
BSAREC_C = 1
TIME_SPAN_FOR = {"micro_video": 512, "ml-1m": 2048, "kuairand": 512}

device = "cpu"
set_seed(0)
rng = np.random.default_rng(0)


def extra_kwargs(model_name, dataset_name):
    if model_name == "bsarec":
        return dict(alpha=BSAREC_ALPHA_FOR[dataset_name], c=BSAREC_C)
    if model_name == "tisasrec":
        return dict(time_span=TIME_SPAN_FOR[dataset_name])
    return {}


def build_ours_model(model_name, dataset_name, n_user, m_item):
    model_class = MODEL_REGISTRY[model_name]
    debiased_class = build_unshared_debias_model(model_class)
    return debiased_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, score_norm=SCORE_NORM,
        **extra_kwargs(model_name, dataset_name),
    ).to(device)


def ours_path(model_name, dataset_name, seed, lam):
    suffix = "cfhawkesanova" if model_name == "mf" else "hawkesanova"
    return (
        f"./weights_hawkes_anova_generalize/{dataset_name}/_{model_name}_lambdacen{lam}_tau{HAWKES_TAU}"
        f"_scorenorm{SCORE_NORM}_e500_seed{seed}_ablation{ABLATION}_{suffix}.pt"
    )


@torch.no_grad()
def extract_representations(model, model_name, hist_item_np, hist_time_np, user_np):
    model.eval()
    reps = []
    for start in range(0, len(user_np), BATCH_SIZE):
        end = start + BATCH_SIZE
        hist_t = torch.tensor(hist_item_np[start:end], dtype=torch.long, device=device)
        if model_name == "tisasrec":
            time_t = torch.tensor(hist_time_np[start:end], dtype=torch.long, device=device) * 24 * 60 * 60
            h = model.encode_user(hist_t, time_t)
        else:
            user_t = torch.tensor(user_np[start:end], dtype=torch.long, device=device)
            h = model.encode_user(hist_t, user_t)
        h = F.normalize(h, dim=-1, eps=1e-8)
        reps.append(h.cpu().numpy())
    return np.concatenate(reps, axis=0)


def l_uni(reps):
    n = reps.shape[0]
    pair_idx = rng.choice(n, size=(N_PAIRS, 2))
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    d2 = np.sum((reps[pair_idx[:, 0]] - reps[pair_idx[:, 1]]) ** 2, axis=1)
    return np.log(np.mean(np.exp(-2.0 * d2)))


#%%
results = []  # rows: dataset, backbone, gamma, l_uni, is_winning

for dataset_name in DATASETS:
    print(f"\n########## {dataset_name} ##########")
    dataset = UserItemTime("./data", dataset_name, "d", 50, MAX_SEQ_LEN)

    last_event = {}
    for (user, item), t in dataset.test_user_item_time.items():
        if (user not in last_event) or (t > last_event[user][1]):
            last_event[user] = (item, t)
    all_users = np.array(sorted(last_event.keys()))
    sampled_users = (
        rng.choice(all_users, size=N_USERS_SAMPLE, replace=False)
        if len(all_users) > N_USERS_SAMPLE else all_users
    )
    sampled_users.sort()
    events = [(u, last_event[u][0], last_event[u][1]) for u in sampled_users]
    hist_item_np, hist_time_np = dataset.build_histories(events, MAX_SEQ_LEN)
    user_np = np.array([u for (u, v, t) in events], dtype=np.int64)
    print(f"[data] {len(sampled_users)} users sampled from {len(all_users)} test-active users")

    for model_name in BACKBONES:
        winning_gamma = WINNING_LAMBDA[dataset_name][model_name]
        for gamma in GAMMA_GRID:
            model = build_ours_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
            lam_str = f"{gamma:.1f}"  # filenames use e.g. "0.0", "0.1", "0.5", "1.0", "3.0", "10.0"
            ckpt = torch.load(ours_path(model_name, dataset_name, SEED, lam_str), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            reps = extract_representations(model, model_name, hist_item_np, hist_time_np, user_np)
            val = l_uni(reps)
            is_winning = abs(gamma - winning_gamma) < 1e-9
            results.append((dataset_name, model_name, gamma, val, is_winning))
            print(f"[{dataset_name:>11s} | {model_name:>8s} | gamma={gamma:>4g}] "
                  f"L_uni = {val:.4f}{'  <-- Full' if is_winning else ''}")


#%%
csv_path = "./figures/table_gamma_uniformity.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "model_name", "gamma", "l_uni", "is_winning"])
    for row in results:
        writer.writerow(row)
print(f"\n[saved] {csv_path}")
