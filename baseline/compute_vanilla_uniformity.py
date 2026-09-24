#%%
"""
Uniformity L_uni(P_t) of the Vanilla backbones (no popularity/time
decomposition at all), from the norm_backbone_* checkpoints, at the same
single seed (1) used for the gamma_cen sweep in compute_gamma_uniformity.py
-- so the "Vanilla" reference point and the gamma_cen curve in
analysis_gamma_centering.py come from one consistent measurement protocol.

(Table 2/3 already reports this Vanilla number averaged over 4 seeds in
figures/table_luni_all_datasets.csv; the seed-1-only value computed here is
typically within its reported std and is used purely so every point on the
figure's x-axis is measured the same way.)

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/compute_vanilla_uniformity.py
"""
import csv
import torch
import torch.nn.functional as F
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY

DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
SEED = 1
RECDIM = 128
DROPOUT = 0.2
MAX_SEQ_LEN = 50
N_HEADS = 1
DEPTH = 0
BASELINE_TAU = 0.5
N_USERS_SAMPLE = 4000
BATCH_SIZE = 512
N_PAIRS = 20000

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


def build_baseline_model(model_name, dataset_name, n_user, m_item):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=BASELINE_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, **extra_kwargs(model_name, dataset_name),
    ).to(device)


def baseline_path(model_name, dataset_name, seed):
    return f"./weights/{dataset_name}/norm_backbone_{model_name}_e500_seed{seed}.pt"


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
results = []
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

    for model_name in BACKBONES:
        model = build_baseline_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
        ckpt = torch.load(baseline_path(model_name, dataset_name, SEED), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        reps = extract_representations(model, model_name, hist_item_np, hist_time_np, user_np)
        val = l_uni(reps)
        results.append((dataset_name, model_name, val))
        print(f"[{dataset_name:>11s} | {model_name:>8s}] Vanilla L_uni = {val:.4f}")

#%%
csv_path = "./figures/table_vanilla_uniformity.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "model_name", "l_uni"])
    for row in results:
        writer.writerow(row)
print(f"\n[saved] {csv_path}")
