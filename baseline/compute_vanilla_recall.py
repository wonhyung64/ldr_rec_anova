#%%
"""
Recall@10 of the Vanilla backbones (no popularity/time decomposition at
all) on the held-out test set, from the same norm_backbone_* checkpoints
used for the Vanilla column of Table 2/3's uniformity table
(analysis_luni_table.py). These are not present in assets/sen_gamma.xlsx
(that sweep only logs "Ours" runs), so we evaluate them here with the same
scoring / top-N procedure as the original training scripts (cf.py,
seq_rec.py, seq_rec_tisasrec.py: model.score_all_items + computeTopNAccuracy),
just batched for speed.

This gives the gamma_cen < 0 ("Vanilla") reference point plotted to the
left of gamma_cen = 0 ("Ours w/o centering") in analysis_gamma_centering.py.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/compute_vanilla_recall.py
"""
import csv
import torch
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY, score_all
from module.procedure import computeTopNAccuracy

DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
SEED = 1
RECDIM = 128
DROPOUT = 0.2
MAX_SEQ_LEN = 50
N_HEADS = 1
DEPTH = 0
BASELINE_TAU = 0.5
BATCH_SIZE = 256
TOPKS = [10]

BSAREC_ALPHA_FOR = {"micro_video": 0.7, "ml-1m": 0.7, "kuairand": 0.9}
BSAREC_C = 1
TIME_SPAN_FOR = {"micro_video": 512, "ml-1m": 2048, "kuairand": 512}

device = "cpu"
set_seed(0)


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
def evaluate_recall(model, model_name, dataset, users, items_gt, hist_item_np, hist_time_np):
    model.eval()
    pred_list, gt_list = [], []
    n = len(users)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        hist_t = torch.tensor(hist_item_np[start:end], dtype=torch.long, device=device)
        if model_name == "tisasrec":
            time_t = torch.tensor(hist_time_np[start:end], dtype=torch.long, device=device) * 24 * 60 * 60
            pred = score_all(model, hist_t, time_t)
        else:
            user_t = torch.tensor(users[start:end], dtype=torch.long, device=device)
            pred = score_all(model, hist_t, user_t)
        pred = pred.cpu()
        for row_idx, user in enumerate(users[start:end]):
            exclude_items = list(dataset._allPos[user])
            pred[row_idx, exclude_items] = -9999
        _, pred_k = torch.topk(pred, k=max(TOPKS), dim=-1)
        pred_list.extend(pred_k.tolist())
        gt_list.extend([[it] for it in items_gt[start:end]])
    _, recall, _, _ = computeTopNAccuracy(gt_list, pred_list, TOPKS)
    return recall[0]


#%%
results = []
for dataset_name in DATASETS:
    print(f"\n########## {dataset_name} ##########")
    dataset = UserItemTime("./data", dataset_name, "d", 50, MAX_SEQ_LEN)

    test_items = list(dataset.test_user_item_time.items())
    users = np.array([u for (u, i), t in test_items], dtype=np.int64)
    items_gt = np.array([i for (u, i), t in test_items], dtype=np.int64)
    events = [(u, 0, t) for (u, i), t in test_items]
    hist_item_np, hist_time_np = dataset.build_histories(events, MAX_SEQ_LEN)
    print(f"[data] {len(test_items)} test queries over {len(set(users.tolist()))} users")

    for model_name in BACKBONES:
        model = build_baseline_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
        ckpt = torch.load(baseline_path(model_name, dataset_name, SEED), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        recall10 = evaluate_recall(model, model_name, dataset, users, items_gt, hist_item_np, hist_time_np)
        results.append((dataset_name, model_name, recall10))
        print(f"[{dataset_name:>11s} | {model_name:>8s}] Vanilla test_recall_10 = {recall10:.4f}")

#%%
csv_path = "./figures/table_vanilla_recall.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "model_name", "test_recall_10"])
    for row in results:
        writer.writerow(row)
print(f"\n[saved] {csv_path}")
