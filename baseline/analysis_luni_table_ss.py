#%%
"""
Loss-function control, uniformity: L_uni(P_t) of the vanilla backbones trained
with the 15-negative sampled-softmax loss (cf_ss.py / seq_rec_ss.py /
seq_rec_tisasrec_ss.py, ./weights_sampled_softmax) vs +Ours, for all 6 backbones
x all 3 datasets, mean +- std over 4 seeds.

Protocol, sampling order and the +Ours checkpoints are identical to
analysis_luni_table.py (only the "Vanilla" checkpoints differ: BCE-trained
norm_backbone_* -> sampled-softmax-trained *_ss_*), so the +Ours column
reproduces figures/table_luni_all_datasets.csv exactly.

L_uni = log E[exp(-2 * ||h - h'||^2)] over sampled pairs of normalized
user-context representations (more negative = more uniform).

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_luni_table_ss.py
    (loss tag other than ss, e.g. BPR:  --loss-tag bpr --weights-dir ./weights_bpr)
"""
import argparse
import csv
import os

import torch
import torch.nn.functional as F
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

parser = argparse.ArgumentParser()
parser.add_argument("--weights-dir", default="./weights_sampled_softmax")
parser.add_argument("--loss-tag", default="ss")
parser.add_argument("--vanilla-tau", type=float, default=0.1)
parser.add_argument("--vanilla-score-norm", default="unnormalized")
parser.add_argument("--vanilla-corr", type=int, default=1, help="only used in the ss checkpoint name")
parser.add_argument("--out-csv", default="./figures/table_luni_sampled_softmax.csv")
cli = parser.parse_args()

DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
DISPLAY_NAME = {"mf": "MF", "grurec": "GRU", "sasrec": "SASRec",
                "tisasrec": "TiSASRec", "fearec": "FEARec", "bsarec": "BSARec"}
DISPLAY_DATASET = {"micro_video": "Micro Video", "ml-1m": "MovieLens-1M", "kuairand": "KuaiRand"}
SEEDS = [1, 2, 3, 4]
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
DECIMALS = 4

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


#%%
def extra_kwargs(model_name, dataset_name):
    if model_name == "bsarec":
        return dict(alpha=BSAREC_ALPHA_FOR[dataset_name], c=BSAREC_C)
    if model_name == "tisasrec":
        return dict(time_span=TIME_SPAN_FOR[dataset_name])
    return {}


def build_vanilla_model(model_name, dataset_name, n_user, m_item):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=cli.vanilla_tau, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, score_norm=cli.vanilla_score_norm,
        **extra_kwargs(model_name, dataset_name),
    ).to(device)


def build_ours_model(model_name, dataset_name, n_user, m_item):
    model_class = MODEL_REGISTRY[model_name]
    debiased_class = build_unshared_debias_model(model_class)
    return debiased_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, score_norm=SCORE_NORM,
        **extra_kwargs(model_name, dataset_name),
    ).to(device)


def vanilla_path(model_name, dataset_name, seed):
    return (f"{cli.weights_dir}/{dataset_name}/_{model_name}_{cli.loss_tag}_tau{cli.vanilla_tau}"
            f"_scorenorm{cli.vanilla_score_norm}_corr{cli.vanilla_corr}_e500_seed{seed}.pt")


def ours_path(model_name, dataset_name, seed):
    lam = WINNING_LAMBDA[dataset_name][model_name]
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
results = {}  # (dataset, backbone, approach) -> (mean, std) over seeds

for dataset_name in DATASETS:
    print(f"\n########## {dataset_name} ##########", flush=True)
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
        for approach, build_fn, path_fn in [
            ("Vanilla", build_vanilla_model, vanilla_path),
            ("Ours", build_ours_model, ours_path),
        ]:
            vals = []
            for seed in SEEDS:
                model = build_fn(model_name, dataset_name, dataset.n_user, dataset.m_item)
                ckpt = torch.load(path_fn(model_name, dataset_name, seed), map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                reps = extract_representations(model, model_name, hist_item_np, hist_time_np, user_np)
                vals.append(l_uni(reps))
            mean, std = np.mean(vals), np.std(vals)
            results[(dataset_name, model_name, approach)] = (mean, std)
            print(f"[{dataset_name:>11s} | {DISPLAY_NAME[model_name]:>8s} | {approach:>8s}] "
                  f"L_uni = {mean:.{DECIMALS}f} +- {std:.{DECIMALS}f}", flush=True)


#%%
def fmt(mean, std):
    return f"{mean:.{DECIMALS}f}±{std:.{DECIMALS}f}"


print("\n\n========== markdown ==========\n")
header = ["Backbone"] + [f"{DISPLAY_DATASET[d]} / {a}" for d in DATASETS for a in ("Vanilla", "Ours")]
print("| " + " | ".join(header) + " |")
print("|" + "|".join(["---"] * len(header)) + "|")
for model_name in BACKBONES:
    row = [DISPLAY_NAME[model_name]]
    for d in DATASETS:
        for a in ("Vanilla", "Ours"):
            row.append(fmt(*results[(d, model_name, a)]))
    print("| " + " | ".join(row) + " |")

print("\n\n========== LaTeX ==========\n")
print(r"\begin{tabular}{c|" + "cc|" * len(DATASETS) + "}")
print(r"\toprule")
print(" & ".join(["\\textbf{Backbone}"] +
                 [f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{DISPLAY_DATASET[d]}}}}}" for d in DATASETS]) + r" \\")
print(" & ".join([""] + ["Vanilla", "Ours"] * len(DATASETS)) + r" \\")
print(r"\midrule")
for model_name in BACKBONES:
    row = [f"\\textbf{{{DISPLAY_NAME[model_name]}}}"]
    for d in DATASETS:
        for a in ("Vanilla", "Ours"):
            m, s = results[(d, model_name, a)]
            row.append(f"{m:.{DECIMALS}f}$\\pm${s:.{DECIMALS}f}")
    print(" & ".join(row) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")


#%%
os.makedirs(os.path.dirname(cli.out_csv) or ".", exist_ok=True)
with open(cli.out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "backbone", "approach", "lambda_cen", "L_uni_mean", "L_uni_std"])
    for dataset_name in DATASETS:
        for model_name in BACKBONES:
            for approach in ["Vanilla", "Ours"]:
                mean, std = results[(dataset_name, model_name, approach)]
                lam = WINNING_LAMBDA[dataset_name][model_name] if approach == "Ours" else ""
                writer.writerow([dataset_name, DISPLAY_NAME[model_name], approach, lam,
                                 f"{mean:.{DECIMALS}f}", f"{std:.{DECIMALS}f}"])
print(f"\n[saved] {cli.out_csv}")
