#%%
"""
Corollary 1 support table: empirical uniformity loss L_uni(P_t), Vanilla vs
+Ours, for all 6 backbones x all 3 datasets, mean +- std over 4 seeds.

L_uni = log E[exp(-2 * ||h - h'||^2)] over sampled pairs of normalized
user-context representations h_u(t) (Corollary 1's LHS quantity). Less
negative (closer to 0) = more collapsed; more negative = more uniform.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_luni_table.py
"""
import torch
import torch.nn.functional as F
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

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
BASELINE_TAU = 0.5
SCORE_NORM = "normalized"
ABLATION = "shared"
N_USERS_SAMPLE = 4000
BATCH_SIZE = 512
N_PAIRS = 20000

# validated lambda_cen per (dataset, backbone) -- the one checkpointed over
# 4 seeds in weights_hawkes_anova_generalize/<dataset>, i.e. what the paper's
# own Table 2 reports as "+Ours"
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


def build_baseline_model(model_name, dataset_name, n_user, m_item):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=BASELINE_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, **extra_kwargs(model_name, dataset_name),
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


def baseline_path(model_name, dataset_name, seed):
    return f"./weights/{dataset_name}/norm_backbone_{model_name}_e500_seed{seed}.pt"


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
# ----------------------------------------------------------------------
# main sweep
# ----------------------------------------------------------------------
results = {}  # (dataset, backbone, approach) -> (mean, std) over 4 seeds

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
        for approach, build_fn, path_fn in [
            ("Vanilla", build_baseline_model, baseline_path),
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
                  f"L_uni = {mean:.4f} +- {std:.4f}")


#%%
# ----------------------------------------------------------------------
# render as a Backbone x (Dataset x {Vanilla, Ours}) table, mean +- std
# ----------------------------------------------------------------------
def fmt(mean, std):
    return f"{mean:.3f}$\\pm${std:.3f}"


print("\n\n========== LaTeX table ==========\n")
ncols = 1 + 2 * len(DATASETS)
print(r"\begin{tabular}{c|" + "cc|" * len(DATASETS) + "}")
print(r"\toprule")
header1 = ["\\textbf{Backbone}"]
for d in DATASETS:
    header1.append(f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{DISPLAY_DATASET[d]}}}}}")
print(" & ".join(header1) + r" \\")
header2 = [""]
for d in DATASETS:
    header2 += ["Vanilla", "Ours"]
print(" & ".join(header2) + r" \\")
print(r"\midrule")
for model_name in BACKBONES:
    row = [f"\\textbf{{{DISPLAY_NAME[model_name]}}}"]
    for d in DATASETS:
        m_v, s_v = results[(d, model_name, "Vanilla")]
        m_o, s_o = results[(d, model_name, "Ours")]
        row.append(fmt(m_v, s_v))
        row.append(fmt(m_o, s_o))
    print(" & ".join(row) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")

print("\n\n========== plain-text table ==========\n")
col_w = 16
header = f"{'Backbone':>10s} | " + " | ".join(
    f"{DISPLAY_DATASET[d]:^{2*col_w+3}s}" for d in DATASETS
)
print(header)
sub = f"{'':>10s} | " + " | ".join(f"{'Vanilla':>{col_w}s} {'Ours':>{col_w}s}" for _ in DATASETS)
print(sub)
print("-" * len(header))
for model_name in BACKBONES:
    cells = []
    for d in DATASETS:
        m_v, s_v = results[(d, model_name, "Vanilla")]
        m_o, s_o = results[(d, model_name, "Ours")]
        cells.append(f"{fmt(m_v, s_v):>{col_w}s} {fmt(m_o, s_o):>{col_w}s}")
    print(f"{DISPLAY_NAME[model_name]:>10s} | " + " | ".join(cells))


#%%
import csv
csv_path = "./figures/table_luni_all_datasets.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset", "backbone", "approach", "lambda_cen", "L_uni_mean", "L_uni_std"])
    for dataset_name in DATASETS:
        for model_name in BACKBONES:
            for approach in ["Vanilla", "Ours"]:
                mean, std = results[(dataset_name, model_name, approach)]
                lam = WINNING_LAMBDA[dataset_name][model_name] if approach == "Ours" else ""
                writer.writerow([dataset_name, DISPLAY_NAME[model_name], approach, lam,
                                  f"{mean:.4f}", f"{std:.4f}"])
print(f"\n[saved] {csv_path}")
