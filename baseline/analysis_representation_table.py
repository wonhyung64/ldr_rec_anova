#%%
"""
Theorem 1 / Corollary 1 support table (Micro-Video only).

For each backbone, compares:
  - "Backbone"  : plain backbone, no popularity-separation architecture at all
                  (weights/micro_video/norm_backbone_*.pt)
  - "+Ours"     : Hawkes+ANOVA-centering model at its validated lambda_cen
                  (weights_hawkes_anova_generalize/micro_video/*, the lambda_cen
                  value that was re-run over 4 seeds -- i.e. the one actually
                  reported in the paper's Table 2)

on three representation-geometry diagnostics computed from the learned,
unit-normalized user-context representations h_u(t):
  - PopAlign  : |correlation| between each user's projection onto the top
                (uncentered) singular direction of {h_u(t)} and that user's
                recent-item popularity exposure -- diagnoses whether the
                dominant shared direction is actually popularity-related.
  - PairDist  : mean pairwise squared distance over sampled user pairs
                (Theorem 1 LHS).
  - L_uni     : empirical uniformity loss, log E[exp(-2||h-h'||^2)]
                (Corollary 1 LHS; less negative = more collapsed).

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_representation_table.py
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import TruncatedSVD

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

DATASET = "micro_video"
SEEDS = [1, 2, 3, 4]
RECDIM = 128
DROPOUT = 0.2
MAX_SEQ_LEN = 50
N_HEADS = 1
DEPTH = 0
N_USERS_SAMPLE = 4000
BATCH_SIZE = 512
N_PAIRS = 20000

# validated lambda_cen per backbone: the one checkpointed over 4 seeds in
# weights_hawkes_anova_generalize/micro_video (i.e. what Table 2 reports)
WINNING_LAMBDA = {
    "mf": 1.0,
    "grurec": 1.0,
    "sasrec": 3.0,
    "tisasrec": 0.5,
    "fearec": 0.1,
    "bsarec": 0.1,
}
BSAREC_ALPHA, BSAREC_C = 0.7, 1  # micro_video setting (report_hawkes_anova_generalize.sh)
HAWKES_TAU = 0.1
BASELINE_TAU = 0.5
SCORE_NORM = "normalized"
ABLATION = "shared"

device = "cpu"
set_seed(0)
rng = np.random.default_rng(0)


#%%
# ----------------------------------------------------------------------
# data: one most-recent test-time context per user + recent-item popularity
# ----------------------------------------------------------------------
dataset = UserItemTime("./data", DATASET, "d", 50, MAX_SEQ_LEN)

pop_dict = np.load(f"./data/{DATASET}/item_popularity_recent_3d.npy", allow_pickle=True).item()
pop_array = np.zeros(dataset.m_item, dtype=np.float64)
for item, cnt in pop_dict.items():
    if item < dataset.m_item:
        pop_array[item] = cnt
log_pop_array = np.log1p(pop_array)

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

# per-user recent-popularity exposure: mean popularity of the real (non-padding)
# items in their history window
is_real = hist_item_np != dataset.m_item
user_pop_exposure = np.array([
    log_pop_array[hist_item_np[i, is_real[i]]].mean() if is_real[i].any() else 0.0
    for i in range(len(sampled_users))
])

print(f"[data] {len(sampled_users)} users sampled from {len(all_users)} test-active users")


#%%
# ----------------------------------------------------------------------
# model construction helpers
# ----------------------------------------------------------------------
def extra_kwargs(model_name):
    if model_name == "bsarec":
        return dict(alpha=BSAREC_ALPHA, c=BSAREC_C)
    if model_name == "tisasrec":
        return dict(time_span=512)  # micro_video (ml-1m alone uses 2048)
    return {}


def build_baseline_model(model_name):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_users=dataset.n_user, num_items=dataset.m_item, embedding_k=RECDIM,
        device=device, tau=BASELINE_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, **extra_kwargs(model_name),
    ).to(device)


def build_ours_model(model_name):
    model_class = MODEL_REGISTRY[model_name]
    debiased_class = build_unshared_debias_model(model_class)
    return debiased_class(
        num_users=dataset.n_user, num_items=dataset.m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, score_norm=SCORE_NORM, **extra_kwargs(model_name),
    ).to(device)


def baseline_path(model_name, seed):
    return f"./weights/{DATASET}/norm_backbone_{model_name}_e500_seed{seed}.pt"


def ours_path(model_name, seed):
    lam = WINNING_LAMBDA[model_name]
    suffix = "cfhawkesanova" if model_name == "mf" else "hawkesanova"
    return (
        f"./weights_hawkes_anova_generalize/{DATASET}/_{model_name}_lambdacen{lam}_tau{HAWKES_TAU}"
        f"_scorenorm{SCORE_NORM}_e500_seed{seed}_ablation{ABLATION}_{suffix}.pt"
    )


@torch.no_grad()
def extract_representations(model, model_name):
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


def geometry_metrics(reps):
    svd = TruncatedSVD(n_components=1)
    proj = svd.fit_transform(reps)[:, 0]
    pop_align = abs(np.corrcoef(proj, user_pop_exposure)[0, 1])

    n = reps.shape[0]
    pair_idx = rng.choice(n, size=(N_PAIRS, 2))
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    d2 = np.sum((reps[pair_idx[:, 0]] - reps[pair_idx[:, 1]]) ** 2, axis=1)
    pair_dist = d2.mean()
    l_uni = np.log(np.mean(np.exp(-2.0 * d2)))
    return pop_align, pair_dist, l_uni


#%%
# ----------------------------------------------------------------------
# sweep all backbones x {Backbone, +Ours} x 4 seeds
# ----------------------------------------------------------------------
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
DISPLAY_NAME = {"mf": "MF", "grurec": "GRU", "sasrec": "SASRec",
                "tisasrec": "TiSASRec", "fearec": "FEARec", "bsarec": "BSARec"}

table_rows = []
for model_name in BACKBONES:
    for approach, build_fn, path_fn in [
        ("Backbone", build_baseline_model, baseline_path),
        ("+Ours", build_ours_model, ours_path),
    ]:
        metrics_per_seed = []
        for seed in SEEDS:
            model = build_fn(model_name)
            ckpt = torch.load(path_fn(model_name, seed), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            reps = extract_representations(model, model_name)
            metrics_per_seed.append(geometry_metrics(reps))
        metrics_per_seed = np.array(metrics_per_seed)  # [4, 3]
        mean = metrics_per_seed.mean(axis=0)
        std = metrics_per_seed.std(axis=0)
        table_rows.append((DISPLAY_NAME[model_name], approach, mean, std))
        print(f"[{DISPLAY_NAME[model_name]:>8s} | {approach:>8s}] "
              f"PopAlign={mean[0]:.3f}+-{std[0]:.3f}  "
              f"PairDist={mean[1]:.3f}+-{std[1]:.3f}  "
              f"L_uni={mean[2]:.3f}+-{std[2]:.3f}   (lambda_cen={WINNING_LAMBDA[model_name]})")


#%%
# ----------------------------------------------------------------------
# pretty-print as a Table-2-style block + dump CSV
# ----------------------------------------------------------------------
import csv

csv_path = "./figures/table_representation_geometry_microvideo.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["backbone", "approach", "lambda_cen",
                      "PopAlign_mean", "PopAlign_std",
                      "PairDist_mean", "PairDist_std",
                      "L_uni_mean", "L_uni_std"])
    for name, approach, mean, std in table_rows:
        model_key = [k for k, v in DISPLAY_NAME.items() if v == name][0]
        lam = WINNING_LAMBDA[model_key] if approach == "+Ours" else ""
        writer.writerow([name, approach, lam,
                          f"{mean[0]:.4f}", f"{std[0]:.4f}",
                          f"{mean[1]:.4f}", f"{std[1]:.4f}",
                          f"{mean[2]:.4f}", f"{std[2]:.4f}"])
print(f"[saved] {csv_path}")
