#%%
"""
Does the learned utility f_Psi(x_u(t), v) already satisfy Assumption 1
(sum_u f(x_u(t), v) = 0 for every item v and time t) when the centering
coefficient is gamma_cen = 0?

Reviewers point out that Fig. 2 shows nearly the same Recall@10 / uniformity at
gamma_cen = 0 as at the best gamma_cen > 0. Our hypothesis is that, with the
separate popularity component pi_Phi(v|t), the utility is *already* driven
towards the centered solution without the explicit penalty. This script
checks it on the trained checkpoints of the compute_gamma_uniformity.py sweep
(Ours, gamma_cen in GAMMA_GRID, seed 1) plus the Vanilla backbone (no
popularity component) as a reference.

For every checkpoint we build the user-by-item utility matrix S[u, v] =
f(x_u(t_u), v) over a sample of test-time contexts and all items, and report

  rcen_unb        U-statistic estimate of R_cen = E_{t,v}[ (E_u f)^2 ]
                  (exactly the quantity l_cen estimates; unbiased, so no
                  finite-sample floor), computed within time bins
  item_frac       rcen_unb / E[f^2]: share of the utility's second moment
                  that is an item-main effect, i.e. a violation of centering.
                  0 = perfectly centered, 1 = utility is purely item-dependent
  ubar_norm       || mean_u normalized user representation ||. Because f is a
                  cosine score, E_u f(.,v) = ubar . v_hat / tau, so centering
                  is equivalent to a vanishing mean user direction (the shared
                  direction c_t of Theorem 1)
  pop_corr        Spearman between item mean utility m_v = E_u f(., v) and
                  log training popularity: does f carry a popularity signal?
  rank_overlap10  mean top-10 overlap between ranking by f and ranking by the
                  centered f - m_v (per user). 1 = the residual non-centering
                  does not change any top-10 list

Run from the repo root:
    PYTHONPATH=baseline /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/analysis_centering_check.py
"""
import csv
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

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
BASELINE_TAU = 0.5
SCORE_NORM = "normalized"
ABLATION = "shared"
N_USERS_SAMPLE = 4000
N_TIME_BINS = 5
BATCH_SIZE = 512
TOPK = 10

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
    debiased_class = build_unshared_debias_model(MODEL_REGISTRY[model_name])
    return debiased_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, score_norm=SCORE_NORM,
        **extra_kwargs(model_name, dataset_name),
    ).to(device)


def build_vanilla_model(model_name, dataset_name, n_user, m_item):
    return MODEL_REGISTRY[model_name](
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=BASELINE_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, **extra_kwargs(model_name, dataset_name),
    ).to(device)


def ours_path(model_name, dataset_name, seed, gamma):
    suffix = "cfhawkesanova" if model_name == "mf" else "hawkesanova"
    return (
        f"./weights_hawkes_anova_generalize/{dataset_name}/_{model_name}_lambdacen{gamma:.1f}_tau{HAWKES_TAU}"
        f"_scorenorm{SCORE_NORM}_e500_seed{seed}_ablation{ABLATION}_{suffix}.pt"
    )


def vanilla_path(model_name, dataset_name, seed):
    return f"./weights/{dataset_name}/norm_backbone_{model_name}_e500_seed{seed}.pt"


@torch.no_grad()
def utility_matrix(model, model_name, hist_item_np, hist_time_np, user_np):
    """S[u, v] = f(x_u, v) = <u_hat, v_hat> / tau for all items, plus the
    normalized user representations (needed for ||ubar||)."""
    model.eval()
    item_w = F.normalize(model.item_embedding.weight[: model.num_items], dim=-1, eps=1e-8)
    S, U = [], []
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
        U.append(h)
        S.append(h @ item_w.T / model.tau)
    return torch.cat(S), torch.cat(U)


def u_stat_mean_sq(S):
    """Unbiased estimate of mean_v (E_u S[u, v])^2 (U-statistic, no diagonal)."""
    n = S.shape[0]
    return ((S.sum(0) ** 2 - (S ** 2).sum(0)) / (n * (n - 1))).mean().item()


def topk_overlap(S, m, k=TOPK):
    top_f = torch.topk(S, k, dim=1).indices
    top_c = torch.topk(S - m.unsqueeze(0), k, dim=1).indices
    hits = (top_f.unsqueeze(2) == top_c.unsqueeze(1)).any(dim=2).sum(dim=1).float()
    return (hits / k).mean().item()


def centering_stats(S, U, time_bin, log_pop):
    n_bins = int(time_bin.max()) + 1
    rcen, w = 0.0, 0
    for b in range(n_bins):
        Sb = S[torch.tensor(time_bin == b)]
        if Sb.shape[0] < 3:
            continue
        rcen += u_stat_mean_sq(Sb) * Sb.shape[0]
        w += Sb.shape[0]
    rcen /= w
    second_moment = (S ** 2).mean().item()
    m = S.mean(0)
    within_sd = S.std(0).mean().item()
    return dict(
        rcen_unb=rcen,
        second_moment=second_moment,
        item_frac=rcen / second_moment,
        item_sd_between=m.std().item(),
        item_sd_within=within_sd,
        ubar_norm=U.mean(0).norm().item(),
        pop_corr=spearmanr(m.numpy(), log_pop).correlation,
        rank_overlap10=topk_overlap(S, m),
    )


#%%
if __name__ == "__main__":
    results = []
    for dataset_name in DATASETS:
        print(f"\n########## {dataset_name} ##########")
        dataset = UserItemTime("./data", dataset_name, "d", 50, MAX_SEQ_LEN)

        counts = np.bincount(dataset.trainItem, minlength=dataset.m_item)
        log_pop = np.log1p(counts)

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
        times = np.array([t for (u, v, t) in events], dtype=np.float64)
        # equal-count time bins: Assumption 1 is a per-time statement
        order = np.argsort(times, kind="stable")
        time_bin = np.empty(len(times), dtype=np.int64)
        time_bin[order] = np.arange(len(times)) * N_TIME_BINS // len(times)
        print(f"[data] {len(sampled_users)} users / {dataset.m_item} items, {N_TIME_BINS} time bins")

        for model_name in BACKBONES:
            ckpts = [("vanilla", None)] + [("ours", g) for g in GAMMA_GRID]
            for kind, gamma in ckpts:
                if kind == "vanilla":
                    model = build_vanilla_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
                    path = vanilla_path(model_name, dataset_name, SEED)
                else:
                    model = build_ours_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
                    path = ours_path(model_name, dataset_name, SEED, gamma)
                ckpt = torch.load(path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                S, U = utility_matrix(model, model_name, hist_item_np, hist_time_np, user_np)
                stats = centering_stats(S, U, time_bin, log_pop)
                is_winning = kind == "ours" and abs(gamma - WINNING_LAMBDA[dataset_name][model_name]) < 1e-9
                tag = "vanilla" if kind == "vanilla" else f"g={gamma:g}"
                row = dict(dataset=dataset_name, model_name=model_name, kind=kind,
                           gamma=(np.nan if gamma is None else gamma), is_winning=is_winning, **stats)
                results.append(row)
                print(f"[{dataset_name:>11s}|{model_name:>8s}|{tag:>8s}] "
                      f"item_frac={stats['item_frac']:.4f} Rcen={stats['rcen_unb']:.4f} "
                      f"|ubar|={stats['ubar_norm']:.3f} pop_corr={stats['pop_corr']:+.3f} "
                      f"overlap@10={stats['rank_overlap10']:.3f}{'  <-- Full' if is_winning else ''}", flush=True)
                sys.stdout.flush()

    csv_path = "./figures/table_centering_check.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {csv_path}")
