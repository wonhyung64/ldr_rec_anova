#%%
"""
Direct evaluation of the centering condition of Assumption 1,

    c(v) = (1/|C|) sum_{x in C} f(x, v)        (should be 0 for every item v)

on the trained checkpoints (Vanilla / Ours gamma_cen=0 / Ours gamma_cen=tuned),
for 3 datasets x 6 backbones.

Contexts C: ALL test events (u, v, t) of the dataset; x_u(t) is the user's
history before t. For ALL items v, c(v) is the average of f(x, v) over every
test-time context (no sampling, no time binning). We report the mean and the
standard deviation of c(v) over items. Besides the raw value, we store c
rescaled by tau (= cosine units), since Vanilla and Ours are trained with
different temperatures (tau = 0.5 vs 0.1), and the sampling-noise floor of
the estimate, E_v[Var_x(f) / |C|].

Run from the repo root:
    PYTHONPATH=baseline /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/analysis_centering_assumption1.py
"""
import csv
import torch
import torch.nn.functional as F
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from analysis_centering_check import (
    DATASETS, BACKBONES, WINNING_LAMBDA, SEED, MAX_SEQ_LEN, HAWKES_TAU, BASELINE_TAU,
    build_ours_model, build_vanilla_model, ours_path, vanilla_path, device,
)

N_TIME_BINS = 1  # pooled over all test contexts
BATCH_SIZE = 512

set_seed(0)


@torch.no_grad()
def binned_item_sums(model, model_name, hist_item_np, hist_time_np, user_np, time_bin):
    """Per time bin b: sum_u f(x_u, v), sum_u f(x_u, v)^2 (over all items v)."""
    model.eval()
    m_item = model.num_items
    item_w = F.normalize(model.item_embedding.weight[:m_item], dim=-1, eps=1e-8)
    sum1 = torch.zeros(N_TIME_BINS, m_item, dtype=torch.float64)
    sum2 = torch.zeros(N_TIME_BINS, m_item, dtype=torch.float64)
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
        S = (h @ item_w.T / model.tau).double()
        bins = torch.tensor(time_bin[start:end])
        for b in range(N_TIME_BINS):
            sel = bins == b
            if sel.any():
                sum1[b] += S[sel].sum(0)
                sum2[b] += (S[sel] ** 2).sum(0)
    return sum1, sum2


def centering_stats(sum1, sum2, n_bin, tau):
    n = torch.tensor(n_bin, dtype=torch.float64).unsqueeze(1)
    c = sum1 / n                                  # c(v), (1 x items)
    var_u = (sum2 / n - c ** 2).clamp(min=0)      # Var_x f(., v)
    noise_sd = (var_u / n).mean().sqrt().item()
    return dict(
        c_mean=c.mean().item(),
        c_std=c.std(unbiased=False).item(),
        c_abs_mean=c.abs().mean().item(),
        c_mean_cos=c.mean().item() * tau,
        c_std_cos=c.std(unbiased=False).item() * tau,
        c_noise_sd=noise_sd,
        f_std_over_users=var_u.mean().sqrt().item(),
    )


#%%
if __name__ == "__main__":
    results = []
    for dataset_name in DATASETS:
        print(f"\n########## {dataset_name} ##########", flush=True)
        dataset = UserItemTime("./data", dataset_name, "d", 50, MAX_SEQ_LEN)

        pairs = list(dataset.test_user_item_time.items())
        events = [(u, v, t) for (u, v), t in pairs]  # all test-time contexts
        time_bin = np.zeros(len(events), dtype=np.int64)
        n_bin = np.bincount(time_bin, minlength=N_TIME_BINS)
        hist_item_np, hist_time_np = dataset.build_histories(events, MAX_SEQ_LEN)
        user_np = np.array([u for (u, _, _) in events], dtype=np.int64)
        print(f"[data] {len(events)} test contexts, {dataset.m_item} items", flush=True)

        for model_name in BACKBONES:
            settings = [
                ("vanilla", None),
                ("gamma0", 0.0),
                ("tuned", WINNING_LAMBDA[dataset_name][model_name]),
            ]
            for kind, gamma in settings:
                if kind == "vanilla":
                    model = build_vanilla_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
                    path, tau = vanilla_path(model_name, dataset_name, SEED), BASELINE_TAU
                else:
                    model = build_ours_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
                    path, tau = ours_path(model_name, dataset_name, SEED, gamma), HAWKES_TAU
                ckpt = torch.load(path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                sum1, sum2 = binned_item_sums(model, model_name, hist_item_np, hist_time_np, user_np, time_bin)
                st = centering_stats(sum1, sum2, n_bin, tau)
                results.append(dict(dataset=dataset_name, model_name=model_name, setting=kind,
                                    gamma=(np.nan if gamma is None else gamma), tau=tau, **st))
                print(f"[{dataset_name:>11s}|{model_name:>8s}|{kind:>7s}] "
                      f"c mean={st['c_mean']:+.4f} std={st['c_std']:.4f} "
                      f"(cos: {st['c_mean_cos']:+.4f} / {st['c_std_cos']:.4f}) "
                      f"noise_sd={st['c_noise_sd']:.4f} f_sd={st['f_std_over_users']:.3f}", flush=True)

    csv_path = "./figures/table_centering_assumption1.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {csv_path}")
