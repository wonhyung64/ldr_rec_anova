#%%
"""
Centering value of Assumption 1 at every test-time context, for every item:

    c(v, t) = (1/|U|) sum_{u in U} f(x_u(t), v)

where x_u(t) is the context of user u at time t (u's history before t) and U is
the set of ALL users (users with at least one interaction in the dataset).
t runs over the time of every test event ("test-time context") and v over all
items.

Because f is a cosine score, f(x, v) = <u_hat(x), v_hat> / tau and therefore

    c(v, t) = < ubar(t), v_hat > / tau,     ubar(t) = (1/|U|) sum_u u_hat(x_u(t)),

so the full (|test contexts| x |V|) matrix c is exactly represented by the mean
user representation ubar(t_i) of every test time and the item representations
V = {v_hat}; we store (ubar, V, tau) per checkpoint (the dense matrix would be
~70 GB). Materialize with load_c(path).

ubar(t) for all test times is computed exactly, but cheaply: all users are
encoded once just before the first test time, and afterwards a user's
representation only changes when that user has an interaction, so every
interaction after t_0 is encoded once and ubar(t) is obtained by cumulating the
representation changes (contexts only depend on the user's history).

Per checkpoint (figures/centering_c/{dataset}_{backbone}_{setting}.npz,
setting in {vanilla, gamma0, tuned}):
    ubar (n_ctx, d) float32, V (|V|, d) float32, tau, gamma (nan for Vanilla),
    n_users, n_users_empty_hist (users without any history at the first test time)
Shared: figures/centering_c/contexts_{dataset}.npz  (user, item, time of every test context,
        row i <-> ubar[i]).
Summary: figures/table_centering_c.csv  (mean / std of c over all (test time, item) pairs
        and over items of the time-averaged c).

Run from the repo root:
    PYTHONPATH=baseline /Users/wonhyung64/miniforge3/envs/rank/bin/python baseline/save_centering_values.py
"""
import csv
import os
import torch
import torch.nn.functional as F
import numpy as np

from module.utils import set_seed
from module.dataset import UserItemTime
from analysis_centering_check import (
    DATASETS, BACKBONES, WINNING_LAMBDA, SEED, MAX_SEQ_LEN, HAWKES_TAU, BASELINE_TAU,
    build_ours_model, build_vanilla_model, ours_path, vanilla_path, device,
)

OUT_DIR = "./figures/centering_c"
CSV_PATH = "./figures/table_centering_c.csv"
BATCH_SIZE = 512
N_CHECK_TIMES = 3

set_seed(0)
rng = np.random.default_rng(0)


def load_c(path, rows=None, cols=None):
    """c(v, t_i) = <ubar(t_i), v_hat> / tau for the selected test times / items (default all)."""
    z = np.load(path)
    A, V = z["ubar"], z["V"]
    if rows is not None:
        A = A[rows]
    if cols is not None:
        V = V[cols]
    return A @ V.T / float(z["tau"])


@torch.no_grad()
def encode(model, model_name, hist_items, hist_times, users):
    """Normalized context representation u_hat for every (history, user) row."""
    model.eval()
    reps = []
    # BSARec's nn.MultiheadAttention is called on [batch, seq, d] without batch_first, so it attends
    # ACROSS the users of a batch and a user's representation depends on its batch mates. Encode one
    # user at a time (as the test-time scoring does) so that the context representation is well defined.
    bs = 1 if model_name == "bsarec" else BATCH_SIZE
    for start in range(0, len(users), bs):
        end = start + bs
        hist_t = torch.tensor(hist_items[start:end], dtype=torch.long, device=device)
        if model_name == "tisasrec":
            time_t = torch.tensor(hist_times[start:end], dtype=torch.long, device=device) * 24 * 60 * 60
            h = model.encode_user(hist_t, time_t)
        else:
            user_t = torch.tensor(users[start:end], dtype=torch.long, device=device)
            h = model.encode_user(hist_t, user_t)
        reps.append(F.normalize(h, dim=-1, eps=1e-8).double())
    return torch.cat(reps).numpy()


def aggregate(A, V, tau):
    """Exact mean / std of c over all (time, item) pairs without materializing c, via
    c = A V^T / tau:  E[c] = mean(A) . mean(V) / tau,  E[c^2] = mean_i a_i^T G a_i / tau^2, G = V^T V / |V|.
    Also the same for the time-averaged c(v) = abar . v_hat / tau (std over items)."""
    A, V = A.astype(np.float64), V.astype(np.float64)
    G = V.T @ V / len(V)
    vbar = V.mean(0)
    abar = A.mean(0)
    mean_all = abar @ vbar / tau
    m2_all = np.einsum("id,de,ie->i", A, G, A).mean() / tau ** 2
    mean_pool = mean_all
    m2_pool = abar @ G @ abar / tau ** 2
    return dict(
        c_mean=mean_all, c_std=np.sqrt(max(m2_all - mean_all ** 2, 0.0)),
        c_pooled_std=np.sqrt(max(m2_pool - mean_pool ** 2, 0.0)),
        c_mean_cos=mean_all * tau, c_std_cos=np.sqrt(max(m2_all - mean_all ** 2, 0.0)) * tau,
        c_pooled_std_cos=np.sqrt(max(m2_pool - mean_pool ** 2, 0.0)) * tau,
    )


#%%
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for dataset_name in DATASETS:
        print(f"\n########## {dataset_name} ##########", flush=True)
        dataset = UserItemTime("./data", dataset_name, "d", 50, MAX_SEQ_LEN)

        pairs = list(dataset.test_user_item_time.items())
        ctx_user = np.array([u for (u, _), _ in pairs], dtype=np.int64)
        ctx_item = np.array([v for (_, v), _ in pairs], dtype=np.int64)
        ctx_time = np.array([t for _, t in pairs], dtype=np.float64)
        np.savez_compressed(f"{OUT_DIR}/contexts_{dataset_name}.npz", user=ctx_user, item=ctx_item, time=ctx_time)
        t0 = ctx_time.min()

        users = np.array([u for u in range(dataset.n_user) if len(dataset.user_interactions[u]) > 0], dtype=np.int64)
        user_pos = {int(u): i for i, u in enumerate(users)}
        n_U = len(users)

        # users' contexts right before the first test time
        init_items, init_times = dataset.get_histories_for_users_at_times(users, [t0] * n_U, MAX_SEQ_LEN, w_time=True)
        n_empty = int((init_items == dataset.m_item).all(1).sum())

        # every interaction from t0 on, in time order: the user's context changes after it
        ev = sorted(((t, int(u)) for u in users for (t, _) in dataset.user_interactions[int(u)] if t >= t0))
        ev_time = np.array([t for t, _ in ev], dtype=np.float64)
        ev_user = np.array([u for _, u in ev], dtype=np.int64)
        ev_items, ev_hist_times = dataset.get_histories_for_users_at_times(
            ev_user, np.nextafter(ev_time, np.inf), MAX_SEQ_LEN, w_time=True)
        prev_ev = np.full(len(ev), -1, dtype=np.int64)       # previous event of the same user
        last = {}
        for e, u in enumerate(ev_user):
            prev_ev[e] = last.get(int(u), -1)
            last[int(u)] = e
        ev_user_pos = np.array([user_pos[int(u)] for u in ev_user], dtype=np.int64)
        k_i = np.searchsorted(ev_time, ctx_time, side="left")  # events with time < t_i
        print(f"[data] {len(ctx_time)} test times, {dataset.m_item} items, |U|={n_U} "
              f"({n_empty} without history at t0), {len(ev)} interactions after t0", flush=True)

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

                R0 = encode(model, model_name, init_items, init_times, users)           # (|U|, d)
                Rev = encode(model, model_name, ev_items, ev_hist_times, ev_user)        # (n_ev, d)
                before = np.where((prev_ev >= 0)[:, None], Rev[np.maximum(prev_ev, 0)], R0[ev_user_pos])
                cum = np.concatenate([np.zeros((1, R0.shape[1])), np.cumsum(Rev - before, axis=0)])
                ubar = (R0.sum(0) + cum[k_i]) / n_U                                      # (n_ctx, d)

                if kind == "vanilla":
                    # exactness check of the incremental scheme against direct encoding of all users
                    for i in rng.choice(len(ctx_time), size=N_CHECK_TIMES, replace=False):
                        h_it, h_tm = dataset.get_histories_for_users_at_times(
                            users, [ctx_time[i]] * n_U, MAX_SEQ_LEN, w_time=True)
                        direct = encode(model, model_name, h_it, h_tm, users).mean(0)
                        assert np.abs(direct - ubar[i]).max() < 1e-6, (dataset_name, model_name, i)

                V = F.normalize(model.item_embedding.weight[: model.num_items], dim=-1, eps=1e-8) \
                    .detach().numpy().astype(np.float32)
                out = f"{OUT_DIR}/{dataset_name}_{model_name}_{kind}.npz"
                np.savez(out, ubar=ubar.astype(np.float32), V=V, tau=tau,
                         gamma=np.nan if gamma is None else gamma, n_users=n_U, n_users_empty_hist=n_empty)

                st = aggregate(ubar, V, tau)
                rows.append(dict(dataset=dataset_name, model_name=model_name, setting=kind,
                                 gamma=np.nan if gamma is None else gamma, tau=tau, **st))
                print(f"[{dataset_name:>11s}|{model_name:>8s}|{kind:>7s}] c mean={st['c_mean']:+.4f} "
                      f"std={st['c_std']:.4f} (time-avg c: std={st['c_pooled_std']:.4f}) "
                      f"cos: {st['c_mean_cos']:+.4f}/{st['c_std_cos']:.4f}", flush=True)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {CSV_PATH}\n[done]")
