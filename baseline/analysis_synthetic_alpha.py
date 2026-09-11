#%%
"""
Semi-synthetic controlled experiment: sweeping the popularity-confounding
strength alpha and checking Theorem 1 / Corollary 1's qualitative prediction
directly, rather than only observing "Ours has lower uniformity than Vanilla"
on real data (as the existing Table 3 / analysis_representation.py do).

Data-generating process. For a fixed ground-truth utility f*(u, v) = <z_u, w_v>
(static per-user/item latents, so the MF backbone's inductive bias matches the
generator exactly) and a time-varying, long-tailed popularity signal pi0(v|t)
(each item has a Gaussian "trend window" superimposed on a Zipf base rate),
events are drawn from

    p_alpha(v | x_u(t)) ~ pi0(v|t)^alpha * exp{f*(x_u(t), v)},

for alpha in {0, 0.5, 1, 2, 4}. alpha=0 removes popularity confounding
entirely; alpha=1 is the paper's own formulation (Sec 3.2, eq. 2); alpha>1
progressively strengthens the popularity-utility confound while the ground
truth preference ranking f* is held fixed. All events across all users are
pooled and split chronologically 80/10/10 (train/valid/test), mirroring the
paper's own dataset protocol (Sec 5.1.1).

For every alpha, two MF backbones are trained on the SAME synthetic
interactions:
  - "Vanilla": plain BPR/logistic loss (module/model.py, no popularity model),
    exactly the training objective in cf.py.
  - "Ours": the paper's Hawkes-popularity + ANOVA-centered utility model
    (module/debias.build_unshared_debias_model + module/hawkes_choice.py),
    exactly the training objective in debiased_cf_hawkes_anova.py.

Two metrics are then computed directly from each trained model:
  1. Representation uniformity L_uni = log E[exp(-tau*||h-h'||^2)] on the
     L2-normalized user embeddings (tau=2.0, matching analysis_representation.py
     and analysis_luni_table.py exactly) -- this is the quantity Theorem 1 /
     Corollary 1 bound.
  2. Ground-truth preference-ranking recovery: Recall@10 between each user's
     TRUE top-10 items under f* alone (excluding train history) and the
     model's own utility-only ranking (score_all_items(), which for "Ours" is
     f_Psi alone -- the Hawkes prior lives in a separate model.prior() branch
     never touched by score_all_items -- i.e. exactly the "gamma=0,
     fully popularity-debiased ranking" described in Sec 4.4). This avoids
     needing to pick an inference-time blending weight gamma at all.

If Theorem 1's mechanism is real, Vanilla's L_uni should approach 0 (representation
collapse) as alpha grows while its ranking-recovery Recall@10 degrades (the
learned embeddings increasingly encode popularity instead of preference),
whereas Ours should stay comparatively stable on both axes because popularity
is routed through the separate Hawkes branch instead of being absorbed into
the (centered) utility embeddings.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_synthetic_alpha.py

Outputs:
    ./figures/synthetic_alpha_sweep.{png,pdf}
    ./analysis_representation/synthetic_alpha_sweep.csv
"""
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY, score_pair, score_all
from module.debias import build_unshared_debias_model
from module.hawkes_choice import importance_corrected_choice_loss, anova_centering_penalty
from module.procedure import computeTopNAccuracy

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
ALPHA_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]
SEEDS = [1, 2, 3]

N_USER = 800
N_ITEM = 200
D_TRUE = 16            # ground-truth latent dimension for f*
T_DAYS = 30.0          # observation horizon
EVENTS_PER_USER = 25
POP_WIDTH = 4.0        # width (days) of each item's popularity "trend window"
POP_ZIPF_EXP = 0.8     # long-tail exponent of the base popularity rate
UTILITY_SCALE = 3.0    # scales f* so its spread is comparable to alpha=1 log-popularity spread

RECDIM = 32
BASELINE_TAU = 0.5     # Vanilla score_norm temperature (matches analysis_representation.py)
HAWKES_TAU = 0.1       # Ours score_norm temperature (matches analysis_representation.py)
SCORE_NORM = "normalized"
LAMBDA_CEN = 1.0
MAX_SEQ_LEN = 10       # MF ignores history; only sizes the dummy placeholder tensor
TIME_LEN = 50          # capped Hawkes history width per item (matches the real-data scripts' default)
USER_BUCKET_DAYS = 1.0

BATCH_SIZE = 512
CONTRAST_SIZE = 8
EPOCHS = 30
LR = 1e-3

TOPK = 10
UNIFORMITY_TAU = 2.0
N_PAIRS_UNIFORMITY = 20000

device = "cpu"


# ----------------------------------------------------------------------
# semi-synthetic generator
# ----------------------------------------------------------------------
def generate_events(alpha, rng):
    z = rng.normal(size=(N_USER, D_TRUE)) / np.sqrt(D_TRUE)
    w = rng.normal(size=(N_ITEM, D_TRUE)) / np.sqrt(D_TRUE)
    true_utility = UTILITY_SCALE * (z @ w.T)   # f*(u, v), time-invariant -- matches MF's inductive bias

    ranks = np.arange(1, N_ITEM + 1)
    zipf_weight = 1.0 / ranks ** POP_ZIPF_EXP
    perm = rng.permutation(N_ITEM)
    base_pop = np.empty(N_ITEM)
    base_pop[perm] = zipf_weight
    log_base_pop = np.log(base_pop)
    trend_center = rng.uniform(0, T_DAYS, size=N_ITEM)

    def log_pi0(t):
        return log_base_pop - (t - trend_center) ** 2 / (2 * POP_WIDTH ** 2)

    events = []
    for u in range(N_USER):
        times = np.sort(rng.uniform(0, T_DAYS, size=EVENTS_PER_USER))
        seen = np.zeros(N_ITEM, dtype=bool)
        for t in times:
            logits = alpha * log_pi0(t) + true_utility[u]
            logits = np.where(seen, -np.inf, logits)
            logits = logits - logits.max()
            p = np.exp(logits)
            p /= p.sum()
            v = int(rng.choice(N_ITEM, p=p))
            seen[v] = True
            events.append((u, v, float(t)))

    events.sort(key=lambda e: e[2])
    return events, true_utility


def split_events(events):
    n = len(events)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    return events[:n_train], events[n_train:n_train + n_valid], events[n_train + n_valid:]


def events_to_item_dict(events):
    d = {}
    for (u, v, _t) in events:
        d.setdefault(u, []).append(v)
    return d


def build_time_dict(events):
    time_dict = {}
    for (u, v, t) in events:
        time_dict.setdefault(u, {})[v] = t
    return time_dict


# ----------------------------------------------------------------------
# in-memory UserItemTime: replicates UserItemTime.__init__ (module/dataset.py)
# without touching disk, reusing its instance methods unchanged so the rest
# of the training/eval pipeline (get_pair_item_uniform, prepare_user_timebucket_
# sampler, build_histories, ...) works identically to the real-data scripts.
# ----------------------------------------------------------------------
class SyntheticUserItemTime(UserItemTime):
    def __init__(self, all_events, train_events, valid_events, test_events, n_user, n_item, time_len, seq_len):
        self.time_unit = "s"   # no unit conversion -- generated times are already plain "day" floats
        self.time_dict = build_time_dict(all_events)
        self.train_dict = events_to_item_dict(train_events)
        self.valid_dict = events_to_item_dict(valid_events)
        self.test_dict = events_to_item_dict(test_events)

        self.trainUniqueUsers, self.trainUser, self.trainItem, self.trainDataSize = self.load_set(self.train_dict)
        self.validUniqueUsers, self.validUser, self.validItem, self.validDataSize = self.load_set(self.valid_dict)
        self.testUniqueUsers, self.testUser, self.testItem, self.testDataSize = self.load_set(self.test_dict)

        self.m_item = n_item
        self.n_user = n_user

        self.UserItemNet = csr_matrix(
            (np.ones(self.trainDataSize), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        self._allPos = self.getUserPosItems(list(range(self.n_user)), self.UserItemNet)

        self.train_user_item_time = self.set_to_pair(self.train_dict, self.time_dict, self.time_unit)
        self.valid_user_item_time = self.set_to_pair(self.valid_dict, self.time_dict, self.time_unit)
        self.test_user_item_time = self.set_to_pair(self.test_dict, self.time_dict, self.time_unit)

        self.item_time_array = self.time_dict_to_array(self.time_dict, time_len)
        if self.item_time_array.shape[0] < self.m_item:
            # items past the max ever-interacted id (e.g. never sampled at high
            # alpha) are dropped by time_dict_to_array's range(); pad them back
            # in as "no history" rows so every downstream index < m_item is valid.
            pad_n = self.m_item - self.item_time_array.shape[0]
            fill = float(self.item_time_array.max()) if self.item_time_array.size else 0.0
            pad = np.full((pad_n, self.item_time_array.shape[1]), fill)
            self.item_time_array = np.concatenate([self.item_time_array, pad], axis=0)

        self.user_interactions = self.build_user_interactions(self.time_dict, self.time_unit)
        self.split_train_hot_n_cold()
        self.train_hist_item_list, self.train_hist_time_list = self.build_histories(self.train_hot_events, seq_len)


# ----------------------------------------------------------------------
# training loops (trimmed copies of cf.py / debiased_cf_hawkes_anova.py,
# driven by the in-memory synthetic dataset instead of file-backed data)
# ----------------------------------------------------------------------
def train_vanilla(dataset, seed):
    set_seed(seed)
    model = MODEL_REGISTRY["mf"](
        num_users=dataset.n_user, num_items=dataset.m_item, embedding_k=RECDIM,
        device=device, tau=BASELINE_TAU, depth=0, max_seq_len=MAX_SEQ_LEN,
        n_heads=1, dropout=0.0, score_norm=SCORE_NORM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mini_batch = BATCH_SIZE // CONTRAST_SIZE
    batch_num = dataset.trainDataSize // mini_batch + 1
    hot_ratio = dataset.hotDataSize / dataset.trainDataSize
    hot_mini_batch = round(mini_batch * hot_ratio)
    cold_mini_batch = mini_batch - hot_mini_batch
    hot_idxs = np.arange(dataset.hotDataSize)
    cold_idxs = np.arange(dataset.coldDataSize)
    dummy_hist = torch.full((mini_batch, MAX_SEQ_LEN), dataset.m_item, dtype=torch.long, device=device)

    dataset.get_pair_item_uniform(k=CONTRAST_SIZE - 1, w_time=False)

    for _epoch in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(hot_idxs)
        for idx in range(batch_num):
            hot_sample_idx = hot_idxs[hot_mini_batch * idx: hot_mini_batch * (idx + 1)]
            cold_sample_idx = cold_idxs[cold_mini_batch * idx: cold_mini_batch * (idx + 1)]

            anchor_user = torch.tensor(
                np.concatenate([dataset.cold_user_list[cold_sample_idx], dataset.hot_user_list[hot_sample_idx]]),
                dtype=torch.long, device=device)
            pos_item = torch.tensor(
                np.concatenate([dataset.cold_pos_item_list[cold_sample_idx], dataset.hot_pos_item_list[hot_sample_idx]]),
                dtype=torch.long, device=device)
            neg_item = torch.tensor(
                np.concatenate([dataset.cold_neg_item_list[cold_sample_idx], dataset.hot_neg_item_list[hot_sample_idx]]),
                dtype=torch.long, device=device)
            if anchor_user.shape[0] == 0:
                continue
            batch_hist = dummy_hist[:anchor_user.shape[0]]

            pos_score = score_pair(model, pos_item, batch_hist, anchor_user)
            neg_score = score_pair(model, neg_item, batch_hist, anchor_user)
            loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dataset.get_pair_item_uniform(k=CONTRAST_SIZE - 1, w_time=False)

    return model


def train_ours(dataset, seed, lambda_cen=LAMBDA_CEN):
    set_seed(seed)
    debiased_class = build_unshared_debias_model(MODEL_REGISTRY["mf"])
    model = debiased_class(
        num_users=dataset.n_user, num_items=dataset.m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=0, max_seq_len=MAX_SEQ_LEN,
        n_heads=1, dropout=0.0, score_norm=SCORE_NORM,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    mini_batch = BATCH_SIZE // CONTRAST_SIZE
    batch_num = dataset.trainDataSize // mini_batch + 1
    hot_ratio = dataset.hotDataSize / dataset.trainDataSize
    hot_mini_batch = round(mini_batch * hot_ratio)
    cold_mini_batch = mini_batch - hot_mini_batch
    hot_idxs = np.arange(dataset.hotDataSize)
    cold_idxs = np.arange(dataset.coldDataSize)
    num_negatives = CONTRAST_SIZE - 1
    dummy_hist = torch.full((mini_batch, MAX_SEQ_LEN), dataset.m_item, dtype=torch.long, device=device)

    dataset.get_pair_item_uniform(k=num_negatives, w_time=True)
    dataset.prepare_user_timebucket_sampler(bucket_size=USER_BUCKET_DAYS, w_cold=True)
    dataset.get_pair_user_event_timebucket_fast(w_cold=True)

    for _epoch in range(1, EPOCHS + 1):
        model.train()
        np.random.shuffle(hot_idxs)
        for idx in range(batch_num):
            hot_sample_idx = hot_idxs[hot_mini_batch * idx: hot_mini_batch * (idx + 1)]
            cold_sample_idx = cold_idxs[cold_mini_batch * idx: cold_mini_batch * (idx + 1)]

            anchor_user = torch.tensor(
                np.concatenate([dataset.cold_user_list[cold_sample_idx], dataset.hot_user_list[hot_sample_idx]]),
                dtype=torch.long, device=device)
            if anchor_user.shape[0] == 0:
                continue
            pos_item = torch.tensor(
                np.concatenate([dataset.cold_pos_item_list[cold_sample_idx], dataset.hot_pos_item_list[hot_sample_idx]]),
                dtype=torch.long, device=device)
            neg_item = torch.tensor(
                np.concatenate([dataset.cold_neg_item_list[cold_sample_idx], dataset.hot_neg_item_list[hot_sample_idx]]),
                dtype=torch.long, device=device)
            pos_time = torch.tensor(
                np.concatenate([dataset.cold_event_time_list[cold_sample_idx], dataset.hot_event_time_list[hot_sample_idx]]),
                dtype=torch.float32, device=device)
            pos_time_all = torch.tensor(
                np.concatenate([dataset.cold_pos_time_all[cold_sample_idx], dataset.hot_pos_time_all[hot_sample_idx]]),
                dtype=torch.float32, device=device)
            neg_time_all = torch.tensor(
                np.concatenate([dataset.cold_neg_time_all[cold_sample_idx], dataset.hot_neg_time_all[hot_sample_idx]]),
                dtype=torch.float32, device=device)

            candidate_items = torch.cat([pos_item.unsqueeze(-1), neg_item], dim=-1)
            candidate_time_all = torch.cat([pos_time_all.unsqueeze(1), neg_time_all], dim=1)
            batch_hist = dummy_hist[:anchor_user.shape[0]]

            choice_loss = importance_corrected_choice_loss(
                model, candidate_items, batch_hist, anchor_user, pos_time, candidate_time_all,
                dataset.m_item, num_negatives,
            )

            dataset.get_pair_user_event_timebucket_fast(w_cold=True)
            center_user_a = torch.tensor(
                np.concatenate([dataset.cold_neg_user_list[cold_sample_idx, 0], dataset.hot_neg_user_list[hot_sample_idx, 0]]),
                dtype=torch.long, device=device)
            dataset.get_pair_user_event_timebucket_fast(w_cold=True)
            center_user_b = torch.tensor(
                np.concatenate([dataset.cold_neg_user_list[cold_sample_idx, 0], dataset.hot_neg_user_list[hot_sample_idx, 0]]),
                dtype=torch.long, device=device)

            center_loss = anova_centering_penalty(
                model, candidate_items, batch_hist, center_user_a, batch_hist, center_user_b,
            )

            total_loss = choice_loss + lambda_cen * center_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            dataset.get_pair_item_uniform(k=num_negatives, w_time=True)

    return model


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def compute_uniformity(model, rng):
    h = model.user_embedding.weight.detach().cpu().numpy()
    h_norm = h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-8, None)
    n = h_norm.shape[0]
    idx = rng.integers(0, n, size=(N_PAIRS_UNIFORMITY, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    d2 = np.sum((h_norm[idx[:, 0]] - h_norm[idx[:, 1]]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-UNIFORMITY_TAU * d2))))


@torch.no_grad()
def compute_ranking_recovery(model, dataset, true_utility):
    model.eval()
    user_idx_all = np.arange(dataset.n_user)
    dummy_hist = torch.full((len(user_idx_all), MAX_SEQ_LEN), dataset.m_item, dtype=torch.long, device=device)
    user_t = torch.tensor(user_idx_all, dtype=torch.long, device=device)
    pred_scores = score_all(model, dummy_hist, user_t).cpu().numpy()   # utility-only ranking (gamma=0 for Ours)

    gt_list, pred_list = [], []
    for u in user_idx_all:
        exclude = list(dataset._allPos[u])
        if len(exclude) >= N_ITEM - TOPK:
            continue

        true_scores = true_utility[u].copy()
        true_scores[exclude] = -np.inf
        true_topk = list(np.argsort(-true_scores)[:TOPK])

        pred_scores_u = pred_scores[u].copy()
        pred_scores_u[exclude] = -1e9
        pred_topk = list(np.argsort(-pred_scores_u)[:TOPK])

        gt_list.append(true_topk)
        pred_list.append(pred_topk)

    _, recall, ndcg, _ = computeTopNAccuracy(gt_list, pred_list, [TOPK])
    return recall[0], ndcg[0]


# ----------------------------------------------------------------------
# main sweep
# ----------------------------------------------------------------------
def main():
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./analysis_representation", exist_ok=True)

    results = []
    for alpha in ALPHA_GRID:
        for seed in SEEDS:
            print(f"\n=== alpha={alpha} seed={seed} ===")
            data_rng = np.random.default_rng(hash((seed, alpha)) % (2 ** 32))
            events, true_utility = generate_events(alpha, data_rng)
            train_events, valid_events, test_events = split_events(events)
            dataset = SyntheticUserItemTime(
                events, train_events, valid_events, test_events, N_USER, N_ITEM, TIME_LEN, MAX_SEQ_LEN)
            print(f"[data] {len(events)} events | train={len(train_events)} "
                  f"valid={len(valid_events)} test={len(test_events)}")

            for approach, train_fn in [("vanilla", train_vanilla), ("ours", train_ours)]:
                model = train_fn(dataset, seed)
                metric_rng = np.random.default_rng(hash((seed, alpha, approach)) % (2 ** 32))
                luni_val = compute_uniformity(model, metric_rng)
                recall10, ndcg10 = compute_ranking_recovery(model, dataset, true_utility)
                print(f"  [{approach:>7s}] L_uni={luni_val:.4f}  "
                      f"Recall@10(gt-recovery)={recall10:.4f}  NDCG@10={ndcg10:.4f}")
                results.append(dict(alpha=alpha, seed=seed, approach=approach,
                                     l_uni=luni_val, recall_10=recall10, ndcg_10=ndcg10))

    csv_path = "./analysis_representation/synthetic_alpha_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "seed", "approach", "l_uni", "recall_10", "ndcg_10"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {csv_path}")

    def agg(approach, key):
        means, stds = [], []
        for alpha in ALPHA_GRID:
            vals = [r[key] for r in results if r["approach"] == approach and r["alpha"] == alpha]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    colors = {"vanilla": "#4C72B0", "ours": "#DD8452"}
    labels = {"vanilla": "Vanilla", "ours": "Ours"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for approach in ["vanilla", "ours"]:
        m, s = agg(approach, "l_uni")
        axes[0].errorbar(ALPHA_GRID, m, yerr=s, marker="o", capsize=3,
                          label=labels[approach], color=colors[approach])
    axes[0].set_xlabel(r"popularity strength $\alpha$")
    axes[0].set_ylabel(r"uniformity loss $\mathcal{L}_{\mathrm{uni}}$")
    axes[0].set_title("Representation uniformity")
    axes[0].legend()

    for approach in ["vanilla", "ours"]:
        m, s = agg(approach, "recall_10")
        axes[1].errorbar(ALPHA_GRID, m, yerr=s, marker="o", capsize=3,
                          label=labels[approach], color=colors[approach])
    axes[1].set_xlabel(r"popularity strength $\alpha$")
    axes[1].set_ylabel("Recall@10 (ground-truth ranking recovery)")
    axes[1].set_title("Preference-ranking recovery")
    axes[1].legend()

    fig.suptitle(r"Semi-synthetic sweep of popularity-confounding strength $\alpha$"
                 "\n" r"($\alpha{=}0$: no popularity confound, $\alpha{=}1$: paper's own formulation)", y=1.08)
    fig.tight_layout()
    out_path = "./figures/synthetic_alpha_sweep.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
