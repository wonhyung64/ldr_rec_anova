#%%
"""
Direct test of Proposition 1 (Identifiability under Centering): does the model
actually recover the decomposition pi_hat(.|t) ~= pi0(.|t) and f_hat in [f*],
rather than merely fitting the observed choice distribution p0(v|x_u(t)) well?

This is a different question from the rho-sweep in analysis_synthetic_rho.py
(Figure 1), which only checks that ranking is recovered downstream. Here we
check the decomposition itself, so the synthetic DGP is set up to match
Proposition 1's own hypotheses exactly:

  - rho = 1 (not swept): events are drawn from
        p0(v | x_u(t)) ~ pi0(v|t) * exp{f*(u, v)},
    i.e. exactly Eq. 2 / the model Proposition 1 is stated for, unlike Figure 1
    which sweeps rho to control confounding strength.
  - Assumption 2 (centered utility) is satisfied by construction: f*(u, v) =
    s * <z_u, w_v> with z_u re-centered so that sum_u f*(u, v) = 0 for every v
    (exact because f* is time-invariant here, so "for every v, t" reduces to
    "for every v"), and the scale s calibrated per seed (see generate_events)
    so f*'s marginal spread matches log pi0(t)'s -- otherwise recovering
    utility from the observed choices is dominated by a trivial SNR
    imbalance, unrelated to the identifiability question this script tests
    (verified empirically: with a naive fixed scale, even the pi0-oracle
    variant below could not recover utility above noise, since popularity
    swamped it in the generative process by ~6.5x).

Three variants are compared, all using the SAME MF backbone and SAME
importance-corrected choice loss (module/hawkes_choice.py):

  - "no_centering": Ours' architecture (Hawkes pi_Phi + MF f_Psi) trained with
    lambda_cen = 0, i.e. the non-identifiability of Eq. 3 is present.
  - "ours": the same architecture with the centering penalty active
    (lambda_cen = LAMBDA_CEN), i.e. Proposition 1's identification constraint.
  - "oracle_pi0": pi_0(v|t) is FIXED to its true (ground-truth) value -- no
    Hawkes parameters are learned at all -- and only f_Psi is trained, with
    lambda_cen = 0. Fixing pi0 already removes the popularity/utility
    reallocation freedom Eq. 3 describes (pi0 is not a trainable target, so no
    shared component can be routed through it), so the centering penalty is
    not needed for identification here; the only ambiguity left is the
    additive per-context constant C(x_u(t)) Definition 1 already tolerates,
    which the evaluation metric below removes by construction. This isolates
    whether imperfect recovery of "ours" traces back to the decomposition
    formulation itself or to Hawkes popularity-estimation error, since here
    that error is zero by construction.

Metrics computed directly from each trained model (NOT from downstream
ranking):

  1. Popularity recovery: mean KL(pi0(.|t) || pi_hat(.|t)) over a grid of
     evaluation timestamps T_eval, i.e. exactly the D_KL in Proposition 1's
     pi_tilde(.|t) = pi0(.|t) claim. For "oracle_pi0" this is ~0 by
     construction (sanity check). NOTE: pi_Phi is a Hawkes intensity, a
     strongly restricted parametric family, not an arbitrary item-time
     function -- so a priori there is no reason lambda_cen (which acts on
     f_Psi) should move this metric, and empirically it does not (see
     run_significance_tests output); it is reported as a check on the
     Hawkes fit itself; the identifiability claim is about the DECOMPOSITION
     given both components are estimated, not about lambda_cen improving
     pi_Phi's accuracy specifically.
  2. Utility recovery ("Ranking-equivalent Utility MSE", primary): this is
     the metric that actually corresponds to Proposition 1's f_hat in [f*]
     (Definition 1). For each user u, remove the ranking-preserving additive
     constant by centering ACROSS ITEMS (not across users):
         f_c(u, v)  = f_hat(u, v)  - mean_v' f_hat(u, v')
         f*_c(u, v) = f*(u, v)     - mean_v' f*(u, v')      [already 0 here]
     and report MSE(f_c, f*_c) pooled over all (u, v), with NO additional
     scale correction -- Definition 1 only tolerates an additive shift, not a
     multiplicative one, so a scale mismatch is a genuine recovery failure,
     not a nuisance to regress away. See compute_utility_recovery().
  3. Diagnostics (not headline results): (a) held-out choice NLL under the
     full model p_{Phi,Psi}(v|x_u(t)), to check whether variants that recover
     the decomposition differently nonetheless fit the observed choices
     similarly (the strongest version of the identifiability story is
     comparable NLL + divergent utility recovery); (b) correlation between
     f_c and f*_c, a scale-invariant companion to the primary MSE;
     (c) "centering residual" mean_v[(1/|U|) sum_u f_hat(u,v)]^2 -- this is
     EXACTLY the paper's own R_cen (Sec. 4.3) evaluated on f_hat, so a priori
     the model trained to directly minimize it ("ours") is expected to score
     low; it is logged as a mechanistic check that the penalty is doing what
     it is supposed to, NOT as evidence of utility recovery.

30 seeds (matching analysis_synthetic_rho.py) x 3 variants = 90 independent
runs, parallelized across processes (N_WORKERS).

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_synthetic_decomposition.py

Outputs:
    ./figures/synthetic_decomposition_recovery.{png,pdf}
    ./analysis_representation/synthetic_decomposition_recovery.csv
    ./analysis_representation/synthetic_decomposition_significance.txt
"""
import os
import csv
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix

from module.utils import set_seed
from module.model import MODEL_REGISTRY, score_pair, score_all
from module.debias import build_unshared_debias_model
from module.hawkes_choice import importance_corrected_choice_loss, anova_centering_penalty
from analysis_synthetic_rho import SyntheticUserItemTime, split_events

torch.set_num_threads(1)   # each worker process handles one run; avoid oversubscribing cores

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
RHO = 1.0                    # Proposition 1's own model (Eq. 2), not swept
VARIANTS = ["no_centering", "ours", "oracle_pi0"]
SEEDS = list(range(1, 31))   # 30 seeds, matching analysis_synthetic_rho.py
N_WORKERS = min(8, os.cpu_count() or 1)

N_USER = 800
N_ITEM = 200
D_TRUE = 16
T_DAYS = 30.0
EVENTS_PER_USER = 25
POP_WIDTH = 4.0         # same as analysis_synthetic_rho.py's default (unchanged, so the
                        # popularity generator here is structurally identical to Figure 1's)
POP_ZIPF_EXP = 0.8
N_SCALE_CALIB_SAMPLES = 2000   # for calibrating f*'s scale against log pi0(t)'s spread

RECDIM = 32
HAWKES_TAU = 0.1
SCORE_NORM = "normalized"
LAMBDA_CEN = 1.0
MAX_SEQ_LEN = 10
TIME_LEN = 50
USER_BUCKET_DAYS = 1.0

BATCH_SIZE = 512
CONTRAST_SIZE = 8
EPOCHS = 80            # more than analysis_synthetic_rho.py's 30: recovering the utility
                        # embeddings themselves (not just downstream ranking) needs more
                        # optimization steps to converge (verified empirically).
LR = 3e-3

N_EVAL_TIMES = 20
EVAL_MARGIN = 2.0   # days kept away from [0, T_DAYS] boundary

device = "cpu"


#%%
# semi-synthetic generator: rho=1, and Assumption 2 (centered utility) exact
def _log_pi0_cross_item_std(log_base_pop, trend_center, rng, n_samples=N_SCALE_CALIB_SAMPLES):
    """Average (over random t) cross-item std of log pi0(.|t), used to scale-
    calibrate f* below so neither component trivially dominates the other."""
    ts = rng.uniform(0, T_DAYS, size=n_samples)
    stds = [
        (log_base_pop - (t - trend_center) ** 2 / (2 * POP_WIDTH ** 2)).std()
        for t in ts
    ]
    return float(np.mean(stds))


def generate_events(rng):
    z = rng.normal(size=(N_USER, D_TRUE)) / np.sqrt(D_TRUE)
    z = z - z.mean(axis=0, keepdims=True)   # center so sum_u f*(u,v)=0 for every v (Assumption 2)
    w = rng.normal(size=(N_ITEM, D_TRUE)) / np.sqrt(D_TRUE)
    f_raw = z @ w.T   # f*(u,v) up to scale, time-invariant, exactly centered across users

    ranks = np.arange(1, N_ITEM + 1)
    zipf_weight = 1.0 / ranks ** POP_ZIPF_EXP
    perm = rng.permutation(N_ITEM)
    base_pop = np.empty(N_ITEM)
    base_pop[perm] = zipf_weight
    log_base_pop = np.log(base_pop)
    trend_center = rng.uniform(0, T_DAYS, size=N_ITEM)

    # calibrate f*'s scale to match log pi0(t)'s cross-item spread, so recovery is not
    # dominated by a trivial signal-to-noise imbalance between the two components
    target_std = _log_pi0_cross_item_std(log_base_pop, trend_center, rng)
    utility_scale = target_std / (f_raw.std() + 1e-8)
    true_utility = utility_scale * f_raw

    def log_pi0(t):
        return log_base_pop - (t - trend_center) ** 2 / (2 * POP_WIDTH ** 2)

    events = []
    for u in range(N_USER):
        times = np.sort(rng.uniform(0, T_DAYS, size=EVENTS_PER_USER))
        seen = np.zeros(N_ITEM, dtype=bool)
        for t in times:
            logits = RHO * log_pi0(t) + true_utility[u]
            logits = np.where(seen, -np.inf, logits)
            logits = logits - logits.max()
            p = np.exp(logits)
            p /= p.sum()
            v = int(rng.choice(N_ITEM, p=p))
            seen[v] = True
            events.append((u, v, float(t)))

    events.sort(key=lambda e: e[2])
    return events, true_utility, log_base_pop, trend_center


# ----------------------------------------------------------------------
# Oracle popularity model: pi0(v|t) fixed to its ground-truth value, only
# f_Psi (plain MF utility) is learned. `prior()` matches HawkesDebias's
# signature so the SAME importance_corrected_choice_loss applies unchanged;
# batch_time_all (Hawkes history) is unused.
# ----------------------------------------------------------------------
def build_oracle_model(model_class):
    class OraclePopularity(model_class):
        def set_popularity_truth(self, log_base_pop, trend_center, pop_width):
            self.register_buffer("log_base_pop_t", torch.tensor(log_base_pop, dtype=torch.float32))
            self.register_buffer("trend_center_t", torch.tensor(trend_center, dtype=torch.float32))
            self.pop_width = pop_width

        def prior(self, batch_items, pos_time, batch_time_all):
            t = pos_time.view(-1, 1)
            log_pi0 = (self.log_base_pop_t[batch_items]
                       - (t - self.trend_center_t[batch_items]) ** 2 / (2 * self.pop_width ** 2))
            return torch.exp(log_pi0)

        def prior_parameters_from_embeddings(self):
            raise NotImplementedError("Oracle popularity has no learned Hawkes parameters.")

    return OraclePopularity


# ----------------------------------------------------------------------
# training loop: shared by "no_centering" / "ours" (Hawkes pi_Phi + MF f_Psi)
# and "oracle_pi0" (fixed true pi0 + MF f_Psi), differing only in model_class
# and whether prior_parameters_from_embeddings() supplies mu/alpha/beta.
# lambda_cen=0 for both "no_centering" and "oracle_pi0" (see module docstring
# for why the oracle needs no centering penalty); lambda_cen=LAMBDA_CEN only
# for "ours".
# ----------------------------------------------------------------------
def train_variant(dataset, seed, variant, log_base_pop, trend_center, lambda_cen):
    set_seed(seed)
    if variant == "oracle_pi0":
        model_class = build_oracle_model(MODEL_REGISTRY["mf"])
    else:
        model_class = build_unshared_debias_model(MODEL_REGISTRY["mf"])
    model = model_class(
        num_users=dataset.n_user, num_items=dataset.m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=0, max_seq_len=MAX_SEQ_LEN,
        n_heads=1, dropout=0.0, score_norm=SCORE_NORM,
    ).to(device)
    if variant == "oracle_pi0":
        model.set_popularity_truth(log_base_pop, trend_center, POP_WIDTH)

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

            if lambda_cen > 0:
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
            else:
                total_loss = choice_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            dataset.get_pair_item_uniform(k=num_negatives, w_time=True)

    return model


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
@torch.no_grad()
def compute_popularity_kl(model, dataset, eval_times, log_base_pop, trend_center, variant):
    item_time_array = torch.tensor(dataset.item_time_array, dtype=torch.float32)
    log_base_pop_t = torch.tensor(log_base_pop, dtype=torch.float32)
    trend_center_t = torch.tensor(trend_center, dtype=torch.float32)

    if variant != "oracle_pi0":
        mu, alpha, beta = model.prior_parameters_from_embeddings()

    kls = []
    for t in eval_times:
        log_pi0_true = log_base_pop_t - (t - trend_center_t) ** 2 / (2 * POP_WIDTH ** 2)
        log_pi0_true = log_pi0_true - torch.logsumexp(log_pi0_true, dim=0)

        if variant == "oracle_pi0":
            log_pi_hat = log_pi0_true.clone()
        else:
            delta = (t - item_time_array).clamp(min=0.0)
            mask = (item_time_array < t).float()
            h = (torch.exp(-beta * delta) * mask).sum(-1)
            lam = mu + alpha * h
            log_lam = torch.log(lam + 1e-12)
            log_pi_hat = log_lam - torch.logsumexp(log_lam, dim=0)

        pi0_true = torch.exp(log_pi0_true)
        kl = torch.sum(pi0_true * (log_pi0_true - log_pi_hat)).item()
        kls.append(kl)
    return float(np.mean(kls))


@torch.no_grad()
def get_f_hat(model, dataset):
    model.eval()
    user_idx_all = np.arange(dataset.n_user)
    dummy_hist = torch.full((len(user_idx_all), MAX_SEQ_LEN), dataset.m_item, dtype=torch.long, device=device)
    user_t = torch.tensor(user_idx_all, dtype=torch.long, device=device)
    return score_all(model, dummy_hist, user_t).cpu().numpy()


def compute_utility_recovery(f_hat, true_utility):
    """Primary utility-recovery metric: ranking-equivalence-adjusted MSE.

    Per Definition 1, f_hat in [f*] iff f_hat(x,v) = f*(x,v) + C(x) for some
    C constant across items. We therefore remove, for each user u, the
    item-mean (the least-squares-optimal estimate of that additive constant)
    from BOTH f_hat and f* before comparing -- NOT a per-user row-mean of
    f_hat alone or any additional scale correction, since Definition 1 only
    licenses an additive shift, not a multiplicative one:
        f_c(u,v)  = f_hat(u,v) - mean_v' f_hat(u,v')
        f*_c(u,v) = f*(u,v)    - mean_v' f*(u,v')
    Equivalently, E_f = mean_u min_C mean_v [f_hat(u,v)-f*(u,v)-C]^2.
    Returns (mse, corr); corr is a scale-invariant companion diagnostic.
    """
    f_hat_c = f_hat - f_hat.mean(axis=1, keepdims=True)
    f_true_c = true_utility - true_utility.mean(axis=1, keepdims=True)
    mse = float(np.mean((f_hat_c - f_true_c) ** 2))
    corr = float(np.corrcoef(f_hat_c.flatten(), f_true_c.flatten())[0, 1])
    return mse, corr


def compute_centering_residual(f_hat):
    """Diagnostic only (NOT a recovery metric): mean_v[(1/|U|) sum_u f_hat(u,v)]^2,
    i.e. the paper's own R_cen (Sec. 4.3) evaluated on f_hat post-hoc. "ours"
    directly minimizes this during training, so a low value here confirms the
    penalty is doing its job mechanistically -- it is not independent
    evidence of recovering f*, which is what compute_utility_recovery checks.
    """
    pop_mean = f_hat.mean(axis=0)
    return float(np.mean(pop_mean ** 2))


@torch.no_grad()
def compute_held_out_nll(model, dataset, variant, f_hat, chunk_size=400):
    """Mean negative log-likelihood of held-out (test-split) interactions
    under the full model p_{Phi,Psi}(v|x_u(t)) (Eq. 4), normalizing exactly
    over the full item catalog (feasible here since N_ITEM=200). Diagnostic:
    if two variants fit held-out choices comparably while differing sharply
    in utility recovery, that is the strongest form of the identifiability
    argument (same observed distribution, different decomposition).
    """
    test_items = list(dataset.test_user_item_time.items())
    if not test_items:
        return float("nan")
    users = np.array([uv[0] for uv, _t in test_items])
    items = np.array([uv[1] for uv, _t in test_items])
    times = np.array([t for _uv, t in test_items], dtype=np.float32)

    if variant != "oracle_pi0":
        mu, alpha, beta = model.prior_parameters_from_embeddings()
        item_time_array = torch.tensor(dataset.item_time_array, dtype=torch.float32)

    neg_log_p = []
    for start in range(0, len(test_items), chunk_size):
        sl = slice(start, start + chunk_size)
        util_chunk = torch.tensor(f_hat[users[sl]], dtype=torch.float32)   # (b, n_item)
        t_chunk = torch.tensor(times[sl], dtype=torch.float32)

        if variant == "oracle_pi0":
            t_col = t_chunk.view(-1, 1)
            log_lam = (model.log_base_pop_t.view(1, -1)
                       - (t_col - model.trend_center_t.view(1, -1)) ** 2 / (2 * model.pop_width ** 2))
        else:
            t_3d = t_chunk.view(-1, 1, 1)
            delta = (t_3d - item_time_array.unsqueeze(0)).clamp(min=0.0)
            mask = (item_time_array.unsqueeze(0) < t_3d).float()
            h = (torch.exp(-beta * delta) * mask).sum(-1)
            lam = mu.view(1, -1) + alpha.view(1, -1) * h
            log_lam = torch.log(lam + 1e-12)

        log_unnorm = log_lam + util_chunk
        log_Z = torch.logsumexp(log_unnorm, dim=1)
        item_idx = torch.tensor(items[sl], dtype=torch.long)
        log_p_true = log_unnorm[torch.arange(len(item_idx)), item_idx] - log_Z
        neg_log_p.append((-log_p_true).numpy())

    return float(np.mean(np.concatenate(neg_log_p)))


# ----------------------------------------------------------------------
# one independent (seed, variant) run -- top-level so it is picklable for
# multiprocessing
# ----------------------------------------------------------------------
def run_single(task):
    seed, variant = task
    torch.set_num_threads(1)

    data_rng = np.random.default_rng(hash((seed, "decomposition")) % (2 ** 32))
    events, true_utility, log_base_pop, trend_center = generate_events(data_rng)
    train_events, valid_events, test_events = split_events(events)
    dataset = SyntheticUserItemTime(
        events, train_events, valid_events, test_events, N_USER, N_ITEM, TIME_LEN, MAX_SEQ_LEN)

    lambda_cen = LAMBDA_CEN if variant == "ours" else 0.0
    model = train_variant(dataset, seed, variant, log_base_pop, trend_center, lambda_cen)

    eval_times = np.linspace(EVAL_MARGIN, T_DAYS - EVAL_MARGIN, N_EVAL_TIMES)
    kl_pop = compute_popularity_kl(model, dataset, eval_times, log_base_pop, trend_center, variant)

    f_hat = get_f_hat(model, dataset)
    mse_util, corr_util = compute_utility_recovery(f_hat, true_utility)
    centering_residual = compute_centering_residual(f_hat)
    held_out_nll = compute_held_out_nll(model, dataset, variant, f_hat)

    return dict(seed=seed, variant=variant, kl_popularity=kl_pop, mse_utility=mse_util,
                corr_utility=corr_util, centering_residual=centering_residual,
                held_out_nll=held_out_nll)


# ----------------------------------------------------------------------
# statistical significance: paired comparisons across the three variants,
# paired by seed (all three variants are trained on the IDENTICAL synthetic
# dataset for a given seed).
# ----------------------------------------------------------------------
def run_significance_tests(results):
    by_key = {(r["seed"], r["variant"]): r for r in results}
    pairs = [("no_centering", "ours"), ("ours", "oracle_pi0"), ("no_centering", "oracle_pi0")]

    lines = [f"Paired comparisons across variants, paired by seed (n={len(SEEDS)} seeds)"]
    for metric, label in [("kl_popularity", "Popularity KL"),
                           ("mse_utility", "Ranking-equivalent Utility MSE (primary)"),
                           ("corr_utility", "Utility correlation f_c vs f*_c (diagnostic)"),
                           ("held_out_nll", "Held-out Choice NLL (diagnostic)"),
                           ("centering_residual", "Centering residual = R_cen(f_hat) (diagnostic; "
                                                   "'ours' directly minimizes this during training)")]:
        lines.append(f"\n--- {label} ---")
        for va, vb in pairs:
            a = np.array([by_key[(s, va)][metric] for s in SEEDS])
            b = np.array([by_key[(s, vb)][metric] for s in SEEDS])
            diff = a - b
            t_stat, t_p = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            lines.append(
                f"  {va:>12s} vs {vb:<12s}: {va}={a.mean():.4e}(sd={a.std():.4e})  "
                f"{vb}={b.mean():.4e}(sd={b.std():.4e})  diff={diff.mean():+.4e}  "
                f"paired-t p={t_p:.2e}  wilcoxon p={w_p:.2e}")

    report = "\n".join(lines)
    print("\n########## statistical significance ##########")
    print(report)

    txt_path = "./analysis_representation/synthetic_decomposition_significance.txt"
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print(f"\n[saved] {txt_path}")


# ----------------------------------------------------------------------
# main sweep
# ----------------------------------------------------------------------
def main():
    os.makedirs("./figures", exist_ok=True)
    os.makedirs("./analysis_representation", exist_ok=True)

    tasks = [(seed, variant) for seed in SEEDS for variant in VARIANTS]
    print(f"[sweep] {len(tasks)} runs ({len(SEEDS)} seeds x {len(VARIANTS)} variants), "
          f"{N_WORKERS} parallel workers")

    results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single, tasks), 1):
            results.append(res)
            if i % 10 == 0 or i == len(tasks):
                print(f"[progress] {i}/{len(tasks)} runs done -- last: "
                      f"seed={res['seed']} {res['variant']:>12s}: "
                      f"KL={res['kl_popularity']:.4f} MSE={res['mse_utility']:.4f} "
                      f"NLL={res['held_out_nll']:.4f}")

    results.sort(key=lambda r: (r["seed"], VARIANTS.index(r["variant"])))

    csv_path = "./analysis_representation/synthetic_decomposition_recovery.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "variant", "kl_popularity", "mse_utility",
                                                "corr_utility", "centering_residual", "held_out_nll"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[saved] {csv_path}")

    run_significance_tests(results)

    def agg(variant, key):
        vals = [r[key] for r in results if r["variant"] == variant]
        return float(np.mean(vals)), float(np.std(vals))

    labels = {"no_centering": "Ours w/o\ncentering", "ours": "Ours", "oracle_pi0": r"Oracle" "\n" r"$\pi_0$"}
    colors = {"no_centering": "#C44E52", "ours": "#DD8452", "oracle_pi0": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    for ax, key, title, ylabel in [
        (axes[0], "kl_popularity", "Popularity Recovery", r"$D_{\mathrm{KL}}(\pi_0 \,\|\, \hat\pi_\Phi)\downarrow$"),
        (axes[1], "mse_utility", "Utility Recovery", r"Ranking-equivalent Utility MSE $E_f\downarrow$"),
    ]:
        means, stds = [], []
        for variant in VARIANTS:
            m, s = agg(variant, key)
            means.append(m)
            stds.append(s / np.sqrt(len(SEEDS)))
        bar_colors = [colors[v] for v in VARIANTS]
        ax.bar(range(len(VARIANTS)), means, yerr=stds, capsize=4, color=bar_colors,
               tick_label=[labels[v] for v in VARIANTS])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle("Recovery of Popularity and Utility Components (Proposition 1), "
                  r"$\rho{=}1$" f", MF backbone, error bars = SEM over {len(SEEDS)} seeds", y=1.04)
    fig.tight_layout()
    out_path = "./figures/synthetic_decomposition_recovery.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
