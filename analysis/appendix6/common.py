"""Shared utilities for Appendix A.6 ("Analysis of Tail-Item Degradation on KuaiRand")
follow-up analyses.

Reuses the repository's existing dataset/model/checkpoint code wherever possible:
  - baseline/module/dataset.py:UserItemTime            (chronological split, time_dict, etc.)
  - baseline/module/model.py:MODEL_REGISTRY             (MF/GRU/SASRec/TiSASRec/FEARec/BSARec)
  - baseline/module/debias.py:build_unshared_debias_model (the "Ours" architecture,
    ablation="shared" CLI value -> separate p_item_embedding Hawkes head; see
    baseline/debiased_seq_rec_hawkes_anova.py lines 62-63)
  - baseline/analysis_kuairand_tail_diagnosis.py         (checkpoint loading, eval-batch
    construction, additional_feat dispatch for TiSASRec's history-timestamp signature,
    ranking utilities) -- generalized here to all three datasets rather than KuaiRand only.

No training code is modified. No model is retrained. Only already-saved checkpoints in
weights/<dataset> (backbone) and weights_hawkes_anova_generalize/<dataset> (Ours) are
loaded.

Head/Tail terminology note (see report, Sec. "Existing code inspection"): the repository's
own Head/Tail split (data/<dataset>/testing_dict_tail_recent_3d.npy etc., used for
Table 1/4) is a *static*, one-time classification -- empirically verified against
item_popularity_recent_3d.npy to be based on a fixed window near the *end* of the
dataset's time range, not a per-query-time snapshot. The "local empirical popularity"
analysis introduced here (Sec. 3 of the task spec) is a *different*, complementary,
query-time-varying quantity p_emp(v|t) computed fresh at many snapshot times; it is not a
re-derivation of the paper's Head/Tail labels. Where an analysis needs the paper's actual
Tail *evaluation set* (e.g. rank-displacement, eta-sweep), we reuse
`ds.test_tail_recent_3d_dict` directly, exactly as Table 1/4 do.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

BASELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "baseline")
REPO_ROOT = os.path.dirname(BASELINE_DIR)
sys.path.insert(0, BASELINE_DIR)

from module.dataset import UserItemTime          # noqa: E402
from module.model import MODEL_REGISTRY          # noqa: E402
from module.debias import build_unshared_debias_model  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "appendix6")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures", "appendix6")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ["micro_video", "ml-1m", "kuairand"]
DATASET_LABELS = {"micro_video": "Micro-Video", "ml-1m": "MovieLens", "kuairand": "KuaiRand"}
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
SEEDS = [1, 2, 3, 4]

# Colors/markers held fixed across every figure in this appendix, per the plotting spec.
DATASET_COLOR = {"kuairand": "#d62728", "micro_video": "#1f77b4", "ml-1m": "#2ca02c"}
DATASET_MARKER = {"kuairand": "o", "micro_video": "s", "ml-1m": "^"}
BACKBONE_COLOR = {
    "mf": "#7f7f7f", "grurec": "#bcbd22", "sasrec": "#d62728",
    "tisasrec": "#9467bd", "fearec": "#8c564b", "bsarec": "#17becf",
}

# GAMMA_FOR (lambda_cen, training-time centering weight, paper's gamma in Sec. 4.3/L(Phi,Psi))
# and ALPHA1_FOR (eta, the CLI --alpha1 value report_0831-4.sh used to launch each
# (dataset, backbone) run) are needed only to build the *checkpoint filename*; eta itself
# is swept freely at inference in these analyses regardless of this "default" value.
GAMMA_FOR = {
    ("micro_video", "mf"): 1.0, ("micro_video", "grurec"): 1.0, ("micro_video", "sasrec"): 3.0,
    ("micro_video", "tisasrec"): 0.5, ("micro_video", "fearec"): 0.1, ("micro_video", "bsarec"): 0.1,
    ("ml-1m", "mf"): 3.0, ("ml-1m", "grurec"): 10.0, ("ml-1m", "sasrec"): 0.5,
    ("ml-1m", "tisasrec"): 1.0, ("ml-1m", "fearec"): 0.1, ("ml-1m", "bsarec"): 0.1,
    ("kuairand", "mf"): 0.1, ("kuairand", "grurec"): 10.0, ("kuairand", "sasrec"): 0.5,
    ("kuairand", "tisasrec"): 0.1, ("kuairand", "fearec"): 0.1, ("kuairand", "bsarec"): 0.1,
}
ALPHA1_FOR = {
    ("micro_video", "mf"): 0.3, ("micro_video", "grurec"): 0.3, ("micro_video", "sasrec"): 0.1,
    ("micro_video", "tisasrec"): 0.1, ("micro_video", "fearec"): 0.1, ("micro_video", "bsarec"): 0.1,
    ("ml-1m", "mf"): 0.7, ("ml-1m", "grurec"): 0.5, ("ml-1m", "sasrec"): 0.5,
    ("ml-1m", "tisasrec"): 0.5, ("ml-1m", "fearec"): 0.3, ("ml-1m", "bsarec"): 0.5,
    ("kuairand", "mf"): 0.3, ("kuairand", "grurec"): 0.5, ("kuairand", "sasrec"): 0.5,
    ("kuairand", "tisasrec"): 0.5, ("kuairand", "fearec"): 0.5, ("kuairand", "bsarec"): 0.5,
}
BSAREC_ALPHA = {"micro_video": 0.7, "ml-1m": 0.7, "kuairand": 0.9}
TISASREC_TIME_SPAN = {"micro_video": 512, "ml-1m": 2048, "kuairand": 512}


def load_dataset(dataset_name, rescale_item_time_array=True):
    """UserItemTime(...) exactly as every training/eval script in baseline/ constructs it
    (time_unit="d", time_len=50, seq_len=50). If rescale_item_time_array, applies the same
    /86400 rescale that debiased_seq_rec_hawkes_anova.py line 48 applies before Hawkes
    inference (item_time_array is built from time_dict, which is in SECONDS, but query
    times from set_to_pair are in DAYS)."""
    t0 = time.time()
    ds = UserItemTime(os.path.join(REPO_ROOT, "data"), dataset_name, "d", 50, 50)
    if rescale_item_time_array:
        ds.item_time_array = ds.item_time_array / 86400.0
    print(f"[load_dataset:{dataset_name}] built in {time.time()-t0:.1f}s "
          f"(n_user={ds.n_user}, m_item={ds.m_item}, "
          f"train/valid/test={ds.trainDataSize}/{ds.validDataSize}/{ds.testDataSize})")
    return ds


def _model_kwargs(dataset_name, model_name, device, tau):
    kwargs = dict(num_users=None, num_items=None, embedding_k=128, device=device, tau=tau,
                  depth=0, max_seq_len=50, n_heads=1, dropout=0.2)
    if model_name == "bsarec":
        kwargs["c"] = 3
        kwargs["alpha"] = BSAREC_ALPHA[dataset_name]
    if model_name == "tisasrec":
        kwargs["time_span"] = TISASREC_TIME_SPAN[dataset_name]
    return kwargs


def build_backbone_model(dataset_name, model_name, ds, device="cpu"):
    kwargs = _model_kwargs(dataset_name, model_name, device, tau=0.5)
    kwargs["num_users"], kwargs["num_items"] = ds.n_user, ds.m_item
    return MODEL_REGISTRY[model_name](**kwargs).to(device)


def build_ours_model(dataset_name, model_name, ds, device="cpu"):
    kwargs = _model_kwargs(dataset_name, model_name, device, tau=0.1)
    kwargs["num_users"], kwargs["num_items"] = ds.n_user, ds.m_item
    kwargs["score_norm"] = "normalized"
    debiased_cls = build_unshared_debias_model(MODEL_REGISTRY[model_name])
    return debiased_cls(**kwargs).to(device)


def load_backbone_ckpt(dataset_name, model_name, seed, ds, device="cpu"):
    model = build_backbone_model(dataset_name, model_name, ds, device)
    path = f"{REPO_ROOT}/weights/{dataset_name}/_backbone_{model_name}_e500_seed{seed}.pt"
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_ours_ckpt(dataset_name, model_name, seed, ds, device="cpu", eta_override=None):
    model = build_ours_model(dataset_name, model_name, ds, device)
    lam = GAMMA_FOR[(dataset_name, model_name)]
    suffix = "cfhawkesanova" if model_name == "mf" else "hawkesanova"
    path = (f"{REPO_ROOT}/weights_hawkes_anova_generalize/{dataset_name}/"
            f"_{model_name}_lambdacen{lam}_tau0.1_scorenormnormalized_e500_seed{seed}"
            f"_ablationshared_{suffix}.pt")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    eta = eta_override if eta_override is not None else ALPHA1_FOR[(dataset_name, model_name)]
    return model, eta


def additional_feat(model_name, users, hist_time_np, device):
    """TiSASRec's encode_user takes history TIMESTAMPS (seconds), every other backbone
    takes the user id. See seq_rec_tisasrec.py / debiased_seq_rec_tisasrec_hawkes_anova.py."""
    if model_name == "tisasrec":
        return torch.tensor(hist_time_np, dtype=torch.long, device=device) * 24 * 60 * 60
    return torch.tensor(users, dtype=torch.long, device=device)


def build_eval_batch(ds, data_split, max_seq_len=50):
    """Reproduces exactly what every eval loop in baseline/*.py does: set_to_pair -> per-
    event history construction via ds.build_histories. Returns arrays, not a DataLoader,
    so callers can batch/chunk themselves."""
    pairs = ds.set_to_pair(data_split, ds.time_dict, ds.time_unit)
    users, items, times = zip(*[(u, v, t) for (u, v), t in pairs.items()])
    users = np.array(users, np.int64)
    items = np.array(items, np.int64)
    times = np.array(times, np.float64)
    events = list(zip(users.tolist(), [0] * len(users), times.tolist()))
    hist_item_np, hist_time_np = ds.build_histories(events, max_seq_len)
    return users, items, times, hist_item_np, hist_time_np


def run_utility_scores(model, ds, users, hist_item_np, hist_time_np, model_name,
                        batch_size=256, device="cpu", mask_seen=True):
    """f_Psi(x_u(t), v) for every candidate item v, i.e. the cosine(u,v)/tau utility score
    (works for both backbone and Ours checkpoints -- Ours additionally normalizes, backbone
    training used raw dot product, so this function is only meaningful for Ours checkpoints;
    use run_backbone_scores for the Vanilla backbone's own raw-dot-product score)."""
    N, m_item = len(users), ds.m_item
    out = np.zeros((N, m_item), dtype=np.float32)
    with torch.no_grad():
        v_all = F.normalize(model.get_item_repr(torch.arange(m_item, device=device)), dim=-1, eps=1e-8)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            hist_t = torch.tensor(hist_item_np[s:e], dtype=torch.long, device=device)
            feat_t = additional_feat(model_name, users[s:e], hist_time_np[s:e], device)
            u = F.normalize(model.encode_user(hist_t, feat_t), dim=-1, eps=1e-8)
            out[s:e] = (torch.matmul(u, v_all.T) / model.tau).cpu().numpy()
    if mask_seen:
        for i, u in enumerate(users):
            out[i, list(ds._allPos[u])] = -9999.0
    return out


def run_backbone_scores(model, ds, users, hist_item_np, hist_time_np, model_name,
                         batch_size=256, device="cpu", mask_seen=True):
    """Raw dot-product score, exactly as seq_rec.py / seq_rec_tisasrec.py / cf.py compute
    it for the Vanilla backbone (no normalization, no tau)."""
    N, m_item = len(users), ds.m_item
    out = np.zeros((N, m_item), dtype=np.float32)
    with torch.no_grad():
        v_all = model.get_item_repr(torch.arange(m_item, device=device))
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            hist_t = torch.tensor(hist_item_np[s:e], dtype=torch.long, device=device)
            feat_t = additional_feat(model_name, users[s:e], hist_time_np[s:e], device)
            u = model.encode_user(hist_t, feat_t)
            out[s:e] = torch.matmul(u, v_all.T).cpu().numpy()
    if mask_seen:
        for i, u in enumerate(users):
            out[i, list(ds._allPos[u])] = -9999.0
    return out


def hawkes_lambda_at_times(model, ds, query_times, batch_size=64):
    """lambda_v(t; Phi) = mu_v + alpha_v * sum_{t_j<t} exp(-beta(t-t_j)) for every item v,
    at every query time in query_times (float array, in days -- same units as
    ds.item_time_array after load_dataset's rescale). Returns (N_times, m_item) array of
    raw (unnormalized) intensities -- caller divides by row-sum for pi_Phi(v|t).

    Uses the same closed-form batched computation as
    analysis_kuairand_tail_diagnosis.run_ours_scores's logprob loop (a direct sum over the
    item's up-to-50 stored raw event times, masked to those preceding t); the model's own
    architecture already caps stored history at 50 events per item (module/dataset.py
    time_dict_to_array, time_len=50), so there is no separate recursive-state update to
    implement beyond what that fixed-size cap already gives us -- summing over <=50 stored
    timestamps per item per chunk is the efficient form here.
    """
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()
    mu, alpha, beta = mu.cpu().numpy(), alpha.cpu().numpy(), float(beta.cpu().numpy())
    item_time_array = ds.item_time_array.astype(np.float32)
    N = len(query_times)
    out = np.zeros((N, ds.m_item), dtype=np.float32)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        t_batch = np.asarray(query_times[s:e], dtype=np.float32).reshape(-1, 1, 1)
        mask = item_time_array[None, :, :] < t_batch
        delta = np.clip(t_batch - item_time_array[None, :, :], 0.0, None)
        h = (np.exp(-beta * delta) * mask).sum(-1)
        out[s:e] = mu[None, :] + alpha[None, :] * h
    return out, mu, alpha, beta


def rank_of_true_item(scores, items):
    """1-indexed rank (1 = best) of the true item within each row's full candidate set."""
    true_score = scores[np.arange(len(items)), items]
    return (scores > true_score[:, None]).sum(axis=1) + 1


def mask_seen(scores, ds, users, value=-9999.0):
    for i, u in enumerate(users):
        scores[i, list(ds._allPos[u])] = value
    return scores


# ---------------------------------------------------------------------------
# Concentration / distributional statistics (Sec. 3.2 / 5 of the task spec)
# ---------------------------------------------------------------------------

def normalized_entropy(p, eps=1e-12):
    """H_norm = -sum p log p / log|support|. p need not be normalized; support is the
    number of entries with p > 0 (catalog size if p is defined over the whole catalog with
    zero-probability entries included, as specified in Sec. 3.1)."""
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p / total
    support = len(p)
    nz = p > eps
    h = -(p[nz] * np.log(p[nz])).sum()
    return float(h / np.log(support)) if support > 1 else np.nan


def effective_support(p, eps=1e-12):
    """N_eff = exp(H) (Shannon entropy, nats), i.e. the "effective number of items"."""
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return np.nan
    p = p / total
    nz = p > eps
    h = -(p[nz] * np.log(p[nz])).sum()
    return float(np.exp(h))


def top_q_mass(p, q_fracs=(0.01, 0.05, 0.20)):
    """C_q = cumulative probability mass held by the top q-fraction of items by p."""
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return {q: np.nan for q in q_fracs}
    p_sorted = np.sort(p)[::-1] / total
    n = len(p_sorted)
    cum = np.cumsum(p_sorted)
    out = {}
    for q in q_fracs:
        k = max(1, int(round(q * n)))
        out[q] = float(cum[k - 1])
    return out


def gini(p):
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return np.nan
    x = np.sort(p) / total
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1] if cum[-1] > 0 else 0)) / n) if cum[-1] > 0 else np.nan


def jensen_shannon_divergence(p, q, eps=1e-12, base=2.0):
    """JSD in the given log base (base=2 -> bounded in [0,1] bit units). p, q need not be
    pre-normalized."""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > eps
        return float((a[nz] * np.log(a[nz] / np.maximum(b[nz], eps))).sum())

    js_nats = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return js_nats / np.log(base)


def load_item_head_tail_labels(dataset_name, kind="overall"):
    """Item-level Head/Tail labels from the repository's own (static) split
    (data/<dataset>/testing_dict_{head,tail}_{kind}.npy), verified disjoint at the item
    level (0 overlap on KuaiRand). kind in {"overall", "recent_3d", "recent_7d"}. Returns
    (head_items: set[int], tail_items: set[int]) -- items never appearing as a test
    positive are in neither set."""
    path = os.path.join(REPO_ROOT, "data", dataset_name)
    head_dict = np.load(f"{path}/testing_dict_head_{kind}.npy", allow_pickle=True).item()
    tail_dict = np.load(f"{path}/testing_dict_tail_{kind}.npy", allow_pickle=True).item()
    head_items, tail_items = set(), set()
    for u, items in head_dict.items():
        head_items.update(items)
    for u, items in tail_dict.items():
        tail_items.update(items)
    return head_items, tail_items


def cumulative_popularity_curve(p, n_points=200):
    """Time-average-ready cumulative mass curve: sorts items by p descending, returns
    (percentile_grid in [0,100], cumulative mass at each percentile) via interpolation, so
    curves from different catalog sizes / different snapshots can be averaged directly."""
    p = np.asarray(p, dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return np.linspace(0, 100, n_points), np.full(n_points, np.nan)
    p_sorted = np.sort(p)[::-1] / total
    n = len(p_sorted)
    cum = np.cumsum(p_sorted)
    item_pct = (np.arange(1, n + 1) / n) * 100.0
    grid = np.linspace(0, 100, n_points)
    curve = np.interp(grid, item_pct, cum)
    return grid, curve
