#%%
"""
Analysis of Representation: geometric mechanism behind the uniformity gap
between Vanilla backbones and the proposed method.

The existing table already shows that +Ours substantially lowers the
empirical uniformity loss

    L_uni = log E_{h,h'}[exp(-tau * ||h - h'||_2^2)]      (tau = 2)

across all six backbones and three datasets. This script does NOT repeat
that table. It investigates *why* the gap exists, via three complementary,
representation-level analyses run on a single representative backbone
(SASRec by default, swappable with --backbone):

  1. Common-PCA geometry:  Vanilla vs Ours user-context representations,
     projected onto ONE jointly-fit 2D PCA basis, plotted as scatter + KDE
     contours on shared axes.

  2. Alignment to the dominant shared direction: for normalized context
     vectors h_bar_i, the mean direction c_hat = mean(h_bar_i) / ||...||
     and the per-context cosine alignment a_i = c_hat^T h_bar_i. The mean
     resultant length R = ||mean(h_bar_i)|| in [0, 1] is the same quantity
     that appears in Theorem 1's concentration bound (R -> 1 as a shared
     component with strength rho dominates; R -> 0 for dispersed
     directions).

  3. Temporal popularity concentration vs. representation concentration:
     for each of several time buckets in the test period, relate the
     EMPIRICAL item-popularity concentration D_KL(p_hat(.|b) || Unif(V))
     (data-derived, model-independent -- see note below) to the
     representation concentration R_b computed from each model's own
     context vectors restricted to that bucket.

     NOTE on popularity source: Vanilla has no learned popularity
     component at all (only +Ours has the Hawkes branch), so a model-based
     popularity estimate cannot be defined identically for both models. We
     therefore use the empirical interaction-frequency distribution within
     each bucket (item counts vs. Unif(V)) as a model-independent reference
     measure of "how concentrated was engagement in this period" -- this is
     the same quantity for Vanilla and Ours by construction, which is what
     validation check (6)/(same buckets for Vanilla and Ours) requires, and
     it avoids attributing a Vanilla-side data point to a popularity model
     Vanilla never sees. This is a deliberate, documented deviation from a
     literal "use the learned Hawkes values" reading -- see the printed
     report for the alternative (Ours' own learned pi_hat(v|t)) if needed.

This is an empirical, correlational analysis; see the printed report for
the exact non-causal language used in the caption.

Usage (from the repo root):
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_representation.py \
        --backbone sasrec --datasets micro_video ml-1m kuairand --seeds 1 2 3 4

Outputs land in ./analysis_representation/:
    representation_geometry_full.{pdf,png}
    representation_geometry_compact.{pdf,png}
    uniformity_summary.csv
    dominant_alignment_summary.csv
    temporal_concentration_summary.csv
    representations_<dataset>_<backbone>_<approach>_<seed>.npz   (cache)
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

# ----------------------------------------------------------------------
# constants shared with the earlier analysis scripts (report_hawkes_anova_
# generalize.sh / analysis_luni_table.py) -- copied rather than imported so
# this script stays self-contained per-file, per the "add a separate
# analysis script" guidance.
# ----------------------------------------------------------------------
DATASETS_ALL = ["micro_video", "ml-1m", "kuairand"]
DISPLAY_DATASET = {"micro_video": "Micro-Video", "ml-1m": "MovieLens-1M", "kuairand": "KuaiRand"}
DISPLAY_BACKBONE = {"mf": "MF", "grurec": "GRU", "sasrec": "SASRec",
                     "tisasrec": "TiSASRec", "fearec": "FEARec", "bsarec": "BSARec"}

WINNING_LAMBDA = {
    "micro_video": {"mf": 1.0, "grurec": 1.0, "sasrec": 3.0, "tisasrec": 0.5, "fearec": 0.1, "bsarec": 0.1},
    "ml-1m":       {"mf": 3.0, "grurec": 10.0, "sasrec": 0.5, "tisasrec": 1.0, "fearec": 0.1, "bsarec": 0.1},
    "kuairand":    {"mf": 0.1, "grurec": 10.0, "sasrec": 0.5, "tisasrec": 0.1, "fearec": 0.1, "bsarec": 0.1},
}
BSAREC_ALPHA_FOR = {"micro_video": 0.7, "ml-1m": 0.7, "kuairand": 0.9}
BSAREC_C = 1
TIME_SPAN_FOR = {"micro_video": 512, "ml-1m": 2048, "kuairand": 512}

RECDIM = 128
DROPOUT = 0.2
MAX_SEQ_LEN = 50
N_HEADS = 1
DEPTH = 0
HAWKES_TAU = 0.1          # tau used inside the +Ours score_norm (unrelated to uniformity's tau)
BASELINE_TAU = 0.5
SCORE_NORM = "normalized"
ABLATION = "shared"
UNIFORMITY_TAU = 2.0      # tau in L_uni = log E[exp(-tau * ||h-h'||^2)] -- matches the existing table

VANILLA_COLOR = "#4C72B0"   # muted blue
OURS_COLOR = "#DD8452"      # muted orange -- identical palette in every panel
VANILLA_LABEL, OURS_LABEL = "Vanilla", "Ours"

EVAL_SAMPLE_SEED = 12345   # fixes WHICH interactions are evaluated -- shared by Vanilla/Ours/all seeds


# ----------------------------------------------------------------------
# model construction (mirrors analysis_luni_table.py / analysis_uniformity_
# wangisola.py so checkpoints load identically to how they were trained)
# ----------------------------------------------------------------------
def extra_kwargs(model_name, dataset_name):
    if model_name == "bsarec":
        return dict(alpha=BSAREC_ALPHA_FOR[dataset_name], c=BSAREC_C)
    if model_name == "tisasrec":
        return dict(time_span=TIME_SPAN_FOR[dataset_name])
    return {}


def build_baseline_model(model_name, dataset_name, n_user, m_item, device):
    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=BASELINE_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, **extra_kwargs(model_name, dataset_name),
    ).to(device)


def build_ours_model(model_name, dataset_name, n_user, m_item, device):
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


def load_model(approach, model_name, dataset_name, seed, n_user, m_item, device):
    if approach == "vanilla":
        model = build_baseline_model(model_name, dataset_name, n_user, m_item, device)
        ckpt = torch.load(baseline_path(model_name, dataset_name, seed), map_location=device)
    else:
        model = build_ours_model(model_name, dataset_name, n_user, m_item, device)
        ckpt = torch.load(ours_path(model_name, dataset_name, seed), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ----------------------------------------------------------------------
# evaluation-context sampling: the SAME (user, item, time) interactions are
# used for Vanilla and Ours and every seed (validation check #2)
# ----------------------------------------------------------------------
def sample_eval_events(dataset, n_sample, rng_seed=EVAL_SAMPLE_SEED):
    rng = np.random.default_rng(rng_seed)
    items = list(dataset.test_user_item_time.items())  # [((u, v), t), ...]
    n = min(n_sample, len(items))
    idx = rng.choice(len(items), size=n, replace=False)
    idx.sort()
    events = [(items[i][0][0], items[i][0][1], float(items[i][1])) for i in idx]
    return events


@torch.no_grad()
def encode_batch(model, model_name, hist_item_np, hist_time_np, user_np, batch_size, device):
    reps = []
    for start in range(0, len(user_np), batch_size):
        end = start + batch_size
        hist_t = torch.tensor(hist_item_np[start:end], dtype=torch.long, device=device)
        if model_name == "tisasrec":
            time_t = torch.tensor(hist_time_np[start:end], dtype=torch.long, device=device) * 24 * 60 * 60
            h = model.encode_user(hist_t, time_t)
        else:
            user_t = torch.tensor(user_np[start:end], dtype=torch.long, device=device)
            h = model.encode_user(hist_t, user_t)
        reps.append(h.cpu().numpy())
    return np.concatenate(reps, axis=0)


def get_or_extract_representations(dataset, dataset_name, model_name, approach, seed,
                                    events, cache_dir, device, batch_size=512):
    """h_i = h_Psi(x_{u_i}(t_i)): the user-context vector returned by
    encode_user(), taken BEFORE score_all_items()/residual_score() match it
    against any candidate item embedding and BEFORE the F.normalize() those
    functions apply for scoring. Cached raw (unnormalized); normalization
    is applied explicitly downstream, only for geometry/uniformity metrics
    (validation check #1)."""
    cache_path = f"{cache_dir}/representations_{dataset_name}_{model_name}_{approach}_{seed}.npz"
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["h"], data["t"], data["u"], data["v"]

    hist_item_np, hist_time_np = dataset.build_histories(events, MAX_SEQ_LEN)
    user_np = np.array([e[0] for e in events], dtype=np.int64)
    item_np = np.array([e[1] for e in events], dtype=np.int64)
    time_np = np.array([e[2] for e in events], dtype=np.float64)

    model = load_model(approach, model_name, dataset_name, seed, dataset.n_user, dataset.m_item, device)
    h = encode_batch(model, model_name, hist_item_np, hist_time_np, user_np, batch_size, device)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, h=h.astype(np.float32), t=time_np, u=user_np, v=item_np)
    return h, time_np, user_np, item_np


def normalize_rows(h):
    return h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-8, None)


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def l_uni(h_norm, rng, tau=UNIFORMITY_TAU, n_pairs=20000):
    n = h_norm.shape[0]
    idx = rng.integers(0, n, size=(n_pairs, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    d2 = np.sum((h_norm[idx[:, 0]] - h_norm[idx[:, 1]]) ** 2, axis=1)
    val = float(np.log(np.mean(np.exp(-tau * d2))))
    assert val <= 1e-6, f"L_uni should be <= 0, got {val}"
    return val


def alignment_stats(h_norm):
    m = h_norm.mean(axis=0)
    R = float(np.linalg.norm(m))
    assert -1e-6 <= R <= 1 + 1e-6, f"R={R} outside [0,1]"
    c_hat = m / max(R, 1e-8)
    a = h_norm @ c_hat
    assert np.all(a >= -1 - 1e-6) and np.all(a <= 1 + 1e-6), "cosine alignment outside [-1,1]"
    return dict(R=R, c_hat=c_hat, a=a,
                mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()), p90=float(np.percentile(a, 90)))


def quantile_bucket_edges(times, n_buckets):
    qs = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(times, qs)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    # de-duplicate degenerate edges (very bursty timestamps) by nudging
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def empirical_popularity_kl(item_ids, m_item, eps=1e-12):
    if len(item_ids) == 0:
        return np.nan
    counts = np.bincount(item_ids, minlength=m_item).astype(np.float64)
    p = counts / counts.sum()
    p = np.clip(p, eps, None)
    u = 1.0 / m_item
    return float(np.sum(p * np.log(p / u)))


# ----------------------------------------------------------------------
# per-dataset pipeline
# ----------------------------------------------------------------------
def run_dataset(dataset_name, model_name, seeds, n_sample, n_time_buckets, cache_dir, device, rng):
    print(f"\n########## {dataset_name} ({model_name}) ##########")
    dataset = UserItemTime("./data", dataset_name, "d", 50, MAX_SEQ_LEN)
    events = sample_eval_events(dataset, n_sample)
    print(f"[data] {len(events)} shared evaluation interactions (Vanilla and Ours, all seeds)")

    per_seed = {"vanilla": {}, "ours": {}}  # seed -> dict(h_norm, t, v, l_uni, align)
    for approach in ["vanilla", "ours"]:
        for seed in seeds:
            h, t, u, v = get_or_extract_representations(
                dataset, dataset_name, model_name, approach, seed, events, cache_dir, device)
            h_norm = normalize_rows(h)
            luni_val = l_uni(h_norm, rng)
            align = alignment_stats(h_norm)
            per_seed[approach][seed] = dict(h_norm=h_norm, t=t, v=v, l_uni=luni_val, align=align)
            print(f"  [{approach:>7s} seed={seed}] L_uni={luni_val:.4f}  R={align['R']:.4f}  "
                  f"mean_cos={align['mean']:.4f}")

    # -------- median-performance seed for the qualitative PCA panel --------
    ours_luni_by_seed = {s: per_seed["ours"][s]["l_uni"] for s in seeds}
    order = sorted(seeds, key=lambda s: ours_luni_by_seed[s])
    median_seed = order[len(order) // 2]
    print(f"[median seed] chosen by median +Ours L_uni across seeds {seeds}: seed={median_seed} "
          f"(L_uni={ours_luni_by_seed[median_seed]:.4f}); used for BOTH Vanilla and Ours in the "
          f"qualitative PCA panel for a paired comparison.")

    # -------- panel 1 data: joint PCA on the median seed --------
    h_van = per_seed["vanilla"][median_seed]["h_norm"]
    h_ours = per_seed["ours"][median_seed]["h_norm"]
    # same shared `events` list -> Vanilla and Ours always have equal counts here
    n_common = min(len(h_van), len(h_ours))
    from sklearn.decomposition import PCA
    concat = np.concatenate([h_van[:n_common], h_ours[:n_common]], axis=0)
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(concat)
    proj_van, proj_ours = proj[:n_common], proj[n_common:]
    panel1 = dict(proj_van=proj_van, proj_ours=proj_ours,
                  explained_var=pca.explained_variance_ratio_, median_seed=median_seed)

    # -------- panel 2 data: alignment distribution pooled over all seeds --------
    a_van_all = np.concatenate([per_seed["vanilla"][s]["align"]["a"] for s in seeds])
    a_ours_all = np.concatenate([per_seed["ours"][s]["align"]["a"] for s in seeds])
    panel2 = dict(a_van=a_van_all, a_ours=a_ours_all)

    # -------- panel 3 data: temporal popularity vs. representation concentration --------
    all_test_items = np.array([item for (_, item), _ in dataset.test_user_item_time.items()], dtype=np.int64)
    all_test_times = np.array([t for _, t in dataset.test_user_item_time.items()], dtype=np.float64)
    edges = quantile_bucket_edges(all_test_times, n_time_buckets)

    bucket_rows = []
    for b in range(n_time_buckets):
        lo, hi = edges[b], edges[b + 1]
        pop_mask = (all_test_times >= lo) & (all_test_times < hi)
        kl_b = empirical_popularity_kl(all_test_items[pop_mask], dataset.m_item)

        row = dict(bucket=b, lo=lo, hi=hi, n_pop=int(pop_mask.sum()), kl=kl_b)
        for approach in ["vanilla", "ours"]:
            Rs = []
            for seed in seeds:
                t_s = per_seed[approach][seed]["t"]
                h_s = per_seed[approach][seed]["h_norm"]
                mask = (t_s >= lo) & (t_s < hi)
                if mask.sum() < 3:
                    continue
                m = h_s[mask].mean(axis=0)
                Rs.append(float(np.linalg.norm(m)))
            row[f"R_{approach}_mean"] = float(np.mean(Rs)) if Rs else np.nan
            row[f"R_{approach}_std"] = float(np.std(Rs)) if Rs else np.nan
            row[f"n_{approach}"] = len(Rs)
        bucket_rows.append(row)

    panel3 = dict(rows=bucket_rows)

    summary = dict(
        dataset=dataset_name, model=model_name, seeds=seeds, median_seed=median_seed,
        l_uni={approach: {s: per_seed[approach][s]["l_uni"] for s in seeds} for approach in ["vanilla", "ours"]},
        align={approach: {s: {k: v for k, v in per_seed[approach][s]["align"].items() if k not in ("a", "c_hat")}
                           for s in seeds} for approach in ["vanilla", "ours"]},
    )
    return panel1, panel2, panel3, summary


# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------
def setup_style():
    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_pca_panel(ax, panel1, show_legend=False):
    proj = np.concatenate([panel1["proj_van"], panel1["proj_ours"]], axis=0)
    lim = np.abs(proj).max() * 1.1

    ax.scatter(*panel1["proj_van"].T, s=4, alpha=0.15, color=VANILLA_COLOR, linewidths=0, rasterized=True)
    ax.scatter(*panel1["proj_ours"].T, s=4, alpha=0.15, color=OURS_COLOR, linewidths=0, rasterized=True)
    for proj, color in [(panel1["proj_van"], VANILLA_COLOR), (panel1["proj_ours"], OURS_COLOR)]:
        try:
            kde = gaussian_kde(proj.T)
            grid = np.linspace(-lim, lim, 120)
            GX, GY = np.meshgrid(grid, grid)
            dens = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            levels = np.unique(np.quantile(dens, [0.5, 0.75, 0.9, 0.97]))
            if len(levels) >= 2:
                ax.contour(GX, GY, dens, levels=levels, colors=[color], linewidths=0.9, alpha=0.9)
        except (np.linalg.LinAlgError, ValueError):
            pass
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"PC1 ({panel1['explained_var'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({panel1['explained_var'][1]*100:.1f}%)")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        handles = [plt.Line2D([0], [0], color=VANILLA_COLOR, lw=2, label=VANILLA_LABEL),
                   plt.Line2D([0], [0], color=OURS_COLOR, lw=2, label=OURS_LABEL)]
        ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.2)


def plot_alignment_panel(ax, panel2, show_legend=False):
    grid = np.linspace(-1, 1, 400)
    for a, color, label in [(panel2["a_van"], VANILLA_COLOR, VANILLA_LABEL),
                             (panel2["a_ours"], OURS_COLOR, OURS_LABEL)]:
        kde = gaussian_kde(a, bw_method=0.08)
        dens = kde(grid)
        ax.plot(grid, dens, color=color, lw=1.3, label=label)
        ax.fill_between(grid, dens, color=color, alpha=0.15)
        ax.axvline(a.mean(), color=color, lw=0.8, ls="--", alpha=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"cosine alignment $a_i = \hat c^\top \bar h_i$")
    ax.set_ylabel("density")
    ax.set_yticks([])
    if show_legend:
        ax.legend(loc="upper left", frameon=False, handlelength=1.2)


def plot_temporal_panel(ax, panel3, show_legend=False, annotate=True):
    rows = panel3["rows"]
    kl = np.array([r["kl"] for r in rows])
    stats_text = []
    for approach, color, label in [("vanilla", VANILLA_COLOR, VANILLA_LABEL), ("ours", OURS_COLOR, OURS_LABEL)]:
        Rm = np.array([r[f"R_{approach}_mean"] for r in rows])
        Rs = np.array([r[f"R_{approach}_std"] for r in rows])
        valid = ~np.isnan(kl) & ~np.isnan(Rm)
        ax.errorbar(kl[valid], Rm[valid], yerr=Rs[valid], fmt="o", ms=4, elinewidth=0.8, capsize=2,
                    color=color, alpha=0.85, label=label)
        if valid.sum() >= 3:
            coeffs = np.polyfit(kl[valid], Rm[valid], 1)
            xs = np.linspace(kl[valid].min(), kl[valid].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color=color, lw=1.2, alpha=0.9)
            r_p, p_p = pearsonr(kl[valid], Rm[valid])
            r_s, p_s = spearmanr(kl[valid], Rm[valid])
            stats_text.append(f"{label}: r={r_p:.2f}, $\\rho_s$={r_s:.2f} (n={valid.sum()})")
    ax.set_xlabel(r"popularity concentration $D_{KL}(\hat p_b \Vert \mathrm{Unif})$")
    ax.set_ylabel(r"context concentration $R_b$")
    if annotate:
        ax.text(0.03, 0.97, "\n".join(stats_text), transform=ax.transAxes, va="top", ha="left", fontsize=7)
    if show_legend:
        ax.legend(loc="lower right", frameon=False, handlelength=1.2)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analysis of Representation figure")
    parser.add_argument("--backbone", type=str, default="sasrec", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS_ALL)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-sample", type=int, default=4000)
    parser.add_argument("--n-time-buckets", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="./analysis_representation")
    parser.add_argument("--cache-dir", type=str, default=None,
                         help="defaults to <out-dir> if not given")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cache_dir = args.cache_dir or args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    set_seed(0)
    rng = np.random.default_rng(0)
    setup_style()

    panel1_by_ds, panel2_by_ds, panel3_by_ds, summary_by_ds = {}, {}, {}, {}
    for dataset_name in args.datasets:
        p1, p2, p3, summ = run_dataset(dataset_name, args.backbone, args.seeds, args.n_sample,
                                        args.n_time_buckets, cache_dir, args.device, rng)
        panel1_by_ds[dataset_name] = p1
        panel2_by_ds[dataset_name] = p2
        panel3_by_ds[dataset_name] = p3
        summary_by_ds[dataset_name] = summ

    # ==================== CSV summaries ====================
    write_uniformity_csv(args.out_dir, summary_by_ds, args.backbone)
    write_alignment_csv(args.out_dir, summary_by_ds, args.backbone)
    write_temporal_csv(args.out_dir, panel3_by_ds, args.backbone)

    # ==================== full 3xN figure ====================
    n_ds = len(args.datasets)
    fig, axes = plt.subplots(n_ds, 3, figsize=(9.5, 3.0 * n_ds))
    if n_ds == 1:
        axes = axes[None, :]
    for row, dataset_name in enumerate(args.datasets):
        plot_pca_panel(axes[row, 0], panel1_by_ds[dataset_name], show_legend=(row == 0))
        plot_alignment_panel(axes[row, 1], panel2_by_ds[dataset_name], show_legend=(row == 0))
        plot_temporal_panel(axes[row, 2], panel3_by_ds[dataset_name], show_legend=(row == 0))
        axes[row, 0].set_ylabel(f"{DISPLAY_DATASET[dataset_name]}\n" + axes[row, 0].get_ylabel())
    col_titles = ["(a) common-PCA geometry", "(b) alignment to dominant direction",
                  "(c) popularity vs. context concentration"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title)
    fig.suptitle(f"Geometry of user-context representations ({DISPLAY_BACKBONE[args.backbone]})", y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{args.out_dir}/representation_geometry_full.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out_dir}/representation_geometry_full.{{pdf,png}}")

    # ==================== compact 5-panel figure ====================
    make_compact_figure(args.out_dir, args.backbone, args.datasets, panel1_by_ds, panel2_by_ds, panel3_by_ds)

    # ==================== caption ====================
    write_caption(args.out_dir, args.backbone, args.datasets, args.seeds)


def write_uniformity_csv(out_dir, summary_by_ds, backbone):
    import csv
    path = f"{out_dir}/uniformity_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "approach", "seed", "L_uni"])
        for dataset_name, summ in summary_by_ds.items():
            for approach in ["vanilla", "ours"]:
                for seed, val in summ["l_uni"][approach].items():
                    w.writerow([dataset_name, backbone, approach, seed, f"{val:.6f}"])
    print(f"[saved] {path}")


def write_alignment_csv(out_dir, summary_by_ds, backbone):
    import csv
    path = f"{out_dir}/dominant_alignment_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "approach", "seed", "R", "mean_cos", "median_cos", "std_cos", "p90_cos"])
        for dataset_name, summ in summary_by_ds.items():
            for approach in ["vanilla", "ours"]:
                for seed, stats in summ["align"][approach].items():
                    w.writerow([dataset_name, backbone, approach, seed,
                                f"{stats['R']:.6f}", f"{stats['mean']:.6f}", f"{stats['median']:.6f}",
                                f"{stats['std']:.6f}", f"{stats['p90']:.6f}"])
    print(f"[saved] {path}")


def write_temporal_csv(out_dir, panel3_by_ds, backbone):
    import csv
    path = f"{out_dir}/temporal_concentration_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "bucket", "t_lo", "t_hi", "n_pop_events", "kl_popularity",
                    "R_vanilla_mean", "R_vanilla_std", "n_vanilla_seeds",
                    "R_ours_mean", "R_ours_std", "n_ours_seeds"])
        for dataset_name, panel3 in panel3_by_ds.items():
            for row in panel3["rows"]:
                w.writerow([dataset_name, backbone, row["bucket"], f"{row['lo']:.4f}", f"{row['hi']:.4f}",
                            row["n_pop"], f"{row['kl']:.6f}",
                            f"{row['R_vanilla_mean']:.6f}", f"{row['R_vanilla_std']:.6f}", row["n_vanilla"],
                            f"{row['R_ours_mean']:.6f}", f"{row['R_ours_std']:.6f}", row["n_ours"]])
    print(f"[saved] {path}")


def make_compact_figure(out_dir, backbone, datasets, panel1_by_ds, panel2_by_ds, panel3_by_ds):
    fig = plt.figure(figsize=(9.5, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    for col, dataset_name in enumerate(datasets[:3]):
        ax = fig.add_subplot(gs[0, col])
        plot_pca_panel(ax, panel1_by_ds[dataset_name], show_legend=(col == 0))
        ax.set_title(f"({chr(97+col)}) {DISPLAY_DATASET[dataset_name]}")

    ax_d = fig.add_subplot(gs[1, 0])
    a_van_pool = np.concatenate([panel2_by_ds[d]["a_van"] for d in datasets])
    a_ours_pool = np.concatenate([panel2_by_ds[d]["a_ours"] for d in datasets])
    plot_alignment_panel(ax_d, dict(a_van=a_van_pool, a_ours=a_ours_pool), show_legend=True)
    ax_d.set_title("(d) alignment, pooled across datasets")

    ax_e = fig.add_subplot(gs[1, 1:])
    pooled_rows = []
    for dataset_name in datasets:
        rows = panel3_by_ds[dataset_name]["rows"]
        kl = np.array([r["kl"] for r in rows])
        valid = ~np.isnan(kl)
        if valid.sum() < 2:
            continue
        z = (kl - np.nanmean(kl)) / (np.nanstd(kl) + 1e-8)
        for r, zi in zip(rows, z):
            r2 = dict(r)
            r2["kl"] = zi
            pooled_rows.append(r2)
    plot_temporal_panel(ax_e, dict(rows=pooled_rows), show_legend=True)
    ax_e.set_xlabel(r"popularity concentration (per-dataset $z$-score of $D_{KL}$)")
    ax_e.set_title("(e) popularity vs. context concentration, pooled across datasets")

    fig.suptitle(f"Geometry of user-context representations ({DISPLAY_BACKBONE[backbone]}) -- compact", y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_dir}/representation_geometry_compact.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir}/representation_geometry_compact.{{pdf,png}}")


def write_caption(out_dir, backbone, datasets, seeds):
    caption = f"""Figure: Geometry of user-context representations under temporal popularity bias ({DISPLAY_BACKBONE[backbone]}).
Column (a) visualizes Vanilla and Ours in a single, jointly-fit PCA space (fit once on the pooled
representations of both models at a representative seed, then used to project each); column (b) compares
their cosine alignment to each model's own dominant representation direction c_hat = mean(h_bar_i) / ||...||;
column (c) relates the empirical, model-independent temporal popularity concentration
D_KL(p_hat(.|b) || Unif(V)) of each test-period time bucket to the directional concentration
R_b = ||mean_{{i in b}} h_bar_i||_2 of each model's own context vectors in that bucket. Lower R and broader,
less contour-concentrated PCA geometry indicate reduced dominance of a shared representation direction.
Quantitative panels (b, c) aggregate {len(seeds)} random seeds ({seeds}); panel (a) uses a single
median-performance seed (by +Ours L_uni) for a paired qualitative comparison, documented per dataset in the
run log. These patterns are consistent with the theoretical concentration mechanism (Theorem 1 / Corollary 1)
by which a sufficiently strong shared direction absorbed into user-context representations increases their
pairwise concentration; they are reported as an empirical, correlational analysis and do not establish that
non-identifiability of popularity and utility causes representation collapse in general.
"""
    path = f"{out_dir}/figure_caption.txt"
    with open(path, "w") as f:
        f.write(caption)
    print(f"[saved] {path}")
    print("\n" + caption)


if __name__ == "__main__":
    main()
