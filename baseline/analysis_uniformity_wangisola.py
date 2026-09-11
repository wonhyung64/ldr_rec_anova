#%%
"""
Wang & Isola (2020)-style uniformity visualization, swept over lambda_cen.

For every (dataset, backbone) pair and every lambda_cen in the grid, this
projects the learned, unit-normalized user-context representations h_u(t)
onto their top-2 UNCENTERED singular directions (uncentered so the shared
popularity-absorption direction Theorem 1 is about isn't subtracted out),
renormalizes each projection to the unit circle exactly as Wang & Isola's
2D encoder output already is, fits a plain 2D Gaussian KDE on those (x, y)
points (no von-Mises trick needed -- Euclidean KDE already handles the
angular wraparound correctly because +-pi are close in (x, y) too), and
plots:
  - top row:    the KDE evaluated along the unit circle itself (the "ring")
  - bottom row: a histogram of the angles theta = atan2(y, x)

One PNG per (dataset, backbone), with one column per lambda_cen value.
Also prints the mean resultant length R (1 = collapsed to a point,
0 = uniform on the circle) for every (dataset, backbone, lambda_cen) so the
backbone with the cleanest / most dramatic lambda_cen -> uniformity trend
can be picked out without eyeballing all 18 figures.

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_uniformity_wangisola.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import TruncatedSVD

from module.utils import set_seed
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

DATASETS = ["micro_video", "ml-1m", "kuairand"]
BACKBONES = ["mf", "grurec", "sasrec", "tisasrec", "fearec", "bsarec"]
DISPLAY_NAME = {"mf": "MF", "grurec": "GRU", "sasrec": "SASRec",
                "tisasrec": "TiSASRec", "fearec": "FEARec", "bsarec": "BSARec"}
LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0]
SEED = 1
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


def build_model(model_name, dataset_name, n_user, m_item):
    model_class = MODEL_REGISTRY[model_name]
    debiased_class = build_unshared_debias_model(model_class)
    return debiased_class(
        num_users=n_user, num_items=m_item, embedding_k=RECDIM,
        device=device, tau=HAWKES_TAU, depth=DEPTH, max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS, dropout=DROPOUT, score_norm=SCORE_NORM,
        **extra_kwargs(model_name, dataset_name),
    ).to(device)


def checkpoint_path(model_name, dataset_name, lam):
    suffix = "cfhawkesanova" if model_name == "mf" else "hawkesanova"
    return (
        f"./weights_hawkes_anova_generalize/{dataset_name}/_{model_name}_lambdacen{lam}_tau{HAWKES_TAU}"
        f"_scorenorm{SCORE_NORM}_e500_seed{SEED}_ablation{ABLATION}_{suffix}.pt"
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


def to_unit_circle(reps):
    """Uncentered top-2 SVD projection, renormalized onto S^1 (Wang & Isola's
    encoder already outputs S^1 directly; we approximate that view of a
    128-d unit-hypersphere representation)."""
    svd = TruncatedSVD(n_components=2)
    xy = svd.fit_transform(reps)
    r = np.linalg.norm(xy, axis=1, keepdims=True)
    xy_unit = xy / np.clip(r, 1e-8, None)
    return xy_unit


#%%
summary = {}  # (dataset, backbone, lambda) -> resultant_R
circle_grid = np.linspace(-np.pi, np.pi, 720)
circle_xy = np.stack([np.cos(circle_grid), np.sin(circle_grid)], axis=0)  # [2, 720]

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
        fig, axes = plt.subplots(2, len(LAMBDA_GRID), figsize=(2.6 * len(LAMBDA_GRID), 5.6),
                                  gridspec_kw=dict(height_ratios=[2.2, 1]))

        for col, lam in enumerate(LAMBDA_GRID):
            model = build_model(model_name, dataset_name, dataset.n_user, dataset.m_item)
            ckpt = torch.load(checkpoint_path(model_name, dataset_name, lam), map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            reps = extract_representations(model, model_name, hist_item_np, hist_time_np, user_np)

            xy_unit = to_unit_circle(reps)
            angles = np.arctan2(xy_unit[:, 1], xy_unit[:, 0])
            resultant_R = np.abs(np.mean(np.exp(1j * angles)))
            summary[(dataset_name, model_name, lam)] = resultant_R

            # plain 2D Gaussian KDE (no periodic trick needed in Euclidean x,y)
            kde = gaussian_kde(xy_unit.T)
            ring_density = kde(circle_xy)

            ax_top, ax_bot = axes[0, col], axes[1, col]

            sizes = 20 + 300 * (ring_density - ring_density.min()) / (np.ptp(ring_density) + 1e-12)
            ax_top.scatter(circle_xy[0], circle_xy[1], c=ring_density, cmap="turbo",
                            s=sizes, alpha=0.6, linewidths=0)
            ax_top.scatter(xy_unit[:, 0], xy_unit[:, 1], s=3, c="black", alpha=0.15, linewidths=0)
            ax_top.set_xlim(-1.2, 1.2)
            ax_top.set_ylim(-1.2, 1.2)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_top.set_aspect("equal")
            for spine in ax_top.spines.values():
                spine.set_visible(False)
            ax_top.set_title(rf"$\lambda_{{\mathrm{{cen}}}}={lam}$" + "\n" + rf"$R={resultant_R:.2f}$",
                              fontsize=11)

            ax_bot.hist(angles, bins=60, range=(-np.pi, np.pi), color="#4C72B0")
            ax_bot.set_xlim(-np.pi, np.pi)
            ax_bot.set_xticks([-np.pi, 0, np.pi])
            ax_bot.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=8)
            ax_bot.set_yticks([])

            print(f"[{dataset_name:>11s} | {DISPLAY_NAME[model_name]:>8s} | lambda_cen={lam:>5}] R={resultant_R:.4f}")

        fig.suptitle(f"{DISPLAY_NAME[model_name]} on {dataset_name}: uniformity of $h_u(t)$ on $S^1$ "
                      "(top-2 uncentered SVD projection) vs. $\\lambda_{cen}$", y=1.03, fontsize=12)
        fig.tight_layout()
        out_path = f"./figures/wangisola_{dataset_name}_{model_name}.png"
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_path}")


#%%
# ----------------------------------------------------------------------
# pick the "prettiest" backbone: largest, most monotonic drop in R as
# lambda_cen increases (tight cluster at low lambda -> spread at high lambda)
# ----------------------------------------------------------------------
print("\n########## ranking backbones by lambda_cen -> uniformity trend ##########")
for dataset_name in DATASETS:
    print(f"\n-- {dataset_name} --")
    scored = []
    for model_name in BACKBONES:
        Rs = [summary[(dataset_name, model_name, lam)] for lam in LAMBDA_GRID]
        r_range = max(Rs) - min(Rs)
        # monotonicity: fraction of consecutive steps that decrease
        diffs = np.diff(Rs)
        monotonic_frac = np.mean(diffs <= 0)
        scored.append((model_name, r_range, monotonic_frac, Rs))
        print(f"  {DISPLAY_NAME[model_name]:>8s}: R={['%.3f' % r for r in Rs]}  "
              f"range={r_range:.3f}  monotonic_frac={monotonic_frac:.2f}")
    best = max(scored, key=lambda x: x[1] * (0.5 + 0.5 * x[2]))
    print(f"  ==> prettiest for {dataset_name}: {DISPLAY_NAME[best[0]]} "
          f"(range={best[1]:.3f}, monotonic_frac={best[2]:.2f})")
