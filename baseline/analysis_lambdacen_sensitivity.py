#%%
"""
Theorem 1 / Corollary 1 sensitivity check.

Sweeps the ANOVA-centering penalty weight lambda_cen (the empirically
controllable knob that enforces Assumption 2 / Definition of centered
utility) for SASRec+Ours on Micro-Video, and visualizes how the learned
user-context representations h_u(t) are distributed on the unit
hypersphere. Weak centering (small lambda_cen) is expected to let a
popularity-aligned shared direction leak into f_Psi, concentrating
representations in a narrow arc (Theorem 1); strong centering should
spread them more uniformly (Corollary 1).

Run from the repo root:
    /Users/wonhyung64/miniforge3/envs/openmmlab/bin/python baseline/analysis_lambdacen_sensitivity.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde
from sklearn.decomposition import TruncatedSVD

from matplotlib.colors import PowerNorm

from module.utils import set_seed, set_device
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.debias import build_unshared_debias_model

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
DATASET = "micro_video"
MODEL_NAME = "sasrec"
LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 3.0, 10.0]
TAU = 0.1
SCORE_NORM = "normalized"
ABLATION = "shared"          # CLI value used at training time -> build_unshared_debias_model
RECDIM = 128
DROPOUT = 0.2
MAX_SEQ_LEN = 50
N_HEADS = 1
DEPTH = 0
SEED = 1
WEIGHTS_DIR = f"./weights_hawkes_anova_generalize/{DATASET}"
N_USERS_SAMPLE = 4000
BATCH_SIZE = 512
KDE_BW = 0.12
OUT_PATH = "./figures/fig_lambdacen_sensitivity_sasrec_microvideo.png"

set_seed(SEED)
device = "cpu"


#%%
# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
dataset = UserItemTime("./data", DATASET, "d", 50, MAX_SEQ_LEN)

# one (most-recent) test-time context per user, so every user contributes
# exactly one h_u(t) and the geometry isn't dominated by heavy testers
last_event = {}
for (user, item), t in dataset.test_user_item_time.items():
    if (user not in last_event) or (t > last_event[user][1]):
        last_event[user] = (item, t)

rng = np.random.default_rng(0)
all_users = np.array(sorted(last_event.keys()))
if len(all_users) > N_USERS_SAMPLE:
    sampled_users = rng.choice(all_users, size=N_USERS_SAMPLE, replace=False)
else:
    sampled_users = all_users
sampled_users.sort()

events = [(u, last_event[u][0], last_event[u][1]) for u in sampled_users]
hist_item_np, _ = dataset.build_histories(events, MAX_SEQ_LEN)
user_np = np.array([u for (u, v, t) in events], dtype=np.int64)

print(f"[data] {len(sampled_users)} users sampled from {len(all_users)} test-active users")


#%%
# ----------------------------------------------------------------------
# model factory + checkpoint loading
# ----------------------------------------------------------------------
def build_model():
    model_class = MODEL_REGISTRY[MODEL_NAME]
    debiased_class = build_unshared_debias_model(model_class)  # ablation == "shared"
    model = debiased_class(
        num_users=dataset.n_user,
        num_items=dataset.m_item,
        embedding_k=RECDIM,
        device=device,
        tau=TAU,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        score_norm=SCORE_NORM,
    ).to(device)
    return model


def checkpoint_path(lam):
    return (
        f"{WEIGHTS_DIR}/_{MODEL_NAME}_lambdacen{lam}_tau{TAU}_scorenorm{SCORE_NORM}"
        f"_e500_seed{SEED}_ablation{ABLATION}_hawkesanova.pt"
    )


# extreme baseline: plain backbone with NO popularity-separation architecture
# at all (not even the lambda_cen=0 Hawkes-branch scaffolding) -- this is the
# "unconstrained absorption" regime Theorem 1 is about.
BASELINE_LABEL = "no separation\n(backbone only)"
BASELINE_TAU = 0.5   # norm_backbone_* checkpoints were trained with the default --tau (0.5)
BASELINE_PATH = f"./weights/{DATASET}/norm_backbone_{MODEL_NAME}_e500_seed{SEED}.pt"


def build_baseline_model():
    model_class = MODEL_REGISTRY[MODEL_NAME]
    model = model_class(
        num_users=dataset.n_user,
        num_items=dataset.m_item,
        embedding_k=RECDIM,
        device=device,
        tau=BASELINE_TAU,
        depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        n_heads=N_HEADS,
        dropout=DROPOUT,
    ).to(device)
    return model


@torch.no_grad()
def extract_representations(model):
    model.eval()
    reps = []
    for start in range(0, len(user_np), BATCH_SIZE):
        end = start + BATCH_SIZE
        hist_t = torch.tensor(hist_item_np[start:end], dtype=torch.long, device=device)
        user_t = torch.tensor(user_np[start:end], dtype=torch.long, device=device)
        h = model.encode_user(hist_t, user_t)
        h = F.normalize(h, dim=-1, eps=1e-8)
        reps.append(h.cpu().numpy())
    return np.concatenate(reps, axis=0)


#%%
# ----------------------------------------------------------------------
# sweep: [no-separation baseline] + lambda_cen grid
# ----------------------------------------------------------------------
settings = [(BASELINE_LABEL, None)] + [(rf"$\lambda_{{\mathrm{{cen}}}}={lam}$", lam) for lam in LAMBDA_GRID]

results = {}
for label, lam in settings:
    if lam is None:
        model = build_baseline_model()
        ckpt = torch.load(BASELINE_PATH, map_location=device)
    else:
        model = build_model()
        ckpt = torch.load(checkpoint_path(lam), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    reps = extract_representations(model)

    # Uncentered SVD (NOT covariance-PCA): a shared direction c_t absorbed by
    # every user's representation shows up as the dominant *singular*
    # direction only if we do not subtract the mean first. Centering would
    # remove exactly the shared component Theorem 1 is about, which is why
    # the earlier mean-subtracted-PCA version of this plot showed no signal.
    svd = TruncatedSVD(n_components=2)
    xy = svd.fit_transform(reps)  # raw projection, norm <= 1 since reps are unit vectors
    radii = np.linalg.norm(xy, axis=1)
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    resultant_R = np.abs(np.mean(np.exp(1j * angles)))  # 1 = all one angle, 0 = uniform on circle

    # pairwise-distance / uniformity diagnostics (Theorem 1 / Corollary 1 LHS),
    # computed on the full normalized 128-d representations, not the 2D
    # projection used only for the plot
    n = reps.shape[0]
    pair_idx = rng.choice(n, size=(min(20000, n * (n - 1) // 2), 2))
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    d2 = np.sum((reps[pair_idx[:, 0]] - reps[pair_idx[:, 1]]) ** 2, axis=1)
    pair_dist = d2.mean()
    l_uni = np.log(np.mean(np.exp(-2.0 * d2)))

    results[label] = dict(xy=xy, radii=radii, angles=angles, pair_dist=pair_dist, l_uni=l_uni,
                           resultant_R=resultant_R, explained_var=svd.explained_variance_ratio_.sum())
    print(f"[{label.strip()!s:>28}] PairDist={pair_dist:.4f}  L_uni={l_uni:.4f}  "
          f"mean_radius={radii.mean():.3f}  resultant_R={resultant_R:.4f}  "
          f"top-2-SVD explained var={results[label]['explained_var']:.3f}")


#%%
# ----------------------------------------------------------------------
# 2D KDE over the unit disc (top-2 raw SVD plane) + plot
# ----------------------------------------------------------------------
grid_lin = np.linspace(-1.05, 1.05, 220)
GX, GY = np.meshgrid(grid_lin, grid_lin)
disc_mask = (GX ** 2 + GY ** 2) <= 1.0
cmap = plt.get_cmap("YlOrBr")  # low density -> pale, high density -> dark (NOT reversed)

fig, axes = plt.subplots(1, len(settings), figsize=(3.1 * len(settings), 3.6),
                          subplot_kw=dict(aspect="equal"))

for ax, (label, lam) in zip(axes, settings):
    xy = results[label]["xy"]

    kde = gaussian_kde(xy.T)
    dens = np.full(GX.shape, np.nan)
    dens[disc_mask] = kde(np.vstack([GX[disc_mask], GY[disc_mask]]))
    dens_masked = np.ma.masked_invalid(dens)
    panel_norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=np.nanmax(dens))

    ax.pcolormesh(GX, GY, dens_masked, cmap=cmap, norm=panel_norm, shading="auto")
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="black", alpha=0.25, linewidths=0)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="0.4", linewidth=1.0))

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        label + "\n" +
        rf"$\mathcal{{L}}_{{uni}}={results[label]['l_uni']:.2f}$, $R={results[label]['resultant_R']:.2f}$",
        fontsize=11,
    )

fig.suptitle(
    "SASRec on Micro-Video: user-context representations $h_u(t)$ projected onto their\n"
    "top-2 (uncentered) singular directions — a shared popularity direction absorbed into\n"
    r"$f_\Psi$ pulls points toward one corner of the disc; $R$ = mean resultant length (1 = collapsed, 0 = uniform)",
    y=1.12, fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"[saved] {OUT_PATH}")
