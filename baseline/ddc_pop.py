#%%
# DDC: Disentangled Debiased CF via Analytical Vector Decomposition
# "Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition"
import re
import os
import wandb
import torch
import inspect
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset_pop import UserItemTime
from module.model import MODEL_REGISTRY


def get_epoch(path):
    match = re.search(r"_e(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def compute_popularity_directions(item_emb_weight, pop_norm, k):
    """Analytically find k popularity directions via sequential deflation."""
    # item_emb_weight: [m_item, d], pop_norm: [m_item]
    directions = []
    residual = item_emb_weight.detach().clone()
    for _ in range(k):
        d = residual.T @ pop_norm  # [d]
        d = F.normalize(d, dim=0, eps=1e-8)
        directions.append(d)
        # deflate: remove this direction from the embedding matrix
        residual = residual - (residual @ d).unsqueeze(-1) * d.unsqueeze(0)
    return directions  # list of [d] tensors


def ddc_score_all(model, hist_item_idx, user_idx, pop_directions):
    """DDC debiased score: project out popularity directions from item embeddings."""
    u = model.encode_user(hist_item_idx, user_idx)  # [B, d]
    v_all = model.item_embedding.weight[:model.num_items]  # [m_item, d]

    # Sequentially remove each popularity direction
    v_int = v_all
    for d in pop_directions:
        v_int = v_int - (v_int @ d).unsqueeze(-1) * d.unsqueeze(0)

    u = F.normalize(u, dim=-1, eps=1e-8)
    v_int = F.normalize(v_int, dim=-1, eps=1e-8)
    return torch.matmul(u, v_int.T) / model.tau  # [B, m_item]


#%%
args = parse_args()
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)

wandb_login = False
file_dir = inspect.getfile(inspect.currentframe())
file_name = file_dir.split("/")[-1]
if file_name.endswith(".py"):
    try:
        wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
    except Exception:
        wandb_login = False

if wandb_login:
    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    args.expt_name = f"{file_name.split('.')[-2]}_{args.model_name}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec_pop", config=vars(args))
    wandb.run.name = args.expt_name


#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

all_item_idxs = np.arange(dataset.m_item)

# Compute log-normalized item popularity from training interactions
item_pop = np.bincount(dataset.trainItem, minlength=dataset.m_item).astype(np.float32)
item_pop_log = np.log1p(item_pop)
item_pop_norm = torch.tensor(
    item_pop_log / (item_pop_log.sum() + 1e-12), dtype=torch.float32, device=args.device
)


#%%
model_name = getattr(args, "model_name", "mf").lower()
if model_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
model_class = MODEL_REGISTRY[model_name]

model = model_class(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    tau=args.tau,
    depth=args.depth,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    dropout=args.dropout,
).to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.decay,
)


#%%
dataset.get_pair_item_uniform(k=args.contrast_size - 1)

epoch = 0

save_dir = Path(args.save_path)
save_prefix = f"_ddc_{args.model_name}_lr{args.lr}_decay{args.decay}_kpop{args.k_pop}"
pattern = f"{save_prefix}_e???_seed{args.seed}.pt"
matched_files = sorted(save_dir.glob(pattern))
if len(matched_files) > 0:
    recent_file = max(matched_files, key=get_epoch)
    checkpoint = torch.load(recent_file, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print("MODEL LOADED!")
else:
    raise ValueError

#%%
# Test

eval_datasets = [
    ("head_overall", dataset.test_head_overall_dict),
    ("head_recent_3d", dataset.test_head_recent_3d_dict),
    ("head_recent_7d", dataset.test_head_recent_7d_dict),
    ("tail_overall", dataset.test_tail_overall_dict),
    ("tail_recent_3d", dataset.test_tail_recent_3d_dict),
    ("tail_recent_7d", dataset.test_tail_recent_7d_dict),
]

with torch.no_grad():
    v_emb = model.item_embedding.weight[:model.num_items]
    pop_directions = compute_popularity_directions(v_emb, item_pop_norm, args.k_pop)

for (split_name, data_split) in eval_datasets:

    pred_list, gt_list = [], []
    model.eval()

    for (user, item), pos_time_val in dataset.set_to_pair(data_split, dataset.time_dict, dataset.time_unit).items():
        hist_item_np, _ = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            pred = ddc_score_all(model, hist_item_t, user_t, pop_directions).squeeze(0).cpu()

        exclude_items = list(dataset._allPos[user])
        pred[exclude_items] = -9999
        _, pred_k = torch.topk(pred, k=max(args.topks))
        pred_list.append(pred_k.cpu())
        gt_list.append([item])

    test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

    if wandb_login:
        wandb_var.log(dict(zip([f"test_{split_name}_precision_{k}_{epoch}" for k in args.topks], test_results[0])))
        wandb_var.log(dict(zip([f"test_{split_name}_recall_{k}_{epoch}" for k in args.topks], test_results[1])))
        wandb_var.log(dict(zip([f"test_{split_name}_ndcg_{k}_{epoch}" for k in args.topks], test_results[2])))
        wandb_var.log(dict(zip([f"test_{split_name}_mrr_{k}_{epoch}" for k in args.topks], test_results[3])))

if wandb_login:
    wandb_var.finish()

# %%
