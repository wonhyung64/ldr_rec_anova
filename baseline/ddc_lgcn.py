#%%
# DDC-LightGCN: DDC with LightGCN backbone
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
from module.dataset import UserItemTime
from module.ddc_lgcn_model import DDCLightGCN, build_norm_adj


def get_epoch(path):
    match = re.search(r"_e(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def compute_popularity_directions(item_emb, pop_norm, k):
    """Analytically find k popularity directions via sequential deflation."""
    directions = []
    residual = item_emb.detach().clone()
    for _ in range(k):
        d = residual.T @ pop_norm  # [d]
        d = F.normalize(d, dim=0, eps=1e-8)
        directions.append(d)
        residual = residual - (residual @ d).unsqueeze(-1) * d.unsqueeze(0)
    return directions


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
    args.expt_name = f"{file_name.split('.')[-2]}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec4", config=vars(args))
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

# Build normalised adjacency from all training interactions
all_users = np.array([u for (u, v, t) in dataset.train_events])
all_items = np.array([v for (u, v, t) in dataset.train_events])
adj = build_norm_adj(all_users, all_items, dataset.n_user, dataset.m_item, args.device)

print(f"Graph edges: {len(all_users) * 2}")

# Log-normalised item popularity for DDC decomposition
item_pop = np.bincount(dataset.trainItem, minlength=dataset.m_item).astype(np.float32)
item_pop_log = np.log1p(item_pop)
item_pop_norm = torch.tensor(
    item_pop_log / (item_pop_log.sum() + 1e-12), dtype=torch.float32, device=args.device
)


#%%
model = DDCLightGCN(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    n_layers=args.n_layers,
    tau=args.tau,
    adj=adj,
).to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.decay,
)


#%%
dataset.get_pair_item_uniform(k=args.contrast_size - 1, w_time=False)

epoch = 0

save_dir = Path(args.save_path)
save_prefix = f"_ddc_lgcn_lr{args.lr}_decay{args.decay}_nlayers{args.n_layers}_kpop{args.k_pop}"
pattern = f"{save_prefix}_e???_seed{args.seed}.pt"
matched_files = sorted(save_dir.glob(pattern))
if len(matched_files) > 0:
    recent_file = max(matched_files, key=get_epoch)
    checkpoint = torch.load(recent_file, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print("MODEL LOADED!")


#%%
while epoch < args.epochs:
    epoch += 1
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx  = hot_idxs[hot_mini_batch  * idx: (idx + 1) * hot_mini_batch]
        cold_sample_idx = cold_idxs[cold_mini_batch * idx: (idx + 1) * cold_mini_batch]

        hot_anchor_user  = torch.tensor(dataset.hot_user_list[hot_sample_idx],    dtype=torch.long, device=args.device)
        hot_pos_item     = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_neg_item     = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        cold_anchor_user = torch.tensor(dataset.cold_user_list[cold_sample_idx],    dtype=torch.long, device=args.device)
        cold_pos_item    = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item    = torch.tensor(dataset.cold_neg_item_list[cold_sample_idx], dtype=torch.long, device=args.device)

        anchor_user = torch.cat([cold_anchor_user, hot_anchor_user], dim=0)
        pos_item    = torch.cat([cold_pos_item,    hot_pos_item],    dim=0)
        neg_item    = torch.cat([cold_neg_item,    hot_neg_item],    dim=0)

        # BPR loss (standard LightGCN training)
        pos_score = model.bpr_score(anchor_user, pos_item)          # [B]
        neg_score = model.bpr_score(anchor_user, neg_item)          # [B, N]

        loss = -(F.logsigmoid(pos_score.unsqueeze(-1) - neg_score)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        dataset.get_pair_item_uniform(k=args.contrast_size - 1, w_time=False)

    print(f"[Epoch {epoch:>4d} Train Loss] {epoch_loss / batch_num:.4f}")

    if epoch % 100 == 0:
        save_name = f"{save_prefix}_e{epoch:03d}_seed{args.seed}.pt"
        for old_file in save_dir.glob(f"{save_prefix}_e???_seed{args.seed}.pt"):
            if old_file.name != save_name:
                old_file.unlink()
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }, f"{args.save_path}/{save_name}")

    if epoch % args.evaluate_interval == 0:
        model.eval()
        with torch.no_grad():
            _, v_emb = model.propagate()
            pop_directions = compute_popularity_directions(v_emb, item_pop_norm, args.k_pop)

        for flag, user_item_time in [("valid", dataset.valid_user_item_time),
                                      ("test",  dataset.test_user_item_time)]:
            pred_list, gt_list = [], []

            for (user, item), _ in user_item_time.items():
                user_t = torch.tensor([user], dtype=torch.long, device=args.device)

                with torch.no_grad():
                    scores = model.ddc_score_all(user_t, pop_directions).squeeze(0).cpu()

                exclude_items = list(dataset._allPos[user])
                scores[exclude_items] = -9999
                _, pred_k = torch.topk(scores, k=max(args.topks))
                pred_list.append(pred_k.cpu())
                gt_list.append([item])

            results = computeTopNAccuracy(gt_list, pred_list, args.topks)

            print(
                f"[Epoch {epoch} {flag}] "
                f"Recall@{args.topks}: {results[1]} | "
                f"NDCG@{args.topks}: {results[2]}"
            )

            if wandb_login:
                wandb_var.log(dict(zip([f"{flag}_precision_{k}_{epoch}" for k in args.topks], results[0])))
                wandb_var.log(dict(zip([f"{flag}_recall_{k}_{epoch}"    for k in args.topks], results[1])))
                wandb_var.log(dict(zip([f"{flag}_ndcg_{k}_{epoch}"      for k in args.topks], results[2])))
                wandb_var.log(dict(zip([f"{flag}_mrr_{k}_{epoch}"       for k in args.topks], results[3])))

if wandb_login:
    wandb_var.finish()

# %%
