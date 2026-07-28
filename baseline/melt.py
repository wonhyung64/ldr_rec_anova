#%%
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
from module.melt_model import MELT


def get_epoch(path):
    match = re.search(r"_e(\d+)_", path.name)
    return int(match.group(1)) if match else -1


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
batch_num = dataset.hotDataSize // mini_batch + 1

hot_idxs = np.arange(dataset.hotDataSize)

all_item_idxs = np.arange(dataset.m_item)


#%% Build tail user/item masks from training interaction counts
item_counts = np.zeros(dataset.m_item, dtype=np.int64)
for u, item_list in dataset.train_dict.items():
    for item in item_list:
        item_counts[item] += 1

user_counts = np.array(
    [len(dataset.train_dict.get(u, [])) for u in range(dataset.n_user)],
    dtype=np.int64,
)

item_median = np.median(item_counts[item_counts > 0])
user_median = np.median(user_counts[user_counts > 0])

tail_item_ids = np.where(item_counts <= item_median)[0]
tail_user_ids = np.where(user_counts <= user_median)[0]


#%% Build padded item→user interaction table (capped at MAX_U per item)
MAX_U = 50
item_user_dict = {}
for u, item_list in dataset.train_dict.items():
    for item in item_list:
        if item not in item_user_dict:
            item_user_dict[item] = []
        item_user_dict[item].append(u)

item_user_pad = np.full((dataset.m_item, MAX_U), dataset.n_user, dtype=np.int64)
item_user_mask = np.zeros((dataset.m_item, MAX_U), dtype=bool)
rng = np.random.default_rng(args.seed)
for i in range(dataset.m_item):
    users = item_user_dict.get(i, [])
    if len(users) > MAX_U:
        users = rng.choice(users, MAX_U, replace=False).tolist()
    n = len(users)
    if n > 0:
        item_user_pad[i, :n] = users[:n]
        item_user_mask[i, :n] = True


#%% Build model
model = MELT(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    tau=args.tau,
    depth=args.depth,
    max_seq_len=args.max_seq_len,
    n_heads=args.n_heads,
    dropout=args.dropout,
    melt_alpha=args.melt_alpha,
    n_proto=args.n_proto,
    tail_user_ids=tail_user_ids,
    tail_item_ids=tail_item_ids,
    item_user_pad=item_user_pad,
    item_user_mask=item_user_mask,
).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

dataset.get_pair_item_uniform(k=args.contrast_size - 1)


#%% Load checkpoint if exists
epoch = 0
save_dir = Path(args.save_path)
save_prefix = f"_melt_lr{args.lr}_alpha{args.melt_alpha}_nproto{args.n_proto}"
pattern = f"{save_prefix}_e*_seed{args.seed}.pt"
matched_files = sorted(save_dir.glob(pattern))
if len(matched_files) > 0:
    recent_file = max(matched_files, key=get_epoch)
    checkpoint = torch.load(recent_file, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print("MODEL LOADED!")


#%% Training loop
while epoch < args.epochs:
    epoch += 1
    torch.cuda.empty_cache()
    model.train()
    model.clear_item_cache()
    np.random.shuffle(hot_idxs)
    epoch_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[mini_batch * idx: (idx + 1) * mini_batch]

        anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        pos_score = model.residual_score(pos_item, anchor_hist_items, anchor_user)
        neg_score = model.residual_score(neg_item, anchor_hist_items, anchor_user)

        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] {epoch_loss / batch_num:.4f}")

    if epoch % args.pair_reset_interval == 0:
        dataset.get_pair_item_uniform(k=args.contrast_size - 1)

    if epoch % 100 == 0:
        save_name = f"{save_prefix}_e{epoch}_seed{args.seed}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        }, f"{args.save_path}/{save_name}")

        # Delete previous epoch checkpoints
        for prev_f in save_dir.glob(pattern):
            if get_epoch(prev_f) < epoch:
                prev_f.unlink()


if epoch % args.evaluate_interval == 0:
    pred_list, gt_list = [], []
    model.eval()
    model.precompute_item_enhancements()

    for (user, item), pos_time_val in dataset.valid_user_item_time.items():
        hist_item_np, _ = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            pred = model.score_all_items(hist_item_t, user_t).squeeze(0).cpu()

        exclude_items = list(dataset._allPos[user])
        pred[exclude_items] = -9999
        _, pred_k = torch.topk(pred, k=max(args.topks))
        pred_list.append(pred_k.cpu())
        gt_list.append([item])

    valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

    if wandb_login:
        wandb_var.log(dict(zip([f"valid_precision_{k}_{epoch}" for k in args.topks], valid_results[0])))
        wandb_var.log(dict(zip([f"valid_recall_{k}_{epoch}" for k in args.topks], valid_results[1])))
        wandb_var.log(dict(zip([f"valid_ndcg_{k}_{epoch}" for k in args.topks], valid_results[2])))
        wandb_var.log(dict(zip([f"valid_mrr_{k}_{epoch}" for k in args.topks], valid_results[3])))


    pred_list, gt_list = [], []
    model.eval()
    model.precompute_item_enhancements()

    for (user, item), pos_time_val in dataset.test_user_item_time.items():
        hist_item_np, _ = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            pred = model.score_all_items(hist_item_t, user_t).squeeze(0).cpu()

        exclude_items = list(dataset._allPos[user])
        pred[exclude_items] = -9999
        _, pred_k = torch.topk(pred, k=max(args.topks))
        pred_list.append(pred_k.cpu())
        gt_list.append([item])

    test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

    if wandb_login:
        wandb_var.log(dict(zip([f"test_precision_{k}_{epoch}" for k in args.topks], test_results[0])))
        wandb_var.log(dict(zip([f"test_recall_{k}_{epoch}" for k in args.topks], test_results[1])))
        wandb_var.log(dict(zip([f"test_ndcg_{k}_{epoch}" for k in args.topks], test_results[2])))
        wandb_var.log(dict(zip([f"test_mrr_{k}_{epoch}" for k in args.topks], test_results[3])))


if wandb_login:
    wandb_var.finish()

# %%
