#%%
import os
import re
import math
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from torch import optim
from datetime import datetime

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset import UserItemTime
from module.model import build_model, score_pair, score_all
from module.bsarec import BSARec


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
    args.expt_name = f"{file_name.split('.')[-2]}_{args.model_name}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec_norm_backbone", config=vars(args))
    # wandb_var = wandb.init(project="ldr_rec_backbone", config=vars(args))
    wandb.run.name = args.expt_name


#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

all_item_idxs = np.arange(dataset.m_item)


#%%
if args.model_name == "bsarec":
    model = BSARec(
        num_users=dataset.n_user,
        num_items=dataset.m_item,
        embedding_k=args.recdim,
        device=args.device,
        tau=args.tau,
        depth=args.depth,
        max_seq_len=args.max_seq_len,
        n_heads=args.n_heads,
        dropout=args.dropout,
        c=args.c,
        alpha=args.alpha,
        ).to(args.device)
else:
    model = build_model(args, dataset, mini_batch)


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


#%%
epoch = 0


save_dir = Path(args.save_path)
pattern = f"norm_backbone_{args.model_name}_e???_seed{args.seed}.pt"
# pattern = f"_backbone_{args.model_name}_e???_seed{args.seed}.pt"
matched_files = sorted(save_dir.glob(pattern))
if len(matched_files) > 0:
    recent_file = max(matched_files, key=get_epoch)
    checkpoint = torch.load(recent_file, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print("MODEL LOADED!")



while epoch < args.epochs: 
    epoch += 1 
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_user_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]
        anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_times = torch.tensor(dataset.train_hist_time_list[hot_sample_idx], dtype=torch.long, device=args.device)

        pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
        neg_score = score_pair(model, neg_item, anchor_hist_items, anchor_user)
        # u = model.encode_user(anchor_hist_items, anchor_user)
        # mini_batch, recdim = u.shape
        # v = model.get_item_repr(pos_item).reshape(mini_batch, -1, recdim)
        # pos_score = torch.sum(u.unsqueeze(1) * v, dim=-1)
        # v = model.get_item_repr(neg_item).reshape(mini_batch, -1, recdim)
        # neg_score = torch.sum(u.unsqueeze(1) * v, dim=-1)

        user_loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).sum()
        optimizer.zero_grad()
        user_loss.backward()
        optimizer.step()

        epoch_user_loss += user_loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] ldr: {epoch_user_loss / batch_num:.4f}")

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_user_loss,
        }, f"{args.save_path}/norm_backbone_{args.model_name}_e{epoch}_seed{args.seed}.pt")
        # }, f"{args.save_path}/_backbone_{args.model_name}_e{epoch}_seed{args.seed}.pt")



    if epoch % args.pair_reset_interval == 0:
        print("Reset uniform negative users")
        dataset.get_pair_item_uniform(k=args.contrast_size-1)

    if epoch % args.evaluate_interval == 0:
        pred_list = []
        gt_list = []

        model.eval()
        for (user, item), pos_time_val in dataset.valid_user_item_time.items():
            hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
            hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
            user_t = torch.tensor([user], dtype=torch.long, device=args.device)

            with torch.no_grad():
                pred = score_all(model, hist_item_t, user_t).squeeze(0).cpu()
                # u = model.encode_user(hist_item_t, user_t)
                # v_all = model.get_item_repr(torch.arange(model.num_items, device=hist_item_t.device))
                # pred = torch.matmul(u, v_all.T).squeeze(0).cpu()

            exclude_items = list(dataset._allPos[user])
            pred[exclude_items] = -9999
            _, pred_k = torch.topk(pred, k=max(args.topks))
            pred_list.append(pred_k.cpu())
            gt_list.append([item])

        valid_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

        if wandb_login:
            wandb_var.log({
                "train_ldr": epoch_user_loss / batch_num,
            })
            wandb_var.log(dict(zip([f"valid_precision_{k}" for k in args.topks], valid_results[0])))
            wandb_var.log(dict(zip([f"valid_recall_{k}" for k in args.topks], valid_results[1])))
            wandb_var.log(dict(zip([f"valid_ndcg_{k}" for k in args.topks], valid_results[2])))
            wandb_var.log(dict(zip([f"valid_mrr_{k}" for k in args.topks], valid_results[3])))



pred_list = []
gt_list = []


model.eval()

for (user, item), pos_time_val in dataset.test_user_item_time.items():
    hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
    hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
    user_t = torch.tensor([user], dtype=torch.long, device=args.device)

    with torch.no_grad():
        pred = score_all(model, hist_item_t, user_t).squeeze(0).cpu()
        # u = model.encode_user(hist_item_t, user_t)
        # v_all = model.get_item_repr(torch.arange(model.num_items, device=hist_item_t.device))
        # pred = torch.matmul(u, v_all.T).squeeze(0).cpu()

    exclude_items = list(dataset._allPos[user])
    pred[exclude_items] = -9999
    _, pred_k = torch.topk(pred, k=max(args.topks))
    pred_list.append(pred_k.cpu())
    gt_list.append([item])

test_results = computeTopNAccuracy(gt_list, pred_list, args.topks)

if wandb_login:
    wandb_var.log(dict(zip([f"test_precision_{k}" for k in args.topks], test_results[0])))
    wandb_var.log(dict(zip([f"test_recall_{k}" for k in args.topks], test_results[1])))
    wandb_var.log(dict(zip([f"test_ndcg_{k}" for k in args.topks], test_results[2])))
    wandb_var.log(dict(zip([f"test_mrr_{k}" for k in args.topks], test_results[3])))
    wandb_var.finish()



#%%

