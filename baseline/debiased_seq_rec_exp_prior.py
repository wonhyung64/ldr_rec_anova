#%%
import re
import os
import math
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset import UserItemTime
from module.model import score_pair, score_all, MODEL_REGISTRY
from module.debias import build_debias_model, build_unshared_debias_model, build_linear_debias_model, build_unshared_linear_debias_model
from module.sampler import make_prior_snapshot, sample_epoch_negatives


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
    wandb_var = wandb.init(project="ldr_rec_prior", config=vars(args))
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

#%%
beta = getattr(args, 'beta', 3)


def compute_exp_prior(item_time_array, query_time, beta, eps=1e-9):
    valid = (item_time_array > 0) & (item_time_array < query_time)
    delta = np.where(valid, query_time - item_time_array, 0.0)
    weights = np.where(valid, np.exp(-beta * delta), 0.0)
    counts = weights.sum(axis=1) + eps
    return counts / counts.sum()


def sample_exp_prior_negatives(item_time_array, event_times, beta, num_negatives, eps=1e-9):
    negs = np.zeros((len(event_times), num_negatives), dtype=np.int64)
    for i, t in enumerate(event_times):
        prob = compute_exp_prior(item_time_array, t, beta, eps)
        negs[i] = np.random.choice(len(prob), size=num_negatives, p=prob)
    return negs


#%%
model_name = getattr(args, "model_name", "grurec").lower()
if model_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
model_class = MODEL_REGISTRY[model_name]

if args.ablation == "shared":
    debiased_class = build_unshared_debias_model(model_class)
elif args.ablation == "linear":
    debiased_class = build_linear_debias_model(model_class)
elif args.ablation == "none":
    debiased_class = build_debias_model(model_class)
elif args.ablation == "both":
    debiased_class = build_unshared_linear_debias_model(model_class)


if args.model_name == "bsarec":
    model = debiased_class(
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
    model = debiased_class(
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
dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)

item_hot_negs = sample_exp_prior_negatives(dataset.item_time_array, dataset.hot_event_time_list, beta, args.contrast_size-1)
item_cold_negs = sample_exp_prior_negatives(dataset.item_time_array, dataset.cold_event_time_list, beta, args.contrast_size-1)


epoch = 0

save_dir = Path(args.save_path)
pattern = f"{args.model_name}_lambda{args.lambda1}_e???_seed{args.seed}_exp_prior.pt"
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
    epoch_item_loss = 0.0


    for idx in range(batch_num):
        user_loss = torch.zeros(1).to(args.device)
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]

        """USER"""
        hot_anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        hot_neg_item = torch.tensor(item_hot_negs[hot_sample_idx], dtype=torch.long, device=args.device)
        pos_score = score_pair(model, hot_pos_item, anchor_hist_items, hot_anchor_user)
        neg_score = score_pair(model, hot_neg_item, anchor_hist_items, hot_anchor_user)
        user_loss += -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean()

        epoch_user_loss += user_loss.item()


        total_loss = user_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] user: {epoch_user_loss / batch_num:.4f} / item: {epoch_item_loss / batch_num:.4f}")

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_user_loss,
        }, f"{args.save_path}/{args.model_name}_lambda{args.lambda1}_e{epoch}_seed{args.seed}_exp_prior.pt")


    if epoch % args.pair_reset_interval == 0:
        print("Reset negative users")
        item_hot_negs = sample_exp_prior_negatives(dataset.item_time_array, dataset.hot_event_time_list, beta, args.contrast_size-1)
        item_cold_negs = sample_exp_prior_negatives(dataset.item_time_array, dataset.cold_event_time_list, beta, args.contrast_size-1)


if epoch % args.evaluate_interval == 0:
    pred_list = []
    gt_list = []

    model.eval()

    for (user, item), pos_time_val in dataset.valid_user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            resid = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

        window_prob = compute_exp_prior(dataset.item_time_array, pos_time_val, beta)
        item_log_prob = torch.tensor(np.log(window_prob), dtype=torch.float32)

        pred = (item_log_prob * args.alpha1 + resid * (1-args.alpha1)).cpu()

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


    pred_list = []
    gt_list = []

    model.eval()

    for (user, item), pos_time_val in dataset.test_user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            resid = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

        window_prob = compute_exp_prior(dataset.item_time_array, pos_time_val, beta)
        item_log_prob = torch.tensor(np.log(window_prob), dtype=torch.float32)

        pred = (item_log_prob * args.alpha1 + resid * (1-args.alpha1)).cpu()

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
