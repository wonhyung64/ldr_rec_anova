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
    wandb_var = wandb.init(project="ldr_rec3", config=vars(args))
    wandb.run.name = args.expt_name

# args.model_name = "bsarec"
# args.dataset = "kuairand"
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

if args.dr_anchor != "item":
    snapshot = make_prior_snapshot(model)
    hot_negs = sample_epoch_negatives(
        snapshot=snapshot,
        train_events=dataset.train_hot_events,
        num_items=dataset.m_item,
        num_negatives=args.contrast_size-1,
    )

if args.dr_anchor != "user":
    dataset.prepare_user_timebucket_sampler()
    dataset.get_pair_user_event_timebucket_fast()


epoch = 0

save_dir = Path(args.save_path)
pattern = f"_{args.model_name}_lambda{args.lambda1}_e???_seed{args.seed}_ablation{args.ablation}.pt"
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

        if args.dr_anchor != "item":
            hot_neg_item = torch.tensor(hot_negs[hot_sample_idx], dtype=torch.long, device=args.device)
            pos_score = score_pair(model, hot_pos_item, anchor_hist_items, hot_anchor_user)
            neg_score = score_pair(model, hot_neg_item, anchor_hist_items, hot_anchor_user)
            user_loss += -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean() * args.lambda1

        if args.dr_anchor != "user":
            neg_hist_items_np = dataset.get_histories_for_users_at_times(
                dataset.hot_neg_user_list[hot_sample_idx, 0],
                dataset.hot_event_time_list[hot_sample_idx],
                max_seq_len=args.max_seq_len,
            )
            neg_hist_items = torch.tensor(neg_hist_items_np, device=args.device)

            pos_score = score_pair(model, hot_pos_item, anchor_hist_items, hot_anchor_user)
            neg_score = score_pair(model, hot_pos_item, neg_hist_items, hot_anchor_user)
            user_loss += -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean()

            dataset.get_pair_user_event_timebucket_fast()

        epoch_user_loss += user_loss.item()


        """ITEM"""
        cold_sample_idx = cold_idxs[cold_mini_batch*idx : (idx + 1)*cold_mini_batch]
        cold_pos_item = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        pos_item = torch.cat([cold_pos_item, hot_pos_item], dim=0)

        hot_neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item = torch.tensor(dataset.cold_neg_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        neg_item = torch.cat([cold_neg_item, hot_neg_item], dim=0)

        hot_pos_time = torch.Tensor(dataset.hot_event_time_list[hot_sample_idx]).reshape(-1, 1, 1).to(args.device)
        cold_pos_time = torch.Tensor(dataset.cold_event_time_list[cold_sample_idx]).reshape(-1, 1, 1).to(args.device)
        pos_time = torch.cat([cold_pos_time, hot_pos_time], dim=0)

        hot_pos_time_all = torch.Tensor(dataset.hot_pos_time_all[hot_sample_idx]).to(args.device)
        cold_pos_time_all = torch.Tensor(dataset.cold_pos_time_all[cold_sample_idx]).to(args.device)
        pos_time_all = torch.cat([cold_pos_time_all, hot_pos_time_all], dim=0)

        hot_neg_time_all = torch.Tensor(dataset.hot_neg_time_all[hot_sample_idx]).to(args.device)
        cold_neg_time_all = torch.Tensor(dataset.cold_neg_time_all[cold_sample_idx]).to(args.device)
        neg_time_all = torch.cat([cold_neg_time_all, hot_neg_time_all], dim=0)

        batch_items = torch.concat([pos_item.unsqueeze(-1), neg_item], -1).reshape(pos_item.shape[0], -1)
        batch_time_all = torch.concat([pos_time_all.unsqueeze(1), neg_time_all], 1)

        logits = model.prior(batch_items, pos_time, batch_time_all)
        log_logits = torch.log(logits + 1e-9)
        item_loss = -nn.functional.log_softmax(log_logits, dim=-1)[:, 0].mean() * (1-args.lambda1)
        epoch_item_loss += item_loss.item()

        dataset.get_pair_item_uniform(k=args.contrast_size-1, w_time=True)


        total_loss = item_loss + user_loss
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
        }, f"{args.save_path}/_{args.model_name}_lambda{args.lambda1}_e{epoch}_seed{args.seed}_ablation{args.ablation}.pt")


    if epoch % args.pair_reset_interval == 0:
        if args.dr_anchor != "item":
            print("Reset negative users")
            snapshot = make_prior_snapshot(model)
            hot_negs = sample_epoch_negatives(
                snapshot=snapshot,
                train_events=dataset.train_hot_events,
                num_items=dataset.m_item,
                num_negatives=args.contrast_size-1,
            )



if epoch % args.evaluate_interval == 0:
    pred_list = []
    gt_list = []

    model.eval()
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()

    for (user, item), pos_time_val in dataset.valid_user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            resid = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

        pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)

        item_logits_list = []
        for idx2 in range(dataset.m_item // args.batch_size + 1):
            item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
            if len(item_idx) == 0:
                continue

            batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
            batch_time_mask = batch_time_all < pos_time_t
            batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)

            with torch.no_grad():
                time_intensity = (torch.exp(-beta * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                logits = (mu[item_idx] + alpha[item_idx] * time_intensity.squeeze(-1)).flatten()
            item_logits_list.append(logits)

        item_logits = torch.concat(item_logits_list)
        item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)

        pred = (item_log_prob.cpu() + resid.cpu()).cpu()

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
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()

    for (user, item), pos_time_val in dataset.test_user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            resid = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

        pos_time_t = torch.tensor([pos_time_val], dtype=torch.float32).to(args.device)

        item_logits_list = []
        for idx2 in range(dataset.m_item // args.batch_size + 1):
            item_idx = all_item_idxs[idx2 * args.batch_size: (idx2 + 1) * args.batch_size]
            if len(item_idx) == 0:
                continue

            batch_time_all = torch.tensor(dataset.item_time_array[item_idx], dtype=torch.float32).to(args.device)
            batch_time_mask = batch_time_all < pos_time_t
            batch_time_delta = (pos_time_t - batch_time_all).clamp(min=0.0)

            with torch.no_grad():
                time_intensity = (torch.exp(-beta * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                logits = (mu[item_idx] + alpha[item_idx] * time_intensity.squeeze(-1)).flatten()
            item_logits_list.append(logits)

        item_logits = torch.concat(item_logits_list)
        item_log_prob = torch.log(item_logits + 1e-12) - torch.log(item_logits.sum() + 1e-12)

        pred = (item_log_prob.cpu() + resid.cpu()).cpu()

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
