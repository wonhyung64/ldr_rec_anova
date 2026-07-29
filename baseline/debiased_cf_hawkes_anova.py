#%%
import re
import os
import wandb
import torch
import inspect
import numpy as np
from datetime import datetime
from pathlib import Path

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset import UserItemTime
from module.model import score_all, MODEL_REGISTRY
from module.debias import build_debias_model, build_unshared_debias_model, build_linear_debias_model, build_unshared_linear_debias_model
from module.hawkes_choice import importance_corrected_choice_loss, anova_centering_penalty


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
    wandb_var = wandb.init(project="anova_hawkes", config=vars(args))
    wandb.run.name = args.expt_name

#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)
dataset.item_time_array = dataset.item_time_array / 86400.0

# CF backbones (MF/NCF) have no sequence dependence, so unlike the sequential
# scripts, cold events (first interaction per user) carry just as much signal
# as hot ones and are used here too.
mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

num_negatives = args.contrast_size - 1

# ResidualBase.encode_user ignores history for this backbone family (MF/NCF
# only look up user_idx), so a constant all-padding placeholder keeps the
# score_pair/score_all API shape uniform across backbones without spending
# time building histories that are never read.
dummy_hist = torch.full((mini_batch, args.max_seq_len), dataset.m_item, dtype=torch.long, device=args.device)

#%%
model_name = getattr(args, "model_name", "mf").lower()
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
    score_norm=args.score_norm,
    ).to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.decay,
)

#%%
dataset.get_pair_item_uniform(k=num_negatives, w_time=True)
dataset.prepare_user_timebucket_sampler(bucket_size=args.user_bucket_size, w_cold=True)
dataset.get_pair_user_event_timebucket_fast(w_cold=True)


epoch = 0

save_dir = Path(args.save_path)
pattern = f"_{args.model_name}_lambdacen{args.lambda_cen}_tau{args.tau}_scorenorm{args.score_norm}_e???_seed{args.seed}_ablation{args.ablation}_cfhawkesanova.pt"
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
    epoch_choice_loss = 0.0
    epoch_center_loss = 0.0


    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[hot_mini_batch*idx : (idx + 1)*hot_mini_batch]
        cold_sample_idx = cold_idxs[cold_mini_batch*idx : (idx + 1)*cold_mini_batch]

        """CANDIDATE SET: observed item (col 0, hot+cold combined) + uniform negatives"""
        anchor_user = torch.tensor(
            np.concatenate([dataset.cold_user_list[cold_sample_idx], dataset.hot_user_list[hot_sample_idx]]),
            dtype=torch.long, device=args.device,
        )
        pos_item = torch.tensor(
            np.concatenate([dataset.cold_pos_item_list[cold_sample_idx], dataset.hot_pos_item_list[hot_sample_idx]]),
            dtype=torch.long, device=args.device,
        )
        neg_item = torch.tensor(
            np.concatenate([dataset.cold_neg_item_list[cold_sample_idx], dataset.hot_neg_item_list[hot_sample_idx]]),
            dtype=torch.long, device=args.device,
        )
        pos_time = torch.tensor(
            np.concatenate([dataset.cold_event_time_list[cold_sample_idx], dataset.hot_event_time_list[hot_sample_idx]]),
            dtype=torch.float32, device=args.device,
        )
        pos_time_all = torch.tensor(
            np.concatenate([dataset.cold_pos_time_all[cold_sample_idx], dataset.hot_pos_time_all[hot_sample_idx]]),
            dtype=torch.float32, device=args.device,
        )
        neg_time_all = torch.tensor(
            np.concatenate([dataset.cold_neg_time_all[cold_sample_idx], dataset.hot_neg_time_all[hot_sample_idx]]),
            dtype=torch.float32, device=args.device,
        )

        candidate_items = torch.cat([pos_item.unsqueeze(-1), neg_item], dim=-1)
        candidate_time_all = torch.cat([pos_time_all.unsqueeze(1), neg_time_all], dim=1)
        batch_hist = dummy_hist[:anchor_user.shape[0]]

        """CHOICE LOSS: l_choice(Phi, Psi)"""
        choice_loss = importance_corrected_choice_loss(
            model, candidate_items, batch_hist, anchor_user,
            pos_time, candidate_time_all, dataset.m_item, num_negatives,
        )

        """ANOVA-CENTERING PENALTY: R_cen(Psi), estimated from two independently
        sampled users active near each event's time bucket"""
        dataset.get_pair_user_event_timebucket_fast(w_cold=True)
        center_user_a = torch.tensor(
            np.concatenate([dataset.cold_neg_user_list[cold_sample_idx, 0], dataset.hot_neg_user_list[hot_sample_idx, 0]]),
            dtype=torch.long, device=args.device,
        )
        dataset.get_pair_user_event_timebucket_fast(w_cold=True)
        center_user_b = torch.tensor(
            np.concatenate([dataset.cold_neg_user_list[cold_sample_idx, 0], dataset.hot_neg_user_list[hot_sample_idx, 0]]),
            dtype=torch.long, device=args.device,
        )

        center_loss = anova_centering_penalty(
            model, candidate_items, batch_hist, center_user_a, batch_hist, center_user_b,
        )

        total_loss = choice_loss + args.lambda_cen * center_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_choice_loss += choice_loss.item()
        epoch_center_loss += center_loss.item()

        dataset.get_pair_item_uniform(k=num_negatives, w_time=True)

    print(f"[Epoch {epoch:>4d} Train Loss] choice: {epoch_choice_loss / batch_num:.4f} / center: {epoch_center_loss / batch_num:.4f}")

    if wandb_login:
        wandb_var.log({
            "train_choice_loss": epoch_choice_loss / batch_num,
            "train_center_loss": epoch_center_loss / batch_num,
        })

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"{args.save_path}/_{args.model_name}_lambdacen{args.lambda_cen}_tau{args.tau}_scorenorm{args.score_norm}_e{epoch}_seed{args.seed}_ablation{args.ablation}_cfhawkesanova.pt")


all_item_idxs = np.arange(dataset.m_item)

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
            resid = score_all(model, hist_item_t, user_t).squeeze(0)

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
    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()

    for (user, item), pos_time_val in dataset.test_user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            resid = score_all(model, hist_item_t, user_t).squeeze(0)

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


if args.debiased_eval == "true":
    eval_datasets = [
        ("head_overall", dataset.test_head_overall_dict),
        ("head_recent_3d", dataset.test_head_recent_3d_dict),
        ("head_recent_7d", dataset.test_head_recent_7d_dict),
        ("tail_overall", dataset.test_tail_overall_dict),
        ("tail_recent_3d", dataset.test_tail_recent_3d_dict),
        ("tail_recent_7d", dataset.test_tail_recent_7d_dict),
    ]

    with torch.no_grad():
        mu, alpha, beta = model.prior_parameters_from_embeddings()


    for (split_name, data_split) in eval_datasets:
        pred_list, gt_list = [], []
        model.eval()

        for (user, item), pos_time_val in dataset.set_to_pair(data_split, dataset.time_dict, dataset.time_unit).items():
            hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
            hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
            user_t = torch.tensor([user], dtype=torch.long, device=args.device)

            with torch.no_grad():
                resid = score_all(model, hist_item_t, user_t).squeeze(0)

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

            pred = (item_log_prob * args.alpha1 + resid * (1-args.alpha1)).cpu()

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
