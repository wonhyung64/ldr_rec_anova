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
from module.debias_two_timescale_fixedlong import build_two_timescale_fixed_long_debias_model
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

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.hotDataSize // mini_batch + 1
hot_idxs = np.arange(dataset.hotDataSize)

num_negatives = args.contrast_size - 1

#%%
# Second iteration of the two-timescale Hawkes prior: beta_long is now a
# FIXED buffer (not learnable) instead of drifting toward beta_short over
# training, and alpha_long_mlp's init bias is loosened from -6 to -2 so it
# isn't starved of gradient from the start. See module.debias_two_timescale_fixedlong
# for the full rationale. mu_v is unchanged, and (as with the prior two
# Hawkes variants) there is no --ablation branching here.
model_name = getattr(args, "model_name", "sasrec").lower()
if model_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
model_class = MODEL_REGISTRY[model_name]
debiased_class = build_two_timescale_fixed_long_debias_model(
    model_class, short_half_life=args.short_half_life, long_half_life=args.long_half_life,
)

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
dataset.prepare_user_timebucket_sampler(bucket_size=args.user_bucket_size)
dataset.get_pair_user_event_timebucket_fast()


epoch = 0

save_dir = Path(args.save_path)
pattern = (
    f"_{args.model_name}_lambdacen{args.lambda_cen}_tau{args.tau}_scorenorm{args.score_norm}"
    f"_shl{args.short_half_life}_lhl{args.long_half_life}_e???_seed{args.seed}_hawkestwotimescalefixedlong.pt"
)
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
        hot_sample_idx = hot_idxs[mini_batch*idx : (idx + 1)*mini_batch]

        """CANDIDATE SET: observed hot item (col 0) + uniform negatives (cols 1..M)"""
        hot_anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_neg_item = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        hot_pos_time = torch.tensor(dataset.hot_event_time_list[hot_sample_idx], dtype=torch.float32, device=args.device)
        hot_pos_time_all = torch.tensor(dataset.hot_pos_time_all[hot_sample_idx], dtype=torch.float32, device=args.device)
        hot_neg_time_all = torch.tensor(dataset.hot_neg_time_all[hot_sample_idx], dtype=torch.float32, device=args.device)

        candidate_items = torch.cat([hot_pos_item.unsqueeze(-1), hot_neg_item], dim=-1)
        candidate_time_all = torch.cat([hot_pos_time_all.unsqueeze(1), hot_neg_time_all], dim=1)

        """CHOICE LOSS: l_choice(Phi, Psi) -- model.prior()'s signature and return
        value are unchanged (still a single [B, 1+M] intensity tensor), so this
        call is identical to debiased_seq_rec_hawkes_anova.py"""
        choice_loss = importance_corrected_choice_loss(
            model, candidate_items, anchor_hist_items, hot_anchor_user,
            hot_pos_time, candidate_time_all, dataset.m_item, num_negatives,
        )

        """ANOVA-CENTERING PENALTY: R_cen(Psi) -- unaffected by the Hawkes
        excitation change, since it only uses score_pair (utility), not model.prior()"""
        dataset.get_pair_user_event_timebucket_fast()
        center_user_a_np = dataset.hot_neg_user_list[hot_sample_idx, 0].copy()
        center_hist_a_np = dataset.get_histories_for_users_at_times(
            center_user_a_np, dataset.hot_event_time_list[hot_sample_idx], args.max_seq_len,
        )

        dataset.get_pair_user_event_timebucket_fast()
        center_user_b_np = dataset.hot_neg_user_list[hot_sample_idx, 0].copy()
        center_hist_b_np = dataset.get_histories_for_users_at_times(
            center_user_b_np, dataset.hot_event_time_list[hot_sample_idx], args.max_seq_len,
        )

        center_user_a = torch.tensor(center_user_a_np, dtype=torch.long, device=args.device)
        center_hist_a = torch.tensor(center_hist_a_np, dtype=torch.long, device=args.device)
        center_user_b = torch.tensor(center_user_b_np, dtype=torch.long, device=args.device)
        center_hist_b = torch.tensor(center_hist_b_np, dtype=torch.long, device=args.device)

        center_loss = anova_centering_penalty(
            model, candidate_items, center_hist_a, center_user_a, center_hist_b, center_user_b,
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
        log_dict = {
            "train_choice_loss": epoch_choice_loss / batch_num,
            "train_center_loss": epoch_center_loss / batch_num,
        }
        # Diagnostics only (see module.debias_two_timescale_fixedlong.prior):
        # reuses the last batch's already-built candidate tensors, so this is
        # an extra no-grad forward pass purely for logging, not part of the
        # loss. Includes mean_alpha_short/long and mean_h_short/long (not just
        # their products) so it's possible to tell apart "alpha_long learned
        # to matter" from "h_long is just naturally small" -- exactly the
        # ambiguity the first run's diagnostics couldn't resolve.
        with torch.no_grad():
            diag = {}
            model.prior(candidate_items, hot_pos_time, candidate_time_all, diagnostics=diag)
        wandb_var.log({**log_dict, **{f"hawkes_{k}": v for k, v in diag.items()}})

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, (
            f"{args.save_path}/_{args.model_name}_lambdacen{args.lambda_cen}_tau{args.tau}_scorenorm{args.score_norm}"
            f"_shl{args.short_half_life}_lhl{args.long_half_life}_e{epoch}_seed{args.seed}_hawkestwotimescalefixedlong.pt"
        ))


all_item_idxs = np.arange(dataset.m_item)

if epoch % args.evaluate_interval == 0:
    pred_list = []
    gt_list = []

    model.eval()
    with torch.no_grad():
        mu, alpha_short, alpha_long, beta_short, beta_long = model.prior_parameters_from_embeddings()

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
                time_intensity_short = (torch.exp(-beta_short * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                time_intensity_long = (torch.exp(-beta_long * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                logits = (
                    mu[item_idx]
                    + alpha_short[item_idx] * time_intensity_short.squeeze(-1)
                    + alpha_long[item_idx] * time_intensity_long.squeeze(-1)
                ).flatten()
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
        mu, alpha_short, alpha_long, beta_short, beta_long = model.prior_parameters_from_embeddings()

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
                time_intensity_short = (torch.exp(-beta_short * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                time_intensity_long = (torch.exp(-beta_long * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                logits = (
                    mu[item_idx]
                    + alpha_short[item_idx] * time_intensity_short.squeeze(-1)
                    + alpha_long[item_idx] * time_intensity_long.squeeze(-1)
                ).flatten()
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
        mu, alpha_short, alpha_long, beta_short, beta_long = model.prior_parameters_from_embeddings()


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
                    time_intensity_short = (torch.exp(-beta_short * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                    time_intensity_long = (torch.exp(-beta_long * batch_time_delta) * batch_time_mask).sum(-1, keepdim=True)
                    logits = (
                        mu[item_idx]
                        + alpha_short[item_idx] * time_intensity_short.squeeze(-1)
                        + alpha_long[item_idx] * time_intensity_long.squeeze(-1)
                    ).flatten()
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
