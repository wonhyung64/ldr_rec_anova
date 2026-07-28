#%%
import re
import os
import copy
import wandb
import torch
import inspect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime

from module.utils import parse_args, set_seed, set_device
from module.procedure import computeTopNAccuracy
from module.dataset import UserItemTime
from module.model import score_pair, score_all, MODEL_REGISTRY
from module.debias import build_debias_model, build_unshared_debias_model, build_linear_debias_model, build_unshared_linear_debias_model
from module.sampler import make_prior_snapshot, sample_epoch_negatives, FenwickTree


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
    wandb_var = wandb.init(project="ldr_rec_fixed", config=vars(args))
    wandb.run.name = args.expt_name


#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)

# `item_time_array` is built in module/dataset.py's time_dict_to_array from
# the raw (unix-second) time_dict, while every query time used in this
# script (hot/cold_event_time_list, valid/test event times) is in "day"
# units (time_unit="d" above divides by 86400 in set_to_pair). Left
# unconverted, item_time_array (~1e9) is never < a day-scale query time
# (~1e4), so the window condition `item_time_array < query_time` is always
# false and compute_window_prior degenerates to a uniform distribution
# regardless of window_size. Align item_time_array to the same day-unit
# scale so the window actually reflects recent interactions.
dataset.item_time_array = dataset.item_time_array / 86400.0

mini_batch = args.batch_size // args.contrast_size
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

all_item_idxs = np.arange(dataset.m_item)

#%%
# ---------------------------------------------------------------------------
# Fast window-prior helpers.
#
# The original `debiased_cf_window_prior.py` recomputes, from scratch, a full
# probability vector over *all* m_item items (via `compute_window_prior`) for
# every single training example, every single batch, every single epoch. That
# is O(batch_num * mini_batch * m_item * time_len) numpy work per epoch, done
# through a python-level list comprehension, even though the training loss
# only ever needs the probability at the handful (`contrast_size`) of sampled
# items, plus the normalization total. `debiased_cf_weighted.py` avoids this
# entirely by using a closed-form (Hawkes) prior evaluated only on the sampled
# items via a vectorized GPU call.
#
# Below we keep the exact same math as `compute_window_prior`, but:
#   1. Only gather item_time_array rows for the items actually used in the
#      loss (numerator) instead of scanning every item.
#   2. Replace the O(m_item * time_len) normalization sum with two
#      np.searchsorted lookups against a single sorted array of all valid
#      timestamps in item_time_array (built once, since item_time_array never
#      changes during training).
# This produces numerically identical results to the original per-item
# `compute_window_prior`, just without the redundant O(m_item) work.
# ---------------------------------------------------------------------------
window_size = getattr(args, 'window_size', 3)


def compute_window_prior(item_time_array, pos_mask, query_time, window_size, eps=1e-9):
    """Full per-item distribution, kept for the (infrequent) eval-time ranking
    over all items, where a full distribution is actually required."""
    in_window = pos_mask & (item_time_array >= query_time - window_size) & (item_time_array < query_time)
    counts = in_window.sum(axis=1).astype(np.float64) + eps
    return counts / counts.sum()


def build_window_prior_context(item_time_array):
    pos_mask = item_time_array > 0
    m_item, time_len = item_time_array.shape
    flat_sorted = np.sort(item_time_array[pos_mask])

    # Flattened (time, item) arrivals, sorted by time, used by the sliding-
    # window Fenwick-tree negative sampler below.
    flat_items = np.repeat(np.arange(m_item), time_len)
    flat_times_all = item_time_array.reshape(-1)
    arrival_times = flat_times_all[pos_mask.reshape(-1)]
    arrival_items = flat_items[pos_mask.reshape(-1)]
    arrival_order = np.argsort(arrival_times, kind="stable")
    arrival_times = arrival_times[arrival_order].astype(np.float64)
    arrival_items = arrival_items[arrival_order].astype(np.int64)

    return {
        "pos_mask": pos_mask,
        "flat_sorted": flat_sorted,
        "m_item": m_item,
        "arrival_times": arrival_times,
        "arrival_items": arrival_items,
    }


def window_prior_denominator(flat_sorted, m_item, query_times, window_size, eps=1e-9):
    """Vectorized total normalization mass for a batch of query times, via
    binary search on the (constant) sorted timestamp array instead of an
    O(m_item * time_len) scan per query time."""
    qt = np.asarray(query_times, dtype=np.float64)
    hi = np.searchsorted(flat_sorted, qt, side="left")
    lo = np.searchsorted(flat_sorted, qt - window_size, side="left")
    counts_total = (hi - lo).astype(np.float64)
    return counts_total + m_item * eps


def window_prior_probs_for_items(item_time_array, ctx, items, query_times, window_size, eps=1e-9):
    """Returns prob[b, c] == compute_window_prior(item_time_array, query_times[b], window_size)[items[b, c]],
    computed without ever materializing the (B, m_item) distribution."""
    gathered = item_time_array[items]  # (B, C, T) - cheap gather, only touches sampled items
    q = np.asarray(query_times, dtype=np.float64).reshape(-1, 1, 1)
    in_window = (gathered > 0) & (gathered >= q - window_size) & (gathered < q)
    numerator = in_window.sum(axis=-1).astype(np.float64) + eps  # (B, C)

    denom = window_prior_denominator(ctx["flat_sorted"], ctx["m_item"], query_times, window_size, eps)  # (B,)
    return numerator / denom[:, None]


def sample_window_prior_negatives_fast(ctx, event_times, window_size, num_negatives, eps=1e-9):
    """Same sampling semantics as `np.random.choice(m_item, p=window_prior)`
    per event (i.i.d., with replacement, no exclusion of the positive item -
    matching the original's behavior exactly), but computed with a sliding
    window over a Fenwick tree instead of rebuilding the full (m_item,)
    distribution from scratch for every event.

    Because item_time_array never changes and every query window is a
    contiguous [t - window_size, t) interval, the per-item in-window count
    only changes at the ~m_item*time_len timestamps where some item's
    interaction enters or leaves the window. Sweeping query times in
    increasing order and maintaining running counts in a Fenwick tree turns
    the original O(N * m_item * time_len) scan into O((m_item*time_len + N)
    * log m_item)."""
    arrival_times = ctx["arrival_times"]
    arrival_items = ctx["arrival_items"]
    m_item = ctx["m_item"]

    n = len(event_times)
    negs = np.zeros((n, num_negatives), dtype=np.int64)
    if n == 0:
        return negs

    tree = FenwickTree(m_item)
    for i in range(m_item):
        tree.add(i, eps)

    n_arrivals = len(arrival_times)
    enter_ptr = 0
    exit_ptr = 0

    order = np.argsort(event_times, kind="stable")
    for oi in order:
        t = float(event_times[oi])
        while enter_ptr < n_arrivals and arrival_times[enter_ptr] < t:
            tree.add(int(arrival_items[enter_ptr]), 1.0)
            enter_ptr += 1
        lower = t - window_size
        while exit_ptr < n_arrivals and arrival_times[exit_ptr] < lower:
            tree.add(int(arrival_items[exit_ptr]), -1.0)
            exit_ptr += 1

        total = tree.total()
        for j in range(num_negatives):
            mass = np.random.rand() * total
            negs[oi, j] = tree.sample(mass)

    return negs


window_ctx = build_window_prior_context(dataset.item_time_array)


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

item_hot_negs = sample_window_prior_negatives_fast(window_ctx, dataset.hot_event_time_list, window_size, args.contrast_size-1)
item_cold_negs = sample_window_prior_negatives_fast(window_ctx, dataset.cold_event_time_list, window_size, args.contrast_size-1)

if args.dr_anchor != "item":
    hot_negs = item_hot_negs
    cold_negs = item_cold_negs

if args.dr_anchor != "user":
    dataset.prepare_user_timebucket_sampler(w_cold=True)
    dataset.get_pair_user_event_timebucket_fast(w_cold=True)

epoch = 0

save_dir = Path(args.save_path)
pattern = f"{args.model_name}_lambda{args.lambda1}_e???_seed{args.seed}_window_prior_fast_timefix.pt"
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
        cold_sample_idx = cold_idxs[cold_mini_batch*idx : (idx + 1)*cold_mini_batch]

        """USER"""
        hot_anchor_user = torch.tensor(dataset.hot_user_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_pos_item = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        cold_anchor_user = torch.tensor(dataset.cold_user_list[cold_sample_idx], dtype=torch.long, device=args.device)
        cold_pos_item = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)

        anchor_user = torch.cat([cold_anchor_user, hot_anchor_user], dim=0)
        pos_item = torch.cat([cold_pos_item, hot_pos_item], dim=0)
        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        if args.dr_anchor != "item":
            hot_neg_item = torch.tensor(hot_negs[hot_sample_idx], dtype=torch.long, device=args.device)
            cold_neg_item = torch.tensor(cold_negs[cold_sample_idx], dtype=torch.long, device=args.device)
            neg_item = torch.cat([cold_neg_item, hot_neg_item], dim=0)

            pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
            neg_score = score_pair(model, neg_item, anchor_hist_items, anchor_user)
            user_loss += -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean() * args.lambda1

        if args.dr_anchor != "user":

            hot_neg_user = torch.tensor(dataset.hot_neg_user_list[hot_sample_idx, 0], dtype=torch.long, device=args.device)
            cold_neg_user = torch.tensor(dataset.cold_neg_user_list[cold_sample_idx, 0], dtype=torch.long, device=args.device)
            neg_user = torch.cat([cold_neg_user, hot_neg_user], dim=0)

            pos_score = score_pair(model, pos_item, anchor_hist_items, anchor_user)
            neg_score = score_pair(model, pos_item, anchor_hist_items, neg_user)
            user_loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1, keepdim=True)).mean()


            dataset.get_pair_user_event_timebucket_fast(w_cold=True)

        epoch_user_loss += user_loss.item()


        """ITEM"""
        hot_neg_item = item_hot_negs[hot_sample_idx]
        cold_neg_item = item_cold_negs[cold_sample_idx]
        neg_item_np = np.concatenate([cold_neg_item, hot_neg_item], axis=0)

        pos_item_np = np.concatenate([
            dataset.cold_pos_item_list[cold_sample_idx],
            dataset.hot_pos_item_list[hot_sample_idx],
        ], axis=0)
        batch_items_np = np.concatenate([pos_item_np[:, None], neg_item_np], axis=1)

        hot_event_times = dataset.hot_event_time_list[hot_sample_idx]
        cold_event_times = dataset.cold_event_time_list[cold_sample_idx]
        event_times = np.concatenate([cold_event_times, hot_event_times])

        window_probs = window_prior_probs_for_items(dataset.item_time_array, window_ctx, batch_items_np, event_times, window_size)
        logits = torch.tensor(window_probs, dtype=torch.float32, device=args.device)
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
        },  f"{args.save_path}/{args.model_name}_lambda{args.lambda1}_e{epoch}_seed{args.seed}_window_prior_fast_timefix.pt")


    if epoch % args.pair_reset_interval == 0:
        print("Reset Negs")
        item_hot_negs = sample_window_prior_negatives_fast(window_ctx, dataset.hot_event_time_list, window_size, args.contrast_size-1)
        item_cold_negs = sample_window_prior_negatives_fast(window_ctx, dataset.cold_event_time_list, window_size, args.contrast_size-1)
        if args.dr_anchor != "item":
            hot_negs = item_hot_negs
            cold_negs = item_cold_negs


if epoch % args.evaluate_interval == 0:
    item_nll_list = []
    pred_list = []
    gt_list = []

    model.eval()

    for (user, item), pos_time_val in dataset.valid_user_item_time.items():
        hist_item_np, hist_time_np = dataset.build_histories(zip([user], [0], [pos_time_val]), args.max_seq_len)
        hist_item_t = torch.tensor(hist_item_np, dtype=torch.long, device=args.device)
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            resid = score_all(model, hist_item_t, user_t).squeeze(0).cpu()

        window_prob = compute_window_prior(dataset.item_time_array, window_ctx["pos_mask"], pos_time_val, window_size)
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

        window_prob = compute_window_prior(dataset.item_time_array, window_ctx["pos_mask"], pos_time_val, window_size)
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
