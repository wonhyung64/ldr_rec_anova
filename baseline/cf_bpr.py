#%%
"""
BPR loss control for the CF backbone (MF).  Counterpart of cf.py / cf_ss.py.

Trains the same MF with BPR, -log sigmoid(s_pos - s_neg), averaged over the
M = contrast_size - 1 = 15 uniform negatives per positive (use
--contrast-size=2 for classic 1-negative BPR).  Scoring (cosine/tau), N(0, 0.02)
item init, optimizer and data regime are identical to cf_ss.py and "+Ours"; no
Hawkes prior / centering.  Like cf_ss.py, cold events are trained on too.
"""
import os
import wandb
import torch
import inspect
import numpy as np
from datetime import datetime

from torch import optim

from module.utils import parse_args, set_seed, set_device
from module.dataset import UserItemTime
from module.model import MODEL_REGISTRY
from module.bpr import bpr_loss
from module.sampled_softmax import (
    init_item_embedding, sample_uniform_negatives,
    resume_if_possible, save_checkpoint, evaluate_valid, evaluate_test_all,
    with_epoch_suffix, dump_result_json,
)


#%%
args = parse_args()
args.loss_tag = "bpr"
if args.result_dir == "./results/sampled_softmax":
    args.result_dir = "./results/bpr"
set_seed(args.seed)
args.device = set_device(args.device)
args.save_path = f"{args.weights_path}/{args.dataset}"
os.makedirs(args.save_path, exist_ok=True)

wandb_login = False
file_name = os.path.basename(inspect.getfile(inspect.currentframe()))
try:
    wandb_login = wandb.login(key=open(f"{args.cred_path}/wandb_key.txt", 'r').readline())
except Exception:
    wandb_login = False

if wandb_login:
    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    args.expt_name = f"{file_name.split('.')[-2]}_{args.model_name}_{expt_num}"
    wandb_var = wandb.init(project="ldr_rec_bpr", config=vars(args))
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

num_negatives = args.contrast_size - 1

# MF ignores the history; a constant all-padding placeholder keeps the
# residual_score API shape shared with the sequential backbones.
dummy_hist = torch.full((mini_batch, args.max_seq_len), dataset.m_item, dtype=torch.long, device=args.device)


#%%
model_name = args.model_name.lower()
if model_name not in ("mf", "ncf"):
    raise ValueError(f"cf_ss.py supports mf/ncf, got {model_name}")

model = MODEL_REGISTRY[model_name](
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
init_item_embedding(model, args.item_init_std)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)


#%%
epoch = resume_if_possible(model, optimizer, args)
valid_metrics = None

while epoch < args.epochs:
    epoch += 1
    torch.cuda.empty_cache()
    model.train()
    np.random.shuffle(hot_idxs)
    epoch_loss = 0.0
    n_steps = 0

    for idx in range(batch_num):
        hot_sample_idx = hot_idxs[hot_mini_batch * idx: (idx + 1) * hot_mini_batch]
        cold_sample_idx = cold_idxs[cold_mini_batch * idx: (idx + 1) * cold_mini_batch]
        if len(hot_sample_idx) + len(cold_sample_idx) == 0:
            continue

        anchor_user = torch.tensor(
            np.concatenate([dataset.cold_user_list[cold_sample_idx], dataset.hot_user_list[hot_sample_idx]]),
            dtype=torch.long, device=args.device,
        )

        """CANDIDATE SET: observed item (col 0, hot+cold combined) + M uniform negatives"""
        pos_np = np.concatenate([dataset.cold_item_list[cold_sample_idx], dataset.hot_item_list[hot_sample_idx]])
        neg_np = sample_uniform_negatives(pos_np, dataset.m_item, num_negatives)
        candidate_items = torch.tensor(
            np.concatenate([pos_np[:, None], neg_np], axis=1), dtype=torch.long, device=args.device)

        loss = bpr_loss(
            model, candidate_items, dummy_hist[:anchor_user.shape[0]], anchor_user,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_steps += 1

    print(f"[Epoch {epoch:>4d} Train Loss] bpr: {epoch_loss / max(n_steps, 1):.4f}")

    if wandb_login:
        wandb_var.log({"train_bpr_loss": epoch_loss / max(n_steps, 1)})

    if epoch % 100 == 0:
        save_checkpoint(model, optimizer, args, epoch)

    if epoch % args.evaluate_interval == 0 and epoch < args.epochs:
        valid_metrics = evaluate_valid(model, dataset, args, mode="user_nohist")
        print(f"[Epoch {epoch:>4d}] valid recall@10 {valid_metrics['valid_recall_10']:.4f}  ndcg@10 {valid_metrics['valid_ndcg_10']:.4f}")
        if wandb_login:
            wandb_var.log(valid_metrics)


#%%
"""FINAL EVALUATION: valid, test (Table 1), head/tail x overall/recent_3d/recent_7d (Table 2, Fig. 2)"""
valid_metrics = evaluate_valid(model, dataset, args, mode="user_nohist")
test_metrics = evaluate_test_all(model, dataset, args, mode="user_nohist")

print(f"[Final e{epoch}] valid recall@10 {valid_metrics['valid_recall_10']:.4f} / test recall@10 {test_metrics['test_recall_10']:.4f} ndcg@10 {test_metrics['test_ndcg_10']:.4f}")

dump_result_json(args, epoch, test_metrics, valid_metrics)

if wandb_login:
    wandb_var.log(with_epoch_suffix({**valid_metrics, **test_metrics}, epoch))
    wandb_var.finish()
