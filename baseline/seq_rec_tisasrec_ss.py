#%%
"""
Sampled-softmax loss control for TiSASRec.  Counterpart of seq_rec_tisasrec.py.

Same recipe as seq_rec_ss.py (importance-corrected l_choice with M = 15
uniform negatives, uniform popularity prior, no ANOVA-centering, cosine/tau
scoring, N(0, 0.02) item init), but TiSASRec's encode_user consumes the
history *timestamps* (seconds) instead of a user id, so they are threaded
through exactly as in seq_rec_tisasrec.py / the +Ours TiSASRec script.
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
from module.tisasrec import TiSASRec
from module.sampled_softmax import (
    init_item_embedding, sample_uniform_negatives, sampled_softmax_loss,
    resume_if_possible, save_checkpoint, evaluate_valid, evaluate_test_all,
    with_epoch_suffix, dump_result_json,
)


#%%
args = parse_args()
args.model_name = "tisasrec"
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
    wandb_var = wandb.init(project="ldr_rec_sampled_softmax", config=vars(args))
    wandb.run.name = args.expt_name


#%%
dataset = UserItemTime("./data", args.dataset, "d", 50, args.max_seq_len)

mini_batch = args.batch_size // args.contrast_size
batch_num = (dataset.hotDataSize + mini_batch - 1) // mini_batch
hot_idxs = np.arange(dataset.hotDataSize)
num_negatives = args.contrast_size - 1


#%%
if args.dataset == "ml-1m":
    time_span = 2048
else:
    time_span = 512

model = TiSASRec(
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
    time_span=time_span,
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

    for idx in range(batch_num):
        sample_idx = hot_idxs[mini_batch * idx: (idx + 1) * mini_batch]

        anchor_hist_items = torch.tensor(dataset.train_hist_item_list[sample_idx], dtype=torch.long, device=args.device)
        anchor_hist_times = torch.tensor(dataset.train_hist_time_list[sample_idx], dtype=torch.long, device=args.device) * 24 * 60 * 60

        """CANDIDATE SET: observed item (col 0) + M uniform negatives (cols 1..M)"""
        pos_np = dataset.hot_item_list[sample_idx]
        neg_np = sample_uniform_negatives(pos_np, dataset.m_item, num_negatives)
        candidate_items = torch.tensor(
            np.concatenate([pos_np[:, None], neg_np], axis=1), dtype=torch.long, device=args.device)

        loss = sampled_softmax_loss(
            model, candidate_items, anchor_hist_items, anchor_hist_times,
            dataset.m_item, num_negatives, correction=bool(args.ss_correction),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch:>4d} Train Loss] sampled-softmax: {epoch_loss / batch_num:.4f}")

    if wandb_login:
        wandb_var.log({"train_ss_loss": epoch_loss / batch_num})

    if epoch % 100 == 0:
        save_checkpoint(model, optimizer, args, epoch)

    if epoch % args.evaluate_interval == 0 and epoch < args.epochs:
        valid_metrics = evaluate_valid(model, dataset, args, mode="time")
        print(f"[Epoch {epoch:>4d}] valid recall@10 {valid_metrics['valid_recall_10']:.4f}  ndcg@10 {valid_metrics['valid_ndcg_10']:.4f}")
        if wandb_login:
            wandb_var.log(valid_metrics)


#%%
"""FINAL EVALUATION: valid, test (Table 1), head/tail x overall/recent_3d/recent_7d (Table 2, Fig. 2)"""
valid_metrics = evaluate_valid(model, dataset, args, mode="time")
test_metrics = evaluate_test_all(model, dataset, args, mode="time")

print(f"[Final e{epoch}] valid recall@10 {valid_metrics['valid_recall_10']:.4f} / test recall@10 {test_metrics['test_recall_10']:.4f} ndcg@10 {test_metrics['test_ndcg_10']:.4f}")

dump_result_json(args, epoch, test_metrics, valid_metrics)

if wandb_login:
    wandb_var.log(with_epoch_suffix({**valid_metrics, **test_metrics}, epoch))
    wandb_var.finish()
