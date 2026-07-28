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
from module.paac_model import PAACModel


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
batch_num = dataset.trainDataSize // mini_batch + 1

hot_ratio = dataset.hotDataSize / dataset.trainDataSize
hot_mini_batch = round(mini_batch * hot_ratio)
hot_idxs = np.arange(dataset.hotDataSize)
cold_mini_batch = mini_batch - hot_mini_batch
cold_idxs = np.arange(dataset.coldDataSize)

# Item popularity split: top 20% by interaction count = popular
item_freq = np.bincount(dataset.trainItem, minlength=dataset.m_item)
pop_threshold = np.percentile(item_freq[item_freq > 0], 80.0)
is_pop_item = item_freq >= pop_threshold          # bool mask [m_item]
pop_item_arr = np.where(is_pop_item)[0].astype(np.int64)
cold_item_arr = np.where(~is_pop_item)[0].astype(np.int64)


#%%
model = PAACModel(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    tau=args.tau,
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
pattern = f"paac_lr{args.lr}_tau{args.tau}_alpha{args.alpha}_gamma{args.gamma}_e???_seed{args.seed}.pt"
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
    epoch_bpr_loss  = 0.0
    epoch_aln_loss  = 0.0
    epoch_con_loss  = 0.0

    for idx in range(batch_num):
        hot_sample_idx  = hot_idxs[hot_mini_batch * idx : (idx + 1) * hot_mini_batch]
        cold_sample_idx = cold_idxs[cold_mini_batch * idx : (idx + 1) * cold_mini_batch]

        hot_anchor_user  = torch.tensor(dataset.hot_user_list[hot_sample_idx],  dtype=torch.long, device=args.device)
        hot_pos_item     = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx],  dtype=torch.long, device=args.device)
        hot_neg_item     = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx],  dtype=torch.long, device=args.device)

        cold_anchor_user = torch.tensor(dataset.cold_user_list[cold_sample_idx], dtype=torch.long, device=args.device)
        cold_pos_item    = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item    = torch.tensor(dataset.cold_neg_item_list[cold_sample_idx], dtype=torch.long, device=args.device)

        anchor_user = torch.cat([cold_anchor_user, hot_anchor_user], dim=0)
        pos_item    = torch.cat([cold_pos_item,    hot_pos_item],    dim=0)
        neg_item    = torch.cat([cold_neg_item,    hot_neg_item],    dim=0)

        # ---- BPR recommendation loss (interest branch) ----
        bpr = model.bpr_loss(anchor_user, pos_item, neg_item)

        # ---- Popularity-Aware Alignment ----
        # For each interaction in the batch, pair with a random item from the
        # OPPOSITE popularity group to form (pop_item, cold_item) pairs.
        B = anchor_user.shape[0]
        pos_np = pos_item.cpu().numpy()
        is_pop_batch = is_pop_item[pos_np]             # bool [B]

        # For pop interactions → sample a cold item; for cold → sample a pop item
        paired_np = np.empty(B, dtype=np.int64)
        n_pop  = int(is_pop_batch.sum())
        n_cold = B - n_pop
        if n_pop > 0:
            paired_np[is_pop_batch]  = cold_item_arr[np.random.randint(0, len(cold_item_arr), n_pop)]
        if n_cold > 0:
            paired_np[~is_pop_batch] = pop_item_arr[np.random.randint(0, len(pop_item_arr), n_cold)]

        paired_item = torch.tensor(paired_np, dtype=torch.long, device=args.device)

        # pop_for_align: the "popular" side; cold_for_align: the "cold" side
        pop_for_align  = torch.where(
            torch.tensor(is_pop_batch, device=args.device), pos_item, paired_item
        )
        cold_for_align = torch.where(
            torch.tensor(is_pop_batch, device=args.device), paired_item, pos_item
        )

        aln = model.alignment_loss(anchor_user, pop_for_align, cold_for_align)

        # ---- Popularity-Aware Contrast ----
        con = model.contrast_loss(anchor_user, pop_for_align, cold_for_align)

        total_loss = bpr + args.alpha * aln + args.gamma * con

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_bpr_loss += bpr.item()
        epoch_aln_loss += aln.item()
        epoch_con_loss += con.item()

        dataset.get_pair_item_uniform(k=args.contrast_size - 1, w_time=False)

    print(
        f"[Epoch {epoch:>4d} Train Loss] "
        f"bpr: {epoch_bpr_loss / batch_num:.4f} / "
        f"align: {epoch_aln_loss / batch_num:.4f} / "
        f"con: {epoch_con_loss / batch_num:.4f}"
    )

    if epoch % 100 == 0:
        save_name = f"paac_lr{args.lr}_tau{args.tau}_alpha{args.alpha}_gamma{args.gamma}_e{epoch}_seed{args.seed}.pt"
        save_path = f"{args.save_path}/{save_name}"

        # Delete previous checkpoint for this config
        prev_pattern = f"paac_lr{args.lr}_tau{args.tau}_alpha{args.alpha}_gamma{args.gamma}_e???_seed{args.seed}.pt"
        for old_file in save_dir.glob(prev_pattern):
            if old_file != Path(save_path):
                old_file.unlink()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_bpr_loss,
        }, save_path)

    if epoch % args.evaluate_interval == 0:
        model.eval()

        for flag, user_item_time in [("valid", dataset.valid_user_item_time),
                                      ("test",  dataset.test_user_item_time)]:
            pred_list, gt_list = [], []

            for (user, item), _ in user_item_time.items():
                user_t = torch.tensor([user], dtype=torch.long, device=args.device)

                with torch.no_grad():
                    scores = model.score_all_items(user_t).squeeze(0).cpu()

                exclude_items = list(dataset._allPos[user])
                scores[exclude_items] = -9999
                _, pred_k = torch.topk(scores, k=max(args.topks))
                pred_list.append(pred_k.cpu())
                gt_list.append([item])

            results = computeTopNAccuracy(gt_list, pred_list, args.topks)

            if wandb_login:
                wandb_var.log(dict(zip([f"{flag}_precision_{k}_{epoch}" for k in args.topks], results[0])))
                wandb_var.log(dict(zip([f"{flag}_recall_{k}_{epoch}"    for k in args.topks], results[1])))
                wandb_var.log(dict(zip([f"{flag}_ndcg_{k}_{epoch}"      for k in args.topks], results[2])))
                wandb_var.log(dict(zip([f"{flag}_mrr_{k}_{epoch}"       for k in args.topks], results[3])))

if wandb_login:
    wandb_var.finish()

# %%
