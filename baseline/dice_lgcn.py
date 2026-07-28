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
from module.dice_lgcn_model import DICELightGCN, build_norm_adj


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

all_item_idxs = np.arange(dataset.m_item)


#%%
# Build interest graph: all training interactions
int_users = np.array([u for (u, v, t) in dataset.train_events])
int_items = np.array([v for (u, v, t) in dataset.train_events])
int_adj = build_norm_adj(int_users, int_items, dataset.n_user, dataset.m_item, args.device)

# Build conformity graph: hot (repeated / popular) interactions only
con_users = np.array([u for (u, v, t) in dataset.train_hot_events])
con_items = np.array([v for (u, v, t) in dataset.train_hot_events])
con_adj = build_norm_adj(con_users, con_items, dataset.n_user, dataset.m_item, args.device)

print(f"Interest graph edges: {len(int_users) * 2}  |  Conformity graph edges: {len(con_users) * 2}")


#%%
model = DICELightGCN(
    num_users=dataset.n_user,
    num_items=dataset.m_item,
    embedding_k=args.recdim,
    device=args.device,
    n_layers=args.n_layers,
    tau=args.tau,
    int_adj=int_adj,
    con_adj=con_adj,
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
pattern = f"dice_lgcn_lambda{args.lambda1}_lr{args.lr}_alpha{args.alpha}_nlayers{args.n_layers}_e???_seed{args.seed}.pt"
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
    epoch_int_loss = 0.0
    epoch_con_loss = 0.0
    epoch_dis_loss = 0.0

    for idx in range(batch_num):
        hot_sample_idx  = hot_idxs[hot_mini_batch  * idx : (idx + 1) * hot_mini_batch]
        cold_sample_idx = cold_idxs[cold_mini_batch * idx : (idx + 1) * cold_mini_batch]

        hot_anchor_user  = torch.tensor(dataset.hot_user_list[hot_sample_idx],   dtype=torch.long, device=args.device)
        hot_pos_item     = torch.tensor(dataset.hot_pos_item_list[hot_sample_idx], dtype=torch.long, device=args.device)
        hot_neg_item     = torch.tensor(dataset.hot_neg_item_list[hot_sample_idx], dtype=torch.long, device=args.device)

        cold_anchor_user = torch.tensor(dataset.cold_user_list[cold_sample_idx],   dtype=torch.long, device=args.device)
        cold_pos_item    = torch.tensor(dataset.cold_pos_item_list[cold_sample_idx], dtype=torch.long, device=args.device)
        cold_neg_item    = torch.tensor(dataset.cold_neg_item_list[cold_sample_idx], dtype=torch.long, device=args.device)

        anchor_user = torch.cat([cold_anchor_user, hot_anchor_user], dim=0)
        pos_item    = torch.cat([cold_pos_item,    hot_pos_item],    dim=0)
        neg_item    = torch.cat([cold_neg_item,    hot_neg_item],    dim=0)

        # --- Interest BPR (all interactions, random negatives) ---
        int_pos = model.interest_score(anchor_user, pos_item)       # [B]
        int_neg = model.interest_score(anchor_user, neg_item)       # [B, N]
        int_loss = -(F.logsigmoid(int_pos.unsqueeze(-1) - int_neg)).mean()

        # --- Conformity BPR (hot = popular positive, cold item as negative) ---
        if hot_anchor_user.shape[0] > 0 and cold_anchor_user.shape[0] > 0:
            n_hot = hot_anchor_user.shape[0]
            cold_neg_for_con = cold_pos_item[np.arange(n_hot) % cold_anchor_user.shape[0]]
            con_pos = model.conformity_score(hot_anchor_user, hot_pos_item)
            con_neg = model.conformity_score(hot_anchor_user, cold_neg_for_con)
            con_loss = -(F.logsigmoid(con_pos - con_neg)).mean()
        else:
            con_loss = torch.zeros(1, device=args.device).squeeze()

        # --- Discrepancy loss ---
        dis_loss = model.discrepancy_loss(anchor_user, pos_item)

        total_loss = (
            args.lambda1         * int_loss
            + (1 - args.lambda1) * con_loss
            + args.alpha         * dis_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_int_loss += int_loss.item()
        epoch_con_loss += con_loss.item() if isinstance(con_loss, torch.Tensor) else con_loss
        epoch_dis_loss += dis_loss.item()

        dataset.get_pair_item_uniform(k=args.contrast_size - 1, w_time=False)

    print(
        f"[Epoch {epoch:>4d} Train Loss] "
        f"int: {epoch_int_loss / batch_num:.4f} / "
        f"con: {epoch_con_loss / batch_num:.4f} / "
        f"dis: {epoch_dis_loss / batch_num:.4f}"
    )

    if epoch % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_int_loss,
        }, f"{args.save_path}/dice_lgcn_lambda{args.lambda1}_lr{args.lr}_alpha{args.alpha}_nlayers{args.n_layers}_e{epoch}_seed{args.seed}.pt")

    if epoch % args.evaluate_interval == 0:
        model.eval()

        for flag, user_item_time in [("valid", dataset.valid_user_item_time),
                                      ("test",  dataset.test_user_item_time)]:
            pred_list, gt_list = [], []

            for (user, item), _ in user_item_time.items():
                user_t = torch.tensor([user], dtype=torch.long, device=args.device)

                with torch.no_grad():
                    scores = model.score_all_items(user_t).squeeze(0).cpu()   # [M]

                exclude_items = list(dataset._allPos[user])
                scores[exclude_items] = -9999
                _, pred_k = torch.topk(scores, k=max(args.topks))
                pred_list.append(pred_k.cpu())
                gt_list.append([item])

            results = computeTopNAccuracy(gt_list, pred_list, args.topks)

            print(
                f"[Epoch {epoch} {flag}] "
                f"Recall@{args.topks}: {results[1]} | "
                f"NDCG@{args.topks}: {results[2]}"
            )

            if wandb_login:
                wandb_var.log(dict(zip([f"{flag}_precision_{k}_{epoch}" for k in args.topks], results[0])))
                wandb_var.log(dict(zip([f"{flag}_recall_{k}_{epoch}"    for k in args.topks], results[1])))
                wandb_var.log(dict(zip([f"{flag}_ndcg_{k}_{epoch}"      for k in args.topks], results[2])))
                wandb_var.log(dict(zip([f"{flag}_mrr_{k}_{epoch}"       for k in args.topks], results[3])))

if wandb_login:
    wandb_var.finish()

# %%
