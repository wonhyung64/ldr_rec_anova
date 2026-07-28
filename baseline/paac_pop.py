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
from module.dataset_pop import UserItemTime
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
    wandb_var = wandb.init(project="ldr_rec_pop", config=vars(args))
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
else:
    raise ValueError

#%%
eval_datasets = [
    ("head_overall", dataset.test_head_overall_dict),
    ("head_recent_3d", dataset.test_head_recent_3d_dict),
    ("head_recent_7d", dataset.test_head_recent_7d_dict),
    ("tail_overall", dataset.test_tail_overall_dict),
    ("tail_recent_3d", dataset.test_tail_recent_3d_dict),
    ("tail_recent_7d", dataset.test_tail_recent_7d_dict),
]


for (split_name, data_split) in eval_datasets:

    pred_list, gt_list = [], []
    model.eval()

    for (user, item), pos_time_val in dataset.set_to_pair(data_split, dataset.time_dict, dataset.time_unit).items():
        user_t = torch.tensor([user], dtype=torch.long, device=args.device)

        with torch.no_grad():
            scores = model.score_all_items(user_t).squeeze(0).cpu()

        exclude_items = list(dataset._allPos[user])
        scores[exclude_items] = -9999
        _, pred_k = torch.topk(scores, k=max(args.topks))
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
