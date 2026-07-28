import torch
import torch.nn as nn
import torch.nn.functional as F
from .mf import MF
from .ncf import NCF
from .grurec import GRURec
from .sasrec import SASRec
# from .mlp4rec import MLP4Rec
# from .bert4rec import BERT4Rec
from .tisasrec import TiSASRec
from .fearec import FEARec
from .bsarec import BSARec


MODEL_REGISTRY = {
    "mf": MF,
    "ncf": NCF,
    "grurec": GRURec,
    "sasrec": SASRec,
    "tisasrec": TiSASRec,
    "fearec": FEARec,
    "bsarec": BSARec,
}


def build_model(args, dataset, mini_batch):
    model_name = getattr(args, "model_name", "grurec").lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name={model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(
        num_users=dataset.n_user,
        num_items=dataset.m_item,
        embedding_k=args.recdim,
        device=args.device,
        tau=getattr(args, "tau", 1.0),
        depth=getattr(args, "depth", 1),
        max_seq_len=getattr(args, "max_seq_len", 50),
        n_heads=getattr(args, "n_heads", 2),
        dropout=getattr(args, "dropout", 0.1),
    ).to(args.device)


def score_pair(model, item_idx, hist_item_idx, additional_feat):
    return model.residual_score(item_idx, hist_item_idx, additional_feat)


def score_all(model, hist_item_idx, additional_feat):
    return model.score_all_items(hist_item_idx, additional_feat)

