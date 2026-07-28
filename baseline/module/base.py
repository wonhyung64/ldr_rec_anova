import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBase(nn.Module):
    def __init__(self, num_users, num_items, embedding_k, device, tau=1.0, depth=1, max_seq_len=50, n_heads=2, dropout=0.1, norm_first = True, score_norm="normalized"):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dropout = dropout
        self.padding_item_id = self.num_items
        self.norm_first = norm_first
        self.score_norm = score_norm

        self.item_embedding = nn.Embedding(self.num_items + 1, self.embedding_k, padding_idx=self.padding_item_id)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)

    def encode_user(self, hist_item_idx, user_idx=None):
        raise NotImplementedError

    def get_item_repr(self, item_idx):
        return self.item_embedding(item_idx)

    def residual_score(self, item_idx, hist_item_idx, additional_feat):
        u = self.encode_user(hist_item_idx, additional_feat)
        mini_batch, recdim = u.shape
        v = self.get_item_repr(item_idx).reshape(mini_batch, -1, recdim)
        if self.score_norm == "normalized":
            u = F.normalize(u, dim=-1, eps=1e-8)
            v = F.normalize(v, dim=-1, eps=1e-8)
            return torch.sum(u.unsqueeze(1) * v, dim=-1) / self.tau
        return torch.sum(u.unsqueeze(1) * v, dim=-1)

    def score_all_items(self, hist_item_idx, additional_feat):
        u = self.encode_user(hist_item_idx, additional_feat)
        v_all = self.get_item_repr(torch.arange(self.num_items, device=hist_item_idx.device))
        if self.score_norm == "normalized":
            u = F.normalize(u, dim=-1, eps=1e-8)
            v_all = F.normalize(v_all, dim=-1, eps=1e-8)
            return torch.matmul(u, v_all.T) / self.tau
        return torch.matmul(u, v_all.T)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model, padding_item_id):
        super().__init__()
        self.padding_item_id = padding_item_id
        self.pos_emb = torch.nn.Embedding(max_len+1, d_model)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(1, L+1, device=x.device).unsqueeze(0).expand(B, -1)
        hist_mask = x != self.padding_item_id
        pos = hist_mask * pos
        return self.pos_emb(pos)
