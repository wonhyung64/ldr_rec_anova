import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(users, items, num_users, num_items, device):
    """Build D^{-1/2} A D^{-1/2} for a bipartite user-item graph."""
    N = num_users + num_items
    row = np.concatenate([users, items + num_users])
    col = np.concatenate([items + num_users, users])
    data = np.ones(len(row), dtype=np.float32)
    A = sp.csr_matrix((data, (row, col)), shape=(N, N))

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    norm_A = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.stack([norm_A.row, norm_A.col], axis=0).astype(np.int64)
    )
    values = torch.from_numpy(norm_A.data)
    return torch.sparse_coo_tensor(indices, values, (N, N)).to(device)


class DDCLightGCN(nn.Module):
    """
    LightGCN backbone for DDC.
    Training: standard BPR on all interactions.
    Inference: DDC analytical debiasing applied to propagated item embeddings.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_k,
        device,
        n_layers=3,
        tau=1.0,
        adj=None,
        **kwargs,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.n_layers = n_layers
        self.tau = tau

        self.user_emb = nn.Embedding(num_users, embedding_k)
        self.item_emb = nn.Embedding(num_items, embedding_k)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.register_buffer("_adj_indices", adj._indices() if adj is not None else torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("_adj_values",  adj._values()  if adj is not None else torch.zeros(0))
        self._adj_size = (num_users + num_items, num_users + num_items)

    def _get_adj(self):
        return torch.sparse_coo_tensor(self._adj_indices, self._adj_values, self._adj_size)

    def propagate(self):
        """LightGCN propagation; returns (user_emb, item_emb) as layer mean."""
        adj = self._get_adj()
        e = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj, e)
            layers.append(e)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users:]

    def bpr_score(self, user_idx, item_idx):
        """Pair score for BPR training (no debiasing)."""
        u_all, v_all = self.propagate()
        u = F.normalize(u_all[user_idx], dim=-1)
        v = F.normalize(v_all[item_idx], dim=-1)
        if v.dim() == 2:
            return (u * v).sum(-1) / self.tau           # [B]
        else:
            return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau  # [B, N]

    def score_all_items(self, user_idx):
        """Standard (biased) score against all items — used in baseline eval."""
        u_all, v_all = self.propagate()
        u = F.normalize(u_all[user_idx], dim=-1)
        v = F.normalize(v_all, dim=-1)
        return torch.matmul(u, v.T) / self.tau  # [B, M]

    def ddc_score_all(self, user_idx, pop_directions):
        """DDC debiased score: project out popularity directions from item embeddings."""
        u_all, v_all = self.propagate()
        u = F.normalize(u_all[user_idx], dim=-1)

        v_int = v_all
        for d in pop_directions:
            v_int = v_int - (v_int @ d).unsqueeze(-1) * d.unsqueeze(0)

        u = F.normalize(u, dim=-1, eps=1e-8)
        v_int = F.normalize(v_int, dim=-1, eps=1e-8)
        return torch.matmul(u, v_int.T) / self.tau  # [B, M]
