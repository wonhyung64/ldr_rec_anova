import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(users, items, num_users, num_items, device):
    """
    Build D^{-1/2} A D^{-1/2} for a bipartite user-item graph.
    Node ordering: [users (0..N-1), items (N..N+M-1)].
    Returns a torch sparse COO tensor on `device`.
    """
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


class DICELightGCN(nn.Module):
    """
    DICE (WWW 2021) with LightGCN as the backbone.

    Two separate LightGCN propagation paths:
      - Interest path: graph built from ALL training interactions
      - Conformity path: graph built from HOT (repeated / popular) interactions

    Training loss = lambda1 * L_int  +  (1-lambda1) * L_con  +  alpha * L_dis
    Inference     = interest-only score
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_k,
        device,
        n_layers=3,
        tau=1.0,
        int_adj=None,   # prebuilt normalised adjacency (interest graph)
        con_adj=None,   # prebuilt normalised adjacency (conformity graph)
        **kwargs,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.n_layers = n_layers
        self.tau = tau

        # Initial (layer-0) embeddings — separate for interest / conformity
        self.int_user_emb = nn.Embedding(num_users, embedding_k)
        self.int_item_emb = nn.Embedding(num_items, embedding_k)
        self.con_user_emb = nn.Embedding(num_users, embedding_k)
        self.con_item_emb = nn.Embedding(num_items, embedding_k)

        for emb in [self.int_user_emb, self.int_item_emb,
                    self.con_user_emb, self.con_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

        # Adjacency matrices stored as buffers (not model parameters)
        self.register_buffer("_int_adj_indices", int_adj._indices() if int_adj is not None else torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("_int_adj_values",  int_adj._values()  if int_adj is not None else torch.zeros(0))
        self.register_buffer("_con_adj_indices", con_adj._indices() if con_adj is not None else torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("_con_adj_values",  con_adj._values()  if con_adj is not None else torch.zeros(0))
        self._adj_size = (num_users + num_items, num_users + num_items)

    def _get_adj(self, which):
        if which == "int":
            return torch.sparse_coo_tensor(
                self._int_adj_indices, self._int_adj_values, self._adj_size
            )
        else:
            return torch.sparse_coo_tensor(
                self._con_adj_indices, self._con_adj_values, self._adj_size
            )

    def _propagate(self, adj, user_e0, item_e0):
        """LightGCN propagation; returns (user_emb, item_emb) as layer mean."""
        e = torch.cat([user_e0, item_e0], dim=0)   # [N+M, K]
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj, e)
            layers.append(e)
        final = torch.stack(layers, dim=0).mean(dim=0)  # [N+M, K]
        return final[: self.num_users], final[self.num_users :]

    def get_int_embeddings(self):
        adj = self._get_adj("int")
        return self._propagate(adj, self.int_user_emb.weight, self.int_item_emb.weight)

    def get_con_embeddings(self):
        adj = self._get_adj("con")
        return self._propagate(adj, self.con_user_emb.weight, self.con_item_emb.weight)

    # ------------------------------------------------------------------ #
    #  Scoring helpers                                                     #
    # ------------------------------------------------------------------ #
    def _score(self, u_emb, v_emb_table, user_idx, item_idx):
        u = F.normalize(u_emb[user_idx], dim=-1)
        if item_idx.dim() == 1:
            v = F.normalize(v_emb_table[item_idx], dim=-1)
            return (u * v).sum(-1) / self.tau
        else:
            v = F.normalize(v_emb_table[item_idx], dim=-1)   # [B, N, K]
            return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau

    def interest_score(self, user_idx, item_idx):
        u_all, v_all = self.get_int_embeddings()
        return self._score(u_all, v_all, user_idx, item_idx)

    def conformity_score(self, user_idx, item_idx):
        u_all, v_all = self.get_con_embeddings()
        return self._score(u_all, v_all, user_idx, item_idx)

    def discrepancy_loss(self, user_idx, item_idx):
        u_int_all, v_int_all = self.get_int_embeddings()
        u_con_all, v_con_all = self.get_con_embeddings()
        u_int = F.normalize(u_int_all[user_idx], dim=-1)
        u_con = F.normalize(u_con_all[user_idx], dim=-1)
        v_int = F.normalize(v_int_all[item_idx], dim=-1)
        v_con = F.normalize(v_con_all[item_idx], dim=-1)
        return (u_int * u_con).sum(-1).abs().mean() + (v_int * v_con).sum(-1).abs().mean()

    def score_all_items(self, user_idx):
        """Inference: interest-only score against all items."""
        u_all, v_all = self.get_int_embeddings()
        u = F.normalize(u_all[user_idx], dim=-1)
        v = F.normalize(v_all, dim=-1)
        return torch.matmul(u, v.T) / self.tau   # [B, M]
