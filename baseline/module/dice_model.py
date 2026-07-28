import torch
import torch.nn as nn
import torch.nn.functional as F


class DICEModel(nn.Module):
    """
    DICE: Disentangling User Interest and Conformity for Recommendation
    with Causal Embedding (WWW 2021).

    Maintains separate interest / conformity embeddings for users and items.
    During training: BPR on interest embeddings + BPR on conformity embeddings
                     + discrepancy loss to keep them orthogonal.
    During inference: interest-only score.
    """

    def __init__(self, num_users, num_items, embedding_k, device, tau=1.0, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau

        self.int_user_emb = nn.Embedding(num_users, embedding_k)
        self.int_item_emb = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)
        self.con_user_emb = nn.Embedding(num_users, embedding_k)
        self.con_item_emb = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)

        for emb in [self.int_user_emb, self.int_item_emb,
                    self.con_user_emb, self.con_item_emb]:
            nn.init.normal_(emb.weight, std=0.02)

    def _score(self, user_emb, item_emb_table, user_idx, item_idx):
        """Normalised dot-product score. item_idx: [B] or [B, N]."""
        u = F.normalize(user_emb(user_idx), dim=-1)         # [B, K]
        if item_idx.dim() == 1:
            v = F.normalize(item_emb_table(item_idx), dim=-1)   # [B, K]
            return (u * v).sum(-1) / self.tau                    # [B]
        else:
            v = F.normalize(item_emb_table(item_idx), dim=-1)   # [B, N, K]
            return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau  # [B, N]

    def interest_score(self, user_idx, item_idx):
        return self._score(self.int_user_emb, self.int_item_emb, user_idx, item_idx)

    def conformity_score(self, user_idx, item_idx):
        return self._score(self.con_user_emb, self.con_item_emb, user_idx, item_idx)

    def discrepancy_loss(self, user_idx, item_idx):
        """Penalise cosine similarity between interest and conformity embeddings."""
        u_int = F.normalize(self.int_user_emb(user_idx), dim=-1)
        u_con = F.normalize(self.con_user_emb(user_idx), dim=-1)
        v_int = F.normalize(self.int_item_emb(item_idx), dim=-1)
        v_con = F.normalize(self.con_item_emb(item_idx), dim=-1)
        return (u_int * u_con).sum(-1).abs().mean() + (v_int * v_con).sum(-1).abs().mean()

    def score_all_items(self, user_idx):
        """Inference: interest-only score against all items."""
        u = F.normalize(self.int_user_emb(user_idx), dim=-1)
        v_all = F.normalize(
            self.int_item_emb(torch.arange(self.num_items, device=user_idx.device)),
            dim=-1,
        )
        return torch.matmul(u, v_all.T) / self.tau   # [B, M]
