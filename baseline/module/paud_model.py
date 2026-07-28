import torch
import torch.nn as nn
import torch.nn.functional as F


class PAUDModel(nn.Module):
    """
    Popularity-aware Sequential Recommendation with User Desire (PAUD).

    Decomposes interaction into:
      - Desire  : user sequential preference  (GRU encoder)
      - Conformity: per-user scalar conformity weight * log item popularity

    Ranking score:
      s(u, i) = cos_sim(desire(u), e_i) / tau  +  c_u * log_pop(i)

    Reference: "Popularity-aware Sequential Recommendation with User Desire"
    """

    def __init__(self, num_users, num_items, embedding_k, device,
                 tau=0.1, depth=2, max_seq_len=50, dropout=0.2,
                 log_pop=None, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.padding_item_id = num_items

        self.item_embedding = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)
        num_layers = max(1, depth)
        self.gru = nn.GRU(
            input_size=embedding_k,
            hidden_size=embedding_k,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # per-user conformity scalar
        self.user_conformity = nn.Embedding(num_users, 1)

        if log_pop is not None:
            self.register_buffer('log_pop', log_pop.float())
        else:
            self.register_buffer('log_pop', torch.zeros(num_items))

        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_conformity.weight)

    # ------------------------------------------------------------------
    def _desire_repr(self, hist_item_idx):
        """GRU last hidden state as user desire. hist_item_idx: [B, L]"""
        emb = self.item_embedding(hist_item_idx)
        _, h_n = self.gru(emb)
        return h_n[-1]  # [B, D]

    def _desire_score_pair(self, desire, item_idx):
        """item_idx: [B] or [B, N]"""
        u = F.normalize(desire, dim=-1)
        if item_idx.dim() == 1:
            v = F.normalize(self.item_embedding(item_idx), dim=-1)
            return (u * v).sum(-1) / self.tau          # [B]
        v = F.normalize(self.item_embedding(item_idx), dim=-1)  # [B, N, D]
        return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau  # [B, N]

    def _conformity_score_pair(self, user_idx, item_idx):
        """item_idx: [B] or [B, N]"""
        c = self.user_conformity(user_idx)  # [B, 1]
        if item_idx.dim() == 1:
            return (c * self.log_pop[item_idx].unsqueeze(-1)).squeeze(-1)  # [B]
        return c * self.log_pop[item_idx]  # [B, N]

    # ------------------------------------------------------------------
    def score_pair(self, hist_item_idx, user_idx, item_idx):
        desire = self._desire_repr(hist_item_idx)
        return (self._desire_score_pair(desire, item_idx)
                + self._conformity_score_pair(user_idx, item_idx))

    def score_all_items(self, hist_item_idx, user_idx):
        desire = self._desire_repr(hist_item_idx)
        u = F.normalize(desire, dim=-1)
        v_all = F.normalize(
            self.item_embedding(torch.arange(self.num_items, device=self.device)), dim=-1
        )
        desire_all = torch.matmul(u, v_all.T) / self.tau   # [B, num_items]

        c = self.user_conformity(user_idx)                  # [B, 1]
        conformity_all = c * self.log_pop.unsqueeze(0)      # [B, num_items]

        return desire_all + conformity_all
