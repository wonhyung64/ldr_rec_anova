import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_item_popularity(train_events, num_items):
    """
    Compute normalised log-popularity for each item from training events.
    Returns a float32 tensor of shape [num_items], values in [0, 1].
    """
    counts = np.zeros(num_items, dtype=np.float32)
    for (_, v, _) in train_events:
        counts[v] += 1.0
    log_counts = np.log1p(counts)
    max_lc = log_counts.max()
    if max_lc > 0:
        log_counts /= max_lc
    return torch.from_numpy(log_counts)   # [M]


class TIDEModel(nn.Module):
    """
    TIDE: "Popularity bias is not always evil – Disentangling benign
    and harmful bias for recommendation."

    Causal decomposition
    --------------------
    Total score  = interest(u, v) + conformity(u) * pop(v)

    * interest(u, v) = dot(u_int, v_int) / tau
      captures genuine user-item affinity (benign).

    * conformity(u) * pop(v)
      = dot(u_con, v_con) / tau * pop_norm[v]
      captures the harmful bandwagon effect, where pop_norm[v] is the
      pre-computed normalised log-popularity of item v.

    Training  : optimise total score via BPR losses + discrepancy loss.
    Inference : replace each item's actual popularity with the global mean
                (do-calculus intervention), which removes the DIFFERENTIAL
                harmful effect while keeping the average popularity signal.

                score_debiased(u, v) = interest(u, v)
                                     + conformity(u) * mean_pop
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_k,
        device,
        tau=1.0,
        pop_scores=None,   # precomputed tensor [M], normalised log-pop
        **kwargs,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau

        # Interest embeddings
        self.int_user_emb = nn.Embedding(num_users, embedding_k)
        self.int_item_emb = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)
        # Conformity embeddings
        self.con_user_emb = nn.Embedding(num_users, embedding_k)
        self.con_item_emb = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)

        for emb in [self.int_user_emb, self.int_item_emb,
                    self.con_user_emb, self.con_item_emb]:
            nn.init.normal_(emb.weight, std=0.02)

        # Pre-computed item popularity (fixed, not learnable)
        if pop_scores is None:
            pop_scores = torch.zeros(num_items)
        # Register as buffer so it moves with .to(device) and is saved in checkpoints
        self.register_buffer("pop_scores", pop_scores.float())          # [M]
        self.register_buffer("mean_pop", pop_scores.float().mean().unsqueeze(0))  # [1]

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _interest(self, user_idx, item_idx):
        """dot(u_int, v_int); item_idx may be [B] or [B, N]."""
        u = F.normalize(self.int_user_emb(user_idx), dim=-1)
        if item_idx.dim() == 1:
            v = F.normalize(self.int_item_emb(item_idx), dim=-1)
            return (u * v).sum(-1) / self.tau                          # [B]
        else:
            v = F.normalize(self.int_item_emb(item_idx), dim=-1)      # [B, N, K]
            return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau  # [B, N]

    def _conformity(self, user_idx, item_idx):
        """dot(u_con, v_con); item_idx may be [B] or [B, N]."""
        u = F.normalize(self.con_user_emb(user_idx), dim=-1)
        if item_idx.dim() == 1:
            v = F.normalize(self.con_item_emb(item_idx), dim=-1)
            return (u * v).sum(-1) / self.tau
        else:
            v = F.normalize(self.con_item_emb(item_idx), dim=-1)
            return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau

    def _pop(self, item_idx):
        """Popularity weight for item_idx ([B] or [B, N])."""
        return self.pop_scores[item_idx]                               # same shape as item_idx

    # ------------------------------------------------------------------ #
    #  Training scores                                                     #
    # ------------------------------------------------------------------ #
    def total_score(self, user_idx, item_idx):
        """interest + conformity * pop  (used during training)."""
        return self._interest(user_idx, item_idx) + \
               self._conformity(user_idx, item_idx) * self._pop(item_idx)

    def interest_score(self, user_idx, item_idx):
        """Pure interest component (for interest BPR loss)."""
        return self._interest(user_idx, item_idx)

    def conformity_score(self, user_idx, item_idx):
        """Conformity component (for conformity BPR loss)."""
        return self._conformity(user_idx, item_idx) * self._pop(item_idx)

    def discrepancy_loss(self, user_idx, item_idx):
        """Cosine similarity penalty to keep int / con embeddings orthogonal."""
        u_int = F.normalize(self.int_user_emb(user_idx), dim=-1)
        u_con = F.normalize(self.con_user_emb(user_idx), dim=-1)
        v_int = F.normalize(self.int_item_emb(item_idx), dim=-1)
        v_con = F.normalize(self.con_item_emb(item_idx), dim=-1)
        return (u_int * u_con).sum(-1).abs().mean() + \
               (v_int * v_con).sum(-1).abs().mean()

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #
    def score_all_items(self, user_idx):
        """
        Debiased inference: do(pop = mean_pop).
        score = interest(u, v) + conformity(u, v) * mean_pop
        This removes differential harmful bias while keeping the
        average benign popularity signal.
        """
        u_int = F.normalize(self.int_user_emb(user_idx), dim=-1)       # [B, K]
        u_con = F.normalize(self.con_user_emb(user_idx), dim=-1)       # [B, K]

        v_int = F.normalize(
            self.int_item_emb(torch.arange(self.num_items, device=user_idx.device)),
            dim=-1,
        )                                                                 # [M, K]
        v_con = F.normalize(
            self.con_item_emb(torch.arange(self.num_items, device=user_idx.device)),
            dim=-1,
        )                                                                 # [M, K]

        interest  = torch.matmul(u_int, v_int.T) / self.tau             # [B, M]
        conformity = torch.matmul(u_con, v_con.T) / self.tau            # [B, M]
        return interest + conformity * self.mean_pop                     # [B, M]
