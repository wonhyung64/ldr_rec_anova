import torch
import torch.nn as nn
import torch.nn.functional as F


class PAACModel(nn.Module):
    """
    Popularity-Aware Alignment and Contrast (PAAC) for mitigating popularity bias.

    Reference: "Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias"

    Training objectives:
    - L_BPR   : standard BPR on interest embeddings
    - L_align : align user interest representations across popularity groups
    - L_con   : popularity-aware contrastive loss in popularity embedding space
    Inference : interest embeddings only (popularity signal discarded)
    """

    def __init__(self, num_users, num_items, embedding_k, device, tau=0.1, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau

        # Interest branch (used for final ranking)
        self.int_user_emb = nn.Embedding(num_users, embedding_k)
        self.int_item_emb = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)

        # Popularity branch (trained to capture conformity, discarded at inference)
        self.pop_item_emb = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)

        for emb in [self.int_user_emb, self.int_item_emb, self.pop_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _int_score(self, user_idx, item_idx):
        """Normalised dot-product in interest space. item_idx: [B] or [B, N]."""
        u = F.normalize(self.int_user_emb(user_idx), dim=-1)
        if item_idx.dim() == 1:
            v = F.normalize(self.int_item_emb(item_idx), dim=-1)
            return (u * v).sum(-1) / self.tau
        v = F.normalize(self.int_item_emb(item_idx), dim=-1)
        return torch.bmm(v, u.unsqueeze(-1)).squeeze(-1) / self.tau

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def bpr_loss(self, user_idx, pos_item_idx, neg_item_idx):
        """Standard BPR recommendation loss on interest embeddings."""
        pos = self._int_score(user_idx, pos_item_idx)         # [B]
        neg = self._int_score(user_idx, neg_item_idx)         # [B, N] or [B]
        if neg.dim() == 1:
            neg = neg.unsqueeze(-1)
        return -(F.logsigmoid(pos.unsqueeze(-1) - neg)).mean()

    def alignment_loss(self, user_idx, pop_item_idx, cold_item_idx):
        """
        Popularity-Aware Alignment (PA):
        User interest representations should be invariant to item popularity.
        Minimises distance between context-augmented user representations
        formed with popular vs. unpopular items.
        """
        u   = self.int_user_emb(user_idx)
        e_p = self.int_item_emb(pop_item_idx)
        e_c = self.int_item_emb(cold_item_idx)
        z_pop  = F.normalize(u + e_p, dim=-1)
        z_cold = F.normalize(u + e_c, dim=-1)
        return (z_pop - z_cold).pow(2).sum(-1).mean()

    def contrast_loss(self, user_idx, pop_item_idx, cold_item_idx):
        """
        Popularity-Aware Contrast (PC):
        In the popularity embedding space, popular items should score higher
        than cold items for the same user (captures conformity signal and keeps
        it separated from the interest branch).
        """
        u     = F.normalize(self.int_user_emb(user_idx), dim=-1)
        e_pop = F.normalize(self.pop_item_emb(pop_item_idx), dim=-1)
        e_cld = F.normalize(self.pop_item_emb(cold_item_idx), dim=-1)
        s_pos = (u * e_pop).sum(-1) / self.tau   # [B]
        s_neg = (u * e_cld).sum(-1) / self.tau   # [B]
        return -(F.logsigmoid(s_pos - s_neg)).mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def score_all_items(self, user_idx):
        """Interest-only score against all items (used at inference)."""
        u = F.normalize(self.int_user_emb(user_idx), dim=-1)
        v_all = F.normalize(
            self.int_item_emb(torch.arange(self.num_items, device=self.device)),
            dim=-1,
        )
        return torch.matmul(u, v_all.T) / self.tau
