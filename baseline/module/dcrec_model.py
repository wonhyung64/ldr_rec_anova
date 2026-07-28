import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ResidualBase, PositionalEncoding


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        out = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return out.transpose(-1, -2)


class DCRec(ResidualBase):
    """
    Debiased Contrastive Learning for Sequential Recommendation.
    Implements intent-disentangled augmentation and debiased InfoNCE.
    Backbone: SASRec-style self-attention encoder.
    """

    def __init__(self, *args, n_intents=128, lambda_cl=0.1, aug_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_intents = n_intents
        self.lambda_cl = lambda_cl
        self.aug_ratio = aug_ratio
        self.mask_token_id = self.num_items + 1  # padding=num_items, mask=num_items+1

        # item embedding supports padding_idx (num_items) and mask token (num_items+1)
        self.item_embedding = nn.Embedding(self.num_items + 2, self.embedding_k, padding_idx=self.padding_item_id)

        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k, self.padding_item_id)
        self.emb_dropout = nn.Dropout(p=self.dropout)

        n_layers = max(2, self.depth)
        self.attention_layernorms = nn.ModuleList([nn.LayerNorm(self.embedding_k, eps=1e-8) for _ in range(n_layers)])
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(self.embedding_k, self.n_heads, self.dropout) for _ in range(n_layers)])
        self.forward_layernorms = nn.ModuleList([nn.LayerNorm(self.embedding_k, eps=1e-8) for _ in range(n_layers)])
        self.forward_layers = nn.ModuleList([PointWiseFeedForward(self.embedding_k, self.dropout) for _ in range(n_layers)])
        self.last_layernorm = nn.LayerNorm(self.embedding_k, eps=1e-8)

        # Intent prototype embeddings (K x d)
        self.intent_prototypes = nn.Embedding(n_intents, self.embedding_k)
        nn.init.xavier_uniform_(self.intent_prototypes.weight)

    def _sasrec_forward(self, hist_item_idx):
        seqs = self.item_embedding(hist_item_idx)
        seqs *= self.embedding_k ** 0.5
        seqs = seqs + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)

        for attn_ln, attn_layer, fwd_ln, fwd_layer in zip(
            self.attention_layernorms, self.attention_layers,
            self.forward_layernorms, self.forward_layers
        ):
            seqs_t = seqs.transpose(0, 1)
            x = attn_ln(seqs_t)
            mha_out, _ = attn_layer(x, x, x)
            seqs = seqs + mha_out.transpose(0, 1)
            seqs = seqs + fwd_layer(fwd_ln(seqs))

        return self.last_layernorm(seqs)[:, -1, :]  # last position

    def encode_user(self, hist_item_idx, user_idx=None):
        return self._sasrec_forward(hist_item_idx)

    # ------------------------------------------------------------------ #
    # Augmentation operations                                              #
    # ------------------------------------------------------------------ #

    def _aug_crop(self, seq):
        """Keep a contiguous subsequence of ratio aug_ratio."""
        B, L = seq.shape
        keep = max(1, int(L * (1 - self.aug_ratio)))
        out = seq.clone()
        for b in range(B):
            valid = (seq[b] != self.padding_item_id).nonzero(as_tuple=True)[0]
            n = len(valid)
            if n <= 1:
                continue
            crop_len = max(1, int(n * (1 - self.aug_ratio)))
            start = torch.randint(0, n - crop_len + 1, (1,)).item()
            selected = valid[start: start + crop_len]
            new_row = torch.full((L,), self.padding_item_id, dtype=seq.dtype, device=seq.device)
            new_row[-crop_len:] = seq[b][selected]
            out[b] = new_row
        return out

    def _aug_mask(self, seq):
        """Randomly replace items with mask token."""
        B, L = seq.shape
        out = seq.clone()
        mask = (seq != self.padding_item_id) & (torch.rand(B, L, device=seq.device) < self.aug_ratio)
        out[mask] = self.mask_token_id
        return out

    def _aug_reorder(self, seq):
        """Shuffle a random contiguous subsequence."""
        B, L = seq.shape
        out = seq.clone()
        for b in range(B):
            valid = (seq[b] != self.padding_item_id).nonzero(as_tuple=True)[0]
            n = len(valid)
            if n <= 1:
                continue
            reorder_len = max(1, int(n * self.aug_ratio))
            start = torch.randint(0, n - reorder_len + 1, (1,)).item()
            idxs = valid[start: start + reorder_len]
            perm = idxs[torch.randperm(len(idxs))]
            out[b][idxs] = seq[b][perm]
        return out

    def augment(self, seq):
        aug_type = torch.randint(0, 3, (1,)).item()
        if aug_type == 0:
            return self._aug_crop(seq)
        elif aug_type == 1:
            return self._aug_mask(seq)
        else:
            return self._aug_reorder(seq)

    # ------------------------------------------------------------------ #
    # Intent-disentangled debiased contrastive loss                        #
    # ------------------------------------------------------------------ #

    def _intent_distribution(self, u):
        """Soft intent assignment: p_k = softmax(u · e_k / τ)."""
        e = self.intent_prototypes.weight  # (K, d)
        u_norm = F.normalize(u, dim=-1, eps=1e-8)
        e_norm = F.normalize(e, dim=-1, eps=1e-8)
        logits = u_norm @ e_norm.T / self.tau  # (B, K)
        return F.softmax(logits, dim=-1)

    def debiased_cl_loss(self, seq1, seq2):
        """
        Debiased intent-level InfoNCE between two augmented views.

        False-negative weight w_ij = 1 - cos_sim(p_i, p_j) where p is the
        intent distribution. This down-weights in-batch negatives that share
        similar intents with the anchor.
        """
        u1 = self._sasrec_forward(seq1)  # (B, d)
        u2 = self._sasrec_forward(seq2)  # (B, d)

        # Intent distributions
        p1 = self._intent_distribution(u1)  # (B, K)
        p2 = self._intent_distribution(u2)  # (B, K)

        # Normalize user representations for cosine similarity
        z1 = F.normalize(u1, dim=-1, eps=1e-8)
        z2 = F.normalize(u2, dim=-1, eps=1e-8)

        # Pairwise similarity matrix: (B, B)
        sim = torch.matmul(z1, z2.T) / self.tau

        # False-negative probability via intent distribution similarity (B, B)
        intent_sim = torch.matmul(p1, p2.T)  # cosine between prob vectors (already normalized by softmax)
        # Weight: 1 = true negative, 0 = likely false negative
        fn_weight = 1.0 - intent_sim
        fn_weight = fn_weight.clamp(min=0.0)

        B = z1.shape[0]
        diag_mask = torch.eye(B, dtype=torch.bool, device=z1.device)

        # Debiased denominator: positives on diagonal, negatives weighted by fn_weight
        pos_sim = sim[diag_mask]  # (B,)

        # Mask diagonal when computing denominator
        neg_sim = sim.masked_fill(diag_mask, float('-inf'))  # (B, B)
        # Apply false-negative debiasing: subtract log(fn_weight) effectively
        # or equivalently weight: exp(s_ij) * w_ij
        # Use log-sum-exp with weighting
        neg_weighted = neg_sim + torch.log(fn_weight.clamp(min=1e-8))
        neg_weighted = neg_weighted.masked_fill(diag_mask, float('-inf'))

        log_denom = torch.logsumexp(
            torch.cat([pos_sim.unsqueeze(1), neg_weighted], dim=1), dim=1
        )
        loss = -(pos_sim - log_denom).mean()
        return loss
