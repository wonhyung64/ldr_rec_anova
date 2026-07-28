import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base import PositionalEncoding


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x.transpose(-1, -2))))))
        return out.transpose(-1, -2)


class MELT(nn.Module):
    """
    MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation
    SIGIR 2023

    Mutual enhancement mechanism:
    1. Tail item enhancement: augment item embeddings with attention-weighted user embeddings
       from users who have interacted with the item.
    2. Tail user enhancement: augment user sequence representations with attention-weighted
       prototypes from a learnable prototype bank.
    Both enhancements are applied only to tail users/items (below-median interaction count).

    Ranking score: normalize(u_enhanced) · normalize(i_enhanced) / tau
    """
    def __init__(self, num_users, num_items, embedding_k, device,
                 tau=1.0, depth=2, max_seq_len=50, n_heads=2, dropout=0.1,
                 melt_alpha=1.0, n_proto=128, norm_first=True,
                 tail_user_ids=None, tail_item_ids=None,
                 item_user_pad=None, item_user_mask=None, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.melt_alpha = melt_alpha
        self.n_proto = n_proto
        self.max_seq_len = max_seq_len
        self.padding_item_id = num_items
        self.norm_first = norm_first

        self.item_embedding = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)
        self.user_embedding = nn.Embedding(num_users, embedding_k)

        # SASRec backbone
        self.pos_enc = PositionalEncoding(max_seq_len, embedding_k, num_items)
        self.emb_dropout = nn.Dropout(p=dropout)

        n_layers = max(2, depth)
        self.attention_layernorms = nn.ModuleList([nn.LayerNorm(embedding_k, eps=1e-8) for _ in range(n_layers)])
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(embedding_k, n_heads, dropout) for _ in range(n_layers)])
        self.forward_layernorms = nn.ModuleList([nn.LayerNorm(embedding_k, eps=1e-8) for _ in range(n_layers)])
        self.forward_layers = nn.ModuleList([PointWiseFeedForward(embedding_k, dropout) for _ in range(n_layers)])
        self.last_layernorm = nn.LayerNorm(embedding_k, eps=1e-8)

        # Learnable prototype bank for tail user enhancement
        self.proto_bank = nn.Embedding(n_proto, embedding_k)
        nn.init.xavier_uniform_(self.proto_bank.weight)

        # Projections for item enhancement (item queries interacting-user keys)
        self.item_q = nn.Linear(embedding_k, embedding_k, bias=False)
        self.user_k = nn.Linear(embedding_k, embedding_k, bias=False)

        # Projections for user enhancement (user queries prototype keys)
        self.user_q = nn.Linear(embedding_k, embedding_k, bias=False)
        self.proto_k = nn.Linear(embedding_k, embedding_k, bias=False)

        # Tail masks: True = tail (should be enhanced)
        tail_u = torch.zeros(num_users, dtype=torch.bool)
        if tail_user_ids is not None:
            tail_u[torch.tensor(tail_user_ids, dtype=torch.long)] = True
        self.register_buffer('tail_user_mask', tail_u)

        tail_i = torch.zeros(num_items, dtype=torch.bool)
        if tail_item_ids is not None:
            tail_i[torch.tensor(tail_item_ids, dtype=torch.long)] = True
        self.register_buffer('tail_item_mask', tail_i)

        # Padded item→user interaction table: [num_items, max_u_per_item]
        if item_user_pad is not None:
            self.register_buffer('item_user_pad', torch.tensor(item_user_pad, dtype=torch.long))
            self.register_buffer('item_user_valid', torch.tensor(item_user_mask, dtype=torch.bool))
        else:
            self.register_buffer('item_user_pad', torch.zeros(num_items, 1, dtype=torch.long))
            self.register_buffer('item_user_valid', torch.zeros(num_items, 1, dtype=torch.bool))

    def _enhance_items(self, item_idx):
        """
        Compute enhanced item embeddings for items in item_idx: [N].
        For tail items: base_emb + alpha * attn_pool(interacting user embeddings).
        For head items: base_emb unchanged.
        """
        base = self.item_embedding(item_idx)                                      # [N, k]

        u_idx = self.item_user_pad[item_idx]                                      # [N, max_u]
        u_valid = self.item_user_valid[item_idx]                                  # [N, max_u]

        u_embs = self.user_embedding(u_idx.clamp(0, self.num_users - 1))          # [N, max_u, k]
        u_embs = u_embs * u_valid.unsqueeze(-1).float()

        q = self.item_q(base).unsqueeze(1)                                        # [N, 1, k]
        k = self.user_k(u_embs)                                                   # [N, max_u, k]
        attn = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / (self.embedding_k ** 0.5)  # [N, max_u]
        attn = attn.masked_fill(~u_valid, float('-inf'))

        has_any = u_valid.any(-1)                                                  # [N]
        safe_attn = attn.clone()
        safe_attn[~has_any] = 0.0
        weights = torch.where(
            has_any.unsqueeze(-1),
            F.softmax(safe_attn, dim=-1),
            torch.zeros_like(safe_attn),
        )
        context = torch.bmm(weights.unsqueeze(1), u_embs).squeeze(1)              # [N, k]

        is_tail = self.tail_item_mask[item_idx].unsqueeze(-1).float()              # [N, 1]
        return base + self.melt_alpha * context * is_tail

    def _sasrec_encode(self, seqs):
        """SASRec transformer layers over [B, L, k] sequence embeddings."""
        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                out, _ = self.attention_layers[i](x, x, x)
                seqs = seqs + out
                seqs = seqs.transpose(0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                out, _ = self.attention_layers[i](seqs, seqs, seqs)
                seqs = self.attention_layernorms[i](seqs + out)
                seqs = seqs.transpose(0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))
        return self.last_layernorm(seqs)[:, -1, :]

    def _encode_user(self, hist_item_idx, user_idx):
        """
        Encode user with mutual enhancement.

        Step 1 (item→user): Run SASRec over enhanced history item embeddings.
        Step 2 (user enhancement): For tail users, augment sequence repr via prototype bank.
        """
        B, L = hist_item_idx.shape

        # Step 1: get enhanced embeddings for all items in history
        hist_flat = hist_item_idx.reshape(-1)                                      # [B*L]
        is_pad = (hist_flat == self.padding_item_id)
        hist_clamped = hist_flat.clamp(0, self.num_items - 1)

        enh_flat = self._enhance_items(hist_clamped)                               # [B*L, k]
        pad_emb = self.item_embedding(
            torch.full((B * L,), self.padding_item_id, device=self.device, dtype=torch.long)
        )
        enh_flat = torch.where(is_pad.unsqueeze(-1), pad_emb, enh_flat)

        seqs = enh_flat.reshape(B, L, self.embedding_k)
        seqs = seqs * (self.embedding_k ** 0.5) + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)

        u_repr = self._sasrec_encode(seqs)                                         # [B, k]

        # Step 2: enhance tail users via prototype bank
        q = self.user_q(u_repr)                                                    # [B, k]
        k = self.proto_k(self.proto_bank.weight)                                   # [n_proto, k]
        attn = torch.matmul(q, k.T) / (self.embedding_k ** 0.5)                  # [B, n_proto]
        context = torch.matmul(F.softmax(attn, dim=-1), self.proto_bank.weight)   # [B, k]

        is_tail = self.tail_user_mask[user_idx].unsqueeze(-1).float()             # [B, 1]
        return u_repr + self.melt_alpha * context * is_tail                        # [B, k]

    def residual_score(self, item_idx, hist_item_idx, user_idx):
        """BPR-style score for (user, item) pairs. item_idx: [B] or [B, K]."""
        u = self._encode_user(hist_item_idx, user_idx)                            # [B, k]
        B = u.shape[0]

        v = self._enhance_items(item_idx.reshape(-1)).reshape(B, -1, self.embedding_k)  # [B, K, k]

        u_n = F.normalize(u, dim=-1, eps=1e-8).unsqueeze(1)                       # [B, 1, k]
        v_n = F.normalize(v, dim=-1, eps=1e-8)                                    # [B, K, k]
        return (u_n * v_n).sum(-1) / self.tau                                     # [B, K]

    def score_all_items(self, hist_item_idx, user_idx):
        """Score user against all items. Uses cached item embeddings if available."""
        u = self._encode_user(hist_item_idx, user_idx)                            # [B, k]

        if hasattr(self, '_cached_item_embs'):
            v_all = self._cached_item_embs
        else:
            all_idx = torch.arange(self.num_items, device=self.device)
            v_all = self._enhance_items(all_idx)                                  # [num_items, k]

        u_n = F.normalize(u, dim=-1, eps=1e-8)
        v_n = F.normalize(v_all, dim=-1, eps=1e-8)
        return torch.matmul(u_n, v_n.T) / self.tau                               # [B, num_items]

    def precompute_item_enhancements(self, batch_size=512):
        """Precompute and cache enhanced embeddings for all items before eval loop."""
        all_idx = torch.arange(self.num_items, device=self.device)
        chunks = []
        for i in range(0, self.num_items, batch_size):
            with torch.no_grad():
                chunks.append(self._enhance_items(all_idx[i:i + batch_size]))
        self._cached_item_embs = torch.cat(chunks, dim=0)

    def clear_item_cache(self):
        if hasattr(self, '_cached_item_embs'):
            del self._cached_item_embs
