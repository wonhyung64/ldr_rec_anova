import torch
import torch.nn as nn
import torch.nn.functional as F
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


class R2Rec(nn.Module):
    """
    R2Rec: Reembedding and Reweighting are Needed for Tail Item Sequential Recommendation

    Key ideas:
    1. Reembedding: Tail item embeddings are replaced by a softmax attention-weighted
       combination of K learnable prototype vectors, which are shared across all items
       and thus benefit from aggregate interaction signal.
    2. Reweighting: BPR training loss is scaled per sample by the inverse frequency
       of the positive item, so tail items receive higher gradient signal.

    Ranking score: normalize(u) · normalize(ẽ_i) / τ
    where ẽ_i = prototype-reembedded representation for tail items, original otherwise.
    """

    def __init__(self, num_users, num_items, embedding_k, device,
                 tau=1.0, depth=2, max_seq_len=50, n_heads=2, dropout=0.1,
                 n_proto=128, norm_first=True,
                 tail_item_ids=None, item_counts=None, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.max_seq_len = max_seq_len
        self.padding_item_id = num_items
        self.norm_first = norm_first
        self.n_proto = n_proto

        self.item_embedding = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)
        self.user_embedding = nn.Embedding(num_users, embedding_k)

        # SASRec backbone
        self.pos_enc = PositionalEncoding(max_seq_len, embedding_k, num_items)
        self.emb_dropout = nn.Dropout(p=dropout)

        n_layers = max(2, depth)
        self.attention_layernorms = nn.ModuleList(
            [nn.LayerNorm(embedding_k, eps=1e-8) for _ in range(n_layers)]
        )
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(embedding_k, n_heads, dropout=dropout, batch_first=False) for _ in range(n_layers)]
        )
        self.forward_layernorms = nn.ModuleList(
            [nn.LayerNorm(embedding_k, eps=1e-8) for _ in range(n_layers)]
        )
        self.forward_layers = nn.ModuleList(
            [PointWiseFeedForward(embedding_k, dropout) for _ in range(n_layers)]
        )
        self.last_layernorm = nn.LayerNorm(embedding_k, eps=1e-8)

        # Prototype bank for reembedding tail items
        self.proto_bank = nn.Embedding(n_proto, embedding_k)
        nn.init.xavier_uniform_(self.proto_bank.weight)

        # Tail item mask (True = tail item)
        tail_i = torch.zeros(num_items, dtype=torch.bool)
        if tail_item_ids is not None:
            tail_i[torch.tensor(tail_item_ids, dtype=torch.long)] = True
        self.register_buffer('tail_item_mask', tail_i)

        # Per-item reweighting: max_count / count(i), clamped to [1, max_weight]
        if item_counts is not None:
            counts = torch.tensor(item_counts, dtype=torch.float32).clamp(min=1.0)
            weights = counts.max() / counts
            self.register_buffer('item_weights', weights)
        else:
            self.register_buffer('item_weights', torch.ones(num_items, dtype=torch.float32))

    def _reembed_items(self, item_idx):
        """
        Reembedding: replace tail item embeddings with prototype-weighted representations.
        For head items the original embedding is returned unchanged.
        """
        e = self.item_embedding(item_idx.clamp(0, self.num_items - 1))  # [N, d]
        P = self.proto_bank.weight                                        # [K, d]
        attn = e @ P.T / (self.embedding_k ** 0.5)                       # [N, K]
        e_reembed = F.softmax(attn, dim=-1) @ P                          # [N, d]

        is_tail = self.tail_item_mask[item_idx.clamp(0, self.num_items - 1)].unsqueeze(-1).float()
        return e * (1.0 - is_tail) + e_reembed * is_tail

    def _sasrec_encode(self, seqs):
        """SASRec transformer layers. seqs: [B, L, d] → [B, d]"""
        B, L, d = seqs.shape
        # causal mask
        attn_mask = torch.triu(torch.ones(L, L, device=seqs.device), diagonal=1).bool()

        for i in range(len(self.attention_layers)):
            seqs_t = seqs.transpose(0, 1)  # [L, B, d]
            if self.norm_first:
                x = self.attention_layernorms[i](seqs_t)
                out, _ = self.attention_layers[i](x, x, x, attn_mask=attn_mask)
                seqs_t = seqs_t + out
                seqs = seqs_t.transpose(0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                out, _ = self.attention_layers[i](seqs_t, seqs_t, seqs_t, attn_mask=attn_mask)
                seqs_t = self.attention_layernorms[i](seqs_t + out)
                seqs = seqs_t.transpose(0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        return self.last_layernorm(seqs)[:, -1, :]  # [B, d]

    def encode_user(self, hist_item_idx, user_idx=None):
        """Encode user from interaction history using SASRec."""
        B, L = hist_item_idx.shape
        is_pad = (hist_item_idx == self.padding_item_id)

        hist_clamped = hist_item_idx.clamp(0, self.num_items - 1)
        hist_flat = hist_clamped.reshape(-1)
        enh_flat = self._reembed_items(hist_flat)

        pad_emb = self.item_embedding(
            torch.full((B * L,), self.padding_item_id, device=self.device, dtype=torch.long)
        )
        enh_flat = torch.where(is_pad.reshape(-1).unsqueeze(-1), pad_emb, enh_flat)

        seqs = enh_flat.reshape(B, L, self.embedding_k)
        seqs = seqs * (self.embedding_k ** 0.5) + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)

        return self._sasrec_encode(seqs)  # [B, d]

    def residual_score(self, item_idx, hist_item_idx, user_idx):
        """BPR-style pairwise score. item_idx: [B] or [B, K]."""
        u = self.encode_user(hist_item_idx, user_idx)   # [B, d]
        B = u.shape[0]
        v = self._reembed_items(item_idx.reshape(-1)).reshape(B, -1, self.embedding_k)  # [B, K, d]

        u_n = F.normalize(u, dim=-1, eps=1e-8).unsqueeze(1)   # [B, 1, d]
        v_n = F.normalize(v, dim=-1, eps=1e-8)                  # [B, K, d]
        return (u_n * v_n).sum(-1) / self.tau                   # [B, K]

    def score_all_items(self, hist_item_idx, user_idx):
        """Score user against all items."""
        u = self.encode_user(hist_item_idx, user_idx)  # [B, d]

        if hasattr(self, '_cached_item_embs'):
            v_all = self._cached_item_embs
        else:
            all_idx = torch.arange(self.num_items, device=self.device)
            v_all = self._reembed_items(all_idx)        # [num_items, d]

        u_n = F.normalize(u, dim=-1, eps=1e-8)
        v_n = F.normalize(v_all, dim=-1, eps=1e-8)
        return torch.matmul(u_n, v_n.T) / self.tau    # [B, num_items]

    def precompute_item_embeddings(self, batch_size=512):
        all_idx = torch.arange(self.num_items, device=self.device)
        chunks = []
        for i in range(0, self.num_items, batch_size):
            with torch.no_grad():
                chunks.append(self._reembed_items(all_idx[i:i + batch_size]))
        self._cached_item_embs = torch.cat(chunks, dim=0)

    def clear_item_cache(self):
        if hasattr(self, '_cached_item_embs'):
            del self._cached_item_embs
