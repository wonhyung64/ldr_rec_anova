import math
import torch
import torch.nn as nn
from .base import ResidualBase


class TiSASRec(ResidualBase):
    def __init__(self, *args, time_span=256, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_span = time_span
        self.emb_dropout = nn.Dropout(p=self.dropout)

        # absolute positional embeddings for K/V
        # 0 reserved for padding
        self.abs_pos_k_emb = nn.Embedding(self.max_seq_len + 1, self.embedding_k, padding_idx=0)
        self.abs_pos_v_emb = nn.Embedding(self.max_seq_len + 1, self.embedding_k, padding_idx=0)

        # time interval embeddings for K/V
        # 0 reserved for padding pair
        # actual delta d in [0, time_span] -> index d+1
        self.time_matrix_k_emb = nn.Embedding(self.time_span + 2, self.embedding_k, padding_idx=0)
        self.time_matrix_v_emb = nn.Embedding(self.time_span + 2, self.embedding_k, padding_idx=0)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.embedding_k, eps=1e-8)

        for _ in range(max(2, self.depth)):
            self.attention_layernorms.append(
                nn.LayerNorm(self.embedding_k, eps=1e-8)
            )
            self.attention_layers.append(
                TimeAwareMultiHeadSelfAttention(
                    self.embedding_k,
                    self.n_heads,
                    self.dropout,
                )
            )
            self.forward_layernorms.append(
                nn.LayerNorm(self.embedding_k, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(self.embedding_k, self.dropout)
            )

    def _build_position_ids(self, hist_item_idx):
        # [B, L]
        B, L = hist_item_idx.shape
        pos = torch.arange(1, L + 1, device=hist_item_idx.device).unsqueeze(0).expand(B, -1)
        valid = hist_item_idx.ne(self.padding_item_id)
        pos = pos * valid.long()
        return pos

    def _build_time_matrix(self, hist_timestamps, hist_item_idx):
        """
        hist_timestamps: [B, L] (already discretized integer timestamps)
        return: [B, L, L]
          0 -> padding pair
          1 -> interval 0
          ...
          time_span + 1 -> clipped max interval
        """
        hist_timestamps = hist_timestamps.long()
        valid = hist_item_idx.ne(self.padding_item_id)         # [B, L]
        pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)   # [B, L, L]
        delta = torch.abs(
            hist_timestamps.unsqueeze(2) - hist_timestamps.unsqueeze(1)
        )  # [B, L, L]
        delta = torch.clamp(delta, max=self.time_span)
        delta = delta + 1
        delta = delta * pair_valid.long()
        return delta

    def _get_last_hidden(self, seqs, hist_item_idx):
        """
        마지막 non-padding 위치의 hidden state를 가져온다.
        left-padding / right-padding 둘 다 안전하게 동작.
        seqs: [B, L, D]
        """
        B, L, D = seqs.shape
        valid = hist_item_idx.ne(self.padding_item_id)  # [B, L]
        pos = torch.arange(L, device=hist_item_idx.device).unsqueeze(0).expand(B, -1)
        last_idx = (pos * valid.long()).max(dim=1).values  # [B]
        return seqs[torch.arange(B, device=seqs.device), last_idx]  # [B, D]

    def _item_score(self, user_emb, item_idx):
        """
        user_emb: [B, D]
        item_idx: [B] or [B, N]
        return:   [B] or [B, N]
        """
        item_emb = self.item_embedding(item_idx)

        if item_emb.dim() == 2:
            # [B, D]
            return (user_emb * item_emb).sum(dim=-1)

        elif item_emb.dim() == 3:
            # [B, N, D]
            return (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)

        raise ValueError(f"Unexpected item_emb dim: {item_emb.dim()}")

    def encode_user(self, hist_item_idx, hist_timestamps):
        """
        hist_item_idx:   [B, L]
        hist_timestamps: [B, L]
        """
        timeline_mask = hist_item_idx.eq(self.padding_item_id)  # [B, L]

        seqs = self.item_embedding(hist_item_idx)
        seqs = seqs * (self.item_embedding.embedding_dim ** 0.5)
        seqs = self.emb_dropout(seqs)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        pos_ids = self._build_position_ids(hist_item_idx)               # [B, L]
        time_mat = self._build_time_matrix(hist_timestamps, hist_item_idx)  # [B, L, L]

        abs_pos_k = self.abs_pos_k_emb(pos_ids)     # [B, L, D]
        
        abs_pos_v = self.abs_pos_v_emb(pos_ids)     # [B, L, D]
        time_mat_k = self.time_matrix_k_emb(time_mat)  # [B, L, L, D]
        time_mat_v = self.time_matrix_v_emb(time_mat)  # [B, L, L, D]

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                attn_out = self.attention_layers[i](
                    x,
                    abs_pos_k,
                    abs_pos_v,
                    time_mat_k,
                    time_mat_v,
                    padding_mask=timeline_mask,
                )
                seqs = seqs + attn_out
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                attn_out = self.attention_layers[i](
                    seqs,
                    abs_pos_k,
                    abs_pos_v,
                    time_mat_k,
                    time_mat_v,
                    padding_mask=timeline_mask,
                )
                seqs = self.attention_layernorms[i](seqs + attn_out)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

            seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        seqs = self.last_layernorm(seqs)
        u = self._get_last_hidden(seqs, hist_item_idx)  # [B, D]
        return u


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(self.conv1(inputs.transpose(-1, -2)))
                )
            )
        )
        return outputs.transpose(-1, -2)


class TimeAwareMultiHeadSelfAttention(nn.Module):
    """
    TiSASRec-style attention without causal mask.
    Only padding mask is applied.

    score(i, j) =
        <Q_i, K_j>
      + <Q_i, P^K_j>
      + <Q_i, R^K_{i,j}>

    out(i) =
      sum_j a_ij * (V_j + P^V_j + R^V_{i,j})
    """
    def __init__(self, hidden_units, n_heads, dropout_rate):
        super().__init__()
        assert hidden_units % n_heads == 0

        self.hidden_units = hidden_units
        self.n_heads = n_heads
        self.head_dim = hidden_units // n_heads

        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.out_proj = nn.Linear(hidden_units, hidden_units)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_dropout = nn.Dropout(dropout_rate)

    def _split_heads(self, x):
        # [B, L, D] -> [B*H, L, Dh]
        B, L, D = x.shape
        x = x.view(B, L, self.n_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B * self.n_heads, L, self.head_dim)

    def _split_heads_2d(self, x):
        # [B, L, L, D] -> [B*H, L, L, Dh]
        B, L, _, D = x.shape
        x = x.view(B, L, L, self.n_heads, self.head_dim)
        x = x.permute(0, 3, 1, 2, 4).contiguous()
        return x.view(B * self.n_heads, L, L, self.head_dim)

    def _merge_heads(self, x, B, L):
        # [B*H, L, Dh] -> [B, L, D]
        x = x.view(B, self.n_heads, L, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, L, self.hidden_units)

    def forward(
        self,
        x,              # [B, L, D]
        abs_pos_k,      # [B, L, D]
        abs_pos_v,      # [B, L, D]
        time_mat_k,     # [B, L, L, D]
        time_mat_v,     # [B, L, L, D]
        padding_mask,   # [B, L], True if padding
    ):
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)                    # [BH, L, Dh]
        k = self._split_heads(k)                    # [BH, L, Dh]
        v = self._split_heads(v)                    # [BH, L, Dh]
        abs_pos_k = self._split_heads(abs_pos_k)    # [BH, L, Dh]
        abs_pos_v = self._split_heads(abs_pos_v)    # [BH, L, Dh]
        time_mat_k = self._split_heads_2d(time_mat_k)  # [BH, L, L, Dh]
        time_mat_v = self._split_heads_2d(time_mat_v)  # [BH, L, L, Dh]

        # [BH, L, L]
        attn_scores = torch.matmul(q, k.transpose(1, 2))
        attn_scores = attn_scores + torch.matmul(q, abs_pos_k.transpose(1, 2))
        attn_scores = attn_scores + torch.einsum("bid,bijd->bij", q, time_mat_k)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # key padding mask only
        key_pad = padding_mask.repeat_interleave(self.n_heads, dim=0)  # [BH, L]
        attn_scores = attn_scores.masked_fill(key_pad.unsqueeze(1), float("-inf"))

        # padded query rows -> softmax NaN 방지
        query_pad = padding_mask.repeat_interleave(self.n_heads, dim=0)  # [BH, L]
        attn_scores = attn_scores.masked_fill(query_pad.unsqueeze(-1), 0.0)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # padded query rows는 output 0으로
        attn_weights = attn_weights.masked_fill(query_pad.unsqueeze(-1), 0.0)

        out = torch.matmul(attn_weights, v)
        out = out + torch.matmul(attn_weights, abs_pos_v)
        out = out + torch.einsum("bij,bijd->bid", attn_weights, time_mat_v)

        out = self._merge_heads(out, B, L)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out
