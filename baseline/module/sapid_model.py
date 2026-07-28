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


class SAPID(nn.Module):
    """
    SAPID: Sequentially Diversified and Accurate Recommendations in
    Chronological Order for a Series of Users.

    Sequential backbone: SASRec-style Transformer encoder.

    Ranking score:
        score(u, i) = cosine_sim(u_repr, i_emb) / tau
                      - gamma * norm_log_popularity(i)

    where norm_log_popularity(i) = log(freq(i)+1) / log(max_freq+1).
    The diversity penalty discounts items that have been consumed by many
    preceding users in chronological order, promoting set-level diversity
    across the series of users.
    """

    def __init__(self, num_users, num_items, embedding_k, device,
                 tau=1.0, depth=2, max_seq_len=50, n_heads=2, dropout=0.1,
                 gamma=0.1, item_counts=None, norm_first=True, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.max_seq_len = max_seq_len
        self.padding_item_id = num_items

        if item_counts is not None:
            log_counts = torch.log(torch.tensor(item_counts, dtype=torch.float32) + 1.0)
            max_log = log_counts.max().clamp(min=1.0)
            norm_log_pop = log_counts / max_log
        else:
            norm_log_pop = torch.zeros(num_items)
        self.register_buffer("log_popularity", norm_log_pop)

        self.item_embedding = nn.Embedding(num_items + 1, embedding_k, padding_idx=num_items)
        self.pos_enc = PositionalEncoding(max_seq_len, embedding_k, num_items)
        self.emb_dropout = nn.Dropout(p=dropout)

        n_layers = max(2, depth)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(embedding_k, eps=1e-8)

        for _ in range(n_layers):
            self.attention_layernorms.append(nn.LayerNorm(embedding_k, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(embedding_k, n_heads, dropout))
            self.forward_layernorms.append(nn.LayerNorm(embedding_k, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(embedding_k, dropout))

    def encode_user(self, hist_item_idx, user_idx=None):
        seqs = self.item_embedding(hist_item_idx)
        seqs = seqs * (self.embedding_k ** 0.5)
        seqs = seqs + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            x = self.attention_layernorms[i](seqs)
            mha_out, _ = self.attention_layers[i](x, x, x)
            seqs = seqs + mha_out
            seqs = seqs.transpose(0, 1)
            seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))

        return self.last_layernorm(seqs)[:, -1, :]

    def residual_score(self, item_idx, hist_item_idx, user_idx=None):
        u = self.encode_user(hist_item_idx, user_idx)
        B, K = u.shape
        v = self.item_embedding(item_idx).reshape(B, -1, K)

        u_norm = F.normalize(u, dim=-1, eps=1e-8)
        v_norm = F.normalize(v, dim=-1, eps=1e-8)
        relevance = (u_norm.unsqueeze(1) * v_norm).sum(-1) / self.tau

        pop = self.log_popularity[item_idx]
        score = relevance - self.gamma * pop
        return score

    def score_all_items(self, hist_item_idx, user_idx=None):
        u = self.encode_user(hist_item_idx, user_idx)
        v_all = self.item_embedding(torch.arange(self.num_items, device=hist_item_idx.device))

        u_norm = F.normalize(u, dim=-1, eps=1e-8)
        v_norm = F.normalize(v_all, dim=-1, eps=1e-8)
        relevance = torch.matmul(u_norm, v_norm.T) / self.tau

        pop = self.log_popularity
        score = relevance - self.gamma * pop.unsqueeze(0)
        return score
