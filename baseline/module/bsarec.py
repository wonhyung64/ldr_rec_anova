import math
import torch
import torch.nn as nn
from module.base import ResidualBase, PositionalEncoding


class BSARec(ResidualBase):
    def __init__(self, *args, c=3, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k, self.padding_item_id)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout)
        self.n_layers = max(2, self.depth)
        self.layer = nn.ModuleList()
        self.c = c
        self.alpha = alpha
        for i in range(self.n_layers):
            self.layer_ramp = BSARecBlock(self.dropout, self.embedding_k, self.n_heads, self.c, self.alpha)
            self.layer.append(self.layer_ramp)

    def encode_user(self, hist_item_idx, user_idx=None):
        seqs = self.item_embedding(hist_item_idx)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        seqs = seqs + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)
        for layer_module in self.layer:
            hidden_states = layer_module(seqs)
        return hidden_states[:, -1, :]


class BSARecBlock(nn.Module):
    def __init__(self, dropout_rate, embedding_k, n_heads, c, alpha):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(dropout_rate, embedding_k, n_heads, c, alpha)
        self.feed_forward = FeedForward(embedding_k, dropout_rate)

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class BSARecLayer(nn.Module):
    def __init__(self, dropout_rate, embedding_k, n_heads, c, alpha):
        super(BSARecLayer, self).__init__()
        self.filter_layer = FrequencyLayer(dropout_rate, embedding_k, c)
        self.attention_layer = nn.MultiheadAttention(embedding_k, n_heads, dropout_rate)
        self.alpha = alpha

    def forward(self, input_tensor):
        dsp = self.filter_layer(input_tensor)
        gsp, _ = self.attention_layer(input_tensor, input_tensor, input_tensor)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    

class FrequencyLayer(nn.Module):
    def __init__(self, dropout_rate, embedding_k, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(embedding_k, eps=1e-8)
        self.c = c
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, embedding_k))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        _, seq_len, __ = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(dropout_rate)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.gelu(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
