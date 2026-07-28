import torch
import torch.nn as nn
from .base import ResidualBase, PositionalEncoding


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.embedding_k, self.padding_item_id)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.embedding_k, eps=1e-8)

        for _ in range(max(2, self.depth)):
            new_attn_layernorm = nn.LayerNorm(self.embedding_k, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  nn.MultiheadAttention(self.embedding_k, self.n_heads, self.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.embedding_k, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.embedding_k, self.dropout)
            self.forward_layers.append(new_fwd_layer)

    def encode_user(self, hist_item_idx, user_idx=None):
        seqs = self.item_embedding(hist_item_idx)
        seqs *= self.item_embedding.embedding_dim ** 0.5
        seqs = seqs + self.pos_enc(hist_item_idx)
        seqs = self.emb_dropout(seqs)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        u = self.last_layernorm(seqs)[:,-1,:]

        return u
