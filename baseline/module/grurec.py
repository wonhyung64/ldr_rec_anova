import torch
from .base import ResidualBase


class GRURec(ResidualBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_gru = torch.nn.GRU(
            input_size=self.embedding_k,
            hidden_size=self.embedding_k,
            num_layers=max(2,self.depth),
            batch_first=True,
        )

    def encode_user(self, hist_item_idx, user_idx=None):
        hist_emb = self.item_embedding(hist_item_idx)
        _, h_n = self.user_gru(hist_emb)
        u = h_n[-1]
        return u
