import torch
from .base import ResidualBase


class MF(ResidualBase):
    """Matrix Factorization baseline: static user/item embeddings."""
    def encode_user(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("MFResidual requires user_idx.")
        return self.user_embedding(user_idx)
