import torch
from .base import ResidualBase


class NCF(ResidualBase):
    """Neural Collaborative Filtering baseline with static user/item embeddings."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(8, self.embedding_k)
        layers = [torch.nn.Linear(self.embedding_k * 2, hidden), torch.nn.ReLU()]
        for _ in range(max(0, self.depth - 1)):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hidden, 1, bias=False)]
        self.mlp = torch.nn.Sequential(*layers)

    def encode_user(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        return self.user_embedding(user_idx)

    def residual_score(self, item_idx, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        u = self.user_embedding(user_idx)
        mini_batch, recdim = u.shape
        v = self.get_item_repr(item_idx).reshape(mini_batch, -1, recdim)
        u_expanded = u.unsqueeze(1).expand(-1, v.size(1), -1)
        x = torch.cat([v, u_expanded], dim=-1)
        h = self.mlp(x).squeeze(-1)
        return h

    def score_all_items(self, hist_item_idx, user_idx=None):
        if user_idx is None:
            raise ValueError("NCFResidual requires user_idx.")
        u = self.user_embedding(user_idx)
        v_all = self.get_item_repr(torch.arange(self.num_items, device=u.device)).unsqueeze(0).expand(u.size(0), -1, -1)
        u_expand = u.unsqueeze(1).expand(-1, self.num_items, -1)
        h = self.mlp(torch.cat([u_expand, v_all], dim=-1)).squeeze(-1)
        return h
