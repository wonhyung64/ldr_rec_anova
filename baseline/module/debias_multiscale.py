import math
import torch
import torch.nn as nn


EPS = 1e-8


def build_multiscale_debias_model(model_class, half_lives):
    """Hawkes-prior wrapper with a flexible, signed K-component exponential
    excitation ("basis expansion" Hawkes kernel).

    This generalizes both the single- and two-timescale variants
    (module.debias / module.debias_two_timescale*):

        lambda_v(t) = softplus( b_v + sum_k w_k(v) * H_k(t) ) + eps

        H_k(t) = sum_{t_j < t, item_j=v} exp(-beta_k * (t - t_j)),  k = 1..K

    Motivation (from the no-training data diagnostic across micro_video,
    ml-1m, kuairand): the (1-day, 7-day) two-timescale split didn't match any
    dataset's actual timescale, AND micro_video's best-fitting second
    component had a NEGATIVE partial coefficient -- something alpha_v >= 0
    (softplus) can never express. This module fixes both problems at once:

    - beta_1..beta_K are FIXED, non-learnable buffers spanning a wide,
      dataset-specific, log-spaced half-life grid (passed in by the caller),
      rather than 1-2 learnable betas that (per the earlier two-timescale
      runs) just drift/collapse toward each other during training.
    - the per-item mixing weights w_k(v) are UNCONSTRAINED (signed) --
      positivity is enforced once, on the total (b_v + sum_k w_k*H_k), not on
      each w_k individually. This lets the data decide the sign and relative
      importance of every timescale, including "irrelevant" (w_k ~= 0) and
      "actively suppressive" (w_k < 0) components.

    Setting K=1 with a non-negative w_1 recovers the vanilla single-timescale
    model; K=2 with both w_k >= 0 recovers the (additive) two-timescale
    model. Here neither constraint is imposed, so the fit is a strict
    superset of both.
    """
    half_lives = list(half_lives)
    K = len(half_lives)

    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eps = EPS
            self.K = K
            self.softplus = nn.Softplus()

            self.register_buffer("betas", torch.tensor([math.log(2) / hl for hl in half_lives], dtype=torch.float32))

            hidden_dim = self.embedding_k // 2
            self.mu_trunk, mu_out_dim = self._build_trunk(self.embedding_k, hidden_dim, self.depth)
            self.mu_head = nn.Linear(mu_out_dim, 1, bias=False)

            self.weight_trunk, w_out_dim = self._build_trunk(self.embedding_k, hidden_dim, self.depth)
            self.weight_head = nn.Linear(w_out_dim, self.K, bias=True)

            self.reset_parameters()

        @staticmethod
        def _build_trunk(input_dim, hidden_dim, depth):
            layers = []
            in_dim = input_dim
            for _ in range(max(depth, 1)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Softplus())
                in_dim = hidden_dim
            return nn.Sequential(*layers), in_dim

        def reset_parameters(self):
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            for module in list(self.mu_trunk) + [self.mu_head] + list(self.weight_trunk):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # weight_head zero-initialized (both weight and bias) so every
            # w_k(v) starts at exactly 0: at init, lambda_v(t) ~= softplus(b_v),
            # i.e. identical to the original static-mu vanilla model, with no
            # excitation contribution from any timescale until training moves
            # weight_head away from zero.
            nn.init.zeros_(self.weight_head.weight)
            nn.init.zeros_(self.weight_head.bias)

        def _mu_logit_and_weights(self, item_vec):
            mu_logit = self.mu_head(self.mu_trunk(item_vec))    # [..., 1]
            w = self.weight_head(self.weight_trunk(item_vec))   # [..., K]
            return mu_logit, w

        def prior_parameters_from_embeddings(self, diagnostics=None):
            z = self.item_embedding.weight
            mu_logit, w = self._mu_logit_and_weights(z)
            mu_logit = mu_logit.squeeze(-1)   # [V]

            if diagnostics is not None:
                diagnostics["mean_mu_logit"] = mu_logit.detach().mean().item()
                for hl, wk in zip(half_lives, w.detach().unbind(dim=-1)):
                    diagnostics[f"mean_w_hl{hl}"] = wk.mean().item()

            return mu_logit, w, self.betas

        def prior(self, batch_items, pos_time, batch_time_all, diagnostics=None):
            item_vec = self.item_embedding(batch_items)              # [B, C, d]
            mB, C = batch_items.shape

            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query                              # [B, C, L]: strictly-before, no self-leakage
            delta = (query - batch_time_all).clamp(min=0.0)            # [B, C, L]

            # H_k for all K components at once: mask/delta don't depend on
            # beta, so they're computed once and reused across every timescale.
            decay = torch.exp(-delta.unsqueeze(-1) * self.betas.view(1, 1, 1, -1))  # [B, C, L, K]
            H = (decay * mask.unsqueeze(-1)).sum(dim=2)                             # [B, C, K]

            mu_logit, w = self._mu_logit_and_weights(item_vec)
            mu_logit = mu_logit.reshape(mB, C)
            w = w.reshape(mB, C, self.K)

            excitation = (w * H).sum(dim=-1)                          # [B, C], SIGNED
            intensity = self.softplus(mu_logit + excitation) + self.eps

            if diagnostics is not None:
                diagnostics["mean_mu_logit"] = mu_logit.detach().mean().item()
                diagnostics["mean_excitation"] = excitation.detach().mean().item()
                diagnostics["mean_intensity"] = intensity.detach().mean().item()
                diagnostics["max_intensity"] = intensity.detach().max().item()
                for i, hl in enumerate(half_lives):
                    diagnostics[f"mean_w_hl{hl}"] = w[..., i].detach().mean().item()
                    diagnostics[f"mean_H_hl{hl}"] = H[..., i].detach().mean().item()
                    diagnostics[f"mean_contrib_hl{hl}"] = (w[..., i] * H[..., i]).detach().mean().item()

            return intensity

    return HawkesDebias
