import math
import torch
import torch.nn as nn


DEFAULT_FREQUENCIES = (1, 2, 4, 8)


def build_time_varying_debias_model(model_class, frequencies=DEFAULT_FREQUENCIES):
    """Hawkes-prior wrapper with a time-varying baseline mu_v(t).

    This is `module.debias.build_debias_model` with exactly one change:
    mu_v -> mu_v(t). The excitation term alpha_v * H_v(t) and beta are
    untouched; mu_v(t) is a residual correction on top of the original static
    baseline logit b_v = static_mu_mlp(z_v):

        mu_v(t) = softplus(b_v + temporal_mu_mlp([z_v, e_t, z_v*e_t])) + eps

    where e_t is a small Fourier-time encoding of the (train-period-only)
    normalized timestamp. temporal_mu_mlp's final layer is zero-initialized,
    so mu_v(t) ~= softplus(b_v) at initialization (same as the old model).
    """

    class TimeVaryingHawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eps = 1e-8
            self.frequencies = tuple(frequencies)
            time_feat_dim = 1 + 2 * len(self.frequencies)

            self.softplus = nn.Softplus()
            self.static_mu_mlp = self._build_mlp(self.embedding_k, self.embedding_k // 2, self.depth)
            self.excitation_net = self._build_mlp(self.embedding_k, self.embedding_k // 2, self.depth)
            self.log_beta = nn.Parameter(torch.zeros(()))

            self.time_encoder = nn.Sequential(
                nn.Linear(time_feat_dim, self.embedding_k // 2),
                nn.Softplus(),
                nn.Linear(self.embedding_k // 2, self.embedding_k),
            )
            self.temporal_mu_mlp = nn.Sequential(
                nn.Linear(self.embedding_k * 3, self.embedding_k // 2),
                nn.Softplus(),
                nn.Linear(self.embedding_k // 2, 1),
            )

            # Training-period time range used to normalize t; fixed once via
            # set_time_normalization() before training so validation/test
            # timestamps are transformed with the same (train-only) range.
            self.register_buffer("train_time_min", torch.zeros(()))
            self.register_buffer("train_time_max", torch.ones(()))

            self.reset_parameters()

        @staticmethod
        def _build_mlp(input_dim, hidden_dim, depth):
            layers = []
            in_dim = input_dim
            for _ in range(max(depth, 1)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Softplus())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1, bias=False))
            return nn.Sequential(*layers)

        def reset_parameters(self):
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            for module in list(self.static_mu_mlp) + list(self.excitation_net) + list(self.time_encoder) \
                    + list(self.temporal_mu_mlp[:-1]):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            # Zero-init only the final layer of temporal_mu_mlp so delta_v(t) ~= 0
            # at initialization: mu_v(t) ~= softplus(b_v), matching the old model.
            nn.init.zeros_(self.temporal_mu_mlp[-1].weight)
            nn.init.zeros_(self.temporal_mu_mlp[-1].bias)

        def set_time_normalization(self, train_time_min, train_time_max):
            """Fix t_normalized's reference range. Call once after construction
            with values computed from TRAINING events only — validation/test
            timestamps must never influence train_time_min/max (no leakage)."""
            self.train_time_min.fill_(float(train_time_min))
            self.train_time_max.fill_(float(train_time_max))

        def current_beta(self):
            return self.softplus(self.log_beta) + 1e-6

        def fourier_time_features(self, t):
            """phi(t) for a tensor t of any shape [...]; returns [..., 1+2L].

            t_normalized is NOT clamped to [0, 1]: out-of-range validation/test
            timestamps are allowed to extrapolate past the fitted sin/cos phase
            rather than being clipped to the training-period boundary.
            """
            span = torch.clamp(self.train_time_max - self.train_time_min, min=self.eps)
            t_norm = (t - self.train_time_min) / span
            feats = [t_norm]
            for f in self.frequencies:
                feats.append(torch.sin(2 * math.pi * f * t_norm))
                feats.append(torch.cos(2 * math.pi * f * t_norm))
            return torch.stack(feats, dim=-1)

        def _mu_time_varying(self, item_vec, time_emb_expanded, out_shape, diagnostics=None):
            """item_vec, time_emb_expanded: same shape [..., d]; out_shape: item_vec.shape[:-1]."""
            static_logit = self.static_mu_mlp(item_vec).reshape(out_shape)
            temporal_input = torch.cat([item_vec, time_emb_expanded, item_vec * time_emb_expanded], dim=-1)
            temporal_delta = self.temporal_mu_mlp(temporal_input).reshape(out_shape)
            mu_t = self.softplus(static_logit + temporal_delta) + self.eps

            if diagnostics is not None:
                diagnostics["static_baseline_logit_mean"] = static_logit.detach().mean().item()
                diagnostics["temporal_delta_mean"] = temporal_delta.detach().mean().item()
                diagnostics["temporal_delta_std"] = temporal_delta.detach().std().item()
                diagnostics["mu_t_mean"] = mu_t.detach().mean().item()

            return mu_t

        def prior_parameters_from_embeddings(self, query_time, diagnostics=None):
            """Full item-table (mu_v(t), alpha_v, beta) at a single scalar query_time."""
            z = self.item_embedding.weight
            t = query_time if torch.is_tensor(query_time) else torch.tensor(float(query_time), device=z.device)
            t = t.reshape(())

            time_feat = self.fourier_time_features(t)                       # [F]
            time_emb = self.time_encoder(time_feat)                         # [d]
            time_emb_expanded = time_emb.unsqueeze(0).expand(z.shape[0], -1)  # [V, d] view

            mu_t = self._mu_time_varying(z, time_emb_expanded, (z.shape[0],), diagnostics=diagnostics)
            alpha = self.softplus(self.excitation_net(z)).reshape(z.shape[0]) + 1e-8
            beta = self.current_beta()
            return mu_t, alpha, beta

        def prior(self, batch_items, pos_time, batch_time_all, diagnostics=None):
            item_vec = self.item_embedding(batch_items)                     # [B, C, d]
            beta = self.current_beta()
            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query
            delta = (query - batch_time_all).clamp(min=0.0)
            h = (torch.exp(-beta * delta) * mask).sum(dim=-1)                # [B, C]
            mB, C = h.shape

            # Same event time -> same e_t for every candidate at that event;
            # expand (a view, not a copy) rather than materializing [B,C,d].
            time_feat = self.fourier_time_features(pos_time.reshape(-1))     # [B, F]
            time_emb = self.time_encoder(time_feat)                         # [B, d]
            time_emb_expanded = time_emb.unsqueeze(1).expand(-1, C, -1)      # [B, C, d] view

            mu_t = self._mu_time_varying(item_vec, time_emb_expanded, (mB, C), diagnostics=diagnostics)
            alpha = self.softplus(self.excitation_net(item_vec)).reshape(mB, C) + 1e-8

            intensity = mu_t + alpha * h + self.eps

            if diagnostics is not None:
                diagnostics["hawkes_excitation_mean"] = (alpha * h).detach().mean().item()
                diagnostics["intensity_mean"] = intensity.detach().mean().item()
                diagnostics["intensity_max"] = intensity.detach().max().item()

            return intensity

    return TimeVaryingHawkesDebias
