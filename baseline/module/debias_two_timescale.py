import math
import torch
import torch.nn as nn


EPS = 1e-8


def _inv_softplus(y):
    """x such that softplus(x) == y, for y > 0. softplus(x) = log(1+exp(x))."""
    return math.log(math.expm1(y))


def build_two_timescale_debias_model(model_class, short_half_life=1.0, long_half_life=7.0):
    """Hawkes-prior wrapper with a two-timescale excitation.

    This is `module.debias.build_debias_model` with exactly one change: the
    single alpha_v * H_v(t) excitation term is replaced with

        alpha_short_v * H_short_v(t) + alpha_long_v * H_long_v(t)

    mu_v and its network are untouched. Timestamps in this codebase are
    already expressed in days (see debiased_seq_rec_hawkes_anova.py, which
    divides raw seconds by 86400), so short_half_life/long_half_life default
    to 1 and 7 days directly, with no further unit conversion.

    Both H_short and H_long are computed by reusing the SAME masked-sum
    mechanism the original model already uses for H_v(t) (mask/delta over
    each item's precomputed, length-bounded history array
    `dataset.item_time_array`) rather than a literal per-item recursive
    running-state update -- mask/delta don't depend on beta, so they are
    computed once and reused for both timescales.
    """

    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eps = EPS
            self.softplus = nn.Softplus()

            self.mu_mlp = self._build_mlp(self.embedding_k, self.embedding_k // 2, self.depth)
            self.alpha_short_mlp = self._build_mlp(self.embedding_k, self.embedding_k // 2, self.depth)
            self.alpha_long_mlp = self._build_alpha_long_mlp(self.embedding_k, self.embedding_k // 2, self.depth)

            # beta_long = softplus(beta_long_raw) + eps
            # beta_short = beta_long + softplus(beta_gap_raw)
            # => beta_short > beta_long > 0 is guaranteed by construction,
            # for any value the two raw parameters take during training.
            self.beta_long_raw = nn.Parameter(torch.zeros(()))
            self.beta_gap_raw = nn.Parameter(torch.zeros(()))

            self.reset_parameters()
            self._init_betas(short_half_life, long_half_life)

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

        @staticmethod
        def _build_alpha_long_mlp(input_dim, hidden_dim, depth, init_bias=-6.0):
            """Same structure as _build_mlp, but the final layer keeps a bias
            (unlike the other nets' bias=False final layer) so it can be
            initialized to a strongly negative constant: with weight=0, the
            pre-softplus output is exactly init_bias regardless of z_v, so
            alpha_long_v starts at softplus(init_bias) (~0.0025 for -6),
            small next to alpha_short_v (~O(0.1-1) under standard Xavier init).
            """
            layers = []
            in_dim = input_dim
            for _ in range(max(depth, 1)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Softplus())
                in_dim = hidden_dim
            final = nn.Linear(in_dim, 1, bias=True)
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, init_bias)
            layers.append(final)
            return nn.Sequential(*layers)

        def reset_parameters(self):
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            # alpha_short_mlp uses the exact same build + init as the original
            # single-timescale excitation network, so it starts out behaving
            # like the original alpha_v (there is no separately-trained
            # "existing" alpha network instance in this fresh model to copy
            # weights from; matching its initialization scheme is the natural
            # reading of "initialize alpha_short_mlp using the existing alpha
            # network").
            for module in list(self.mu_mlp) + list(self.alpha_short_mlp):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # alpha_long_mlp's final layer was already explicitly initialized
            # (zero weight, negative bias) in _build_alpha_long_mlp; only its
            # earlier hidden layer(s) get the standard Xavier init here.
            for module in list(self.alpha_long_mlp[:-1]):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def _init_betas(self, short_half_life, long_half_life):
            target_beta_long = math.log(2) / long_half_life
            target_beta_short = math.log(2) / short_half_life
            target_gap = target_beta_short - target_beta_long
            if target_gap <= 0:
                raise ValueError(
                    f"short_half_life ({short_half_life}) must be < long_half_life ({long_half_life}) "
                    "so that beta_short_init > beta_long_init."
                )
            with torch.no_grad():
                self.beta_long_raw.fill_(_inv_softplus(target_beta_long - self.eps))
                self.beta_gap_raw.fill_(_inv_softplus(target_gap))

        def current_beta_long(self):
            return self.softplus(self.beta_long_raw) + self.eps

        def current_beta_short(self):
            return self.current_beta_long() + self.softplus(self.beta_gap_raw)

        def current_beta(self):
            """Kept for interface parity with the single-timescale model."""
            return self.current_beta_short()

        def half_lives(self):
            beta_short = self.current_beta_short()
            beta_long = self.current_beta_long()
            return math.log(2) / beta_short.item(), math.log(2) / beta_long.item()

        def prior_parameters_from_embeddings(self, diagnostics=None):
            z = self.item_embedding.weight
            mu = self.softplus(self.mu_mlp(z)).squeeze(-1) + self.eps
            alpha_short = self.softplus(self.alpha_short_mlp(z)).squeeze(-1) + self.eps
            alpha_long = self.softplus(self.alpha_long_mlp(z)).squeeze(-1) + self.eps
            beta_short = self.current_beta_short()
            beta_long = self.current_beta_long()

            if diagnostics is not None:
                self._fill_diagnostics(diagnostics, mu, alpha_short, alpha_long, beta_short, beta_long)

            return mu, alpha_short, alpha_long, beta_short, beta_long

        def prior(self, batch_items, pos_time, batch_time_all, diagnostics=None):
            item_vec = self.item_embedding(batch_items)
            beta_short = self.current_beta_short()
            beta_long = self.current_beta_long()

            query = pos_time.view(-1, 1, 1)
            mask = batch_time_all < query                      # strictly-before mask: no self-leakage
            delta = (query - batch_time_all).clamp(min=0.0)     # mask/delta reused for both timescales

            h_short = (torch.exp(-beta_short * delta) * mask).sum(dim=-1)
            h_long = (torch.exp(-beta_long * delta) * mask).sum(dim=-1)
            mB, C = h_short.shape

            mu = self.softplus(self.mu_mlp(item_vec)).reshape(mB, C) + self.eps
            alpha_short = self.softplus(self.alpha_short_mlp(item_vec)).reshape(mB, C) + self.eps
            alpha_long = self.softplus(self.alpha_long_mlp(item_vec)).reshape(mB, C) + self.eps

            short_excitation = alpha_short * h_short
            long_excitation = alpha_long * h_long
            intensity = mu + short_excitation + long_excitation + self.eps

            if diagnostics is not None:
                diagnostics["mean_mu"] = mu.detach().mean().item()
                diagnostics["mean_short_excitation"] = short_excitation.detach().mean().item()
                diagnostics["mean_long_excitation"] = long_excitation.detach().mean().item()
                diagnostics["mean_total_intensity"] = intensity.detach().mean().item()
                diagnostics["beta_short"] = beta_short.item()
                diagnostics["beta_long"] = beta_long.item()
                diagnostics["half_life_short"] = math.log(2) / beta_short.item()
                diagnostics["half_life_long"] = math.log(2) / beta_long.item()
                denom = short_excitation.detach().sum() + long_excitation.detach().sum() + self.eps
                diagnostics["short_fraction"] = (short_excitation.detach().sum() / denom).item()
                diagnostics["long_fraction"] = (long_excitation.detach().sum() / denom).item()

            return intensity

        @staticmethod
        def _fill_diagnostics(diagnostics, mu, alpha_short, alpha_long, beta_short, beta_long):
            diagnostics["mean_mu"] = mu.detach().mean().item()
            diagnostics["beta_short"] = beta_short.item()
            diagnostics["beta_long"] = beta_long.item()
            diagnostics["half_life_short"] = math.log(2) / beta_short.item()
            diagnostics["half_life_long"] = math.log(2) / beta_long.item()

    return HawkesDebias
