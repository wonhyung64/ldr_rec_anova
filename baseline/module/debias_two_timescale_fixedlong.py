import math
import torch
import torch.nn as nn


EPS = 1e-8


def _inv_softplus(y):
    """x such that softplus(x) == y, for y > 0. softplus(x) = log(1+exp(x))."""
    return math.log(math.expm1(y))


def build_two_timescale_fixed_long_debias_model(model_class, short_half_life=1.0, long_half_life=7.0,
                                                 alpha_long_init_bias=-2.0):
    """Two-timescale Hawkes prior, second iteration.

    Same excitation split as module.debias_two_timescale
    (alpha_short_v*H_short_v(t) + alpha_long_v*H_long_v(t)), but with two
    fixes motivated by what the first run's diagnostics showed:

    1. In the first version beta_long was learnable and, over training, drifted
       up from its 7-day-half-life init to within a few multiples of
       beta_short (half_life_long collapsed from ~7 to ~0.3-1 by epoch 500,
       and long_fraction fell to ~0.0005 within the first ~30 steps and never
       recovered) -- the model was effectively using one (increasingly fast)
       timescale, not two. Here beta_long is a FIXED, non-learnable buffer:
       only beta_short (via beta_gap_raw) is trained, so the long component is
       structurally forced to keep operating at the configured long_half_life
       and can no longer collapse toward beta_short.
    2. alpha_long_mlp's initial bias was -6 (alpha_long_v ~= 0.0025 at init),
       which likely starved it of gradient signal early (short already
       explains the loss well, so the marginal gradient pushing alpha_long up
       was tiny -- a rich-get-richer dead-branch pattern). Loosened to -2
       (alpha_long_v ~= 0.12 at init) so it has a more realistic chance to
       receive gradient and actually be used.

    Everything else (mu_v, alpha_short_v's structure/init, the masked-sum
    Hawkes computation reused for both timescales, ordering beta_short >
    beta_long) is unchanged from module.debias_two_timescale.
    """

    class HawkesDebias(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eps = EPS
            self.softplus = nn.Softplus()

            self.mu_mlp = self._build_mlp(self.embedding_k, self.embedding_k // 2, self.depth)
            self.alpha_short_mlp = self._build_mlp(self.embedding_k, self.embedding_k // 2, self.depth)
            self.alpha_long_mlp = self._build_alpha_long_mlp(
                self.embedding_k, self.embedding_k // 2, self.depth, init_bias=alpha_long_init_bias,
            )

            # beta_long is now a FIXED buffer (not a Parameter): the optimizer
            # never touches it, so the long component cannot drift toward the
            # short timescale. Only beta_short (via beta_gap_raw) is learned:
            # beta_short = beta_long_fixed + softplus(beta_gap_raw), which
            # still guarantees beta_short > beta_long for any beta_gap_raw.
            self.beta_gap_raw = nn.Parameter(torch.zeros(()))
            self.register_buffer("beta_long_fixed", torch.zeros(()))

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
        def _build_alpha_long_mlp(input_dim, hidden_dim, depth, init_bias):
            """Same structure as _build_mlp, but the final layer keeps a bias
            so it can be initialized to a (mildly) negative constant: with
            weight=0, the pre-softplus output is exactly init_bias regardless
            of z_v, so alpha_long_v starts at softplus(init_bias) - small next
            to alpha_short_v, but not so small it starves out of gradient.
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
            for module in list(self.mu_mlp) + list(self.alpha_short_mlp):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # alpha_long_mlp's final layer was already explicitly initialized
            # in _build_alpha_long_mlp; only its earlier hidden layer(s) get
            # the standard Xavier init here.
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
                self.beta_long_fixed.fill_(target_beta_long)
                self.beta_gap_raw.fill_(_inv_softplus(target_gap))

        def current_beta_long(self):
            return self.beta_long_fixed

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
                diagnostics["mean_mu"] = mu.detach().mean().item()
                diagnostics["mean_alpha_short"] = alpha_short.detach().mean().item()
                diagnostics["mean_alpha_long"] = alpha_long.detach().mean().item()
                diagnostics["beta_short"] = beta_short.item()
                diagnostics["beta_long"] = beta_long.item()
                diagnostics["half_life_short"] = math.log(2) / beta_short.item()
                diagnostics["half_life_long"] = math.log(2) / beta_long.item()

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
                # Split out alpha_* and h_* (not just their products) so it's
                # possible to tell apart "alpha_long learned to matter" from
                # "h_long is just naturally small" -- the first version only
                # logged the products, which conflated the two.
                diagnostics["mean_mu"] = mu.detach().mean().item()
                diagnostics["mean_alpha_short"] = alpha_short.detach().mean().item()
                diagnostics["mean_alpha_long"] = alpha_long.detach().mean().item()
                diagnostics["mean_h_short"] = h_short.detach().mean().item()
                diagnostics["mean_h_long"] = h_long.detach().mean().item()
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

    return HawkesDebias
