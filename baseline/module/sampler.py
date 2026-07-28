import math
import torch
import numpy as np
from dataclasses import dataclass


def sample_epoch_negatives(
    snapshot,
    train_events,
    num_items,
    num_negatives,
):
    """
    Pre-sample negatives once per epoch from the frozen prior snapshot.
    The sampler uses the exact base/excitation mixture decomposition.
    """
    negatives = np.empty((len(train_events), num_negatives), dtype=np.int64)
    tree = FenwickTree(num_items)
    c_weights = np.zeros(num_items, dtype=np.float64)
    total_c = 0.0
    g = 1.0
    prev_time = float(train_events[0][2]) if train_events else 0.0

    def sample_one_excluding(pos_item: int, rho: float) -> int:
        for _ in range(20):
            if np.random.rand() < rho or total_c <= 0.0:
                candidate = snapshot.base_sampler.sample()
            else:
                mass = np.random.rand() * max(tree.total(), 1e-12)
                candidate = tree.sample(mass)
            if candidate != pos_item:
                return int(candidate)
        # Fallback: uniform rejection if the prior mass is too concentrated.
        candidate = np.random.randint(0, num_items - 1)
        candidate += candidate >= pos_item
        return int(candidate)

    for idx, (_, item, t) in enumerate(train_events):
        t = float(t)
        delta = max(t - prev_time, 0.0)
        if delta > 0.0:
            g *= math.exp(-snapshot.beta * delta)
        excitation_mass = g * total_c
        rho = snapshot.base_mass / (snapshot.base_mass + excitation_mass + 1e-12)

        for j in range(num_negatives):
            negatives[idx, j] = sample_one_excluding(item, rho)

        delta_c = snapshot.alpha[item] / max(g, 1e-12)
        c_weights[item] += delta_c
        total_c += delta_c
        tree.add(item, delta_c)
        prev_time = t

    return negatives


class FenwickTree:
    """Fenwick tree for positive weights, used by the excitation sampler."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.tree = np.zeros(self.size + 1, dtype=np.float64)

    def add(self, index: int, delta: float) -> None:
        i = int(index) + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def total(self) -> float:
        return float(self.prefix_sum(self.size - 1))

    def prefix_sum(self, index: int) -> float:
        i = int(index) + 1
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def sample(self, mass: float) -> int:
        if mass < 0.0 or mass >= self.total() + 1e-9:
            raise ValueError("mass must be in [0, total)")
        idx = 0
        bit_mask = 1 << (self.size.bit_length() - 1)
        running = 0.0
        while bit_mask != 0:
            nxt = idx + bit_mask
            if nxt <= self.size and running + self.tree[nxt] <= mass:
                idx = nxt
                running += self.tree[nxt]
            bit_mask >>= 1
        return min(idx, self.size - 1)


@torch.no_grad()
def make_prior_snapshot(model):
    model.eval()
    mu_t, alpha_t, beta_t = model.prior_parameters_from_embeddings()
    mu = mu_t.detach().cpu().numpy().astype(np.float64)
    alpha = alpha_t.detach().cpu().numpy().astype(np.float64)
    beta = float(beta_t.detach().cpu().item())
    base_mass = float(mu.sum())
    return PriorSnapshot(
        mu=mu,
        alpha=alpha,
        beta=beta,
        base_mass=base_mass,
        base_sampler=BaseSampler(mu),
    )


class BaseSampler:
    """CDF sampler for the frozen base distribution p_base(v) ∝ mu_v."""

    def __init__(self, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        self.cdf = np.cumsum(weights / total)
        self.cdf[-1] = 1.0

    def sample(self) -> int:
        u = np.random.rand()
        return int(np.searchsorted(self.cdf, u, side="right"))


@dataclass
class PriorSnapshot:
    mu: np.ndarray
    alpha: np.ndarray
    beta: float
    base_mass: float
    base_sampler: BaseSampler
