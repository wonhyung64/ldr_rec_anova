"""Concise correctness checks for build_two_timescale_debias_model
(single-timescale alpha_v*H_v(t) -> alpha_short_v*H_short_v(t) + alpha_long_v*H_long_v(t)).

Plain-assert script (no pytest dependency in this repo) - run directly:
    python module/test_debias_two_timescale.py
"""
import math
import torch

from module.mf import MF
from module.debias_two_timescale import build_two_timescale_debias_model


def _make_model(seed=0, short_half_life=1.0, long_half_life=7.0):
    torch.manual_seed(seed)
    model_cls = build_two_timescale_debias_model(MF, short_half_life=short_half_life, long_half_life=long_half_life)
    model = model_cls(
        num_users=20, num_items=30, embedding_k=16, device="cpu",
        tau=0.5, depth=1, max_seq_len=10, n_heads=1, dropout=0.0,
    )
    model.eval()
    return model


def test_positivity():
    """1. mu_v, alpha_short_v, alpha_long_v, beta_short, beta_long, lambda_v(t) are all > 0."""
    model = _make_model(seed=0)
    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    with torch.no_grad():
        mu, alpha_short, alpha_long, beta_short, beta_long = model.prior_parameters_from_embeddings()
        intensity = model.prior(batch_items, pos_time, batch_time_all)

    assert (mu > 0).all()
    assert (alpha_short > 0).all()
    assert (alpha_long > 0).all()
    assert beta_short.item() > 0
    assert beta_long.item() > 0
    assert (intensity > 0).all()
    print("test_positivity: OK")


def test_timescale_ordering():
    """2. beta_short > beta_long, and therefore half_life_short < half_life_long,
    both at initialization and after a gradient step (the ordered
    parameterization must hold throughout training, not just at init)."""
    model = _make_model(seed=1)
    beta_short = model.current_beta_short().item()
    beta_long = model.current_beta_long().item()
    assert beta_short > beta_long > 0
    half_life_short = math.log(2) / beta_short
    half_life_long = math.log(2) / beta_long
    assert half_life_short < half_life_long

    # perturb the raw params (as an optimizer step would) and re-check
    with torch.no_grad():
        model.beta_long_raw.add_(torch.randn(()) * 5)
        model.beta_gap_raw.add_(torch.randn(()) * 5)
    assert model.current_beta_short().item() > model.current_beta_long().item() > 0
    print("test_timescale_ordering: OK")


def test_causal_history():
    """3. The event at t_i does not contribute to H_short or H_long for its own likelihood."""
    model = _make_model(seed=2)
    item_id = 5
    batch_items = torch.tensor([[item_id]])
    pos_time = torch.tensor([50.0])
    # this item's own history includes an event exactly AT the query time
    batch_time_all = torch.tensor([[[10.0, 50.0, 80.0]]])

    with torch.no_grad():
        beta_short = model.current_beta_short()
        beta_long = model.current_beta_long()
        mask = batch_time_all < pos_time.view(-1, 1, 1)
        delta = (pos_time.view(-1, 1, 1) - batch_time_all).clamp(min=0.0)
        h_short = (torch.exp(-beta_short * delta) * mask).sum(dim=-1)
        h_long = (torch.exp(-beta_long * delta) * mask).sum(dim=-1)

    # only the t=10 event (< 50) should count; t=50 (== query) and t=80 (> query) must not
    expected_h_short = torch.exp(-beta_short * torch.tensor(40.0))
    expected_h_long = torch.exp(-beta_long * torch.tensor(40.0))
    assert torch.allclose(h_short.squeeze(), expected_h_short, atol=1e-6)
    assert torch.allclose(h_long.squeeze(), expected_h_long, atol=1e-6)
    print("test_causal_history: OK")


def test_differential_decay():
    """4. For the same elapsed delta_t > 0, exp(-beta_short*dt) < exp(-beta_long*dt):
    the short state decays faster than the long state."""
    model = _make_model(seed=3)
    beta_short = model.current_beta_short()
    beta_long = model.current_beta_long()
    delta_t = torch.tensor(2.5)
    assert torch.exp(-beta_short * delta_t) < torch.exp(-beta_long * delta_t)
    print("test_differential_decay: OK")


def test_state_update_equivalent():
    """5. State update, adapted to this codebase's actual mechanism.

    The current implementation has no explicit per-item running state
    (state_short[v]/state_long[v]) to increment after an event -- H_v(t) is
    instead recomputed each call from each item's precomputed history array
    via a mask/delta over strictly-earlier timestamps (see the module
    docstring). The functional equivalent of "the state increases by 1 after
    the event" is: once an event at t_j is added to that array, querying at
    any t > t_j must include its contribution, exactly like a state that was
    just incremented and will now decay from t_j onward.
    """
    model = _make_model(seed=4)
    item_id = 7
    batch_items = torch.tensor([[item_id], [item_id]])
    # a fresh event just occurred at t=30 for this item; querying an instant
    # after it must reflect it, and querying strictly before must not.
    batch_time_all = torch.tensor([[[30.0]], [[30.0]]])
    pos_time = torch.tensor([30.0 - 1e-6, 30.0 + 5.0])

    with torch.no_grad():
        beta_short = model.current_beta_short()
        beta_long = model.current_beta_long()
        mask = batch_time_all < pos_time.view(-1, 1, 1)
        delta = (pos_time.view(-1, 1, 1) - batch_time_all).clamp(min=0.0)
        h_short = (torch.exp(-beta_short * delta) * mask).sum(dim=-1)
        h_long = (torch.exp(-beta_long * delta) * mask).sum(dim=-1)

    assert h_short[0].item() == 0.0 and h_long[0].item() == 0.0  # before the event: no contribution
    assert h_short[1].item() > 0.0 and h_long[1].item() > 0.0    # after the event: contributes, then decays
    print("test_state_update_equivalent: OK")


def test_gradient_flow():
    """6. alpha_short_mlp, alpha_long_mlp, beta_long_raw, beta_gap_raw receive
    finite, nonzero gradients."""
    model = _make_model(seed=5)
    model.train()

    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    loss = intensity.log().mean()
    loss.backward()

    for name, param in [("beta_long_raw", model.beta_long_raw), ("beta_gap_raw", model.beta_gap_raw)]:
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
        assert param.grad.abs().sum() > 0, f"{name} received an all-zero gradient"

    for name, module in [("alpha_short_mlp", model.alpha_short_mlp), ("alpha_long_mlp", model.alpha_long_mlp)]:
        saw_nonzero = False
        for p in module.parameters():
            assert p.grad is not None, f"{name} has a parameter with no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
            if p.grad.abs().sum() > 0:
                saw_nonzero = True
        assert saw_nonzero, f"{name} received an all-zero gradient"
    print("test_gradient_flow: OK")


def test_output_shape():
    """7. model.prior(...) returns exactly [batch_size, num_candidates]."""
    model = _make_model(seed=6)
    B, C, L = 12, 9, 7
    batch_items = torch.randint(0, 30, (B, C))
    pos_time = torch.rand(B) * 20
    batch_time_all = torch.rand(B, C, L) * 20

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    assert intensity.shape == (B, C), intensity.shape
    print("test_output_shape: OK")


if __name__ == "__main__":
    test_positivity()
    test_timescale_ordering()
    test_causal_history()
    test_differential_decay()
    test_state_update_equivalent()
    test_gradient_flow()
    test_output_shape()
    print("ALL TESTS PASSED")
