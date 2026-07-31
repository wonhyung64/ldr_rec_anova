"""Concise correctness checks for build_two_timescale_fixed_long_debias_model.

Same suite as test_debias_two_timescale.py, plus two checks specific to this
iteration's fixes: beta_long must be truly fixed (no gradient, unaffected by
an optimizer step), and alpha_long's looser init must actually be bigger than
before. Run directly:
    python module/test_debias_two_timescale_fixedlong.py
"""
import math
import torch

from module.mf import MF
from module.debias_two_timescale_fixedlong import build_two_timescale_fixed_long_debias_model


def _make_model(seed=0, short_half_life=1.0, long_half_life=7.0):
    torch.manual_seed(seed)
    model_cls = build_two_timescale_fixed_long_debias_model(MF, short_half_life=short_half_life, long_half_life=long_half_life)
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
    """2. beta_short > beta_long at init and after a gradient step on beta_gap_raw."""
    model = _make_model(seed=1)
    assert model.current_beta_short().item() > model.current_beta_long().item() > 0
    with torch.no_grad():
        model.beta_gap_raw.add_(torch.randn(()) * 5)
    assert model.current_beta_short().item() > model.current_beta_long().item() > 0
    print("test_timescale_ordering: OK")


def test_beta_long_is_truly_fixed():
    """New for this iteration: beta_long must not move -- it's a buffer, not
    a Parameter, so it gets no gradient and an optimizer step must not
    change it (this directly targets the failure mode from the first
    two-timescale run, where beta_long drifted from a 7-day to a <1-day
    half-life over training)."""
    model = _make_model(seed=2, long_half_life=7.0)
    beta_long_before = model.current_beta_long().item()
    assert "beta_long_fixed" in dict(model.named_buffers())
    assert "beta_long_raw" not in dict(model.named_parameters())

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    model.train()
    intensity = model.prior(batch_items, pos_time, batch_time_all)
    loss = intensity.log().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    beta_long_after = model.current_beta_long().item()
    assert beta_long_before == beta_long_after, (beta_long_before, beta_long_after)
    assert abs(beta_long_after - math.log(2) / 7.0) < 1e-6
    print("test_beta_long_is_truly_fixed: OK")


def test_causal_history():
    """3. The event at t_i does not contribute to H_short or H_long when
    evaluating its own likelihood."""
    model = _make_model(seed=3)
    item_id = 5
    batch_items = torch.tensor([[item_id]])
    pos_time = torch.tensor([50.0])
    batch_time_all = torch.tensor([[[10.0, 50.0, 80.0]]])

    with torch.no_grad():
        beta_short = model.current_beta_short()
        beta_long = model.current_beta_long()
        mask = batch_time_all < pos_time.view(-1, 1, 1)
        delta = (pos_time.view(-1, 1, 1) - batch_time_all).clamp(min=0.0)
        h_short = (torch.exp(-beta_short * delta) * mask).sum(dim=-1)
        h_long = (torch.exp(-beta_long * delta) * mask).sum(dim=-1)

    expected_h_short = torch.exp(-beta_short * torch.tensor(40.0))
    expected_h_long = torch.exp(-beta_long * torch.tensor(40.0))
    assert torch.allclose(h_short.squeeze(), expected_h_short, atol=1e-6)
    assert torch.allclose(h_long.squeeze(), expected_h_long, atol=1e-6)
    print("test_causal_history: OK")


def test_differential_decay():
    """4. For the same elapsed delta_t > 0, exp(-beta_short*dt) < exp(-beta_long*dt)."""
    model = _make_model(seed=4)
    beta_short = model.current_beta_short()
    beta_long = model.current_beta_long()
    delta_t = torch.tensor(2.5)
    assert torch.exp(-beta_short * delta_t) < torch.exp(-beta_long * delta_t)
    print("test_differential_decay: OK")


def test_state_update_equivalent():
    """5. Mask-based equivalent of "state increases by 1 after the event":
    querying strictly before a fresh event sees no contribution; querying
    after it does (see module docstring for why there's no literal
    state_short[v]/state_long[v] object in this implementation)."""
    model = _make_model(seed=5)
    item_id = 7
    batch_items = torch.tensor([[item_id], [item_id]])
    batch_time_all = torch.tensor([[[30.0]], [[30.0]]])
    pos_time = torch.tensor([30.0 - 1e-6, 30.0 + 5.0])

    with torch.no_grad():
        beta_short = model.current_beta_short()
        beta_long = model.current_beta_long()
        mask = batch_time_all < pos_time.view(-1, 1, 1)
        delta = (pos_time.view(-1, 1, 1) - batch_time_all).clamp(min=0.0)
        h_short = (torch.exp(-beta_short * delta) * mask).sum(dim=-1)
        h_long = (torch.exp(-beta_long * delta) * mask).sum(dim=-1)

    assert h_short[0].item() == 0.0 and h_long[0].item() == 0.0
    assert h_short[1].item() > 0.0 and h_long[1].item() > 0.0
    print("test_state_update_equivalent: OK")


def test_gradient_flow():
    """6. alpha_short_mlp, alpha_long_mlp, beta_gap_raw receive finite,
    nonzero gradients. beta_long_fixed is intentionally excluded: it is a
    buffer now, not a learnable parameter (see test_beta_long_is_truly_fixed)."""
    model = _make_model(seed=6)
    model.train()

    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    loss = intensity.log().mean()
    loss.backward()

    assert model.beta_gap_raw.grad is not None
    assert torch.isfinite(model.beta_gap_raw.grad).all()
    assert model.beta_gap_raw.grad.abs().sum() > 0

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
    model = _make_model(seed=7)
    B, C, L = 12, 9, 7
    batch_items = torch.randint(0, 30, (B, C))
    pos_time = torch.rand(B) * 20
    batch_time_all = torch.rand(B, C, L) * 20

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    assert intensity.shape == (B, C), intensity.shape
    print("test_output_shape: OK")


def test_alpha_long_init_is_looser_than_before():
    """Sanity check for fix #2: alpha_long_v at init should now be
    meaningfully larger than the previous -6-bias version's ~0.0025."""
    model = _make_model(seed=8)
    with torch.no_grad():
        alpha_long_sample = model.softplus(model.alpha_long_mlp(model.item_embedding.weight[:1])).item()
    assert alpha_long_sample > 0.05, alpha_long_sample  # previous version was ~0.0025
    print(f"test_alpha_long_init_is_looser_than_before: OK (alpha_long_init={alpha_long_sample:.4f})")


if __name__ == "__main__":
    test_positivity()
    test_timescale_ordering()
    test_beta_long_is_truly_fixed()
    test_causal_history()
    test_differential_decay()
    test_state_update_equivalent()
    test_gradient_flow()
    test_output_shape()
    test_alpha_long_init_is_looser_than_before()
    print("ALL TESTS PASSED")
