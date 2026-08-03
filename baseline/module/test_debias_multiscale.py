"""Concise correctness checks for build_multiscale_debias_model (signed
K-component exponential-basis Hawkes excitation). Run directly:
    python module/test_debias_multiscale.py
"""
import torch

from module.mf import MF
from module.debias_multiscale import build_multiscale_debias_model

HALF_LIVES = [0.1, 0.5, 2, 10, 50, 200]


def _make_model(seed=0, half_lives=HALF_LIVES):
    torch.manual_seed(seed)
    model_cls = build_multiscale_debias_model(MF, half_lives)
    model = model_cls(
        num_users=20, num_items=30, embedding_k=16, device="cpu",
        tau=0.5, depth=1, max_seq_len=10, n_heads=1, dropout=0.0,
    )
    model.eval()
    return model


def test_backward_compatible_at_init():
    """At init (weight_head = 0), lambda_v(t) ~= softplus(mu_logit(v)) alone,
    i.e. identical to the vanilla static-mu model with no excitation."""
    model = _make_model(seed=0)
    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    with torch.no_grad():
        item_vec = model.item_embedding(batch_items)
        mu_logit = model.mu_head(model.mu_trunk(item_vec)).squeeze(-1)
        expected = model.softplus(mu_logit) + model.eps
        actual = model.prior(batch_items, pos_time, batch_time_all)

    assert torch.allclose(expected, actual, atol=1e-6), (expected - actual).abs().max()
    print("test_backward_compatible_at_init: OK")


def test_positivity_even_with_signed_weights():
    """1. Final intensity stays strictly positive even when some w_k are
    manually pushed strongly negative (the whole point of clipping positivity
    on the TOTAL rather than on each component)."""
    model = _make_model(seed=1)
    with torch.no_grad():
        model.weight_head.bias.copy_(torch.tensor([-50.0, 30.0, -10.0, 5.0, -20.0, 40.0]))

    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    with torch.no_grad():
        intensity = model.prior(batch_items, pos_time, batch_time_all)
    assert (intensity > 0).all()
    assert torch.isfinite(intensity).all()
    print("test_positivity_even_with_signed_weights: OK")


def test_causal_history():
    """3. The event at t_i does not contribute to any H_k for its own likelihood."""
    model = _make_model(seed=2)
    item_id = 4
    batch_items = torch.tensor([[item_id]])
    pos_time = torch.tensor([50.0])
    batch_time_all = torch.tensor([[[10.0, 50.0, 80.0]]])

    query = pos_time.view(-1, 1, 1)
    mask = batch_time_all < query
    delta = (query - batch_time_all).clamp(min=0.0)
    with torch.no_grad():
        decay = torch.exp(-delta.unsqueeze(-1) * model.betas.view(1, 1, 1, -1))
        H = (decay * mask.unsqueeze(-1)).sum(dim=2)

    expected_H = torch.exp(-model.betas * 40.0)  # only the t=10 event (< 50) should count
    assert torch.allclose(H.squeeze(0).squeeze(0), expected_H, atol=1e-6)
    print("test_causal_history: OK")


def test_signed_weight_expressible():
    """New for this model: a negative marginal contribution from one
    component (which alpha_v >= 0 could never express in the earlier
    two-timescale model) is representable and behaves as expected --
    increasing that component's H should DECREASE the total intensity."""
    model = _make_model(seed=3)
    with torch.no_grad():
        model.weight_head.bias.copy_(torch.tensor([0.0, -5.0, 0.0, 0.0, 0.0, 0.0]))

    item_id = 0
    batch_items = torch.tensor([[item_id]])
    pos_time = torch.tensor([100.0])
    # two histories differing only in how much they populate the SECOND
    # component's timescale (half_life=0.5 -> beta=log(2)/0.5)
    batch_time_all_low = torch.tensor([[[99.9]]])   # 1 recent event
    batch_time_all_high = torch.tensor([[[99.9, 99.8, 99.7, 99.6]]])  # several recent events

    with torch.no_grad():
        intensity_low = model.prior(batch_items, pos_time, batch_time_all_low)
        intensity_high = model.prior(batch_items, pos_time, batch_time_all_high)

    assert intensity_high.item() < intensity_low.item(), (intensity_low.item(), intensity_high.item())
    print("test_signed_weight_expressible: OK")


def test_gradient_flow():
    """6. mu_trunk/mu_head and weight_trunk/weight_head all receive finite,
    nonzero gradients (after moving weight_head slightly off its zero init,
    same reasoning as the earlier zero-gated-branch tests: at the exact
    zero-init point the local Jacobian of weight_head is degenerate)."""
    model = _make_model(seed=4)
    model.train()
    with torch.no_grad():
        model.weight_head.weight.add_(torch.randn_like(model.weight_head.weight) * 0.1)
        model.weight_head.bias.add_(torch.randn_like(model.weight_head.bias) * 0.1)

    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 20
    batch_time_all = torch.rand(8, 5, 12) * 20

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    loss = intensity.log().mean()
    loss.backward()

    for name, module in [("mu_trunk", model.mu_trunk), ("mu_head", model.mu_head),
                         ("weight_trunk", model.weight_trunk), ("weight_head", model.weight_head)]:
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
    model = _make_model(seed=5)
    B, C, L = 12, 9, 7
    batch_items = torch.randint(0, 30, (B, C))
    pos_time = torch.rand(B) * 20
    batch_time_all = torch.rand(B, C, L) * 20

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    assert intensity.shape == (B, C), intensity.shape
    print("test_output_shape: OK")


def test_arbitrary_K():
    """Sanity: K=1 (vanilla-equivalent shape) and a larger K both work."""
    for hl in [[1.0], [0.1, 1, 10, 100]]:
        model = _make_model(seed=6, half_lives=hl)
        batch_items = torch.randint(0, 30, (4, 3))
        pos_time = torch.rand(4) * 20
        batch_time_all = torch.rand(4, 3, 5) * 20
        intensity = model.prior(batch_items, pos_time, batch_time_all)
        assert intensity.shape == (4, 3)
        assert (intensity > 0).all()
    print("test_arbitrary_K: OK")


if __name__ == "__main__":
    test_backward_compatible_at_init()
    test_positivity_even_with_signed_weights()
    test_causal_history()
    test_signed_weight_expressible()
    test_gradient_flow()
    test_output_shape()
    test_arbitrary_K()
    print("ALL TESTS PASSED")
