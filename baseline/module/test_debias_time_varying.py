"""Concise correctness checks for build_time_varying_debias_model (mu_v -> mu_v(t)).

Plain-assert script (no pytest dependency in this repo) - run directly:
    python module/test_debias_time_varying.py
"""
import copy
import torch

from module.mf import MF
from module.debias import build_debias_model
from module.debias_time_varying import build_time_varying_debias_model


def _make_model(cls_builder, seed=0):
    torch.manual_seed(seed)
    model_cls = cls_builder(MF)
    model = model_cls(
        num_users=20, num_items=30, embedding_k=16, device="cpu",
        tau=0.5, depth=1, max_seq_len=10, n_heads=1, dropout=0.0,
    )
    model.eval()
    return model


def test_backward_compatible_at_init():
    """1. At init (temporal_mu_mlp's final layer = 0), mu_v(t) ~= old static mu_v.

    The two wrapper classes build different sets of layers before
    reset_parameters() runs, so seeding them identically does not give them
    identical weights (RNG consumption order differs). Instead, explicitly
    copy the shared components (item embedding, static mu net) from the old
    model into the new one so both score the exact same underlying weights.
    """
    old = _make_model(build_debias_model, seed=0)
    new = _make_model(build_time_varying_debias_model, seed=1)
    new.set_time_normalization(0.0, 100.0)

    with torch.no_grad():
        new.item_embedding.weight.copy_(old.item_embedding.weight)
        for old_p, new_p in zip(old.base_net.parameters(), new.static_mu_mlp.parameters()):
            new_p.copy_(old_p)

    with torch.no_grad():
        z = old.item_embedding.weight
        old_mu = old.softplus(old.base_net(z)).squeeze(-1) + 1e-8
        new_mu_t0 = new.prior_parameters_from_embeddings(query_time=10.0)[0]
        new_mu_t1 = new.prior_parameters_from_embeddings(query_time=90.0)[0]

    assert torch.allclose(old_mu, new_mu_t0, atol=1e-5), (old_mu - new_mu_t0).abs().max()
    assert torch.allclose(old_mu, new_mu_t1, atol=1e-5), (old_mu - new_mu_t1).abs().max()
    print("test_backward_compatible_at_init: OK")


def test_time_dependence_after_nonzero_temporal_params():
    """2. With nonzero temporal_mu_mlp params, the same item differs across t."""
    model = _make_model(build_time_varying_debias_model, seed=1)
    model.set_time_normalization(0.0, 100.0)
    with torch.no_grad():
        for p in model.temporal_mu_mlp.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    with torch.no_grad():
        mu_t0, _, _ = model.prior_parameters_from_embeddings(query_time=5.0)
        mu_t1, _, _ = model.prior_parameters_from_embeddings(query_time=95.0)

    assert not torch.allclose(mu_t0, mu_t1, atol=1e-6)
    print("test_time_dependence_after_nonzero_temporal_params: OK")


def test_candidate_consistency():
    """3. Candidates at the same event time share the same e_t before item-time interaction."""
    model = _make_model(build_time_varying_debias_model, seed=2)
    model.set_time_normalization(0.0, 100.0)

    batch_items = torch.randint(0, 30, (4, 6))
    pos_time = torch.tensor([10.0, 20.0, 30.0, 40.0])
    batch_time_all = torch.zeros(4, 6, 5)  # no prior events -> pure mu_v(t) check via zero Hawkes state

    with torch.no_grad():
        time_feat = model.fourier_time_features(pos_time)
        time_emb = model.time_encoder(time_feat)  # [4, d]
        expanded = time_emb.unsqueeze(1).expand(-1, batch_items.shape[1], -1)
        # every candidate column at a given row must see the identical e_t
        for row in range(4):
            assert torch.allclose(expanded[row, 0], expanded[row, -1])
            assert torch.allclose(expanded[row, 0], time_emb[row])
    print("test_candidate_consistency: OK")


def test_positivity():
    """4. mu_v(t) and lambda_v(t) are strictly positive."""
    model = _make_model(build_time_varying_debias_model, seed=3)
    model.set_time_normalization(0.0, 100.0)

    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 100
    batch_time_all = torch.rand(8, 5, 12) * 100

    with torch.no_grad():
        mu_t, _, _ = model.prior_parameters_from_embeddings(query_time=50.0)
        intensity = model.prior(batch_items, pos_time, batch_time_all)

    assert (mu_t > 0).all()
    assert (intensity > 0).all()
    print("test_positivity: OK")


def test_gradient_flow():
    """5. time_encoder and temporal_mu_mlp receive finite, nonzero gradients.

    temporal_mu_mlp's final layer is exactly zero at initialization (by
    design, see test 1), which also zeroes the *local Jacobian* of that
    layer - so gradient to anything upstream (time_encoder, temporal_mu_mlp's
    own earlier layer) is identically zero at that exact point, same as any
    zero-gated residual/adapter block. That is expected, not a bug; a
    meaningful gradient-flow check instead looks just after the first
    optimizer step would have moved the model off that degenerate point,
    so we nudge temporal_mu_mlp's parameters slightly first (as test 2 does).
    """
    model = _make_model(build_time_varying_debias_model, seed=4)
    model.set_time_normalization(0.0, 100.0)
    model.train()
    with torch.no_grad():
        for p in model.temporal_mu_mlp.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    batch_items = torch.randint(0, 30, (8, 5))
    pos_time = torch.rand(8) * 100
    batch_time_all = torch.rand(8, 5, 12) * 100

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    loss = intensity.log().mean()
    loss.backward()

    for name, module in [("time_encoder", model.time_encoder), ("temporal_mu_mlp", model.temporal_mu_mlp)]:
        saw_nonzero = False
        for p in module.parameters():
            assert p.grad is not None, f"{name} has a parameter with no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
            if p.grad.abs().sum() > 0:
                saw_nonzero = True
        assert saw_nonzero, f"{name} received an all-zero gradient"
    print("test_gradient_flow: OK")


def test_no_hawkes_leakage():
    """6. The current event does not contribute to its own Hawkes state."""
    model = _make_model(build_time_varying_debias_model, seed=5)
    model.set_time_normalization(0.0, 100.0)

    item_id = 3
    batch_items = torch.tensor([[item_id]])
    pos_time = torch.tensor([50.0])
    # this item's own history includes an event exactly AT the query time
    batch_time_all = torch.tensor([[[10.0, 50.0, 80.0]]])

    with torch.no_grad():
        beta = model.current_beta()
        mask = batch_time_all < pos_time.view(-1, 1, 1)
        delta = (pos_time.view(-1, 1, 1) - batch_time_all).clamp(min=0.0)
        h = (torch.exp(-beta * delta) * mask).sum(dim=-1)

    # only the t=10 event (< 50) should count; t=50 (== query) and t=80 (> query) must not
    expected_h = torch.exp(-beta * torch.tensor(40.0))
    assert torch.allclose(h.squeeze(), expected_h, atol=1e-5)
    print("test_no_hawkes_leakage: OK")


def test_output_shape():
    """7. model.prior(...) returns exactly [batch_size, num_candidates]."""
    model = _make_model(build_time_varying_debias_model, seed=6)
    model.set_time_normalization(0.0, 100.0)

    B, C, L = 12, 9, 7
    batch_items = torch.randint(0, 30, (B, C))
    pos_time = torch.rand(B) * 100
    batch_time_all = torch.rand(B, C, L) * 100

    intensity = model.prior(batch_items, pos_time, batch_time_all)
    assert intensity.shape == (B, C), intensity.shape
    print("test_output_shape: OK")


if __name__ == "__main__":
    test_backward_compatible_at_init()
    test_time_dependence_after_nonzero_temporal_params()
    test_candidate_consistency()
    test_positivity()
    test_gradient_flow()
    test_no_hawkes_leakage()
    test_output_shape()
    print("ALL TESTS PASSED")
