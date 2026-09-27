"""Gradient-inversion tests: batched per-window gradients equal the per-window loop; DP noise/clipping; iDLG."""

from __future__ import annotations

import torch

from fedguard.attack.gradient_inversion import flat_grad, observed_gradient, per_window_grads
from fedguard.models.patchtst import PatchTST
from fedguard.train.loops import make_loss


def small():
    torch.manual_seed(0)
    return PatchTST(107, 24, d_model=16, n_layers=1, n_heads=2, d_ff=16, dropout=0.0, head_hidden=8).eval()


def test_batched_grads_match_loop_and_are_differentiable():
    model = small()
    loss_fn = make_loss("bce", 10.0)
    x = torch.randn(3, 24, 107, requires_grad=True)
    m = torch.ones(3, 24, dtype=torch.bool)
    y = torch.tensor([0.0, 1.0, 0.0])
    g = per_window_grads(model, loss_fn, x, m, y)
    for i in range(3):
        ref = flat_grad(model, loss_fn(model(x[i : i + 1], m[i : i + 1]), y[i : i + 1]))
        torch.testing.assert_close(g[i], ref, rtol=1e-4, atol=1e-6)
    # second-order: the matching loss is differentiable w.r.t. the dummy inputs, and window 0's loss only
    # depends on window 0
    g[0].pow(2).sum().backward()
    assert x.grad[0].abs().sum() > 0 and torch.all(x.grad[1:] == 0)


def test_observed_gradient_clip_and_noise():
    model = small()
    loss_fn = make_loss("bce", 10.0)
    x, m, y = torch.randn(1, 24, 107), torch.ones(1, 24, dtype=torch.bool), torch.tensor([1.0])
    raw = observed_gradient(model, x, m, y, loss_fn, None, 0.0, torch.Generator().manual_seed(0))
    clipped = observed_gradient(model, x, m, y, loss_fn, 1e-3, 0.0, torch.Generator().manual_seed(0))
    assert torch.isclose(clipped.norm(), torch.tensor(1e-3), rtol=1e-3)
    noisy = observed_gradient(model, x, m, y, loss_fn, 1.0, 5.0, torch.Generator().manual_seed(0))
    assert (noisy - raw * min(1, 1 / raw.norm())).std() > 4.0  # noise std = sigma * C
    assert float(raw[-1]) < 0  # iDLG: y = 1 -> negative output-bias gradient
