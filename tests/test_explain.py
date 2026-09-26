"""Explanation tests: IG completeness, per-variable grouping, attention rollout normalisation."""

from __future__ import annotations

import numpy as np
import torch

from fedguard.data.features import channel_spec
from fedguard.explain.attention import rollout
from fedguard.explain.integrated_gradients import integrated_gradients, per_variable
from fedguard.models.patchtst import PatchTST


def test_ig_completeness_and_grouping():
    spec = channel_spec(True, True, ["Age", "Gender", "ICULOS", "HospAdmTime"])
    model = PatchTST(
        spec.n_channels, 24, d_model=16, n_layers=1, n_heads=2, d_ff=16, dropout=0.0, head_hidden=8
    ).eval()
    torch.manual_seed(0)
    x = torch.randn(3, 24, spec.n_channels)
    m = torch.ones(3, 24, dtype=torch.bool)
    attr = integrated_gradients(model, x, m, n_steps=200)
    with torch.no_grad():
        gap = model(x, m) - model(torch.zeros_like(x), m)
    torch.testing.assert_close(attr.sum((1, 2)), gap, atol=2e-2, rtol=2e-2)  # completeness (logit)
    names, pv = per_variable(attr.numpy(), spec)
    assert names[:2] == ["HR", "O2Sat"] and "HospAdmTime" in names and len(names) == 34 + 4
    np.testing.assert_allclose(pv.sum(-1), attr.numpy().sum(-1), atol=1e-4)


def test_rollout_sums_to_one_over_real_hours():
    model = PatchTST(10, 24, d_model=16, n_layers=2, n_heads=2, d_ff=16, dropout=0.0, head_hidden=8)
    x = torch.randn(2, 24, 10)
    m = torch.ones(2, 24, dtype=torch.bool)
    m[0, :20] = False
    x = x * m.unsqueeze(-1)
    r = rollout(model, x, m)
    assert r.shape == (2, 24)
    np.testing.assert_allclose(r.sum(1), 1.0, atol=1e-5)
    assert np.all(r[0, :20] == 0)
    assert model.attention_maps()[0] is None  # keep_attention switched off again
