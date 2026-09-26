"""Model tests (M2): shapes, padding, Opacus compatibility, MC Dropout, baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from fedguard.data.features import channel_spec
from fedguard.models.baselines import GRUClassifier, handcrafted_features
from fedguard.models.patchtst import PatchTST, num_patches
from fedguard.models.uncertainty import mc_dropout_mode, mc_predict

C, L = 107, 24


def small(dropout: float = 0.2, lookback: int = L) -> PatchTST:
    return PatchTST(C, lookback, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=dropout, head_hidden=16)


def batch(b: int = 6, lookback: int = L, n_real: list[int] | None = None):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, lookback, C, generator=g)
    m = torch.ones(b, lookback, dtype=torch.bool)
    for i, r in enumerate(n_real or []):
        m[i, : lookback - r] = False
    x = x * m.unsqueeze(-1)
    return x, m


@pytest.mark.parametrize("lookback,expected", [(12, 5), (24, 11), (48, 23)])
def test_patch_count(lookback, expected):
    assert num_patches(lookback, 4, 2) == expected
    x, m = batch(3, lookback)
    assert small(lookback=lookback)(x, m).shape == (3,)


def test_padding_mask_is_respected():
    model = small(0.0).eval()
    x, m = batch(4, n_real=[1, 3, 10, 24])
    out = model(x, m)
    x2 = x.clone()
    x2[~m] = torch.randn(int((~m).sum()), C)  # garbage in padded hours
    torch.testing.assert_close(model(x2, m), out)
    # changing a real hour does change the output
    x3 = x.clone()
    x3[0, -1] += 5.0
    assert not torch.allclose(model(x3, m)[0], out[0])


def test_opacus_validator_and_per_sample_grads():
    from opacus import GradSampleModule
    from opacus.validators import ModuleValidator

    model = small()
    assert ModuleValidator.validate(model, strict=True) == []
    gsm = GradSampleModule(model)
    x, m = batch(5, n_real=[2, 24, 24, 7, 24])
    loss = nn.functional.binary_cross_entropy_with_logits(gsm(x, m), torch.ones(5))
    loss.backward()
    for name, p in gsm.named_parameters():
        assert p.grad_sample is not None, name
        assert p.grad_sample.shape == (5, *p.shape), (name, p.grad_sample.shape)
    # per-sample gradients sum to the batch gradient (mean reduction -> divide by 5)
    for name, p in gsm.named_parameters():
        torch.testing.assert_close(p.grad_sample.sum(0) / 5, p.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_opacus_make_private_step_runs():
    from opacus import PrivacyEngine

    warnings.filterwarnings("ignore")
    model = small()
    x, m = batch(64)
    y = (torch.rand(64) > 0.8).float()
    ds = torch.utils.data.TensorDataset(x, m, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=16)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    pe = PrivacyEngine(accountant="rdp")
    model, opt, dl = pe.make_private(
        module=model, optimizer=opt, data_loader=dl, noise_multiplier=1.0, max_grad_norm=1.0
    )
    for xb, mb, yb in dl:
        if len(yb) == 0:
            continue
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(xb, mb), yb).backward()
        opt.step()
    assert pe.get_epsilon(1e-5) > 0


def test_mc_dropout_std():
    x, m = batch(8)
    model = small(0.3).eval()
    mu, sd = mc_predict(model, x, m, T=20, generator_seed=0)
    assert mu.shape == sd.shape == (8,) and (sd > 0).all()
    assert not model.training  # previous (eval) mode restored
    model.train()
    mc_predict(model, x, m, T=2)
    assert model.training  # previous (train) mode restored
    mu0, sd0 = mc_predict(small(0.0), x, m, T=10)
    assert torch.all(sd0 < 1e-6)  # no dropout -> no spread
    # plain eval mode is deterministic
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x, m), model(x, m))
    with mc_dropout_mode(model):
        assert all(d.training for d in model.modules() if isinstance(d, nn.Dropout))
        assert not any(mod.training for mod in model.modules() if isinstance(mod, nn.LayerNorm))


def test_attention_maps_available():
    model = small(0.0).eval()
    model.set_keep_attention(True)
    x, m = batch(2, n_real=[3, 24])
    model(x, m)
    maps = model.attention_maps()
    assert len(maps) == 2 and maps[0].shape == (2, 4, 12, 12)
    torch.testing.assert_close(maps[0].sum(-1), torch.ones(2, 4, 12))
    # padded patches receive ~zero attention
    assert maps[0][0, :, :, 1:10].max() < 1e-6


def test_gru_and_handcrafted():
    x, m = batch(4, n_real=[1, 5, 24, 24])
    assert GRUClassifier(C, hidden=8)(x, m).shape == (4,)
    spec = channel_spec(True, True, ["Age", "Gender", "ICULOS", "HospAdmTime"])
    f = handcrafted_features(x.numpy(), m.numpy(), spec)
    assert f.shape[0] == 4 and np.isfinite(f).all()
    # 34 last + 2 windows x (mean,min,max,slope) x 34 + mask last + mask count + 5 static
    assert f.shape[1] == 34 + 8 * 34 + 2 * 34 + 5
