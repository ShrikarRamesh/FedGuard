"""Privacy tests (M5): accounting <= target, budget rules, exhaustion, patient-level sampling, public norm."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fedguard.data.features import DYN
from fedguard.fl.engine import FLEngine
from fedguard.privacy.accounting import calibrate_noise, check_delta, epsilon_after, planned_steps
from fedguard.privacy.budgets import allocate
from fedguard.privacy.public_norm import REFERENCE
from tests.test_fl import tiny_cfg, tiny_scenario  # noqa: F401  (fixture re-export)


@pytest.mark.parametrize("eps", [1.0, 3.0, 8.0])
@pytest.mark.parametrize("n", [3700, 4900])
def test_accountant_eps_below_target_after_planned_steps(eps, n):
    q, steps = planned_steps(n, 256, 5, 40)
    assert q == pytest.approx(1 / np.ceil(n / 256))
    sigma = calibrate_noise(eps, 1e-5, q, steps)
    spent = epsilon_after(sigma, q, steps, 1e-5)
    assert spent <= eps and spent > 0.95 * eps  # tight but never over
    assert epsilon_after(sigma, q, steps // 2, 1e-5) < spent  # monotone in steps


def test_delta_must_be_below_one_over_n():
    check_delta(1e-5, 4000)
    with pytest.raises(ValueError):
        check_delta(1e-3, 4000)


def test_budget_rules():
    n = {"a": 1000, "b": 2000, "c": 4000}
    assert allocate("uniform", 3.0, n) == {"a": 3.0, "b": 3.0, "c": 3.0}
    ad = allocate("adaptive", 3.0, n, a=0.55)
    assert ad["c"] == pytest.approx(3.0)
    assert ad["a"] == pytest.approx(3.0 * (0.55 + 0.45 * 0.25))
    assert ad["a"] < ad["b"] < ad["c"]
    inv = allocate("inverse", 3.0, n, a=0.55)
    assert inv["a"] == pytest.approx(3.0) and inv["a"] > inv["b"] > inv["c"]
    assert allocate("equal_noise", 3.0, n) == {"a": None, "b": None, "c": None}
    with pytest.raises(ValueError):
        allocate("nope", 3.0, n)


def test_public_norm_is_data_independent_and_complete():
    assert list(REFERENCE) == DYN
    for v, (c, s, lo, hi) in REFERENCE.items():
        assert s > 0 and lo < hi and lo <= c <= hi, v


def _events(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_dp_clients_stop_at_budget_and_spend_at_most_target(tiny_scenario, tmp_path):  # noqa: F811
    cfg = tiny_cfg("fedavg_dp", **{"fl.rounds": 5, "privacy.r_max": 2, "privacy.logical_batch_size": 4,
                                   "privacy.local_epochs": 1, "privacy.epsilon": 2.0})  # fmt: skip
    eng = FLEngine(tiny_scenario, cfg, tmp_path, 0, torch.device("cpu"))
    res = eng.run()
    ev = _events(tmp_path / "events.jsonl")
    for c, cl in eng.clients.items():
        assert cl.dp.exhausted and cl.dp.participations == 2  # R_max reached, then stops
        assert res.client_summary[c]["eps"] <= 2.0 + 1e-9
        assert res.client_summary[c]["eps"] == pytest.approx(eng.budgets[c]["planned_eps"], rel=1e-6)
        assert cl.dp.engine.accountant.history[0][1] == pytest.approx(eng.budgets[c]["sample_rate"])
        assert not cl.can_participate()
    assert any(e["type"] == "all_budgets_exhausted" for e in ev)
    assert res.versions == 2  # rounds 3..5 had no participants
    assert eng.norm == "public"


def test_adaptive_dp_budgets_follow_rule(tiny_scenario, tmp_path):  # noqa: F811
    cfg = tiny_cfg("fedguard", **{"fl.total_updates": 4, "privacy.r_max": 1, "privacy.logical_batch_size": 4,
                                  "privacy.local_epochs": 1})  # fmt: skip
    eng = FLEngine(tiny_scenario, cfg, tmp_path, 0, torch.device("cpu"))
    n = {c: cl.n_patients for c, cl in eng.clients.items()}
    exp = allocate("adaptive", 3.0, n, 0.55)
    for c, b in eng.budgets.items():
        assert b["target_eps"] == pytest.approx(exp[c]) and b["planned_eps"] <= exp[c]
    eng.run()


def _small_model(dropout=0.0):
    from fedguard.models.patchtst import PatchTST

    return PatchTST(107, 24, d_model=16, n_layers=1, n_heads=2, d_ff=16, dropout=dropout, head_hidden=8)


def test_k1_per_patient_clipping_equals_opacus():
    """k = 1: our per-patient clipped sum == Opacus' per-sample clipped sum on the same batch."""
    from opacus import GradSampleModule

    from fedguard.privacy.dp import per_patient_clipped_sum
    from fedguard.train.loops import make_loss

    torch.manual_seed(0)
    model = _small_model()
    x = torch.randn(6, 24, 107)
    m = torch.ones(6, 24, dtype=torch.bool)
    m[1, :10] = False
    x = x * m.unsqueeze(-1)
    y = torch.tensor([0.0, 1, 0, 0, 1, 0])
    loss_fn = make_loss("bce", 10.0)
    summed, norms = per_patient_clipped_sum(model, loss_fn, x[:, None], m[:, None], y[:, None], clip=0.05)
    gsm = GradSampleModule(
        _copy(model), loss_reduction="sum"
    )  # per-sample loss summed -> raw per-sample grads
    loss_sum = torch.nn.functional.binary_cross_entropy_with_logits(
        gsm(x, m), y, pos_weight=torch.tensor(10.0), reduction="sum"
    )
    loss_sum.backward()
    gs = {n.removeprefix("_module."): p.grad_sample for n, p in gsm.named_parameters()}
    ref_norms = torch.sqrt(sum((g.reshape(6, -1) ** 2).sum(1) for g in gs.values()))
    torch.testing.assert_close(norms, ref_norms, rtol=1e-4, atol=1e-6)
    fac = (0.05 / (ref_norms + 1e-6)).clamp(max=1)
    for n, g in gs.items():
        torch.testing.assert_close(summed[n], torch.einsum("p,p...->...", fac, g), rtol=1e-4, atol=1e-7)
    assert (norms > 0.05).any()  # clipping was actually exercised


def _copy(model):
    import copy

    return copy.deepcopy(model)


def test_k_windows_gradient_is_mean_over_patient_windows():
    from fedguard.privacy.dp import per_patient_clipped_sum
    from fedguard.train.loops import make_loss

    torch.manual_seed(1)
    model = _small_model()
    x = torch.randn(2, 3, 24, 107)
    m = torch.ones(2, 3, 24, dtype=torch.bool)
    y = torch.tensor([[0.0, 1, 0], [0, 0, 0]])
    loss_fn = make_loss("bce", 2.0)
    summed, norms = per_patient_clipped_sum(model, loss_fn, x, m, y, clip=1e6)  # no clipping
    model.zero_grad()
    (loss_fn(model(x[0], m[0]), y[0]) + loss_fn(model(x[1], m[1]), y[1])).backward()
    for n, p in model.named_parameters():
        torch.testing.assert_close(summed[n], p.grad, rtol=1e-4, atol=1e-6)


def test_patient_dpsgd_accounting(tiny_scenario, tmp_path):  # noqa: F811
    cfg = tiny_cfg("fedavg_dp", **{"fl.rounds": 3, "privacy.r_max": 2, "privacy.logical_batch_size": 4,
                                   "privacy.local_epochs": 1, "privacy.windows_per_patient": 4})  # fmt: skip
    eng = FLEngine(tiny_scenario, cfg, tmp_path, 0, torch.device("cpu"))
    res = eng.run()
    for c, cl in eng.clients.items():
        assert cl.dp.custom is not None and cl.dp.exhausted
        assert cl.dp.custom.steps == eng.budgets[c]["planned_steps"]  # accountant stepped exactly as planned
        assert res.client_summary[c]["eps"] == pytest.approx(eng.budgets[c]["planned_eps"], rel=1e-6)
        assert res.client_summary[c]["eps"] <= eng.budgets[c]["target_eps"] + 1e-9


def test_patient_window_dataset_one_window_per_patient(tiny_scenario):  # noqa: F811
    from fedguard.fl.client import PatientWindowDataset

    arr = tiny_scenario.arrays("A_MICU", "train", "public")
    ds = PatientWindowDataset(arr, 24, seed=0)
    assert len(ds) == arr.n_patients
    rows = ds._rows(np.arange(len(ds)))
    starts, ends = arr.offsets[:-1], arr.offsets[1:]
    assert np.all((rows >= starts) & (rows < ends))  # each sampled window belongs to that patient
    x, m, y = ds.__getitems__([0, 1])
    assert x.shape[0] == 2
