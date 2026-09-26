"""Evaluation tests: official utility equivalence, patient bootstrap, calibration, NaN handling."""

from __future__ import annotations

import numpy as np

from fedguard.eval import _official_2019 as official
from fedguard.eval import metrics as M
from fedguard.eval.utility import normalized_utility, official_utility, row_utility


def cohort(seed: int = 0, n: int = 60):
    rng = np.random.default_rng(seed)
    labels, preds, offs = [], [], [0]
    for i in range(n):
        T = int(rng.integers(3, 80))
        lab = np.zeros(T, int)
        if i % 3 == 0:
            lab[int(rng.integers(0, T)) :] = 1
        labels.append(lab)
        preds.append((rng.random(T) > 0.7).astype(int))
        offs.append(offs[-1] + T)
    return np.concatenate(labels), np.concatenate(preds), np.array(offs)


def test_official_docstring_example():
    # example from the official file: labels [0,0,0,0,1,1], predictions [0,0,1,1,1,1] -> 3.388888888888889
    lab, pr = np.array([0, 0, 0, 0, 1, 1]), np.array([0, 0, 1, 1, 1, 1])
    assert np.isclose(official.compute_prediction_utility(lab, pr), 3.388888888888889)
    assert np.isclose(row_utility(lab, pr, np.array([0, 6])).sum(), 3.388888888888889)


def test_vectorised_utility_matches_official():
    for seed in range(5):
        y, p, offs = cohort(seed)
        per_patient_official = [
            official.compute_prediction_utility(y[s:e], p[s:e])
            for s, e in zip(offs[:-1], offs[1:], strict=True)
        ]
        u = row_utility(y, p, offs)
        per_patient_ours = [u[s:e].sum() for s, e in zip(offs[:-1], offs[1:], strict=True)]
        np.testing.assert_allclose(per_patient_ours, per_patient_official, atol=1e-9)
        assert np.isclose(normalized_utility(y, p, offs), official_utility(y, p, offs))
    y, p, offs = cohort(1)
    assert np.isclose(normalized_utility(y, np.zeros_like(y), offs), 0.0)


def test_metrics_nan_when_not_computable():
    y = np.zeros(10)
    p = np.linspace(0, 1, 10)
    assert np.isnan(M.auroc(y, p)) and np.isnan(M.auprc(y, p))
    s = M.summary(y, p)
    assert s["prevalence"] == 0 and np.isfinite(s["brier"])
    assert np.isnan(M.summary(np.array([]), np.array([]))["ece"])


def test_ece():
    y = np.array([0, 1, 0, 1])
    assert M.ece(y, np.array([0.0, 1.0, 0.0, 1.0])) == 0.0
    assert np.isclose(M.ece(np.array([0, 0]), np.array([0.9, 0.9])), 0.9)


def test_bootstrap_resamples_patients():
    rng = np.random.default_rng(0)
    # 30 patients; each patient's rows share one label so row- and patient-level bootstraps differ
    patient = np.repeat(np.arange(30), 20)
    y = np.repeat(rng.integers(0, 2, 30), 20).astype(float)
    p = np.clip(y * 0.3 + rng.random(600) * 0.7, 0, 1)
    ci = M.bootstrap_ci(y, p, patient, n_boot=200, seed=1)
    lo, hi = ci["auroc"]
    assert lo <= M.auroc(y, p) <= hi and hi - lo > 0
    # identical under shuffling of rows (grouping is by patient id, not position)
    perm = rng.permutation(600)
    assert M.bootstrap_ci(y[perm], p[perm], patient[perm], n_boot=200, seed=1)["auroc"] == ci["auroc"]
    # each resample keeps whole patients: with one class per patient, drawn row counts are multiples of 20
    order = np.argsort(patient, kind="stable")
    _, starts, counts = np.unique(patient[order], return_index=True, return_counts=True)
    draw = np.random.default_rng(5).integers(0, 30, 30)
    c = counts[draw]
    idx = np.repeat(starts[draw], c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
    assert len(idx) == 600 and set(np.unique(patient[order][idx])) <= set(
        np.unique(patient[order][starts[draw]])
    )
