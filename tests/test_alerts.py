"""Alert tests (M6): episode collapsing, refractory merging, true/false alarms, tuning uses validation only."""

from __future__ import annotations

import numpy as np

from fedguard.alerts.metrics import episode_starts, evaluate_alerts
from fedguard.alerts.policy import threshold_only, uncertainty_gated
from fedguard.alerts.tuning import apply, tune
from fedguard.eval.predictions import Predictions


def b(s: str) -> np.ndarray:
    return np.array([c == "1" for c in s])


def test_consecutive_hours_collapse():
    assert episode_starts(b("0011100")) == [2]
    assert episode_starts(b("1111111"), refractory=0) == [0]
    assert episode_starts(b("0000")) == []


def test_refractory_merging():
    #                 0123456789012345
    seq = b("1100000110000001")
    assert episode_starts(seq, refractory=6) == [0, 15]  # hour 7 run starts 6 h after end (1) -> merged
    assert episode_starts(seq, refractory=0) == [0, 7, 15]
    assert episode_starts(b("1000000010"), refractory=6) == [0, 8]  # gap of 7 > 6 -> new episode


def test_true_and_false_alarms():
    # septic patient: first positive at hour 10 -> onset 16 -> true window [4, 19]
    lab = np.zeros(30, int)
    lab[10:] = 1
    alert = np.zeros(30, bool)
    alert[1] = True  # false (too early)
    alert[12] = True  # true: lead = 16 - 12 = 4 h
    alert[27] = True  # false (too late)
    # non-septic patient with two separated episodes -> 2 false
    lab2 = np.zeros(20, int)
    alert2 = np.zeros(20, bool)
    alert2[[2, 15]] = True
    labels = np.concatenate([lab, lab2])
    alerts = np.concatenate([alert, alert2])
    offs = np.array([0, 30, 50])
    m = evaluate_alerts(alerts, labels, offs, refractory=6)
    assert (m.n_true, m.n_false, m.n_septic, m.n_detected) == (1, 4, 1, 1)
    assert m.sensitivity == 1.0 and m.median_lead_h == 4.0
    assert np.isclose(m.false_per_100h, 100 * 4 / 50)


def test_policies():
    mean = np.array([0.2, 0.7, 0.8, 0.9])
    std = np.array([0.01, 0.2, 0.05, 0.01])
    assert threshold_only(mean, 0.6).tolist() == [False, True, True, True]
    assert uncertainty_gated(mean, std, 0.6, 0.1).tolist() == [False, False, True, True]


def _preds(seed: int, n_pat: int = 60) -> Predictions:
    rng = np.random.default_rng(seed)
    pid, hour, y, mu, sd = [], [], [], [], []
    for i in range(n_pat):
        T = int(rng.integers(20, 60))
        lab = np.zeros(T, int)
        if i % 4 == 0:
            lab[int(rng.integers(5, T)) :] = 1
        m = np.clip(0.2 + 0.6 * lab + rng.normal(0, 0.15, T), 0, 1)
        s = np.abs(rng.normal(0.05, 0.03, T)) + 0.15 * (rng.random(T) < 0.2) * (
            lab == 0
        )  # noisy false spikes
        pid += [i] * T
        hour += list(range(T))
        y += list(lab)
        mu += list(m)
        sd += list(s)
    a = lambda v, t: np.asarray(v, t)  # noqa: E731
    return Predictions(
        a(pid, np.int64),
        a(hour, np.int32),
        a(y, np.int8),
        a(mu, np.float32),
        a(mu, np.float32),
        a(sd, np.float32),
    )


def test_tuning_uses_validation_only_and_respects_constraint():
    val, test = _preds(0), _preds(1)
    params = tune(val, refractory=6, max_sens_drop=0.02)
    # constraint holds on validation
    chosen = [g for g in params["grid_gate_at_tau_r"] if g["tau_s"] == params["tau_s"]]
    if chosen:
        assert chosen[0]["sensitivity"] >= params["val_threshold_only"]["sensitivity"] - 0.02 - 1e-12
    # thresholds do not depend on the test set
    params2 = tune(val, refractory=6)
    assert params2["tau_r"] == params["tau_r"] and params2["tau_s"] == params["tau_s"]
    res = apply(test, params)
    assert res["threshold_only"]["tau_r"] == params["tau_r"]
    assert (
        res["fedguard"]["false_per_100h"] <= res["threshold_only"]["false_per_100h"] + 1e-12
        or params["tau_s"] == np.inf
    )
