"""PhysioNet/CinC 2019 normalised utility score.

``official_utility`` calls the vendored, unmodified official implementation (``_official_2019``;
BSD-2-Clause, (c) 2019 PhysioNet). ``normalized_utility`` is a vectorised re-implementation with identical
parameters, used for fast threshold sweeps; ``tests/test_eval.py`` checks it matches the official one.

Parameters (official): dt_early = -12, dt_optimal = -6, dt_late = +3 relative to t_sepsis, where
t_sepsis = argmax(labels) + 6 (the label is already shifted 6 h early); max_u_tp = 1, min_u_fn = -2,
u_fp = -0.05, u_tn = 0. Normalised so that the best possible predictions score 1 and no predictions 0.
"""

from __future__ import annotations

import numpy as np

from fedguard.eval import _official_2019 as official

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3
MAX_U_TP, MIN_U_FN, U_FP, U_TN = 1.0, -2.0, -0.05, 0.0


def _t_sepsis_per_row(labels: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (hour index within patient, t_sepsis of that patient or +inf)."""
    lens = np.diff(offsets)
    starts = np.repeat(offsets[:-1], lens)
    hour = np.arange(offsets[-1]) - starts
    pos_rows = np.flatnonzero(labels == 1)
    patient_of = np.repeat(np.arange(len(lens)), lens)
    first = np.full(len(lens), np.inf)
    if len(pos_rows):
        # first positive row per patient (rows are sorted within patient)
        p = patient_of[pos_rows]
        uniq, first_idx = np.unique(p, return_index=True)
        first[uniq] = hour[pos_rows[first_idx]]
    t_sepsis = first - DT_OPTIMAL  # = first positive + 6
    return hour.astype(float), np.repeat(t_sepsis, lens)


def row_utility(labels: np.ndarray, preds: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-row utility of binary ``preds`` (identical to official compute_prediction_utility, per row)."""
    t, ts = _t_sepsis_per_row(np.asarray(labels), np.asarray(offsets))
    preds = np.asarray(preds).astype(bool)
    septic = np.isfinite(ts)
    d = np.where(septic, t - ts, 0.0)
    m1 = MAX_U_TP / (DT_OPTIMAL - DT_EARLY)
    b1 = -m1 * DT_EARLY
    m2 = -MAX_U_TP / (DT_LATE - DT_OPTIMAL)
    b2 = -m2 * DT_LATE
    m3 = MIN_U_FN / (DT_LATE - DT_OPTIMAL)
    b3 = -m3 * DT_OPTIMAL
    early = d <= DT_OPTIMAL
    tp = np.where(early, np.maximum(m1 * d + b1, U_FP), m2 * d + b2)
    fn = np.where(early, 0.0, m3 * d + b3)
    u_septic = np.where(preds, tp, fn)
    u_septic = np.where(d <= DT_LATE, u_septic, 0.0)
    u_non = np.where(preds, U_FP, U_TN)
    return np.where(septic, u_septic, u_non)


def best_predictions(labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Official 'best' predictions: 1 on [t_sepsis - 12, t_sepsis + 3] for septic patients."""
    t, ts = _t_sepsis_per_row(np.asarray(labels), np.asarray(offsets))
    return (np.isfinite(ts) & (t >= ts + DT_EARLY) & (t <= ts + DT_LATE)).astype(np.int8)


def normalized_utility(labels: np.ndarray, preds: np.ndarray, offsets: np.ndarray) -> float:
    """Cohort normalised utility (vectorised). ``offsets`` delimit patients in the row arrays."""
    labels = np.asarray(labels)
    obs = row_utility(labels, preds, offsets).sum()
    best = row_utility(labels, best_predictions(labels, offsets), offsets).sum()
    inaction = row_utility(labels, np.zeros_like(labels), offsets).sum()
    if best == inaction:
        return float("nan")
    return float((obs - inaction) / (best - inaction))


def official_utility(labels: np.ndarray, preds: np.ndarray, offsets: np.ndarray) -> float:
    """Cohort normalised utility computed with the vendored official per-patient function (slow)."""
    kw = dict(dt_early=DT_EARLY, dt_optimal=DT_OPTIMAL, dt_late=DT_LATE, max_u_tp=MAX_U_TP,
              min_u_fn=MIN_U_FN, u_fp=U_FP, u_tn=U_TN)  # fmt: skip
    labels = np.asarray(labels)
    best = best_predictions(labels, offsets)
    obs = bst = ina = 0.0
    for s, e in zip(offsets[:-1], offsets[1:], strict=True):
        lab = labels[s:e]
        obs += official.compute_prediction_utility(lab, np.asarray(preds[s:e]), **kw)
        bst += official.compute_prediction_utility(lab, best[s:e], **kw)
        ina += official.compute_prediction_utility(lab, np.zeros(e - s), **kw)
    return float((obs - ina) / (bst - ina))
