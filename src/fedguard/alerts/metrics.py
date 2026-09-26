"""Alarm-episode metrics (D22).

Episodes: a maximal run of consecutive alerting hours is one episode. After an episode ends at hour e,
a new run starting at hour s <= e + refractory is merged into the same episode (the alarm is still
"active"); a run starting later is a new episode. Each episode is one alarm, at its start hour.

True alarm: an episode that starts within [onset - 12 h, onset + 3 h] of a septic patient (the PhysioNet
utility window; onset = t_sepsis = first positive label + 6). Every other episode is a false alarm.

Metrics: false alarms per 100 patient-hours, patient-level sensitivity (septic patients with >= 1 true
alarm), median lead time (onset - start of the first true alarm, hours; positive = before onset), the
number of alarms per 100 patient-hours, and the normalised PhysioNet utility of the hourly alert state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedguard.eval.utility import normalized_utility

EARLY, LATE = 12, 3  # true-alarm window relative to onset


def episode_starts(alert: np.ndarray, refractory: int = 6) -> list[int]:
    """Start hours of alarm episodes for ONE patient's hourly boolean alert sequence."""
    a = np.asarray(alert, dtype=bool)
    if not a.any():
        return []
    padded = np.concatenate([[False], a, [False]])
    d = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(d == 1)
    run_ends = np.flatnonzero(d == -1) - 1  # inclusive
    starts, last_end = [], -(10**9)
    for s, e in zip(run_starts, run_ends, strict=True):
        if starts and s <= last_end + refractory:
            last_end = e  # merged into the active episode
            continue
        starts.append(int(s))
        last_end = e
    return starts


@dataclass
class AlertSummary:
    false_per_100h: float
    alarms_per_100h: float
    sensitivity: float
    median_lead_h: float
    utility: float
    n_false: int
    n_true: int
    n_septic: int
    n_detected: int
    patient_hours: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def evaluate_alerts(
    alert: np.ndarray, labels: np.ndarray, offsets: np.ndarray, refractory: int = 6,
    exclude_ambiguous_lead: np.ndarray | None = None,
) -> AlertSummary:  # fmt: skip
    """Alert metrics over a cohort. ``offsets`` delimit patients; ``labels`` are hourly SepsisLabel."""
    alert = np.asarray(alert, dtype=bool)
    labels = np.asarray(labels)
    n_false = n_true = n_septic = n_detected = 0
    leads: list[float] = []
    for i, (s, e) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        lab = labels[s:e]
        starts = episode_starts(alert[s:e], refractory)
        pos = np.flatnonzero(lab == 1)
        if len(pos) == 0:
            n_false += len(starts)
            continue
        n_septic += 1
        onset = int(pos[0]) + 6
        true = [h for h in starts if onset - EARLY <= h <= onset + LATE]
        n_true += len(true)
        n_false += len(starts) - len(true)
        if true:
            n_detected += 1
            if exclude_ambiguous_lead is None or not exclude_ambiguous_lead[i]:
                leads.append(float(onset - true[0]))
    hours = int(offsets[-1])
    return AlertSummary(
        false_per_100h=100.0 * n_false / hours if hours else float("nan"),
        alarms_per_100h=100.0 * (n_false + n_true) / hours if hours else float("nan"),
        sensitivity=n_detected / n_septic if n_septic else float("nan"),
        median_lead_h=float(np.median(leads)) if leads else float("nan"),
        utility=normalized_utility(labels, alert.astype(np.int8), offsets),
        n_false=n_false, n_true=n_true, n_septic=n_septic, n_detected=n_detected, patient_hours=hours,
    )  # fmt: skip
