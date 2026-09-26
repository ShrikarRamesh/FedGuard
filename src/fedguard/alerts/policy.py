"""Alert policies on hourly MC-Dropout outputs (mean risk, std).

* threshold-only:         alert(t) = mean(t) > tau_r
* uncertainty-gated:      alert(t) = mean(t) > tau_r  and  std(t) < tau_s          (FedGuard)

Both policies use the same MC-Dropout mean, so the only difference is the uncertainty gate.
"""

from __future__ import annotations

import numpy as np


def threshold_only(mean: np.ndarray, tau_r: float) -> np.ndarray:
    return np.asarray(mean) > tau_r


def uncertainty_gated(mean: np.ndarray, std: np.ndarray, tau_r: float, tau_s: float) -> np.ndarray:
    return (np.asarray(mean) > tau_r) & (np.asarray(std) < tau_s)
