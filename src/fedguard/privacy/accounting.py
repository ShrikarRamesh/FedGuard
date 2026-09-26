"""RDP accounting helpers (Opacus RDPAccountant) and noise calibration for a planned training length.

A client i trains with Poisson sampling rate q_i = 1 / ceil(n_i / B) (patients; Opacus' definition) for at
most steps_i = local_epochs * R_max * ceil(n_i / B) steps. The noise multiplier is the smallest sigma (to a
tolerance) whose RDP bound after ``steps_i`` steps is <= target eps at delta. The accountant then tracks
the eps actually spent; a client stops when it reaches R_max participations or eps >= target.
"""

from __future__ import annotations

import math

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def planned_steps(n_patients: int, batch_size: int, local_epochs: int, r_max: int) -> tuple[float, int]:
    """(sample rate q, total planned optimizer steps) for one client, exactly as Opacus does it:
    ``DPDataLoader`` uses q = 1 / len(loader) with len(loader) = ceil(n / B) steps per epoch."""
    if batch_size > n_patients:
        raise ValueError(f"batch {batch_size} > n_patients {n_patients}")
    steps_per_epoch = math.ceil(n_patients / batch_size)
    return 1.0 / steps_per_epoch, local_epochs * r_max * steps_per_epoch


def calibrate_noise(target_eps: float, delta: float, q: float, steps: int) -> float:
    """Noise multiplier achieving ``target_eps`` at ``delta`` after ``steps`` Poisson-sampled steps (RDP)."""
    return float(get_noise_multiplier(target_epsilon=target_eps, target_delta=delta, sample_rate=q, steps=steps,
                                      accountant="rdp", epsilon_tolerance=0.01))  # fmt: skip


def epsilon_after(noise_multiplier: float, q: float, steps: int, delta: float) -> float:
    """RDP epsilon after ``steps`` steps (0 steps -> 0)."""
    if steps == 0:
        return 0.0
    acc = RDPAccountant()
    acc.history = [(noise_multiplier, q, steps)]
    return float(acc.get_epsilon(delta))


def check_delta(delta: float, n_patients: int) -> None:
    """Require delta < 1 / n (standard; otherwise a mechanism leaking one random record is 'private')."""
    if not delta < 1.0 / n_patients:
        raise ValueError(f"delta={delta} must be < 1/n = {1.0 / n_patients:.2e} for n={n_patients} patients")
