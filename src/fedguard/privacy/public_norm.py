"""Data-independent ("public") normalisation for DP clients (D3).

DP-SGD's guarantee assumes the per-sample inputs do not depend on other patients' data. Standardising
with a client's own training mean/std violates that (changing one patient shifts everyone's inputs). DP
runs therefore standardise with fixed clinical reference values that were written down *before* looking
at the data: value' = (clip(value, lo, hi) - center) / scale. Centers/scales follow standard adult
reference ranges (center approx. mid-normal, scale approx. half the normal range, widened for ICU spread);
lo/hi are generous physiological plausibility bounds. They are hyperparameters, not statistics of the data.
"""

from __future__ import annotations

import numpy as np

from fedguard.data.features import DYN

# variable: (center, scale, lo, hi)   units as in PhysioNet 2019
REFERENCE: dict[str, tuple[float, float, float, float]] = {
    "HR": (80, 20, 20, 250),
    "O2Sat": (97, 3, 50, 100),
    "Temp": (37, 0.8, 30, 43),
    "SBP": (120, 25, 40, 280),
    "MAP": (85, 15, 20, 200),
    "DBP": (70, 15, 10, 180),
    "Resp": (16, 5, 3, 70),
    "EtCO2": (40, 5, 5, 100),
    "BaseExcess": (0, 4, -30, 30),
    "HCO3": (24, 4, 5, 50),
    "FiO2": (0.5, 0.2, 0.2, 1.0),
    "pH": (7.4, 0.08, 6.8, 7.8),
    "PaCO2": (40, 8, 10, 150),
    "SaO2": (97, 3, 50, 100),
    "AST": (30, 50, 0, 1000),
    "BUN": (20, 15, 0, 250),
    "Alkalinephos": (100, 60, 0, 1500),
    "Calcium": (9, 1, 2, 20),
    "Chloride": (104, 6, 60, 150),
    "Creatinine": (1.0, 1.0, 0, 20),
    "Bilirubin_direct": (0.5, 2, 0, 40),
    "Glucose": (120, 40, 10, 1000),
    "Lactate": (1.5, 1.5, 0, 30),
    "Magnesium": (2, 0.4, 0, 10),
    "Phosphate": (3.5, 1, 0, 20),
    "Potassium": (4.1, 0.6, 1, 10),
    "Bilirubin_total": (1, 2, 0, 50),
    "TroponinI": (0, 2, 0, 50),
    "Hct": (35, 6, 10, 70),
    "Hgb": (11, 2, 3, 25),
    "PTT": (32, 10, 10, 200),
    "WBC": (10, 5, 0, 200),
    "Fibrinogen": (300, 100, 30, 1500),
    "Platelets": (200, 100, 0, 1500),
}
assert list(REFERENCE) == DYN, "reference table must follow DYN order"


def public_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(center, scale, lo, hi) as float32 arrays in DYN order."""
    t = np.array([REFERENCE[v] for v in DYN], dtype=np.float32)
    return t[:, 0], t[:, 1], t[:, 2], t[:, 3]
