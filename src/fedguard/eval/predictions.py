"""Row-level predictions saved in every run dir (``preds_<split>.npz``) and the standard evaluation report.

Saved arrays (one entry per patient-hour, patients contiguous, hours ascending):
    patient_idx  int64   index into the processed patients table
    hour         int32   hour within the stay (0-based)
    y            int8    SepsisLabel
    p            float32 deterministic probability (eval mode)
    p_mc_mean / p_mc_std float32 (optional) MC-Dropout mean / std
Downstream steps (alerts, report, export) read only these files, never re-run models on test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fedguard.data.windows import ClientArrays
from fedguard.eval import metrics as M
from fedguard.eval.utility import normalized_utility


@dataclass
class Predictions:
    patient_idx: np.ndarray
    hour: np.ndarray
    y: np.ndarray
    p: np.ndarray
    p_mc_mean: np.ndarray | None = None
    p_mc_std: np.ndarray | None = None

    @classmethod
    def from_arrays(cls, arr: ClientArrays, p: np.ndarray, **mc: np.ndarray) -> Predictions:
        lens = np.diff(arr.offsets)
        return cls(
            patient_idx=np.repeat(arr.patient_idx, lens),
            hour=(np.arange(arr.offsets[-1]) - np.repeat(arr.offsets[:-1], lens)).astype(np.int32),
            y=arr.label.astype(np.int8),
            p=np.asarray(p, np.float32),
            p_mc_mean=mc.get("p_mc_mean"),
            p_mc_std=mc.get("p_mc_std"),
        )

    def save(self, path: Path) -> None:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: Path) -> Predictions:
        z = np.load(path)
        return cls(**{k: z[k] for k in z.files})

    def subset(self, mask: np.ndarray) -> Predictions:
        return Predictions(**{k: (v[mask] if v is not None else None) for k, v in self.__dict__.items()})

    def offsets(self) -> np.ndarray:
        """Patient boundaries (rows are contiguous per patient)."""
        change = np.flatnonzero(np.diff(self.patient_idx)) + 1
        return np.concatenate([[0], change, [len(self.patient_idx)]]).astype(np.int64)


def evaluate(
    preds: Predictions, clients: pd.Series, n_boot: int = 1000, seed: int = 0, key: str = "p"
) -> dict[str, Any]:
    """Global metrics (+ patient-bootstrap 95% CIs) and per-client metrics for probability array ``key``."""
    p = getattr(preds, key)
    out: dict[str, Any] = {"global": M.summary(preds.y, p)}
    if n_boot:
        out["global"]["ci95"] = M.bootstrap_ci(preds.y, p, preds.patient_idx, n_boot=n_boot, seed=seed)
    node_of_row = clients.to_numpy()[preds.patient_idx]
    out["per_client"] = {}
    for c in sorted(pd.unique(node_of_row[pd.notna(node_of_row)])):
        mask = node_of_row == c
        out["per_client"][c] = M.summary(preds.y[mask], p[mask])
    out["calibration"] = M.reliability_curve(preds.y.astype(float), p.astype(float))
    return out


def utility_at(preds: Predictions, threshold: float, key: str = "p") -> float:
    """Normalised PhysioNet utility of predictions binarised at ``threshold``."""
    return normalized_utility(preds.y, (getattr(preds, key) >= threshold).astype(np.int8), preds.offsets())
