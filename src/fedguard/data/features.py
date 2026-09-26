"""Feature definitions and data-independent transforms for PhysioNet 2019.

Channel layout of a model input window ``x[L, C]`` (all float32):

    values  (34)  forward-filled, standardised with *training* stats of the client/pool; never-seen -> 0
    masks   (34)  1 if the variable was actually measured at that hour          [features.use_masks]
    deltas  (34)  log1p(min(hours since last measurement, cap)) / log1p(cap);   [features.use_deltas]
                  never measured so far -> 1.0
    static  (k)   Age, Gender, ICULOS, HospAdmTime (+ HospAdmTime_obs) with fixed, data-independent scaling

Only the ``values`` block depends on data statistics (see ``windows.NormStats``); everything else is a
fixed transform, so it cannot leak across splits or clients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 34 dynamic variables in file order (vitals then labs).
DYN: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]  # fmt: skip
VITALS: list[str] = DYN[:8]
DEMOGRAPHIC: list[str] = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
COLUMNS: list[str] = DYN + DEMOGRAPHIC + [LABEL]

# Raw static columns stored per row in processed data, in this order.
STATIC_RAW: list[str] = ["Age", "Gender", "HospAdmTime", "ICULOS"]


@dataclass(frozen=True)
class ChannelSpec:
    """Names of the input channels and the clinical variable each channel belongs to."""

    names: list[str]
    groups: list[str]  # variable name per channel (value/mask/delta of HR -> "HR")
    n_dyn: int

    @property
    def n_channels(self) -> int:
        return len(self.names)


def channel_spec(use_masks: bool, use_deltas: bool, static: list[str]) -> ChannelSpec:
    """Build the channel layout for a feature config."""
    names, groups = list(DYN), list(DYN)
    if use_masks:
        names += [f"{v}_mask" for v in DYN]
        groups += list(DYN)
    if use_deltas:
        names += [f"{v}_delta" for v in DYN]
        groups += list(DYN)
    for s in static:
        names.append(s)
        groups.append(s)
        if s == "HospAdmTime":
            names.append("HospAdmTime_obs")
            groups.append(s)
    return ChannelSpec(names=names, groups=groups, n_dyn=len(DYN))


def scale_deltas(hours_since: np.ndarray, cap: float) -> np.ndarray:
    """log1p(min(h, cap)) / log1p(cap); NaN (never measured so far) -> 1.0."""
    h = np.where(np.isnan(hours_since), cap, np.minimum(hours_since, cap))
    return (np.log1p(h) / np.log1p(cap)).astype(np.float32)


def scale_static(static_raw: np.ndarray, static: list[str], hospadm_clip: float) -> np.ndarray:
    """Fixed (data-independent) scaling of static columns ``STATIC_RAW`` -> selected static channels."""
    cols = {name: static_raw[:, i] for i, name in enumerate(STATIC_RAW)}
    out: list[np.ndarray] = []
    for s in static:
        v = cols[s]
        if s == "Age":
            out.append(np.nan_to_num((v - 60.0) / 20.0))
        elif s == "Gender":
            out.append(np.nan_to_num(v))
        elif s == "ICULOS":
            out.append(np.log1p(np.nan_to_num(np.maximum(v, 0.0))) / 5.0)
        elif s == "HospAdmTime":
            # hours from hospital to ICU admission (usually negative); positive values clip to 0
            obs = ~np.isnan(v)
            h = np.clip(-np.nan_to_num(v), 0.0, hospadm_clip)
            out.append(np.where(obs, np.log1p(h) / np.log1p(hospadm_clip), 0.0))
            out.append(obs.astype(np.float64))
        else:  # pragma: no cover - guarded by pydantic Literal
            raise ValueError(f"unknown static feature {s}")
    return np.stack(out, axis=1).astype(np.float32) if out else np.zeros((len(static_raw), 0), np.float32)


def causal_ffill_and_age(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For ONE patient's hourly rows: forward-fill each column (past only) and hours since last measurement.

    Returns (ffilled, hours_since) with NaN where the variable has not been measured yet.
    Row index is the hour; nothing from row > t influences row t.
    """
    n, c = values.shape
    measured = ~np.isnan(values)
    rows = np.arange(n)[:, None]
    last = np.where(measured, rows, -1)
    last = np.maximum.accumulate(last, axis=0)  # index of most recent measurement at or before t
    seen = last >= 0
    idx = np.where(seen, last, 0)
    ffilled = np.where(seen, values[idx, np.arange(c)[None, :]], np.nan).astype(np.float32)
    hours = np.where(seen, rows - last, np.nan).astype(np.float32)
    return ffilled, hours
