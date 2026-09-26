"""Loading processed data, train-only normalisation, and lazy causal windows.

A window for (patient p, hour t) is rows ``max(start_p, t-L+1) .. t`` of that patient only, left-padded
with zeros to length L, with ``pad_mask`` True on real rows. Nothing after hour t is ever read.
Windows are gathered lazily from per-client contiguous arrays, so RAM holds one row per patient-hour
(not L copies).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fedguard.config import DataConfig
from fedguard.data.features import DYN, ChannelSpec, channel_spec, scale_deltas, scale_static

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


@dataclass
class ProcessedData:
    """Processed arrays (memory-mapped) + patient table."""

    dir: Path
    patients: pd.DataFrame
    offsets: np.ndarray
    values_raw: np.ndarray
    values_ffill: np.ndarray
    hours_since: np.ndarray
    static_raw: np.ndarray
    label: np.ndarray

    @classmethod
    def load(cls, d: Path, mmap: bool = True) -> ProcessedData:
        d = Path(d)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"no processed data in {d}; run `fedguard data process` first")
        mode = "r" if mmap else None
        return cls(
            dir=d,
            patients=pd.read_csv(d / "patients.csv", dtype={"patient_id": str}),
            offsets=np.load(d / "offsets.npy"),
            values_raw=np.load(d / "values_raw.npy", mmap_mode=mode),
            values_ffill=np.load(d / "values_ffill.npy", mmap_mode=mode),
            hours_since=np.load(d / "hours_since.npy", mmap_mode=mode),
            static_raw=np.load(d / "static_raw.npy", mmap_mode=mode),
            label=np.load(d / "label.npy", mmap_mode=mode),
        )

    def rows_of(self, patient_idx: np.ndarray) -> np.ndarray:
        """Concatenated row indices of the given patients (in the given order)."""
        starts, ends = self.offsets[patient_idx], self.offsets[patient_idx + 1]
        if len(starts) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(s, e) for s, e in zip(starts, ends, strict=True)])


@dataclass
class NormStats:
    """Per-variable mean/std of the dynamic values, computed from *measured* values of training patients."""

    mean: np.ndarray
    std: np.ndarray
    n_patients: int
    n_measurements: np.ndarray

    @classmethod
    def fit(cls, data: ProcessedData, train_patient_idx: np.ndarray) -> NormStats:
        """Fit on the given (training) patients only. Each actual measurement counts once (D5)."""
        if len(train_patient_idx) == 0:
            raise ValueError("cannot fit normalisation stats on zero patients")
        v = np.asarray(data.values_raw[data.rows_of(np.sort(train_patient_idx))], dtype=np.float64)
        n = (~np.isnan(v)).sum(0)
        with warnings.catch_warnings():  # all-NaN columns (variable never measured) are handled below
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(v, axis=0)
            std = np.nanstd(v, axis=0)
        mean[n == 0] = 0.0
        std[(n < 2) | ~np.isfinite(std) | (std < 1e-6)] = 1.0
        return cls(
            mean.astype(np.float32), std.astype(np.float32), int(len(train_patient_idx)), n.astype(np.int64)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": DYN,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_patients": self.n_patients,
            "n_measurements": self.n_measurements.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormStats:
        return cls(
            np.asarray(d["mean"], np.float32),
            np.asarray(d["std"], np.float32),
            int(d["n_patients"]),
            np.asarray(d["n_measurements"], np.int64),
        )


@dataclass
class ClientArrays:
    """Model-ready rows for a set of patients (one client or a pool), standardised with given stats."""

    features: np.ndarray  # float32 [R, C]
    label: np.ndarray  # int8 [R]
    offsets: np.ndarray  # int64 [P+1] into features
    patient_idx: np.ndarray  # indices into ProcessedData.patients
    spec: ChannelSpec

    @property
    def n_patients(self) -> int:
        return len(self.patient_idx)


def build_client_arrays(
    data: ProcessedData, patient_idx: np.ndarray, stats: NormStats, cfg: DataConfig
) -> ClientArrays:
    """Assemble features for ``patient_idx`` (value block standardised with ``stats``, NaN -> 0)."""
    patient_idx = np.asarray(patient_idx, dtype=np.int64)
    rows = data.rows_of(patient_idx)
    f = cfg.features
    spec = channel_spec(f.use_masks, f.use_deltas, list(f.static))
    blocks = [np.nan_to_num((np.asarray(data.values_ffill[rows]) - stats.mean) / stats.std, nan=0.0)]
    if f.use_masks:
        blocks.append((~np.isnan(np.asarray(data.values_raw[rows]))).astype(np.float32))
    if f.use_deltas:
        blocks.append(scale_deltas(np.asarray(data.hours_since[rows]), f.delta_cap_hours))
    blocks.append(scale_static(np.asarray(data.static_raw[rows]), list(f.static), f.hospadm_clip_hours))
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    assert features.shape[1] == spec.n_channels, (features.shape, spec.n_channels)
    lens = data.offsets[patient_idx + 1] - data.offsets[patient_idx]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return ClientArrays(features, np.asarray(data.label[rows]).astype(np.int8), offsets, patient_idx, spec)


class WindowDataset:
    """torch-style Dataset: one sample per (patient, hour). ``__getitem__`` -> (x[L, C], pad_mask[L], y).

    ``__getitems__`` gathers a whole batch with vectorised indexing (used by torch DataLoader >= 2.1).
    """

    def __init__(self, arrays: ClientArrays, lookback: int):
        self.a = arrays
        self.L = int(lookback)
        lens = np.diff(arrays.offsets)
        self.row = np.arange(arrays.offsets[-1], dtype=np.int64)  # sample i ends at feature row i
        self.start = np.repeat(arrays.offsets[:-1], lens)  # first row of the sample's patient
        self.patient_pos = np.repeat(np.arange(len(lens)), lens)  # position within arrays.patient_idx
        self.hour = self.row - self.start

    def __len__(self) -> int:
        return len(self.row)

    def labels(self) -> np.ndarray:
        return self.a.label

    def gather(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch of windows: x [B, L, C] float32, pad_mask [B, L] bool (True = real row), y [B] float32."""
        idx = np.asarray(idx, dtype=np.int64)
        t = self.row[idx]
        rows = t[:, None] - np.arange(self.L - 1, -1, -1)[None, :]  # [B, L], last column = t
        valid = rows >= self.start[idx][:, None]  # never crosses into the previous patient
        x = self.a.features[np.where(valid, rows, 0)]
        x[~valid] = 0.0
        return x.astype(np.float32, copy=False), valid, self.a.label[t].astype(np.float32)

    def __getitem__(self, i: int):
        x, m, y = self.gather(np.array([i]))
        return x[0], m[0], y[0]

    def __getitems__(self, indices: list[int]):
        return self.gather(np.asarray(indices))


def client_patient_indices(
    patients: pd.DataFrame, clients: pd.Series, client: str | None, split: str
) -> np.ndarray:
    """Patient indices of ``client`` (None = all included clients, i.e. the pool) in ``split``."""
    included = clients.notna()
    sel = included & (patients["split"] == SPLIT_CODE[split])
    if client is not None:
        sel &= clients == client
    return np.flatnonzero(sel.to_numpy())
