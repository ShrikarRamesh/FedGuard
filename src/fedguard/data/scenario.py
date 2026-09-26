"""A federated scenario: processed data + partition + per-client / pooled normalisation.

Normalisation rules (CLAUDE.md rule 2, D15):
* ``site`` mode: every patient is standardised with the training stats of *its own* client. Used for
  local-only and federated models, and for evaluating them (each hospital standardises its own data).
* ``pooled`` mode: every patient is standardised with stats pooled over all clients' training patients.
  Used only for the centralized upper bound.
* ``public`` mode: fixed clinical reference values, no data statistics at all (DP runs, D3).
Stats are always fit on training patients only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fedguard.config import DataConfig
from fedguard.data.physionet2019 import assign_clients
from fedguard.data.windows import (
    ClientArrays,
    NormStats,
    ProcessedData,
    WindowDataset,
    build_client_arrays,
    client_patient_indices,
)

NormMode = Literal["site", "pooled", "public"]


@dataclass
class Scenario:
    data: ProcessedData
    cfg: DataConfig
    clients: pd.Series  # client name per patient (NaN = excluded)
    _stats: dict[str, NormStats] = field(default_factory=dict)

    @classmethod
    def load(cls, processed_dir: Path, cfg: DataConfig) -> Scenario:
        data = ProcessedData.load(processed_dir)
        return cls(data=data, cfg=cfg, clients=assign_clients(data.patients, cfg))

    @property
    def client_names(self) -> list[str]:
        return sorted(self.clients.dropna().unique())

    def patients(self, client: str | None, split: str) -> np.ndarray:
        return client_patient_indices(self.data.patients, self.clients, client, split)

    def stats(self, client: str | None) -> NormStats:
        """Train-only stats of one client (``None`` = pooled over all clients). Cached."""
        key = client or "__pooled__"
        if key not in self._stats:
            self._stats[key] = NormStats.fit(self.data, self.patients(client, "train"))
        return self._stats[key]

    def arrays(self, client: str | None, split: str, norm: NormMode) -> ClientArrays:
        """Rows of ``client`` (None = all clients) in ``split``, standardised per ``norm``."""
        if norm in ("pooled", "public"):
            st = self.stats(None) if norm == "pooled" else NormStats.public()
            return build_client_arrays(self.data, self.patients(client, split), st, self.cfg)
        names = [client] if client is not None else self.client_names
        parts = [
            build_client_arrays(self.data, self.patients(c, split), self.stats(c), self.cfg)
            for c in names
            if len(self.patients(c, split))
        ]
        return concat_arrays(parts)

    def dataset(self, client: str | None, split: str, norm: NormMode) -> WindowDataset:
        return WindowDataset(self.arrays(client, split, norm), self.cfg.lookback)

    def n_train_patients(self, client: str) -> int:
        return len(self.patients(client, "train"))


def concat_arrays(parts: list[ClientArrays]) -> ClientArrays:
    """Concatenate ClientArrays (patients keep their own normalisation)."""
    if not parts:
        raise ValueError("no patients to concatenate")
    if len(parts) == 1:
        return parts[0]
    offs = [parts[0].offsets]
    base = parts[0].offsets[-1]
    for p in parts[1:]:
        offs.append(p.offsets[1:] + base)
        base += p.offsets[-1]
    return ClientArrays(
        features=np.concatenate([p.features for p in parts]),
        label=np.concatenate([p.label for p in parts]),
        offsets=np.concatenate(offs),
        patient_idx=np.concatenate([p.patient_idx for p in parts]),
        spec=parts[0].spec,
    )
