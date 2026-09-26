"""Synthetic PhysioNet-2019-format files for tests (no real patient data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedguard.data.features import COLUMNS, DYN


def make_patient(rng: np.random.Generator, n_rows: int, septic: bool, unit: str) -> pd.DataFrame:
    """One synthetic stay: vitals often measured, labs rarely, unit flags per ``unit`` (MICU/SICU/UNK/BOTH)."""
    vals = rng.normal(0, 1, size=(n_rows, len(DYN))) * 10 + 80
    measure_p = np.array([0.9] * 8 + [0.1] * (len(DYN) - 8))
    vals[rng.random((n_rows, len(DYN))) > measure_p] = np.nan
    df = pd.DataFrame(vals, columns=DYN)
    df["Age"] = float(rng.integers(20, 90))
    df["Gender"] = float(rng.integers(0, 2))
    u1, u2 = {"MICU": (1.0, 0.0), "SICU": (0.0, 1.0), "UNK": (np.nan, np.nan), "BOTH": (1.0, 1.0)}[unit]
    df["Unit1"], df["Unit2"] = u1, u2
    df["HospAdmTime"] = -float(rng.uniform(0, 100)) if rng.random() > 0.1 else np.nan
    df["ICULOS"] = np.arange(1, n_rows + 1, dtype=float)
    label = np.zeros(n_rows, dtype=int)
    if septic:
        label[int(rng.integers(0, n_rows)) :] = 1
    df["SepsisLabel"] = label
    return df[COLUMNS]


def write_fixture(raw_dir: Path, n_per_hospital: int = 40, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Write ``training_setA``/``training_setB`` .psv files; returns {patient_id: DataFrame}."""
    rng = np.random.default_rng(seed)
    units = ["MICU", "SICU", "UNK"]
    out = {}
    for subset, base in (("training_setA", 0), ("training_setB", 100000)):
        d = Path(raw_dir) / subset
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_hospital):
            pid = f"p{base + i + 1:06d}"
            unit = "BOTH" if i == 0 else units[i % 3]
            df = make_patient(rng, int(rng.integers(5, 60)), septic=(i % 4 == 0), unit=unit)
            df.to_csv(d / f"{pid}.psv", sep="|", index=False, na_rep="NaN")
            out[pid] = df
    return out
