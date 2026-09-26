"""PhysioNet/CinC 2019: parse raw .psv files, assign hospital x unit strata, make patient-level splits,
and define federated partitions. Logic adapted from the team's ``prepare_data.py`` starter (see D4, D5).

Label semantics (documented in docs/data.md): the organisers already shifted ``SepsisLabel`` 6 h early.
For septic patients it is 1 for t >= t_sepsis - 6, so predicting ``SepsisLabel[t]`` from rows <= t is
6-hour-ahead prediction. We never shift it again. Onset hour = first positive row + 6.

Processed layout (``<data_dir>/<processed_subdir>/``):
    values_raw.npy   float32 [R, 34]  as measured, NaN if not measured that hour
    values_ffill.npy float32 [R, 34]  causal forward fill within patient, NaN before first measurement
    hours_since.npy  float32 [R, 34]  hours since last measurement, NaN before first measurement
    static_raw.npy   float32 [R, 4]   Age, Gender, HospAdmTime, ICULOS (unscaled)
    label.npy        int8    [R]      SepsisLabel, byte-identical to the raw files
    offsets.npy      int64   [P+1]    patient p owns rows offsets[p]:offsets[p+1]
    patients.csv     one row per patient (id, hospital, unit, stratum, split, n_rows, ever_septic, onset ...)
    splits.csv       patient_id, node, split  (node = hospital_unit stratum)
    manifest.json    counts, config, data-quality checks
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fedguard.config import DataConfig
from fedguard.data.download import SUBSETS, select_subset
from fedguard.data.features import COLUMNS, DYN, LABEL, STATIC_RAW, causal_ffill_and_age
from fedguard.utils.io import write_json
from fedguard.utils.seed import rng_for

HOSPITAL_OF_SUBSET = {"training_setA": "A", "training_setB": "B"}
SPLIT_NAMES = ("train", "val", "test")
ONSET_SHIFT_HOURS = 6


class DataFormatError(ValueError):
    """A raw file does not match the PhysioNet 2019 format."""


@dataclass
class ParsedPatient:
    patient_id: str
    hospital: str
    unit: str  # MICU | SICU | UNK
    unit_conflict: bool  # unit flags disagree across rows or both set
    values_raw: np.ndarray
    values_ffill: np.ndarray
    hours_since: np.ndarray
    static_raw: np.ndarray
    label: np.ndarray
    iculos_consecutive: bool


def unit_of(unit1: np.ndarray, unit2: np.ndarray) -> tuple[str, bool]:
    """Unit from the Unit1 (MICU) / Unit2 (SICU) flags. Uses any row with a non-NaN flag; if the flags
    ever indicate both units the patient is ``UNK`` with ``conflict=True`` (counted in the manifest)."""
    micu = bool(np.any(unit1 == 1))
    sicu = bool(np.any(unit2 == 1))
    if micu and sicu:
        return "UNK", True
    return ("MICU" if micu else "SICU" if sicu else "UNK"), False


def parse_file(path: Path, hospital: str) -> ParsedPatient:
    """Parse one patient file. Raises DataFormatError on unexpected columns or empty files."""
    df = pd.read_csv(path, sep="|")
    if list(df.columns) != COLUMNS:
        raise DataFormatError(f"{path}: unexpected columns {list(df.columns)[:5]}...")
    if len(df) == 0:
        raise DataFormatError(f"{path}: no rows")
    values = df[DYN].to_numpy(np.float32)
    ffill, hours = causal_ffill_and_age(values)
    unit, conflict = unit_of(df["Unit1"].to_numpy(), df["Unit2"].to_numpy())
    iculos = df["ICULOS"].to_numpy()
    label = df[LABEL].to_numpy()
    if np.isnan(label).any() or not np.isin(label, (0, 1)).all():
        raise DataFormatError(f"{path}: SepsisLabel must be 0/1")
    return ParsedPatient(
        patient_id=Path(path).stem,
        hospital=hospital,
        unit=unit,
        unit_conflict=conflict,
        values_raw=values,
        values_ffill=ffill,
        hours_since=hours,
        static_raw=df[STATIC_RAW].to_numpy(np.float32),
        label=label.astype(np.int8),
        iculos_consecutive=bool(np.all(np.diff(iculos) == 1)),
    )


def _parse_chunk(args: tuple[list[str], str]) -> list[ParsedPatient]:
    paths, hospital = args
    return [parse_file(Path(p), hospital) for p in paths]


def list_raw_files(raw_dir: Path, subset_per_hospital: int | None, seed: int) -> list[tuple[Path, str]]:
    """(path, hospital) for every raw file to process, sorted; optionally a deterministic subset."""
    out: list[tuple[Path, str]] = []
    for subset in SUBSETS:
        d = Path(raw_dir) / subset
        names = [p.name for p in d.glob("*.psv")] if d.exists() else []
        for n in select_subset(names, subset_per_hospital, seed):
            out.append((d / n, HOSPITAL_OF_SUBSET[subset]))
    if not out:
        raise FileNotFoundError(f"no .psv files under {raw_dir}; run `fedguard data download` first")
    return out


def parse_all(
    files: list[tuple[Path, str]],
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[ParsedPatient]:
    """Parse files in parallel (process pool; ``workers=1`` runs inline). Output order == input order."""
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    chunk = 250
    jobs: list[tuple[list[str], str]] = []
    for hosp in dict.fromkeys(h for _, h in files):  # preserves input order of hospitals
        paths = [str(p) for p, h in files if h == hosp]
        jobs += [(paths[i : i + chunk], hosp) for i in range(0, len(paths), chunk)]
    if [p for j in jobs for p in j[0]] != [str(p) for p, _ in files]:
        raise ValueError("files must be grouped by hospital (as list_raw_files returns them)")
    out: list[ParsedPatient] = []
    if workers == 1:
        for i, j in enumerate(jobs, 1):
            out.extend(_parse_chunk(j))
            if progress:
                progress(i, len(jobs))
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_parse_chunk, jobs), 1):
            out.extend(res)
            if progress:
                progress(i, len(jobs))
    return out


def stratified_split(
    ever_septic: np.ndarray, fracs: tuple[float, float, float], rng: np.random.Generator
) -> np.ndarray:
    """0/1/2 = train/val/test, stratified on ever-septic (same rounding as the starter script)."""
    split = np.zeros(len(ever_septic), dtype=np.int8)
    for cls in (0, 1):
        idx = np.where(ever_septic == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        n_tr, n_va = int(fracs[0] * len(idx)), int(fracs[1] * len(idx))
        split[idx[n_tr : n_tr + n_va]] = 1
        split[idx[n_tr + n_va :]] = 2
    return split


def build_patient_table(
    parsed: list[ParsedPatient], fracs: tuple[float, float, float], seed: int
) -> pd.DataFrame:
    """Per-patient table with stratum, split and label summaries. Splits are made per hospital x unit
    stratum with an RNG derived from (seed, stratum), so they do not depend on processing order (D4)."""
    rows = []
    start = 0
    for p in parsed:
        n = len(p.label)
        pos = np.flatnonzero(p.label == 1)
        first = int(pos[0]) if len(pos) else -1
        rows.append(
            {
                "patient_id": p.patient_id,
                "hospital": p.hospital,
                "unit": p.unit,
                "stratum": f"{p.hospital}_{p.unit}",
                "row_start": start,
                "n_rows": n,
                "ever_septic": int(len(pos) > 0),
                "first_pos_row": first,
                "onset_hour": first + ONSET_SHIFT_HOURS if first >= 0 else -1,
                "onset_ambiguous": int(
                    first == 0
                ),  # positive from the first row: onset <= 6 h, exact hour unknown
                "label_monotone": int(len(pos) == 0 or bool(np.all(p.label[first:] == 1))),
                "unit_conflict": int(p.unit_conflict),
                "iculos_consecutive": int(p.iculos_consecutive),
            }
        )
        start += n
    df = pd.DataFrame(rows)
    df["split"] = -1
    # sort within stratum by patient id before permuting, so the result is order-independent
    for stratum, g in df.groupby("stratum", sort=True):
        g = g.sort_values("patient_id")
        s = stratified_split(g["ever_septic"].to_numpy(), fracs, rng_for(seed, "split", stratum))
        df.loc[g.index, "split"] = s
    assert (df["split"] >= 0).all()
    df["split_name"] = df["split"].map(dict(enumerate(SPLIT_NAMES)))
    return df


def write_processed(
    parsed: list[ParsedPatient], patients: pd.DataFrame, out_dir: Path, manifest: dict
) -> None:
    """Write arrays + tables to ``out_dir`` (overwrites)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.concatenate([[0], np.cumsum([len(p.label) for p in parsed])]).astype(np.int64)
    assert (offsets[:-1] == patients["row_start"].to_numpy()).all()
    np.save(out_dir / "values_raw.npy", np.concatenate([p.values_raw for p in parsed]))
    np.save(out_dir / "values_ffill.npy", np.concatenate([p.values_ffill for p in parsed]))
    np.save(out_dir / "hours_since.npy", np.concatenate([p.hours_since for p in parsed]))
    np.save(out_dir / "static_raw.npy", np.concatenate([p.static_raw for p in parsed]))
    np.save(out_dir / "label.npy", np.concatenate([p.label for p in parsed]))
    np.save(out_dir / "offsets.npy", offsets)
    patients.to_csv(out_dir / "patients.csv", index=False)
    splits = pd.DataFrame(
        {"patient_id": patients["patient_id"], "node": patients["stratum"], "split": patients["split_name"]}
    )
    splits.to_csv(out_dir / "splits.csv", index=False)
    write_json(out_dir / "manifest.json", manifest)


def process(cfg: DataConfig, raw_dir: Path, out_dir: Path, workers: int | None = None, progress=None) -> dict:
    """Full processing step. Returns the manifest (counts + data-quality checks)."""
    files = list_raw_files(raw_dir, cfg.subset_per_hospital, cfg.split_seed)
    parsed = parse_all(files, workers=workers, progress=progress)
    patients = build_patient_table(parsed, tuple(cfg.split_fracs), cfg.split_seed)
    manifest = {
        "n_patients": int(len(patients)),
        "n_rows": int(patients["n_rows"].sum()),
        "patients_per_stratum": patients["stratum"].value_counts().sort_index().to_dict(),
        "split_seed": cfg.split_seed,
        "split_fracs": list(cfg.split_fracs),
        "subset_per_hospital": cfg.subset_per_hospital,
        "checks": {
            "unit_conflicts": int(patients["unit_conflict"].sum()),
            "non_monotone_labels": int((patients["label_monotone"] == 0).sum()),
            "onset_ambiguous_positive_from_first_row": int(patients["onset_ambiguous"].sum()),
            "non_consecutive_iculos": int((patients["iculos_consecutive"] == 0).sum()),
        },
    }
    write_processed(parsed, patients, out_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------------------------
# Federated partitions


def assign_clients(patients: pd.DataFrame, cfg: DataConfig, seed: int | None = None) -> pd.Series:
    """Client name per patient (``NaN`` = excluded) for ``cfg.partition`` and ``cfg.unk_policy``.

    * ``unit``: node = hospital_unit. UNK patients: exclude | separate (A_UNK/B_UNK nodes).
      ``merge_into_hospital`` is not defined for this partition (a unit cannot be invented) -> error.
    * ``hospital``: node = A | B. UNK: exclude | separate (A_UNK/B_UNK) | merge_into_hospital (keep in A/B).
    * ``dirichlet``: k clients with label skew, class proportions ~ Dir(alpha); UNK: exclude or keep.
    Splits are untouched: a client's train/val/test are its patients' pre-assigned splits.
    """
    unk = patients["unit"] == "UNK"
    if cfg.partition == "unit":
        if cfg.unk_policy == "merge_into_hospital":
            raise ValueError(
                "unk_policy=merge_into_hospital is only defined for partition=hospital|dirichlet"
            )
        client = patients["stratum"].astype(object)
        if cfg.unk_policy == "exclude":
            client = client.where(~unk)
        return client
    if cfg.partition == "hospital":
        client = patients["hospital"].astype(object)
        if cfg.unk_policy == "exclude":
            client = client.where(~unk)
        elif cfg.unk_policy == "separate":
            client = client.where(~unk, patients["hospital"] + "_UNK")
        return client
    # dirichlet
    keep = ~unk if cfg.unk_policy == "exclude" else pd.Series(True, index=patients.index)
    client = pd.Series(np.nan, index=patients.index, dtype=object)
    rng = rng_for(cfg.split_seed if seed is None else seed, "dirichlet", cfg.dirichlet_k, cfg.dirichlet_alpha)
    names = [f"D{i}" for i in range(cfg.dirichlet_k)]
    for cls in (0, 1):
        idx = patients.index[keep & (patients["ever_septic"] == cls)]
        idx = idx[np.argsort(patients.loc[idx, "patient_id"].to_numpy())]
        idx = idx[rng.permutation(len(idx))]
        props = rng.dirichlet(np.full(cfg.dirichlet_k, cfg.dirichlet_alpha))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for name, part in zip(names, np.split(idx, cuts), strict=True):
            client.loc[part] = name
    return client
