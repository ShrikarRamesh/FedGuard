"""eICU-CRD adapter (STUB): the realistic many-hospital federation (node = ``hospitalid``).

eICU-CRD (full, 208 hospitals) is credentialed. The eICU-CRD *demo* is open access but is still placed
manually. FedGuard never downloads either. Implement the TODOs once files are local.

Expected local layout (``$FEDGUARD_DATA_DIR/raw/eicu-crd/2.0/``)::

    patient.csv.gz  vitalPeriodic.csv.gz  vitalAperiodic.csv.gz  lab.csv.gz
    diagnosis.csv.gz  infusionDrug.csv.gz  medication.csv.gz  microLab.csv.gz  hospital.csv.gz

Cohort definition:
    * adults (age >= 18; "> 89" mapped to 90), unit stays >= 24 h, first unit stay per hospital stay;
    * offsets are minutes from unit admission, binned to hours; vitals from vitalPeriodic (5-min
      medians -> hourly last value), labs from lab.csv mapped to the 34 PhysioNet variables (TODO map);
    * Sepsis-3 onset: suspected infection (antibiotics + culture within the Sepsis-3 windows) with
      SOFA increase >= 2 (TODO: SOFA components from available tables; document gaps);
    * label shift identical to PhysioNet 2019 (1 for t >= t_sepsis - 6 h);
    * node = ``hospitalid``; the 8-node ablation keeps the 8 largest hospitals by eligible stays.
"""

from __future__ import annotations

from pathlib import Path

EXPECTED_FILES = [
    "patient.csv.gz", "vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz", "lab.csv.gz", "diagnosis.csv.gz",
    "infusionDrug.csv.gz", "medication.csv.gz", "microLab.csv.gz", "hospital.csv.gz",
]  # fmt: skip


def check_layout(root: Path) -> list[str]:
    """Return the expected files that are missing under ``root`` (read-only; never downloads)."""
    return [f for f in EXPECTED_FILES if not (Path(root) / f).exists()]


def process(root: Path, out_dir: Path, n_hospitals: int = 8) -> None:
    """Build processed arrays from local eICU-CRD files (not implemented yet)."""
    missing = check_layout(root)
    raise NotImplementedError(
        "eICU-CRD adapter is a documented stub. "
        + (f"Missing local files under {root}: {missing}. " if missing else "")
        + "Place eICU-CRD v2.0 files as described in this module's docstring (never downloaded "
        "automatically), then implement the variable mapping, hourly binning and Sepsis-3 TODOs in "
        "fedguard/data/adapters/eicu.py."
    )
