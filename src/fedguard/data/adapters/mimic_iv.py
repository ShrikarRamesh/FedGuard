"""MIMIC-IV adapter (STUB): external validation once PhysioNet credentialing is approved.

MIMIC-IV is credentialed data. FedGuard never downloads it. Place the files locally yourself, after
completing the CITI course and signing the data use agreement, then implement the TODOs below.

Expected local layout (``$FEDGUARD_DATA_DIR/raw/mimic-iv/3.1/``)::

    hosp/patients.csv.gz  hosp/admissions.csv.gz  hosp/labevents.csv.gz  hosp/d_labitems.csv.gz
    icu/icustays.csv.gz   icu/chartevents.csv.gz  icu/d_items.csv.gz
    derived/sepsis3.csv   # from the MIT-LCP `mimic-code` concepts (sepsis3 view), exported to CSV

Cohort definition (to match PhysioNet 2019 as closely as possible):
    * adults (anchor_age >= 18), first ICU stay per admission, ICU stay >= 24 h;
    * hourly bins from ICU intime; each of the 34 PhysioNet variables mapped from chartevents/labevents
      itemids (mapping table TODO, reviewed by a clinician on the team);
    * Sepsis-3 onset t_sepsis from `mimic-code` sepsis3 (suspected infection + SOFA >= 2);
    * label SepsisLabel[t] = 1 for t >= t_sepsis - 6 h (same 6-hour shift as the challenge), truncated
      at t_sepsis + 9 h as in the challenge;
    * node = ``first_careunit`` (MICU, SICU, CVICU, ...), which gives a single-hospital multi-unit federation.

Output: the same processed layout as ``physionet2019.write_processed`` so every downstream step works.
"""

from __future__ import annotations

from pathlib import Path

EXPECTED_FILES = [
    "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/labevents.csv.gz", "hosp/d_labitems.csv.gz",
    "icu/icustays.csv.gz", "icu/chartevents.csv.gz", "icu/d_items.csv.gz", "derived/sepsis3.csv",
]  # fmt: skip


def check_layout(root: Path) -> list[str]:
    """Return the expected files that are missing under ``root`` (read-only; never downloads)."""
    return [f for f in EXPECTED_FILES if not (Path(root) / f).exists()]


def process(root: Path, out_dir: Path) -> None:
    """Build processed arrays from local MIMIC-IV files (not implemented yet)."""
    missing = check_layout(root)
    raise NotImplementedError(
        "MIMIC-IV adapter is a documented stub. "
        + (f"Missing local files under {root}: {missing}. " if missing else "")
        + "Place credentialed MIMIC-IV v3.1 files as described in this module's docstring (never downloaded "
        "automatically), export the mimic-code sepsis3 concept, then implement the itemid mapping and hourly "
        "binning TODOs in fedguard/data/adapters/mimic_iv.py."
    )
