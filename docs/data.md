# Data: PhysioNet/CinC Challenge 2019

All numbers on this page come from `fedguard data process` and `fedguard data eda` on the full download (run 2026-09-26). The tables are in `results/eda/`. They replace any prevalence figures in the synopsis.

## Source

- **PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis from Clinical Data**, v1.0.0, open access, CC-BY 4.0. Reyna MA et al., *Crit Care Med* 2020;48(2):210–217.
- Downloaded with `fedguard data download` from the public S3 bucket `physionet-open` (`challenge-2019/1.0.0/training/`). Each file was checked against its S3 MD5 and the counts were verified: **training_setA 20,336 files, training_setB 20,000 files** (40,336 patients, 1,552,210 patient-hours).
- The organisers' hidden test set (hospital C) is not public and is not used.

## File format

One pipe-separated file per ICU stay, one row per hour: 34 dynamic variables (8 vitals, 26 labs), then `Age`, `Gender`, `Unit1` (MICU), `Unit2` (SICU), `HospAdmTime` (hours between hospital and ICU admission), `ICULOS` (ICU hour counter), and `SepsisLabel`. Values are NaN when not measured that hour.

## Label semantics: do not shift again

The organisers define t_sepsis by Sepsis-3 and **already shifted the label 6 h earlier**. For septic patients `SepsisLabel[t] = 1` for t ≥ t_sepsis − 6, and 0 before. Predicting `SepsisLabel[t]` from rows ≤ t is therefore 6-hour-ahead prediction, and FedGuard never shifts it again.

- **Onset hour** (used by alert metrics and the bedside demo) = first positive row index + 6.
- **426 patients are positive from their first row.** Their true onset is at most 6 h after ICU admission and the exact hour is unknown. They are flagged `onset_ambiguous` in `patients.csv`; alert lead-time statistics report results with and without them.
- Checks on the full data: labels are monotone (never 1 → 0) for all 2,932 septic patients, and `ICULOS` increases by exactly 1 per row for every patient, so the row index is the hour.

## Nodes (hospital system × ICU unit)

Hospital A = `training_setA`, B = `training_setB`. Unit = MICU if `Unit1 == 1` in any row, SICU if `Unit2 == 1`, otherwise UNK. No patient has both flags set (0 conflicts).

| Stratum | Patients | Septic patients | Septic patient rate | Patient-hours | Positive-hour rate | Median LOS (h) |
|---|---:|---:|---:|---:|---:|---:|
| A_MICU | 5,344 | 576 | 10.78% | 204,894 | 2.67% | 39 |
| A_SICU | 5,470 | 222 | 4.06% | 199,156 | 1.08% | 37 |
| A_UNK | 9,522 | 992 | 10.42% | 386,165 | 2.46% | 39 |
| B_MICU | 6,923 | 390 | 5.63% | 262,007 | 1.39% | 38 |
| B_SICU | 6,982 | 428 | 6.13% | 274,193 | 1.48% | 39 |
| B_UNK | 6,095 | 324 | 5.32% | 225,795 | 1.36% | 38 |
| **All** | **40,336** | **2,932** | **7.27%** | **1,552,210** | **1.80%** | 38 |

**Main 4-node federation** (`partition: unit`, `unk_policy: exclude`): 24,719 patients, 1,616 septic (6.54%), 940,250 patient-hours, **positive-hour prevalence 1.63%**. That is the reference line for AUPRC.

**UNK patients are 38.7% of the data** (15,617 patients). Excluding them is the spec's choice for the main experiments (a unit cannot be invented), but it discards a large, septic-rich part of hospital A. See D13 in `docs/decisions.md`. Counts for all three policies are in `results/eda/summary_{unit_exclude,unit_separate,hospital_merge_into_hospital}.csv`.

### Non-IID structure (why federation is hard here)

- **Label skew:** the septic-patient rate ranges from 4.1% (A_SICU) to 10.8% (A_MICU).
- **Feature/measurement skew:** measurement practice differs sharply by site (`results/eda/missingness.png`). Missing fraction of patient-hours:

  | Variable | A_MICU | A_SICU | B_MICU | B_SICU |
  |---|---:|---:|---:|---:|
  | DBP | 73.7% | 21.4% | 12.5% | 10.2% |
  | Temp | 73.6% | 49.7% | 73.6% | 50.9% |
  | HCO3 | 92.4% | 91.9% | 99.9% | 99.7% |
  | BaseExcess | 94.6% | 80.4% | 99.9% | 99.6% |

  The measurement masks alone largely identify the site.
- **Stay length** is similar across nodes (median 37–39 h, 90th percentile 52–56 h).

## Splits

Patient-level 70/15/15 splits are stratified on "ever septic" within each hospital × unit stratum (including UNK). Each stratum uses an RNG derived from `(split_seed = 42, "split", stratum)`, so splits don't depend on file order. Partitions (unit, hospital, Dirichlet) only regroup already-split patients. The global test set is the union of client test sets, which is identical across node-count ablations with the same UNK policy (D4). `data/processed/splits.csv` records `patient_id, node, split` for every patient.

## Features and windows

- **Values (34):** causal forward-fill within the patient. A value never measured so far is NaN, which becomes 0 after standardisation (= the training mean). Standardisation uses the mean/std of *measured* values of the **training patients** of the relevant client (per-node for FL and local runs, pooled for centralized).
- **Masks (34):** 1 if measured that hour.
- **Deltas (34):** log1p(min(hours since last measurement, 48)) / log1p(48); 1.0 if never measured.
- **Static (fixed, data-independent scaling):** Age (−60)/20, Gender, log1p(ICULOS)/5, HospAdmTime as log1p(clip(−h, 0, 720))/log1p(720) plus an observed flag (HospAdmTime is missing for some stays).
- **Windows:** one sample per patient-hour with lookback L = 24. The window is left-padded with zeros plus a padding mask, never crosses into another patient, and never reads rows after t. Windows are gathered lazily from contiguous per-client arrays.

Processed layout: see the docstring of `src/fedguard/data/physionet2019.py`.

## Reproduce

```bash
fedguard data download     # ~40k files, resumable, MD5-verified
fedguard data process      # ~22 s with 15 parser processes (16-thread laptop CPU)
fedguard data eda          # results/eda/*.csv, *.png
```
