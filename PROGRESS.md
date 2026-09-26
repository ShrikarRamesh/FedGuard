# FedGuard progress

Only real, reproducible numbers appear here. Anything not yet run says **not run yet**.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Scaffold: package, configs, CLI skeleton, CLAUDE.md, PROGRESS.md, `make test` | **Done** (2026-09-26) |
| M1 | Data: download, verify, nodes, splits, windows, EDA | **Done** (2026-09-26) |
| M2 | Models: PatchTST (Opacus-safe), baselines, MC Dropout | next |
| M3 | Centralized + local-only, 3 seeds | not started |
| M4 | FL engine (sync/async, FedAvg/FedProx) + Flower app + cross-check + deployment scripts | not started |
| M5 | DP: Opacus per client, uniform/adaptive budgets, accounting, ε sweep, `docs/privacy.md` | not started; patient-level DP approved (D2); D3 open |
| M6 | Alerts, calibration, explanations | not started |
| M7 | Gradient-inversion attack | not started |
| M8 | Full experiments + ablations + `fedguard report` | not started |
| M9 | Export + Streamlit app | not started |
| M10 | Polish: README, docstrings, `make smoke` green | not started |

## Environment (checked 2026-09-26)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.11.0 venv at `C:\Users\Shrikar\.venvs\fedguard` (outside OneDrive; system default is 3.14, not used) |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB, driver 592.82, CUDA 13.1; `torch.cuda.is_available() == True` |
| Disk | about 499 GB free on C: |
| Tools | git 2.45, uv 0.11.24; **no** `make`, **no** `aws` CLI (D11) |

### Library versions (installed and verified against the actual API)

| Library | Version | API checks done |
|---|---|---|
| torch | 2.14.0+cu130 | CUDA OK |
| opacus | 1.6.0 | `make_private_with_epsilon(module, optimizer, data_loader, target_epsilon, target_delta, epochs, max_grad_norm, poisson_sampling=True, ...)`; `BatchMemoryManager`; `ModuleValidator`; per-sample grads for Linear/LayerNorm/Embedding verified. **Found the `arange`-indexed Embedding bug (D1).** |
| flwr | 1.38.0 | Message API: `flwr.serverapp.ServerApp`, `flwr.clientapp.ClientApp`, `flwr.app.{ArrayRecord, MetricRecord, ConfigRecord, Message, Context}`, `flwr.serverapp.strategy.{FedAvg, FedProx}` with `.start(grid, initial_arrays, num_rounds, timeout, evaluate_fn=...)` |
| captum | 0.9.0 | `IntegratedGradients.attribute(inputs, baselines, target, additional_forward_args, n_steps, internal_batch_size)` |
| streamlit | 1.64.0 | `st.fragment(run_every=...)`, `streamlit.testing.v1.AppTest.from_file` |
| lightgbm 4.7.0 · scikit-learn 1.9.1 · numpy 2.4.6 · pandas 3.0.6 · pyarrow 25.0.1 · typer 0.20.1 · omegaconf 2.3.1 · pydantic 2.13.5 · matplotlib 3.11.2 · plotly 7.1.0 | | |

---

## M0: Scaffold (done 2026-09-26)

**Built**
- `pyproject.toml` with pinned deps (`pip install -e .[dev]`), ruff + black config.
- `src/fedguard/` package skeleton matching the layout in CLAUDE.md. Implemented so far:
  - `config.py`: YAML composition (`defaults:` groups, same-group inheritance, `_fast:` blocks for `--fast`, dotlist overrides) and a pydantic `DataConfig`.
  - `utils/seed.py`: `seed_everything`, and order-independent `derived_seed`/`rng_for` (SHA-256 based).
  - `utils/io.py`: paths (`FEDGUARD_DATA_DIR`, `FEDGUARD_RUNS_DIR`) and atomic JSON/JSONL writers.
  - `utils/logging.py`: console + file logger, and an offline CSV `MetricsLogger` with an optional W&B mirror.
  - `utils/runs.py`: run dirs `runs/<exp>/<timestamp>_<seed>/` with `config.yaml`, `meta.json` (git commit + dirty flag, library versions, argv, config hash), a DONE marker and `find_completed` for resumable sweeps.
  - `cli.py`: Typer CLI with every planned command (`data download|process|eda`, `train centralized|local`, `fl run`, `alerts`, `attack`, `report`, `export`), each taking `--fast`, `--seed`, `-o`. `info` and `show-config` work; the others exit with code 2 and name their milestone.
- Configs: `data.yaml`, `model/patchtst.yaml`, `train/{centralized,local}.yaml`, `fl/{fedavg,fedprox,fedguard_sync,fedguard_async}.yaml`, `privacy/{none,uniform,adaptive}.yaml`, `experiments/centralized.yaml`.
- `Makefile` plus `scripts/make.ps1` (same targets), `.gitignore`, `README.md` stub, `docs/decisions.md` (D1–D12), and `app/static/fedguard_demo.html` (copy of `FedGuard live demo.html`).

**Commands and results**
```
.\scripts\make.ps1 lint    ->  ruff: All checks passed! / black: 20 files would be left unchanged
.\scripts\make.ps1 test    ->  8 passed in 3.5 s
.\scripts\make.ps1 smoke   ->  fedguard info prints paths + versions (CUDA True)
```

**Issues found and fixed**
- Black's multiprocessing hangs on this Windows machine, so `workers = 1` is set in `[tool.black]`.
- PowerShell 5.1 turns native stderr into errors when redirected with `2>&1`, so don't redirect in `make.ps1` calls.

**Real numbers:** none yet (no data).

---

## M1: Data (done 2026-09-26)

**Built** (logic moved from `prepare_data.py`, which is removed from the root and kept in git history)
- `data/download.py`: pure-Python S3 download (32 threads, retries, resumable, **per-file MD5** from S3 ETags), physionet.org fallback, and `verify()` with exact counts.
- `data/physionet2019.py`: parallel parsing (process pool), causal ffill plus hours-since-measured per patient, unit assignment, and a patient table with onset hour and data-quality flags. Splits are per stratum with order-independent RNGs (D4). Partitions: `unit`, `hospital`, `dirichlet`, with `unk_policy` handling.
- `data/features.py`: 107-channel layout (34 values + 34 masks + 34 deltas + 5 static) and fixed, data-independent transforms.
- `data/windows.py`: memory-mapped `ProcessedData`; `NormStats.fit` on measured values of training patients only (D5); `build_client_arrays`; lazy causal `WindowDataset` with vectorised `__getitems__` batching.
- `data/eda.py`: per-node tables (all three UNK policies), missingness, LOS, prevalence plots.
- `data/adapters/{mimic_iv,eicu}.py`: documented stubs (layout, cohort, Sepsis-3 derivation) that raise `NotImplementedError`; they never download.
- CLI: `fedguard data download|process|eda` (all with `--fast`); `make smoke` runs the three steps.
- Docs: `docs/data.md`; decisions D13 (UNK = 38.7%) and D14 added; D2 marked adopted.

**Commands and results**
```
fedguard data download        -> 40,036 downloaded + 300 already present; Verified: {training_setA: 20336, training_setB: 20000}
fedguard data process         -> 40,336 patients, 1,552,210 rows in 22 s; checks: 0 unit conflicts,
                                 0 non-monotone labels, 0 non-consecutive ICULOS, 426 onset-ambiguous
fedguard data eda             -> results/eda/ (9 files)
.\scripts\make.ps1 smoke      -> fast download + process + EDA in 20 s (190 patients in the 4 nodes)
.\scripts\make.ps1 test       -> 24 passed in 14.3 s
```

**Real numbers (main 4-node federation, `unit/exclude`)**

| Node | Patients | Septic | Septic rate | Patient-hours | Positive-hour rate |
|---|---:|---:|---:|---:|---:|
| A_MICU | 5,344 | 576 | 10.78% | 204,894 | 2.67% |
| A_SICU | 5,470 | 222 | 4.06% | 199,156 | 1.08% |
| B_MICU | 6,923 | 390 | 5.63% | 262,007 | 1.39% |
| B_SICU | 6,982 | 428 | 6.13% | 274,193 | 1.48% |
| **All 4** | **24,719** | **1,616** | **6.54%** | **940,250** | **1.63%** |

The full dataset (including UNK) has 40,336 patients, 2,932 septic (7.27%) and 1.80% positive hours, matching the published challenge figures. The synopsis's "18% positives" is wrong by an order of magnitude. The AUPRC reference line is **1.63%**.

**Findings that affect later milestones**
- UNK is 38.7% of patients (D13). Centralized uses the same 24,719 patients for a like-for-like upper bound.
- Strong measurement-practice skew across sites (DBP 74% missing in A_MICU vs 10% in B_SICU; HCO3 almost never charted at B). This is genuinely non-IID, and the masks identify the site.
- The HTML demo's hard-coded node sizes (7,400 / 5,200 / 3,600 / 2,100) are illustrative only. The real sizes are above, and the Streamlit app will use them.
- `.gitignore` bug fixed: an unanchored `data/` also ignored `src/fedguard/data/`. It is now `/data/`.

**Open issues:** none blocking. D3 (DP-safe normalisation) is resolved in M5.

---

## Open questions for the team

- **Q3 (D3, M5):** How to handle data-dependent normalisation stats and pos_weight inside DP clients. DP-estimated stats are recommended.

Resolved on 2026-09-26: Q1, patient-level DP adopted (D2); Q2, data and runs moved to `C:\Users\Shrikar\fedguard-work\` (D11).
