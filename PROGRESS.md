# FedGuard progress

Only real, reproducible numbers appear here. Anything not yet run says **not run yet**.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Scaffold: package, configs, CLI skeleton, CLAUDE.md, PROGRESS.md, `make test` | **Done** (2026-09-26) |
| M1 | Data: download, verify, nodes, splits, windows, EDA | Next, waiting on Q2 (data location) |
| M2 | Models: PatchTST (Opacus-safe), baselines, MC Dropout | not started |
| M3 | Centralized + local-only, 3 seeds | not started |
| M4 | FL engine (sync/async, FedAvg/FedProx) + Flower app + cross-check + deployment scripts | not started |
| M5 | DP: Opacus per client, uniform/adaptive budgets, accounting, ε sweep, `docs/privacy.md` | not started; needs Q1 (D2) and D3 |
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

## Open questions for the team

- **Q1 (D2, blocks M5):** Adopt **patient-level DP** (one random window per Poisson-sampled patient) instead of window-level "record" DP? Window-level ε would not protect a patient. This is recommended.
- **Q2 (blocks the M1 download):** The repo is in OneDrive. The raw data is 40,336 small `.psv` files, and processed arrays, checkpoints and runs will follow. Keep `data/` and `runs/` in the repo (spec layout, synced by OneDrive), or relocate them with `FEDGUARD_DATA_DIR` / `FEDGUARD_RUNS_DIR` to a non-synced folder such as `C:\Users\Shrikar\fedguard-work\`? Relocating is recommended.
- **Q3 (D3, M5):** How to handle data-dependent normalisation stats and pos_weight inside DP clients. DP-estimated stats are recommended.
