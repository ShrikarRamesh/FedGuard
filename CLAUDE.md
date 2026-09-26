# CLAUDE.md: FedGuard

FedGuard is a final-year B.E. (CSE, AI & ML) research project: *Privacy-Preserving Federated Learning for Real-Time ICU Patient Deterioration Monitoring*. The full build spec is `CLAUDE_CODE_BUILD_PROMPT.md`, the scope is `FedGuard_Plan.md`, progress is in `PROGRESS.md`, and every deviation from the plan is in `docs/decisions.md`.

Correctness, honesty and reproducibility matter more than speed. A smaller system that is verifiably right beats a large one that is plausibly wrong.

## Environment

- Python **3.11** virtualenv lives **outside OneDrive** at `C:\Users\Shrikar\.venvs\fedguard` (activate: `C:\Users\Shrikar\.venvs\fedguard\Scripts\Activate.ps1`). The default `python` on PATH is 3.14. Don't use it.
- `make` is not installed on the Windows dev machine. Use `scripts\make.ps1 <target>` (same targets as the `Makefile`).
- `aws` CLI is not installed; `fedguard data download` uses pure-Python HTTPS against the public PhysioNet S3 bucket.
- GPU: RTX 4050 Laptop, 6 GB VRAM, CUDA 13.1 driver, torch cu130 wheels.
- **Data and runs live outside OneDrive:** `FEDGUARD_DATA_DIR=C:\Users\Shrikar\fedguard-work\data`, `FEDGUARD_RUNS_DIR=C:\Users\Shrikar\fedguard-work\runs` (user-level env vars). Raw PhysioNet 2019 is downloaded and verified; processed arrays are in `<data>/processed/` (fast subset in `processed_fast/`).
- **Privacy unit is the patient** (D2, approved): DP datasets sample patients, one random window each; δ < 1/(training patients).

## Non-negotiable rules

1. **Never fabricate results.** Every number in any table, plot, README, `results.json` or the app must come from code actually run, traceable to a run directory. If something hasn't been run, say "not run yet". Demo or placeholder data must be visibly labelled as such everywhere it appears.
2. **No data leakage.**
   - Splits are **patient-level**, never row-level. No patient appears in more than one of train/val/test.
   - Inputs at hour *t* use **only rows ≤ t** (causal). Forward-fill only; never backward-fill or interpolate across the future.
   - Normalisation statistics come from **training patients only**. In federated settings each client uses **its own** training statistics. Centralized uses pooled training statistics.
   - All thresholds (alert thresholds, uncertainty gates, decision thresholds) are tuned on **validation** data only. Test data is touched only for final reporting.
3. **Privacy claims must be exact.** Report ε as the **total** privacy loss over all training for each client, at a stated δ, from the Opacus RDP accountant. Never report "ε per round" as the guarantee. State the unit of privacy (patient vs. record/window); see `docs/decisions.md`.
4. **Reproducibility.** Every run takes a config file and a seed, writes to its own run directory (`runs/<experiment>/<timestamp>_<seed>/`) containing the resolved config, git commit hash, library versions, logs, metrics and checkpoints. Main results use **3 seeds**, reported as mean ± std.
5. **Verify before claiming.** Don't say something works until the tests pass and a smoke run completed. Show the command and its output summary.
6. **Fast mode everywhere.** Every CLI command supports `--fast` (tiny patient subset, 1–2 epochs or rounds) so the whole pipeline can be smoke-tested end to end in under ~5 minutes on CPU.
7. **Credentialed data.** MIMIC-IV and eICU-CRD require PhysioNet credentialing. Never attempt to download them. Build adapters that work once the user places the files locally, and ask the user before touching them.
8. **No secrets in code.** W&B keys, etc. come from environment variables. W&B is optional (`--wandb`); CSV/JSON logging always works offline.

### Library-specific rules learned the hard way
- **Opacus + `nn.Embedding`:** never index an embedding with a bare `arange(P)`. Opacus treats the index tensor's first dim as the batch dim and `optimizer.step()` crashes. Always index with a batch-expanded `[B, P]` id tensor. Verified with opacus 1.6.0 / torch 2.14 (decision D1).
- Check the **installed** version of Flower, Opacus, Captum and Streamlit and read its source/docs before writing code against it. Versions are recorded in `PROGRESS.md`.
- Code must run on Linux and Windows: `pathlib` everywhere, `num_workers=0` fallback on Windows, no shell tricks inside Python.

## Repository layout

```
./                            # repo root
├── CLAUDE.md
├── PROGRESS.md
├── README.md
├── FedGuard_Plan.md
├── pyproject.toml            # package + pinned deps; `pip install -e .[dev]`
├── Makefile                  # common commands (PowerShell equivalent: scripts/make.ps1)
├── configs/
│   ├── data.yaml
│   ├── model/patchtst.yaml
│   ├── train/{centralized,local}.yaml
│   ├── fl/{fedavg,fedprox,fedguard_sync,fedguard_async}.yaml
│   ├── privacy/{none,uniform,adaptive}.yaml
│   └── experiments/          # one YAML per experiment / ablation grid
├── src/fedguard/
│   ├── cli.py                # Typer CLI: fedguard <command>
│   ├── config.py             # OmegaConf config loading + pydantic validation
│   ├── utils/{seed,io,logging,runs}.py
│   ├── data/
│   │   ├── download.py       # PhysioNet 2019 download + checksum/count verification
│   │   ├── physionet2019.py  # parsing, node assignment, splits (from prepare_data.py)
│   │   ├── windows.py        # lazy WindowDataset, causal windows
│   │   ├── features.py       # feature lists, masks, time-since-measured
│   │   ├── eda.py            # summary tables + plots
│   │   └── adapters/{mimic_iv,eicu}.py   # stubs with clear TODOs + expected file layout
│   ├── models/
│   │   ├── patchtst.py       # own implementation, Opacus-compatible
│   │   ├── baselines.py      # logistic regression, LightGBM, GRU
│   │   └── uncertainty.py    # MC Dropout inference
│   ├── train/{loops,centralized,local}.py
│   ├── fl/
│   │   ├── engine.py         # in-house event-driven FL simulator (sync + async)
│   │   ├── aggregators.py    # FedAvg, FedProx, staleness-weighted async
│   │   ├── client.py         # local training (optionally DP) for one client
│   │   └── flower_app/       # Flower ServerApp/ClientApp for simulation + real deployment
│   ├── privacy/{dp,budgets,accounting}.py
│   ├── alerts/{policy,metrics}.py
│   ├── explain/{integrated_gradients,attention}.py
│   ├── attack/gradient_inversion.py
│   ├── eval/{metrics,utility,report}.py
│   └── export/results_json.py   # writes results.json for the HTML + Streamlit apps
├── app/
│   ├── streamlit_app.py
│   ├── pages/{1_Train_together,2_Try_to_steal_data,3_At_the_bedside,4_Results}.py
│   ├── theme.py
│   ├── .streamlit/config.toml
│   └── static/fedguard_demo.html   # copy of "FedGuard live demo.html"
├── scripts/
│   ├── make.ps1                     # PowerShell equivalent of the Makefile
│   ├── run_all_main.sh / .ps1
│   ├── run_ablations.sh / .ps1
│   └── deploy/                      # Flower multi-laptop launch scripts + instructions
├── tests/
├── docs/{decisions,privacy,data,results,demo}.md
├── data/        (gitignored; override location with FEDGUARD_DATA_DIR)
├── runs/        (gitignored; override location with FEDGUARD_RUNS_DIR)
└── results/     (aggregated outputs; small files may be committed)
```

## Hard contract

`results/results.json` must match the schema in section 11.1 of `CLAUDE_CODE_BUILD_PROMPT.md` and is validated by a test (`tests/test_export.py`). Method names: `Local-only`, `FedAvg`, `FedProx`, `FedAvg + DP`, `FedGuard`, `Centralized`.

## Working conventions

- Update `PROGRESS.md` at the end of every milestone: what was built, commands run, test results, real numbers, open issues.
- Record every deviation from `FedGuard_Plan.md` / the build prompt in `docs/decisions.md` (numbered D1, D2, ...).
- Commit after each milestone.
