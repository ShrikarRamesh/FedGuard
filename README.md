# FedGuard

**Privacy-preserving federated learning for real-time ICU sepsis early warning.** Final-year B.E. project (CSE, AI & ML) by Shrikar Ramesh, Suhruth R Bharadwaj, Tejas Amarnath and Triambak TS.

Several hospitals train one model that warns 6 hours before sepsis, without sharing patient data:

- **Federated learning:** synchronous FedAvg/FedProx and a staleness-weighted asynchronous FedGuard server.
- **Patient-level differential privacy:** DP-SGD with Opacus. ε is the total per hospital, from the RDP accountant.
- **Uncertainty-gated alerts:** MC Dropout. An alert fires only when the risk is high and the model is confident.
- **Explanations:** Integrated Gradients and attention rollout.
- **Privacy attack demo:** gradient inversion.
- **Live deployment:** Flower across laptops, with a Streamlit app.

| Read this | For |
|---|---|
| [docs/results.md](docs/results.md) | all results (real numbers only, each traceable to a run directory) |
| [docs/privacy.md](docs/privacy.md) | the exact privacy guarantee, its assumptions and what is not protected |
| [docs/decisions.md](docs/decisions.md) | every design decision and deviation from the plan, with reasons |
| [docs/data.md](docs/data.md) | dataset, label semantics, nodes, splits, features, measured prevalences |
| [docs/demo.md](docs/demo.md) | demo runbook: Streamlit app, HTML front end, 4-laptop Flower deployment |
| [PROGRESS.md](PROGRESS.md) | milestone log with commands run, test results and wall-clock times |

## Setup (Python 3.11)

**Linux:**

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # or .../whl/cpu
pip install -e ".[dev]"
make test          # ~1-2 min on CPU
make smoke         # whole pipeline end to end on a tiny subset, ~2 min
```

**Windows (PowerShell).** `make` is not needed; `scripts\make.ps1` has the same targets:

```powershell
py -3.11 -m venv C:\Users\<you>\.venvs\fedguard      # keep venvs out of OneDrive/Dropbox folders
C:\Users\<you>\.venvs\fedguard\Scripts\Activate.ps1
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev]"
.\scripts\make.ps1 test
.\scripts\make.ps1 smoke
```

Optional environment variables:
- `FEDGUARD_DATA_DIR` and `FEDGUARD_RUNS_DIR` move `data/` and `runs/`, e.g. off a synced folder.
- `FEDGUARD_RESULTS_DIR` moves `results/`.
- `WANDB_API_KEY` together with `--wandb` mirrors metrics to Weights & Biases (optional; CSV/JSON logs always work offline).

## Data

PhysioNet/CinC Challenge 2019 (open access, CC-BY 4.0; 40,336 ICU stays from two hospital systems). No login needed:

```bash
fedguard data download      # MD5-verified, resumable; fails loudly unless 20,336 + 20,000 files are present
fedguard data process       # parse, causal forward-fill, hospital x unit nodes, patient-level splits
fedguard data eda           # results/eda/: per-node prevalence, missingness, length of stay
```

MIMIC-IV and eICU-CRD are credentialed. FedGuard never downloads them; `src/fedguard/data/adapters/` documents the expected local layout.

## Commands

Every command accepts `--fast` (tiny deterministic subset, 1–2 epochs/rounds) and `-o key=value` config overrides. Each run writes `runs/<experiment>/<timestamp>_<seed>/` containing:
- `config.yaml` and `meta.json` (git commit and library versions);
- logs, `metrics.csv` and `metrics.json`;
- checkpoints;
- predictions (`preds_{val,test}.npz`);
- for FL runs, `events.jsonl`.

| Command | What it does |
|---|---|
| `fedguard train centralized [--model patchtst\|gru\|lr\|lgbm]` | pooled data, pooled normalisation (upper bound) |
| `fedguard train local --all-nodes` | each hospital node alone (lower bound) |
| `fedguard fl run -c experiments/{fedavg,fedprox,fedavg_dp,fedguard,fedguard_async_nodp}` | in-house simulated-clock FL engine |
| `fedguard alerts -e fedguard` | alert thresholds tuned on validation, alert metrics + calibration on test |
| `fedguard explain -e fedguard` | global Integrated Gradients + attention rollout |
| `fedguard mc-ablation -e fedguard` | MC-Dropout T in {5, 10, 20, 50} |
| `fedguard attack -e fedavg --seed 0` | gradient inversion with and without DP |
| `fedguard report` | `results/summary.csv`, Markdown/LaTeX tables, figures |
| `fedguard export` | `results/results.json` for the HTML and Streamlit front ends (schema-validated) |
| `fedguard fl eval-checkpoints <flower run>` | score a Flower deployment run with the same evaluation code |

## Reproduce everything

```bash
python scripts/run_experiments.py --stage tune      --workers 2   # async mixing (validation)
python scripts/run_experiments.py --stage tune_dp   --workers 1   # DP-SGD hyperparameters (validation)
python scripts/run_experiments.py --stage main      --workers 2   # 6 methods x 3 seeds (+ async without DP)
python scripts/run_experiments.py --stage baselines --workers 2
python scripts/run_experiments.py --stage sweep     --workers 2   # eps in {1,2,3,5,8} x rules x 3 seeds
python scripts/run_experiments.py --stage ablations --workers 2   # seed 0
fedguard mc-ablation -e fedguard && fedguard alerts -e fedguard && fedguard alerts -e centralized_patchtst
fedguard explain -e fedguard && fedguard attack -e fedavg --seed 0
fedguard report && fedguard export
```

`scripts/run_all_main.{ps1,sh}` and `scripts/run_ablations.{ps1,sh}` wrap the same stages. Every stage is **resumable**: a run whose experiment, seed and config hash already completed is skipped. Measured wall-clock times on an RTX 4050 Laptop (6 GB) are in PROGRESS.md.

## Demo

```bash
cd app && streamlit run streamlit_app.py
```

Four screens:
1. Train together: replay recorded training, or watch a live Flower deployment.
2. Try to steal data: the gradient-inversion attack.
3. At the bedside: replay a test patient hour by hour.
4. Results.

Everything shown comes from pipeline artefacts. `-- --demo` shows labelled synthetic data for layout only. `app/static/fedguard_demo.html` is the stand-alone front end; it loads `results/results.json`. The multi-laptop runbook is in [docs/demo.md](docs/demo.md).

## Repository layout

```
configs/            data, model, train, fl, privacy, eval + experiments/ (composable YAML; `_fast` blocks)
src/fedguard/       cli, config, utils, data, models, train, fl (engine, aggregators, client, flower_app),
                    privacy, alerts, explain, attack, eval, export
app/                Streamlit app (theme, data loaders, 4 pages, static HTML + bundled fonts)
scripts/            experiment runner, smoke pipeline, make.ps1, deploy/ (Flower launch scripts)
tests/              pytest (synthetic fixtures; no real data needed)
docs/               decisions, privacy, data, results, demo
results/            aggregated outputs (tables, figures, results.json)
```

## Licences and attribution

- Data: PhysioNet/CinC Challenge 2019 (Reyna et al., *Crit Care Med* 2020), CC-BY 4.0.
- `src/fedguard/eval/_official_2019.py` is the official challenge scoring code, vendored unmodified (BSD-2-Clause, © 2019 PhysioNet; `LICENSES/`).
- Atkinson Hyperlegible font: SIL Open Font License (`app/static/fonts/OFL.txt`).
- The team has not yet chosen a licence for FedGuard's own code.
