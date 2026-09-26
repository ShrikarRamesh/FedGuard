# Build FedGuard from the ground up

You are the lead engineer building **FedGuard**, a final-year B.E. (CSE, AI & ML) research project: *Privacy-Preserving Federated Learning for Real-Time ICU Patient Deterioration Monitoring*. You will build the complete system (data pipeline, models, federated learning, differential privacy, uncertainty-aware alerting, explanations, privacy-attack demo, evaluation, experiment runner, a Streamlit demo app, and a multi-laptop Flower deployment) as a clean, tested, reproducible Python project.

This is research code that will be defended in front of an academic panel. **Correctness, honesty and reproducibility matter more than speed.** A smaller system that is verifiably right beats a large one that is plausibly wrong.

---

## 0. Before writing any code

1. Read these files in the repo root completely:
   - `FedGuard_Plan.md`: the project plan, design decisions and experiment list. This is the source of truth for scope.
   - `prepare_data.py`: an existing, tested starter for the data step. Reuse its logic and move it into the package; don't rewrite it from scratch without reason.
   - `fedguard_demo.html`: the presentation front end. Its `results.json` schema (section 11 below) is a **hard contract** your pipeline must produce.
2. Create `CLAUDE.md` in the repo root containing the **Non-negotiable rules** (section 1) and the **Repository layout** (section 2), so they persist across sessions.
3. Create `PROGRESS.md` with the milestone list (section 13). Update it at the end of every milestone: what was built, commands run, test results, real numbers obtained, open issues.
4. Check the environment: OS, Python version, GPU (`nvidia-smi`), free disk space. The primary dev machine is a laptop with an **NVIDIA RTX 4050 (6 GB VRAM)**; an RTX 4090 workstation may be used for large sweeps later. Code must run on **Linux and Windows** (use `pathlib`, `num_workers=0` fallback on Windows, no shell-specific tricks inside Python).
5. For every third-party library whose API changes often (**Flower, Opacus, Captum, Streamlit**), check the **installed version** and read its actual docs or source before writing code against it. Do not write code from memory of an older API. Note versions in `PROGRESS.md`.
6. Present a short implementation plan for Milestone 1 and then proceed. If anything in `FedGuard_Plan.md` is technically wrong or infeasible, **say so explicitly, propose a fix, and record the decision** in `docs/decisions.md` rather than silently deviating.

---

## 1. Non-negotiable rules

1. **Never fabricate results.** Every number in any table, plot, README, `results.json` or the app must come from code you actually ran, traceable to a run directory. If something hasn't been run, say "not run yet". Demo or placeholder data must be visibly labelled as such everywhere it appears.
2. **No data leakage.**
   - Splits are **patient-level**, never row-level. No patient appears in more than one of train/val/test.
   - Inputs at hour *t* use **only rows ≤ t** (causal). Forward-fill only; never backward-fill or interpolate across the future.
   - Normalisation statistics come from **training patients only**. In federated settings each client uses **its own** training statistics. Centralized uses pooled training statistics.
   - All thresholds (alert thresholds, uncertainty gates, decision thresholds) are tuned on **validation** data only. Test data is touched only for final reporting.
3. **Privacy claims must be exact.** Report ε as the **total** privacy loss over all training for each client, at a stated δ, from the Opacus RDP accountant. Never report "ε per round" as the guarantee.
4. **Reproducibility.** Every run takes a config file and a seed, writes to its own run directory (`runs/<experiment>/<timestamp>_<seed>/`) containing the resolved config, git commit hash, library versions, logs, metrics and checkpoints. Main results use **3 seeds**, reported as mean ± std.
5. **Verify before claiming.** Don't say something works until the tests pass and a smoke run completed. Show the command and its output summary.
6. **Fast mode everywhere.** Every CLI command supports `--fast` (tiny patient subset, 1–2 epochs or rounds) so the whole pipeline can be smoke-tested end to end in under ~5 minutes on CPU.
7. **Credentialed data.** MIMIC-IV and eICU-CRD require PhysioNet credentialing. Never attempt to download them. Build adapters that work once the user places the files locally, and ask the user before touching them.
8. **No secrets in code.** W&B keys, etc. come from environment variables. W&B is optional (`--wandb`); CSV/JSON logging always works offline.

---

## 2. Repository layout

```
fedguard/
├── CLAUDE.md
├── PROGRESS.md
├── README.md
├── FedGuard_Plan.md
├── pyproject.toml            # package + pinned deps; `pip install -e .[dev]`
├── Makefile                  # common commands (also document PowerShell equivalents in README)
├── configs/
│   ├── data.yaml
│   ├── model/patchtst.yaml
│   ├── train/{centralized,local}.yaml
│   ├── fl/{fedavg,fedprox,fedguard_sync,fedguard_async}.yaml
│   ├── privacy/{none,uniform,adaptive}.yaml
│   └── experiments/          # one YAML per experiment / ablation grid
├── src/fedguard/
│   ├── cli.py                # Typer CLI: fedguard <command>
│   ├── config.py             # pydantic/OmegaConf config loading + validation
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
│   ├── train/
│   │   ├── loops.py          # train/eval loops, class weighting, early stopping
│   │   ├── centralized.py
│   │   └── local.py
│   ├── fl/
│   │   ├── engine.py         # in-house event-driven FL simulator (sync + async)
│   │   ├── aggregators.py    # FedAvg, FedProx, staleness-weighted async
│   │   ├── client.py         # local training (optionally DP) for one client
│   │   └── flower_app/       # Flower ServerApp/ClientApp for simulation + real deployment
│   ├── privacy/
│   │   ├── dp.py             # Opacus wrapping, per-client target ε
│   │   ├── budgets.py        # uniform / adaptive ε allocation rules
│   │   └── accounting.py     # RDP accounting, budget tracking, exhaustion
│   ├── alerts/
│   │   ├── policy.py         # threshold-only vs uncertainty-gated alerting
│   │   └── metrics.py        # alarm episodes, false alarms / 100 patient-hours, lead time
│   ├── explain/{integrated_gradients,attention}.py
│   ├── attack/gradient_inversion.py
│   ├── eval/
│   │   ├── metrics.py        # AUROC, AUPRC, bootstrap CIs, calibration (ECE)
│   │   ├── utility.py        # official PhysioNet 2019 utility score (vendored, attributed)
│   │   └── report.py         # aggregate runs -> tables, plots
│   └── export/results_json.py   # writes results.json for the HTML + Streamlit apps
├── app/
│   ├── streamlit_app.py
│   ├── pages/{1_Train_together,2_Try_to_steal_data,3_At_the_bedside,4_Results}.py
│   ├── theme.py              # colours, Plotly templates
│   ├── .streamlit/config.toml
│   └── static/fedguard_demo.html
├── scripts/
│   ├── run_all_main.sh / .ps1       # all main experiments, 3 seeds
│   ├── run_ablations.sh / .ps1
│   └── deploy/                      # Flower multi-laptop launch scripts + instructions
├── tests/
├── docs/{decisions,privacy,data,results,demo}.md
├── data/        (gitignored)
├── runs/        (gitignored)
└── results/     (aggregated outputs; small files may be committed)
```

---

## 3. Data (Milestone 1)

**Primary dataset: PhysioNet/CinC Challenge 2019 (Early Prediction of Sepsis)**, open access, CC-BY 4.0.
- Download: `aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data/raw/physionet2019/`, with a fallback to `wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/training/`. Implement in `fedguard data download` with retries, then **verify**: training_setA should contain 20,336 `.psv` files and training_setB 20,000. Fail loudly if not.
- Each file is one patient, one row per hour, pipe-separated, 40 variables plus `SepsisLabel`. The label is Sepsis-3 based and **already shifted 6 hours early** by the organisers, so predicting `SepsisLabel[t]` from rows ≤ t *is* 6-hour-ahead prediction. Do not shift again. Document this in `docs/data.md`.

**Federated nodes** = hospital system × ICU unit: `A_MICU`, `A_SICU`, `B_MICU`, `B_SICU`, using `Unit1` (MICU) and `Unit2` (SICU).
- Patients with neither unit flag are `UNK`. Make this a config option `unk_policy: exclude | separate | merge_into_hospital`. Report counts for all three in EDA, use `exclude` for the main 4-node experiments, and state the choice in `docs/decisions.md`.
- Also support `partition: hospital` (2 nodes, A and B) and `partition: dirichlet` (k synthetic nodes via Dirichlet over patients, α configurable) for the node-count ablation (2 / 4 / 8 nodes).

**Splits:** per node, patient-level, stratified on "ever septic", 70/15/15, fixed seed. Save `data/processed/splits.csv` (patient_id, node, split). The global test set is the union of node test sets.

**Features** (config-driven):
- All 34 dynamic variables (vitals + labs), forward-filled causally; values never observed so far get the train mean (0 after standardisation).
- Binary **measurement masks** for each variable (missingness is informative).
- Optional **hours since last measurement** per variable (log-scaled, capped).
- Static: Age, Gender, ICULOS (log-scaled), HospAdmTime (clipped, scaled).
- Standardisation: per-node train stats for FL and local runs; pooled train stats for centralized. Store stats in the processed files.

**Windows:** one sample per (patient, hour). Lookback L = 24 hours (configurable), left-padded with zeros plus a padding mask. Build windows **lazily** (store per-patient contiguous arrays plus offsets) so RAM stays small. Label = `SepsisLabel[t]`.

**EDA** (`fedguard data eda`): per node, report patients, septic patients, septic patient rate, rows, positive row rate, missingness per variable, and length-of-stay distribution. Save as CSV + PNG plots into `results/eda/` and summarise in `PROGRESS.md`. **These measured prevalences replace any numbers in the synopsis.**

**Adapters:** `mimic_iv.py` and `eicu.py` are documented stubs describing the expected local file layout, the cohort definition (adults, ICU stay ≥ 24 h), hourly binning, and Sepsis-3 derivation (via the `mimic-code` concepts for MIMIC). They raise a clear `NotImplementedError` with instructions until the user provides data. For eICU, the node = `hospitalid` (the realistic many-hospital federation).

**Acceptance:** download verified; processed files exist for every node; EDA tables and plots saved; data tests pass (section 12).

---

## 4. Models (Milestone 2)

### 4.1 PatchTST (own implementation, `models/patchtst.py`)
Do **not** use the HuggingFace PatchTST: its default BatchNorm is incompatible with DP-SGD.

- Input `x: [B, L, C]` (C = values + masks + optional deltas + static, broadcast over time) plus padding mask.
- **Channel-mixing patch embedding:** split time into patches (patch_len = 4 h, stride = 2 h → 11 patches for L = 24). Flatten each patch (patch_len × C) and project with `nn.Linear` to `d_model`. The original channel-independent PatchTST is impractical with about 70 channels; document this choice.
- Learned positional embedding via `nn.Embedding(num_patches, d_model)` indexed by `arange`, and a CLS token via `nn.Embedding(1, d_model)`. **Do not use bare `nn.Parameter` tensors.** Opacus needs per-sample-gradient support for every parameterised module, and `Linear`, `LayerNorm` and `Embedding` are supported.
- Transformer encoder: **implement attention yourself with `nn.Linear` for Q/K/V/out** (pre-LN blocks, GELU MLP, dropout on attention and MLP). This keeps it fully Opacus-compatible and exposes attention weights for explanations. Defaults: d_model 128, 4 layers, 8 heads, ff 256, dropout 0.2 (config).
- Mask padded patches in attention. Head: CLS → 2-layer MLP with dropout → 1 logit.
- `ModuleValidator.validate(model, strict=True)` must return no errors. Add a unit test that wraps the model with Opacus and confirms per-sample gradients exist for every parameter.

### 4.2 Baselines (`models/baselines.py`)
- Logistic regression and **LightGBM** on hand-crafted features (last value, mean/min/max/slope over 6 h and 24 h, masks), as a sanity floor.
- A small **GRU** on the same windows, as a non-transformer deep baseline.

### 4.3 Training details
- Loss: BCE-with-logits with **positive-class weighting** from train prevalence (a weighted sampler is incompatible with Opacus Poisson sampling, so use loss weighting). Optional focal loss.
- AdamW, lr 1e-4 (config), gradient clipping for non-DP runs, early stopping on **validation AUPRC**.
- fp32 by default (AMP interacts badly with Opacus); make batch size fit 6 GB VRAM. For DP runs use Opacus `BatchMemoryManager` for large logical batches.

### 4.4 Uncertainty (`models/uncertainty.py`)
- MC Dropout: `model.eval()`, then re-enable only `nn.Dropout` modules. T = 50 stochastic passes (config). Return per-sample mean and std of the sigmoid probabilities. Vectorise (repeat the batch or loop over T with no_grad) and keep it fast.

**Acceptance:** centralized PatchTST, GRU, LightGBM and logistic regression train and evaluate in `--fast` mode and in full mode, with real metrics recorded. Model tests pass.

---

## 5. Centralized and local-only training (Milestone 3)

- `fedguard train centralized`: pool all node training data (pooled normalisation). This is the upper bound.
- `fedguard train local --node A_MICU` (and `--all-nodes`): each node trains alone on its own data. This is the lower bound. Evaluate each local model on (a) its own node's test set and (b) the global test set. Report both and the average across nodes.

**Acceptance:** 3-seed results for both, saved and aggregated. Record the real local-only vs centralized gap in `PROGRESS.md`. This gap is the project's headline motivation.

---

## 6. Federated learning (Milestone 4)

### 6.1 In-house FL engine (`fl/engine.py`), used for all reported experiments
An **event-driven simulator with a simulated clock**, so sync vs async comparisons are fair and deterministic.

- Each client has a compute-speed multiplier (config, e.g. `[1.0, 1.35, 0.8, 2.3]`) plus seeded jitter. Local training time in simulated seconds = multiplier × local work.
- Clients can be scripted to go offline and come back (config: list of `[client, t_off, t_on]`) for robustness experiments.
- **Sync mode (FedAvg / FedProx):** each round, the server sends the global model and waits for all online clients (with a timeout for offline ones), then aggregates weighted by n_i. FedProx adds the proximal term μ‖w − w_global‖² locally (μ config).
- **Async mode (FedGuard):** the server merges each update on arrival. Staleness τ = current_version − version_client_started_from. Mixing: `w_global ← (1 − α_i)·w_global + α_i·w_client`, with `α_i = α₀ · (n_i / N) · exp(−λ·τ)` (α₀, λ configurable; defaults chosen on validation and documented). Also implement FedAsync's polynomial staleness function as an option. Cap maximum staleness.
- Local training: E local epochs (default 5 from the plan; tune down if it diverges on non-IID data and document), batch 64, lr 1e-4.
- Log every event to `events.jsonl` (time, client, event type, staleness, weight, global version, validation metrics at intervals, ε spent per client). **The Streamlit "Train together" page replays these real logs.**
- Track communication cost: bytes of parameters sent per update and in total.

### 6.2 Flower implementation (`fl/flower_app/`)
- Implement FedAvg and FedProx as a Flower app using the **current Flower API of the installed version** (read the docs; recent versions use the Message API with `ServerApp`/`ClientApp` and `flwr run`).
- **Cross-check:** the in-house sync FedAvg and Flower FedAvg, with the same seed and config, must reach comparable validation AUROC (within noise). Record this in `docs/decisions.md`. This justifies using the in-house engine for async experiments.
- **Real deployment for the live demo:** scripts plus `docs/demo.md` to run a SuperLink on the server laptop and one SuperNode per client laptop over a LAN, each loading **only its own node's data**. First test it on one machine with 4 local processes. The server writes the same `events.jsonl` format so the Streamlit app can show live training.

**Acceptance:** FedAvg, FedProx, sync and async runs all complete (fast and full); cross-check documented; async-vs-sync wall-clock comparison under load imbalance produced; `events.jsonl` produced.

---

## 7. Differential privacy (Milestone 5)

- Record-level DP-SGD per client with **Opacus**: per-sample clipping (C = 1.0 default), Gaussian noise, **Poisson sampling**, RDP accountant.
- Each client i has a target total budget ε_i at δ_i (default δ = 1e-5, and assert δ < 1/n_i). Because `make_private_with_epsilon` needs the total training length up front, fix each client's **maximum number of participations R_max** (config) and compute the noise multiplier for E × R_max local epochs. A client stops participating when it reaches R_max or its accountant reports ε ≥ ε_i. In async mode fast clients may exhaust their budget early; this is expected, log it.
- **Budget allocation (`privacy/budgets.py`):**
  - `uniform`: ε_i = ε for all.
  - `adaptive` (FedGuard): ε_i = ε · (a + (1 − a) · n_i / n_max), default a = 0.55, as in the plan (larger hospitals get a higher budget). Make the rule pluggable and also implement the inverse rule and an "equal noise multiplier" rule as ablations. This design choice may be questioned by the panel, so let the data decide.
- **Formal statement (`docs/privacy.md`):** each client's training satisfies (ε_i, δ_i)-DP with respect to its own records (RDP accountant, converted to (ε, δ)). Every patient record lives in exactly one client, so by parallel composition any individual record is protected at the ε_i of its hospital, and asynchrony and staleness don't affect the guarantee because only local DP-SGD steps touch data. Write this as a short, precise proposition with proof sketch and assumptions: honest-but-curious server, no secure aggregation, and post-processing immunity for everything the server computes. List what is **not** protected (e.g. hospital-level membership, dataset size n_i disclosure).
- Sweep ε ∈ {1, 2, 3, 5, 8} plus no-DP, for uniform vs adaptive.

**Acceptance:** accountant unit tests pass (ε reported ≤ target); DP runs complete; privacy–utility curve generated from real runs; `docs/privacy.md` written.

---

## 8. Alerting, uncertainty and explanations (Milestone 6)

**Alert policies (`alerts/policy.py`):**
- *Threshold-only:* alert if mean risk > τ_r.
- *Uncertainty-gated (FedGuard):* alert if mean risk > τ_r **and** MC-Dropout std < τ_σ.
- Tune τ_r and τ_σ on **validation** data. Objective: minimise false alarms subject to a sensitivity drop of at most 2 percentage points vs threshold-only at the same τ_r; also report the utility-score-optimal setting. Report the grid.

**Alert metrics (`alerts/metrics.py`):**
- **Alarm episodes:** consecutive alerting hours collapse into one episode; after an alarm, apply a refractory period (default 6 h, config) before counting a new one.
- A true alarm is an episode starting within [onset − 12 h, onset + 3 h] for a septic patient (matching the challenge utility window). All other episodes are false.
- Report: false alarms per 100 patient-hours, patient-level sensitivity, median lead time before onset, and the PhysioNet utility score.

**Calibration:** ECE (15 bins) and reliability diagrams for the deterministic model and the MC-Dropout mean.

**Explanations (`explain/`):**
- Integrated Gradients (Captum) with respect to the input window, aggregated per variable per hour (combine value + mask + delta channels of the same variable).
- Attention rollout from the custom attention module, mapped back to hours.
- Produce per-patient attribution matrices (hours × variables) for the bedside page, and a global feature-importance plot.

**Acceptance:** real false-alarm reduction and sensitivity numbers on the test set, calibration plots, explanation plots; no threshold chosen using test data.

---

## 9. Privacy attack demo (Milestone 7)

`attack/gradient_inversion.py`: a DLG/iDLG-style gradient-inversion attack.
- Setting: the attacker (curious server) sees one client update computed from **a single patient window** (batch size 1, the best case for the attacker; state this). It optimises a dummy input (and label, or use iDLG label inference) to match the observed gradients (L2 or cosine loss, L-BFGS or Adam, several restarts).
- Compare: (a) no DP, raw gradient; (b) gradient after DP clipping + noise at ε ∈ {1, 3, 8}.
- Metrics: per-variable Pearson correlation and MSE between the true and reconstructed window (on the observed channels), averaged over N random test patients (e.g. 50), with CIs.
- Export example curves (true vs reconstructed HR, MAP, Resp over 24 h) for the app.
- Be honest in `docs/results.md` if the attack is only partially successful even without DP on this model; report what actually happens.

---

## 10. Evaluation and experiment runner (Milestone 8)

- Metrics (`eval/metrics.py`): AUROC, AUPRC (always reported **with prevalence**), patient-level bootstrap 95% CIs (1,000 resamples over patients), ECE, Brier score.
- `eval/utility.py`: vendor the official scoring function from the PhysioNet 2019 Python example repository (`github.com/physionetchallenges/python-example-2019`, `evaluate_sepsis_score.py`) with license attribution, and test it against the official example.
- **Main experiments** (3 seeds each): Local-only, Centralized, FedAvg, FedProx, FedAvg + uniform DP, FedGuard (async + adaptive DP), each at ε = 3 plus the full ε sweep for the DP methods.
- **Ablations:** node count 2 / 4 / 8 (hospital / unit / Dirichlet partitions); sync vs async under load imbalance and client dropout; uniform vs adaptive vs alternative ε rules; MC-Dropout T ∈ {5, 10, 20, 50} vs ECE and alert metrics; lookback L ∈ {12, 24, 48}.
- `scripts/run_all_main.*` and `scripts/run_ablations.*` run everything resumably (skip completed run directories).
- `fedguard report`: aggregate all runs into `results/summary.csv`, LaTeX/Markdown tables for the report, and publication-quality plots (matplotlib, consistent style) in `results/figures/`. Fill `docs/results.md` with **only** real numbers.

---

## 11. Exports and the Streamlit app (Milestone 9)

### 11.1 `results.json` (hard contract with `fedguard_demo.html`)
`fedguard export` writes `results/results.json`:
```json
{
  "methods": [ {"name": "Local-only", "auroc": 0.0, "auprc": 0.0}, "..." ],
  "prevalence": 0.0,
  "privacy": { "eps": [1, 2, 3, 5, 8], "uniform": [], "adaptive": [], "no_dp": 0.0 },
  "alerts": {
    "threshold_only": {"false_per_100h": 0.0, "sensitivity": 0.0},
    "fedguard":       {"false_per_100h": 0.0, "sensitivity": 0.0}
  },
  "patients": [
    { "name": "Test patient <id>", "onset_hour": null,
      "hr": [], "map": [], "resp": [], "temp": [], "spo2": [],
      "risk_mean": [], "risk_std": [],
      "attr_features": ["HR", "MAP", "..."],
      "attr": [[]]
    }
  ]
}
```
- Method names used in the HTML: `Local-only`, `FedAvg`, `FedProx`, `FedAvg + DP`, `FedGuard`, `Centralized`. Values are 3-seed means. Also write `results/results_full.json` with stds and CIs.
- Vitals in `patients` are **raw clinical units** (not standardised), forward-filled for display, one value per hour, all arrays the same length. `attr` has one row per hour and one column per `attr_features` entry.
- Patient selection for the demo: choose a small set (e.g. 3 septic, 3 non-septic including at least one where threshold-only fires falsely and FedGuard doesn't) **by a documented, reproducible rule**, and record the selection rule in `docs/demo.md`. Never hand-edit predictions.
- Add a test that loads `results.json` and validates it against a JSON schema matching the above.

### 11.2 Streamlit app (`app/`)
A polished, presentation-grade multipage app mirroring the four screens of `fedguard_demo.html`, but driven entirely by **real pipeline artefacts**:

1. **Train together:** replay a recorded `events.jsonl` from a chosen run (dropdown), or **live mode** that tails the `events.jsonl` of a running Flower deployment (`st.fragment(run_every=...)` or equivalent in the installed Streamlit version). Show a network diagram of hospitals ↔ server with update events, global validation AUROC over time with local-only and centralized reference lines from real runs, per-client ε spent vs budget, and an event log. Include a sync vs async comparison view from real runs.
2. **Try to steal data:** load attack outputs; show true vs reconstructed curves, with a selector for no DP / ε values and the aggregate reconstruction metrics.
3. **At the bedside:** replay exported test patients hour by hour with an ICU-monitor-style panel (dark monitor, conventional colours: HR green, MAP red, SpO₂ cyan, Resp yellow, Temp white), a risk chart with an uncertainty band, the alert threshold, threshold-only alarm markers vs FedGuard alert markers, an onset marker, live threshold sliders that recompute alarms, an alarm tally, and the attribution heatmap.
4. **Results:** AUROC/AUPRC bars with CIs and a prevalence line, the privacy–utility curve, the alert comparison, calibration plots, and the ablation tables.

Requirements:
- Visual identity consistent with the HTML demo: background `#EDF2F1`, ink `#15283A`, accent `#1F6E8C`, alert `#C8413A`; node colours `#2E7D7A`, `#4C5FB8`, `#B7791F`, `#B0487A`; Atkinson Hyperlegible for text; large, projector-readable numbers. Put this in `app/theme.py` and `.streamlit/config.toml`, with a Plotly template.
- If an artefact is missing, show a clear message naming the command that produces it (e.g. "Run `fedguard export` first"). **Never silently fall back to synthetic data.** A `--demo` flag may load clearly labelled synthetic data, with a persistent "DEMO DATA" banner.
- Cache loaded artefacts (`st.cache_data`). The app must start in under ~3 s and work fully offline.
- Serve `app/static/fedguard_demo.html` as a link or download so either front end can be used.
- Add a smoke test (Streamlit `AppTest`, if available in the installed version) that each page renders without exceptions, both with artefacts present and with them missing.

---

## 12. Tests (continuous, pytest)

At minimum:
- **Data:** file counts after download; no patient in more than one split; every patient in exactly one node; windows are causal (perturbing rows > t doesn't change window t); normalisation stats computed from train patients only; labels unchanged from the raw files; `--fast` subset is deterministic.
- **Model:** output shapes; padding mask respected; `ModuleValidator` clean; Opacus per-sample gradients for all parameters; MC Dropout gives nonzero std only when dropout is active.
- **FL:** FedAvg of identical client models equals the model; weights sum correctly; async staleness weights decrease with τ; the simulated clock is deterministic for a fixed seed; offline clients are handled.
- **Privacy:** accountant ε ≤ target after planned steps; adaptive budgets follow the rule; a client stops after budget exhaustion.
- **Alerts and metrics:** episode collapsing and refractory logic on hand-made sequences; utility score matches the official implementation on a fixture; bootstrap is resampled by patient.
- **Export:** `results.json` validates against the schema.
- **App:** pages render.

Tests must run in under ~2 minutes on CPU (use tiny synthetic fixtures). Add `make test` and `make smoke` (end-to-end `--fast` pipeline: download check → process → train → FL → DP → alerts → export → app smoke test).

---

## 13. Milestones (work in this order; update `PROGRESS.md` after each)

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold: package, configs, CLI skeleton, CLAUDE.md, PROGRESS.md, CI-free `make test` | `pip install -e .[dev]` works, empty tests pass |
| M1 | Data: download, verify, nodes, splits, windows, EDA | Data tests pass; EDA with real prevalences recorded |
| M2 | Models: PatchTST (Opacus-safe), baselines, MC Dropout | Model tests pass; fast runs work |
| M3 | Centralized + local-only, 3 seeds | Real upper/lower bounds recorded |
| M4 | FL engine (sync/async, FedAvg/FedProx) + Flower app + cross-check + deployment scripts | FL tests pass; cross-check documented; 4-process local deployment works |
| M5 | DP: Opacus per client, uniform/adaptive budgets, accounting, ε sweep, `docs/privacy.md` | Privacy tests pass; privacy–utility curve from real runs |
| M6 | Alerts, calibration, explanations | Real alert metrics on test; thresholds from validation only |
| M7 | Gradient-inversion attack | Real attack metrics with/without DP |
| M8 | Full experiments + ablations + `fedguard report` | `results/summary.csv`, figures, `docs/results.md` with real numbers |
| M9 | Export + Streamlit app | `results.json` validates; app renders all pages from real artefacts |
| M10 | Polish: README (setup, commands, reproduce-all, demo runbook), docstrings, type hints, `make smoke` green | A fresh clone can reproduce everything by following README |

At the end of each milestone, **stop and give me a short report**: what was built, tests and results, real numbers, deviations from the plan and why, and anything that needs my decision. Continue to the next milestone unless something needs my input.

---

## 14. Engineering standards

- Python 3.11, type hints throughout, docstrings on public functions, `ruff` + `black` formatting.
- Pin dependency versions in `pyproject.toml` after confirming they install together (torch with CUDA for the local GPU, opacus, flwr, captum, lightgbm, scikit-learn, pandas, numpy, pyarrow, typer, pydantic or omegaconf, matplotlib, plotly, streamlit, pytest, wandb as an optional extra).
- Deterministic seeding (Python, NumPy, torch, CUDA flags where feasible); note any remaining nondeterminism.
- Checkpoints: best-on-validation plus last; resumable long runs.
- Clear errors with actionable messages. No bare `except`.
- Keep the core model small enough that a full federated DP experiment fits on the RTX 4050 in reasonable time; report per-experiment wall-clock times in `PROGRESS.md` so we can plan runs on the 4090.
- Commit after each milestone with a descriptive message (if git is initialised; initialise it in M0 with a proper `.gitignore` for `data/`, `runs/`, checkpoints and secrets).

---

## 15. What "done" looks like

A panel member can clone the repo and:
1. Run `make smoke` and see the whole pipeline pass in minutes.
2. Follow the README to download the data and reproduce every table and figure in `docs/results.md`.
3. Run `streamlit run app/streamlit_app.py` and walk through the four screens powered by real results.
4. Follow `docs/demo.md` to run live federated training across four laptops, with the app showing it live.
5. Read `docs/privacy.md` and `docs/decisions.md` and understand exactly what is guaranteed, what isn't, and why each design choice was made.

Start with Section 0 now.
