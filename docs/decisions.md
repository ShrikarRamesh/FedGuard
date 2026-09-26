# Design decisions and deviations

Every deviation from `FedGuard_Plan.md` or `CLAUDE_CODE_BUILD_PROMPT.md` is recorded here with the reason. Status is **Adopted** (implemented or will be), **Proposed** (needs team sign-off), or **Open** (to be resolved at the named milestone).

---

## D1. Positional/CLS embeddings must be indexed with a batch-expanded id tensor (Adopted)

**Spec said:** "Learned positional embedding via `nn.Embedding(num_patches, d_model)` indexed by `arange`."

**Problem (verified, opacus 1.6.0, torch 2.14.0):** Opacus's per-sample-gradient hook for `nn.Embedding` treats the *index tensor's* first dimension as the batch dimension. Indexing with `arange(P)` gives `grad_sample` of shape `[P, num_embeddings, d]` instead of `[B, ...]`, and `DPOptimizer.step()` crashes with `stack expects each tensor to be equal size`. The same applies to the CLS embedding indexed by `zeros(1)`.

**Fix:** index with `arange(P).unsqueeze(0).expand(B, P)` (and `zeros(B, 1)` for CLS). Verified that the per-sample gradients then have batch dimension B for every parameter and that a DP step completes. A unit test guards this (M2).

## D2. Unit of privacy: patient-level DP instead of window-level "record" DP (Adopted 2026-09-26, approved by the team)

**Spec said:** "Record-level DP-SGD per client", with one training sample per (patient, hour) window.

**Problem:** if the DP "record" is a window, the (ε, δ) guarantee protects **one patient-hour window**, not a patient. A patient contributes on average about 38 windows (up to about 336), and consecutive windows share 23 of 24 hours of data. By group privacy the patient-level guarantee for k windows degrades to about kε (with a much worse δ), so "ε = 3" would be meaningless for a patient. Presenting it as patient protection would violate rule 3 (exact privacy claims).

**Fix (adopted):** make the **patient** the unit of privacy. The DP dataset element is a patient. Opacus Poisson-samples *patients* with rate q; each sampled patient contributes **one** window chosen uniformly at random (randomness independent of other patients) from its own stay, and that single per-sample gradient is clipped to C. Every patient contributes at most one clipped gradient of norm ≤ C per step. The standard subsampled-Gaussian RDP analysis therefore holds under add/remove-one-*patient* adjacency. For each fixed draw of the window-selection randomness the bound holds, and Rényi divergence is jointly quasi-convex, so it also holds for the mixture. Then δ < 1/n_i uses n_i = number of training *patients*.

**Cost:** a DP "epoch" is one expected pass over patients (one window each), so DP runs see fewer windows per epoch than non-DP runs. That is the honest price of patient-level privacy and will show up in the privacy–utility curve.

**Possible later extension:** k windows per sampled patient with gradients summed per patient *before* clipping. This needs a custom per-patient aggregation around Opacus grad samples plus a manual accountant, so it is deferred.

## D3. Data-dependent preprocessing inside DP clients (Open; resolve in M5)

Per-client normalisation statistics and the positive-class weight are computed from the client's private training data. They never leave the client, but the model update is a function of them. Strictly, changing one patient changes every other patient's standardised input, which breaks the sensitivity argument of DP-SGD. Options:
1. **DP-estimated statistics:** clipped per-feature sums and counts via the Gaussian mechanism at a small ε_pre, composed into the reported total ε. (**Recommended.**)
2. Public, data-independent normalisation (clinical reference ranges) and a configured pos_weight for DP runs.
3. List it as an unprotected leak in `docs/privacy.md`. (Weakest; only if 1 and 2 hurt utility badly.)

Non-DP runs keep exact per-node training statistics as the spec requires.

## D4. Splits are assigned once per patient, independent of the federated partition (Adopted)

Patient-level 70/15/15 splits are stratified on "ever septic" **within each hospital × unit stratum** (including UNK), using a per-stratum RNG seeded from `(seed, stratum name)` so the result does not depend on processing order. `prepare_data.py` used one RNG consumed sequentially across nodes. Partitions (`unit` = 4 nodes, `hospital` = 2 nodes, `dirichlet` = k nodes) only regroup already-split patients into clients. The global test set is therefore identical across the 2/4/8-node ablation, which keeps it comparable.

## D5. Processed data stores raw (unstandardised) forward-filled values; stats are computed at load time (Adopted)

`prepare_data.py` standardised with per-node statistics at preprocessing time. Centralized runs need pooled statistics, Dirichlet partitions need per-synthetic-client statistics, and D3 may need DP statistics, so standardisation moves to load time and is computed from the training patients of whichever client or pool is being built. Statistics are computed over **actually measured** values of training patients (each measurement counted once). Using forward-filled rows over-weights long stays and stale values. The statistics are saved alongside each run.

## D6. Static features (Adopted)

The spec adds `HospAdmTime` to the starter's Age/Gender/ICULOS. All static scaling is fixed and data-independent, so there is no leakage: Age (age − 60)/20, Gender as-is, log1p(ICULOS)/5, and HospAdmTime as log1p(clip(−h, 0, 720))/log1p(720). HospAdmTime is missing for some stays, so it gets an extra `HospAdmTime_obs` channel. Missing Age/Gender map to 0. The input has 34 values + 34 masks + 34 deltas + 5 static = **107 channels**.

## D7. Privacy sweep grid (Adopted)

The plan lists ε ∈ {1, 3, 8, ∞}; the build prompt lists ε ∈ {1, 2, 3, 5, 8} plus no-DP. We use the build prompt's grid (a superset).

## D8. Alert thresholds in the plan are placeholders (Adopted)

The plan diagram shows "risk > 0.65 AND spread < 0.12" and the HTML demo defaults to those values. These are **not** used. τ_r and τ_σ are tuned on validation data only (M6). The uncertainty measure is the **standard deviation** of MC-Dropout probabilities, as the plan decided.

## D9. HTML demo file name and its own alarm logic (Adopted)

The repo contains `FedGuard live demo.html`; the build prompt calls it `fedguard_demo.html`. It is copied unchanged to `app/static/fedguard_demo.html`. Its client-side tally counts an alarm as false only if it fires more than 12 h before onset; it does not count alarms after onset + 3 h as false. The reported metrics come from `alerts/metrics.py`, which uses the full [onset − 12 h, onset + 3 h] window; the HTML is presentation-only.

## D10. Attention implementation (Adopted)

The plan's risk table suggests Opacus's `DPMultiheadAttention`. Per the build prompt we implement attention ourselves with `nn.Linear` Q/K/V/out projections instead. That is fully Opacus-compatible and exposes attention weights for rollout.

## D11. Tooling on the Windows dev machine (Adopted)

- `aws` CLI is not installed, so the downloader uses pure-Python HTTPS against the public S3 bucket `physionet-open` (anonymous ListObjectsV2 plus GET) with retries, falling back to `https://physionet.org/files/challenge-2019/1.0.0/training/`.
- `make` is not installed, so `scripts/make.ps1` mirrors every `Makefile` target.
- The repo is inside OneDrive, so the virtualenv lives outside it (`C:\Users\Shrikar\.venvs\fedguard`). **Decided 2026-09-26:** `data/` and `runs/` live at `C:\Users\Shrikar\fedguard-work\{data,runs}`, set by the user-level environment variables `FEDGUARD_DATA_DIR` / `FEDGUARD_RUNS_DIR`. `scripts/make.ps1` picks them up even in older terminals. `results/` (small aggregated outputs) stays in the repo.

## D13. UNK patients are 38.7% of the data; `exclude` kept for the main 4-node runs (Adopted, flagged)

Measured in M1: 15,617 of 40,336 patients have neither `Unit1` nor `Unit2` set (A_UNK 9,522, B_UNK 6,095). A_UNK is septic-rich (10.4% septic patients). The main 4-node federation uses `unk_policy: exclude` as specified, because a unit cannot be invented for these patients. It keeps 24,719 patients (61.3%). Consequences, stated in the report:
- The "Centralized" upper bound is centralized **over the same 24,719 patients**, not over all 40,336, so the comparison stays like-for-like.
- `unit/separate` (6 nodes, with A_UNK and B_UNK as their own clients) and `hospital/merge_into_hospital` (2 nodes, all 40,336 patients) are available as ablations; their counts are in `results/eda/`.
- `merge_into_hospital` is undefined for the unit partition and raises an error rather than assigning UNK patients to an arbitrary unit.

## D14. Processing details from the full data (Adopted)

- Unit assignment uses any row with a non-NaN flag (flags were never contradictory: 0 conflicts).
- Onset hour = first positive row + 6. The 426 patients positive from their first row are flagged `onset_ambiguous`, and lead-time statistics are reported with and without them.
- The fast subset is a seeded random sample of 150 files per hospital (seed 42), identical between `download --fast` and `process --fast`. It is so small that some clients have one septic patient, so downstream metrics must return NaN (not crash) when a split has no positives.

## D15. Normalisation at evaluation time (Adopted)

- **Centralized** models use pooled training statistics, both in training and in evaluation.
- **Local-only and non-DP federated** models: every patient, in training and in evaluation, is standardised with the training statistics of *its own* hospital node. Each site standardises its own data, which is how a deployed model would be used. This includes a local model evaluated on other nodes' patients in the global test set.
- **DP** runs use the public reference normalisation everywhere (D3).

## D16. Local work is K optimiser steps, not E full epochs (Adopted)

The plan's default of 5 local epochs per round would cost about 5 centralized epochs per round, roughly 2.5 h per FL run on the RTX 4050, which is infeasible for 3 seeds × all methods. Measured throughput is about 13k windows/s.
- Non-DP clients run **K = 500 AdamW steps of batch 64** (32k windows) per participation, over **R = 40** rounds. That is about 8 passes over each client's own data, roughly the compute of the centralized run (which early-stops after about 8 epochs).
- Async runs get the same total number of local jobs (160 = 40 × 4).
- AdamW state resets every participation (standard FedAvg). FedProx uses μ = 0.01.

## D17. DP clients use AdamW on the privatised gradient (Adopted)

Adam only sees noisy clipped gradients, so this is post-processing and changes nothing in the guarantee. It avoids tuning an SGD learning rate for a transformer under DP. Details:
- AdamW state persists across a client's participations; it is a function of that client's own noisy outputs only.
- Clipping C = 1.
- Clients run no additional gradient-norm clipping on top.

## D18. Simulated clock details (Adopted)

- Job time = speed_i × samples/1000 × jitter (clipped at 0.5, seeded) plus 2 × model bytes / 100 Mbit/s.
- **Sync:** the server waits for every online client; stragglers are the cost of synchrony. The `sync_timeout` (150 s) applies only to clients that go offline mid-job.
- **Lost updates:** a client scheduled to be offline during its job is known in advance. It does not train and releases nothing, so its accountant is not charged.
- **Async:** a client restarts from the newest global model immediately after each merge. Updates with staleness > 8 are dropped.
- Local training seeds torch per (client, participation), so results do not depend on the order clients are simulated in (needed for the Flower cross-check, D19).

## D19. Flower implementation and cross-check (Adopted)

- **Runtime:** Flower 1.38 Message API (`ServerApp` with the built-in `FedAvg`/`FedProx` strategies, `ClientApp`) on the **Deployment Engine**. The Simulation Engine needs Ray, which is not installed and not needed. The Control API in 1.38 is HTTP; the SuperLink is started with `--host/--port 9093`.
- **Shared code:** the ClientApp reuses FedGuard's `FLClient` (same model, data, per-client seed, local steps, aggregation weights). The participation counter comes from the server's `server-round`, because subprocess isolation runs every message in a fresh process.
- **No patient data on the server.** Clients report their own validation AUROC for the live view. The server saves every round's global model, and `fedguard fl eval-checkpoints` scores those checkpoints with exactly the in-house evaluation.
- **Cross-check (fast config, 2 rounds, 4 local SuperNodes):** all 8 per-client training losses match the in-house sync FedAvg to within 1e-7 (e.g. A_MICU round 1: 1.2944918632507325 in both). The full 40-round comparison is in PROGRESS.md.

## D20. Async mixing parameters chosen on validation (Adopted)

α₀ ∈ {0.5, 1, 2} at λ = 0.35, then λ ∈ {0.1, 1.0} at the best α₀. Selection uses the best global **validation** AUPRC of async FL without DP (seed 0). The chosen values are written into `configs/fl/fedguard_async.yaml`; the results are in PROGRESS.md and `results/tables/async_tuning.md`. Experiment queues run as parallel processes (up to 3 on the GPU). The simulated clock makes results independent of real-time contention.

## D21. Ablations use one seed (Adopted)

Main results (6 methods) and the privacy sweep use 3 seeds. Ablations use seed 0 only:
- node count 2/4/8;
- sync vs async under dropout and imbalance;
- budget rules;
- lookback 12/48;
- MC-Dropout T.

With one seed their differences are indicative, not significant, and they are reported as such (n = 1).

## D22. Alert episodes and operating point (Adopted)

- **Episodes:** a maximal run of consecutive alerting hours is one episode. A new run starting ≤ 6 h (the refractory period) after the previous episode ended is merged into it.
- **True alarm:** an episode starting in [onset − 12 h, onset + 3 h] of a septic patient (onset = first positive label + 6). Every other episode is false.
- **Risk score:** both policies use the **MC-Dropout mean** (T = 50), so the only difference is the gate.
- **Operating point (validation only):**
  - τ_r = argmax validation utility of threshold-only alerting;
  - τ_σ = the gate minimising validation false alarms/100 patient-hours subject to patient-level sensitivity ≥ threshold-only − 2 percentage points.
- The jointly utility-optimal (τ_r, τ_σ) and the full grids are also reported.
- Lead time is reported with and without the 426 onset-ambiguous patients.

## D23. Gradient-inversion threat model (Adopted)

- **Victim:** the trained FedAvg global model (seed 0, no DP).
- **What the server observes:** the gradient of a single full 24-hour window, batch size 1, dropout off, with the loss (including pos_weight) and padding mask known. This is the attacker's best case.
- **Conditions:**
  - raw gradient;
  - clipping to C = 1 only;
  - clipping plus Gaussian noise with the noise multiplier FedGuard's **largest** client uses at total ε ∈ {1, 3, 8} (the least noise of any client).
- **Attack:** iDLG label inference, then cosine gradient matching (Adam, 400 steps, 3 restarts).
- **Metrics:** Pearson r and MSE on the vital-sign value channels, over 50 test patients (half from septic hours), with patient-bootstrap CIs.

## D24. DP-SGD hyperparameters chosen on validation (Adopted)

The first FedAvg + DP run (ε = 3, logical batch 256, 5 patient-epochs per round, R_max = 40) needed noise multipliers of about 5. Validation AUROC plateaued near 0.62; test AUROC was 0.628.
- **Grid:** logical batch {1024, 2048} (physical 256 via BatchMemoryManager) × patient-epochs per round {2, 5} × lr {5e-4, 1e-3, 2e-3}, using 4 of the combinations.
- **Selection:** FedAvg + uniform DP at ε = 3, seed 0, on best validation AUPRC. The model and rounds are unchanged.
- **Caveat:** this selection is not included in ε (docs/privacy.md). The chosen setting is used for every DP run.

## D12. Library versions (Adopted)

Built against torch 2.14.0+cu130, opacus 1.6.0, flwr 1.38.0 (Message API: `ServerApp`/`ClientApp`, `flwr.serverapp.strategy.FedAvg/FedProx`), captum 0.9.0, streamlit 1.64.0, and Python 3.11. Exact pins are in `pyproject.toml`.
