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

α₀ ∈ {0.5, 1, 2} at λ = 0.35, selected on the best global **validation** AUPRC of async FL without DP (seed 0). One tuning run takes about 58 min under GPU contention, so λ is kept at 0.35 for the main runs. Measured so far (best val AUPRC / test AUROC): α₀ = 0.5 → 0.0620 / 0.742; α₀ = 2.0 → 0.0703 / 0.759. The best value was at the edge of the grid, so α₀ = 4.0 was added: 0.0800 / 0.766. **Chosen: α₀ = 4.0.** The grid cannot usefully extend further: α = min(1, α₀ · n_i/N · s(τ)) with n_i/N ≈ 0.2–0.3 is already about 0.8–1.0 at α₀ = 4, so larger α₀ only saturates at full replacement. The improvement is monotonic in α₀, so the α₀ = 1.0 run (rerun after a GPU out-of-memory failure) cannot change the choice; it is reported for completeness. Its result, 0.0669 / 0.754, confirms the monotonic trend (0.5 < 1 < 2 < 4). Interpretation: in this setting the newest client model should largely replace the global model, with staleness down-weighting as the only brake. The λ ∈ {0.1, 1.0} runs move to the ablation stage and are reported there (they were planned as a second tuning stage). The chosen values are written into `configs/fl/fedguard_async.yaml`; the results are in PROGRESS.md and `results/tables/async_tuning.md`. Experiment queues run as parallel processes (up to 3 on the GPU). The simulated clock makes results independent of real-time contention.

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

## D25. k windows per sampled patient (implemented; selected on validation if it helps)

One random window per sampled patient (D2) discards most of each patient's ~38 windows of signal per step.
- **Mechanism:** `privacy/dp.py` computes each sampled patient's gradient of its mean loss over k windows, drawn uniformly with replacement from the patient's own stay. It uses `torch.func` (vmap over patients; dropout masks independent per patient), clips that per-patient vector to C, sums, and adds N(0, σ²C²) noise.
- **Guarantee:** each patient still contributes one clipped vector per step, so Opacus' `RDPAccountant` (stepped with (σ, q) every step) bounds the privacy loss exactly as before. The guarantee is identical to D2.
- **Verified:**
  - for k = 1 the per-patient clipped sum equals Opacus' per-sample clipped sum on the same batch;
  - for k > 1 a patient's gradient equals the average over its windows;
  - the accountant takes exactly the planned number of steps.
- **Use:** Opacus remains the implementation for k = 1. k ∈ {4, 8} enters the D24 validation selection.

## D26. Clipping norm C = 1 is not tuned: measured gradient norms show it is not the bottleneck (Adopted)

Per-sample gradient norms of the DP loss, measured on 512 random + 256 positive A_MICU training windows (dropout off):
- **Untrained model:** median 3.0 (random windows), 29 (positive); 100% are clipped at C = 1.
- **Trained FedAvg + DP (ε = 3):** median 0.006 for random (mostly negative) windows, of which only 3% are clipped; 149 for positive windows, 100% clipped. Per-patient norms with k = 8 have median 0.007.

The DP model has collapsed toward "always low risk". The rare positive windows (~1.6% of samples) carry the useful signal, and they are clipped to C in any case.
- **Why C doesn't help:** noise is σC and AdamW is scale-invariant, so changing C does not change positives' signal relative to the noise. A much smaller C would only raise the relative weight of the uninformative negatives.
- **The binding constraint** is signal versus noise dimension: about 0.6M parameters with ~4k patients per client and 1.6% positives. Required noise multipliers for the smallest client (δ = 1e-5, R_max = 40):
  - batch 1024, 2 epochs/round: σ = 18.3 / 9.7 / 6.8 / 4.4 / 3.0 at ε = 1 / 2 / 3 / 5 / 8;
  - batch 512, 2 epochs/round: σ = 13.0 / 6.9 / 4.8 / 3.2 / 2.2 at the same ε.
- **Implication:** ε = 8 improves the noise-to-signal ratio about 2.3× over ε = 3. That is still noise-dominated for the positive class, so only partial recovery is expected.
- **Decision:** no further batch/epoch/lr/C variations. The ~0.63 AUROC at ε = 3 is reported as a real privacy–utility cost, and the ε sweep (1–8, 3 seeds, plus the like-for-like no-DP reference) is the headline privacy result.
- **Not pursued (listed as future work):** a smaller model for DP, and oversampling positive hours within a septic patient's own stay (DP-valid, but it changes calibration).

## D27. One time-boxed smaller-model DP run (Adopted by the team, 2026-09-26)

Fewer parameters is the standard remedy for DP noise: the noise norm grows with √d, while the clipped signal does not.
- **Run:** a single configuration (`tune_dp_small`): PatchTST with d_model 64, 2 layers, 4 heads, d_ff 128, head 32, **97,409 parameters versus 594,945** (6.1× fewer, about 2.5× smaller noise norm).
- **Held fixed:** the same patient-level DP (FedAvg + uniform, ε = 3, δ = 1e-5, seed 0, batch 1024 / 5 patient-epochs / lr 5e-4, C = 1, R_max = 40).
- **Decision rule, set before the run:**
  - If it clearly beats the ~0.63 test AUROC of the full-size DP runs, the ε sweep and all DP methods use the small model. The model-size change is reported explicitly: the DP arms then differ in architecture from the non-DP arms, and the like-for-like no-DP reference uses the small model too.
  - Otherwise the plan stays as in D26, and this run is cited as tried.
- **Operational:** at most 3 concurrent GPU jobs. `scripts/run_experiments.py --max-gpu-jobs` (default 3) counts FedGuard jobs machine-wide before starting a new one. This follows one CUDA out-of-memory failure (async α₀ = 1.0, rerun) while four jobs shared the 6 GB GPU.

## D28. DP selection closed: standard Opacus DP-SGD, batch 1024, 5 patient-epochs/round, lr 5e-4 (Adopted)

All runs are FedAvg + uniform patient-level DP, ε = 3, δ = 1e-5, seed 0. "Val AUPRC" is the selection metric.

| Configuration | Val AUPRC | Test AUROC (95% CI) | Test AUPRC |
|---|---|---|---|
| batch 256, 5 epochs, lr 5e-4 (initial) | 0.0252 | 0.628 | 0.024 |
| **batch 1024, 5 epochs, lr 5e-4 (chosen)** | **0.0263** | **0.636 (0.603–0.667)** | 0.028 |
| batch 1024, 2 epochs, lr 1e-3 | 0.0261 | 0.635 (0.603–0.666) | 0.027 |
| k = 8 windows/patient, batch 512, 2 epochs, lr 1e-3 (D25) | 0.0279 | 0.644 (0.618–0.673) | 0.026 |
| k = 4 windows/patient, batch 1024, 2 epochs, lr 1e-3 (D25) | 0.0244 | 0.621 (0.591–0.652) | 0.026 |
| small PatchTST, 97k params, batch 1024, 5 epochs, lr 5e-4 (D27) | 0.0158 | 0.512 (0.479–0.546) | 0.017 |

Test prevalence is 0.0166. A batch-2048 configuration and an lr 2e-3 configuration were stopped unrun to keep at most 3 GPU jobs (D27).

- **Result:** every full-size configuration lands at 0.62–0.64 test AUROC with overlapping CIs.
- **k-window DP (D25) is not adopted.** k = 8 has the nominally best validation AUPRC, but k = 4 is worse than plain DP, so the mechanism does not clearly help, and k = 8 costs 2× the compute.
- **Small model (D27) collapsed.** Validation AUROC stayed at about 0.49 from round 5, and training loss rose (A_MICU 0.94 → 3.00) under noise multipliers of 9.5–10.6. It is cited as tried, not adopted, and was not investigated further (time-boxed).
- **Chosen:** the best standard (k = 1, Opacus) configuration by validation AUPRC.
- **Consequence:** ~0.63 test AUROC at ε = 3 is reported as a genuine privacy–utility cost. The ε sweep (1–8, 3 seeds, like-for-like no-DP reference) is the headline privacy result.

## D29. Alert threshold grid includes the model's own validation-risk quantiles (Adopted; bug fix)

The first alert evaluation of FedGuard (DP, ε = 3) reported zero alarms for all seeds. The fixed τ_r grid started at 0.05, but the DP models' validation risks never exceed 0.0054 (median 0.0023): they predict near the 1.7% base rate, with ECE 0.016. The τ_r grid is now the fixed grid ∪ the 50th–99.9th percentiles of the model's own **validation** mean risk, so it is still validation-only. The test `test_threshold_grid_adapts_to_compressed_risks` covers this.

After the fix:
- **FedGuard seed 0:** τ_r = 0.004, 0.16 false alarms/100 h, sensitivity 2.4%.
- **FedGuard seeds 1–2:** never alerting has the highest validation utility, so the tuned policy raises no alarms.

**Finding:** at ε = 3 the DP models (test AUROC ~0.62) cannot support a useful alerting policy.

## D30. Source of the alert headline and the bedside demo (Adopted)

The spec intends "FedGuard" alerts. With DP at ε = 3 there is no useful alerting (D29), and a bedside replay would show flat ~0.003 risk curves. So:
- `results.json → alerts` and the bedside patients come from **async federated learning without DP**: FedGuard's aggregation (α₀ = 4) with MC-Dropout uncertainty gating.
  - MC predictions are added post hoc from the saved best checkpoint with `fedguard mc-predict`, T = 50.
  - Test AUROC 0.766, seed 0.
- Everywhere they appear (`results_full.json`, the app, docs/results.md) they are labelled "async FL, no DP". The DP-model alert results and the centralized-model alert results are reported next to them.
- Seed 0 test results (validation-tuned thresholds):
  - async FL without DP: threshold-only 1.10 false alarms/100 patient-hours at 41.6% sensitivity; gated 0.95 at 40.0% (−14% false alarms, −1.6 points sensitivity);
  - centralized: 1.10 at 45.7% → 0.78 at 43.7% (−29%, −2.0 points).
- The uncertainty gate's benefit is therefore shown on federated and centralized models without DP. Combining it with DP at ε = 3 is not useful with this model and data.

## D31. Privacy sweep extended to large ε, single seed (Adopted)

The planned sweep (ε ∈ {1, 2, 3, 5, 8}, adaptive rule) shows no recovery. Seed 0 test AUROC:

| ε | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| Adaptive | 0.646 | 0.642 | 0.637 | 0.621 | 0.610 |
| Uniform | – | 0.642 | 0.636 | 0.619 | – |

The like-for-like no-DP reference (public normalisation) is 0.753. At ε = 8 the required noise multipliers are still 4.1–5.0.
- **Extension:** the queue `sweep_ext` adds ε ∈ {16, 32, 64, 256}, adaptive rule, **seed 0 only**. Noise multipliers for the smallest / largest client: 2.67 / 2.41 at ε = 16, 1.65 / 1.51 at 32, 1.09 / 1.01 at 64, 0.55 / 0.53 at 256.
- **Purpose:** locate where accuracy starts recovering. It is reported as a single-seed extension, separate from the 3-seed grid.
- **Caveat:** ε ≥ 16 offers little meaningful formal protection. Those points characterise the privacy–utility trade-off; they are not recommended operating points.

## D32. The ε sweep degrades as ε grows: DP recipe under diagnosis (Open; blocks the privacy result)

Seed-0 test AUROC of FedGuard (adaptive rule) falls monotonically as ε grows:

| ε | 1 | 2 | 3 | 5 | 8 | 16 (extension) |
|---|---|---|---|---|---|---|
| Test AUROC | 0.646 | 0.642 | 0.637 | 0.621 | 0.610 | 0.585 |

The no-DP reference is 0.753. Less noise producing a worse model means the curve is **not** a clean privacy–utility trade-off. The DP training recipe (patient sampling, clipping, AdamW at lr 5e-4, pos_weight 10), selected only at ε = 3, is likely failing on its own.
- **Hypothesis:** noise inflates AdamW's second-moment estimate and acts as an implicit learning-rate reduction. With less noise the effective step grows and training collapses; the small-model run's rising training loss is consistent with this.
- **Action taken:**
  - the DP sweep, the ε extension and the DP budget-rule ablations are paused; completed runs are kept;
  - three seed-0 diagnostics (`dp_diag`) are running: **σ = 0 with the identical pipeline** (clipping, no noise), which measures the recipe's ceiling and has **no privacy guarantee** (ε reported as ∞); σ = 0 at lr 1e-4; ε = 16 at lr 1e-4.
- **Diagnostic results so far** (seed 0; σ = 0 runs have no privacy guarantee):
  - σ = 0, lr 5e-4 → test AUROC 0.723 (0.695–0.752);
  - σ = 0, lr 1e-4 → 0.666 (0.632–0.698), with validation AUROC rising to 0.61 then **decaying** to 0.53 during training.
  - So even without noise, the DP recipe degrades with training. Noise is not the root cause.
- **Round 2** (σ = 0, lr 5e-4):
  - (a) `privacy.reset_optimizer=true`: clear AdamW state each participation, as the non-DP path does. The suspect is stale moments applied to a new global model each round.
  - (b) `privacy.max_grad_norm=1000`: effectively no clipping. The suspect is class-imbalance clipping bias: positive-window gradient norms (~150) are clipped about 150× at C = 1, while negatives (~0.006) are not.
- **If the recipe is at fault:** it is fixed, and DP results are rerun before any privacy claim. The ~0.62 at ε = 3 is not reported as the privacy cost until then.
- **Operational note:** a second hung job (sweep_uniform ε = 1, seed 0; stuck 3 h 20 min) prompted `scripts/watchdog.py`. It kills jobs whose run dir shows no progress for 45 min; the pid is recorded in `meta.json`.

## D33. LightGBM early-stopping bug fixed; strong tree baseline reported as such (Adopted)

The first LightGBM baseline stopped after **1 tree** (test AUROC 0.722). With `eval_metric="average_precision"`, LightGBM still tracked its default `binary_logloss`, and `lgb.early_stopping` stops when *any* metric fails to improve. Under `scale_pos_weight` ≈ 60, validation log-loss worsens immediately.
- **Fix:** `metric="average_precision"` as the only metric and `first_metric_only=True`.
- **Rerun:** 111 trees; test AUROC **0.817**, AUPRC 0.092. The buggy run is kept in `runs/_superseded/` and excluded from reports.
- **Finding** (centralized, seed 0 unless noted):

  | Model | Test AUROC | Test AUPRC |
  |---|---|---|
  | PatchTST (3 seeds) | 0.816 ± 0.011 | 0.106 |
  | LightGBM | 0.817 | 0.092 |
  | Logistic regression | 0.798 | 0.083 |

  A tuned tree ensemble on hand-crafted features matches the transformer's AUROC; PatchTST leads only in AUPRC.
- **Implication:** the choice of a neural model in FedGuard is justified by what the system needs (gradient-based FedAvg/FedProx, DP-SGD, MC-Dropout uncertainty, Integrated Gradients), not by a large accuracy advantage. The report says this explicitly.

## D34. Root cause of the DP results: optimizer state carried across rounds. Fixed; all DP runs rerun (Adopted; resolves D32)

D17 kept each DP client's AdamW state (first and second moments) across participations, while every participation starts from a *different* global model. The moments accumulated on other weights were applied to the new ones. The non-DP path resets the optimizer each round and was unaffected.

Isolating diagnostics (seed 0, σ = 0, i.e. no noise and no privacy guarantee, lr 5e-4, identical pipeline otherwise):

| Configuration | Test AUROC (95% CI) | Validation AUROC over training |
|---|---|---|
| AdamW state carried across rounds (as in D17) | 0.723 | 0.55 → 0.52 → 0.66 → 0.68 |
| carried state, lr 1e-4 | 0.666 | rises to 0.61, then decays to 0.53 |
| carried state, clipping effectively off (C = 1000) | 0.748 | — |
| **AdamW state reset every participation** (C = 1) | **0.757 (0.730–0.783)** | 0.55 → 0.63 → … → 0.73, steady |
| reference: async FL without DP, same public normalisation | 0.753 | — |

- Carried state and clipping interact: removing either one largely restores training. The reset alone reaches the reference while keeping C = 1, and clipping is required for DP, so the fix is the reset and C stays at 1.
- **Fix:** `privacy.reset_optimizer: true` (default in `configs/privacy/uniform.yaml`). Resetting optimizer state touches no data, so the DP guarantee (docs/privacy.md) is unchanged.
- **Consequences:**
  - Every DP result produced before the fix mismeasured the cost of privacy: FedGuard, FedAvg + DP, the ε sweep and extension, and the DP alert results in D29/D30. That includes the monotonic degradation in D31/D32. These runs are archived in `runs/_superseded/pre_optimizer_reset/` and excluded from all reports.
  - The DP hyperparameter selection (D28) and the gradient-norm analysis (D26) were done on affected models. Their conclusions are revisited after the rerun.
  - Everything is rerun with the fix: FedGuard and FedAvg + DP (3 seeds each), the full ε sweep (3 seeds), the large-ε extension (seed 0), and the DP budget-rule ablations.
- **Lesson** (kept in CLAUDE.md): always run a σ = 0 control of a DP pipeline against the equivalent non-DP pipeline before trusting any privacy–utility curve.

## D35. Gradient-inversion attack: batched, sized to fit, clip-only dropped (Adopted)

- **Why the change:** the first full attack (50 windows × 5 conditions × 3 restarts × 400 steps, one window at a time) hit the 4-hour job timeout at about 0.6 s per double-backward step on the shared GPU.
- **Batching:** the attack now optimises windows in batches. `per_window_grads` uses `torch.func` vmap(grad) to give each window its own batch-of-one gradient, differentiable with respect to the dummy inputs. It is tested equal to the per-window loop, and windows do not interact. This makes it about 18× faster.
- **Final size:** 30 test windows (half from septic hours), 300 Adam steps, 2 restarts, conditions no-DP and DP at ε ∈ {1, 3, 8}.
- **Clip-only dropped:** cosine gradient matching is scale-invariant, so clipping alone cannot change the attack. It was verified identical to no-DP in a 2-window check.
- **2-window check with full-length optimisation** (seed 0, FedAvg victim):
  - no DP: gradient matched almost perfectly (1 − cos = 0.004 / 0.010), but vitals were only partially reconstructed (Pearson r = 0.33 / 0.20); labels recovered 2/2;
  - DP at ε = 1, 3, 8: matching fails (1 − cos ≈ 0.93), r ≈ 0, label inference 1/2 (chance).
- **Interpretation:** the channel-mixing patch embedding makes the input far from uniquely determined by one gradient, so even the undefended attack is only partially successful on this model. This is reported as such.
- **Related (MC-Dropout T ablation, async FL without DP, seed 0):** T ∈ {5, 10, 20, 50} changes the MC-mean ECE only in the 5th decimal (0.1564) and AUROC by +0.002. The MC-mean ECE (0.156) is worse than the deterministic model's (0.113). MC Dropout's value here is the uncertainty gate, not calibration.

## D36. At most 2 concurrent GPU jobs for DP work; ablations paused while DP runs (Adopted by the team, 2026-09-27)

- **Why:** with 3–4 concurrent jobs the 6 GB GPU overflowed. Two FedGuard DP jobs alone use about 5.1 GB. Throughput collapsed: FedAvg + DP seed 2 managed 5 of 40 rounds in 3 h, and FedGuard seed 1 hit the 4 h runner timeout at 156/160 merges. Several earlier hangs have the same likely cause.
- **Policy:** the remaining FedGuard and FedAvg + DP seeds run with `--max-gpu-jobs 2`. The ablation queue is paused; its FL jobs are full trainings and not lighter.
- **Order:** after the DP main runs finish, the 6-method, 3-seed table is produced and sent. Then the ε sweep and the ablations run, also capped at 2.
- **Other changes:**
  - the runner's job timeout is now 8 h (hangs are handled by `scripts/watchdog.py`);
  - an exclusive lock makes the "count jobs, then start one" step atomic across runners;
  - the batched attack is deferred until the GPU is otherwise idle.

## D37. Two concurrent DP trainings stall the GPU; finishing crashed runs from their checkpoint (Adopted, 2026-09-27)

- **What happened:** at 14:39 the watchdog killed the FedGuard seed-0 rerun and seed-1 rerun after about 2 h with no progress. From 12:40 the two DP FedGuard jobs held 5.3 of 6.1 GB at 100% utilisation but made no updates at all; the seed-0 rerun was at 84/160.
  - A single DP job alone runs normally: FedAvg + DP seed 2 finished 21 rounds in its first ~7 min.
  - Most likely cause: WDDM paging VRAM to system memory once two DP jobs (per-sample gradients at physical batch 256) share the 6 GB card. This is the same pattern as the earlier crawls (D36).
- **Confirmed afterwards:** FedGuard seed 2 trained alone in 15.5 min (peak 2.9 GB); FedAvg + DP seed 2 alone took 18.5 min. The same jobs had previously made little or no progress for hours when sharing the GPU.
- **Policy change:** DP trainings now run **one at a time**; the 2-job cap of D36 is not enough. From 15:15 all queues (sweep, extension, ablations) run with `--max-gpu-jobs 1`. Inference-only jobs (evaluation, `mc-predict`, alerts) may share the GPU with one DP training.
- **Finalizing instead of retraining:** the earlier FedGuard seed-0 (`20260927-060327_0`) and seed-1 (`20260927-075956_1`) runs had finished training normally. Each has a `done` event (seed 0: 158 merges; seed 1: 156 merges, with all four clients budget-exhausted) and a `best.pt` written by the training process. Their processes were then killed during the post-training MC-Dropout evaluation (by the watchdog, and by the 4 h runner timeout).
  - New command `fedguard fl finalize <run_dir>` runs the **same** post-training code as `run_fl` (shared `_evaluate_and_finish`) on the saved `best.pt`, in the same run directory.
  - The engine result is rebuilt from `events.jsonl`: evals, simulated time, versions, bytes, best validation AUPRC, per-client participations (dispatches) and budgets. Each client's ε is recomputed exactly with the RDP accountant from its logged noise multiplier, sample rate and participations; the recomputation must match the logged value to 1e-5 or finalize refuses.
  - The config hash must match `meta.json`. Only async runs with the standard patient-level accountant are supported.
- **Guarantees:**
  - Tested (`test_finalize_reproduces_post_training_eval`, with and without DP): after deleting `DONE`, `metrics.json`, `summary.json` and the predictions of a completed run, finalize reproduces the predictions bit-for-bit, the val/test metrics, the per-client summary and ε exactly.
  - The only lossy field is `sim_time` and the eval times, which `events.jsonl` rounds to 1e-4 simulated seconds.
- **Traceability:**
  - Finalized runs carry `finalized_from_checkpoint` in `metrics.json` and `summary.json`, with the training and evaluation wall-clock times split.
  - Their peak GPU memory is recorded as `peak_gpu_mem_mib_finalize`, not as the training peak.
  - The proof-of-compute appendix shows each finalized run as a training segment plus a separate finalize segment.
- **Other run directories:** the killed partial reruns (`fedguard/20260927-094159_0`, `fedguard/20260927-123419_1`, and the partial `fedavg_dp/*_2` directories) have no `DONE` and are excluded from all results. They stay in `runs/` as evidence and appear as killed runs in the appendix.

## D38. Open issue: the post-fix privacy–utility curve is flat for ε ∈ [1, 8] (Under investigation, 2026-09-27)

- **Observation:** with the optimizer-reset fix (D34), FedGuard's test AUROC is about 0.59–0.65 at every ε from 1 to 8, with both budget rules.
  - It tracks the seed, not ε: seed 0 ≈ 0.62–0.645, seeds 1 and 2 ≈ 0.59–0.62.
  - Within each seed the best checkpoint is almost always at the same version (seed 0: v100/v112; seed 1: v148; seed 2: v128).
  - For seed 0 it even declines slightly as ε rises (0.645 → 0.619).
- **Controls:** DP with σ = 0 (clip C = 1, same recipe) reaches 0.757, and the non-DP pipeline with public normalisation reaches 0.753 ± 0.009. So the loss is caused by the noise, but it does not shrink as the noise falls from σ ≈ 26 (ε = 1) to σ ≈ 4 (ε = 8).
- **Leading hypothesis (unverified):**
  - AdamW normalises each coordinate's update, so the size of the noise-driven random walk is roughly independent of σ.
  - With about 600k parameters and a clipped signal norm of at most 1, the per-coordinate signal-to-noise ratio stays far below 1 even at ε = 8.
  - If so, this is a property of the recipe (DP-Adam, model size) rather than a code bug.
- **Update 2026-09-28 00:10:** the extension makes it worse, not better.
  - Seed 0 adaptive test AUROC falls as ε rises: 0.645 (ε 1), 0.643 (2), 0.640 (3), 0.632 (5), 0.619 (8), 0.598 (16), 0.562 (32).
  - FedAvg + DP seed 0 follows the same trend: 0.636, 0.629, 0.636, 0.628, 0.602 for ε 1, 2, 3, 5, 8.
  - Client training losses follow an almost identical path for σ from 32 down to 1.5 (20-merge block means 1.5–2.0, rising from their start). The noise level barely changes training, but σ = 0 (same code path, same Opacus wrapper) reaches 0.757.
  - This is consistent with the AdamW explanation: in coordinates dominated by noise, the per-coordinate normalisation makes each step about lr in size whatever σ is.
  - **Diagnostic queued (`dp_diag_sgd`, seed 0, about 75 min):** DP with SGD (momentum 0.9) in place of AdamW; lr ∈ {0.02, 0.1, 0.5} at ε = 8, selected on validation AUPRC; σ = 0 at lr 0.1; ε = 1 at lr 0.1. The option is `privacy.optimizer` (default `adamw`, so existing configs and their hashes are unchanged); tested in `test_dp_sgd_optimizer_option`.
  - **ε = 32, 64 and 256 (seed 0) never improved on the untrained initial model.** Each selected version 0 as its best checkpoint (validation AUPRC 0.0211 = the initial value). Their identical test AUROC (0.5623) is the **initial model's**, not a trained result, and must not be reported as one. At σ ≈ 0.55 (ε = 256) validation AUROC drifts to about 0.50 during training, while σ = 0 reaches 0.757.
    - A tiny noise level destroying training, while zero noise works, is the signature of AdamW's per-coordinate normalisation. Coordinates whose true gradient is (near) zero receive steps of about lr in size once any noise is present, whatever σ is.
    - The report must flag any run whose best checkpoint is version 0.
  - **SGD diagnostic result (seed 0, 07:21):**
    - SGD with momentum 0.9 gave test AUROC 0.633 / 0.614 / 0.616 at ε = 8 for lr 0.02 / 0.1 / 0.5, and 0.632 at ε = 1 (lr 0.1).
    - **σ = 0 at lr 0.1 also collapsed** (validation AUROC decayed to 0.50; test 0.597). So the "noise-driven random walk" explanation is rejected: SGD fails without any noise when its step is too large.
  - **Revised explanation (Tang et al., DP-AdamBC):** when noise dominates Adam's second moment, DP-Adam behaves like DP-SGD with step lr·B/(σ·C), where B is the expected batch.
    - With lr fixed at the value tuned for ε = 3 (5e-4), the effective step is about 0.016 at ε = 1, about 0.05 at ε = 3, and about 0.85 at ε = 256.
    - This predicts every observation: the inverted curve, the collapse to the initial model at ε ≥ 32, the σ = 0 control working (ordinary Adam), and the SGD failures at effective step ≈ 1.
  - **Fix under test (`dp_lr_rule`, FedGuard seed 0, ε ∈ {3, 8, 32, 256, 1}):** `privacy.lr_rule: sigma_scaled` sets each client's lr to `effective_lr · σ_i · C / (q_i n_i)` with `effective_lr` = 0.045.
    - 0.045 was derived from the lr already tuned at ε = 3 (per-client values 0.040–0.051), not tuned anew. At ε = 3 it reproduces lr ≈ 4.4e-4 to 5.7e-4.
    - σ, C, q and n_i are public, so this is post-processing with no privacy cost. Tested in `test_sigma_scaled_lr_rule`; the default (`fixed`) is unchanged.
    - Caveat: the rule assumes noise dominates. At very large ε it gives a very small lr (2.5e-5 at ε = 256) and may under-train.
  - **σ-scaled lr results (FedGuard seed 0, 2026-09-28):** test AUROC at ε = 1 / 3 / 8 was 0.654 / 0.640 / 0.630; the floor (`lr_min` = 1e-4, the non-DP tuned lr) is inactive there.
    - Without floor: ε = 32 → 0.628, ε = 256 → 0.625. With floor (every client's lr = 1e-4, verified in `events.jsonl`): ε = 32 → 0.627, ε = 256 → **0.596** (ECE 0.069).
    - The rule removes the collapse to the initial model; ε = 3 is unchanged (0.640, identical to Table 7.1 seed 0).
    - **The curve stays flat (about 0.63–0.65 from ε = 1 to 32)**, far below the σ = 0 ceiling (0.757). The floor makes ε = 256 worse: a 4× larger step at σ ≈ 0.55 still hurts, so training remains noise-dominated even there. The best checkpoints come early (versions 4–8 at ε ≥ 8).
    - **Leading explanation:** noise norm ≈ σ·C·√d / B with d ≈ 600k parameters. At ε = 256 this is about 0.45, comparable to the clipped signal (norm ≤ 1); at ε = 8 it is about 3.4. Every tested ε is noise-dominated because of model size. This is a real DP limitation of the recipe, not a code error.
    - The earlier smaller-model check (D27, d_model 64) predates the D34 fix and is therefore invalid.
    - **Decision (user, 2026-09-28):** the floor is dropped. The σ-scaled rule is used without `lr_min`. A time-boxed small-model test (`dp_small_rule`: d_model 64, 2 layers, 4 heads, d_ff 128; FedGuard seed 0; ε = 3 and 32; σ-scaled lr, no floor) decides which rerun to do.
    - **Pre-registered model choice (committed in 8221878 at 2026-09-28 21:29:14, before the test's first run started at 21:29:34; validation data only):** the DP rerun uses the small model if its ε = 3 best validation AUPRC is ≥ 0.0305 (≥ 10% above the large model's 0.0277 under the same rule, seed 0), and its ε = 32 best validation AUPRC is not below its ε = 3 value. Otherwise it uses the current model (d_model 128, 4 layers).
    - **Small-model result (21:51):** best validation AUPRC 0.0173 at ε = 3 and 0.0172 at ε = 32 (test AUROC 0.525 / 0.555), versus 0.0277 for the current model at ε = 3. This fails the pre-registered rule, so **the rerun keeps d_model 128, 4 layers**.
      - Caveat: `effective_lr` was calibrated on the large model and not re-tuned for the small one. Its low scores therefore show that the small model does not help under this recipe, not that it could never help.
    - **Adopted:** `lr_rule: sigma_scaled`, `effective_lr: 0.045` in `configs/privacy/uniform.yaml`, which both FedGuard (adaptive inherits it) and FedAvg + DP use. The old DP runs (fedguard, fedavg_dp, sweep_*, abl_rule_*) moved to `runs/_superseded/pre_lr_rule/` at 21:55.
    - **Rerun (user, 2026-09-28):** starts immediately after the test. One GPU job at a time, hard deadline Tue 2026-09-29 12:00, then `fedguard report` and `fedguard proof`. The σ-scaled lr rule (no floor) is the DP recipe for **both** FedGuard and FedAvg + DP. Superseded DP runs move to `runs/_superseded/pre_lr_rule/`.
    - **Priority order** (so a cutoff loses the least important runs):
      1. Table 7.1 DP rows (FedGuard and FedAvg + DP, ε = 3, 3 seeds);
      2. FedGuard adaptive sweep ε ∈ {1, 2, 5, 8} × 3 seeds;
      3. FedAvg + DP sweep ε ∈ {1, 2, 5, 8}, seed 0;
      4. large-ε extension ε ∈ {16, 32, 64, 256}, seed 0;
      5. uniform-rule sweep, seed 0;
      6. DP budget-rule ablations;
      7. FedGuard explanations;
      8. uniform-rule sweep seeds 1–2;
      9. FedAvg + DP sweep seeds 1–2.
  - If the rule makes utility rise with ε, the ε sweep (every point except ε = 3) needs rerunning with it. That is about 8 GPU-hours, and the user decides. The ε = 3 main results are unaffected unless the ε = 3 rerun differs materially. That is about 10–12 GPU-hours, and the user decides.
- **Next steps:** the ε = 16–256 extension (running tonight) shows whether utility recovers at all. A DP-SGD (non-adaptive optimizer) check at ε = 8 would test the hypothesis. Until then the curve is reported as measured, with this caveat.

## D39. Overnight rerun incidents: OneDrive RAM exhaustion and a watchdog PID-reuse kill (2026-09-29)

- **Outcome of `rerun_lr_rule`** (started 2026-09-28 21:57, finished 09:35): 40 jobs succeeded and 13 failed. Priorities 1–7 completed in full: Table 7.1 DP rows, FedGuard adaptive sweep × 3 seeds, FedAvg + DP sweep seed 0, large-ε extension, uniform-rule sweep seed 0 except ε = 8, budget-rule ablations, and FedGuard explanations and alerts.
- **Failures:**
  - 12 failed with `MemoryError` from 06:28 onwards. They were uniform-rule seeds 1–2 and FedAvg + DP seed 1 ε = 1 / seed 2 ε = 2 and 5. The OneDrive client process had grown to 30.8 GB of private memory, leaving 0.4 GB RAM and 2.1 GB commit free.
    - The graceful `/shutdown` was ignored, so the hung process was force-stopped and OneDrive restarted (now about 0.1 GB).
    - The 3-hourly ~1 GB backup zips written into the OneDrive folder are a plausible but unproven contributor. The OneDrive process had been running since 2026-09-26 16:10, before the backups began.
  - 1 (uniform rule, seed 0, ε = 8, exit 15) was **killed by the watchdog**. An abandoned run dir from 2026-09-27 had recorded PID 3792; Windows reused that PID for the new job, and since the new process was a genuine `fedguard.cli` job, the command-line check passed.
    - Fix: the watchdog now only kills a PID whose process start time lies within 15 min before that run dir's creation time. Tested with a dummy process for both cases (reused PID spared, genuinely stale job killed).
- **Handling:** failed runs have no DONE marker and are excluded from every table. The 13 jobs are retried from 09:36 under the same 12:00 deadline, in priority order: uniform seed 0 ε = 8, the FedAvg + DP seeds, then the uniform-rule seeds 1–2.

## D12. Library versions (Adopted)

Built against torch 2.14.0+cu130, opacus 1.6.0, flwr 1.38.0 (Message API: `ServerApp`/`ClientApp`, `flwr.serverapp.strategy.FedAvg/FedProx`), captum 0.9.0, streamlit 1.64.0, and Python 3.11. Exact pins are in `pyproject.toml`.
