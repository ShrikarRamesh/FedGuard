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

## D12. Library versions (Adopted)

Built against torch 2.14.0+cu130, opacus 1.6.0, flwr 1.38.0 (Message API: `ServerApp`/`ClientApp`, `flwr.serverapp.strategy.FedAvg/FedProx`), captum 0.9.0, streamlit 1.64.0, and Python 3.11. Exact pins are in `pyproject.toml`.
