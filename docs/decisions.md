# Design decisions and deviations

Every deviation from `FedGuard_Plan.md` or `CLAUDE_CODE_BUILD_PROMPT.md` is recorded here with the reason. Status is **Adopted** (implemented or will be), **Proposed** (needs team sign-off), or **Open** (to be resolved at the named milestone).

---

## D1. Positional/CLS embeddings must be indexed with a batch-expanded id tensor (Adopted)

**Spec said:** "Learned positional embedding via `nn.Embedding(num_patches, d_model)` indexed by `arange`."

**Problem (verified, opacus 1.6.0, torch 2.14.0):** Opacus's per-sample-gradient hook for `nn.Embedding` treats the *index tensor's* first dimension as the batch dimension. Indexing with `arange(P)` gives `grad_sample` of shape `[P, num_embeddings, d]` instead of `[B, ...]`, and `DPOptimizer.step()` crashes with `stack expects each tensor to be equal size`. The same applies to the CLS embedding indexed by `zeros(1)`.

**Fix:** index with `arange(P).unsqueeze(0).expand(B, P)` (and `zeros(B, 1)` for CLS). Verified that the per-sample gradients then have batch dimension B for every parameter and that a DP step completes. A unit test guards this (M2).

## D2. Unit of privacy: patient-level DP instead of window-level "record" DP (Proposed; needs team sign-off before M5)

**Spec said:** "Record-level DP-SGD per client", with one training sample per (patient, hour) window.

**Problem:** if the DP "record" is a window, the (ε, δ) guarantee protects **one patient-hour window**, not a patient. A patient contributes on average about 38 windows (up to about 336), and consecutive windows share 23 of 24 hours of data. By group privacy the patient-level guarantee for k windows degrades to about kε (with a much worse δ), so "ε = 3" would be meaningless for a patient. Presenting it as patient protection would violate rule 3 (exact privacy claims).

**Proposed fix:** make the **patient** the unit of privacy. The DP dataset element is a patient. Opacus Poisson-samples *patients* with rate q; each sampled patient contributes **one** window chosen uniformly at random (randomness independent of other patients) from its own stay, and that single per-sample gradient is clipped to C. Every patient contributes at most one clipped gradient of norm ≤ C per step. The standard subsampled-Gaussian RDP analysis therefore holds under add/remove-one-*patient* adjacency. For each fixed draw of the window-selection randomness the bound holds, and Rényi divergence is jointly quasi-convex, so it also holds for the mixture. Then δ < 1/n_i uses n_i = number of training *patients*.

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

The spec adds `HospAdmTime` (clipped, scaled) to the starter's Age/Gender/ICULOS. Missing Gender or Age is imputed to 0 after scaling, with a mask channel. Age uses a fixed affine scale ((age − 60)/20) that is data-independent, so no leakage.

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
- The repo is inside OneDrive, so the virtualenv lives outside it (`C:\Users\Shrikar\.venvs\fedguard`). `data/` and `runs/` can be relocated with `FEDGUARD_DATA_DIR` / `FEDGUARD_RUNS_DIR` (see Q2 in `PROGRESS.md`).

## D12. Library versions (Adopted)

Built against torch 2.14.0+cu130, opacus 1.6.0, flwr 1.38.0 (Message API: `ServerApp`/`ClientApp`, `flwr.serverapp.strategy.FedAvg/FedProx`), captum 0.9.0, streamlit 1.64.0, and Python 3.11. Exact pins are in `pyproject.toml`.
