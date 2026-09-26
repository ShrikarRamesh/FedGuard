# Privacy guarantee of FedGuard

This page states exactly what FedGuard's differential privacy protects, under which assumptions, and what it does **not** protect. Numbers quoted elsewhere (ε per client) are always **total** privacy loss over all of a client's training, at the stated δ, from the Opacus RDP accountant.

## Setting

- K clients (hospital × ICU unit). Client *i* holds a dataset D_i of **patients**; every patient belongs to exactly one client (`splits.csv`, D4).
- The **unit of privacy is the patient** (D2). Two datasets are neighbouring if one is obtained from the other by adding or removing one patient's entire ICU stay.
- Each client trains locally with **DP-SGD** (Opacus 1.6.0):
  - **Poisson sampling of patients** with rate q_i = 1/⌈n_i/B⌉ (n_i = training patients, B = logical batch; exactly Opacus' `DPDataLoader` definition).
  - Each sampled patient contributes **one window** at a uniformly random hour of its own stay. The randomness is independent of every other patient.
  - **Per-sample clipping** to L2 norm C (C = 1).
  - **Gaussian noise** N(0, σ_i² C² I) added to the sum of clipped gradients.
  - An optimiser step (AdamW) on the privatised gradient.
- The client's noise multiplier σ_i is calibrated **before training** so that the RDP bound after its maximum planned number of steps, T_i = E · R_max · ⌈n_i/B⌉ (E patient-epochs per participation, R_max participations), converts to ε_i at δ_i = 10⁻⁵. `privacy/accounting.py` asserts δ_i < 1/n_i. The accountant is updated after every step. A client stops participating after R_max participations or when the accountant reports ε ≥ ε_i, whichever comes first.
- Budgets (`privacy/budgets.py`):
  - uniform: ε_i = ε;
  - **adaptive (FedGuard)**: ε_i = ε · (a + (1 − a) · n_i / n_max), a = 0.55;
  - inverse: ε_i = ε · (a + (1 − a) · n_min / n_i), an ablation;
  - equal-noise: every client uses the σ the uniform rule gives the largest client; its ε_i is whatever the accountant then reports. Also an ablation.
- **Input preprocessing is data-independent** in DP runs (D3):
  - values are clipped to fixed physiological ranges and standardised with fixed clinical reference centres/scales (`privacy/public_norm.py`), not with the client's own statistics;
  - the positive-class weight is a fixed hyperparameter (10), not the client's prevalence;
  - measurement masks, time-since-measured and static features use fixed transforms.

## Proposition

*Assume an honest-but-curious server and honest clients, with no secure aggregation. Then for every client i, the entire sequence of model updates that client i sends during training, over any number of rounds and any interleaving with other clients (synchronous or asynchronous, with any staleness), is (ε_i, δ_i)-differentially private with respect to adding or removing one patient of D_i. Consequently every patient, whose record lives in exactly one client's dataset, is protected at level (ε_i, δ_i) of its own hospital, against anything computed from the released updates: the global models, the aggregation weights, the validation curves and the deployed model.*

**Proof sketch.**
1. *One step.* Fix all randomness other than client i's DP-SGD, including the random hour chosen for each patient. Each sampled patient contributes a single gradient of norm ≤ C, so adding or removing one patient changes the sum of clipped gradients by at most C in L2. With Poisson sampling at rate q_i and Gaussian noise σ_i C, one step is the subsampled Gaussian mechanism. Its Rényi-DP curve ε(α) is the one Opacus' `RDPAccountant` computes (Mironov et al., 2019).
2. *Random window choice.* Each patient's gradient is a randomised function of that patient's record only, with randomness independent of other patients. For every fixed draw of the choices the bound in step 1 holds. Rényi divergence is jointly quasi-convex, so the bound also holds for the mixture over draws.
3. *Composition.* The model a client trains from at each step depends on its own previous privatised outputs and on other clients' updates. Other clients' updates do not depend on D_i, so relative to D_i they are fixed auxiliary inputs. The steps therefore form an adaptive composition, and RDP composes additively over the ≤ T_i steps. The accountant converts the composed RDP to (ε_i, δ_i) at the tracked number of steps. The client never exceeds T_i steps, and it stops when the reported ε reaches ε_i.
4. *Post-processing.* AdamW's update, the server's FedAvg/FedProx averaging or asynchronous mixing, staleness weights and model selection on validation are all functions of the released updates (plus data the client never touches). By post-processing immunity they cannot increase privacy loss. That is why asynchrony and staleness do not affect the guarantee: only local DP-SGD steps touch patient data.
5. *Across clients (parallel composition).* D_1, …, D_K are disjoint in patients. Changing one patient changes exactly one D_i, so the joint mechanism is (max over the affected client's ε_i, δ_i)-DP for that patient, with no summation over clients.

**Data-independent preprocessing matters for step 1.** If client i standardised with its own training mean and standard deviation, removing one patient would shift every other patient's inputs. The one-patient sensitivity argument would then fail. FedGuard's DP runs therefore use fixed public reference normalisation (D3). Non-DP runs use per-client training statistics as the specification requires; they make no privacy claim.

## Assumptions

- Honest-but-curious server: it follows the protocol but may inspect every update it receives. That is the threat model of page 2 of the app, the gradient-inversion attack.
- Clients run the protocol faithfully: correct clipping, noise from a proper RNG, and stopping at the budget. The experiments use Opacus' non-cryptographic RNG (`secure_mode=False`); a deployment should enable `secure_mode`.
- No secure aggregation. The server sees each client's update individually. The guarantee above does not rely on hiding updates.
- The RDP accountant's bound is an upper bound. For some very small test configurations Opacus warns that the optimal Rényi order was at the edge of its grid, which only means the reported ε could be tighter. It never under-reports.

## What is NOT protected

- **Hospital-level membership and participation.** The fact that a hospital takes part, when it sends updates, and how many (R_max, dropouts) are visible to the server.
- **Dataset size n_i.** It enters the aggregation weights (FedAvg weights n_i by training windows; async mixing uses n_i/N) and the adaptive budget rule (training patients). These counts are sent in clear. A curious server learns them exactly, which is disclosure at the level of each hospital, not of a patient.
- **Per-hospital distribution shifts** encoded in the model, such as a site's measurement practice. The masks make sites distinguishable (see docs/data.md). DP bounds the influence of one patient, not of a whole hospital.
- **Hyperparameter selection.** DP-SGD settings (batch size, epochs per round, learning rate; D24) and the async mixing parameters (D20) were chosen on *validation* data of the same clients. That selection is not included in ε (a standard caveat; see Papernot & Steinke, 2022). The reported ε covers training given those hyperparameters.
- **Evaluation.** Validation and test metrics are computed centrally for the research report. In a deployment each hospital would evaluate locally. The Flower deployment already does this; its server holds no patient data.
- **Non-DP runs** (Local-only, FedAvg, FedProx, Centralized, async without DP) have **no** formal guarantee. Page 2 of the app shows what that can mean in practice.

## Reporting rules used in this project

- ε is always the **total** over all training for each client, at δ = 10⁻⁵, taken from the accountant after training (`metrics.json → fl.clients.<name>.eps`). The planned value (`fl.budgets.<name>.planned_eps`) is logged next to it; in all runs, spent ≤ planned ≤ target.
- ε "per round" is never reported as the guarantee.
