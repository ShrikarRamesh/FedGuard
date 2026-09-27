# Proof of compute

_Generated 2026-09-27 12:45 by `fedguard proof` from files on disk only (see the source list in `src/fedguard/eval/proof.py`). Nothing in this document is estimated or typed in by hand, except the section explicitly labelled as manual GPU readings. Regenerate with `fedguard proof`._

## Summary

- Jobs recorded (excluding 95 fast-mode smoke-test jobs): **89** (86 GPU jobs, 3 CPU-only sklearn baselines), from 2026-09-26 17:46 to 2026-09-27 12:45.
- **GPU job-hours: 49.7 h.** This is the sum of the wall-clock durations of all GPU jobs, including failed, killed and superseded ones. Jobs that shared the GPU are each counted in full.
- **GPU busy wall-clock: 18.8 h.** This is the union of GPU job intervals, i.e. time during which at least one job was running.
- Maximum number of GPU jobs running at the same time: **4**.
- GPU jobs with **unknown duration** (excluded from the totals above): 1 (attack -e fedavg --seed 0 --n 30 --iters 300 --restarts 2, started 2026-09-27 12:00).
- GPU jobs by status: done: 49 (24.1 h), superseded (done): 15 (6.2 h), killed/failed: 12 (11.5 h), superseded (killed/failed): 5 (0.6 h), running: 2 (3.2 h), no success marker: 2 (0.0 h), timeout: 1 (4.0 h).
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU.

Failed, killed, timed-out and superseded runs are kept on purpose. They document the bugs and operational problems found and fixed during the project (docs/decisions.md D19–D36): an optimizer-state bug that invalidated the first DP results, hung jobs, a watchdog false positive, GPU memory overflow with 3–4 concurrent jobs, and a LightGBM early-stopping bug.

## What this does and does not capture

- **Captured:** every job that created a run directory (all training and FL runs, including killed, timed-out and archived ones), and every attack / explain / MC-ablation / mc-predict / alerts job launched through `scripts/run_experiments.py`, via its log.
- **Not captured:** commands launched directly from the CLI outside the runner that create no run directory. Examples are direct `fedguard alerts` / `mc-predict` calls and two short gradient-inversion feasibility checks. Their compute is missing from the totals, so the totals are lower bounds.
- **End time of runs that never finished** is their last file write, i.e. when they stopped making progress. A hung job may have held the GPU until it was killed later, so those durations are lower bounds too.
- Fast-mode smoke-test jobs are excluded from the totals.

## Timeline

![compute timeline](figures/compute_timeline.png)

## Peak GPU memory

### From automated logs

Automated samples: 10 (nvidia-smi every 30 s) from 2026-09-27T12:40:15 to 2026-09-27T12:44:52.

- **Peak dedicated GPU memory in the automated log: 5288 / 6141 MiB** at 2026-09-27T12:44:52 (2 FedGuard job(s) running, utilisation 100%).
- Mean memory used: 5256 MiB; mean utilisation: 100%.
- Samples by number of concurrent FedGuard jobs (max memory MiB): 2 job(s): 5288

- No run has recorded its own peak allocation yet (added to summary.json from 2026-09-27; later runs will).

### Manual readings (not logged)

These readings were taken by hand with `nvidia-smi` during the session, **before automated GPU logging existed**. They are real observations but are not backed by a log file; they are listed separately for that reason.

| time | memory used / total (MiB) | concurrent jobs | context |
|---|---|---|---|
| 2026-09-27 06:38 | 5898 / 6141 | 4 | gradient-inversion attack + FedAvg+DP seed 0 + FedGuard seed 0 + 2-node async ablation; near-OOM, the ablation job was stopped (5518 MiB after) |
| 2026-09-27 12:33 | 5441 / 6141 | 3 | FedGuard seed 0 rerun + FedAvg+DP seed 2 + batched attack, utilisation 100%; FedAvg+DP seed 2 had made 5/40 rounds in ~3 h; attack stopped (5209 MiB after) |
| 2026-09-27 12:40 | 5145 / 6141 | 2 | two FedGuard patient-level DP jobs only (seeds 0 and 1; logical batch 1024, physical 256); basis for the 2-job cap (D36) |

## Every run

| run                                                       |   seed | config                                                            | start            | end              | end basis                                                      | wall clock (h)   | status                     | device                             | test AUROC   | test AUPRC   |
|:----------------------------------------------------------|-------:|:------------------------------------------------------------------|:-----------------|:-----------------|:---------------------------------------------------------------|:-----------------|:---------------------------|:-----------------------------------|:-------------|:-------------|
| centralized_patchtst                                      |      0 | patchtst                                                          | 2026-09-26 17:46 | 2026-09-26 17:59 | DONE marker                                                    | 0.21             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.8225       | 0.111        |
| tune_dp                                                   |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 18:08 | 2026-09-26 18:30 | DONE marker                                                    | 0.37             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6284       | 0.0237       |
| tune_async                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 18:16 | 2026-09-26 19:15 | DONE marker                                                    | 0.97             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7423       | 0.069        |
| local_patchtst/A_MICU                                     |      0 | patchtst                                                          | 2026-09-26 18:16 | 2026-09-26 18:21 | DONE marker                                                    | 0.08             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7228       | 0.0523       |
| local_patchtst/A_SICU                                     |      0 | patchtst                                                          | 2026-09-26 18:21 | 2026-09-26 18:30 | DONE marker                                                    | 0.14             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6937       | 0.0478       |
| local_patchtst/B_MICU                                     |      0 | patchtst                                                          | 2026-09-26 18:30 | 2026-09-26 18:37 | DONE marker                                                    | 0.13             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7216       | 0.0583       |
| tune_dp                                                   |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 18:34 | 2026-09-26 18:59 | DONE marker                                                    | 0.42             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6355       | 0.0276       |
| local_patchtst/B_SICU                                     |      0 | patchtst                                                          | 2026-09-26 18:37 | 2026-09-26 18:44 | DONE marker                                                    | 0.12             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.708        | 0.0555       |
| fedavg                                                    |      0 | fedavg/sync, patchtst                                             | 2026-09-26 18:44 | 2026-09-26 19:19 | last file write                                                | 0.58             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| tune_dp                                                   |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 19:00 | 2026-09-26 19:28 | DONE marker                                                    | 0.47             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.635        | 0.0274       |
| tune_dp                                                   |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 19:05 | 2026-09-26 19:57 | DONE marker                                                    | 0.86             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6443       | 0.0261       |
| tune_async                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 19:15 | 2026-09-26 19:19 | last file write                                                | 0.06             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| tune_async                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 19:20 | 2026-09-26 20:18 | DONE marker                                                    | 0.98             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7593       | 0.0813       |
| tune_dp                                                   |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 19:28 | 2026-09-26 19:29 | last file write                                                | 0.01             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| tune_async                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 19:40 | 2026-09-26 19:46 | last file write                                                | 0.11             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| tune_dp                                                   |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 19:57 | 2026-09-26 20:12 | DONE marker                                                    | 0.25             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.621        | 0.0258       |
| tune_dp_small                                             |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 20:12 | 2026-09-26 20:23 | DONE marker                                                    | 0.19             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.512        | 0.0169       |
| tune_async                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 20:19 | 2026-09-26 20:58 | DONE marker                                                    | 0.65             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7655       | 0.0834       |
| tune_async                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 20:24 | 2026-09-26 21:03 | DONE marker                                                    | 0.66             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7538       | 0.0763       |
| _superseded/pre_optimizer_reset/fedguard                  |      0 | fedguard_async/async, DP eps=3.0 adaptive                         | 2026-09-26 20:59 | 2026-09-26 21:26 | DONE marker                                                    | 0.45             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6366       | 0.0254       |
| async_nodp                                                |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 21:03 | 2026-09-26 21:52 | DONE marker                                                    | 0.81             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7655       | 0.0834       |
| _superseded/pre_optimizer_reset/fedguard                  |      1 | fedguard_async/async, DP eps=3.0 adaptive                         | 2026-09-26 21:26 | 2026-09-26 21:56 | DONE marker                                                    | 0.5              | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6172       | 0.0222       |
| fedprox                                                   |      0 | fedprox/sync, patchtst                                            | 2026-09-26 21:36 | 2026-09-26 22:54 | DONE marker                                                    | 1.29             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7851       | 0.1085       |
| async_nodp                                                |      1 | fedguard_async/async, patchtst                                    | 2026-09-26 21:52 | 2026-09-26 22:44 | DONE marker                                                    | 0.86             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7696       | 0.0819       |
| _superseded/pre_optimizer_reset/fedguard                  |      2 | fedguard_async/async, DP eps=3.0 adaptive                         | 2026-09-26 21:56 | 2026-09-26 22:27 | DONE marker                                                    | 0.51             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6133       | 0.023        |
| async_nodp                                                |      2 | fedguard_async/async, patchtst                                    | 2026-09-26 22:27 | 2026-09-26 23:13 | DONE marker                                                    | 0.76             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7588       | 0.0866       |
| fedavg                                                    |      0 | fedavg/sync, patchtst                                             | 2026-09-26 22:44 | 2026-09-26 23:34 | DONE marker                                                    | 0.83             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7842       | 0.1          |
| _superseded/pre_optimizer_reset/fedavg_dp                 |      0 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-26 22:54 | 2026-09-26 23:16 | DONE marker                                                    | 0.37             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6355       | 0.0276       |
| sweep_nodp_public                                         |      0 | fedguard_async/async, patchtst                                    | 2026-09-26 23:13 | 2026-09-27 00:14 | DONE marker                                                    | 1.01             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7528       | 0.0774       |
| centralized_patchtst                                      |      1 | patchtst                                                          | 2026-09-26 23:16 | 2026-09-26 23:44 | DONE marker                                                    | 0.47             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.8226       | 0.1046       |
| _superseded/pre_optimizer_reset/sweep_adaptive            |      0 | fedguard_async/async, DP eps=1.0 adaptive                         | 2026-09-26 23:34 | 2026-09-27 00:00 | DONE marker                                                    | 0.43             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6462       | 0.0277       |
| local_patchtst/A_MICU                                     |      1 | patchtst                                                          | 2026-09-26 23:44 | 2026-09-26 23:51 | DONE marker                                                    | 0.12             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7132       | 0.0522       |
| local_patchtst/A_SICU                                     |      1 | patchtst                                                          | 2026-09-26 23:51 | 2026-09-26 23:57 | DONE marker                                                    | 0.1              | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7126       | 0.0561       |
| local_patchtst/B_MICU                                     |      1 | patchtst                                                          | 2026-09-26 23:57 | 2026-09-27 00:08 | DONE marker                                                    | 0.19             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6865       | 0.0521       |
| _superseded/pre_optimizer_reset/sweep_uniform             |      0 | fedguard_async/async, DP eps=1.0 uniform                          | 2026-09-27 00:01 | 2026-09-27 00:24 | last file write                                                | 0.39             | superseded (killed/failed) | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| local_patchtst/B_SICU                                     |      1 | patchtst                                                          | 2026-09-27 00:08 | 2026-09-27 00:17 | DONE marker                                                    | 0.14             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6939       | 0.0609       |
| _superseded/pre_optimizer_reset/sweep_adaptive            |      0 | fedguard_async/async, DP eps=2.0 adaptive                         | 2026-09-27 00:14 | 2026-09-27 00:38 | DONE marker                                                    | 0.41             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6416       | 0.0266       |
| fedavg                                                    |      1 | fedavg/sync, patchtst                                             | 2026-09-27 00:17 | 2026-09-27 00:57 | DONE marker                                                    | 0.68             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7711       | 0.1027       |
| _superseded/pre_optimizer_reset/sweep_uniform             |      0 | fedguard_async/async, DP eps=2.0 uniform                          | 2026-09-27 00:38 | 2026-09-27 00:56 | DONE marker                                                    | 0.3              | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6418       | 0.0264       |
| _superseded/pre_optimizer_reset/sweep_uniform             |      0 | fedguard_async/async, DP eps=3.0 uniform                          | 2026-09-27 00:56 | 2026-09-27 01:13 | DONE marker                                                    | 0.28             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6357       | 0.0253       |
| fedprox                                                   |      1 | fedprox/sync, patchtst                                            | 2026-09-27 00:58 | 2026-09-27 01:57 | DONE marker                                                    | 1.0              | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7745       | 0.1071       |
| _superseded/pre_optimizer_reset/sweep_adaptive            |      0 | fedguard_async/async, DP eps=5.0 adaptive                         | 2026-09-27 01:13 | 2026-09-27 01:31 | DONE marker                                                    | 0.29             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.621        | 0.024        |
| _superseded/pre_optimizer_reset/sweep_uniform             |      0 | fedguard_async/async, DP eps=5.0 uniform                          | 2026-09-27 01:31 | 2026-09-27 01:49 | DONE marker                                                    | 0.3              | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6194       | 0.0239       |
| _superseded/pre_optimizer_reset/sweep_adaptive            |      0 | fedguard_async/async, DP eps=8.0 adaptive                         | 2026-09-27 01:49 | 2026-09-27 02:12 | DONE marker                                                    | 0.39             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6102       | 0.023        |
| _superseded/pre_optimizer_reset/fedavg_dp                 |      1 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-27 01:58 | 2026-09-27 02:37 | DONE marker                                                    | 0.66             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.5982       | 0.0209       |
| _superseded/pre_optimizer_reset/sweep_uniform             |      0 | fedguard_async/async, DP eps=8.0 uniform                          | 2026-09-27 02:13 | 2026-09-27 02:53 | DONE marker                                                    | 0.67             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6084       | 0.0228       |
| centralized_patchtst                                      |      2 | patchtst                                                          | 2026-09-27 02:38 | 2026-09-27 02:56 | DONE marker                                                    | 0.3              | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.8028       | 0.1016       |
| sweep_nodp_public                                         |      1 | fedguard_async/async, patchtst                                    | 2026-09-27 02:53 | 2026-09-27 03:23 | DONE marker                                                    | 0.51             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7615       | 0.0719       |
| local_patchtst/A_MICU                                     |      2 | patchtst                                                          | 2026-09-27 02:56 | 2026-09-27 03:00 | DONE marker                                                    | 0.07             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7425       | 0.0646       |
| local_patchtst/A_SICU                                     |      2 | patchtst                                                          | 2026-09-27 03:00 | 2026-09-27 03:04 | DONE marker                                                    | 0.07             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6943       | 0.0527       |
| local_patchtst/B_MICU                                     |      2 | patchtst                                                          | 2026-09-27 03:04 | 2026-09-27 03:07 | DONE marker                                                    | 0.05             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.72         | 0.0508       |
| local_patchtst/B_SICU                                     |      2 | patchtst                                                          | 2026-09-27 03:07 | 2026-09-27 03:11 | DONE marker                                                    | 0.07             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7255       | 0.0554       |
| fedavg                                                    |      2 | fedavg/sync, patchtst                                             | 2026-09-27 03:11 | 2026-09-27 03:43 | DONE marker                                                    | 0.54             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7813       | 0.1038       |
| _superseded/pre_optimizer_reset/sweep_adaptive            |      1 | fedguard_async/async, DP eps=1.0 adaptive                         | 2026-09-27 03:24 | 2026-09-27 03:41 | DONE marker                                                    | 0.3              | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6125       | 0.0228       |
| _superseded/pre_optimizer_reset/sweep_uniform             |      1 | fedguard_async/async, DP eps=1.0 uniform                          | 2026-09-27 03:41 | 2026-09-27 03:43 | last file write                                                | 0.03             | superseded (killed/failed) | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| _superseded/pre_optimizer_reset/sweep_adaptive            |      1 | fedguard_async/async, DP eps=2.0 adaptive                         | 2026-09-27 03:43 | 2026-09-27 03:43 | last file write                                                | 0.0              | superseded (killed/failed) | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedprox                                                   |      2 | fedprox/sync, patchtst                                            | 2026-09-27 03:43 | 2026-09-27 04:56 | DONE marker                                                    | 1.21             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7839       | 0.1082       |
| attack -e fedavg --seed 0                                 |      0 |                                                                   | 2026-09-27 03:44 | 2026-09-27 07:44 | log file mtime                                                 | 4.0              | timeout                    | GPU (runner job)                   |              |              |
| _superseded/pre_optimizer_reset/sweep_adaptive_ext        |      0 | fedguard_async/async, DP eps=16.0 adaptive                        | 2026-09-27 03:44 | 2026-09-27 04:05 | DONE marker                                                    | 0.35             | superseded (done)          | NVIDIA GeForce RTX 4050 Laptop GPU | 0.5845       | 0.0214       |
| _superseded/pre_optimizer_reset/sweep_adaptive_ext        |      0 | fedguard_async/async, DP eps=32.0 adaptive                        | 2026-09-27 04:05 | 2026-09-27 04:06 | last file write                                                | 0.01             | superseded (killed/failed) | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| dp_diag                                                   |      0 | fedguard_async/async, DP diag sigma=0.0 (no guarantee)            | 2026-09-27 04:07 | 2026-09-27 04:28 | DONE marker                                                    | 0.34             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7229       | 0.053        |
| dp_diag                                                   |      0 | fedguard_async/async, DP diag sigma=0.0 (no guarantee)            | 2026-09-27 04:28 | 2026-09-27 04:48 | DONE marker                                                    | 0.34             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6657       | 0.0351       |
| dp_diag                                                   |      0 | fedguard_async/async, DP eps=16.0 adaptive                        | 2026-09-27 04:48 | 2026-09-27 05:20 | DONE marker                                                    | 0.53             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6286       | 0.0275       |
| _superseded/pre_optimizer_reset/fedavg_dp                 |      2 | fedavg/sync, DP eps=3.0 uniform                                   | 2026-09-27 04:56 | 2026-09-27 05:07 | last file write                                                | 0.19             | superseded (killed/failed) | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| centralized_lr                                            |      0 | lr                                                                | 2026-09-27 05:08 | 2026-09-27 05:09 | DONE marker                                                    | 0.02             | done                       | CPU (sklearn)                      | 0.7984       | 0.0828       |
| _superseded/centralized_lgbm_bug_best_iter1               |      0 | lgbm                                                              | 2026-09-27 05:09 | 2026-09-27 05:10 | DONE marker                                                    | 0.02             | superseded (done)          | CPU (sklearn)                      | 0.7216       | 0.0477       |
| centralized_gru                                           |      0 | gru                                                               | 2026-09-27 05:10 | 2026-09-27 05:18 | DONE marker                                                    | 0.14             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.8204       | 0.1025       |
| centralized_lgbm                                          |      0 | lgbm                                                              | 2026-09-27 05:11 | 2026-09-27 05:13 | DONE marker                                                    | 0.03             | done                       | CPU (sklearn)                      | 0.8166       | 0.0923       |
| centralized_gru                                           |      1 | gru                                                               | 2026-09-27 05:19 | 2026-09-27 05:24 | DONE marker                                                    | 0.09             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.8246       | 0.1128       |
| dp_diag                                                   |      0 | fedguard_async/async, DP diag sigma=0.0 (no guarantee), opt-reset | 2026-09-27 05:20 | 2026-09-27 05:42 | DONE marker                                                    | 0.36             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7567       | 0.0537       |
| centralized_gru                                           |      2 | gru                                                               | 2026-09-27 05:24 | 2026-09-27 05:30 | DONE marker                                                    | 0.1              | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.8103       | 0.0984       |
| abl_nodes_hospital_fedavg                                 |      0 | fedavg/sync, patchtst, partition=hospital                         | 2026-09-27 05:30 | 2026-09-27 05:55 | DONE marker                                                    | 0.42             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7955       | 0.0978       |
| dp_diag                                                   |      0 | fedguard_async/async, DP diag sigma=0.0 (no guarantee)            | 2026-09-27 05:42 | 2026-09-27 06:03 | DONE marker                                                    | 0.35             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.7475       | 0.055        |
| abl_nodes_hospital_fedguard_async_nodp                    |      0 | fedguard_async/async, patchtst, partition=hospital                | 2026-09-27 05:55 | 2026-09-27 06:38 | last file write                                                | 0.72             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedguard                                                  |      0 | fedguard_async/async, DP eps=3.0 adaptive, opt-reset              | 2026-09-27 06:03 | 2026-09-27 07:00 | last file write                                                | 0.96             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedavg_dp                                                 |      0 | fedavg/sync, DP eps=3.0 uniform, opt-reset                        | 2026-09-27 06:03 | 2026-09-27 07:15 | DONE marker                                                    | 1.21             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6355       | 0.0276       |
| fedavg_dp                                                 |      1 | fedavg/sync, DP eps=3.0 uniform, opt-reset                        | 2026-09-27 07:16 | 2026-09-27 09:41 | DONE marker                                                    | 2.43             | done                       | NVIDIA GeForce RTX 4050 Laptop GPU | 0.6034       | 0.0212       |
| explain -e async_nodp --seeds 0                           |      0 |                                                                   | 2026-09-27 07:44 | 2026-09-27 07:44 | log file mtime                                                 | 0.01             | done                       | GPU (runner job)                   |              |              |
| explain -e fedguard --seeds 0                             |      0 |                                                                   | 2026-09-27 07:44 | 2026-09-27 07:44 | log file mtime                                                 | 0.0              | no success marker          | GPU (runner job)                   |              |              |
| mc-ablation -e async_nodp --seeds 0                       |      0 |                                                                   | 2026-09-27 07:44 | 2026-09-27 07:59 | log file mtime                                                 | 0.25             | done                       | GPU (runner job)                   |              |              |
| abl_nodes_dirichlet_fedavg                                |      0 | fedavg/sync, patchtst, partition=dirichlet                        | 2026-09-27 07:50 | 2026-09-27 11:49 | last file write                                                | 3.99             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedguard                                                  |      1 | fedguard_async/async, DP eps=3.0 adaptive, opt-reset              | 2026-09-27 07:59 | 2026-09-27 10:51 | last file write                                                | 2.87             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedavg_dp                                                 |      2 | fedavg/sync, DP eps=3.0 uniform, opt-reset                        | 2026-09-27 09:41 | 2026-09-27 11:55 | last file write                                                | 2.22             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedguard                                                  |      0 | fedguard_async/async, DP eps=3.0 adaptive, opt-reset              | 2026-09-27 09:41 | 2026-09-27 12:45 | still running (now)                                            | 3.06             | running                    | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| attack -e fedavg --seed 0 --n 30 --iters 300 --restarts 2 |      0 |                                                                   | 2026-09-27 12:00 | 2026-09-27 12:00 | unknown (log holds only the command; killed before any output) | unknown          | no success marker          | GPU (runner job)                   |              |              |
| fedguard                                                  |      1 | fedguard_async/async, DP eps=3.0 adaptive, opt-reset              | 2026-09-27 12:34 | 2026-09-27 12:45 | still running (now)                                            | 0.18             | running                    | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| abl_nodes_dirichlet_fedguard_async_nodp                   |      0 | fedguard_async/async, patchtst, partition=dirichlet               | 2026-09-27 12:34 | 2026-09-27 12:35 | last file write                                                | 0.01             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| fedavg_dp                                                 |      2 | fedavg/sync, DP eps=3.0 uniform, opt-reset                        | 2026-09-27 12:34 | 2026-09-27 12:35 | last file write                                                | 0.01             | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |
| abl_nodes_hospital_fedguard_async_nodp                    |      0 | fedguard_async/async, patchtst, partition=hospital                | 2026-09-27 12:37 | 2026-09-27 12:37 | last file write                                                | 0.0              | killed/failed              | NVIDIA GeForce RTX 4050 Laptop GPU |              |              |

## Commit timeline (`git log --stat`)

```
commit 248ccbb89e0eaf9e5cc76ad295765f6cb159eb05
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-27 12:42:41 +0530

    Proof-of-compute appendix (fedguard proof), GPU memory logger, per-run peak GPU memory
    
    docs/proof_of_compute.md + docs/figures/compute_timeline.png generated only from run dirs, runner logs, the automated nvidia-smi log and git log; manual nvidia-smi readings kept in a separate, labelled file. Snapshot at 2026-09-27 12:4x: 89 jobs, 49.6 GPU job-hours, 18.8 h GPU busy wall-clock, max 4 concurrent jobs.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 docs/decisions.md                 |  10 +
 docs/figures/compute_timeline.png | Bin 0 -> 290602 bytes
 docs/manual_gpu_observations.json |  23 ++
 docs/proof_of_compute.md          | 468 ++++++++++++++++++++++++++++++++++++++
 scripts/gpu_logger.py             |  57 +++++
 scripts/run_experiments.py        |   2 +-
 src/fedguard/cli.py               |  13 ++
 src/fedguard/eval/proof.py        | 396 ++++++++++++++++++++++++++++++++
 src/fedguard/utils/runs.py        |   9 +-
 9 files changed, 976 insertions(+), 2 deletions(-)

commit 25b4da5da72f080d129cc731cec3edf7cdb6b398
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-27 10:36:48 +0530

    Batched gradient-inversion attack, inference heartbeat for the watchdog, atomic GPU slot lock
    
    D35: attack windows inverted in batches via torch.func vmap(grad), tested equal to the per-window loop (~18x faster); clip-only condition dropped. Inference loops touch a heartbeat file so the watchdog does not kill the silent MC-Dropout phase. Runner: exclusive lock around count-then-start of GPU jobs.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 docs/decisions.md                         |    16 +-
 results/attack/attack.json                | 17260 ++++++++++++++++++++++++++++
 scripts/run_experiments.py                |    56 +-
 scripts/watchdog.py                       |     2 +-
 src/fedguard/attack/gradient_inversion.py |    75 +-
 src/fedguard/attack/run.py                |     7 +-
 src/fedguard/fl/runner.py                 |     1 +
 src/fedguard/train/loops.py               |    16 +-
 src/fedguard/train/runner.py              |     1 +
 tests/test_attack.py                      |    42 +
 10 files changed, 17450 insertions(+), 26 deletions(-)

commit 39e2b98056266e4613419c5d2b3ed4a3030bd166
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-27 05:44:25 +0530

    Fix DP training: reset AdamW state each participation (D34); rerun all DP experiments
    
    sigma=0 controls isolated the cause of the poor/degrading DP results: DP clients carried AdamW
    moments across rounds while loading a different global model each round. With a per-participation
    reset the no-noise DP pipeline reaches 0.757 test AUROC (non-DP reference 0.753) vs 0.723 before.
    Pre-fix DP runs archived under runs/_superseded/pre_optimizer_reset and excluded from reports.
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 CLAUDE.md                    |  1 +
 configs/privacy/uniform.yaml |  1 +
 docs/decisions.md            | 20 ++++++++++++++++++++
 tests/test_privacy.py        | 18 ++++++++++++++++++
 4 files changed, 40 insertions(+)

commit b50e975af2df3ddbac205e3f38e4233bafbbf1df
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-27 05:13:23 +0530

    DP recipe diagnosis tooling, watchdog, LightGBM early-stopping fix
    
    - D31-D32: privacy sweep degrades as eps grows; sigma=0 diagnostic override (no guarantee, eps=inf),
      optional per-round optimizer reset; sweep paused pending diagnosis
    - D33: LightGBM stopped after 1 tree (early stopping watched binary_logloss); fixed, AUROC 0.817
    - scripts/watchdog.py kills jobs with no progress for 45 min (pid in meta.json); runner job timeout
    - export --alerts-from, mc-predict, posthoc/sweep_ext/dp_diag stages
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 docs/decisions.md                    | 53 +++++++++++++++++++++++++++++
 scripts/run_experiments.py           | 34 +++++++++++++++++--
 scripts/watchdog.py                  | 65 ++++++++++++++++++++++++++++++++++++
 src/fedguard/fl/client.py            |  6 ++++
 src/fedguard/fl/engine.py            | 27 ++++++++++-----
 src/fedguard/train/sklearn_models.py |  3 +-
 src/fedguard/utils/runs.py           |  2 ++
 7 files changed, 179 insertions(+), 11 deletions(-)

commit 2b45bf6446cdeb6a7cc5d9cd75b7f605749544f6
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-26 22:45:00 +0530

    DP selection closed, alpha0 chosen, alert grid fix, post-hoc MC predictions, job cap + timeout
    
    - D26-D28: gradient-norm analysis, k-window DP (torch.func, tested vs Opacus), small-model run; chosen DP-SGD config
    - D20: async alpha0 = 4.0 on validation; D29: alert risk grid includes validation-risk quantiles
    - D30: alerts/bedside from async FL without DP (DP models cannot alert usefully at eps=3)
    - fedguard mc-predict; experiment runner: machine-wide GPU job cap, per-job timeout
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 Makefile                            |   8 +--
 README.md                           | 124 ++++++++++++++++++++++++++++++++----
 app/pages/3_At_the_bedside.py       |   2 +
 app/pages/4_Results.py              |   4 ++
 configs/fl/fedguard_async.yaml      |   2 +-
 configs/privacy/uniform.yaml        |   8 ++-
 docs/decisions.md                   |  81 ++++++++++++++++++++++-
 pyproject.toml                      |   1 +
 scripts/deploy/xcheck.ps1           |  15 +++++
 scripts/make.ps1                    |   8 +--
 scripts/run_experiments.py          |  72 ++++++++++++++++++---
 scripts/smoke.py                    |  59 +++++++++++++++++
 src/fedguard/alerts/tuning.py       |  13 +++-
 src/fedguard/cli.py                 |  42 +++++++++++-
 src/fedguard/export/results_json.py |  34 ++++++++--
 src/fedguard/fl/client.py           |  33 +++++++++-
 src/fedguard/fl/engine.py           |   6 +-
 src/fedguard/privacy/dp.py          | 122 +++++++++++++++++++++++++++++++++++
 tests/test_alerts.py                |   9 +++
 tests/test_app.py                   |  15 +++++
 tests/test_export.py                |   8 ++-
 tests/test_privacy.py               |  73 +++++++++++++++++++++
 22 files changed, 688 insertions(+), 51 deletions(-)

commit fa9f3c4edebcb0d0e33ff07e6bfb044df9126b8b
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-26 18:56:35 +0530

    M2-M9 code: models, training, FL engine + Flower app, patient-level DP, alerts, explanations, attack, report, export, Streamlit app
    
    - own Opacus-safe PatchTST (+GRU/LR/LightGBM baselines), MC Dropout; vendored official PhysioNet 2019 utility (BSD-2)
    - event-driven simulated-clock FL engine (sync FedAvg/FedProx, async staleness mixing), offline clients, events.jsonl
    - Flower 1.38 ServerApp/ClientApp on the Deployment Engine; per-client losses match in-house FedAvg exactly
    - patient-level DP-SGD (Opacus RDP, Poisson over patients), budget rules, public normalisation for DP (D3)
    - alert episodes/tuning on validation, calibration, IG + attention rollout, gradient-inversion attack
    - resumable experiment queue runner, fedguard report/export, results.json schema, 4-page Streamlit app
    - docs: decisions D15-D24, privacy.md, demo.md; 86 tests
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 .gitignore                                        |   2 +-
 LICENSES/physionet-evaluation-2019.txt            |  25 ++
 app/.streamlit/config.toml                        |  27 ++
 app/data.py                                       | 160 +++++++
 app/pages/1_Train_together.py                     | 216 ++++++++++
 app/pages/2_Try_to_steal_data.py                  |  71 ++++
 app/pages/3_At_the_bedside.py                     | 154 +++++++
 app/pages/4_Results.py                            | 132 ++++++
 app/static/fonts/AtkinsonHyperlegible-Bold.ttf    | Bin 0 -> 55256 bytes
 app/static/fonts/AtkinsonHyperlegible-Regular.ttf | Bin 0 -> 54348 bytes
 app/static/fonts/OFL.txt                          |  92 ++++
 app/streamlit_app.py                              |  71 ++++
 app/theme.py                                      |  67 +++
 configs/eval.yaml                                 |   5 +
 configs/experiments/centralized.yaml              |   6 +-
 configs/experiments/fedavg.yaml                   |   4 +
 configs/experiments/fedavg_dp.yaml                |   4 +
 configs/experiments/fedguard.yaml                 |   6 +
 configs/experiments/fedguard_async_nodp.yaml      |   4 +
 configs/experiments/fedprox.yaml                  |   4 +
 configs/experiments/local.yaml                    |   4 +
 configs/fl/fedavg.yaml                            |  17 +-
 configs/fl/fedguard_async.yaml                    |  10 +-
 configs/model/gru.yaml                            |   9 +
 configs/model/lgbm.yaml                           |   3 +
 configs/model/lr.yaml                             |   3 +
 configs/privacy/uniform.yaml                      |  19 +-
 docs/decisions.md                                 |  79 ++++
 docs/demo.md                                      |  84 ++++
 docs/privacy.md                                   |  58 +++
 pyproject.toml                                    |   3 +
 scripts/deploy/run_flower.ps1                     |  19 +
 scripts/deploy/start_client.ps1                   |  14 +
 scripts/deploy/start_client.sh                    |   7 +
 scripts/deploy/start_local.ps1                    |  45 ++
 scripts/deploy/start_server.ps1                   |   7 +
 scripts/deploy/start_server.sh                    |   5 +
 scripts/deploy/stop_local.ps1                     |  10 +
 scripts/run_ablations.ps1                         |   7 +
 scripts/run_ablations.sh                          |   6 +
 scripts/run_all_main.ps1                          |   8 +
 scripts/run_all_main.sh                           |   5 +
 scripts/run_experiments.py                        | 161 +++++++
 src/fedguard/alerts/mc_ablation.py                |  45 ++
 src/fedguard/alerts/metrics.py                    |  95 +++++
 src/fedguard/alerts/policy.py                     |  19 +
 src/fedguard/alerts/run.py                        |  73 ++++
 src/fedguard/alerts/tuning.py                     |  94 +++++
 src/fedguard/attack/gradient_inversion.py         | 118 ++++++
 src/fedguard/attack/run.py                        | 127 ++++++
 src/fedguard/cli.py                               | 242 +++++++++--
 src/fedguard/config.py                            |  19 +
 src/fedguard/data/physionet2019.py                |  28 ++
 src/fedguard/data/scenario.py                     |  98 +++++
 src/fedguard/data/windows.py                      |  15 +-
 src/fedguard/eval/_official_2019.py               | 493 ++++++++++++++++++++++
 src/fedguard/eval/metrics.py                      | 116 +++++
 src/fedguard/eval/predictions.py                  |  84 ++++
 src/fedguard/eval/report.py                       | 438 +++++++++++++++++++
 src/fedguard/eval/run_artifacts.py                |  40 ++
 src/fedguard/eval/utility.py                      |  89 ++++
 src/fedguard/explain/attention.py                 |  40 ++
 src/fedguard/explain/integrated_gradients.py      |  46 ++
 src/fedguard/explain/run.py                       |  83 ++++
 src/fedguard/export/results_json.py               | 240 +++++++++++
 src/fedguard/fl/aggregators.py                    |  68 +++
 src/fedguard/fl/client.py                         | 222 ++++++++++
 src/fedguard/fl/engine.py                         | 327 ++++++++++++++
 src/fedguard/fl/flower_app/LICENSE                |   7 +
 src/fedguard/fl/flower_app/client_app.py          |  79 ++++
 src/fedguard/fl/flower_app/common.py              |  42 ++
 src/fedguard/fl/flower_app/pyproject.toml         |  33 ++
 src/fedguard/fl/flower_app/server_app.py          | 101 +++++
 src/fedguard/fl/runner.py                         | 108 +++++
 src/fedguard/models/baselines.py                  |  69 +++
 src/fedguard/models/patchtst.py                   | 144 +++++++
 src/fedguard/models/uncertainty.py                |  48 +++
 src/fedguard/privacy/accounting.py                |  44 ++
 src/fedguard/privacy/budgets.py                   |  58 +++
 src/fedguard/privacy/public_norm.py               |  60 +++
 src/fedguard/train/centralized.py                 |  26 ++
 src/fedguard/train/local.py                       |  33 ++
 src/fedguard/train/loops.py                       | 211 +++++++++
 src/fedguard/train/runner.py                      |  89 ++++
 src/fedguard/train/sklearn_models.py              |  72 ++++
 src/fedguard/utils/io.py                          |  30 +-
 src/fedguard/utils/runs.py                        |  15 +
 tests/test_alerts.py                              | 102 +++++
 tests/test_app.py                                 |  91 ++++
 tests/test_data.py                                |  19 +
 tests/test_eval.py                                |  83 ++++
 tests/test_explain.py                             |  41 ++
 tests/test_export.py                              |  67 +++
 tests/test_fl.py                                  | 152 +++++++
 tests/test_models.py                              | 134 ++++++
 tests/test_privacy.py                             |  98 +++++
 tests/test_scaffold.py                            |   9 +-
 97 files changed, 6904 insertions(+), 53 deletions(-)

commit 744c232b8ead640b1b806020a0f738e17b2df90a
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-26 17:35:09 +0530

    M1: PhysioNet 2019 data pipeline - download, verify, strata, splits, causal windows, EDA
    
    - MD5-verified resumable S3 download (40,336 files verified), physionet.org fallback
    - per-stratum order-independent patient-level splits; unit/hospital/dirichlet partitions
    - 107-channel causal features, train-only normalisation, lazy WindowDataset
    - EDA with real prevalences (4-node positive-hour rate 1.63%); docs/data.md, D13-D14
    - fix .gitignore: anchor /data/ so src/fedguard/data/ is tracked
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 .gitignore                                         |   6 +-
 CLAUDE.md                                          |   2 +
 Makefile                                           |   4 +-
 PROGRESS.md                                        |  54 ++-
 docs/data.md                                       |  76 ++++
 docs/decisions.md                                  |  21 +-
 prepare_data.py                                    | 119 ------
 results/eda/eda_summary.json                       | 475 +++++++++++++++++++++
 results/eda/los.png                                | Bin 0 -> 33703 bytes
 results/eda/missingness.csv                        |  35 ++
 results/eda/missingness.png                        | Bin 0 -> 69244 bytes
 results/eda/prevalence.png                         | Bin 0 -> 58803 bytes
 results/eda/summary_by_stratum.csv                 |   8 +
 .../eda/summary_hospital_merge_into_hospital.csv   |   4 +
 results/eda/summary_unit_exclude.csv               |   6 +
 results/eda/summary_unit_separate.csv              |   8 +
 scripts/make.ps1                                   |  15 +-
 src/fedguard/cli.py                                |  79 +++-
 src/fedguard/config.py                             |   8 +
 src/fedguard/data/__init__.py                      |   0
 src/fedguard/data/adapters/__init__.py             |   0
 src/fedguard/data/adapters/eicu.py                 |  45 ++
 src/fedguard/data/adapters/mimic_iv.py             |  48 +++
 src/fedguard/data/download.py                      | 199 +++++++++
 src/fedguard/data/eda.py                           | 146 +++++++
 src/fedguard/data/features.py                      | 112 +++++
 src/fedguard/data/physionet2019.py                 | 289 +++++++++++++
 src/fedguard/data/windows.py                       | 191 +++++++++
 tests/__init__.py                                  |   0
 tests/fixtures_data.py                             |  46 ++
 tests/test_data.py                                 | 304 +++++++++++++
 31 files changed, 2159 insertions(+), 141 deletions(-)

commit 9326b400e6a7b06960ce9d214e90bb25e26e2426
Author: Shrikar Ramesh <myselfshrikar@gmail.com>
Date:   2026-09-26 17:14:00 +0530

    M0: scaffold FedGuard package, configs, CLI skeleton, run provenance, tests
    
    - pyproject with pinned deps (torch 2.14 cu130, opacus 1.6, flwr 1.38, captum 0.9, streamlit 1.64)
    - config composition with --fast blocks, run dirs with git/version provenance
    - CLAUDE.md, PROGRESS.md, docs/decisions.md (D1-D12), Makefile + scripts/make.ps1
    
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>

 .gitignore                             |  42 ++
 CLAUDE.md                              | 104 ++++
 CLAUDE_CODE_BUILD_PROMPT.md            | 362 +++++++++++++
 FedGuard live demo.html                | 935 +++++++++++++++++++++++++++++++++
 FedGuard_Plan.md                       | 241 +++++++++
 Makefile                               |  29 +
 PROGRESS.md                            |  77 +++
 README.md                              |  31 ++
 app/static/fedguard_demo.html          | 935 +++++++++++++++++++++++++++++++++
 configs/data.yaml                      |  29 +
 configs/experiments/centralized.yaml   |   4 +
 configs/fl/fedavg.yaml                 |  17 +
 configs/fl/fedguard_async.yaml         |  11 +
 configs/fl/fedguard_sync.yaml          |   3 +
 configs/fl/fedprox.yaml                |   4 +
 configs/model/patchtst.yaml            |  18 +
 configs/privacy/adaptive.yaml          |   4 +
 configs/privacy/none.yaml              |   1 +
 configs/privacy/uniform.yaml           |  10 +
 configs/train/centralized.yaml         |  17 +
 configs/train/local.yaml               |  17 +
 docs/decisions.md                      |  72 +++
 prepare_data.py                        | 119 +++++
 pyproject.toml                         |  76 +++
 scripts/make.ps1                       |  33 ++
 src/fedguard/__init__.py               |   3 +
 src/fedguard/alerts/__init__.py        |   0
 src/fedguard/attack/__init__.py        |   0
 src/fedguard/cli.py                    | 151 ++++++
 src/fedguard/config.py                 | 134 +++++
 src/fedguard/eval/__init__.py          |   0
 src/fedguard/explain/__init__.py       |   0
 src/fedguard/export/__init__.py        |   0
 src/fedguard/fl/__init__.py            |   0
 src/fedguard/fl/flower_app/__init__.py |   0
 src/fedguard/models/__init__.py        |   0
 src/fedguard/privacy/__init__.py       |   0
 src/fedguard/py.typed                  |   0
 src/fedguard/train/__init__.py         |   0
 src/fedguard/utils/__init__.py         |   0
 src/fedguard/utils/io.py               |  77 +++
 src/fedguard/utils/logging.py          |  76 +++
 src/fedguard/utils/runs.py             | 146 +++++
 src/fedguard/utils/seed.py             |  44 ++
 tests/conftest.py                      |  13 +
 tests/test_scaffold.py                 |  92 ++++
 46 files changed, 3927 insertions(+)
```
