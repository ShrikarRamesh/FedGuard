### main

| method                            |   n_seeds | auroc         | auprc         | own_node_auroc   | own_node_auprc   | prevalence      | ece           |
|:----------------------------------|----------:|:--------------|:--------------|:-----------------|:-----------------|:----------------|:--------------|
| Local-only                        |         3 | 0.711 ± 0.010 | 0.055 ± 0.001 | 0.782 ± 0.015    | 0.089 ± 0.005    | 0.0166 ± 0.0000 | 0.228 ± 0.026 |
| FedAvg                            |         3 | 0.779 ± 0.007 | 0.102 ± 0.002 |                  |                  | 0.0166 ± 0.0000 | 0.051 ± 0.006 |
| FedProx                           |         3 | 0.781 ± 0.006 | 0.108 ± 0.001 |                  |                  | 0.0166 ± 0.0000 | 0.056 ± 0.006 |
| FedAvg + DP                       |         3 | 0.614 ± 0.018 | 0.024 ± 0.003 |                  |                  | 0.0166 ± 0.0000 | 0.026 ± 0.018 |
| FedGuard                          |         3 | 0.620 ± 0.018 | 0.024 ± 0.002 |                  |                  | 0.0166 ± 0.0000 | 0.016 ± 0.000 |
| Centralized                       |         3 | 0.816 ± 0.011 | 0.106 ± 0.005 |                  |                  | 0.0166 ± 0.0000 | 0.213 ± 0.056 |
| Async FL (no DP)                  |         3 | 0.765 ± 0.005 | 0.084 ± 0.002 |                  |                  | 0.0166 ± 0.0000 | 0.104 ± 0.018 |
| GRU (centralized)                 |         3 | 0.818 ± 0.007 | 0.105 ± 0.007 |                  |                  | 0.0166 ± 0.0000 | 0.251 ± 0.006 |
| Logistic regression (centralized) |         1 | 0.798 (n=1)   | 0.083 (n=1)   |                  |                  | 0.0166 (n=1)    | 0.322 (n=1)   |
| LightGBM (centralized)            |         1 | 0.817 (n=1)   | 0.092 (n=1)   |                  |                  | 0.0166 (n=1)    | 0.165 (n=1)   |

### privacy

| epsilon                    | FedGuard (adaptive) AUROC   | FedGuard (adaptive) AUPRC   | FedGuard (uniform rule) AUROC   | FedGuard (uniform rule) AUPRC   | FedAvg + DP (uniform) AUROC   | FedAvg + DP (uniform) AUPRC   |
|:---------------------------|:----------------------------|:----------------------------|:--------------------------------|:--------------------------------|:------------------------------|:------------------------------|
| 1.0                        | not run yet                 | not run yet                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| 2.0                        | not run yet                 | not run yet                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| 3.0                        | 0.620 ± 0.018               | 0.024 ± 0.002               | not run yet                     | not run yet                     | 0.614 ± 0.018                 | 0.024 ± 0.003                 |
| 5.0                        | not run yet                 | not run yet                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| 8.0                        | not run yet                 | not run yet                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| no DP (same normalisation) | 0.757 ± 0.006               | 0.075 ± 0.004               | nan                             | nan                             | nan                           | nan                           |

### alerts

| model          | policy                       | false_per_100h   |   n_seeds | sensitivity   | median_lead_h   | utility     | alarms_per_100h   |
|:---------------|:-----------------------------|:-----------------|----------:|:--------------|:----------------|:------------|:------------------|
| FedGuard (ε=3) | -                            | not run yet      |       nan | nan           | nan             | nan         | nan               |
| Centralized    | threshold-only               | 1.10 (n=1)       |         1 | 0.457 (n=1)   | 3.5 (n=1)       | 0.380 (n=1) | 1.18 (n=1)        |
| Centralized    | uncertainty-gated (FedGuard) | 0.78 (n=1)       |         1 | 0.437 (n=1)   | 3.0 (n=1)       | 0.330 (n=1) | 0.85 (n=1)        |
| Centralized    | gated, utility-optimal       | 1.12 (n=1)       |         1 | 0.453 (n=1)   | 4.0 (n=1)       | 0.379 (n=1) | 1.20 (n=1)        |

### calibration

| model       | ECE deterministic   | ECE MC-Dropout mean   |
|:------------|:--------------------|:----------------------|
| Centralized | 0.2682 (n=1)        | 0.3063 (n=1)          |

### async_tuning

|   alpha0 |   staleness_lambda |   best val AUPRC (selection) |   test AUROC (not used for selection) |
|---------:|-------------------:|-----------------------------:|--------------------------------------:|
|      0.5 |               0.35 |                    0.0619544 |                              0.742333 |
|      1   |               0.35 |                    0.0669289 |                              0.753792 |
|      2   |               0.35 |                    0.070271  |                              0.759347 |
|      4   |               0.35 |                    0.0799778 |                              0.765461 |

### ablation_node_count

| partition                 | algorithm     | AUROC       | AUPRC       |
|:--------------------------|:--------------|:------------|:------------|
| 2 nodes (hospital)        | FedAvg (sync) | 0.796 (n=1) | 0.098 (n=1) |
| 2 nodes (hospital)        | Async (no DP) | not run yet | not run yet |
| 4 nodes (unit, main)      | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) |
| 4 nodes (unit, main)      | Async (no DP) | 0.765 (n=1) | 0.083 (n=1) |
| 8 nodes (Dirichlet a=0.5) | FedAvg (sync) | not run yet | not run yet |
| 8 nodes (Dirichlet a=0.5) | Async (no DP) | not run yet | not run yet |

### ablation_sync_async

| scenario                                   | algorithm     | AUROC       | AUPRC       | simulated time (s)   |
|:-------------------------------------------|:--------------|:------------|:------------|:---------------------|
| default speeds                             | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) | 2912 (n=1)           |
| default speeds                             | Async (no DP) | 0.765 (n=1) | 0.083 (n=1) | 1527 (n=1)           |
| client dropout (B_MICU offline 300-1500 s) | FedAvg (sync) | not run yet | not run yet | not run yet          |
| client dropout (B_MICU offline 300-1500 s) | Async (no DP) | not run yet | not run yet | not run yet          |
| strong imbalance (B_SICU 5x slower)        | FedAvg (sync) | not run yet | not run yet | not run yet          |
| strong imbalance (B_SICU 5x slower)        | Async (no DP) | not run yet | not run yet | not run yet          |

### ablation_budget_rules

| rule                | AUROC       | AUPRC       | eps spent                                          |
|:--------------------|:------------|:------------|:---------------------------------------------------|
| adaptive (FedGuard) | 0.640 (n=1) | 0.026 (n=1) | A_MICU 2.68, A_SICU 2.71, B_MICU 2.98, B_SICU 2.99 |
| uniform             | not run yet | not run yet | not run yet                                        |
| inverse             | not run yet | not run yet | not run yet                                        |
| equal noise         | not run yet | not run yet | not run yet                                        |

### ablation_lookback

|   lookback L (h) | AUROC       | AUPRC       |
|-----------------:|:------------|:------------|
|               12 | not run yet | not run yet |
|               24 | 0.823 (n=1) | 0.111 (n=1) |
|               48 | not run yet | not run yet |
