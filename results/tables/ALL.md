### main

| method                            |   n_seeds | auroc         | auprc         | own_node_auroc   | own_node_auprc   | prevalence      | ece           |
|:----------------------------------|----------:|:--------------|:--------------|:-----------------|:-----------------|:----------------|:--------------|
| Local-only                        |         3 | 0.711 ± 0.010 | 0.055 ± 0.001 | 0.782 ± 0.015    | 0.089 ± 0.005    | 0.0166 ± 0.0000 | 0.228 ± 0.026 |
| FedAvg                            |         3 | 0.779 ± 0.007 | 0.102 ± 0.002 |                  |                  | 0.0166 ± 0.0000 | 0.051 ± 0.006 |
| FedProx                           |         3 | 0.781 ± 0.006 | 0.108 ± 0.001 |                  |                  | 0.0166 ± 0.0000 | 0.056 ± 0.006 |
| FedAvg + DP                       |         3 | 0.615 ± 0.017 | 0.024 ± 0.003 |                  |                  | 0.0166 ± 0.0000 | 0.030 ± 0.024 |
| FedGuard                          |         3 | 0.620 ± 0.018 | 0.024 ± 0.002 |                  |                  | 0.0166 ± 0.0000 | 0.016 ± 0.000 |
| Centralized                       |         3 | 0.816 ± 0.011 | 0.106 ± 0.005 |                  |                  | 0.0166 ± 0.0000 | 0.213 ± 0.056 |
| Async FL (no DP)                  |         3 | 0.765 ± 0.005 | 0.084 ± 0.002 |                  |                  | 0.0166 ± 0.0000 | 0.104 ± 0.018 |
| GRU (centralized)                 |         3 | 0.818 ± 0.007 | 0.105 ± 0.007 |                  |                  | 0.0166 ± 0.0000 | 0.251 ± 0.006 |
| Logistic regression (centralized) |         1 | 0.798 (n=1)   | 0.083 (n=1)   |                  |                  | 0.0166 (n=1)    | 0.322 (n=1)   |
| LightGBM (centralized)            |         1 | 0.817 (n=1)   | 0.092 (n=1)   |                  |                  | 0.0166 (n=1)    | 0.165 (n=1)   |

### privacy

| epsilon                    | FedGuard (adaptive) AUROC   | FedGuard (adaptive) AUPRC   | FedGuard (uniform rule) AUROC   | FedGuard (uniform rule) AUPRC   | FedAvg + DP (uniform) AUROC   | FedAvg + DP (uniform) AUPRC   |
|:---------------------------|:----------------------------|:----------------------------|:--------------------------------|:--------------------------------|:------------------------------|:------------------------------|
| 1.0                        | 0.619 ± 0.030               | 0.026 ± 0.004               | 0.638 (n=1)                     | 0.028 (n=1)                     | 0.614 ± 0.036                 | 0.025 ± 0.002                 |
| 2.0                        | 0.618 ± 0.017               | 0.024 ± 0.002               | 0.637 (n=1)                     | 0.027 (n=1)                     | 0.622 ± 0.019                 | 0.025 ± 0.004                 |
| 3.0                        | 0.620 ± 0.018               | 0.024 ± 0.002               | 0.640 (n=1)                     | 0.026 (n=1)                     | 0.615 ± 0.017                 | 0.024 ± 0.003                 |
| 5.0                        | 0.616 ± 0.015               | 0.024 ± 0.003               | 0.616 ± 0.020                   | 0.024 ± 0.004                   | 0.611 ± 0.029                 | 0.024 ± 0.005                 |
| 8.0                        | 0.615 ± 0.013               | 0.024 ± 0.003               | not run yet                     | not run yet                     | 0.608 ± 0.021                 | 0.024 ± 0.004                 |
| 16.0                       | 0.628 (n=1)                 | 0.027 (n=1)                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| 32.0                       | 0.628 (n=1)                 | 0.027 (n=1)                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| 64.0                       | 0.629 (n=1)                 | 0.028 (n=1)                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| 256.0                      | 0.625 (n=1)                 | 0.028 (n=1)                 | not run yet                     | not run yet                     | not run yet                   | not run yet                   |
| no DP (same normalisation) | 0.753 ± 0.009               | 0.075 ± 0.003               | nan                             | nan                             | nan                           | nan                           |

### alerts

| model          | policy                       |   n_seeds | false_per_100h   | sensitivity   | median_lead_h   | utility        | alarms_per_100h   |
|:---------------|:-----------------------------|----------:|:-----------------|:--------------|:----------------|:---------------|:------------------|
| FedGuard (ε=3) | threshold-only               |         3 | 0.06 ± 0.10      | 0.008 ± 0.014 | 5.5 (n=1)       | -0.001 ± 0.001 | 0.06 ± 0.11       |
| FedGuard (ε=3) | uncertainty-gated (FedGuard) |         3 | 0.06 ± 0.10      | 0.008 ± 0.014 | 5.5 (n=1)       | -0.001 ± 0.001 | 0.06 ± 0.11       |
| FedGuard (ε=3) | gated, utility-optimal       |         3 | 0.13 ± 0.20      | 0.010 ± 0.013 | 4.5 ± 2.1       | -0.002 ± 0.003 | 0.14 ± 0.20       |
| Centralized    | threshold-only               |         1 | 1.10 (n=1)       | 0.457 (n=1)   | 3.5 (n=1)       | 0.380 (n=1)    | 1.18 (n=1)        |
| Centralized    | uncertainty-gated (FedGuard) |         1 | 0.78 (n=1)       | 0.437 (n=1)   | 3.0 (n=1)       | 0.330 (n=1)    | 0.85 (n=1)        |
| Centralized    | gated, utility-optimal       |         1 | 1.12 (n=1)       | 0.453 (n=1)   | 4.0 (n=1)       | 0.379 (n=1)    | 1.20 (n=1)        |

### calibration

| model          | ECE deterministic   | ECE MC-Dropout mean   |
|:---------------|:--------------------|:----------------------|
| FedGuard (ε=3) | 0.0161 ± 0.0001     | 0.0147 ± 0.0004       |
| Centralized    | 0.2682 (n=1)        | 0.3063 (n=1)          |

### async_tuning

|   alpha0 |   staleness_lambda |   best val AUPRC (selection) |   test AUROC (not used for selection) |
|---------:|-------------------:|-----------------------------:|--------------------------------------:|
|      0.5 |               0.35 |                    0.0619544 |                              0.742333 |
|      1   |               0.35 |                    0.0669289 |                              0.753792 |
|      2   |               0.35 |                    0.070271  |                              0.759347 |
|      4   |               0.1  |                    0.0835061 |                              0.760214 |
|      4   |               0.35 |                    0.0799778 |                              0.765461 |
|      4   |               1    |                    0.059045  |                              0.741331 |

### ablation_node_count

| partition                 | algorithm     | AUROC       | AUPRC       |
|:--------------------------|:--------------|:------------|:------------|
| 2 nodes (hospital)        | FedAvg (sync) | 0.796 (n=1) | 0.098 (n=1) |
| 2 nodes (hospital)        | Async (no DP) | 0.754 (n=1) | 0.061 (n=1) |
| 4 nodes (unit, main)      | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) |
| 4 nodes (unit, main)      | Async (no DP) | 0.765 (n=1) | 0.083 (n=1) |
| 8 nodes (Dirichlet a=0.5) | FedAvg (sync) | 0.777 (n=1) | 0.088 (n=1) |
| 8 nodes (Dirichlet a=0.5) | Async (no DP) | 0.738 (n=1) | 0.057 (n=1) |

### ablation_sync_async

| scenario                                   | algorithm     | AUROC       | AUPRC       | simulated time (s)   |
|:-------------------------------------------|:--------------|:------------|:------------|:---------------------|
| default speeds                             | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) | 2912 (n=1)           |
| default speeds                             | Async (no DP) | 0.765 (n=1) | 0.083 (n=1) | 1527 (n=1)           |
| client dropout (B_MICU offline 300-1500 s) | FedAvg (sync) | 0.786 (n=1) | 0.101 (n=1) | 3039 (n=1)           |
| client dropout (B_MICU offline 300-1500 s) | Async (no DP) | 0.790 (n=1) | 0.089 (n=1) | 1975 (n=1)           |
| strong imbalance (B_SICU 5x slower)        | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) | 6313 (n=1)           |
| strong imbalance (B_SICU 5x slower)        | Async (no DP) | 0.763 (n=1) | 0.087 (n=1) | 1740 (n=1)           |

### ablation_budget_rules

| rule                | AUROC       | AUPRC       | eps spent                                          |
|:--------------------|:------------|:------------|:---------------------------------------------------|
| adaptive (FedGuard) | 0.640 (n=1) | 0.026 (n=1) | A_MICU 2.68, A_SICU 2.71, B_MICU 2.98, B_SICU 2.99 |
| uniform             | 0.640 (n=1) | 0.026 (n=1) | A_MICU 3.00, A_SICU 3.00, B_MICU 2.99, B_SICU 2.99 |
| inverse             | 0.641 (n=1) | 0.026 (n=1) | A_MICU 3.00, A_SICU 2.96, B_MICU 2.68, B_SICU 2.68 |
| equal noise         | 0.640 (n=1) | 0.026 (n=1) | A_MICU 3.40, A_SICU 3.40, B_MICU 2.99, B_SICU 2.99 |

### ablation_lookback

|   lookback L (h) | AUROC       | AUPRC       |
|-----------------:|:------------|:------------|
|               12 | 0.823 (n=1) | 0.118 (n=1) |
|               24 | 0.823 (n=1) | 0.111 (n=1) |
|               48 | 0.826 (n=1) | 0.109 (n=1) |
