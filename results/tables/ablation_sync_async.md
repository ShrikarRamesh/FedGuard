| scenario                                   | algorithm     | AUROC       | AUPRC       | simulated time (s)   |
|:-------------------------------------------|:--------------|:------------|:------------|:---------------------|
| default speeds                             | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) | 2912 (n=1)           |
| default speeds                             | Async (no DP) | 0.765 (n=1) | 0.083 (n=1) | 1527 (n=1)           |
| client dropout (B_MICU offline 300-1500 s) | FedAvg (sync) | not run yet | not run yet | not run yet          |
| client dropout (B_MICU offline 300-1500 s) | Async (no DP) | not run yet | not run yet | not run yet          |
| strong imbalance (B_SICU 5x slower)        | FedAvg (sync) | not run yet | not run yet | not run yet          |
| strong imbalance (B_SICU 5x slower)        | Async (no DP) | not run yet | not run yet | not run yet          |