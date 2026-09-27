| partition                 | algorithm     | AUROC       | AUPRC       |
|:--------------------------|:--------------|:------------|:------------|
| 2 nodes (hospital)        | FedAvg (sync) | 0.796 (n=1) | 0.098 (n=1) |
| 2 nodes (hospital)        | Async (no DP) | not run yet | not run yet |
| 4 nodes (unit, main)      | FedAvg (sync) | 0.784 (n=1) | 0.100 (n=1) |
| 4 nodes (unit, main)      | Async (no DP) | 0.765 (n=1) | 0.083 (n=1) |
| 8 nodes (Dirichlet a=0.5) | FedAvg (sync) | not run yet | not run yet |
| 8 nodes (Dirichlet a=0.5) | Async (no DP) | not run yet | not run yet |