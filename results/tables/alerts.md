| model          | policy                       |   n_seeds | false_per_100h   | sensitivity   | median_lead_h   | utility        | alarms_per_100h   |
|:---------------|:-----------------------------|----------:|:-----------------|:--------------|:----------------|:---------------|:------------------|
| FedGuard (ε=3) | threshold-only               |         3 | 0.06 ± 0.10      | 0.008 ± 0.014 | 5.5 (n=1)       | -0.001 ± 0.001 | 0.06 ± 0.11       |
| FedGuard (ε=3) | uncertainty-gated (FedGuard) |         3 | 0.06 ± 0.10      | 0.008 ± 0.014 | 5.5 (n=1)       | -0.001 ± 0.001 | 0.06 ± 0.11       |
| FedGuard (ε=3) | gated, utility-optimal       |         3 | 0.13 ± 0.20      | 0.010 ± 0.013 | 4.5 ± 2.1       | -0.002 ± 0.003 | 0.14 ± 0.20       |
| Centralized    | threshold-only               |         1 | 1.10 (n=1)       | 0.457 (n=1)   | 3.5 (n=1)       | 0.380 (n=1)    | 1.18 (n=1)        |
| Centralized    | uncertainty-gated (FedGuard) |         1 | 0.78 (n=1)       | 0.437 (n=1)   | 3.0 (n=1)       | 0.330 (n=1)    | 0.85 (n=1)        |
| Centralized    | gated, utility-optimal       |         1 | 1.12 (n=1)       | 0.453 (n=1)   | 4.0 (n=1)       | 0.379 (n=1)    | 1.20 (n=1)        |