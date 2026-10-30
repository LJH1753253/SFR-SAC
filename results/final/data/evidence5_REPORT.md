# Five-pair evidence study

Phase: serial_latency
Status: requested_stages_finished

Core and primary policy are frozen. gamma=0; E_ref=0.02 J; eta=0.30; d=0.10 except declared sweeps.
Every result group expects five paired deployments/training seeds. Missing seeds are not filled in.
Energy: mJ/device/slot. Cost and BFSR: percent. Tables show mean and sample SD across pairs.
Original-grid completion uses its original fixed stream; new evidence uses a separate common confirmation stream.
Historical cached tests retain their original identity/provenance; their raw CSV remains in original_ch5_results.tar.gz.

| Experiment | Variant | N | Seeds | Cost (%) | Energy | BFSR (%) |
|---|---|---:|---:|---:|---:|---:|
| actor_matched_penalty | beta2.5 | 70 | 5/5 | 11.744 | 8.714 | 26.522 |
| actor_matched_penalty | beta2.75 | 70 | 5/5 | 10.026 | 9.291 | 51.403 |
| actor_matched_penalty | beta3 | 70 | 5/5 | 8.704 | 9.968 | 71.321 |
| actor_matched_penalty | beta3.25 | 70 | 5/5 | 7.470 | 10.432 | 86.984 |
| actor_matched_penalty | beta4 | 70 | 5/5 | 5.842 | 11.349 | 98.463 |
| actor_matched_penalty | beta5 | 70 | 5/5 | 4.956 | 12.004 | 99.747 |
| background | b0_0.00 | 70 | 5/5 | 9.370 | 8.406 | 95.993 |
| background | b0_0.05 | 70 | 5/5 | 9.381 | 8.594 | 96.229 |
| background | b0_0.10 | 70 | 5/5 | 9.238 | 9.126 | 98.070 |
| budget | d0.08 | 70 | 5/5 | 7.077 | 9.610 | 98.062 |
| budget | d0.09 | 70 | 5/5 | 8.358 | 9.069 | 96.640 |
| budget | d0.10 | 70 | 5/5 | 9.381 | 8.594 | 96.229 |
| budget | d0.11 | 70 | 5/5 | 10.308 | 8.319 | 96.893 |
| budget | d0.12 | 70 | 5/5 | 11.415 | 7.917 | 93.159 |
| dual | batch | 70 | 5/5 | 9.381 | 8.594 | 96.229 |
| dual | episode | 70 | 5/5 | 9.834 | 8.428 | 66.624 |
| dual | step | 70 | 5/5 | 9.500 | 8.581 | 91.475 |
| eta | eta0.10 | 70 | 5/5 | 10.398 | 20.488 | 51.942 |
| eta | eta0.20 | 70 | 5/5 | 8.935 | 12.054 | 95.575 |
| eta | eta0.30 | 70 | 5/5 | 9.381 | 8.594 | 96.229 |
| eta | eta0.40 | 70 | 5/5 | 9.532 | 9.616 | 92.705 |
| eta | eta0.50 | 70 | 5/5 | 9.436 | 11.419 | 99.278 |
| eta | eta0.60 | 70 | 5/5 | 9.476 | 13.462 | 99.407 |
| eta | eta0.70 | 70 | 5/5 | 9.310 | 15.814 | 99.349 |
| eta | eta0.80 | 70 | 5/5 | 9.347 | 18.325 | 95.619 |
| eta | eta0.90 | 70 | 5/5 | 10.567 | 20.558 | 53.062 |
| fair_reference | local_first | 70 | 5/5 | 0.551 | 20.943 | 100.000 |
| fair_reference | sac_lagrangian | 70 | 5/5 | 7.892 | 11.282 | 85.451 |
| fair_rules | constant_budget | 70 | 5/5 | 9.598 | 11.734 | 58.812 |
| fair_rules | constant_matched_mixture | 70 | 5/5 | 9.203 | 11.839 | 63.133 |
| fair_rules | mb_greedy_budget | 70 | 5/5 | 9.398 | 9.790 | 68.164 |
| fair_rules | mb_greedy_matched_mixture | 70 | 5/5 | 9.404 | 9.787 | 66.732 |
| main | dpp_rule | 50 | 5/5 | 9.963 | 11.371 | 51.983 |
| main | dpp_rule | 55 | 5/5 | 10.001 | 12.028 | 51.593 |
| main | dpp_rule | 60 | 5/5 | 10.001 | 12.180 | 51.450 |
| main | dpp_rule | 65 | 5/5 | 10.002 | 12.660 | 51.801 |
| main | dpp_rule | 70 | 5/5 | 10.001 | 12.742 | 51.826 |
| main | penalty_beta5 | 50 | 5/5 | 3.177 | 8.446 | 99.987 |
| main | penalty_beta5 | 55 | 5/5 | 4.009 | 9.846 | 99.897 |
| main | penalty_beta5 | 60 | 5/5 | 4.220 | 10.411 | 99.822 |
| main | penalty_beta5 | 65 | 5/5 | 4.529 | 11.299 | 99.824 |
| main | penalty_beta5 | 70 | 5/5 | 5.038 | 12.045 | 99.581 |
| main | split_exact | 50 | 5/5 | 9.532 | 4.954 | 86.005 |
| main | split_exact | 55 | 5/5 | 9.280 | 6.284 | 93.165 |
| main | split_exact | 60 | 5/5 | 9.089 | 6.943 | 96.902 |
| main | split_exact | 65 | 5/5 | 9.209 | 9.043 | 88.817 |
| main | split_exact | 70 | 5/5 | 9.381 | 8.594 | 96.229 |
| main | td3 | 50 | 5/5 | 9.381 | 6.198 | 70.487 |
| main | td3 | 55 | 5/5 | 9.510 | 8.062 | 67.504 |
| main | td3 | 60 | 5/5 | 9.655 | 8.569 | 63.558 |
| main | td3 | 65 | 5/5 | 9.640 | 9.777 | 62.962 |
| main | td3 | 70 | 5/5 | 9.277 | 10.977 | 72.775 |
| main | vanilla | 50 | 5/5 | 89.932 | 0.504 | 0.000 |
| main | vanilla | 55 | 5/5 | 88.967 | 0.474 | 0.000 |
| main | vanilla | 60 | 5/5 | 90.322 | 0.473 | 0.000 |
| main | vanilla | 65 | 5/5 | 86.653 | 0.492 | 0.000 |
| main | vanilla | 70 | 5/5 | 89.754 | 0.477 | 0.000 |
| mechanism | shared_mlp | 70 | 5/5 | 9.819 | 10.075 | 59.473 |
| mechanism | shared_structured | 70 | 5/5 | 8.913 | 11.888 | 89.135 |
| mechanism | split_mlp | 70 | 5/5 | 11.450 | 10.241 | 28.636 |
| mechanism | split_structured | 70 | 5/5 | 9.385 | 8.587 | 96.050 |
| penalty | beta1 | 70 | 5/5 | 28.129 | 3.747 | 0.000 |
| penalty | beta10 | 70 | 5/5 | 3.998 | 13.475 | 99.921 |
| penalty | beta20 | 70 | 5/5 | 3.996 | 14.934 | 99.829 |
| penalty | beta5 | 70 | 5/5 | 5.038 | 12.045 | 99.581 |

## Matching
Validation-selected penalty beta: None.
All beta arms are tested only after validation selection is locked. A missing match remains missing.
Matched-rule mixtures are genuinely sampled and rolled out; reported test metrics are never interpolated.
Same budget, matched mean failure, BFSR, and per-device reliability are distinct comparisons.

## Limits
Critics predict immediate outcomes; no long-horizon CMDP or hard-safety guarantee.
Dual variants change both update frequency and cost statistic, not just frequency at equal effective gain.
Lyapunov is an inspired heuristic. MB-greedy has full simulator-model information.
All N65 seeds remain included. Aggregate feasibility does not guarantee per-device reliability.
Nominal paired t intervals use five pairs, are exploratory and have no multiple-comparison correction.
Latency is measured serially after experimental jobs; CPU/GPU are separate. Mean P95 is averaged across deployments, not a pooled invocation percentile.
Compact exports omit full replay checkpoints; keep the remote directory for interrupted-training resume.
