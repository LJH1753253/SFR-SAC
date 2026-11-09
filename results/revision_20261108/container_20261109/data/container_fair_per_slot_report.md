# Container Fair-Critics and Per-slot Archive

This archive was copied from the completed container run. No training or optimization was run during archiving.

## Completion

- Fair-Critics: 35 test cases plus 35 parent trainings, all completed.
- Per-slot: 20 tests, all completed; 5 seeds for Actor, Critic refine, Critic scratch, and Numerical optimizer.
- The generic runner report says `73 of 217` because it uses the full revision catalog denominator and includes shared test aliases. The selected target stages are complete and are reported separately here.

## Fair-Critics

| Variant | Pairs | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |
|---|---:|---:|---:|---:|
| SFR-SAC | 5 | 9.38 $\pm$ 0.16 | 8.59 $\pm$ 0.36 | 96.23 $\pm$ 2.31 |
| Safe TD3 | 5 | 9.28 $\pm$ 0.37 | 10.98 $\pm$ 0.51 | 72.78 $\pm$ 9.92 |
| TD3 + structured critics | 5 | 9.35 $\pm$ 0.08 | 8.88 $\pm$ 0.35 | 76.71 $\pm$ 3.67 |
| TD3 + structured critics + split actor | 5 | 10.14 $\pm$ 0.56 | 10.57 $\pm$ 2.96 | 56.67 $\pm$ 15.41 |
| Fixed penalty beta=2 | 3 | 10.35 $\pm$ 3.25 | 8.99 $\pm$ 0.42 | 52.74 $\pm$ 41.28 |
| Fixed penalty beta=2.5 | 3 | 5.14 $\pm$ 0.45 | 10.24 $\pm$ 0.52 | 99.37 $\pm$ 0.42 |
| Fixed penalty beta=3 | 3 | 5.57 $\pm$ 2.75 | 11.65 $\pm$ 1.52 | 92.06 $\pm$ 13.65 |
| Fixed penalty beta=3.5 | 3 | 3.59 $\pm$ 0.54 | 10.89 $\pm$ 0.62 | 99.96 $\pm$ 0.04 |
| Fixed penalty beta=4.5 | 3 | 3.44 $\pm$ 0.47 | 11.16 $\pm$ 0.59 | 99.98 $\pm$ 0.03 |

## Per-slot

| Method | Pairs | Cost (%) | Energy (mJ/device/slot) | BFSR (%) | Median (ms) |
|---|---:|---:|---:|---:|---:|
| Actor | 5 | 9.38 $\pm$ 0.16 | 8.59 $\pm$ 0.38 | 96.37 $\pm$ 2.12 | 1.750 $\pm$ 0.013 |
| Critic refine | 5 | 9.79 $\pm$ 0.04 | 8.36 $\pm$ 0.39 | 84.65 $\pm$ 2.26 | 246.295 $\pm$ 0.993 |
| Critic scratch | 5 | 9.86 $\pm$ 0.04 | 8.53 $\pm$ 0.35 | 74.26 $\pm$ 4.91 | 1172.968 $\pm$ 1.245 |
| Numerical optimizer | 5 | 10.00 $\pm$ 0.00 | 7.71 $\pm$ 0.33 | 53.22 $\pm$ 0.87 | 130.488 $\pm$ 1.318 |

The Actor is the low-latency policy. The per-slot refinements and numerical optimizer reduce energy only by accepting higher failure cost and/or much longer decision time. The Numerical optimizer is a full-information numerical baseline, not a proven global optimum.

## Provenance

- Main source digest: `36c1289d0f1a5bba14e49f93f8713ae2128dec4dc3b66c587520145c9eab5777`
- Revision source digest: `7e9709f3b0c2743d088af104f4fde80d2a7110c84c25ae2f81b89b2a5efeb03a`
- Protocol hash: `25c98d2abc8b0e88`
- Fair test raw archive includes summaries, per-episode metrics, per-device metrics, task/status markers, hardware, and revision markers; large Fair test slot traces and all checkpoints are omitted.
- Per-slot raw archive additionally includes per-slot traces because these files are smaller and useful for reproducing the decision-time summary.
