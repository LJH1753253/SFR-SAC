# Reproducibility Notes

The final protocol uses:

- physical CPU-cycle local energy and PA/circuit TDMA-airtime energy;
- AP-dependent geometry, heterogeneous fixed CPU frequencies, bursty workload
  dynamics, and one-shot slot completion;
- `eta=0.30`, `d=0.10`, `E_ref=0.02 J`, and `gamma=0`;
- 1,600 training episodes, 200 slots per episode, and 100 final evaluation
  episodes;
- five paired deployment/training seeds for the reported groups.

The state and action dimensions are `3N + M + 8` and `2N + NM`, respectively.
For `N=70` and `M=6`, the action dimension is 560. The action contains
continuous offloading and transmit-power branches and an AP-association branch
executed as one-hot AP choices.

The aggregate budget-feasible slot ratio (BFSR) is the fraction of evaluation
slots whose composite service-failure cost does not exceed `d`. It is not a
per-device reliability guarantee.

The compact result package contains the final model and summary artifacts but
not the full replay buffer for every historical run. The full resume-safe run
directory should be preserved when continuing interrupted training.
