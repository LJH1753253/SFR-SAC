# Per-slot Optimization Report

This report was regenerated from the completed historical per-slot outputs. No training or numerical optimization was run during recovery.

## Validated scope

- `N=70`, `d=0.10`, `eta=0.30`, `gamma=0`.
- Five paired training seeds/layouts: `(1000,0)` through `(1004,4)`.
- Four methods, 20 completed tests, 30 evaluation episodes and 6,000 slots per test.
- Critic refine: 30 iterations; critic scratch: 150 iterations; numerical optimizer: 100 iterations.
- Decision time is measured around the optimizer/action function with `time.perf_counter`; it excludes state acquisition, signaling, and actuation overhead.
- Means and sample standard deviations are over the five deployment pairs. Evaluation slots are not treated as independent training replicates.

## Completion reconciliation

The historical controller status records 20 completed of 25 planned tasks and lists the five parent-training IDs under `failed`, but each of those entries has `exit_code=0`. Direct inspection finds `run_status=done`, `model_ep1600.pt`, `model.pt`, and `final_eval_summary.json` for all five parents. All 20 per-slot tasks also have `run_status=done` and complete 30-episode/6,000-slot summaries. The artifact-level completion count is therefore 5/5 parent trainings and 20/20 per-slot tests; the stale controller count is retained in provenance rather than silently overwritten.

## Summary

| Method | Cost (%) | Energy (mJ/device/slot) | BFSR (%) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|
| Actor | 9.373 +/- 0.121 | 9.678 +/- 2.198 | 90.437 +/- 13.898 | 1.212 +/- 0.106 | 2.373 +/- 0.512 |
| Critic refine | 9.727 +/- 0.150 | 9.191 +/- 1.640 | 84.147 +/- 1.823 | 200.389 +/- 24.595 | 291.438 +/- 82.482 |
| Critic scratch | 9.858 +/- 0.028 | 8.531 +/- 0.349 | 74.697 +/- 3.971 | 933.709 +/- 134.725 | 1204.192 +/- 292.072 |
| Numerical optimizer | 9.998 +/- 0.004 | 7.707 +/- 0.326 | 53.240 +/- 0.870 | 101.935 +/- 9.815 | 178.389 +/- 63.558 |

## Interpretation

The Actor is the low-latency method. Critic refinement and critic scratch reduce the energy objective at the cost of substantially longer per-slot decisions. The full-information Numerical optimizer has the lowest mean energy, but its mean cost is close to the 10% budget and its BFSR is lower than the Actor. Therefore, this experiment does not establish Actor energy optimality at an equal failure level. The Numerical optimizer is a full-information numerical baseline, not a mathematically proven global optimum.

## Provenance

- Historical source directory: `D:\PythonCode\SFR-SAC-revision-overnight\outputs\revision_local_overnight`
- Historical main source digest: `9c00e9d775dfde3da37e852d23b6583b93fd9296bb15c52d8f70ab78d6dc9010`
- Historical revision source digest: `0681f3c0d6721a33ceb6b500e0c5856ba4a30499afdbfecb0f576c197cfd7021`
- Historical protocol hash: `fc8a7bdc48652ef2`
- The per-slot run used the historical overnight source/checkpoints and hardware recorded in `data/per_slot_provenance.json`; it must not be described as sharing the current container's Fair-Critics checkpoints.
- Required raw test summaries and CSV files are archived under `data/per_slot_runs/`. Training checkpoints are intentionally not copied.
