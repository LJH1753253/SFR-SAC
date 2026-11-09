# Paper1 SFR-SAC Revision Preflight Audit

Date: 2026-11-06

No formal experiment, full Smoke, numerical per-slot sweep, latency benchmark,
or GPU training was run during this audit.

## Summary

The extension package is the correct Paper1/SFR-SAC project. The ZIP content,
bundle branch `revision-extensions`, and three patch files are consistent. The
bundle is based on main commit `b75f003` and ends at `69ce31a`.

Audit findings: 9 items total, including 2 P0, 4 P1, and 3 P2 items.

## P0

### P0-1: Revision source was not covered by the frozen digest

- Location: `v2/checkpoint.py`, `revision/runner.py`, task identity flow.
- Finding: `source_digest()` covers the frozen main directories but not
  `revision/*.py`.
- Status: **fixed** in `revision/source.py`, `revision/protocol.py`,
  `revision/runner.py`, `revision/worker.py`, and `revision/report.py`.
- Fix: a separate revision digest and per-run `revision_source.json` marker are
  checked without changing historical main-model digests.
- Numerical impact: none for existing results; it changes only acceptance and
  resume identity for revision outputs.

### P0-2: Historical main results are not runnable from the revision branch

- Location: frozen `original_ch5/protocol.py` versus modified v2/agent files.
- Finding: the revision branch changes the main source digest. Historical main
  checkpoints and `original_ch5` outputs must not be accepted by this branch.
- Status: **not merged away by design**. The safe solution is separate
  checkouts: the historical main commit for the old study and this branch for
  revision experiments.
- Numerical impact: no old results are changed. The revision branch cannot be
  used to resume the historical main suite.

## P1

### P1-1: Revision test used a non-Long one-hot index

- Location: `tests/test_revision.py::test_exact_slot_model_matches_simulator`.
- Status: **fixed** by using `dtype=torch.long` for AP indices.
- Numerical impact: test-only; no algorithm or result change.

### P1-2: CI overlap with zero was described as a match

- Location: `README_REVISION.md` and report wording.
- Status: **fixed**. A confidence interval containing zero is now reported as
  inconclusive. Equivalence requires a pre-specified margin and an equivalence
  test.
- Numerical impact: none; this changes interpretation only.

### P1-3: Per-slot optimizers are not global optima

- Location: `revision/per_slot.py`.
- Finding: critic descent and the full-information optimizer use finite
  iterations, relaxed AP logits, multi-start heuristics, and local Adam steps.
- Status: **documented, not a code bug**. Use “full-information per-slot
  numerical optimizer” and “critic-descent controller”, never “global optimum”.

### P1-4: The extension results are not yet scientifically validated

- Location: all eight stage runners.
- Status: **pending GPU/CPU experiment**. Static checks confirm construction and
  interfaces, not convergence, fairness, CPO stability, mismatch behavior, or
  latency values.

## P2

### P2-1: `tests/test_revision.py` contains four tiny training tests

- Status: **not run** during preflight. Static and deterministic tests were run
  with `-k 'not new_arms_train_and_reload'`.

### P2-2: One test used a fixed Unix temporary path

- Location: `tests/test_revision.py::test_catalog_shares_identical_trainings`.
- Status: **fixed** by using pytest's `tmp_path` fixture.
- Numerical impact: test-only.

### P2-3: E6 measures policy computation, not full system latency

- Location: `revision/latency.py`.
- Status: **documented**. Actor forward pass plus action decoding is timed;
  simulator stepping, state collection, signaling, and environment transition
  are excluded.

## Static Verification

- Python compilation: passed for all package Python files.
- Import check: passed for the revision modules and modified runtime.
- Revision deterministic tests: `9 passed, 4 deselected`.
- Dry-run plan: 175 unique trainings, 157 CPU tasks, eight stages.
- State dimension: `4N + M + 8 + NM`; at `N=70, M=6`, `714`.
- Action dimension: `2N + NM`; at `N=70, M=6`, `560`.
- No formal experiment was started.
