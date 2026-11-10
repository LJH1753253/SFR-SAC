# Paper Results Provenance

## Canonical Sources

| Result family | Canonical source | Replication | Source identity |
|---|---|---:|---|
| Main scale comparison and final ablations | `results/final/` | 5 paired seeds | Confirmed final-study exports |
| Fair Critics | `results/revision_20261108/container_20261109/` | 5 pairs for learner comparison; 3 for fixed-penalty rows | Main digest `36c1289d...`; revision digest `7e9709f3...`; protocol `25c98d2abc8b0e88` |
| Per-slot optimization | `results/revision_20261108/container_20261109/` | 5 pairs; 20 tests | Same source and revision identity as Fair Critics |
| Other revision controls | `results/revision_20261108/tables/` | As recorded in each table | Revision provenance in `data/provenance.json` |

## Supplementary Per-Seed Statistics

The tables [near_zero_offloading_N70.csv](per_seed/near_zero_offloading_N70.csv)
and [main_N65_SFR_per_seed.csv](per_seed/main_N65_SFR_per_seed.csv) provide
additional statistics for the original frozen `split_exact` main experiment,
not new training runs. Each uses five paired training and layout seeds, with
100 evaluation episodes of 200 slots per deployment. The original source
digest is `e942cebacf7dd4896d0a5b5647a47fb9c8246d7982c2fe21d72ba58b2f01d7d3`.
Each row records the training task ID, held-out test ID, test random seed,
checkpoint configuration fingerprint, and SHA-256 of the archived `model.pt`;
the task ID and checkpoint configuration fingerprint are distinct identifiers.

The N=70 table records a previously completed post-hoc deterministic inference
diagnostic using the five frozen main-study checkpoints and their original
held-out random streams. The count is over all device-slot decisions with
decoded attempted-offloading ratio `x < 0.01`, not the successfully delivered
fraction. The recorded requested and executed attempted ratios were identical
for all decisions. Across the five models the mean and sample SD are
0.169143% and 0.134897%, respectively; the pooled count is 11,840 of 7,000,000.
The diagnostic was not rerun for this export.

The N=65 table transcribes the original main-study per-seed test summaries.
Its failure and energy values reproduce the N=65 `split_exact` aggregate in
`summary/evidence5_summary.csv`; across deployments they are 9.209346% +/-
0.427495% and 9.042879 +/- 2.440721 mJ/device/slot (mean +/- sample SD).

## Excluded From Canonical Aggregation

`results/revision_20261108/tables/per_slot.csv` is an earlier local Per-slot archive. It is retained for traceability, but excluded from `results/paper/` because its source, checkpoint, and hardware provenance differs from the container rerun.

## Units

- Energy: mJ/device/slot.
- Service-failure cost: percent.
- BFSR: percent of evaluation slots satisfying the budget.
- Decision latency: milliseconds for the measured policy or optimization path.
- Error bars in canonical figures represent sample SD unless stated otherwise.

## Reproduction

```bash
python -m pip install -r requirements.txt
python scripts/plot_paper_results.py
python scripts/validate_paper_results.py
```

The plotting script reads only `results/paper/tables/*.csv`, `results/paper/summary/*.csv`, and `results/paper/per_seed/*.csv`. It does not read historical output directories or hard-code experimental measurements. The validator checks table row counts, method sets, aggregate consistency, source identities, figure/contact-sheet completeness, sensitivity ordering, and core-source immutability on the finalization branch.

## Integrity Notes

- No training or numerical optimization was run during packaging.
- Existing historical result directories were not deleted or modified.
- The finalization work is on a separate branch and is not merged into `main`.
- Full checkpoints and replay buffers remain outside the public result package.
