# Paper Results Provenance

## Canonical Sources

| Result family | Canonical source | Replication | Source identity |
|---|---|---:|---|
| Main scale comparison and final ablations | `results/final/` | 5 paired seeds | Confirmed final-study exports |
| Fair Critics | `results/revision_20261108/container_20261109/` | 5 pairs for learner comparison; 3 for fixed-penalty rows | Main digest `36c1289d...`; revision digest `7e9709f3...`; protocol `25c98d2abc8b0e88` |
| Per-slot optimization | `results/revision_20261108/container_20261109/` | 5 pairs; 20 tests | Same source and revision identity as Fair Critics |
| Other revision controls | `results/revision_20261108/tables/` | As recorded in each table | Revision provenance in `data/provenance.json` |

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
