# Repository Cleanup Plan

This plan is intentionally non-destructive. No historical result, source file, or development document was removed in the finalization branch.

## Keep in the public top-level view

- `agents/`, `envs/`, `v2/`, `cloud24/`, `configs/`, `models/`, `revision/`, `tests/`.
- `scripts/plot_paper_results.py` and `scripts/validate_paper_results.py`.
- `results/paper/` as the primary paper-results entry point.
- `results/final/` as the original formal main-study export.

## Retain for provenance but do not use as the primary reader entry point

- `results/revision_20261108/` and its container/local sub-archives.
- `docs/`, `README_REVISION.md`, `EXPERIMENT_PROTOCOL.md`, and runbooks.
- `diagnostics/` and historical runner scripts.

## Candidate for a later archive-only move, after author review

- Redundant development-era plotting scripts whose outputs are already represented in `results/paper/`.
- Local audit exports and temporary packaging notes that are not required for reproduction.
- Old run instructions that duplicate the canonical README.

## Not removed in this branch

No files are proposed for automatic deletion. The author should review the candidate list after the paper-results branch has been checked, then move selected files to a separate archive branch or release asset. No force push or history rewrite is required.
