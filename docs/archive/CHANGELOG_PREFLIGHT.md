# Preflight Change Log

## Revision Preflight Fixes

### `revision/source.py` (new)

Adds a revision-extension digest built from the frozen main-study digest and
`revision/*.py`. It writes a per-run source marker without changing the old
main-model fingerprint.

### `revision/protocol.py`

Adds the revision digest to the protocol definition, parent specs, test specs,
per-slot specs, latency specs, and manifest.

### `revision/runner.py`

Rejects missing or mismatched revision markers and refuses to resume an output
directory created by another revision-extension source version.

### `revision/worker.py`

Writes the revision source marker before training, held-out testing, per-slot
optimization, or latency execution.

### `revision/report.py`

Excludes source-mismatched task outputs from reports and states that a nominal
CI containing zero is inconclusive rather than equivalent.

### `tests/test_revision.py`

Fixes the AP one-hot test to pass a `torch.long` index tensor. This is a test
correction only.

## Numerical Protocol

No SFR-SAC algorithm, training hyperparameter, environment default, seed,
result file, checkpoint, or historical experiment was modified. No formal
training or large numerical optimization was run.
