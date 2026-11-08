# SFR-SAC IoT-J Remaining Revision Experiments

This package runs only the revision stages that were not covered by the local Per-slot and Fair Critics runs:

1. Model Mismatch
2. CPO / E7
3. E1: Immediate versus Discounted Critics
4. E2: Hinge versus Linear Penalty
5. E3: AP-dependent Channel Gain
6. E6: Decision Latency

The package does not run `fair_critics` or `per_slot`. It uses the frozen revision protocol and keeps each stage in the same resumable output directory.

## Environment

Use the CUDA-enabled Python environment in the container. The expected setup is:

```bash
export PYTHON=/opt/conda/bin/python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
```

The runner is designed for two GPUs and a 32-core host. It uses four concurrent GPU workers and twenty CPU workers by default. Reduce these values only if the container reports memory pressure:

```bash
export GPUS=0,1
export GPU_WORKERS=4
export CPU_WORKERS=20
```

## Smoke and plan

Run the lightweight rehearsal before formal execution:

```bash
bash run_remaining.sh smoke
bash run_remaining.sh dry-run
```

Smoke numbers are not paper results. The dry-run is only a task and resource check.

## Formal background run

Start all remaining stages without an artificial wall-clock deadline:

```bash
export PYTHON=/opt/conda/bin/python
export REVISION_OUTPUT=$PWD/outputs/revision_remaining_20261107
export GPUS=0,1
export GPU_WORKERS=4
export CPU_WORKERS=20

bash run_remaining.sh start
```

`run_remaining.sh` itself starts the controller in the background and runs the stages sequentially. The formal log is:

```text
outputs/revision_remaining_20261107/remaining.log
```

The order is `mismatch`, `cpo`, `horizon`, `penalty_form`, `ap`, and `latency`. Completed tasks are skipped on resume, and the same source/config identity is required before a result is accepted.

## Monitoring and resume

```bash
bash run_remaining.sh status
tail -f outputs/revision_remaining_20261107/remaining.log
nvidia-smi
```

If the SSH session or container is interrupted, reconnect to the same directory and run:

```bash
export PYTHON=/opt/conda/bin/python
export REVISION_OUTPUT=$PWD/outputs/revision_remaining_20261107
export GPUS=0,1
export GPU_WORKERS=4
export CPU_WORKERS=20
bash run_remaining.sh start
```

Do not delete successful run directories. To request a graceful stop:

```bash
bash run_remaining.sh stop
```

After all stages finish, the report is generated at:

```text
outputs/revision_remaining_20261107/report/REVISION_REPORT.md
```

The report uses mean and sample standard deviation across paired deployments. It does not treat individual slots as independent training replicates.
